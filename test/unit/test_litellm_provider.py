# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the LiteLLM provider and its wiring.

Stubs the litellm package so the suite runs without it, and checks the
request each call builds, the shared tool-call shape, error handling, the
api dispatch for provider LITELLM, and that the LiteLLM key never lands in
a queue row.

Main Features:
* generate_text / call_with_tools request shape: drop_params, timeout, optional key and base URL.
* Think-tag stripping, empty responses, tool-call parsing, cap and invalid arguments.
* Configuration errors (bad key, unknown model) are surfaced; transient ones collapse to a generic message.
* validate_ai_config, generate_text and call_with_tools route LITELLM to the provider.
* litellm_api_key_param is stripped from queue payloads and restored from config.LITELLM_API_KEY.
"""

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import config
from tasks.ai import api
from tasks.ai.providers import litellm as litellm_provider
from taskqueue import sql

TOOLS = [
    {"name": "search_database", "description": "Search songs", "inputSchema": {"type": "object"}},
    {"name": "text_search", "description": "Search by text", "inputSchema": {"type": "object"}},
]


class AuthenticationError(Exception):
    """Stands in for litellm.AuthenticationError, which the provider matches by name."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def _chat_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


@pytest.fixture
def fake_litellm(monkeypatch):
    module = types.ModuleType("litellm")
    module.completion = MagicMock(return_value=_chat_response(content="Rainy Day Jazz"))
    monkeypatch.setitem(sys.modules, "litellm", module)
    monkeypatch.setattr(litellm_provider, "_LITELLM_AVAILABLE", True)
    monkeypatch.setattr(config, "AI_REQUEST_TIMEOUT_SECONDS", 42)
    return module.completion


def _ai_config(**overrides):
    cfg = {"provider": "LITELLM", "litellm_model": "anthropic/claude-sonnet-4-5", "litellm_key": "", "litellm_url": ""}
    cfg.update(overrides)
    return cfg


class TestGenerateText:
    def test_builds_request_without_credentials(self, fake_litellm):
        result = litellm_provider.generate_text(
            "anthropic/claude-sonnet-4-5", "Name this playlist", skip_delay=True, temperature=0.3, max_tokens=64
        )

        assert result == "Rainy Day Jazz"
        kwargs = fake_litellm.call_args.kwargs
        assert kwargs["model"] == "anthropic/claude-sonnet-4-5"
        assert kwargs["messages"] == [{"role": "user", "content": "Name this playlist"}]
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 64
        assert kwargs["timeout"] == 42
        assert kwargs["drop_params"] is True
        # Unset, LiteLLM reads the upstream provider's own credentials from the environment.
        assert "api_key" not in kwargs
        assert "api_base" not in kwargs
        assert sys.modules["litellm"].suppress_debug_info is True

    def test_forwards_key_and_base_url_when_set(self, fake_litellm):
        litellm_provider.generate_text(
            "litellm_proxy/playlist-namer", "Name it", "sk-proxy", "http://litellm:4000/", skip_delay=True
        )

        kwargs = fake_litellm.call_args.kwargs
        assert kwargs["api_key"] == "sk-proxy"
        assert kwargs["api_base"] == "http://litellm:4000"

    def test_defaults_match_the_openai_path(self, fake_litellm):
        litellm_provider.generate_text("groq/llama-3.3-70b-versatile", "Name it", skip_delay=True)

        kwargs = fake_litellm.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 8000

    def test_strips_think_tags(self, fake_litellm):
        fake_litellm.return_value = _chat_response(content="<think>pondering</think>  Midnight Drive ")

        assert litellm_provider.generate_text("deepseek/deepseek-reasoner", "Name it", skip_delay=True) == "Midnight Drive"

    def test_waits_for_the_configured_delay(self, fake_litellm, monkeypatch):
        sleeps = []
        monkeypatch.setenv("LITELLM_API_CALL_DELAY_SECONDS", "3")
        monkeypatch.setattr(litellm_provider.time, "sleep", sleeps.append)

        litellm_provider.generate_text("gemini/gemini-2.5-flash", "Name it")

        assert sleeps == [3]

    def test_empty_content_is_an_error(self, fake_litellm):
        fake_litellm.return_value = _chat_response(content="   ")

        assert litellm_provider.generate_text("gemini/gemini-2.5-flash", "x", skip_delay=True) == (
            "Error: LiteLLM returned no content."
        )

    def test_configuration_errors_are_surfaced(self, fake_litellm):
        fake_litellm.side_effect = AuthenticationError("invalid x-api-key")

        result = litellm_provider.generate_text("anthropic/claude-sonnet-4-5", "x", skip_delay=True)

        assert result.startswith("Error: LiteLLM request was rejected (AuthenticationError)")
        assert "invalid x-api-key" in result

    def test_transient_errors_collapse_to_a_generic_message(self, fake_litellm):
        fake_litellm.side_effect = ConnectionError("reset by peer")

        assert litellm_provider.generate_text("anthropic/claude-sonnet-4-5", "x", skip_delay=True) == (
            "Error: LiteLLM service is currently unavailable."
        )

    def test_missing_package_is_reported(self, monkeypatch):
        monkeypatch.setattr(litellm_provider, "_LITELLM_AVAILABLE", False)

        assert litellm_provider.generate_text("anthropic/claude-sonnet-4-5", "x", skip_delay=True).startswith(
            "Error: the litellm package is not installed"
        )


class TestCallWithTools:
    def _call(self, log):
        return litellm_provider.call_with_tools(
            "bedrock/anthropic.claude-sonnet-4", None, None, "system", "songs for a rainy day", TOOLS, log
        )

    def test_returns_the_shared_tool_call_shape(self, fake_litellm):
        fake_litellm.return_value = _chat_response(
            tool_calls=[_tool_call("search_database", json.dumps({"genres": ["jazz"]}))]
        )
        log = []

        result = self._call(log)

        assert result == {"tool_calls": [{"name": "search_database", "arguments": {"genres": ["jazz"]}}]}
        kwargs = fake_litellm.call_args.kwargs
        assert kwargs["tool_choice"] == "required"
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 1024
        assert kwargs["drop_params"] is True
        assert kwargs["messages"][0] == {"role": "system", "content": "system"}
        assert [t["function"]["name"] for t in kwargs["tools"]] == ["search_database", "text_search"]
        assert kwargs["tools"][0]["function"]["parameters"] == {"type": "object"}
        assert log == ["LiteLLM called 1 tools"]

    def test_caps_at_four_calls_and_skips_invalid_arguments(self, fake_litellm):
        calls = [_tool_call("text_search", json.dumps({"query": str(i)})) for i in range(5)]
        calls.insert(0, _tool_call("search_database", "{not json"))
        fake_litellm.return_value = _chat_response(tool_calls=calls)
        log = []

        result = self._call(log)

        assert [c["arguments"]["query"] for c in result["tool_calls"]] == ["0", "1", "2", "3"]
        assert "invalid JSON arguments" in log[0]

    def test_no_tool_calls_returns_the_text(self, fake_litellm):
        fake_litellm.return_value = _chat_response(content="I cannot help with that")

        result = self._call([])

        assert result == {"error": "AI did not call any tools", "ai_response": "I cannot help with that"}

    def test_errors_never_raise(self, fake_litellm):
        fake_litellm.side_effect = TimeoutError("slow")

        assert self._call([]) == {"error": "LiteLLM service is currently unavailable."}


class TestApiDispatch:
    def test_litellm_is_a_valid_provider(self, fake_litellm):
        assert "LITELLM" in api.VALID_PROVIDERS
        assert api.validate_ai_config(_ai_config()) == (True, None)

    def test_model_is_required(self, fake_litellm):
        valid, msg = api.validate_ai_config(_ai_config(litellm_model=""))

        assert valid is False
        assert "litellm_model is empty" in msg

    def test_missing_package_fails_validation(self, monkeypatch):
        monkeypatch.setattr(litellm_provider, "_LITELLM_AVAILABLE", False)

        valid, msg = api.validate_ai_config(_ai_config())

        assert valid is False
        assert "litellm package is not installed" in msg

    def test_generate_text_routes_to_litellm(self, fake_litellm):
        result = api.generate_text(
            "Name it", _ai_config(litellm_key="sk-1", litellm_url="http://litellm:4000"), skip_delay=True
        )

        assert result == "Rainy Day Jazz"
        kwargs = fake_litellm.call_args.kwargs
        assert kwargs["model"] == "anthropic/claude-sonnet-4-5"
        assert kwargs["api_key"] == "sk-1"
        assert kwargs["api_base"] == "http://litellm:4000"

    def test_call_with_tools_routes_to_litellm(self, fake_litellm):
        fake_litellm.return_value = _chat_response(tool_calls=[_tool_call("text_search", '{"query": "rain"}')])

        result = api.call_with_tools("rainy day", TOOLS, _ai_config(), system_prompt="system")

        assert result == {"tool_calls": [{"name": "text_search", "arguments": {"query": "rain"}}]}


class TestQueueSecrets:
    def test_litellm_key_never_lands_in_a_queue_row(self, monkeypatch):
        monkeypatch.setattr(config, "LITELLM_API_KEY", "sk-secret")

        kept, stripped = sql.strip_secret_kwargs({"litellm_api_key_param": "sk-secret", "litellm_model_name_param": "m"})

        assert kept == {"litellm_model_name_param": "m"}
        assert stripped == ["litellm_api_key_param"]
        assert sql.restore_secret_kwargs(kept, stripped)["litellm_api_key_param"] == "sk-secret"
