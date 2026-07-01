# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the AI provider layer and playlist-name helpers.

Mocks the HTTP and SDK calls to exercise tasks.ai name cleaning and the
OpenAI-compatible, Ollama, Gemini, and Mistral text-generation providers.

Main Features:
* Playlist-name sanitizing, unicode normalization, and think-tag stripping.
* Streaming chunk assembly, rate-limit backoff, and parameter fallbacks.
* URL-based OpenAI-vs-Ollama format detection and API-error handling.
"""

import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock as _MagicMock


_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def _ensure_namespace_pkg(name: str, sub_path: str) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(_REPO_ROOT, sub_path)]
    sys.modules[name] = pkg


def _load_submodule(name: str, relpath: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_httpx_stub():
    if 'httpx' in sys.modules:
        return
    try:
        import httpx  # noqa: F401

        return
    except ImportError:
        pass
    httpx_mod = types.ModuleType('httpx')

    class _ReadTimeout(Exception):
        pass

    class _TimeoutException(Exception):
        pass

    class _Client:
        def __init__(self, **kw):
            self.kwargs = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            raise NotImplementedError("stub")

    httpx_mod.ReadTimeout = _ReadTimeout
    httpx_mod.TimeoutException = _TimeoutException
    httpx_mod.Client = _Client
    sys.modules['httpx'] = httpx_mod


def _ensure_google_genai_stub():
    try:
        import google.genai  # noqa: F401

        return
    except (ImportError, ModuleNotFoundError):
        pass
    if 'google' not in sys.modules:
        google_mod = types.ModuleType('google')
        google_mod.__path__ = []
        sys.modules['google'] = google_mod
    from unittest.mock import MagicMock as _mock

    genai_mod = types.ModuleType('google.genai')
    genai_mod.Client = _mock
    genai_types = types.ModuleType('google.genai.types')
    genai_types.Tool = _mock
    genai_types.GenerateContentConfig = _mock
    genai_types.ToolConfig = _mock
    genai_types.FunctionCallingConfig = _mock
    genai_mod.types = genai_types
    sys.modules['google.genai'] = genai_mod
    sys.modules['google.genai.types'] = genai_types


def _ensure_mistralai_stub():
    if 'mistralai' in sys.modules:
        return
    try:
        import mistralai  # noqa: F401

        return
    except (ImportError, ModuleNotFoundError):
        pass
    mod = types.ModuleType('mistralai')
    mod.Mistral = _MagicMock  # type: ignore
    sys.modules['mistralai'] = mod


_ensure_namespace_pkg('tasks', 'tasks')
_ensure_namespace_pkg('tasks.ai', 'tasks/ai')
_ensure_namespace_pkg('tasks.ai.providers', 'tasks/ai/providers')
_ensure_httpx_stub()
_ensure_google_genai_stub()
_ensure_mistralai_stub()
for _name, _relpath in (
    ('tasks.ai.prompts', 'tasks/ai/prompts.py'),
    ('tasks.ai.providers.openai', 'tasks/ai/providers/openai.py'),
    ('tasks.ai.providers.gemini', 'tasks/ai/providers/gemini.py'),
    ('tasks.ai.providers.mistral', 'tasks/ai/providers/mistral.py'),
    ('tasks.ai.api', 'tasks/ai/api.py'),
):
    _load_submodule(_name, _relpath)


from unittest.mock import Mock, patch
import pytest
import requests
import json
import tasks.ai.providers.openai as ai_openai
from tasks.ai.api import clean_playlist_name, get_ai_playlist_name
from tasks.ai.providers.openai import generate_text as get_openai_compatible_playlist_name
from tasks.ai.providers.gemini import generate_text as get_gemini_playlist_name
from tasks.ai.providers.mistral import generate_text as get_mistral_playlist_name
from tasks.ai.prompts import creative_prompt_template


@pytest.fixture(autouse=True)
def _reset_reasoning_cache():
    ai_openai._MODELS_REJECTING_REASONING.clear()
    yield
    ai_openai._MODELS_REJECTING_REASONING.clear()


class TestCleanPlaylistName:
    def test_basic_ascii_name(self):
        name = "Rock Classics"
        assert clean_playlist_name(name) == "Rock Classics"

    def test_removes_special_characters(self):
        name = "Rock★Classics★"
        result = clean_playlist_name(name)
        assert "★" not in result
        assert result == "RockClassics"

    def test_preserves_allowed_punctuation(self):
        name = "Rock & Roll - 80's Hits! (Best)"
        result = clean_playlist_name(name)
        assert result == "Rock & Roll - 80's Hits! (Best)"

    def test_removes_trailing_number_parentheses(self):
        name = "My Playlist (2)"
        result = clean_playlist_name(name)
        assert result == "My Playlist"

    def test_handles_non_string_input(self):
        assert clean_playlist_name(None) == ""
        assert clean_playlist_name(123) == ""
        assert clean_playlist_name([]) == ""

    def test_normalizes_unicode(self):
        name = "Café"
        result = clean_playlist_name(name)
        assert isinstance(result, str)
        assert "Caf" in result

    def test_collapses_multiple_spaces(self):
        name = "Rock    Classics"
        result = clean_playlist_name(name)
        assert result == "Rock Classics"

    def test_strips_leading_trailing_whitespace(self):
        name = "  Rock Classics  "
        result = clean_playlist_name(name)
        assert result == "Rock Classics"

    def test_fixes_text_encoding(self):
        name = "Rock Classics"
        result = clean_playlist_name(name)
        assert isinstance(result, str)


class TestGetOpenAICompatiblePlaylistName:
    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_openai_format_success(self, mock_sleep, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        chunks = [
            b'data: {"choices":[{"delta":{"content":"Sunset"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" Vibes"}}]}\n',
            b'data: {"choices":[{"finish_reason":"stop"}]}\n',
        ]
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4",
            full_prompt="Create a playlist name",
            api_key="test-key",
        )

        assert result == "Sunset Vibes"
        assert mock_sleep.called

    @patch('tasks.ai.providers.openai.requests.post')
    def test_ollama_format_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        chunks = [
            b'{"response":"Morning","done":false}\n',
            b'{"response":" Calm","done":false}\n',
            b'{"response":"","done":true}\n',
        ]
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://localhost:11434/api/generate",
            model_name="deepseek-r1:1.5b",
            full_prompt="Create a playlist name",
            api_key="no-key-needed",
        )

        assert result == "Morning Calm"

    @patch('tasks.ai.providers.openai.requests.post')
    def test_handles_think_tags(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        chunks = [b'{"response":"<think>reasoning here</think>Final Name","done":true}\n']
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://localhost:11434/api/generate",
            model_name="model",
            full_prompt="test",
            api_key="no-key-needed",
        )

        assert result == "Final Name"
        assert "<think>" not in result

    @patch('tasks.ai.providers.openai.requests.post')
    def test_handles_api_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

        result = get_openai_compatible_playlist_name(
            server_url="http://invalid", model_name="model", full_prompt="test", api_key="key"
        )

        assert "Error" in result
        assert "unavailable" in result

    @patch('tasks.ai.providers.openai.requests.post')
    def test_handles_invalid_json(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        chunks = [b'invalid json\n', b'{"response":"Valid","done":true}\n']
        mock_response.iter_lines.return_value = chunks
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://localhost:11434/api/generate",
            model_name="model",
            full_prompt="test",
            api_key="no-key-needed",
        )

        assert result == "Valid"

    @patch('tasks.ai.providers.openai.requests.post')
    def test_openrouter_headers(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Test"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]
        mock_post.return_value = mock_response

        get_openai_compatible_playlist_name(
            server_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="openai/gpt-4",
            full_prompt="test",
            api_key="test-key",
        )

        call_args = mock_post.call_args
        headers = call_args[1]['headers']
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers

    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_combined_content_and_finish_reason_chunk(self, mock_sleep, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Quiet Storm"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="openai/gpt-4o-mini",
            full_prompt="prompt",
            api_key="test-key",
        )

        assert result == "Quiet Storm"
        assert mock_sleep.call_count == 1

    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_content_split_across_chunks_with_final_combined_chunk(self, mock_sleep, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Quiet "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"Storm"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="openai/gpt-4o-mini",
            full_prompt="prompt",
            api_key="test-key",
        )

        assert result == "Quiet Storm"
        assert mock_sleep.call_count == 1

    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_authenticated_ollama_url_uses_ollama_format(self, mock_sleep, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            b'{"response":"Quiet","done":false}\n',
            b'{"response":" Storm","done":false}\n',
            b'{"response":"","done":true}\n',
        ]
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://ollama-proxy.example.com:11434/api/generate",
            model_name="llama3.1:8b",
            full_prompt="prompt",
            api_key="real-bearer-token",
        )

        assert result == "Quiet Storm"
        sent_body = json.loads(mock_post.call_args[1]['data'])
        assert 'prompt' in sent_body
        assert 'messages' not in sent_body

    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_openai_format_detected_from_url_when_global_is_ollama(self, mock_sleep, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines.return_value = [
            b':' + b' OPENROUTER PROCESSING\n',
            b'data: {"choices":[{"index":0,"finish_reason":null,"text":"Late"}]}\n',
            b'data: {"choices":[{"index":0,"finish_reason":null,"text":" Night Blues"}]}\n',
            b'data: {"choices":[{"index":0,"finish_reason":"stop","text":""}]}\n',
            b'data: [DONE]\n',
        ]
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="anthropic/claude-sonnet-4.6",
            full_prompt="prompt",
            api_key="sk-or-v1-test",
        )

        assert result == "Late Night Blues"
        sent_body = json.loads(mock_post.call_args[1]['data'])
        assert 'messages' in sent_body
        assert 'prompt' not in sent_body

    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_rate_limit_retry_with_exponential_backoff(self, mock_sleep, mock_post):
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_429
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Success"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]

        mock_post.side_effect = [mock_response_429, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4",
            full_prompt="test",
            api_key="test-key",
        )

        assert result == "Success"
        assert mock_sleep.call_count >= 2
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert 5 in sleep_calls

    @patch('tasks.ai.providers.openai.os.environ.get')
    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_aggressive_fallback_on_unsupported_parameter(self, mock_sleep, mock_post, mock_env):
        mock_env.return_value = "0"

        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        error_response = {
            'error': {
                'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                'type': 'invalid_request_error',
                'param': 'max_tokens',
                'code': 'unsupported_parameter',
            }
        }
        mock_response_400.json.return_value = error_response
        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Fallback Success"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]

        mock_post.side_effect = [mock_response_400, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            full_prompt="test",
            api_key="test-key",
        )

        assert result == "Fallback Success"
        assert mock_post.call_count == 2

        second_call_data = json.loads(mock_post.call_args_list[1][1]['data'])
        assert 'temperature' not in second_call_data
        assert 'max_tokens' not in second_call_data
        assert second_call_data.get('max_completion_tokens') == 8000

    @patch('tasks.ai.providers.openai.os.environ.get')
    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_ultra_minimal_fallback_after_aggressive_fails(self, mock_sleep, mock_post, mock_env):
        mock_env.return_value = "0"

        mock_response_400_1 = Mock()
        mock_response_400_1.status_code = 400
        error_response_1 = {
            'error': {
                'message': "Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported.",
                'type': 'invalid_request_error',
                'param': 'temperature',
                'code': 'unsupported_value',
            }
        }
        mock_response_400_1.json.return_value = error_response_1
        mock_response_400_1.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400_1
        )

        mock_response_400_2 = Mock()
        mock_response_400_2.status_code = 400
        error_response_2 = {
            'error': {
                'message': "Unsupported parameter: 'max_completion_tokens' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'max_completion_tokens',
                'code': 'unsupported_parameter',
            }
        }
        mock_response_400_2.json.return_value = error_response_2
        mock_response_400_2.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400_2
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Ultra Minimal"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]

        mock_post.side_effect = [mock_response_400_1, mock_response_400_2, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            full_prompt="test",
            api_key="test-key",
        )

        assert result == "Ultra Minimal"
        assert mock_post.call_count == 3

        third_call_data = json.loads(mock_post.call_args_list[2][1]['data'])
        assert 'temperature' not in third_call_data
        assert 'max_tokens' not in third_call_data
        assert 'max_completion_tokens' not in third_call_data

    @patch('tasks.ai.providers.openai.os.environ.get')
    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_rate_limit_then_parameter_error(self, mock_sleep, mock_post, mock_env):
        mock_env.return_value = "0"

        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_429
        )

        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        error_response = {
            'error': {
                'message': "Unsupported parameter: 'temperature' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'temperature',
                'code': 'unsupported_parameter',
            }
        }
        mock_response_400.json.return_value = error_response
        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Combined Success"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]

        mock_post.side_effect = [mock_response_429, mock_response_400, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            full_prompt="test",
            api_key="test-key",
        )

        assert result == "Combined Success"
        assert mock_post.call_count == 3

        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list if call[0][0] >= 5]
        assert len(sleep_calls) >= 1

    @patch('tasks.ai.providers.openai.os.environ.get')
    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_parameter_fallbacks_dont_consume_retry_budget(self, mock_sleep, mock_post, mock_env):
        mock_env.return_value = "0"

        mock_response_400_1 = Mock()
        mock_response_400_1.status_code = 400
        error_response_1 = {
            'error': {
                'message': "Unsupported parameter: 'temperature' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'temperature',
                'code': 'unsupported_parameter',
            }
        }
        mock_response_400_1.json.return_value = error_response_1
        mock_response_400_1.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400_1
        )

        mock_response_400_2 = Mock()
        mock_response_400_2.status_code = 400
        error_response_2 = {
            'error': {
                'message': "Unsupported parameter: 'max_completion_tokens' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'max_completion_tokens',
                'code': 'unsupported_parameter',
            }
        }
        mock_response_400_2.json.return_value = error_response_2
        mock_response_400_2.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400_2
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Final Success"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]

        mock_post.side_effect = [mock_response_400_1, mock_response_400_2, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="model",
            full_prompt="test",
            api_key="test-key",
        )

        assert result == "Final Success"
        assert mock_post.call_count == 3

    @patch('tasks.ai.providers.openai.os.environ.get')
    @patch('tasks.ai.providers.openai.requests.post')
    def test_existing_max_tokens_fallback_still_works(self, mock_post, mock_env):
        mock_env.return_value = "0"

        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        error_response = {
            'error': {
                'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                'type': 'invalid_request_error',
                'param': 'max_tokens',
                'code': 'unsupported_parameter',
            }
        }
        mock_response_400.json.return_value = error_response
        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"Max Tokens Fallback"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]

        mock_post.side_effect = [mock_response_400, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="model",
            full_prompt="test",
            api_key="test-key",
        )

        assert result == "Max Tokens Fallback"
        second_call_data = json.loads(mock_post.call_args_list[1][1]['data'])
        assert 'max_tokens' not in second_call_data
        assert second_call_data.get('max_completion_tokens') == 8000

    @patch('tasks.ai.providers.openai.os.environ.get')
    @patch('tasks.ai.providers.openai.requests.post')
    def test_ultra_minimal_fallback_requires_proper_error_code(self, mock_post, mock_env):
        mock_env.return_value = "0"

        mock_response_400_1 = Mock()
        mock_response_400_1.status_code = 400
        error_response_1 = {
            'error': {
                'message': "Unsupported parameter: 'temperature' is not supported with this model.",
                'type': 'invalid_request_error',
                'param': 'temperature',
                'code': 'unsupported_parameter',
            }
        }
        mock_response_400_1.json.return_value = error_response_1
        mock_response_400_1.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400_1
        )

        mock_response_400_2 = Mock()
        mock_response_400_2.status_code = 400
        error_response_2 = {
            'error': {
                'message': 'Invalid parameter: max_completion_tokens',
                'type': 'invalid_request_error',
                'param': 'max_completion_tokens',
                'code': 'invalid_parameter',
            }
        }
        mock_response_400_2.json.return_value = error_response_2
        mock_response_400_2.text = 'Invalid parameter'
        mock_response_400_2.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400_2
        )

        mock_post.side_effect = [mock_response_400_1, mock_response_400_2]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="model",
            full_prompt="test",
            api_key="test-key",
        )

        assert "Error" in result
        assert mock_post.call_count == 2

    @patch('tasks.ai.providers.openai.os.environ.get')
    @patch('tasks.ai.providers.openai.requests.post')
    @patch('tasks.ai.providers.openai.time.sleep')
    def test_reasoning_effort_dropped_on_null_code_400(self, mock_sleep, mock_post, mock_env):
        mock_env.side_effect = (
            lambda key, default=None: "0" if key == "OPENAI_API_CALL_DELAY_SECONDS" else default
        )

        mock_response_400 = Mock()
        mock_response_400.status_code = 400
        mock_response_400.json.return_value = {
            'error': {
                'message': 'Unrecognized request argument supplied: reasoning_effort',
                'type': 'invalid_request_error',
                'param': None,
                'code': None,
            }
        }
        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_400
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()
        mock_response_success.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"OK Playlist"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]

        mock_post.side_effect = [mock_response_400, mock_response_success]

        result = get_openai_compatible_playlist_name(
            server_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            full_prompt="test",
            api_key="test-key",
        )

        assert result == "OK Playlist"
        assert mock_post.call_count == 2
        first_call_data = json.loads(mock_post.call_args_list[0][1]['data'])
        assert first_call_data.get('reasoning_effort') == 'none'
        second_call_data = json.loads(mock_post.call_args_list[1][1]['data'])
        assert 'reasoning_effort' not in second_call_data
        assert 'temperature' in second_call_data
        assert second_call_data.get('max_tokens') == 8000


class TestGetOllamaPlaylistName:
    @patch('tasks.ai.providers.openai.requests.post')
    def test_calls_with_ollama_format_url(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [b'{"response":"Test Playlist","done":true}']
        mock_post.return_value = mock_response

        result = get_openai_compatible_playlist_name(
            server_url="http://localhost:11434/api/generate",
            model_name="deepseek-r1:1.5b",
            full_prompt="test prompt",
            api_key="no-key-needed",
        )

        assert result == "Test Playlist"


class TestGetGeminiPlaylistName:
    @patch('google.genai.Client')
    @patch('tasks.ai.providers.gemini.time.sleep')
    def test_successful_gemini_call(self, mock_sleep, mock_client_class):
        mock_response = Mock()
        mock_response.text = "Chill Vibes"

        mock_models = Mock()
        mock_models.generate_content.return_value = mock_response

        mock_client = Mock()
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client

        result = get_gemini_playlist_name(
            api_key="valid-key", model_name="gemini-2.5-pro", full_prompt="Create a name"
        )

        assert result == "Chill Vibes"
        mock_client_class.assert_called_once_with(api_key="valid-key")
        assert mock_sleep.called

    def test_rejects_empty_api_key(self):
        result = get_gemini_playlist_name(
            api_key="", model_name="gemini-2.5-pro", full_prompt="test"
        )

        assert "Error" in result
        assert "missing" in result

    def test_rejects_placeholder_api_key(self):
        result = get_gemini_playlist_name(
            api_key="YOUR-GEMINI-API-KEY-HERE", model_name="gemini-2.5-pro", full_prompt="test"
        )

        assert "Error" in result

    @patch('google.genai.Client')
    @patch('tasks.ai.providers.gemini.time.sleep')
    def test_handles_gemini_api_error(self, mock_sleep, mock_client_class):
        mock_models = Mock()
        mock_models.generate_content.side_effect = Exception("API Error")

        mock_client = Mock()
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client

        result = get_gemini_playlist_name(
            api_key="valid-key", model_name="gemini-2.5-pro", full_prompt="test"
        )

        assert "Error" in result
        assert "unavailable" in result


class TestGetMistralPlaylistName:
    @patch('mistralai.Mistral')
    @patch('tasks.ai.providers.mistral.time.sleep')
    def test_successful_mistral_call(self, mock_sleep, mock_mistral_class):
        mock_message = Mock()
        mock_message.content = "Electronic Dreams"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_chat = Mock()
        mock_chat.complete.return_value = mock_response

        mock_client = Mock()
        mock_client.chat = mock_chat
        mock_mistral_class.return_value = mock_client

        result = get_mistral_playlist_name(
            api_key="valid-key", model_name="ministral-3b-latest", full_prompt="Create a name"
        )

        assert result == "Electronic Dreams"
        assert mock_sleep.called

    def test_rejects_empty_api_key(self):
        result = get_mistral_playlist_name(
            api_key="", model_name="ministral-3b-latest", full_prompt="test"
        )

        assert "Error" in result
        assert "missing" in result

    def test_rejects_placeholder_api_key(self):
        result = get_mistral_playlist_name(
            api_key="YOUR-MISTRAL-API-KEY-HERE",
            model_name="ministral-3b-latest",
            full_prompt="test",
        )

        assert "Error" in result


class TestGetAIPlaylistName:
    @staticmethod
    def _ai_config(provider, **extra):
        cfg = {"provider": provider}
        cfg.update(extra)
        return cfg

    @patch('tasks.ai.api.generate_text')
    def test_routes_to_ollama(self, mock_generate):
        mock_generate.return_value = "Test Playlist"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Song 1", "author": "Artist 1"}],
            {"energy": 0.8},
            self._ai_config(
                "OLLAMA",
                ollama_url="http://localhost:11434/api/generate",
                ollama_model="deepseek-r1:1.5b",
            ),
        )

        assert result == "Test Playlist"
        mock_generate.assert_called_once()

    @patch('tasks.ai.api.generate_text')
    def test_routes_to_gemini(self, mock_generate):
        mock_generate.return_value = "Gemini Playlist"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Song 1", "author": "Artist 1"}],
            {},
            self._ai_config("GEMINI", gemini_key="valid-key", gemini_model="gemini-2.5-pro"),
        )

        assert result == "Gemini Playlist"
        mock_generate.assert_called_once()

    @patch('tasks.ai.api.generate_text')
    def test_routes_to_mistral(self, mock_generate):
        mock_generate.return_value = "Mistral Playlist"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Symphony", "author": "Beethoven"}],
            {},
            self._ai_config(
                "MISTRAL", mistral_key="valid-key", mistral_model="ministral-3b-latest"
            ),
        )

        assert result == "Mistral Playlist"
        mock_generate.assert_called_once()

    @patch('tasks.ai.api.generate_text')
    def test_routes_to_openai(self, mock_generate):
        mock_generate.return_value = "OpenAI Playlist"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Track", "author": "Artist"}],
            {},
            self._ai_config(
                "OPENAI",
                openai_url="https://api.openai.com/v1/chat/completions",
                openai_model="gpt-4",
                openai_key="test-key",
            ),
        )

        assert result == "OpenAI Playlist"
        mock_generate.assert_called_once()

    @patch('tasks.ai.api.generate_text')
    def test_handles_none_provider(self, mock_generate):
        mock_generate.return_value = "AI Naming Skipped"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [],
            {},
            self._ai_config("NONE"),
        )

        assert result == "AI Naming Skipped"

    @patch('tasks.ai.api.clean_playlist_name')
    @patch('tasks.ai.api.generate_text')
    def test_applies_length_constraints(self, mock_generate, mock_clean):
        mock_generate.return_value = "Test"
        mock_clean.return_value = "Test"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Test", "author": "Artist"}],
            {},
            self._ai_config(
                "OLLAMA", ollama_url="http://localhost:11434/api/generate", ollama_model="model"
            ),
        )

        assert "Error" in result
        assert "outside" in result

    @patch('tasks.ai.api.clean_playlist_name')
    @patch('tasks.ai.api.generate_text')
    def test_cleans_playlist_name(self, mock_generate, mock_clean):
        mock_generate.return_value = "Rock & Roll - Best Hits!"
        mock_clean.return_value = "Rock & Roll - Best Hits!"

        result = get_ai_playlist_name(
            creative_prompt_template,
            [{"title": "Test", "author": "Artist"}],
            {},
            self._ai_config(
                "OLLAMA", ollama_url="http://localhost:11434/api/generate", ollama_model="model"
            ),
        )

        mock_clean.assert_called_once()
        assert result == "Rock & Roll - Best Hits!"

    @patch('tasks.ai.api.generate_text')
    def test_formats_song_list_in_prompt(self, mock_generate):
        mock_generate.return_value = "Test Playlist Name"
        song_list = [
            {"title": "Song One", "author": "Artist A"},
            {"title": "Song Two", "author": "Artist B"},
        ]

        get_ai_playlist_name(
            creative_prompt_template,
            song_list,
            {},
            self._ai_config(
                "OLLAMA", ollama_url="http://localhost:11434/api/generate", ollama_model="model"
            ),
        )

        prompt = mock_generate.call_args[0][0]
        assert "Song One" in prompt
        assert "Artist A" in prompt
        assert "Song Two" in prompt
        assert "Artist B" in prompt

    def test_prompt_includes_length_requirement(self):
        assert (
            "The title MUST be within the range of 5 to 40 characters long."
            in creative_prompt_template
        )
