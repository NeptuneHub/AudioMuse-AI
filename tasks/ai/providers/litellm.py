# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""LiteLLM client for the playlist AI.

One of the per-provider backends dispatched from ``tasks.ai.api``, using the
LiteLLM SDK. The model id names the upstream provider (``anthropic/...``,
``bedrock/...``, ``vertex_ai/...``, ``azure/...``, ``groq/...``, or a
``litellm_proxy/...`` alias), and LiteLLM reads that provider's credentials
from its usual environment variables. Exposes generate_text and single-turn
call_with_tools, returning the shared {"name","arguments"} tool-call shape.

Main Features:
* LiteLLM translates parameters per model (max_tokens vs max_completion_tokens, unsupported temperature) and drop_params discards fields a model does not accept, so no request is resent with fallbacks.
* Optional LITELLM_API_KEY / LITELLM_API_BASE are passed only when set (e.g. a LiteLLM proxy); otherwise the upstream provider's own environment credentials are used.
* Honors a pre-call delay (env LITELLM_API_CALL_DELAY_SECONDS, default 7s) and config.AI_REQUEST_TIMEOUT_SECONDS; errors collapse to a readable message, never a traceback.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

try:
    import litellm as _litellm_probe  # noqa: F401

    _LITELLM_AVAILABLE = True
    _LITELLM_IMPORT_ERROR = None
except ImportError as _exc:
    _LITELLM_AVAILABLE = False
    _LITELLM_IMPORT_ERROR = str(_exc)

_LITELLM_UNAVAILABLE_MSG = "Error: the litellm package is not installed. Pick a different AI provider."

# Errors a retry or a different prompt cannot fix; surfaced so users can correct the configuration.
_CONFIG_ERRORS = ("AuthenticationError", "NotFoundError", "PermissionDeniedError", "BadRequestError")


def is_available() -> bool:
    return _LITELLM_AVAILABLE


def _completion(model_name: str, api_key: Optional[str], api_base: Optional[str], **kwargs):
    import litellm

    litellm.suppress_debug_info = True
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base.rstrip("/")
    return litellm.completion(
        model=model_name,
        timeout=config.AI_REQUEST_TIMEOUT_SECONDS,
        drop_params=True,
        **kwargs,
    )


def _error_message(exc: Exception) -> str:
    if type(exc).__name__ in _CONFIG_ERRORS:
        return f"LiteLLM request was rejected ({type(exc).__name__}): {getattr(exc, 'message', exc)}"
    return "LiteLLM service is currently unavailable."


def generate_text(
    model_name: str,
    full_prompt: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    *,
    skip_delay: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    if not _LITELLM_AVAILABLE:
        logger.error("LiteLLM provider selected but the package is not installed: %s", _LITELLM_IMPORT_ERROR)
        return _LITELLM_UNAVAILABLE_MSG

    if not skip_delay:
        call_delay = int(os.environ.get("LITELLM_API_CALL_DELAY_SECONDS", "7"))
        if call_delay > 0:
            logger.debug("Waiting for %ss before LiteLLM API call to respect rate limits.", call_delay)
            time.sleep(call_delay)

    from tasks.ai.providers.openai import _strip_thinking

    try:
        logger.debug("Starting LiteLLM call for model '%s'.", model_name)
        response = _completion(
            model_name,
            api_key,
            api_base,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7 if temperature is None else float(temperature),
            max_tokens=8000 if max_tokens is None else int(max_tokens),
        )
        content = response.choices[0].message.content if response.choices else None
        if content and content.strip():
            extracted_text = _strip_thinking(content.strip())
            logger.info("LiteLLM returned: '%s'", extracted_text)
            return extracted_text
        logger.warning("LiteLLM returned no content. Raw response: %s", response)
        return "Error: LiteLLM returned no content."
    except Exception as exc:
        logger.exception("Error calling LiteLLM")
        return f"Error: {_error_message(exc)}"


def call_with_tools(
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
    system_prompt: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
) -> Dict:
    if not _LITELLM_AVAILABLE:
        logger.error("LiteLLM provider selected but the package is not installed: %s", _LITELLM_IMPORT_ERROR)
        return {"error": _LITELLM_UNAVAILABLE_MSG}

    from tasks.ai.providers.openai import _tool_function_specs

    try:
        response = _completion(
            model_name,
            api_key,
            api_base,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=_tool_function_specs(tools),
            tool_choice="required",
            temperature=0,
            max_tokens=1024,
        )
    except Exception as exc:
        logger.exception("Error calling LiteLLM with tools")
        return {"error": _error_message(exc)}

    message = response.choices[0].message if response.choices else None
    tool_calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        try:
            arguments = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            log_messages.append(f"LiteLLM returned invalid JSON arguments for tool '{tc.function.name}'")
            continue
        tool_calls.append({"name": tc.function.name, "arguments": arguments})

    if len(tool_calls) > 4:
        log_messages.append(f"LiteLLM returned {len(tool_calls)} tool calls; capping to first 4")
        tool_calls = tool_calls[:4]

    if not tool_calls:
        text_response = getattr(message, "content", None) or ""
        log_messages.append(f"LiteLLM did not call tools. Response: {text_response[:200]}")
        return {"error": "AI did not call any tools", "ai_response": text_response}

    log_messages.append(f"LiteLLM called {len(tool_calls)} tools")
    return {"tool_calls": tool_calls}
