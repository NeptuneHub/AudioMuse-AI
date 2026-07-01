# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Mistral client for the playlist AI.

One of the per-provider backends dispatched from ``tasks.ai.api``, using the
mistralai SDK. Exposes generate_text and single-turn call_with_tools with
tool_choice="any", returning the shared {"name","arguments"} tool-call shape.

Main Features:
* Probes for the mistralai SDK at import (is_available) since the package has been quarantined on PyPI; when missing, every call returns a clear "pick another provider" message instead of raising.
* Honors a pre-call delay (env MISTRAL_API_CALL_DELAY_SECONDS, default 7s) and config.AI_REQUEST_TIMEOUT_SECONDS; errors collapse to a generic unavailable message, never a traceback.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

try:
    import mistralai as _mistralai_probe  # noqa: F401

    _MISTRAL_AVAILABLE = True
    _MISTRAL_IMPORT_ERROR = None
except ImportError as _exc:
    _MISTRAL_AVAILABLE = False
    _MISTRAL_IMPORT_ERROR = str(_exc)

_MISTRAL_UNAVAILABLE_MSG = (
    "Error: mistralai SDK is not installed. The package is currently "
    "quarantined on PyPI - pick a different AI provider (Gemini / OpenAI / "
    "Ollama) until the SDK is reinstallable."
)


def is_available() -> bool:
    return _MISTRAL_AVAILABLE


def generate_text(
    api_key: str,
    model_name: str,
    full_prompt: str,
    *,
    skip_delay: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    if not _MISTRAL_AVAILABLE:
        logger.error(
            "Mistral provider selected but SDK is not installed: %s", _MISTRAL_IMPORT_ERROR
        )
        return _MISTRAL_UNAVAILABLE_MSG
    if not api_key or api_key == "YOUR-MISTRAL-API-KEY-HERE":
        return "Error: Mistral API key is missing or empty. Please provide a valid API key."

    try:
        from mistralai import Mistral

        if not skip_delay:
            mistral_call_delay = int(os.environ.get("MISTRAL_API_CALL_DELAY_SECONDS", "7"))
            if mistral_call_delay > 0:
                logger.debug(
                    "Waiting for %ss before mistral API call to respect rate limits.",
                    mistral_call_delay,
                )
                time.sleep(mistral_call_delay)

        client = Mistral(api_key=api_key)
        logger.debug("Starting API call for model '%s'.", model_name)

        complete_kwargs = {
            "model": model_name,
            "temperature": 0.9 if temperature is None else float(temperature),
            "timeout_ms": config.AI_REQUEST_TIMEOUT_SECONDS * 1000,
            "messages": [{"role": "user", "content": full_prompt}],
        }
        if max_tokens is not None:
            complete_kwargs["max_tokens"] = int(max_tokens)
        response = client.chat.complete(**complete_kwargs)

        if response and response.choices[0].message.content:
            extracted_text = response.choices[0].message.content
            logger.info("Mistral API returned: '%s'", extracted_text)
            return extracted_text
        logger.warning("Mistral returned no content. Raw response: %s", response)
        return "Error: mistral returned no content."

    except Exception as e:
        logger.error("Error calling Mistral API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."


def call_with_tools(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
) -> Dict:
    if not _MISTRAL_AVAILABLE:
        logger.error(
            "Mistral provider selected but SDK is not installed: %s", _MISTRAL_IMPORT_ERROR
        )
        return {"error": _MISTRAL_UNAVAILABLE_MSG}
    try:
        from mistralai import Mistral

        if not api_key or api_key == "YOUR-MISTRAL-API-KEY-HERE":
            return {"error": "Valid Mistral API key required"}

        client = Mistral(api_key=api_key)

        mistral_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                },
            }
            for tool in tools
        ]

        response = client.chat.complete(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=mistral_tools,
            tool_choice="any",
            temperature=0,
            max_tokens=1024,
        )

        tool_calls = []
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(
                        {
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        }
                    )

        if not tool_calls:
            text_response = response.choices[0].message.content if response.choices else ""
            log_messages.append(f"Mistral did not call tools. Response: {text_response[:200]}")
            return {"error": "AI did not call any tools", "ai_response": text_response}

        log_messages.append(f"Mistral called {len(tool_calls)} tools")
        return {"tool_calls": tool_calls}

    except Exception:
        logger.exception("Error calling Mistral with tools")
        return {"error": "Mistral service is currently unavailable."}
