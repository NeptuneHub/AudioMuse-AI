# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""OpenAI-compatible client (OpenAI, OpenRouter, Ollama) for the playlist AI.

The HTTP backend dispatched from ``tasks.ai.api`` for every non-SDK provider.
generate_text streams SSE completions; call_with_tools does single-turn
function-calling; call_with_tools_ollama tries native /api/chat tool-calling
first (Hermes template), falling back to structured JSON output on
/api/generate when native calls fail.

Main Features:
* Detects Ollama vs OpenAI shape from the URL, adds OpenRouter referer headers, and strips <think>/[/INST] reasoning tags from streamed output.
* Robust 400 fallbacks: retries without reasoning_effort (caching rejecting models), swaps max_tokens->max_completion_tokens, and cycles DeepSeek thinking-off forms; tool-call count is capped to 4 and all failures return a generic error, never a traceback.
* Ollama dual-path: native /api/chat tool-calling (enable_thinking=false for Qwen) with structured-output format=schema fallback; tool names are validated against the registry and invalid names trigger a feedback retry.
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import httpx
import requests

import config

logger = logging.getLogger(__name__)

THINK_END_TAG = "</think>"

_OLLAMA_GENERATE_PATH = "/api/generate"
_OLLAMA_CHAT_PATH = "/api/chat"
_ZEROABLE_ARGS = ("tempo_min", "tempo_max", "energy_min", "min_rating")

_MODELS_REJECTING_REASONING = set()


def _tool_function_specs(tools: List[Dict]) -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in tools
    ]


def _is_ollama_format_url(server_url: str) -> bool:
    s = server_url.lower()
    return _OLLAMA_GENERATE_PATH in s or _OLLAMA_CHAT_PATH in s


def _build_openai_headers(api_key: str, server_url: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "no-key-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    if "openrouter" in server_url.lower():
        headers["HTTP-Referer"] = "https://github.com/NeptuneHub/AudioMuse-AI"
        headers["X-Title"] = "AudioMuse-AI"
    return headers


def generate_text(
    server_url: str,
    model_name: str,
    full_prompt: str,
    api_key: str = "no-key-needed",
    *,
    skip_delay: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    is_ollama_format = _is_ollama_format_url(server_url)
    is_openai_format = not is_ollama_format
    provider_label = "Ollama" if is_ollama_format else "OpenAI/OpenRouter"

    headers = _build_openai_headers(api_key, server_url)

    temp = 0.7 if temperature is None else float(temperature)
    out_tokens = 8000 if max_tokens is None else int(max_tokens)

    is_deepseek = "deepseek" in (model_name or "").lower()

    if is_openai_format:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True,
            "temperature": temp,
            "max_tokens": out_tokens,
        }
        if model_name not in _MODELS_REJECTING_REASONING:
            payload["reasoning_effort"] = "low" if is_deepseek else "none"
    else:
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {"num_predict": out_tokens, "temperature": temp},
            "think": False,
        }

    max_retries = 3
    base_delay = 5
    tried_aggressive_fallback = False
    tried_ultra_minimal_fallback = False

    for attempt in range(max_retries + 1):
        try:
            if is_openai_format and attempt == 0 and not skip_delay:
                openai_call_delay = int(os.environ.get("OPENAI_API_CALL_DELAY_SECONDS", "7"))
                if openai_call_delay > 0:
                    logger.debug(
                        "Waiting for %ss before OpenAI/OpenRouter API call to respect rate limits.",
                        openai_call_delay,
                    )
                    time.sleep(openai_call_delay)

            logger.debug(
                "Starting API call for model '%s' at '%s' (format: %s). Attempt %d/%d",
                model_name,
                server_url,
                "OpenAI" if is_openai_format else "Ollama",
                attempt + 1,
                max_retries + 1,
            )

            response = requests.post(
                server_url, headers=headers, data=json.dumps(payload), stream=True, timeout=960
            )
            response.raise_for_status()
            full_raw_response_content = ""
            raw_sse_lines = []

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="ignore").strip()
                raw_sse_lines.append(line_str)
                if line_str.startswith(":"):
                    continue
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                    if line_str == "[DONE]":
                        break
                try:
                    chunk = json.loads(line_str)
                    if is_openai_format:
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            delta = choice.get("delta")
                            if isinstance(delta, dict):
                                content = delta.get("content")
                                if content is not None:
                                    full_raw_response_content += content
                            elif "text" in choice:
                                text = choice.get("text")
                                if text is not None:
                                    full_raw_response_content += text
                            finish_reason = choice.get("finish_reason")
                            if finish_reason == "length":
                                logger.warning("Response truncated due to max_tokens limit")
                                break
                            elif finish_reason in ("stop", "tool_calls", "content_filter", "error"):
                                break
                    else:
                        if "response" in chunk:
                            full_raw_response_content += chunk["response"]
                        if chunk.get("done"):
                            break
                except json.JSONDecodeError:
                    logger.debug("Could not decode JSON line from stream: %s", line_str)
                    continue

            thought_enders = [THINK_END_TAG, "[/INST]", "[/THOUGHT]"]
            extracted_text = full_raw_response_content.strip()
            for end_tag in thought_enders:
                if end_tag in extracted_text:
                    extracted_text = extracted_text.split(end_tag, 1)[-1].strip()

            if extracted_text:
                logger.info(
                    "%s API returned non-empty content (length=%d chars).",
                    provider_label,
                    len(extracted_text),
                )
                return extracted_text
            logger.warning(
                "%s returned empty content (raw response length: %d chars).",
                provider_label,
                len(full_raw_response_content),
            )
            logger.debug(
                "Raw SSE stream metadata: %d lines received; preview suppressed to avoid sensitive data logging.",
                len(raw_sse_lines),
            )
            if attempt < max_retries:
                sleep_time = base_delay * (2**attempt)
                logger.info("Retrying in %s seconds due to empty content...", sleep_time)
                time.sleep(sleep_time)
                continue
            return "Error: AI returned empty content after retries."

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(
                    "Rate limit exceeded (429). Attempt %d/%d", attempt + 1, max_retries + 1
                )
                if attempt < max_retries:
                    sleep_time = base_delay * (2**attempt)
                    logger.info("Retrying in %s seconds...", sleep_time)
                    time.sleep(sleep_time)
                    continue

            if e.response.status_code == 400 and is_openai_format:
                try:
                    error_body = e.response.json()
                    error_obj = error_body.get("error", {})
                    if not isinstance(error_obj, dict):
                        error_obj = {}
                    error_code = error_obj.get("code", "") or ""
                    error_param = error_obj.get("param", "") or ""
                    error_message = (error_obj.get("message", "") or "").lower()
                    if "reasoning_effort" in payload and (
                        error_param == "reasoning_effort" or "reasoning_effort" in error_message
                    ):
                        logger.info("reasoning_effort rejected (400); retrying without it")
                        payload.pop("reasoning_effort", None)
                        _MODELS_REJECTING_REASONING.add(model_name)
                        continue
                    if error_code in ("unsupported_parameter", "unsupported_value"):
                        if not tried_aggressive_fallback:
                            logger.info(
                                "Unsupported parameter detected (code: %s), switching to max_completion_tokens and removing temperature",
                                error_code,
                            )
                            payload.pop("temperature", None)
                            payload.pop("max_tokens", None)
                            payload.pop("reasoning_effort", None)
                            payload["max_completion_tokens"] = out_tokens
                            tried_aggressive_fallback = True
                            continue
                        elif not tried_ultra_minimal_fallback:
                            logger.info(
                                "Still failing with max_completion_tokens (code: %s), removing it (ultra-minimal mode)",
                                error_code,
                            )
                            payload.pop("max_completion_tokens", None)
                            tried_ultra_minimal_fallback = True
                            continue
                except (json.JSONDecodeError, KeyError, AttributeError):
                    pass

            try:
                error_detail = e.response.text
                logger.exception(
                    "Error calling OpenAI-compatible API. Response body: %s",
                    error_detail,
                )
            except Exception:
                logger.exception("Error calling OpenAI-compatible API")
            return "Error: AI service is currently unavailable."

        except requests.exceptions.RequestException:
            logger.exception("Error calling OpenAI-compatible API")
            return "Error: AI service is currently unavailable."
        except Exception:
            logger.exception(
                "An unexpected error occurred in ai_api_openai.generate_text"
            )
            return "Error: AI service is currently unavailable."

    return "Error: Max retries exceeded."


def call_with_tools(
    server_url: str,
    model_name: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
) -> Dict:
    try:
        functions = _tool_function_specs(tools)

        headers = _build_openai_headers(api_key, server_url)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "tools": functions,
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": 1024,
        }

        is_deepseek = "deepseek" in (model_name or "").lower()
        deepseek_thinking_off_forms = [
            {"thinking": {"type": "disabled"}},
            {"thinking": "none"},
            {"thinking_mode": "non_think"},
        ]
        if is_deepseek:
            payload.update(deepseek_thinking_off_forms[0])
        elif model_name not in _MODELS_REJECTING_REASONING:
            payload["reasoning_effort"] = "none"

        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        log_messages.append(f"Using timeout: {timeout} seconds for OpenAI request")

        def _post(p):
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(server_url, headers=headers, json=p)
                resp.raise_for_status()
                return resp.json()

        try:
            result = _post(payload)
        except httpx.HTTPStatusError as http_err:
            if http_err.response.status_code != 400:
                raise
            if is_deepseek:
                result = None
                for shape in deepseek_thinking_off_forms[1:]:
                    payload.pop("thinking", None)
                    payload.pop("thinking_mode", None)
                    payload.update(shape)
                    log_messages.append(
                        "DeepSeek rejected the thinking-disable parameter; retrying with an alternate form"
                    )
                    try:
                        result = _post(payload)
                        break
                    except httpx.HTTPStatusError as retry_err:
                        if retry_err.response.status_code != 400:
                            raise
                if result is None:
                    payload.pop("thinking", None)
                    payload.pop("thinking_mode", None)
                    log_messages.append(
                        "DeepSeek rejected all thinking-disable forms; retrying without them"
                    )
                    result = _post(payload)
            elif "reasoning_effort" in payload:
                log_messages.append(
                    "reasoning_effort unsupported by this model; retrying without it"
                )
                payload.pop("reasoning_effort", None)
                _MODELS_REJECTING_REASONING.add(model_name)
                result = _post(payload)
            else:
                raise

        tool_calls = []
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    if tc.get("type") == "function":
                        tool_calls.append(
                            {
                                "name": tc["function"]["name"],
                                "arguments": json.loads(tc["function"]["arguments"]),
                            }
                        )

        if len(tool_calls) > 4:
            log_messages.append(f"OpenAI returned {len(tool_calls)} tool calls; capping to first 4")
            tool_calls = tool_calls[:4]

        if not tool_calls:
            text_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            log_messages.append(f"OpenAI did not call tools. Response: {text_response[:200]}")
            return {"error": "AI did not call any tools", "ai_response": text_response}

        log_messages.append(f"OpenAI called {len(tool_calls)} tools")
        return {"tool_calls": tool_calls}

    except httpx.ReadTimeout:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"OpenAI request timed out after {timeout} seconds")
        log_messages.append(
            f"Request timed out after {timeout} seconds. Consider increasing AI_REQUEST_TIMEOUT_SECONDS environment variable."
        )
        return {
            "error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except httpx.TimeoutException:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning("OpenAI request timed out", exc_info=True)
        log_messages.append(f"Request timed out after {timeout} seconds.")
        return {
            "error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except Exception:
        logger.exception("Error calling OpenAI with tools")
        return {"error": "OpenAI service is currently unavailable."}


def _is_droppable_arg(key: str, value) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return True
    return key in _ZEROABLE_ARGS and value == 0


def _clean_call_arguments(tc: Dict, name: str, log_messages: List[str]) -> None:
    if "arguments" not in tc:
        tc["arguments"] = {}
    elif not isinstance(tc["arguments"], dict):
        log_messages.append(f"Coerced non-dict arguments for tool '{name}' to empty dict")
        tc["arguments"] = {}
    args = tc["arguments"]
    for k in [k for k, v in args.items() if _is_droppable_arg(k, v)]:
        log_messages.append(f"   Stripped empty/default arg '{k}={args[k]}' from {name}")
        del args[k]


def _validate_tool_calls(
    tool_calls: List[Dict],
    known_names: set,
    log_messages: List[str],
) -> Dict:
    """Validate and clean a list of raw tool-call dicts against the tool registry.

    Returns ``{"tool_calls": [...], "reasoning": "..."}`` on success or
    ``{"error": "..."}`` with a specific reason.
    """
    valid_calls: List[Dict] = []
    unknown_names: List[str] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict) or "name" not in tc:
            log_messages.append(f"WARN: Skipping invalid tool call: {tc}")
            continue
        name = tc.get("name", "")
        if name not in known_names:
            unknown_names.append(name)
            log_messages.append(
                f"WARN: Unknown tool '{name}' (known: {sorted(known_names)}); dropped"
            )
            continue
        _clean_call_arguments(tc, name, log_messages)
        valid_calls.append(tc)

    if unknown_names:
        msg = (
            f"Unknown tool(s) requested: {', '.join(unknown_names)}. "
            f"Use only: {', '.join(sorted(known_names))}."
        )
        log_messages.append(f"ERROR: {msg}")
        return {"error": msg}

    if not valid_calls:
        return {"error": "No valid tool calls found in Ollama response"}

    log_messages.append(f"OK: Ollama returned {len(valid_calls)} valid tool calls")
    return {"tool_calls": valid_calls}


def _strip_thinking(cleaned: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = (
            cleaned.split(THINK_END_TAG)[-1].strip()
            if THINK_END_TAG in cleaned
            else re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
        )
    return cleaned


def _extract_json_fence(cleaned: str) -> str:
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    return cleaned.strip()


def _tool_calls_from_parsed(parsed, log_messages: List[str]):
    """Return (tool_calls, error): exactly one is non-None."""
    if isinstance(parsed, dict) and "tool_calls" in parsed:
        return parsed["tool_calls"], None
    if isinstance(parsed, list):
        log_messages.append("WARN: Got array directly (expected object with tool_calls field)")
        return parsed, None
    if isinstance(parsed, dict) and "name" in parsed:
        log_messages.append("WARN: Got single tool call object (expected tool_calls array)")
        return [parsed], None
    if isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
        log_messages.append("WARN: Remapped {'tool','arguments'} -> {'name','arguments'} format")
        return [{"name": parsed["tool"], "arguments": parsed["arguments"]}], None
    keys = list(parsed.keys()) if isinstance(parsed, dict) else "N/A"
    log_messages.append(f"WARN: Unexpected JSON structure: {type(parsed)}, keys: {keys}")
    return None, {"error": "Ollama response missing 'tool_calls' field"}


def _parse_ollama_tool_response(
    response_text: str,
    log_messages: List[str],
    tools: List[Dict],
) -> Dict:
    """Parse Ollama's structured JSON response, validate tool names, and clean args.

    Returns ``{"tool_calls": [...], "reasoning": "..."}`` on success or
    ``{"error": "..."}`` with an exact failure reason for the retry feedback loop.
    """
    cleaned = ""
    known_names = {t.get("name") for t in tools if t.get("name")}
    try:
        cleaned = _strip_thinking(response_text.strip())
        log_messages.append(f"Ollama raw response (first 300 chars): {cleaned[:300]}")
        cleaned = _extract_json_fence(cleaned)

        if (
            cleaned.startswith("{")
            and '"tool_calls"' not in cleaned
            and '"properties"' in cleaned
        ):
            log_messages.append("WARN: Ollama returned the schema instead of tool calls")
            return {"error": "Ollama returned schema definition instead of tool calls"}

        parsed = json.loads(cleaned)

        reasoning = None
        if isinstance(parsed, dict):
            r = parsed.get("reasoning")
            if isinstance(r, str) and r.strip():
                reasoning = r.strip()

        tool_calls, error = _tool_calls_from_parsed(parsed, log_messages)
        if error:
            return error
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        result = _validate_tool_calls(tool_calls, known_names, log_messages)
        if "tool_calls" in result and reasoning:
            result["reasoning"] = reasoning
        return result

    except json.JSONDecodeError:
        logger.exception("JSON decode error while parsing Ollama tool response")
        log_messages.append("X: Failed to parse Ollama JSON response.")
        log_messages.append(f"Attempted to parse: {cleaned[:300]}")
        return {
            "error": "Failed to parse Ollama JSON response.",
            "raw_response": response_text[:200],
        }
    except Exception:
        logger.exception("Failed to parse Ollama response")
        log_messages.append("Failed to parse Ollama response.")
        log_messages.append(f"Response was: {response_text[:200]}")
        return {"error": "Failed to parse Ollama tool calls", "raw_response": response_text}


def _coerce_ollama_tool_args(raw_args, name: str, log_messages: List[str]) -> Dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            log_messages.append(
                f"WARN: Could not parse arguments for tool '{name}', using empty dict"
            )
            return {}
        return args if isinstance(args, dict) else {}
    return {}


def _try_native_ollama_tool_call(
    chat_url: str,
    model_name: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
    library_context: Optional[Dict],
    timeout: int,
) -> Optional[Dict]:
    """Attempt native Ollama /api/chat tool-calling (Hermes template path).

    Returns ``{"tool_calls": [...]}`` on success, or ``None`` when the model
    emitted no tool calls (caller should fall back to structured output).
    """
    from tasks.ai.prompts import build_mcp_system_prompt  # noqa: E402

    system_prompt = build_mcp_system_prompt(tools, library_context)
    ollama_tools = _tool_function_specs(tools)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "tools": ollama_tools,
        "stream": False,
        "options": {
            "temperature": config.AI_TOOLCALL_TEMPERATURE,
            "top_p": 0.8,
            "top_k": 20,
            "num_predict": 1536,
        },
    }
    if "qwen" in (model_name or "").lower():
        payload["think"] = False

    log_messages.append("Attempting native Ollama /api/chat tool-calling...")
    with httpx.Client(timeout=timeout) as client:
        response = client.post(chat_url, json=payload)
        response.raise_for_status()
        result = response.json()

    message = result.get("message", {})
    raw_tool_calls = message.get("tool_calls") or []

    if not raw_tool_calls:
        content = message.get("content", "")
        log_messages.append(
            f"Native tool-calling returned 0 tool calls; "
            f"model responded with text (first 150 chars): {content[:150]}"
        )
        return None

    tool_calls: List[Dict] = []
    for tc in raw_tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        tool_calls.append(
            {"name": name, "arguments": _coerce_ollama_tool_args(fn.get("arguments"), name, log_messages)}
        )

    known_names = {t.get("name") for t in tools if t.get("name")}
    validated = _validate_tool_calls(tool_calls, known_names, log_messages)
    if "tool_calls" in validated:
        log_messages.append(
            f"OK: Native tool-calling returned {len(validated['tool_calls'])} tool(s)"
        )
        return validated
    log_messages.append(
        f"Native tool-calling validation failed: {validated.get('error', 'unknown')}"
    )
    return None


def _try_structured_ollama_call(
    generate_url: str,
    model_name: str,
    prompt: str,
    tools: List[Dict],
    log_messages: List[str],
    timeout: int,
) -> Dict:
    """Fallback: prompt-based JSON emission constrained by format=<schema>."""
    from tasks.ai.prompts import build_tool_calls_schema  # noqa: E402

    schema = build_tool_calls_schema(tools)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "think": False,
        "options": {
            "temperature": config.AI_TOOLCALL_TEMPERATURE,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "num_predict": 1536,
        },
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(generate_url, json=payload)
        response.raise_for_status()
        result = response.json()

    if "response" not in result:
        return {"error": "Invalid Ollama response"}

    return _parse_ollama_tool_response(result["response"], log_messages, tools)


def call_with_tools_ollama(
    ollama_url: str,
    model_name: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
    library_context: Optional[Dict] = None,
) -> Dict:
    """Single-turn tool-calling for Ollama with native API first, structured fallback.

    Strategy (two paths, one feedback retry each):
    1. PRIMARY: Native /api/chat with ``tools`` parameter (Hermes template Qwen
       was trained on). Disables thinking via ``enable_thinking=false``.
    2. FALLBACK: Prompt-based JSON emission on /api/generate constrained by
       ``format=<schema>``, used when native path returns no tool calls or when
       the user's URL is already a /api/generate endpoint.
    """
    from tasks.ai.prompts import build_ollama_tool_calling_prompt  # noqa: E402

    is_chat_url = "/api/chat" in ollama_url.lower()
    is_generate_url = "/api/generate" in ollama_url.lower()

    # Determine the actual endpoints for each path
    if is_chat_url:
        chat_url = ollama_url
        generate_url = re.sub(r"/api/chat", "/api/generate", ollama_url, flags=re.IGNORECASE)
    elif is_generate_url:
        chat_url = re.sub(r"/api/generate", "/api/chat", ollama_url, flags=re.IGNORECASE)
        generate_url = ollama_url
    else:
        chat_url = ollama_url.rstrip("/") + "/api/chat"
        generate_url = ollama_url.rstrip("/") + "/api/generate"

    timeout = config.AI_REQUEST_TIMEOUT_SECONDS
    log_messages.append(f"Using timeout: {timeout} seconds for Ollama request")

    # -- Primary path: native tool-calling via /api/chat --
    if not is_generate_url:
        try:
            result = _try_native_ollama_tool_call(
                chat_url, model_name, user_message, tools,
                log_messages, library_context, timeout,
            )
            if result is not None and "tool_calls" in result:
                return result
            log_messages.append("Native path returned no tool calls; falling back to format=schema")
        except Exception:
            logger.warning(
                "Native Ollama tool-calling failed; falling back to structured output", exc_info=True
            )
            log_messages.append("Native /api/chat tool-calling failed; falling back to format=schema")

    # -- Fallback path: prompt-based JSON with format=<schema> --
    log_messages.append("Using Ollama structured-output (format=schema) path")
    try:
        base_prompt = build_ollama_tool_calling_prompt(user_message, tools, library_context)
        prompt = base_prompt
        last_result: Dict = {"error": "Ollama returned no usable tool calls"}
        for attempt in range(2):
            last_result = _try_structured_ollama_call(
                generate_url, model_name, prompt, tools,
                log_messages, timeout,
            )
            if "tool_calls" in last_result:
                if attempt > 0:
                    log_messages.append("OK: structured retry with feedback produced a valid plan")
                return last_result
            if attempt == 0:
                err = last_result.get("error", "invalid output")
                log_messages.append(f"Retrying once with feedback: {err}")
                prompt = (
                    f"{base_prompt}\n\nYour previous reply was invalid ({err}). "
                    "Return ONLY the JSON object in the required shape."
                )
        return last_result

    except httpx.ReadTimeout:
        log_messages.append(
            f"Ollama request timed out after {timeout} seconds. Your model or hardware may be too slow."
        )
        log_messages.append(
            "TIP: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value (e.g., 600 for 10 minutes)"
        )
        return {
            "error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except httpx.TimeoutException:
        log_messages.append(f"Ollama request timed out after {timeout} seconds.")
        log_messages.append(
            "TIP: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value"
        )
        return {
            "error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except Exception:
        logger.exception("Error calling Ollama with tools")
        return {"error": "Ollama service is currently unavailable."}
