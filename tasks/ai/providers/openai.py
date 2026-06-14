"""OpenAI-compatible transport (OpenAI, OpenRouter, Ollama, anything speaking /v1/chat/completions or /api/generate).

Three public functions:
    generate_text(...)          -- single-prompt streaming completion (all providers)
    call_with_tools(...)        -- native function/tool calling (OpenAI, OpenRouter, etc.)
    call_with_tools_ollama(...) -- prompt-based JSON tool calling (Ollama, no native support)

These transports only handle HTTP plumbing. All business prompts come from
`tasks/ai/prompts.py`.
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


def _is_ollama_format_url(server_url: str) -> bool:
    """Detect Ollama endpoints from the URL path (issue #467 fix preserved)."""
    s = server_url.lower()
    return "/api/generate" in s or "/api/chat" in s


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
    """Generate freeform text from an OpenAI-compatible streaming endpoint.

    Handles 429 retries (exponential backoff), 400 ``unsupported_parameter`` /
    ``unsupported_value`` fallback (drop temperature, switch to
    max_completion_tokens, then drop max_completion_tokens), and content
    extraction-before-finish-reason ordering for OpenRouter compatibility.

    NOTE: Detects Ollama vs OpenAI from the URL; the Ollama branch in
    tasks/ai/api.py calls this directly (Ollama has no separate transport).
    """
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
            "reasoning_effort": "low" if is_deepseek else "none",
        }
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
                            # Extract content FIRST (OpenRouter may bundle delta.content
                            # with finish_reason='stop' in a single chunk).
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

            thought_enders = ["</think>", "[/INST]", "[/THOUGHT]"]
            extracted_text = full_raw_response_content.strip()
            for end_tag in thought_enders:
                if end_tag in extracted_text:
                    extracted_text = extracted_text.split(end_tag, 1)[-1].strip()

            if extracted_text:
                # SECURITY: log only length, not content (model output may
                # echo back sensitive data from prompts/tool results).
                logger.info(
                    "%s API returned non-empty content (length=%d chars).",
                    provider_label, len(extracted_text),
                )
                return extracted_text
            logger.warning(
                "%s returned empty content (raw response length: %d chars).",
                provider_label, len(full_raw_response_content),
            )
            logger.debug(
                "Raw SSE stream metadata: %d lines received; preview suppressed to avoid sensitive data logging.",
                len(raw_sse_lines),
            )
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt)
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
                    sleep_time = base_delay * (2 ** attempt)
                    logger.info("Retrying in %s seconds...", sleep_time)
                    time.sleep(sleep_time)
                    continue

            if e.response.status_code == 400 and is_openai_format:
                try:
                    error_body = e.response.json()
                    error_code = error_body.get("error", {}).get("code", "")
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
                logger.error(
                    "Error calling OpenAI-compatible API: %s. Response body: %s",
                    e,
                    error_detail,
                    exc_info=True,
                )
            except Exception:
                logger.error("Error calling OpenAI-compatible API: %s", e, exc_info=True)
            return "Error: AI service is currently unavailable."

        except requests.exceptions.RequestException as e:
            logger.error("Error calling OpenAI-compatible API: %s", e, exc_info=True)
            return "Error: AI service is currently unavailable."
        except Exception:
            logger.error(
                "An unexpected error occurred in ai_api_openai.generate_text", exc_info=True
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
    """Call an OpenAI-compatible /chat/completions endpoint with native tool calling.

    Returns ``{"tool_calls": [...]}`` on success, ``{"error": "..."}`` on failure.
    """
    try:
        functions = [
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
            # Bound generation so a model that won't stop (e.g. a reasoning model
            # that thinks unbounded) can't run forever. Generic OpenAI param,
            # works on any OpenAI-compatible provider (OpenRouter, vLLM, ...).
            # Matches the local Ollama num_predict cap (1024) for consistency.
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
        else:
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
                    log_messages.append("DeepSeek rejected the thinking-disable parameter; retrying with an alternate form")
                    try:
                        result = _post(payload)
                        break
                    except httpx.HTTPStatusError as retry_err:
                        if retry_err.response.status_code != 400:
                            raise
                if result is None:
                    payload.pop("thinking", None)
                    payload.pop("thinking_mode", None)
                    log_messages.append("DeepSeek rejected all thinking-disable forms; retrying without them")
                    result = _post(payload)
            elif "reasoning_effort" in payload:
                log_messages.append("reasoning_effort unsupported by this model; retrying without it")
                payload.pop("reasoning_effort", None)
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

        # Max-items cap: never process a runaway list of tool calls (a tool plan
        # needs only a few). Mirrors the Ollama schema maxItems; generic here.
        if len(tool_calls) > 4:
            log_messages.append(f"OpenAI returned {len(tool_calls)} tool calls; capping to first 4")
            tool_calls = tool_calls[:4]

        if not tool_calls:
            text_response = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            log_messages.append(
                f"OpenAI did not call tools. Response: {text_response[:200]}"
            )
            return {"error": "AI did not call any tools", "ai_response": text_response}

        log_messages.append(f"OpenAI called {len(tool_calls)} tools")
        return {"tool_calls": tool_calls}

    except httpx.ReadTimeout:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"OpenAI request timed out after {timeout} seconds")
        log_messages.append(
            f"\u23f1\ufe0f Request timed out after {timeout} seconds. Consider increasing AI_REQUEST_TIMEOUT_SECONDS environment variable."
        )
        return {
            "error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except httpx.TimeoutException:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning("OpenAI request timed out", exc_info=True)
        log_messages.append(f"\u23f1\ufe0f Request timed out after {timeout} seconds.")
        return {
            "error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except Exception:
        logger.exception("Error calling OpenAI with tools")
        return {"error": "OpenAI service is currently unavailable."}


# ---------------------------------------------------------------------------
# Ollama prompt-based tool calling (Ollama lacks native function calling)
# ---------------------------------------------------------------------------

def call_with_tools_ollama(
    ollama_url: str,
    model_name: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
    library_context: Optional[Dict] = None,
) -> Dict:
    """Prompt-based tool calling for Ollama via /api/generate with structured output.

    Ollama has no native function/tool calling. We build a JSON-output prompt
    (via ``tasks.ai.prompts.build_ollama_tool_calling_prompt``) and set
    ``format`` to the tool-calls JSON Schema so the model is forced to emit
    valid JSON that we parse into ``{"tool_calls": [...]}``.

    Returns ``{"tool_calls": [...]}`` on success, ``{"error": "..."}`` on failure.
    """
    from tasks.ai.prompts import build_ollama_tool_calling_prompt, build_tool_calls_schema  # noqa: E402

    try:
        prompt = build_ollama_tool_calling_prompt(user_message, tools, library_context)

        schema = build_tool_calls_schema(tools)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "think": False,
            # Cap generation so a model can't run forever. A tool call needs
            # only a few hundred tokens.
            "options": {"temperature": 0, "num_predict": 1024},
        }

        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        log_messages.append(f"Using timeout: {timeout} seconds for Ollama request")
        # Single bounded call: the httpx read timeout aborts it at `timeout`
        # seconds so it can never run forever. NO retry -- if the model returns
        # nothing usable, we error out and the chat pipeline falls back (the user
        # still gets a playlist) rather than making a second multi-minute call.
        with httpx.Client(timeout=timeout) as client:
            response = client.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()

        if "response" not in result:
            return {"error": "Invalid Ollama response"}

        response_text = result["response"]

        cleaned = ""
        try:
            cleaned = response_text.strip()
            cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
            if "<think>" in cleaned:
                cleaned = (
                    cleaned.split("</think>")[-1].strip()
                    if "</think>" in cleaned
                    else re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
                )

            log_messages.append(f"Ollama raw response (first 300 chars): {cleaned[:300]}")

            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]
            cleaned = cleaned.strip()

            if cleaned.startswith("{") and '"type"' in cleaned and '"array"' in cleaned:
                log_messages.append(
                    "\u26a0\ufe0f Ollama returned schema instead of tool calls, using fallback"
                )
                return {"error": "Ollama returned schema definition instead of tool calls"}

            log_messages.append(f"Attempting to parse: {cleaned[:200]}")
            parsed = json.loads(cleaned)

            if isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls = parsed["tool_calls"]
                log_messages.append(
                    f"\u2713 Extracted tool_calls array with {len(tool_calls) if isinstance(tool_calls, list) else 1} items"
                )
            elif isinstance(parsed, list):
                tool_calls = parsed
                log_messages.append("\u26a0\ufe0f Got array directly (expected object with tool_calls field)")
            elif isinstance(parsed, dict) and "name" in parsed:
                tool_calls = [parsed]
                log_messages.append(
                    "\u26a0\ufe0f Got single tool call object (expected object with tool_calls array)"
                )
            elif isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
                tool_calls = [{"name": parsed["tool"], "arguments": parsed["arguments"]}]
                log_messages.append(
                    "\u26a0\ufe0f Remapped {'tool','arguments'} -> {'name','arguments'} format"
                )
            else:
                log_messages.append(
                    f"\u26a0\ufe0f Unexpected JSON structure: {type(parsed)}, keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}"
                )
                return {"error": "Ollama response missing 'tool_calls' field"}

            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            valid_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict) and "name" in tc:
                    if "arguments" not in tc:
                        tc["arguments"] = {}
                    args = tc["arguments"]
                    keys_to_remove = []
                    for k, v in args.items():
                        if v is None or v == "" or v == [] or v == {}:
                            keys_to_remove.append(k)
                        elif k in ("tempo_min", "tempo_max", "energy_min", "min_rating") and v == 0:
                            keys_to_remove.append(k)
                    for k in keys_to_remove:
                        log_messages.append(
                            f"   \U0001f9f9 Stripped empty/default arg '{k}={args[k]}' from {tc['name']}"
                        )
                        del args[k]
                    valid_calls.append(tc)
                else:
                    log_messages.append(f"\u26a0\ufe0f Skipping invalid tool call: {tc}")

            if not valid_calls:
                return {"error": "No valid tool calls found in Ollama response"}

            log_messages.append(f"\u2705 Ollama returned {len(valid_calls)} valid tool calls")
            return {"tool_calls": valid_calls}

        except json.JSONDecodeError:
            logger.exception("JSON decode error while parsing Ollama tool response")
            log_messages.append("\u274c Failed to parse Ollama JSON response.")
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

    except httpx.ReadTimeout:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"Ollama request timed out after {timeout} seconds")
        log_messages.append(
            f"\u23f1\ufe0f Ollama request timed out after {timeout} seconds. Your model or hardware may be too slow."
        )
        log_messages.append(
            "\U0001f4a1 Solution: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value (e.g., 600 for 10 minutes)"
        )
        return {
            "error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except httpx.TimeoutException:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning("Ollama request timed out", exc_info=True)
        log_messages.append(f"\u23f1\ufe0f Ollama request timed out after {timeout} seconds.")
        log_messages.append(
            "\U0001f4a1 Solution: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value"
        )
        return {
            "error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except Exception:
        logger.exception("Error calling Ollama with tools")
        return {"error": "Ollama service is currently unavailable."}
