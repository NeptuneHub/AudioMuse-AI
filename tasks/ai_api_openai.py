"""OpenAI-compatible transport (OpenAI, OpenRouter, anything speaking /v1/chat/completions).

Two public functions:
    generate_text(...)   -- single-prompt streaming completion (used for playlist naming, brainstorm, etc.)
    call_with_tools(...) -- non-streaming chat with tool/function calling

These transports only handle HTTP plumbing. All business prompts come from
`tasks/ai_prompts.py`.
"""
import json
import logging
import os
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
) -> str:
    """Generate freeform text from an OpenAI-compatible streaming endpoint.

    Handles 429 retries (exponential backoff), 400 ``unsupported_parameter`` /
    ``unsupported_value`` fallback (drop temperature, switch to
    max_completion_tokens, then drop max_completion_tokens), and content
    extraction-before-finish-reason ordering for OpenRouter compatibility.

    NOTE: Detects Ollama vs OpenAI from the URL for backward compatibility with
    `tasks/ai_api_ollama.generate_text` which delegates here.
    """
    is_ollama_format = _is_ollama_format_url(server_url)
    is_openai_format = not is_ollama_format

    headers = _build_openai_headers(api_key, server_url)

    if is_openai_format:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 8000,
        }
    else:
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {"num_predict": 8000, "temperature": 0.7},
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
                    "OpenAI/OpenRouter API returned non-empty content (length=%d chars).",
                    len(extracted_text),
                )
                return extracted_text
            logger.warning(
                "OpenAI/OpenRouter returned empty content (raw response length: %d chars).",
                len(full_raw_response_content),
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
                            payload["max_completion_tokens"] = 8000
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
        }

        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        log_messages.append(f"Using timeout: {timeout} seconds for OpenAI request")
        with httpx.Client(timeout=timeout) as client:
            response = client.post(server_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        tool_calls = []
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    if tc["type"] == "function":
                        tool_calls.append(
                            {
                                "name": tc["function"]["name"],
                                "arguments": json.loads(tc["function"]["arguments"]),
                            }
                        )

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
    except httpx.TimeoutException as e:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"OpenAI request timed out: {str(e)}")
        log_messages.append(f"\u23f1\ufe0f Request timed out after {timeout} seconds: {str(e)}")
        return {
            "error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except Exception as e:
        logger.exception("Error calling OpenAI with tools")
        return {"error": f"OpenAI error: {str(e)}"}
