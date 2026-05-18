"""Ollama transport.

Ollama lacks native function calling, so `call_with_tools` builds a JSON-output
prompt (via `tasks.ai_prompts.build_ollama_tool_calling_prompt`) and parses the
model's JSON response. `generate_text` delegates to the OpenAI-compatible
streaming code path (since Ollama also exposes /api/generate streaming SSE).
"""
import json
import logging
import re
from typing import Dict, List, Optional

import httpx

import config
from tasks import ai_api_openai
from tasks.ai_prompts import build_ollama_tool_calling_prompt

logger = logging.getLogger(__name__)


def generate_text(
    ollama_url: str,
    model_name: str,
    full_prompt: str,
    *,
    skip_delay: bool = False,
) -> str:
    """Generate freeform text from an Ollama /api/generate endpoint.

    Reuses the OpenAI-compatible transport because the streaming code path is
    shared (it auto-detects Ollama format from the URL).
    """
    return ai_api_openai.generate_text(
        ollama_url, model_name, full_prompt, api_key="no-key-needed", skip_delay=skip_delay
    )


def call_with_tools(
    ollama_url: str,
    model_name: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
    library_context: Optional[Dict] = None,
) -> Dict:
    """Prompt-based tool calling for Ollama (no native function calling).

    Returns ``{"tool_calls": [...]}`` on success, ``{"error": "..."}`` on failure.
    """
    try:
        prompt = build_ollama_tool_calling_prompt(user_message, tools, library_context)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
        }

        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        log_messages.append(f"Using timeout: {timeout} seconds for Ollama request")
        with httpx.Client(timeout=timeout) as client:
            response = client.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()

        if "response" not in result:
            return {"error": "Invalid Ollama response"}

        response_text = result["response"]

        # Thinking models (e.g. Qwen 3.5) return empty response with format=json.
        # Retry without format constraint -- their response field will have clean JSON
        # and the thinking/reasoning stays in the separate 'thinking' field.
        if not response_text and result.get("thinking"):
            log_messages.append("\u2139\ufe0f Thinking model detected -- retrying without format=json")
            payload.pop("format", None)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(ollama_url, json=payload)
                response.raise_for_status()
                result = response.json()
            response_text = result.get("response", "")
            if response_text and "</think>" in response_text:
                response_text = response_text.split("</think>", 1)[-1].strip()

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

        except json.JSONDecodeError as e:
            log_messages.append(f"\u274c JSON decode error: {str(e)}")
            log_messages.append(f"Attempted to parse: {cleaned[:300]}")
            return {
                "error": f"Failed to parse Ollama JSON: {str(e)}",
                "raw_response": response_text[:200],
            }
        except Exception as e:
            log_messages.append(f"Failed to parse Ollama response: {str(e)}")
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
    except httpx.TimeoutException as e:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"Ollama request timed out: {str(e)}")
        log_messages.append(f"\u23f1\ufe0f Ollama request timed out after {timeout} seconds: {str(e)}")
        log_messages.append(
            "\U0001f4a1 Solution: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value"
        )
        return {
            "error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except Exception as e:
        logger.exception("Error calling Ollama with tools")
        return {"error": f"Ollama error: {str(e)}"}
