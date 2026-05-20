"""Google Gemini transport (uses google-genai Client API)."""
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_text(
    api_key: str,
    model_name: str,
    full_prompt: str,
    *,
    skip_delay: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Single-prompt completion via Gemini's generate_content."""
    if not api_key or api_key == "YOUR-GEMINI-API-KEY-HERE":
        return "Error: Gemini API key is missing or empty. Please provide a valid API key."

    try:
        import google.genai as genai

        if not skip_delay:
            gemini_call_delay = int(os.environ.get("GEMINI_API_CALL_DELAY_SECONDS", "7"))
            if gemini_call_delay > 0:
                logger.debug(
                    "Waiting for %ss before Gemini API call to respect rate limits.",
                    gemini_call_delay,
                )
                time.sleep(gemini_call_delay)

        client = genai.Client(api_key=api_key)
        logger.debug("Starting API call for model '%s'.", model_name)

        temp = 0.9 if temperature is None else float(temperature)
        cfg_kwargs = {"temperature": temp}
        if max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = int(max_tokens)
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(**cfg_kwargs),
        )

        if response and hasattr(response, "text") and response.text:
            extracted_text = response.text
            logger.info("Gemini API returned: '%s'", extracted_text)
            return extracted_text
        logger.warning("Gemini returned no content. Raw response: %s", response)
        return "Error: Gemini returned no content."

    except Exception as e:
        logger.error("Error calling Gemini API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."


def call_with_tools(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
) -> Dict:
    """Call Gemini with native function calling. Returns ``{"tool_calls": [...]}`` or ``{"error": ...}``."""
    try:
        import google.genai as genai

        if not api_key or api_key == "YOUR-GEMINI-API-KEY-HERE":
            return {"error": "Valid Gemini API key required"}

        client = genai.Client(api_key=api_key)

        function_declarations = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            }
            for tool in tools
        ]

        tools_list = [genai.types.Tool(function_declarations=function_declarations)]

        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools_list,
                tool_config=genai.types.ToolConfig(
                    function_calling_config=genai.types.FunctionCallingConfig(mode="ANY")
                ),
                temperature=0,
            ),
        )

        log_messages.append(f"Gemini response type: {type(response)}")

        def convert_to_dict(obj):
            """Recursively convert protobuf objects to native Python types."""
            if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
                if hasattr(obj, "items"):
                    return {k: convert_to_dict(v) for k, v in obj.items()}
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            return obj

        tool_calls = []
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        args_dict = {}
                        if hasattr(fc, "args"):
                            args_dict = dict(fc.args) if fc.args else {}
                        elif hasattr(fc, "arguments"):
                            args_dict = (
                                fc.arguments if isinstance(fc.arguments, dict) else {}
                            )
                        tool_calls.append(
                            {"name": fc.name, "arguments": convert_to_dict(args_dict)}
                        )

        if not tool_calls:
            text_response = response.text if hasattr(response, "text") else str(response)
            log_messages.append(f"Gemini did not call tools. Response: {text_response[:200]}")
            return {"error": "AI did not call any tools", "ai_response": text_response}

        log_messages.append(f"Gemini called {len(tool_calls)} tools")
        return {"tool_calls": tool_calls}

    except Exception as e:
        logger.exception("Error calling Gemini with tools")
        return {"error": f"Gemini error: {str(e)}"}
