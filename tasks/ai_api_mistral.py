"""Mistral transport (uses mistralai SDK)."""
import json
import logging
import os
import time
from typing import Dict, List

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
    "quarantined on PyPI — pick a different AI provider (Gemini / OpenAI / "
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
) -> str:
    """Single-prompt completion via Mistral chat.complete."""
    if not _MISTRAL_AVAILABLE:
        logger.error("Mistral provider selected but SDK is not installed: %s",
                     _MISTRAL_IMPORT_ERROR)
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

        response = client.chat.complete(
            model=model_name,
            temperature=0.9,
            timeout_ms=960,
            messages=[{"role": "user", "content": full_prompt}],
        )

        if response and response.choices[0].message.content:
            extracted_text = response.choices[0].message.content
            logger.info("Mistral API returned: '%s'", extracted_text)
            return extracted_text
        logger.warning("Mistral returned no content. Raw response: %s", response)
        return "Error: mistral returned no content."

    except Exception as e:
        logger.error("Error calling Mistral API: %s", e, exc_info=True)
        return "Error: AI service is currently unavailable."
