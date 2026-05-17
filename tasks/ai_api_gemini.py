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

        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(temperature=0.9),
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
