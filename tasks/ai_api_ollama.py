"""Ollama transport.

`generate_text` delegates to the OpenAI-compatible streaming code path (since
Ollama also exposes /api/generate streaming SSE).
"""
import logging
from typing import Dict, List, Optional

from tasks import ai_api_openai

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
