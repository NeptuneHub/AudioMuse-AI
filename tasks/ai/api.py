"""Public AI API facade.

Every AI call in AudioMuse-AI MUST go through this module. Callers pass an
``ai_config`` dict (built from the user's request payload combined with
``config.py`` defaults) and one of the public entry points below.

Detection rule (per user spec):
    1. ``ai_config['provider']`` is the source of truth.
    2. We then validate that the URL/keys configured for that provider are
       consistent with the chosen provider. If they are not, we log a clear
       error and return ``"Error: ..."`` -- we do NOT silently fall back to a
       different provider.

Public entry points:
    validate_ai_config(ai_config)
        Returns ``(is_valid, error_message)``.
    generate_text(prompt, ai_config, *, skip_delay=False)
        Single-prompt freeform text completion.
    call_with_tools(user_message, tools, ai_config, *, system_prompt=None,
                    library_context=None, log_messages=None)
        Provider-native (or prompt-based for Ollama) tool calling. Returns
        ``{"tool_calls": [...]}`` or ``{"error": "..."}``.
    clean_playlist_name(name)
        Sanitize an AI-generated playlist name to standard ASCII.
    get_ai_playlist_name(prompt_template, song_list, other_feature_scores_dict, ai_config)
        High-level helper used by clustering/analysis: format prompt, call
        ``generate_text``, validate length (5-40 chars), retry up to 3 times
        with feedback, return cleaned name.
"""
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import ftfy

from config import MAX_SONGS_IN_AI_PROMPT
from tasks.ai.providers import (
    gemini as ai_api_gemini,
    mistral as ai_api_mistral,
    ollama as ai_api_ollama,
    openai as ai_api_openai,
)
from tasks.ai.prompts import build_mcp_system_prompt

logger = logging.getLogger(__name__)

VALID_PROVIDERS = {"OLLAMA", "OPENAI", "GEMINI", "MISTRAL", "NONE"}


# ---------------------------------------------------------------------------
# Provider detection / validation
# ---------------------------------------------------------------------------

def validate_ai_config(ai_config: Dict) -> Tuple[bool, Optional[str]]:
    """Strictly validate that ``ai_config`` matches its declared provider.

    Returns ``(True, None)`` on success, or ``(False, error_message)`` if the
    provider is unknown or its required URL/key is missing/inconsistent. We
    always log_error on failure and never fall back to another provider.

    SECURITY: ``ai_config`` may contain API keys. Error messages returned to
    the caller include the offending URL/provider for diagnostics, but the
    ``logger.error`` lines log ONLY a static identifier so CodeQL's
    clear-text-logging taint analysis cannot flag a leak via this dict.
    """
    provider = (ai_config.get("provider") or "NONE").upper()

    if provider not in VALID_PROVIDERS:
        msg = (
            f"Unknown AI provider {provider!r}. Valid: {sorted(VALID_PROVIDERS)}"
        )
        logger.error("validate_ai_config: unknown provider")
        return False, msg

    if provider == "NONE":
        return True, None

    if provider == "OLLAMA":
        url = (ai_config.get("ollama_url") or "").lower()
        if not url:
            msg = "Provider=OLLAMA but ollama_url is empty"
            logger.error("validate_ai_config: OLLAMA url empty")
            return False, msg
        if not ("/api/generate" in url or "/api/chat" in url):
            msg = (
                f"Provider=OLLAMA but URL {ai_config.get('ollama_url')!r} does not look like an Ollama endpoint "
                "(expected path /api/generate or /api/chat)"
            )
            logger.error("validate_ai_config: OLLAMA url path mismatch")
            return False, msg
        if not ai_config.get("ollama_model"):
            msg = "Provider=OLLAMA but ollama_model is empty"
            logger.error("validate_ai_config: OLLAMA model empty")
            return False, msg

    elif provider == "OPENAI":
        url = (ai_config.get("openai_url") or "")
        url_l = url.lower()
        key = ai_config.get("openai_key")
        if not url:
            msg = "Provider=OPENAI but openai_url is empty"
            logger.error("validate_ai_config: OPENAI url empty")
            return False, msg
        if "/api/generate" in url_l or "/api/chat" in url_l:
            msg = (
                f"Provider=OPENAI but URL {url!r} looks like an Ollama endpoint. "
                "OpenAI/OpenRouter URLs use /v1/chat/completions."
            )
            logger.error("validate_ai_config: OPENAI url looks like Ollama")
            return False, msg
        if not key or key == "no-key-needed":
            msg = "Provider=OPENAI but openai_key is missing"
            logger.error("validate_ai_config: OPENAI key missing")
            return False, msg
        if not ai_config.get("openai_model"):
            msg = "Provider=OPENAI but openai_model is empty"
            logger.error("validate_ai_config: OPENAI model empty")
            return False, msg

    elif provider == "GEMINI":
        key = ai_config.get("gemini_key")
        if not key or key == "YOUR-GEMINI-API-KEY-HERE":
            msg = "Provider=GEMINI but no API key configured"
            logger.error("validate_ai_config: GEMINI key missing")
            return False, msg
        if not ai_config.get("gemini_model"):
            msg = "Provider=GEMINI but gemini_model is empty"
            logger.error("validate_ai_config: GEMINI model empty")
            return False, msg

    elif provider == "MISTRAL":
        if not ai_api_mistral.is_available():
            msg = ("Provider=MISTRAL but the mistralai SDK is not installed "
                   "(currently quarantined on PyPI). Pick a different provider.")
            logger.error("validate_ai_config: mistralai SDK missing")
            return False, msg
        key = ai_config.get("mistral_key")
        if not key or key == "YOUR-MISTRAL-API-KEY-HERE":
            msg = "Provider=MISTRAL but no API key configured"
            logger.error("validate_ai_config: MISTRAL key missing")
            return False, msg
        if not ai_config.get("mistral_model"):
            msg = "Provider=MISTRAL but mistral_model is empty"
            logger.error("validate_ai_config: MISTRAL model empty")
            return False, msg

    return True, None


# ---------------------------------------------------------------------------
# Generic dispatchers
# ---------------------------------------------------------------------------

def generate_text(prompt: str, ai_config: Dict, *, skip_delay: bool = False,
                  temperature: Optional[float] = None) -> str:
    """Single-prompt freeform text completion.

    Returns the model's text on success, or a string starting with ``"Error: "``
    on failure. Returns ``"AI Naming Skipped"`` when provider is ``NONE``.

    ``temperature=None`` keeps each provider's creative default; pass a value
    (e.g. 0.0 for deterministic tasks like intent classification) to override.
    """
    valid, err = validate_ai_config(ai_config)
    if not valid:
        return f"Error: {err}"

    provider = (ai_config.get("provider") or "NONE").upper()

    if provider == "NONE":
        return "AI Naming Skipped"
    if provider == "OLLAMA":
        return ai_api_ollama.generate_text(
            ai_config["ollama_url"],
            ai_config["ollama_model"],
            prompt,
            skip_delay=skip_delay,
            temperature=temperature,
        )
    if provider == "OPENAI":
        return ai_api_openai.generate_text(
            ai_config["openai_url"],
            ai_config["openai_model"],
            prompt,
            ai_config["openai_key"],
            skip_delay=skip_delay,
            temperature=temperature,
        )
    if provider == "GEMINI":
        return ai_api_gemini.generate_text(
            ai_config["gemini_key"],
            ai_config["gemini_model"],
            prompt,
            skip_delay=skip_delay,
            temperature=temperature,
        )
    if provider == "MISTRAL":
        return ai_api_mistral.generate_text(
            ai_config["mistral_key"],
            ai_config["mistral_model"],
            prompt,
            skip_delay=skip_delay,
            temperature=temperature,
        )

    # Unreachable: validate_ai_config already rejects unknown providers.
    return f"Error: Unsupported provider {provider!r}"


def call_with_tools(
    user_message: str,
    tools: List[Dict],
    ai_config: Dict,
    *,
    system_prompt: Optional[str] = None,
    library_context: Optional[Dict] = None,
    log_messages: Optional[List[str]] = None,
) -> Dict:
    """Call AI with tool definitions and return its tool calls.

    If ``system_prompt`` is None, build the canonical MCP system prompt from
    ``tasks.ai.prompts.build_mcp_system_prompt(tools, library_context)``.

    Returns ``{"tool_calls": [...]}`` on success, ``{"error": "..."}`` on
    failure.
    """
    if log_messages is None:
        log_messages = []

    valid, err = validate_ai_config(ai_config)
    if not valid:
        return {"error": err}

    provider = (ai_config.get("provider") or "NONE").upper()

    if provider == "NONE":
        return {"error": "AI provider is NONE"}

    if system_prompt is None:
        system_prompt = build_mcp_system_prompt(tools, library_context)

    if provider == "OLLAMA":
        # Ollama builds its own JSON-output prompt internally; system_prompt is
        # ignored because the Ollama prompt contains the system text already.
        return ai_api_ollama.call_with_tools(
            ai_config["ollama_url"],
            ai_config["ollama_model"],
            user_message,
            tools,
            log_messages,
            library_context,
        )
    if provider == "OPENAI":
        return ai_api_openai.call_with_tools(
            ai_config["openai_url"],
            ai_config["openai_model"],
            ai_config["openai_key"],
            system_prompt,
            user_message,
            tools,
            log_messages,
        )
    if provider == "GEMINI":
        return ai_api_gemini.call_with_tools(
            ai_config["gemini_key"],
            ai_config["gemini_model"],
            system_prompt,
            user_message,
            tools,
            log_messages,
        )
    if provider == "MISTRAL":
        return ai_api_mistral.call_with_tools(
            ai_config["mistral_key"],
            ai_config["mistral_model"],
            system_prompt,
            user_message,
            tools,
            log_messages,
        )

    return {"error": f"Unsupported provider {provider!r}"}


# ---------------------------------------------------------------------------
# Playlist-naming helpers (high-level)
# ---------------------------------------------------------------------------

def clean_playlist_name(name: str) -> str:
    """Sanitize an AI-generated playlist name to the allowed ASCII subset."""
    if not isinstance(name, str):
        return ""
    name = ftfy.fix_text(name)
    name = unicodedata.normalize("NFKC", name)
    cleaned_name = re.sub(r"[^a-zA-Z0-9\s\-\&\'!\.\,\?\(\)\[\]]", "", name)
    cleaned_name = re.sub(r"\s\(\d+\)$", "", cleaned_name)
    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()
    return cleaned_name


def get_ai_playlist_name(
    prompt_template: str,
    song_list: List[Dict],
    other_feature_scores_dict: Optional[Dict],
    ai_config: Dict,
) -> str:
    """Generate a playlist name via AI, with length validation and retry.

    The prompt template MUST contain the ``{song_list_sample}`` placeholder.
    Names are cleaned and validated against MIN_LENGTH..MAX_LENGTH (5..40);
    on length failure the prompt is re-issued with feedback up to 3 times.
    """
    MIN_LENGTH = 5
    MAX_LENGTH = 40

    # Truncate song list to avoid token-limit issues
    songs_for_prompt = song_list[:MAX_SONGS_IN_AI_PROMPT]
    formatted_song_list = "\n".join(
        [
            f"- {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}"
            for song in songs_for_prompt
        ]
    )
    if len(song_list) > MAX_SONGS_IN_AI_PROMPT:
        logger.info(
            "Truncated song list from %d to %d songs for AI prompt to avoid token limits",
            len(song_list),
            MAX_SONGS_IN_AI_PROMPT,
        )

    full_prompt = prompt_template.format(song_list_sample=formatted_song_list)
    provider = (ai_config.get("provider") or "NONE").upper()
    logger.info("Sending prompt to AI (%s):\n%s", provider, full_prompt)

    max_retries = 3
    current_prompt = full_prompt

    for attempt in range(max_retries):
        name = generate_text(current_prompt, ai_config)

        if name in ("AI Naming Skipped",) or name.startswith("Error"):
            return name

        cleaned_name = clean_playlist_name(name)
        if MIN_LENGTH <= len(cleaned_name) <= MAX_LENGTH:
            return cleaned_name

        logger.warning(
            "AI generated name '%s' (%d chars) outside %d-%d range. Attempt %d/%d",
            cleaned_name,
            len(cleaned_name),
            MIN_LENGTH,
            MAX_LENGTH,
            attempt + 1,
            max_retries,
        )
        if attempt < max_retries - 1:
            feedback = (
                f"\n\nFEEDBACK: The previous title you generated ('{cleaned_name}') was "
                f"{len(cleaned_name)} characters long. It MUST be between {MIN_LENGTH} and "
                f"{MAX_LENGTH} characters. Please try again."
            )
            current_prompt = full_prompt + feedback
            continue
        return (
            f"Error: AI generated name '{cleaned_name}' ({len(cleaned_name)} chars) "
            f"outside {MIN_LENGTH}-{MAX_LENGTH} range after {max_retries} attempts."
        )

    return "Error: Max retries exceeded in get_ai_playlist_name"
