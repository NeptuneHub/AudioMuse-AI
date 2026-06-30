import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import ftfy

import config
from tasks.ai.providers import (
    gemini as ai_api_gemini,
    mistral as ai_api_mistral,
    openai as ai_api_openai,
)
from tasks.ai.prompts import build_mcp_system_prompt

logger = logging.getLogger(__name__)

VALID_PROVIDERS = {"OLLAMA", "OPENAI", "GEMINI", "MISTRAL", "NONE"}


def validate_ai_config(ai_config: Dict) -> Tuple[bool, Optional[str]]:
    provider = (ai_config.get("provider") or "NONE").upper()

    if provider not in VALID_PROVIDERS:
        msg = f"Unknown AI provider {provider!r}. Valid: {sorted(VALID_PROVIDERS)}"
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
        url = ai_config.get("openai_url") or ""
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
            msg = (
                "Provider=MISTRAL but the mistralai SDK is not installed "
                "(currently quarantined on PyPI). Pick a different provider."
            )
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


def generate_text(
    prompt: str,
    ai_config: Dict,
    *,
    skip_delay: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    valid, err = validate_ai_config(ai_config)
    if not valid:
        return f"Error: {err}"

    provider = (ai_config.get("provider") or "NONE").upper()

    if provider == "NONE":
        return "AI Naming Skipped"
    if provider == "OLLAMA":
        return ai_api_openai.generate_text(
            ai_config["ollama_url"],
            ai_config["ollama_model"],
            prompt,
            api_key="no-key-needed",
            skip_delay=skip_delay,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "OPENAI":
        return ai_api_openai.generate_text(
            ai_config["openai_url"],
            ai_config["openai_model"],
            prompt,
            ai_config["openai_key"],
            skip_delay=skip_delay,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "GEMINI":
        return ai_api_gemini.generate_text(
            ai_config["gemini_key"],
            ai_config["gemini_model"],
            prompt,
            skip_delay=skip_delay,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "MISTRAL":
        return ai_api_mistral.generate_text(
            ai_config["mistral_key"],
            ai_config["mistral_model"],
            prompt,
            skip_delay=skip_delay,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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
        return ai_api_openai.call_with_tools_ollama(
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


def clean_playlist_name(name: str) -> str:
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
    MIN_LENGTH = 5
    MAX_LENGTH = 40

    max_songs = config.MAX_SONGS_IN_AI_PROMPT
    songs_for_prompt = song_list[:max_songs]
    formatted_song_list = "\n".join(
        [
            f"- {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}"
            for song in songs_for_prompt
        ]
    )
    if len(song_list) > max_songs:
        logger.info(
            "Truncated song list from %d to %d songs for AI prompt to avoid token limits",
            len(song_list),
            max_songs,
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
