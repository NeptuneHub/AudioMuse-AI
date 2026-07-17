# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Provider-agnostic AI entry point and text cleanup for the playlist AI.

Facade over the ``providers`` subpackage: validates the AI config, then
routes ``generate_text`` and ``call_with_tools`` to the selected vendor
(Ollama/OpenAI/Gemini/Mistral). Also owns playlist-name generation and the
Unicode text hygiene that sibling modules rely on.

Main Features:
* validate_ai_config gates each provider (URL shape, key, model) before any call is made.
* clean_playlist_name repairs mojibake with ftfy + NFKC and strips non-ASCII.
* get_ai_playlist_name asks small local models for one grounded concept, then
  composes and validates the final title in code.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import ftfy

from tasks.ai.providers import (
    gemini as ai_api_gemini,
    mistral as ai_api_mistral,
    openai as ai_api_openai,
)
from tasks.ai.prompts import build_mcp_system_prompt, playlist_concept_prompt_template

logger = logging.getLogger(__name__)

VALID_PROVIDERS = {"OLLAMA", "OPENAI", "GEMINI", "MISTRAL", "NONE"}

_PLAYLIST_DIMENSION_RULES = {
    'contrast': (
        "Return the one ordinary emotion word for feeling both sides at once. "
        "Do not return a rhetorical device or two labels."
    ),
    'function': (
        "Complete this listener intent: I play this music for ___. Return only the "
        "missing one-word purpose noun. Weather, place, object, scenery, and sound "
        "descriptions are invalid."
    ),
    'theme': (
        "Return one single familiar topic noun unifying the lyrics. No adjective, "
        "phrase, metaphor, or mood label."
    ),
    'relationship': (
        "Return one single familiar relationship-topic noun that captures both "
        "the romantic subject and its emotional tone. Ignoring either part is "
        "invalid. No adjective, phrase, metaphor, or mood label."
    ),
    'mood': (
        "Return one familiar mood adjective that reads naturally immediately "
        "before the genre. Do not return a noun, phrase, or copied sound descriptor."
    ),
}

_CONCEPT_CONTAINER_WORDS = {
    'background', 'collection', 'hits', 'instrumental', 'instrumentals', 'mix',
    'music', 'playlist', 'sessions', 'songs', 'sound', 'soundtracks', 'tunes',
    'vibes',
}

_CONCEPT_FILLER_WORDS = {'human'}

_CONTRAST_NON_FEELINGS = {
    'antithesis', 'contrast', 'dichotomy', 'duality', 'irony', 'juxtaposition',
    'oxymoron', 'paradox', 'synthesis',
}

_FUNCTION_NON_NOUNS = {'relax'}

_FUNCTION_BAD_TITLE_WORDS = {'friends', 'movement', 'rest', 'work'}

_FUNCTION_SOUND_DESCRIPTORS = {
    'aggressive', 'calm', 'danceable', 'energetic', 'energy', 'forceful',
    'happy', 'intense', 'joyful', 'relaxed', 'somber', 'upbeat',
}

_MOOD_BAD_TITLE_WORDS = {'joy', 'swing'}


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


def _compose_title(
    display_concept: str, genre_word: str, naming_dimension: str, instrumental: bool
) -> str:
    if naming_dimension in {'mood', 'contrast'}:
        title = f'{display_concept} {genre_word}'
    else:
        title = f'{genre_word} {display_concept}'
    if instrumental:
        title += ' Instrumentals'
    return title


def _concept_problem_basic(raw_concept, concept, words, concept_tokens, naming_dimension):
    if '\n' in raw_concept or '\r' in raw_concept:
        return 'it contained more than one line'
    if not concept:
        return 'the concept is empty'
    if len(concept) > 24:
        return 'the concept is too long'
    if concept_tokens & _CONCEPT_CONTAINER_WORDS:
        return 'container or marketing words are not a concept'
    if concept_tokens & _CONCEPT_FILLER_WORDS:
        return 'the concept contains a redundant filler word'
    if naming_dimension == 'contrast' and concept_tokens & _CONTRAST_NON_FEELINGS:
        return 'a rhetorical term is not an emotion'
    if '&' in concept:
        return 'the concept combined multiple labels'
    if naming_dimension == 'mood' and concept_tokens & _MOOD_BAD_TITLE_WORDS:
        return 'the word is not a mood adjective grounded by the evidence'
    if len(words) != 1:
        return f'{naming_dimension} concepts must be one word'
    return None


def _concept_problem_composed(concept, concept_tokens, title, naming_dimension, taken):
    if naming_dimension == 'function' and (
        concept_tokens & _FUNCTION_NON_NOUNS
        or any(token.endswith('ing') for token in concept_tokens)
    ):
        return 'a verb or gerund is not a purpose noun'
    if naming_dimension == 'function' and concept_tokens & _FUNCTION_BAD_TITLE_WORDS:
        return 'the purpose word does not form a natural playlist title'
    if naming_dimension == 'function' and concept_tokens & _FUNCTION_SOUND_DESCRIPTORS:
        return 'a sound descriptor is not a listening purpose'
    if any(
        concept_tokens & set(re.findall(r"[a-z0-9]+", used_title))
        for used_title in taken
    ):
        return 'the naming concept is already used by another playlist'
    if not (5 <= len(title) <= 40) or not (2 <= len(title.split()) <= 5):
        return 'the composed title is outside the 5-40 character or 2-5 word limit'
    if title.casefold() in taken:
        return 'the composed title is already used'
    return None


def get_ai_playlist_name(
    genre_word: str,
    naming_dimension: str,
    naming_evidence: str,
    ai_config: Dict,
    instrumental: bool = False,
    avoid_names: Optional[List[str]] = None,
) -> Optional[str]:
    dimension_rule = _PLAYLIST_DIMENSION_RULES.get(naming_dimension)
    if not dimension_rule:
        logger.error("Unsupported playlist naming dimension: %s", naming_dimension)
        return None

    avoid_list = avoid_names or []
    taken = {clean_playlist_name(name).casefold() for name in avoid_list}
    recent_names = [clean_playlist_name(name) for name in avoid_list[-8:]]
    avoid_rule = ""
    if recent_names:
        avoid_rule = (
            "Already used titles: "
            + " | ".join(recent_names)
            + ". Do not reuse their naming concept, even with another genre. "
        )
    full_prompt = playlist_concept_prompt_template.format(
        genre=genre_word,
        evidence=naming_evidence,
        dimension_rule=dimension_rule,
        avoid_rule=avoid_rule,
    )
    provider = (ai_config.get("provider") or "NONE").upper()
    logger.info("Sending playlist concept prompt to AI (%s):\n%s", provider, full_prompt)

    genre_pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(genre_word)}(?![a-z0-9])",
        re.IGNORECASE,
    )
    max_retries = 3
    current_prompt = full_prompt

    for attempt in range(max_retries):
        raw_concept = generate_text(
            current_prompt,
            ai_config,
            temperature=0.18,
            max_tokens=20,
        )

        if not isinstance(raw_concept, str):
            return None
        if raw_concept == "AI Naming Skipped" or raw_concept.startswith("Error"):
            return None

        concept = clean_playlist_name(raw_concept)
        concept = genre_pattern.sub('', concept)
        concept = re.sub(r"\s+", " ", concept).strip(' -')
        words = concept.split()
        concept_tokens = set(re.findall(r"[a-z0-9]+", concept.casefold()))
        title = _compose_title(concept.title(), genre_word, naming_dimension, instrumental)

        problem = _concept_problem_basic(
            raw_concept, concept, words, concept_tokens, naming_dimension
        ) or _concept_problem_composed(
            concept, concept_tokens, title, naming_dimension, taken
        )
        if problem is None:
            logger.info(
                "AI playlist concept '%s' composed as '%s' (%s)",
                concept,
                title,
                naming_dimension,
            )
            return title

        logger.warning(
            "AI generated playlist concept '%s' rejected because %s. Attempt %d/%d",
            concept,
            problem,
            attempt + 1,
            max_retries,
        )
        current_prompt = (
            full_prompt
            + f" Previous concept '{concept}' was rejected because {problem}. "
            "Return a different valid concept only."
        )
    return None
