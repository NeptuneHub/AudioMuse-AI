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
* get_ai_playlist_name asks small local models for several grounded candidate
  concepts in a single call, then composes and validates the final title in
  code, keeping the first candidate that passes.
* Validation runs in two passes over the same candidates: the full rules first,
  then a second pass without the stem-collision rule, which is the rule that
  otherwise rejects most concepts once many playlists are already named. Every
  editorial ban list stays in force in both passes.
"""

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
from tasks.ai.playlist_namer import GENRE_DISPLAY
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
    'antithesis', 'contrast', 'dichotomy', 'dissonance', 'duality', 'irony',
    'juxtaposition', 'oxymoron', 'paradox', 'synthesis', 'contradiction',
    'euphoria', 'jubilance',
}

_FUNCTION_NON_NOUNS = {'relax'}

_FUNCTION_BAD_TITLE_WORDS = {
    'exercise', 'friends', 'movement', 'people', 'rest', 'work'
}

_FUNCTION_SOUND_DESCRIPTORS = {
    'aggressive', 'calm', 'danceable', 'energetic', 'energy', 'forceful',
    'happy', 'intense', 'joyful', 'relaxed', 'somber', 'upbeat',
}

_MOOD_BAD_TITLE_WORDS = {'joy', 'swing'}

_KNOWN_GENRE_TITLES = (
    {name.casefold() for name in GENRE_DISPLAY}
    | {name.casefold() for name in GENRE_DISPLAY.values()}
)

_CANDIDATE_SPLIT_PATTERN = re.compile(r"[,;/|\n\r]+")
_LIST_NUMBER_PATTERN = re.compile(r"^\d+\s*[\.\)]\s*")
_LEAD_IN_PATTERN = re.compile(r"^[^:]{0,40}:\s*")
_CANDIDATE_TRIM_CHARS = " *_-.!?()[]'\"`"


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
    name = re.sub(r"_automatic(\s*\(\d+\))?$", "", name, flags=re.IGNORECASE)
    name = name.replace("_", " ")
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


def _stem_token(token):
    if token.endswith('ies') and len(token) > 4:
        return token[:-3] + 'y'
    if token.endswith('s') and not token.endswith('ss') and len(token) > 3:
        return token[:-1]
    return token


def _concept_problem_composed(
    concept, concept_tokens, title, naming_dimension, taken, stem_collision=True
):
    if naming_dimension == 'function' and (
        concept_tokens & _FUNCTION_NON_NOUNS
        or any(token.endswith('ing') for token in concept_tokens)
    ):
        return 'a verb or gerund is not a purpose noun'
    if naming_dimension == 'function' and concept_tokens & _FUNCTION_BAD_TITLE_WORDS:
        return 'the purpose word does not form a natural playlist title'
    if naming_dimension == 'function' and concept_tokens & _FUNCTION_SOUND_DESCRIPTORS:
        return 'a sound descriptor is not a listening purpose'
    if title.removesuffix(' Instrumentals').casefold() in _KNOWN_GENRE_TITLES:
        return 'the composed title is just a genre name'
    concept_stems = {_stem_token(token) for token in concept_tokens}
    if stem_collision and any(
        concept_stems
        & {_stem_token(token) for token in re.findall(r"[a-z0-9]+", used_title)}
        for used_title in taken
    ):
        return 'the naming concept is already used by another playlist'
    if not (5 <= len(title) <= 40) or not (2 <= len(title.split()) <= 5):
        return 'the composed title is outside the 5-40 character or 2-5 word limit'
    if title.casefold() in taken:
        return 'the composed title is already used'
    return None


def _concept_problem_relaxed(concept, concept_tokens, title, naming_dimension, taken):
    return _concept_problem_basic(
        concept, concept, concept.split(), concept_tokens, naming_dimension
    ) or _concept_problem_composed(
        concept, concept_tokens, title, naming_dimension, taken, stem_collision=False
    )


def _parse_concept_candidates(raw_text: str, limit: int) -> List[str]:
    text = (raw_text or "").strip()
    if not text:
        return []
    candidates: List[str] = []
    seen = set()
    for index, chunk in enumerate(_CANDIDATE_SPLIT_PATTERN.split(text)):
        chunk = chunk.strip()
        if index == 0:
            chunk = _LEAD_IN_PATTERN.sub('', chunk)
        chunk = _LIST_NUMBER_PATTERN.sub('', chunk.strip())
        chunk = chunk.strip(_CANDIDATE_TRIM_CHARS)
        if not chunk or chunk.casefold() in seen:
            continue
        seen.add(chunk.casefold())
        candidates.append(chunk)
        if len(candidates) >= limit:
            break
    return candidates


def _evaluate_concept(
    raw_concept, genre_word, genre_pattern, naming_dimension, instrumental, taken
):
    concept = clean_playlist_name(raw_concept)
    concept = genre_pattern.sub('', concept)
    concept = re.sub(r"\s+", " ", concept).strip(" -.,!?()[]")
    words = concept.split()
    concept_tokens = set(re.findall(r"[a-z0-9]+", concept.casefold()))
    title = _compose_title(concept.title(), genre_word, naming_dimension, instrumental)
    problem = _concept_problem_basic(
        raw_concept, concept, words, concept_tokens, naming_dimension
    ) or _concept_problem_composed(
        concept, concept_tokens, title, naming_dimension, taken
    )
    return concept, title, problem


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
        candidate_count=config.AI_NAMING_CANDIDATES,
    )
    provider = (ai_config.get("provider") or "NONE").upper()
    logger.info("Sending playlist concept prompt to AI (%s):\n%s", provider, full_prompt)

    genre_pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(genre_word)}(?![a-z0-9])",
        re.IGNORECASE,
    )
    max_attempts = max(1, config.AI_NAMING_MAX_ATTEMPTS)
    rejected: List[str] = []
    relaxed_pairs: List[Tuple[str, str]] = []
    current_prompt = full_prompt

    for attempt in range(max_attempts):
        raw_response = generate_text(
            current_prompt,
            ai_config,
            temperature=0.7,
        )

        if raw_response == "AI Naming Skipped":
            return None
        if not isinstance(raw_response, str) or raw_response.startswith("Error"):
            logger.warning(
                "AI naming got no usable text from %s on attempt %d/%d; retrying",
                provider,
                attempt + 1,
                max_attempts,
            )
            continue

        for candidate in _parse_concept_candidates(
            raw_response, config.AI_NAMING_CANDIDATES
        ):
            concept, title, problem = _evaluate_concept(
                candidate,
                genre_word,
                genre_pattern,
                naming_dimension,
                instrumental,
                taken,
            )
            if problem is None:
                logger.info(
                    "AI playlist concept '%s' composed as '%s' (%s)",
                    concept,
                    title,
                    naming_dimension,
                )
                return title
            if concept:
                relaxed_pairs.append((concept, title))
                if concept.casefold() not in {word.casefold() for word in rejected}:
                    rejected.append(concept)
            logger.debug(
                "AI playlist concept '%s' rejected because %s. Attempt %d/%d",
                concept,
                problem,
                attempt + 1,
                max_attempts,
            )

        if rejected:
            current_prompt = (
                full_prompt
                + " Already rejected: "
                + ", ".join(rejected)
                + ". Return different words."
            )

    for concept, title in relaxed_pairs:
        concept_tokens = set(re.findall(r"[a-z0-9]+", concept.casefold()))
        if _concept_problem_relaxed(
            concept, concept_tokens, title, naming_dimension, taken
        ) is None:
            logger.info(
                "AI playlist concept '%s' accepted without the stem-collision rule "
                "as '%s' (%s)",
                concept,
                title,
                naming_dimension,
            )
            return title

    logger.warning(
        "No AI playlist concept passed validation for '%s' (%s). Rejected: %s",
        genre_word,
        naming_dimension,
        ", ".join(rejected) or "none",
    )
    return None
