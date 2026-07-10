# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Prompt and JSON-schema templates for the playlist AI.

Central store of every prompt string the AI layer sends: the playlist-naming
template, the single-call tool-router system prompt, the Ollama tool-calling
prompt, and the grounded brainstorm recipe prompt. Consumed by ``planner``,
``api``, ``tool_impl``, and providers.

Main Features:
* Tool prose and the Ollama structured-output grammar are both DERIVED from the get_mcp_tools schemas (names, descriptions, per-argument types and enums), so the routing knowledge lives in one place and stays in sync across providers.
* build_tool_calls_schema emits a typed per-tool grammar (reasoning field first with a hard maxLength, name+arguments branches with enum-locked labels) used to constrain Ollama structured output; prompts stay short with a few diverse worked examples per intent class, including exclusion ('no rap') and language/scene routing rules.
"""

import copy
import json
from typing import Dict, List, Optional

import config


creative_prompt_template = (
    "You are an expert music collector and MUST give a title to this playlist.\n"
    "The title MUST represent the mood and the activity of when you are listening to the playlist.\n"
    "The title MUST use ONLY standard ASCII (a-z, A-Z, 0-9, spaces, and - & ' ! . , ? ( ) [ ]).\n"
    "The title MUST be within the range of 5 to 40 characters long.\n"
    "No special fonts or emojis.\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long/descriptive)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct/literal, not evocative enough)\n"
    "* BAD EXAMPLES: '\\U0001D5DD\\U0001D5C2\\U0001D5C8 \\U0001D5C2\\U0001D5CB\\U0001D5C8\\U0001D5C7\\U0001D5C2 \\U0001D5C9\\U0001D5CB\\U0001D5C8\\U0001D5C7\\U0001D5C2' (Non-standard characters)\n\n"
    "CRITICAL: Your response MUST be ONLY the single playlist name. No explanations, no 'Playlist Name:', no numbering, no extra text or formatting whatsoever.\n\n"
    "This is the playlist:\n{song_list_sample}\n\n"
)


def _get_dynamic_genres(library_context: Optional[Dict]) -> str:
    if library_context and library_context.get('top_genres'):
        return ', '.join(library_context['top_genres'][:10])
    return config.AI_FALLBACK_GENRES


def _render_tool_line(tool: Dict) -> str:
    props = (tool.get('inputSchema') or {}).get('properties') or {}
    args = ", ".join(props.keys())
    return f"- {tool['name']}({args}): {tool['description']}"


def build_mcp_system_prompt(
    tools: List[Dict],
    library_context: Optional[Dict] = None,
) -> str:
    tool_names = {t.get('name') for t in tools}
    tools_block = "\n".join(_render_tool_line(t) for t in tools)

    genres_line = _get_dynamic_genres(library_context)
    voices_line = ", ".join(v for v in config.VOICE_VOCAB if v.endswith('vocalists'))
    moods_line = ", ".join(config.OTHER_FEATURE_LABELS)

    rules: List[str] = []
    finder_options: List[str] = []
    if 'seed_search' in tool_names:
        finder_options.append(
            "the user names a song/artist to imitate ('like X', 'similar to X') -> seed_search"
        )
    if 'text_match' in tool_names:
        finder_options.append(
            "the user describes a sound or a lyric topic -> text_match"
        )
    if 'knowledge_lookup' in tool_names:
        finder_options.append(
            "the user asks for popular/famous/'best of' songs without naming a "
            "specific artist, or for a language/nationality/scene ('French rap', "
            "'Italian pop', 'K-pop') -> knowledge_lookup"
        )
    finder_options.append(
        "the request is plain metadata only -> search_database by itself"
    )
    rules.append("Pick how to FIND songs: " + "; ".join(finder_options) + ".")
    rules.append(
        "Put EVERY stated metadata constraint (genre, voice, mood, year/decade, tempo, "
        "energy, key, scale, rating, artist, album, instrumental) into ONE search_database "
        "call, next to the finder tool when there is one."
    )
    rules.append(
        "Exclusions ('no X', 'without X', 'except X', 'anything but X') go in "
        "search_database exclude_artists/exclude_genres. NEVER put an excluded name in "
        "seeds, in a text_match query, or in the positive artist/genres fields."
    )
    if 'seed_search' in tool_names:
        rules.append(
            "An artist's own songs ('songs by X', 'play X', 'best of X', where X is an "
            "artist's name) -> search_database with artist='X'. Similar to X ('like X', "
            "'sounds like X', 'in the style of X') -> seed_search."
        )
    rules.append(
        "Fill only fields the user asked for, using the closest listed value; when no "
        "field or listed value fits a word, leave it out."
    )
    rules.append("Emit each tool at most once; one finder plus one filter is the usual plan.")
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, start=1))

    return f"""You are a music playlist planner. Turn the user's request into tool calls; the app runs them against the user's own music library.

TOOLS:
{tools_block}

VALUES for search_database:
- genres in this library: {genres_line}
- voices: {voices_line}
- moods: {moods_line}
- scale: major or minor. key: tonic note like C or F# (major/minor goes in scale).
- Decade words map to years: '90s' -> year_min 1990, year_max 1999. energy 0.0-1.0 (calm <= 0.35, intense >= 0.7). tempo 40-200 BPM (slow <= 90, fast >= 130). min_rating 1-5.

HOW TO PLAN:
{rules_block}"""


def _example(reasoning: str, calls: List[Dict]) -> str:
    return json.dumps({"reasoning": reasoning, "tool_calls": calls}, ensure_ascii=True)


def _text_match_modes(tools: List[Dict]) -> set:
    for t in tools:
        if t.get('name') == 'text_match':
            props = (t.get('inputSchema') or {}).get('properties') or {}
            return set((props.get('mode') or {}).get('enum') or [])
    return set()


def _build_examples(tools: List[Dict]) -> List[str]:
    tool_names = {t.get('name') for t in tools}
    modes = _text_match_modes(tools)

    examples: List[str] = []
    if 'search_database' in tool_names:
        examples.append(
            '"energetic songs by Johnny Cash"\n'
            + _example(
                "Johnny Cash's own songs, filtered to high energy.",
                [
                    {
                        "name": "search_database",
                        "arguments": {"artist": "Johnny Cash", "energy_min": 0.65},
                    }
                ],
            )
        )
        examples.append(
            '"aggressive metal from the 80s"\n'
            + _example(
                "Pure metadata: genre metal, mood aggressive, decade 1980s.",
                [
                    {
                        "name": "search_database",
                        "arguments": {
                            "genres": ["metal"],
                            "moods": ["aggressive"],
                            "year_min": 1980,
                            "year_max": 1989,
                        },
                    }
                ],
            )
        )
        examples.append(
            '"party songs but absolutely no rap and nothing by Pitbull"\n'
            + _example(
                "Party mood with a genre and an artist exclusion.",
                [
                    {
                        "name": "search_database",
                        "arguments": {
                            "moods": ["party"],
                            "exclude_genres": ["Hip-Hop"],
                            "exclude_artists": ["Pitbull"],
                        },
                    }
                ],
            )
        )
    if 'seed_search' in tool_names and 'search_database' in tool_names:
        examples.append(
            '"like Get Lucky by Daft Punk but with a female voice"\n'
            + _example(
                "Songs similar to a named track, constrained to female vocals.",
                [
                    {
                        "name": "seed_search",
                        "arguments": {
                            "seeds": [
                                {"type": "song", "title": "Get Lucky", "artist": "Daft Punk"}
                            ]
                        },
                    },
                    {
                        "name": "search_database",
                        "arguments": {"voices": ["female vocalists"]},
                    },
                ],
            )
        )
    if 'seed_search' in tool_names:
        examples.append(
            '"in the style of Oasis but not Blur"\n'
            + _example(
                "Similar to one named artist while removing another's flavor.",
                [
                    {
                        "name": "seed_search",
                        "arguments": {
                            "seeds": [{"type": "artist", "name": "Oasis"}],
                            "blend_mode": "subtract",
                            "subtract": [{"type": "artist", "name": "Blur"}],
                        },
                    }
                ],
            )
        )
    if 'text_match' in tool_names and 'audio' in modes:
        examples.append(
            '"soft acoustic guitar for studying"\n'
            + _example(
                "A sound description, matched by how the music sounds.",
                [
                    {
                        "name": "text_match",
                        "arguments": {"query": "soft acoustic guitar for studying", "mode": "audio"},
                    }
                ],
            )
        )
    if 'text_match' in tool_names and 'lyrics' in modes:
        examples.append(
            '"songs about growing old"\n'
            + _example(
                "A lyric topic, matched by what the words are about.",
                [
                    {
                        "name": "text_match",
                        "arguments": {"query": "growing old", "mode": "lyrics"},
                    }
                ],
            )
        )
    if 'knowledge_lookup' in tool_names:
        examples.append(
            '"greatest disco hits of the 70s"\n'
            + _example(
                "A popularity request that needs world knowledge.",
                [
                    {
                        "name": "knowledge_lookup",
                        "arguments": {"user_request": "greatest disco hits of the 70s"},
                    }
                ],
            )
        )
    return examples


def build_ollama_tool_calling_prompt(
    user_message: str,
    tools: List[Dict],
    library_context: Optional[Dict] = None,
) -> str:
    system_prompt = build_mcp_system_prompt(tools, library_context)
    examples_text = "\n\n".join(_build_examples(tools))

    return f"""{system_prompt}

OUTPUT: return ONLY one JSON object, no other text:
{{"reasoning": "one short sentence: what to find and which tools", "tool_calls": [{{"name": "tool_name", "arguments": {{...}}}}]}}

EXAMPLES:
{examples_text}

Request: "{user_message}"
Fill only fields the user asked for. Return ONLY the JSON object."""


def _inject_unique_items(schema: Dict) -> Dict:
    """Add ``uniqueItems: True`` to every array-typed property recursively.

    Small Ollama models can loop the same value forever in structured-output
    mode; ``uniqueItems`` prevents that.  Injected here rather than in the
    shared ``tools.py`` schema so that only the Ollama structured-output path
    is affected -- Gemini, Mistral, and native OpenAI tool-calling use the
    schema as-is.
    """
    if not isinstance(schema, dict):
        return schema
    # Snapshot keys so we can safely mutate the dict while walking it.
    for key, value in list(schema.items()):
        if key == 'type' and value == 'array':
            schema.setdefault('uniqueItems', True)
        elif isinstance(value, dict):
            _inject_unique_items(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _inject_unique_items(item)
    return schema


def build_tool_calls_schema(tools: List[Dict]) -> Dict:
    branches: List[Dict] = []
    for t in tools:
        name = t.get('name')
        if not name:
            continue
        arg_schema = _inject_unique_items(
            copy.deepcopy(t.get('inputSchema') or {"type": "object"})
        )
        arg_schema['additionalProperties'] = False
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "enum": [name]},
                    "arguments": arg_schema,
                },
                "required": ["name", "arguments"],
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "maxLength": 300,
                "description": "One short sentence: what to find and which tools.",
            },
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"oneOf": branches} if branches else {"type": "object"},
            },
        },
        "required": ["reasoning", "tool_calls"],
    }


def build_ai_brainstorm_prompt(user_request: str) -> str:
    genres_line = ", ".join(config.STRATIFIED_GENRES)
    moods_line = ", ".join(config.OTHER_FEATURE_LABELS)
    voices_line = ", ".join(config.VOICE_VOCAB)
    return f"""You are a music expert. Turn the request into a RECIPE used to search a music library.
You do NOT know which songs are in the library, so you MUST NOT name any songs. Describe and categorise only; the library does the finding.

User request: "{user_request}"

Return ONE JSON object with EXACTLY this shape:
{{"filters": {{"genres": [], "moods": [], "voices": [], "year_min": null, "year_max": null, "energy_min": null, "energy_max": null, "tempo_min": null, "tempo_max": null}}, "sound_descriptions": [], "seed_artists": [], "lyric_themes": []}}

FIELD GUIDE (leave a field empty/null when the request does not imply it -- never invent constraints):
- filters.genres: 0+ values, chosen ONLY from: {genres_line}
- filters.moods: 0+ values, chosen ONLY from: {moods_line}
- filters.voices: 0+ values, chosen ONLY from: {voices_line}
- filters.year_min / year_max: 4-digit years. A decade like "90s" -> 1990 and 1999. "90s and 2000s" -> 1990 and 2009.
- filters.energy_min / energy_max: numbers 0.0 (calm) to 1.0 (intense).
- filters.tempo_min / tempo_max: BPM, 40 to 200.
- sound_descriptions: 2 to {config.AI_BRAINSTORM_SOUND_DESCRIPTIONS_MAX} vivid phrases describing HOW the ideal songs SOUND (instruments, production, era, energy, vibe). This is the most important field. NOT song names.
- seed_artists: up to {config.AI_BRAINSTORM_SEED_ARTISTS_MAX} well-known ARTISTS that exemplify the request. Artists ONLY, never songs. Omit if none are obvious.
- lyric_themes: 0 to {config.AI_BRAINSTORM_LYRIC_THEMES_MAX} short phrases ONLY when the request is about a TOPIC the lyrics should cover (e.g. "heartbreak", "summer roadtrip").

RULES:
- NEVER output a song title anywhere.
- genres / moods / voices MUST come from the lists above, or be left empty.
- Output ONLY the JSON object. No markdown fences, no comments, no extra text.

EXAMPLE -- request "100 of the best rap songs from the 90s and 2000s":
{{"filters": {{"genres": ["Hip-Hop"], "moods": [], "voices": [], "year_min": 1990, "year_max": 2009, "energy_min": 0.5, "energy_max": 1.0, "tempo_min": null, "tempo_max": null}}, "sound_descriptions": ["gritty 90s east coast boom bap hip hop with hard-hitting drums and jazzy samples", "glossy early 2000s mainstream rap with heavy bass and crossover hooks"], "seed_artists": ["Nas", "Jay-Z", "2Pac", "Eminem"], "lyric_themes": []}}

Now produce the JSON recipe for "{user_request}":"""
