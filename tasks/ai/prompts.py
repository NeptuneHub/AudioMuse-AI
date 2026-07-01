# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Prompt and JSON-schema templates for the playlist AI.

Central store of every prompt string the AI layer sends: the playlist-naming
template, the intent classifier, the MCP tool-router system prompt, the
Ollama tool-calling prompt with worked examples, and the grounded brainstorm
recipe prompt. Consumed by ``planner``, ``api``, ``tool_impl``, and providers.

Main Features:
* Prompts are built dynamically from config vocab (genres/voices/moods) and library_context so allowed values stay in sync with the catalog.
* build_tool_calls_schema emits the strict JSON shape (1-4 tool calls, name enum) used to constrain Ollama structured output.
"""

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


INTENT_CLASSES = ["seed", "text", "knowledge", "metadata"]

PRIMARY_INTENTS = ["seed", "text", "knowledge"]


def _get_dynamic_genres(library_context: Optional[Dict]) -> str:
    if library_context and library_context.get('top_genres'):
        return ', '.join(library_context['top_genres'][:10])
    return config.AI_FALLBACK_GENRES


def build_mcp_system_prompt(
    tools: List[Dict],
    library_context: Optional[Dict] = None,
) -> str:
    tool_names = {t['name'] for t in tools}
    has_seed = 'seed_search' in tool_names
    has_text = 'text_match' in tool_names
    has_knowledge = 'knowledge_lookup' in tool_names

    tool_lines: List[str] = []
    if has_seed:
        tool_lines.append(
            "- seed_search(seeds[], blend_mode?, subtract?): songs from one or more SEED songs/artists. "
            "Seeds can be a mix of songs and artists. Use blend_mode='union' (default) for "
            "'similar to A and B'; 'alchemy' for 'A meets B' (2+ seeds); 'subtract' for 'A but not Y'."
        )
    if has_text:
        tool_lines.append(
            "- text_match(query, mode?): semantic text search. mode='audio' (default) for sound/instruments "
            "('calm piano'); mode='lyrics' for lyrical themes ('about heartbreak in the rain'). "
            "NOT for year/genre/mood (use search_database)."
        )
    if has_knowledge:
        tool_lines.append(
            "- knowledge_lookup(user_request): popularity / 'best of' / cultural requests "
            "('best rap of the 90s', 'top festival anthems', 'Grammy winners 2020'). Turns the "
            "request into a grounded library search (genre/year/energy + sound descriptions + "
            "seed artists); never invents song titles."
        )
    tool_lines.append(
        "- search_database(genres?, voices?, moods?, year_min?, year_max?, min_rating?, scale?, "
        "key?, tempo_min?, tempo_max?, energy_min?, energy_max?, artist?, album?): "
        "metadata filter. Can stand alone OR refine any primary pool."
    )
    tools_block = "\n".join(tool_lines)

    genres_line = _get_dynamic_genres(library_context)
    voices_line = ", ".join(config.VOICE_VOCAB)
    moods_line = ", ".join(config.OTHER_FEATURE_LABELS)

    prompt = f"""You are a music playlist router. Return ONLY a JSON object with one or more tool calls. Put EVERY intent in this one response.

TOOLS:
{tools_block}

search_database tag columns are SEPARATE -- do not mix:
- genres : music styles -> {genres_line}
- voices : vocal type -> {voices_line}
- moods  : ONLY these 6 -> {moods_line}
scale: major|minor. year: single year sets year_min=year_max; decade 80s -> 1980..1989. energy 0.0-1.0. tempo 40-200 BPM.

RULES:
1. Only filters the user explicitly mentioned; never invent.
2. seeds: a named TRACK -> {{type:'song',title,artist}} (e.g. "Iron Maiden Run to the Hills" = title 'Run to the Hills', artist 'Iron Maiden'); a bare artist -> {{type:'artist',name}}. Multiple in ONE seed_search: "A and B"=union, "A meets B"=alchemy, "A but not Y"=subtract.
3. ANY descriptor beyond the song/artist (mood, genre, vocal, energy, tempo, year/decade, key, scale) MUST ALSO go in a search_database call. Even one trailing word: "...danceable" -> seed_search(...) AND search_database(moods=["danceable"]).
4. voices: "female voice"/"woman singer" -> ["female vocalists","female vocalist"]; "male voice" -> ["male vocalists"]. Never put a voice or genre in 'moods'.
5. "2024 songs" -> search_database(year_min=2024,year_max=2024), not text_match.
6. A topic/scenario the song should be ABOUT ("about summer", "roadtrip", "songs about heartbreak") -> text_match(mode='lyrics'); any genre/voice/energy/tempo/year mentioned ALONGSIDE -> ALSO a search_database call. Keep BOTH."""

    return prompt


def build_ollama_tool_calling_prompt(
    user_message: str,
    tools: List[Dict],
    library_context: Optional[Dict] = None,
) -> str:
    system_prompt = build_mcp_system_prompt(tools, library_context)

    examples = []
    examples.append(
        '"Similar song to Red Hot Chili Peppers - By The Way and Iron Maiden Run to the Hills"\n'
        '{{"tool_calls": [{{"name": "seed_search", "arguments": {{'
        '"seeds": ['
        '{{"type": "song", "title": "By The Way", "artist": "Red Hot Chili Peppers"}}, '
        '{{"type": "song", "title": "Run to the Hills", "artist": "Iron Maiden"}}'
        '], "blend_mode": "union", "get_songs": 1000}}}}]}}'
    )
    examples.append(
        '"sounds like Iron Maiden and Metallica combined"\n'
        '{{"tool_calls": [{{"name": "seed_search", "arguments": {{'
        '"seeds": ['
        '{{"type": "artist", "name": "Iron Maiden"}}, '
        '{{"type": "artist", "name": "Metallica"}}'
        '], "blend_mode": "alchemy", "get_songs": 200}}}}]}}'
    )
    examples.append(
        '"Similar to By The Way by Red Hot Chili Peppers danceable"\n'
        '{{"tool_calls": ['
        '{{"name": "seed_search", "arguments": {{"seeds": [{{"type": "song", "title": "By The Way", "artist": "Red Hot Chili Peppers"}}], "get_songs": 1000}}}}, '
        '{{"name": "search_database", "arguments": {{"moods": ["danceable"]}}}}'
        ']}}'
    )
    examples.append(
        '"calm piano songs"\n'
        '{{"tool_calls": [{{"name": "text_match", "arguments": {{"query": "calm piano", "mode": "audio"}}}}]}}'
    )
    examples.append(
        '"upbeat pop roadtrip songs about summer with female vocals"\n'
        '{{"tool_calls": ['
        '{{"name": "text_match", "arguments": {{"query": "summer roadtrip", "mode": "lyrics"}}}}, '
        '{{"name": "search_database", "arguments": {{"genres": ["pop"], "voices": ["female vocalists", "female vocalist"], "energy_min": 0.55}}}}'
        ']}}'
    )
    examples_text = "\n\n".join(examples)

    return f"""/no_think
{system_prompt}

=== OUTPUT FORMAT (CRITICAL) ===
Return ONLY a valid JSON object with this EXACT format:
{{
  "tool_calls": [
    {{"name": "tool_name", "arguments": {{"param": "value"}}}}
  ]
}}

=== EXAMPLES ===
{examples_text}

=== COMMON MISTAKES ===
WRONG: only seed_search when a descriptor was added -> also emit search_database
WRONG: putting a voice/genre in 'moods' -> moods is danceable/aggressive/happy/party/relaxed/sad ONLY
WRONG: repeating the same tool -> emit each tool AT MOST once; usually ONE tool call is enough. Output the JSON and STOP.

Do not reason or explain. Go straight to the JSON.
Now analyze this request and return ONLY the JSON:
Request: "{user_message}"
"""


def build_intent_classifier_prompt(user_message: str) -> str:
    return f"""Classify a music request. Return ONLY JSON: {{"primaries": ["seed"|"text"|"knowledge", ...], "needs_filter": true|false}}.

primaries (zero or more, the ways to FIND songs):
- seed: names specific song(s)/artist(s) to find similar/blend/subtract ("similar to By The Way by RHCP", "songs like Madonna").
- text: describes the SOUND, or a LYRIC/TOPIC/SCENARIO theme -- what the song is ABOUT ("calm piano", "songs about heartbreak", "roadtrip songs about summer").
- knowledge: popularity/cultural/historical request ("top pop songs of 2025", "Grammy winners", "songs sampled by Daft Punk").

THEME PRECEDENCE: a lyric/topic/scenario theme is a "text" primary and STAYS even when genre/voice/energy/tempo/year are also present -- those become the filter, they do NOT remove "text".
A popularity word (top/best/popular/radio/Grammy/viral/#1/charts) makes it "knowledge" even with a year/genre: "top rock 2020" = knowledge, "rock 2020" = no primary.
Multiple primaries can co-occur (e.g. a named song AND a theme -> ["seed","text"]).

needs_filter: true when a metadata constraint (year/genre/mood/vocal/tempo/energy/scale/rating) is present.
A PURE metadata filter with no theme/song/artist/popularity -> {{"primaries": [], "needs_filter": true}}.

EXAMPLES:
"songs like Madonna" -> {{"primaries": ["seed"], "needs_filter": false}}
"similar to Pink Floyd with female voice" -> {{"primaries": ["seed"], "needs_filter": true}}
"top pop radio songs of 2025" -> {{"primaries": ["knowledge"], "needs_filter": true}}
"sad jazz from the 90s" -> {{"primaries": [], "needs_filter": true}}
"upbeat pop roadtrip songs about summer with female vocals" -> {{"primaries": ["text"], "needs_filter": true}}
"pop instrumental" -> {{"primaries": [], "needs_filter": true}}
"instrumental jazz" -> {{"primaries": [], "needs_filter": true}}

Request: "{user_message}"
JSON:"""


def build_tool_calls_schema(tools: List[Dict]) -> Dict:
    tool_names = [t['name'] for t in tools if t.get('name')]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "enum": tool_names},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
        },
        "required": ["tool_calls"],
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
