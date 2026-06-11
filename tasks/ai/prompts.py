"""Centralized AI prompt templates and prompt builders.

All business-level prompts used by the AudioMuse-AI features live here, so
that:

* `tasks/ai_api*.py` stay generic transports (no embedded prompts).
* Adding or tweaking a prompt does not require touching transport code.
* Migrating a feature to a new provider only requires plugging the existing
  prompt into the new transport.

Public entry points:
    creative_prompt_template          -- clustering / playlist naming prompt template
    build_mcp_system_prompt(...)      -- canonical MCP tool-decision system prompt
    build_ollama_tool_calling_prompt  -- Ollama-specific JSON-output framing
    build_tool_calls_schema(tools)    -- shared JSON Schema for tool_calls (used by every transport)
    build_ai_brainstorm_prompt(...)   -- ai_brainstorm MCP tool prompt
    get_dynamic_genres                -- helper, exposed for tests
"""
from typing import Dict, List, Optional

import config


# --- Clustering / playlist naming ---------------------------------------------

creative_prompt_template = (
    "You are an expert music collector and MUST give a title to this playlist.\n"
    "The title MUST represent the mood and the activity of when you are listening to the playlist.\n"
    "The title MUST use ONLY standard ASCII (a-z, A-Z, 0-9, spaces, and - & ' ! . , ? ( ) [ ]).\n"
    "The title MUST be within the range of 5 to 40 characters long.\n"
    "No special fonts or emojis.\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long/descriptive)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct/literal, not evocative enough)\n"
    "* BAD EXAMPLES: '\U0001D5DD\U0001D5C2\U0001D5C8 \U0001D5C2\U0001D5CB\U0001D5C8\U0001D5C7\U0001D5C2 \U0001D5C9\U0001D5CB\U0001D5C8\U0001D5C7\U0001D5C2' (Non-standard characters)\n\n"
    "CRITICAL: Your response MUST be ONLY the single playlist name. No explanations, no 'Playlist Name:', no numbering, no extra text or formatting whatsoever.\n\n"
    "This is the playlist:\n{song_list_sample}\n\n"
)


# --- MCP system prompt (used by all providers when calling with tools) --------

_FALLBACK_GENRES = (
    "rock, pop, metal, jazz, electronic, dance, alternative, indie, punk, blues, "
    "hard rock, heavy metal, hip-hop, funk, country, soul"
)


def get_dynamic_genres(library_context: Optional[Dict]) -> str:
    """Return genre list from library context, falling back to defaults."""
    if library_context and library_context.get('top_genres'):
        return ', '.join(library_context['top_genres'][:10])
    return _FALLBACK_GENRES


VOICE_VOCAB = ["female vocalists", "female vocalist", "male vocalists"]

INTENT_CLASSES = ["seed", "text", "knowledge", "metadata"]

PRIMARY_INTENTS = ["seed", "text", "knowledge"]


def build_mcp_system_prompt(tools: List[Dict], library_context: Optional[Dict] = None) -> str:
    """Build the canonical MCP system prompt used by ALL providers."""
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
            "- knowledge_lookup(user_request): cultural/historical world-knowledge fallback "
            "('Grammy winners 2020', 'songs sampled by Daft Punk'). LAST RESORT."
        )
    tool_lines.append(
        "- search_database(genres?, voices?, moods?, year_min?, year_max?, min_rating?, scale?, "
        "key?, tempo_min?, tempo_max?, energy_min?, energy_max?, artist?, album?): "
        "metadata filter. Can stand alone OR refine any primary pool."
    )
    tools_block = "\n".join(tool_lines)

    genres_line = get_dynamic_genres(library_context)
    voices_line = ", ".join(VOICE_VOCAB)
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


# --- Ollama-specific tool-calling framing -------------------------------------
#
# Ollama lacks native function calling, so we ask the model to emit a JSON
# object with a `tool_calls` array. The framing below is a transport-level
# adapter (it tells Ollama HOW to format its answer); the user-facing rules
# come from build_mcp_system_prompt() and are reused verbatim.

def build_ollama_tool_calling_prompt(
    user_message: str,
    tools: List[Dict],
    library_context: Optional[Dict] = None,
) -> str:
    """Build the full Ollama prompt that asks the model to emit JSON tool_calls."""
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
    """Tiny Stage-1 prompt: classify the request into PRIMARY intents + a filter flag.

    Returned JSON shape: {"primaries": [<subset of PRIMARY_INTENTS>], "needs_filter": bool}.
    A request may carry SEVERAL primaries (e.g. ["seed","text"]); a pure metadata
    filter has no primary -> {"primaries": [], "needs_filter": true}.
    """
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
    """JSON Schema for the {tool_calls:[...]} envelope, shared by every transport.

    Argument shapes are kept loose; ``tasks.ai.planner.validate_and_normalize_plan``
    + ``tasks.ai.vocab`` are the source of truth for canonical values.
    Long enums on argument fields slow GBNF/JSON-schema decoding without adding
    safety we don't already have in Python.
    """
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


# --- Free-text MCP prompts (brainstorm) ---------------------------------------

def build_ai_brainstorm_prompt(user_request: str) -> str:
    return f"""You are a music expert with extensive knowledge of songs, artists, and music history. 

User request: "{user_request}"

TASK: Use your knowledge to suggest 25-35 specific songs (with exact artist names) that match this request.

Think about:
- If they want songs similar to an artist \u2192 suggest songs by that artist AND similar artists
- If they want a genre/mood \u2192 suggest famous songs in that genre/mood
- If they want popular/radio hits \u2192 suggest well-known mainstream songs
- If they want a time period \u2192 suggest songs from that era
- If they want a vibe \u2192 suggest songs that match that feeling

CRITICAL REQUIREMENTS:
1. Return ONLY a JSON array of objects
2. Each object MUST have "title" and "artist" fields
3. Be specific with exact song titles and artist names (as they appear in databases)
4. Include variety - different artists when possible
5. Format: [{{"title": "Song Name", "artist": "Artist Name"}}, ...]
6. NO explanations, NO numbering, ONLY the JSON array

Example format:
[
  {{"title": "All the Small Things", "artist": "blink-182"}},
  {{"title": "Basket Case", "artist": "Green Day"}},
  {{"title": "American Idiot", "artist": "Green Day"}}
]

Suggest songs for "{user_request}" now:"""
