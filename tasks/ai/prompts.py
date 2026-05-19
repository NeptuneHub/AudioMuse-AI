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
    build_artist_hits_prompt(...)     -- artist-hits MCP tool prompt
    build_ai_brainstorm_prompt(...)   -- ai_brainstorm MCP tool prompt
    build_vibe_match_prompt(...)      -- vibe_match MCP tool prompt
    get_dynamic_genres / get_dynamic_moods -- helpers, exposed for tests
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
_FALLBACK_MOODS = "danceable, aggressive, happy, party, relaxed, sad"


def get_dynamic_genres(library_context: Optional[Dict]) -> str:
    """Return genre list from library context, falling back to defaults."""
    if library_context and library_context.get('top_genres'):
        return ', '.join(library_context['top_genres'][:15])
    return _FALLBACK_GENRES


def get_dynamic_moods(library_context: Optional[Dict]) -> str:
    """Return mood list from library context, falling back to defaults."""
    if library_context and library_context.get('top_moods'):
        return ', '.join(library_context['top_moods'][:10])
    return _FALLBACK_MOODS


VOICE_VOCAB = ["female vocalists", "female vocalist", "male vocalists"]

INTENT_CLASSES = ["seed", "text", "knowledge", "metadata"]


def build_mcp_system_prompt(tools: List[Dict], library_context: Optional[Dict] = None) -> str:
    """Build the canonical MCP system prompt used by ALL providers."""
    tool_names = {t['name'] for t in tools}
    has_seed = 'seed_search' in tool_names
    has_text = 'text_match' in tool_names
    has_knowledge = 'knowledge_lookup' in tool_names

    lib_section = ""
    if library_context and library_context.get('total_songs', 0) > 0:
        ctx = library_context
        extras = []
        if ctx.get('year_min') and ctx.get('year_max'):
            extras.append(f"years {ctx['year_min']}-{ctx['year_max']}")
        if ctx.get('has_ratings'):
            extras.append(f"{ctx['rated_songs_pct']}% rated 0-5")
        suffix = (" - " + ", ".join(extras)) if extras else ""
        lib_section = f"\nLibrary: {ctx['total_songs']} songs / {ctx['unique_artists']} artists{suffix}\n"

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

    prompt = f"""You are a music playlist router. You MUST return a JSON object containing one or more tool calls. Front-load EVERY intent in a SINGLE response -- you will NOT be called again.
{lib_section}
TOOLS:
{tools_block}

THE THREE TAG COLUMNS IN search_database ARE SEPARATE -- DO NOT MIX:
- genres : music styles. Closed list -> {genres_line}
- voices : vocal type. Closed list  -> {voices_line}
- moods  : real moods. ONLY these 6 -> {moods_line}

SCALE: major | minor
YEAR: single year -> set year_min AND year_max equal. Decade (80s) -> 1980/1989.
ENERGY: 0.0 (calm) - 1.0 (intense). TEMPO: 40-200 BPM.

RULES:
1. ONLY use filters the user EXPLICITLY mentioned. Do NOT invent extra filters.
2. "female voice"/"woman singer"/"female lead" -> voices=["female vocalists","female vocalist"]
3. "male voice"/"man singer" -> voices=["male vocalists"]
4. NEVER put a voice or a genre into 'moods'. moods is ONLY danceable/aggressive/happy/party/relaxed/sad.
5. Multi-intent goes in ONE seed_search call. "songs like A and B" => seeds=[A,B], blend_mode='union'.
   "A meets B" => seeds=[A,B], blend_mode='alchemy'. "A but not Y" => seeds=[A], subtract=[Y], blend_mode='subtract'.
6. Mix primary + search_database when the user adds metadata constraints ('similar to X with female voice').
7. SONG vs ARTIST inside seed_search seeds:
   - Specific TRACK ("By The Way by RHCP", "Iron Maiden Run to the Hills") -> {{type:'song', title, artist}}.
   - Bare artist ("Madonna", "more like Iron Maiden") -> {{type:'artist', name}}.
   - User may not use dashes -- "Iron Maiden Run to the Hills" still means title='Run to the Hills', artist='Iron Maiden'.
8. Year-only queries ('2024 songs') go to search_database (year_min=2024, year_max=2024), NOT text_match."""

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
        '"songs that sound like Pink Floyd with female voice, sad jazz from the 90s"\n'
        '{{"tool_calls": ['
        '{{"name": "seed_search", "arguments": {{"seeds": [{{"type": "artist", "name": "Pink Floyd"}}], "get_songs": 1000}}}}, '
        '{{"name": "search_database", "arguments": {{"voices": ["female vocalists", "female vocalist"], '
        '"genres": ["jazz"], "moods": ["sad"], "year_min": 1990, "year_max": 1999}}}}'
        ']}}'
    )
    examples_text = "\n\n".join(examples)

    return f"""{system_prompt}

=== OUTPUT FORMAT (CRITICAL) ===
Return ONLY a valid JSON object with this EXACT format:
{{
  "tool_calls": [
    {{"name": "tool_name", "arguments": {{"param": "value"}}}}
  ]
}}

=== EXAMPLES ===
{examples_text}

=== COMMON MISTAKES (DO NOT DO THESE) ===
WRONG: text_match(query="2024 songs")           -> year is metadata, use search_database
WRONG: seed_search with blend_mode='alchemy' and only 1 seed
WRONG: search_database(moods=["female voice"])  -> 'moods' is danceable/aggressive/happy/party/relaxed/sad ONLY
WRONG: knowledge_lookup for songs the library can answer via seed_search or search_database

Now analyze this request and return ONLY the JSON:
Request: "{user_message}"
"""


def build_intent_classifier_prompt(user_message: str) -> str:
    """Tiny Stage-1 prompt: classify the user request into ONE of 4 intent classes.

    Returned JSON shape: {"intent": <one of INTENT_CLASSES>, "needs_filter": bool}.
    """
    return f"""You classify a music playlist request into ONE intent class and decide if it also needs a metadata filter. Return JSON ONLY.

INTENT CLASSES:
- "seed":     user names specific song(s) or artist(s) to find similar music to / blend / subtract.
              Examples: "similar to By The Way by RHCP", "songs like Madonna", "Iron Maiden meets Metallica", "A but not Y".
- "text":     user describes the SOUND ('calm piano') or LYRICAL THEME ('songs about heartbreak'). No specific song/artist seed.
- "knowledge": user asks about POPULARITY / CULTURAL / HISTORICAL facts the library can't answer from metadata alone.
              Trigger words: "top", "best", "popular", "famous", "classic", "radio", "radio hits", "trending",
              "viral", "iconic", "#1", "chart", "charts", "Billboard", "Grammy", "Oscar", "soundtrack of",
              "songs sampled by", "covers of", "anthems".
              Examples: "top pop radio songs of 2025", "Grammy winners 2020", "#1 hits of 1985",
              "biggest rock anthems of the 90s", "songs sampled by Daft Punk".
- "metadata": user filters by year / genre / mood / vocal / tempo / energy / scale / rating / album / single-artist
              WITHOUT any popularity or cultural-ranking superlative.
              Examples: "sad jazz from the 90s", "energetic rock", "2024 songs", "songs in minor key",
              "rock songs from 2020".

CRITICAL DISCRIMINATOR — popularity-superlative beats raw filter:
- "rock songs from 2020"           -> metadata    (descriptive filter, no ranking)
- "top rock songs from 2020"       -> knowledge   ("top" = popularity ranking, library has none)
- "best 90s pop"                   -> knowledge   ("best" = cultural ranking)
- "pop songs from 2025"            -> metadata    (no superlative)
- "top pop radio songs of 2025"    -> knowledge   ("top" + "radio" = chart/popularity)
- "viral TikTok songs of 2024"     -> knowledge   ("viral" = cultural)

needs_filter is TRUE when the user adds metadata constraints ON TOP of a seed / text / knowledge query
(e.g. "similar to Pink Floyd WITH female voice" or "top rock songs from 2020"). FALSE for a pure
seed/text/knowledge query or when the intent itself is "metadata" (the filter IS the intent).

OUTPUT FORMAT (return ONLY this JSON, nothing else):
{{"intent": "seed" | "text" | "knowledge" | "metadata", "needs_filter": true | false}}

EXAMPLES:
"Similar to By The Way by RHCP and Iron Maiden Run to the Hills" -> {{"intent": "seed", "needs_filter": false}}
"songs like Pink Floyd with female voice" -> {{"intent": "seed", "needs_filter": true}}
"calm piano songs" -> {{"intent": "text", "needs_filter": false}}
"songs about heartbreak in the rain" -> {{"intent": "text", "needs_filter": false}}
"Top pop radio songs of 2025" -> {{"intent": "knowledge", "needs_filter": true}}
"Grammy-winning rock songs from 2020" -> {{"intent": "knowledge", "needs_filter": true}}
"best 90s pop" -> {{"intent": "knowledge", "needs_filter": true}}
"sad jazz from the 90s in minor key" -> {{"intent": "metadata", "needs_filter": false}}
"2024 songs" -> {{"intent": "metadata", "needs_filter": false}}
"rock songs from 2020" -> {{"intent": "metadata", "needs_filter": false}}

Request: "{user_message}"
JSON:"""


def build_intent_classifier_schema() -> Dict:
    """JSON Schema for the Stage-1 classifier output."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string", "enum": list(INTENT_CLASSES)},
            "needs_filter": {"type": "boolean"},
        },
        "required": ["intent", "needs_filter"],
    }


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


# --- Free-text MCP prompts (artist hits, brainstorm, vibe match) --------------

def build_artist_hits_prompt(artist: str) -> str:
    return f"""You are a music expert. List the most famous and popular songs by the artist "{artist}".

CRITICAL REQUIREMENTS:
1. Return ONLY a JSON array of song titles
2. Include 15-25 of their most famous songs
3. Use exact song titles as they appear on albums
4. Format: ["Song Title 1", "Song Title 2", ...]
5. NO explanations, NO numbering, ONLY the JSON array

Example format:
["Song A", "Song B", "Song C"]

List the famous songs by {artist} now:"""


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


def build_vibe_match_prompt(vibe_description: str) -> str:
    return f"""You are a music database expert. The user wants songs matching this vibe: "{vibe_description}"

Analyze this vibe and return a JSON object with search criteria for a music database.

Database schema:
- mood_vector: Contains genres like 'rock', 'pop', 'jazz', etc.
- other_features: Contains moods like 'danceable', 'party', 'relaxed', etc.
- energy: 0.01-0.15 (higher = more energetic)
- tempo: 40-200 BPM

Return ONLY this JSON structure:
{{
    "genres": ["genre1", "genre2"],
    "moods": ["mood1", "mood2"],
    "energy_min": 0.05,
    "energy_max": 0.12,
    "tempo_min": 100,
    "tempo_max": 140
}}

Return the JSON now:"""
