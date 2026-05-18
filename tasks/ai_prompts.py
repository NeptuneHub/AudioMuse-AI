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
    build_artist_hits_prompt(...)     -- artist-hits MCP tool prompt
    build_ai_brainstorm_prompt(...)   -- ai_brainstorm MCP tool prompt
    build_vibe_match_prompt(...)      -- vibe_match MCP tool prompt
    get_dynamic_genres / get_dynamic_moods -- helpers, exposed for tests
"""
from typing import Dict, List, Optional


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


def build_mcp_system_prompt(tools: List[Dict], library_context: Optional[Dict] = None) -> str:
    """Build the canonical MCP system prompt used by ALL providers."""
    tool_names = [t['name'] for t in tools]
    has_text_search = 'text_search' in tool_names

    lib_section = ""
    if library_context and library_context.get('total_songs', 0) > 0:
        ctx = library_context
        year_range = ''
        if ctx.get('year_min') and ctx.get('year_max'):
            year_range = f"\n- Year range: {ctx['year_min']}-{ctx['year_max']}"
        rating_info = ''
        if ctx.get('has_ratings'):
            rating_info = f"\n- {ctx['rated_songs_pct']}% of songs have ratings (0-5 scale)"
        scale_info = ''
        if ctx.get('scales'):
            scale_info = f"\n- Scales available: {', '.join(ctx['scales'])}"

        lib_section = f"""
=== USER'S MUSIC LIBRARY ===
- {ctx['total_songs']} songs from {ctx['unique_artists']} artists{year_range}{rating_info}{scale_info}
"""

    decision_tree = []
    decision_tree.append("1. Specific song+artist mentioned? -> song_similarity")
    decision_tree.append("2. 'top/best/greatest/hits/famous/popular' + artist? -> ai_brainstorm (cultural knowledge about iconic tracks)")
    decision_tree.append("3. 'songs from [ALBUM]' or 'songs like [ALBUM]'? -> search_database with album filter, OR song_similarity with tracks from the album")
    decision_tree.append("4. 'songs BY/FROM [ARTIST]' (exact catalog)? -> search_database(artist='Artist Name'). Call ONCE per artist.")
    decision_tree.append("5a. Specific year mentioned (e.g., '2026 songs', 'from 2024')? -> search_database with year_min=YEAR AND year_max=YEAR (BOTH the same year)")
    decision_tree.append("5b. Decade mentioned (80s, 90s, 2000s)? -> ALWAYS include year_min/year_max in search_database (e.g., 80s=1980-1989)")
    if has_text_search:
        decision_tree.append("6. Instruments (piano, guitar, ukulele) or SOUND DESCRIPTIONS (romantic, dreamy, chill vibes)? -> text_search (ONLY for audio/sound descriptions \u2014 NEVER pass years, artist names, or metadata like '2026 songs')")
        decision_tree.append("7. 'songs LIKE/SIMILAR TO [ARTIST]' (discover similar)? -> artist_similarity (returns artist's own + similar artists' songs)")
        decision_tree.append("8. MULTIPLE artists blended ('A meets B', 'A + B', 'like A and B combined') OR negation ('X but not Y', 'X without Y')? -> song_alchemy (REQUIRES 2+ items)")
        decision_tree.append("9. Songs NOT in library, trending, award winners (Grammy, Billboard), cultural knowledge? -> ai_brainstorm")
        decision_tree.append("10. Genre/mood/tempo/energy/year/rating filters? -> search_database")
        decision_tree.append("11. 'minor key', 'major key', 'in minor', 'in major'? -> search_database with scale='minor' or scale='major' (NOT genres \u2014 'minor' is a musical scale, not a genre)")
    else:
        decision_tree.append("6. 'songs LIKE/SIMILAR TO [ARTIST]' (discover similar)? -> artist_similarity (returns artist's own + similar artists' songs)")
        decision_tree.append("7. MULTIPLE artists blended ('A meets B', 'A + B', 'like A and B combined') OR negation ('X but not Y', 'X without Y')? -> song_alchemy (REQUIRES 2+ items)")
        decision_tree.append("8. Songs NOT in library, trending, award winners (Grammy, Billboard), cultural knowledge? -> ai_brainstorm")
        decision_tree.append("9. Genre/mood/tempo/energy/year/rating filters? -> search_database")
        decision_tree.append("10. 'minor key', 'major key', 'in minor', 'in major'? -> search_database with scale='minor' or scale='major' (NOT genres \u2014 'minor' is a musical scale, not a genre)")

    decision_text = '\n'.join(decision_tree)

    prompt = f"""You are an expert music playlist curator. Analyze the user's request and call the appropriate tools to build a playlist of 100 songs.
{lib_section}
=== TOOL SELECTION (most specific -> most general) ===
{decision_text}

=== RULES ===
1. Call one or more tools - each returns songs with item_id, title, and artist
2. song_similarity REQUIRES both title AND artist - never leave empty
3. search_database(artist='...') returns ONLY songs BY that artist. artist_similarity returns songs BY + FROM SIMILAR artists.
   - "songs from Madonna" -> search_database(artist="Madonna")  (exact catalog)
   - "songs like Madonna" -> artist_similarity("Madonna")  (discover similar)
   - "top songs of Madonna" -> ai_brainstorm + search_database(artist="Madonna")  (cultural knowledge + catalog)
4. search_database: COMBINE all filters in ONE call. For decades (80s, 90s), ALWAYS set year_min/year_max (e.g., 80s=1980-1989)
5. search_database genres: Use 1-3 SPECIFIC genres, not broad parent genres. 'rock' matches nearly everything - use sub-genres instead (e.g., 'hard rock', 'punk', 'metal'). WRONG: genres=['rock','metal','classic rock','alternative rock'] (too broad). RIGHT: genres=['metal','hard rock'] (specific).
6. For multiple artists: call search_database(artist='...') once per artist for exact catalog, or use song_alchemy to blend their vibes
7. Prefer tool calls over text explanations
8. For complex requests, call MULTIPLE tools in ONE turn for better coverage:
   - "relaxing piano jazz" -> text_search("relaxing piano") + search_database(genres=["jazz"])
   - "energetic songs by Metallica and AC/DC" -> search_database(artist="Metallica") + search_database(artist="AC/DC")
   - "songs from Blink-182 and Green Day" -> search_database(artist="Blink-182") + search_database(artist="Green Day")
9. When a query has BOTH a genre AND a mood from the MOODS list, prefer search_database over text_search:
   - "sad jazz" -> search_database(genres=["jazz"], moods=["sad"])  NOT text_search
   - But "dreamy atmospheric" -> text_search (no specific genre, sound description)
10. For album requests: use search_database(album="Album Name") to get songs FROM an album,
   or song_similarity with a known track from the album to find SIMILAR songs
11. RATING IS A HARD FILTER: If the user asks for rated/starred songs (e.g., "5 star", "highly rated", "my favorites"),
   you MUST include min_rating in EVERY search_database call. Do NOT use other tools (song_similarity, text_search,
   artist_similarity, ai_brainstorm) for rated-song requests since they cannot filter by rating.
   If fewer songs exist than the target, return what's available \u2014 do NOT pad with unrated songs.
12. COMBINE ALL USER FILTERS: When the user specifies multiple criteria (e.g., "rock 5 star songs"), include ALL of them
   in the SAME search_database call (e.g., genres=["rock"], min_rating=5). Never drop a filter to get more results.
   If the combination returns few songs, that's OK \u2014 return what matches. Quality over quantity.
13. STRICT FILTER FIDELITY: ONLY use parameters the user explicitly mentioned. Do NOT invent or add filters on your own.
   - "songs from 2020-2025" \u2192 ONLY year_min=2020, year_max=2025. Do NOT add genres or min_rating.
   - "2026 songs" or "songs from 2026" \u2192 year_min=2026, year_max=2026. Do NOT set year_min=1.
   - "songs after 2010" \u2192 ONLY year_min=2010. Do NOT set year_max.
   - "rock songs" \u2192 genres=["rock"]. Do NOT add min_rating or year filters.
   - "my 5 star jazz" \u2192 genres=["jazz"], min_rating=5. Keep BOTH.
   If the user didn't mention ratings, do NOT use min_rating. If the user didn't mention genres, do NOT add genres.
   If the user mentioned ONE year, do NOT invent the other year boundary.
14. ACCEPT SMALL PLAYLISTS: If search_database with a year/artist/rating filter returns few results, that means the library
   has limited content matching that criteria. Do NOT pad the playlist by dropping filters or using text_search with metadata
   queries (e.g., "2026 songs"). text_search is for AUDIO DESCRIPTIONS ONLY (instruments, moods, textures). STOP and return
   what you have rather than diluting with irrelevant songs.

=== VALID search_database VALUES ===
GENRES: {get_dynamic_genres(library_context)}
MOODS: {get_dynamic_moods(library_context)}
TEMPO: 40-200 BPM
ENERGY: 0.0 (calm) to 1.0 (intense) - use 0.0-0.35 for low, 0.35-0.65 for medium, 0.65-1.0 for high
SCALE: major, minor (IMPORTANT: "minor key" or "major key" \u2192 use scale="minor" or scale="major", NOT genres)
YEAR: year_min and/or year_max. Use BOTH only for ranges (e.g., 1990-1999 for 90s). Use ONLY year_min for "from/since/after YEAR". Use ONLY year_max for "before/until YEAR". For a single year ("2026 songs"), set year_min=2026 AND year_max=2026. Do NOT invent the other boundary.
RATING: min_rating 1-5 (user's personal ratings)
ARTIST: artist name (e.g. 'Madonna', 'Blink-182') - returns ONLY songs by this artist
ALBUM: album name (e.g. 'Abbey Road', 'Thriller') - filters songs from a specific album"""

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
    has_text_search = 'text_search' in [t['name'] for t in tools]

    tools_lines = []
    for tool in tools:
        props = tool['inputSchema'].get('properties', {})
        params_desc = ", ".join([f"{k} ({v.get('type')})" for k, v in props.items()])
        tools_lines.append(f"- {tool['name']}: {params_desc}")
    tools_text = "\n".join(tools_lines)

    system_prompt = build_mcp_system_prompt(tools, library_context)

    examples = []
    examples.append('"Similar to By the Way by Red Hot Chili Peppers"\n{{"tool_calls": [{{"name": "song_similarity", "arguments": {{"song_title": "By the Way", "song_artist": "Red Hot Chili Peppers", "get_songs": 200}}}}]}}')
    if has_text_search:
        examples.append('"calm piano song"\n{{"tool_calls": [{{"name": "text_search", "arguments": {{"description": "calm piano", "get_songs": 200}}}}]}}')
    examples.append('"songs from blink-182 and Green Day"\n{{"tool_calls": [{{"name": "search_database", "arguments": {{"artist": "blink-182", "get_songs": 200}}}}, {{"name": "search_database", "arguments": {{"artist": "Green Day", "get_songs": 200}}}}]}}')
    examples.append('"songs like blink-182"\n{{"tool_calls": [{{"name": "artist_similarity", "arguments": {{"artist": "blink-182", "get_songs": 200}}}}]}}')
    examples.append('"top songs of Madonna"\n{{"tool_calls": [{{"name": "ai_brainstorm", "arguments": {{"user_request": "top songs of Madonna", "get_songs": 200}}}}, {{"name": "search_database", "arguments": {{"artist": "Madonna", "get_songs": 200}}}}]}}')
    examples.append('"energetic rock"\n{{"tool_calls": [{{"name": "search_database", "arguments": {{"genres": ["rock"], "energy_min": 0.65, "get_songs": 200}}}}]}}')
    examples.append('"2026 songs"\n{{"tool_calls": [{{"name": "search_database", "arguments": {{"year_min": 2026, "year_max": 2026, "get_songs": 200}}}}]}}')
    examples.append('"90s pop"\n{{"tool_calls": [{{"name": "search_database", "arguments": {{"genres": ["pop"], "year_min": 1990, "year_max": 1999, "get_songs": 200}}}}]}}')
    examples.append('"songs in minor key"\n{{"tool_calls": [{{"name": "search_database", "arguments": {{"scale": "minor", "get_songs": 200}}}}]}}')
    examples.append('"sounds like Iron Maiden and Metallica combined"\n{{"tool_calls": [{{"name": "song_alchemy", "arguments": {{"add_items": [{{"type": "artist", "id": "Iron Maiden"}}, {{"type": "artist", "id": "Metallica"}}], "get_songs": 200}}}}]}}')
    examples.append('"mix of Daft Punk and Gorillaz"\n{{"tool_calls": [{{"name": "song_alchemy", "arguments": {{"add_items": [{{"type": "artist", "id": "Daft Punk"}}, {{"type": "artist", "id": "Gorillaz"}}], "get_songs": 200}}}}]}}')
    examples_text = "\n\n".join(examples)

    return f"""{system_prompt}

=== TOOL PARAMETERS ===
{tools_text}

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
WRONG: "2026 songs" -> adding genres, min_rating, or moods (user only asked for year!)
WRONG: "electronic music" -> adding min_rating or year filters (user only asked for genre!)
WRONG: "songs from 2020-2025" -> adding genres (user only asked for years!)
CORRECT: Only include filters the user EXPLICITLY mentioned. Nothing extra.
WRONG: "songs in minor key" -> using genres or text_search (user asked for musical scale, not genre!)
CORRECT: "songs in minor key" -> search_database(scale="minor")

IMPORTANT: ONLY include parameters the user explicitly asked for. Do NOT invent extra filters (genres, ratings, moods, energy) the user never mentioned.
For a specific year like "2026 songs", set BOTH year_min and year_max to 2026 (NOT year_min=1).

Now analyze this request and return ONLY the JSON:
Request: "{user_message}"
"""


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
