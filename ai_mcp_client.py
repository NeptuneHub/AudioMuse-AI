"""
MCP Client for AudioMuse-AI
Handles MCP tool calling for different AI providers (Gemini, OpenAI, Mistral, Ollama)
"""
import json
import logging
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger(__name__)


_FALLBACK_GENRES = "rock, pop, metal, jazz, electronic, dance, alternative, indie, punk, blues, hard rock, heavy metal, hip-hop, funk, country, soul"
_FALLBACK_MOODS = "danceable, aggressive, happy, party, relaxed, sad"


def _get_dynamic_genres(library_context: Optional[Dict]) -> str:
    """Return genre list from library context, falling back to defaults."""
    if library_context and library_context.get('top_genres'):
        return ', '.join(library_context['top_genres'][:15])
    return _FALLBACK_GENRES


def _get_dynamic_moods(library_context: Optional[Dict]) -> str:
    """Return mood list from library context, falling back to defaults."""
    if library_context and library_context.get('top_moods'):
        return ', '.join(library_context['top_moods'][:10])
    return _FALLBACK_MOODS


def _build_system_prompt(tools: List[Dict], library_context: Optional[Dict] = None) -> str:
    """Build a single canonical system prompt used by ALL AI providers.

    Args:
        tools: MCP tool definitions (used to list correct tool names)
        library_context: Optional dict from get_library_context() with library stats
    """
    tool_names = [t['name'] for t in tools]
    has_text_search = 'text_search' in tool_names

    # Build library context section
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

    # Build tool decision tree
    decision_tree = []
    decision_tree.append("1. Specific song+artist mentioned? -> song_similarity")
    decision_tree.append("2. 'top/best/greatest/hits/famous/popular' + artist? -> ai_brainstorm (cultural knowledge about iconic tracks)")
    decision_tree.append("3. 'songs from [ALBUM]' or 'songs like [ALBUM]'? -> search_database with album filter, OR song_similarity with tracks from the album")
    decision_tree.append("4. 'songs BY/FROM [ARTIST]' (exact catalog)? -> search_database(artist='Artist Name'). Call ONCE per artist.")
    decision_tree.append("5a. Specific year mentioned (e.g., '2026 songs', 'from 2024')? -> search_database with year_min=YEAR AND year_max=YEAR (BOTH the same year)")
    decision_tree.append("5b. Decade mentioned (80s, 90s, 2000s)? -> ALWAYS include year_min/year_max in search_database (e.g., 80s=1980-1989)")
    if has_text_search:
        decision_tree.append("6. Instruments (piano, guitar, ukulele) or SOUND DESCRIPTIONS (romantic, dreamy, chill vibes)? -> text_search (ONLY for audio/sound descriptions — NEVER pass years, artist names, or metadata like '2026 songs')")
        decision_tree.append("7. 'songs LIKE/SIMILAR TO [ARTIST]' (discover similar)? -> artist_similarity (returns artist's own + similar artists' songs)")
        decision_tree.append("8. MULTIPLE artists blended ('A meets B', 'A + B', 'like A and B combined') OR negation ('X but not Y', 'X without Y')? -> song_alchemy (REQUIRES 2+ items)")
        decision_tree.append("9. Songs NOT in library, trending, award winners (Grammy, Billboard), cultural knowledge? -> ai_brainstorm")
        decision_tree.append("10. Genre/mood/tempo/energy/year/rating filters? -> search_database")
        decision_tree.append("11. 'minor key', 'major key', 'in minor', 'in major'? -> search_database with scale='minor' or scale='major' (NOT genres — 'minor' is a musical scale, not a genre)")
    else:
        decision_tree.append("6. 'songs LIKE/SIMILAR TO [ARTIST]' (discover similar)? -> artist_similarity (returns artist's own + similar artists' songs)")
        decision_tree.append("7. MULTIPLE artists blended ('A meets B', 'A + B', 'like A and B combined') OR negation ('X but not Y', 'X without Y')? -> song_alchemy (REQUIRES 2+ items)")
        decision_tree.append("8. Songs NOT in library, trending, award winners (Grammy, Billboard), cultural knowledge? -> ai_brainstorm")
        decision_tree.append("9. Genre/mood/tempo/energy/year/rating filters? -> search_database")
        decision_tree.append("10. 'minor key', 'major key', 'in minor', 'in major'? -> search_database with scale='minor' or scale='major' (NOT genres — 'minor' is a musical scale, not a genre)")

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
   If fewer songs exist than the target, return what's available — do NOT pad with unrated songs.
12. COMBINE ALL USER FILTERS: When the user specifies multiple criteria (e.g., "rock 5 star songs"), include ALL of them
   in the SAME search_database call (e.g., genres=["rock"], min_rating=5). Never drop a filter to get more results.
   If the combination returns few songs, that's OK — return what matches. Quality over quantity.
13. STRICT FILTER FIDELITY: ONLY use parameters the user explicitly mentioned. Do NOT invent or add filters on your own.
   - "songs from 2020-2025" → ONLY year_min=2020, year_max=2025. Do NOT add genres or min_rating.
   - "2026 songs" or "songs from 2026" → year_min=2026, year_max=2026. Do NOT set year_min=1.
   - "songs after 2010" → ONLY year_min=2010. Do NOT set year_max.
   - "rock songs" → genres=["rock"]. Do NOT add min_rating or year filters.
   - "my 5 star jazz" → genres=["jazz"], min_rating=5. Keep BOTH.
   If the user didn't mention ratings, do NOT use min_rating. If the user didn't mention genres, do NOT add genres.
   If the user mentioned ONE year, do NOT invent the other year boundary.
14. ACCEPT SMALL PLAYLISTS: If search_database with a year/artist/rating filter returns few results, that means the library
   has limited content matching that criteria. Do NOT pad the playlist by dropping filters or using text_search with metadata
   queries (e.g., "2026 songs"). text_search is for AUDIO DESCRIPTIONS ONLY (instruments, moods, textures). STOP and return
   what you have rather than diluting with irrelevant songs.

=== VALID search_database VALUES ===
GENRES: {_get_dynamic_genres(library_context)}
MOODS: {_get_dynamic_moods(library_context)}
TEMPO: 40-200 BPM
ENERGY: 0.0 (calm) to 1.0 (intense) - use 0.0-0.35 for low, 0.35-0.65 for medium, 0.65-1.0 for high
SCALE: major, minor (IMPORTANT: "minor key" or "major key" → use scale="minor" or scale="major", NOT genres)
YEAR: year_min and/or year_max. Use BOTH only for ranges (e.g., 1990-1999 for 90s). Use ONLY year_min for "from/since/after YEAR". Use ONLY year_max for "before/until YEAR". For a single year ("2026 songs"), set year_min=2026 AND year_max=2026. Do NOT invent the other boundary.
RATING: min_rating 1-5 (user's personal ratings)
ARTIST: artist name (e.g. 'Madonna', 'Blink-182') - returns ONLY songs by this artist
ALBUM: album name (e.g. 'Abbey Road', 'Thriller') - filters songs from a specific album"""

    return prompt


def call_ai_with_mcp_tools(
    provider: str,
    user_message: str,
    tools: List[Dict],
    ai_config: Dict,
    log_messages: List[str],
    library_context: Optional[Dict] = None
) -> Dict:
    """
    Call AI provider with MCP tool definitions and handle tool calling flow.

    Args:
        provider: AI provider ('GEMINI', 'OPENAI', 'MISTRAL', 'OLLAMA')
        user_message: The user's natural language request
        tools: List of MCP tool definitions
        ai_config: Configuration dict with API keys, URLs, model names
        log_messages: List to append log messages to
        library_context: Optional library stats dict from get_library_context()

    Returns:
        Dict with 'tool_calls' (list of tool calls) or 'error' (error message)
    """
    if provider == "GEMINI":
        return _call_gemini_with_tools(user_message, tools, ai_config, log_messages, library_context)
    elif provider == "OPENAI":
        return _call_openai_with_tools(user_message, tools, ai_config, log_messages, library_context)
    elif provider == "MISTRAL":
        return _call_mistral_with_tools(user_message, tools, ai_config, log_messages, library_context)
    elif provider == "OLLAMA":
        return _call_ollama_with_tools(user_message, tools, ai_config, log_messages, library_context)
    else:
        return {"error": f"Unsupported AI provider: {provider}"}


def _call_gemini_with_tools(user_message: str, tools: List[Dict], ai_config: Dict, log_messages: List[str], library_context: Optional[Dict] = None) -> Dict:
    """Call Gemini with function calling."""
    try:
        import google.genai as genai

        api_key = ai_config.get('gemini_key')
        model_name = ai_config.get('gemini_model', 'gemini-2.5-pro')

        if not api_key or api_key == "YOUR-GEMINI-API-KEY-HERE":
            return {"error": "Valid Gemini API key required"}

        # Use new google-genai Client API
        client = genai.Client(api_key=api_key)

        # Convert MCP tools to Gemini function declarations
        # Gemini uses a different schema format - need to convert types
        def convert_schema_for_gemini(schema):
            """Convert JSON Schema to Gemini-compatible format."""
            if not isinstance(schema, dict):
                return schema

            result = {}

            # Convert type field
            if 'type' in schema:
                schema_type = schema['type']
                # Gemini uses uppercase type names
                type_map = {
                    'string': 'STRING',
                    'number': 'NUMBER',
                    'integer': 'INTEGER',
                    'boolean': 'BOOLEAN',
                    'array': 'ARRAY',
                    'object': 'OBJECT'
                }
                result['type'] = type_map.get(schema_type, schema_type.upper())

            # Copy description
            if 'description' in schema:
                result['description'] = schema['description']

            # Handle properties recursively
            if 'properties' in schema:
                result['properties'] = {
                    k: convert_schema_for_gemini(v)
                    for k, v in schema['properties'].items()
                }

            # Handle array items
            if 'items' in schema:
                result['items'] = convert_schema_for_gemini(schema['items'])

            # Copy required and enum (Gemini doesn't support 'default')
            for field in ['required', 'enum']:
                if field in schema:
                    result[field] = schema[field]

            return result

        function_declarations = []
        for tool in tools:
            func_decl = {
                "name": tool['name'],
                "description": tool['description'],
                "parameters": convert_schema_for_gemini(tool['inputSchema'])
            }
            function_declarations.append(func_decl)

        # Unified system prompt
        system_instruction = _build_system_prompt(tools, library_context)

        # Prepare tools for new API
        tools_list = [genai.types.Tool(function_declarations=function_declarations)]

        # Generate response with function calling using new API
        # Note: Using 'ANY' mode to force tool calling instead of text response
        # system_instruction gives the prompt proper role separation (not mixed into user content)
        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools_list,
                tool_config=genai.types.ToolConfig(
                    function_calling_config=genai.types.FunctionCallingConfig(mode='ANY')
                )
            )
        )
        
        log_messages.append(f"Gemini response type: {type(response)}")
        
        # Helper to recursively convert protobuf/dict objects to clean dict
        def convert_to_dict(obj):
            """Recursively convert protobuf objects (like RepeatedComposite) to native Python types."""
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                if hasattr(obj, 'items'):  # dict-like
                    return {k: convert_to_dict(v) for k, v in obj.items()}
                else:  # list-like
                    return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            return obj
        
        # Extract function calls from new API response structure
        # New API returns candidates with parts containing function_call
        tool_calls = []
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        # Extract arguments - could be in 'args' dict or 'arguments' field
                        args_dict = {}
                        if hasattr(fc, 'args'):
                            args_dict = dict(fc.args) if fc.args else {}
                        elif hasattr(fc, 'arguments'):
                            args_dict = fc.arguments if isinstance(fc.arguments, dict) else {}
                        
                        tool_calls.append({
                            "name": fc.name,
                            "arguments": convert_to_dict(args_dict)
                        })
        
        if not tool_calls:
            # If no tool calls, Gemini might have returned text
            text_response = response.text if hasattr(response, 'text') else str(response)
            log_messages.append(f"Gemini did not call tools. Response: {text_response[:200]}")
            return {"error": "AI did not call any tools", "ai_response": text_response}
        
        log_messages.append(f"Gemini called {len(tool_calls)} tools")
        return {"tool_calls": tool_calls}
    
    except Exception as e:
        logger.exception("Error calling Gemini with tools")
        return {"error": f"Gemini error: {str(e)}"}


def _call_openai_with_tools(user_message: str, tools: List[Dict], ai_config: Dict, log_messages: List[str], library_context: Optional[Dict] = None) -> Dict:
    """Call OpenAI-compatible API with function calling."""
    try:
        import httpx

        api_url = ai_config.get('openai_url', 'https://api.openai.com/v1/chat/completions')
        api_key = ai_config.get('openai_key', 'no-key-needed')
        model_name = ai_config.get('openai_model', 'gpt-4')

        # Convert MCP tools to OpenAI function format
        functions = []
        for tool in tools:
            functions.append({
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool['inputSchema']
                }
            })

        # Build request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Unified system prompt
        system_prompt = _build_system_prompt(tools, library_context)

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "tools": functions,
            "tool_choice": "required"
        }
        
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        log_messages.append(f"Using timeout: {timeout} seconds for OpenAI/Mistral request")
        with httpx.Client(timeout=timeout) as client:
            response = client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
        
        # Extract tool calls
        tool_calls = []
        if 'choices' in result and result['choices']:
            message = result['choices'][0].get('message', {})
            if 'tool_calls' in message:
                for tc in message['tool_calls']:
                    if tc['type'] == 'function':
                        tool_calls.append({
                            "name": tc['function']['name'],
                            "arguments": json.loads(tc['function']['arguments'])
                        })
        
        if not tool_calls:
            # Check if there's a text response
            text_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            log_messages.append(f"OpenAI did not call tools. Response: {text_response[:200]}")
            return {"error": "AI did not call any tools", "ai_response": text_response}
        
        log_messages.append(f"OpenAI called {len(tool_calls)} tools")
        return {"tool_calls": tool_calls}
    
    except httpx.ReadTimeout:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"OpenAI/Mistral request timed out after {timeout} seconds")
        log_messages.append(f"⏱️ Request timed out after {timeout} seconds. Consider increasing AI_REQUEST_TIMEOUT_SECONDS environment variable.")
        return {"error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."}
    except httpx.TimeoutException as e:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"OpenAI/Mistral request timed out: {str(e)}")
        log_messages.append(f"⏱️ Request timed out after {timeout} seconds: {str(e)}")
        return {"error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."}
    except Exception as e:
        logger.exception("Error calling OpenAI with tools")
        return {"error": f"OpenAI error: {str(e)}"}


def _call_mistral_with_tools(user_message: str, tools: List[Dict], ai_config: Dict, log_messages: List[str], library_context: Optional[Dict] = None) -> Dict:
    """Call Mistral with function calling."""
    try:
        from mistralai import Mistral

        api_key = ai_config.get('mistral_key')
        model_name = ai_config.get('mistral_model', 'mistral-large-latest')

        if not api_key or api_key == "YOUR-GEMINI-API-KEY-HERE":
            return {"error": "Valid Mistral API key required"}

        client = Mistral(api_key=api_key)

        # Convert MCP tools to Mistral function format
        mistral_tools = []
        for tool in tools:
            mistral_tools.append({
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool['inputSchema']
                }
            })

        # Unified system prompt
        system_prompt = _build_system_prompt(tools, library_context)

        # Call Mistral
        response = client.chat.complete(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            tools=mistral_tools,
            tool_choice="any"
        )
        
        # Extract tool calls
        tool_calls = []
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    })
        
        if not tool_calls:
            text_response = response.choices[0].message.content if response.choices else ""
            log_messages.append(f"Mistral did not call tools. Response: {text_response[:200]}")
            return {"error": "AI did not call any tools", "ai_response": text_response}
        
        log_messages.append(f"Mistral called {len(tool_calls)} tools")
        return {"tool_calls": tool_calls}
    
    except Exception as e:
        logger.exception("Error calling Mistral with tools")
        return {"error": f"Mistral error: {str(e)}"}


def _call_ollama_with_tools(user_message: str, tools: List[Dict], ai_config: Dict, log_messages: List[str], library_context: Optional[Dict] = None) -> Dict:
    """
    Call Ollama with tool definitions.
    Note: Ollama's tool calling support varies by model. This uses a prompt-based approach.
    """
    try:
        import httpx

        ollama_url = ai_config.get('ollama_url', 'http://localhost:11434/api/generate')
        model_name = ai_config.get('ollama_model', 'llama3.1:8b')

        # Build tool parameter descriptions for Ollama (it needs explicit param listings)
        tools_list = []
        has_text_search = 'text_search' in [t['name'] for t in tools]
        for tool in tools:
            props = tool['inputSchema'].get('properties', {})
            params_desc = ", ".join([f"{k} ({v.get('type')})" for k, v in props.items()])
            tools_list.append(f"- {tool['name']}: {params_desc}")
        tools_text = "\n".join(tools_list)

        # Use the unified system prompt as base, then add Ollama-specific JSON format instructions
        system_prompt = _build_system_prompt(tools, library_context)

        # Build a few examples for Ollama's JSON output format
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

        prompt = f"""{system_prompt}

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
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        # Always disable thinking for reasoning models (Qwen 3.5, DeepSeek-R1, etc.)
        # Thinking output breaks JSON parsing when format: "json" is set
        payload["think"] = False


        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        log_messages.append(f"Using timeout: {timeout} seconds for Ollama request")
        with httpx.Client(timeout=timeout) as client:
            response = client.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()

        # Parse response
        if 'response' not in result:
            return {"error": "Invalid Ollama response"}

        response_text = result['response']

        # Thinking models (e.g. Qwen 3.5) return empty response with format=json.
        # Retry without format constraint — their response field will have clean JSON
        # and the thinking/reasoning stays in the separate 'thinking' field.
        if not response_text and result.get('thinking'):
            log_messages.append(f"ℹ️ Thinking model detected — retrying without format=json")
            payload.pop("format", None)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(ollama_url, json=payload)
                response.raise_for_status()
                result = response.json()
            response_text = result.get('response', '')
            # Strip <think> tags from thinking model output
            if response_text and '</think>' in response_text:
                response_text = response_text.split('</think>', 1)[-1].strip()

        # Try to extract JSON
        try:
            cleaned = response_text.strip()

            # Safety net: strip <think>...</think> blocks from reasoning models
            # Even with think:false, some models may still include thinking output
            import re
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
            # Also strip if there's an unclosed <think> tag (partial thinking output)
            if '<think>' in cleaned:
                cleaned = cleaned.split('</think>')[-1].strip() if '</think>' in cleaned else re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
            
            log_messages.append(f"Ollama raw response (first 300 chars): {cleaned[:300]}")
            
            # Remove markdown code blocks if present
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]
            cleaned = cleaned.strip()
            
            # Check if this is a schema definition instead of tool calls
            if cleaned.startswith('{') and '"type"' in cleaned and '"array"' in cleaned:
                log_messages.append("⚠️ Ollama returned schema instead of tool calls, using fallback")
                return {"error": "Ollama returned schema definition instead of tool calls"}
            
            # Parse the JSON object (should be {"tool_calls": [...]})
            log_messages.append(f"Attempting to parse: {cleaned[:200]}")
            parsed = json.loads(cleaned)
            
            # Extract tool_calls array from the object
            if isinstance(parsed, dict) and 'tool_calls' in parsed:
                tool_calls = parsed['tool_calls']
                log_messages.append(f"✓ Extracted tool_calls array with {len(tool_calls) if isinstance(tool_calls, list) else 1} items")
            elif isinstance(parsed, list):
                # If it returned an array directly (shouldn't happen with new prompt but handle it)
                tool_calls = parsed
                log_messages.append(f"⚠️ Got array directly (expected object with tool_calls field)")
            elif isinstance(parsed, dict) and 'name' in parsed:
                # Single tool call as object, wrap it
                tool_calls = [parsed]
                log_messages.append(f"⚠️ Got single tool call object (expected object with tool_calls array)")
            elif isinstance(parsed, dict) and 'tool' in parsed and 'arguments' in parsed:
                # Thinking models (e.g. Qwen 3.5) sometimes return {"tool": "name", "arguments": {...}}
                tool_calls = [{"name": parsed["tool"], "arguments": parsed["arguments"]}]
                log_messages.append(f"⚠️ Remapped {{'tool','arguments'}} → {{'name','arguments'}} format")
            else:
                log_messages.append(f"⚠️ Unexpected JSON structure: {type(parsed)}, keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")
                return {"error": "Ollama response missing 'tool_calls' field"}
            
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            
            # Validate tool calls structure and strip empty/default values
            valid_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict) and 'name' in tc:
                    # Ensure arguments is a dict
                    if 'arguments' not in tc:
                        tc['arguments'] = {}
                    # Strip empty/default values that small models hallucinate
                    args = tc['arguments']
                    keys_to_remove = []
                    for k, v in args.items():
                        if v is None or v == '' or v == [] or v == {}:
                            keys_to_remove.append(k)
                        elif k == 'tempo_min' and v == 0:
                            keys_to_remove.append(k)
                        elif k == 'tempo_max' and v == 0:
                            keys_to_remove.append(k)
                        elif k == 'energy_min' and v == 0:
                            keys_to_remove.append(k)
                        elif k == 'min_rating' and v == 0:
                            keys_to_remove.append(k)
                    for k in keys_to_remove:
                        log_messages.append(f"   🧹 Stripped empty/default arg '{k}={args[k]}' from {tc['name']}")
                        del args[k]
                    valid_calls.append(tc)
                else:
                    log_messages.append(f"⚠️ Skipping invalid tool call: {tc}")
            
            if not valid_calls:
                return {"error": "No valid tool calls found in Ollama response"}
            
            log_messages.append(f"✅ Ollama returned {len(valid_calls)} valid tool calls")
            return {"tool_calls": valid_calls}
        
        except json.JSONDecodeError as e:
            log_messages.append(f"❌ JSON decode error: {str(e)}")
            log_messages.append(f"Attempted to parse: {cleaned[:300]}")
            return {"error": f"Failed to parse Ollama JSON: {str(e)}", "raw_response": response_text[:200]}
        
        except Exception as e:
            log_messages.append(f"Failed to parse Ollama response: {str(e)}")
            log_messages.append(f"Response was: {response_text[:200]}")
            return {"error": "Failed to parse Ollama tool calls", "raw_response": response_text}
    
    except httpx.ReadTimeout:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"Ollama request timed out after {timeout} seconds")
        log_messages.append(f"⏱️ Ollama request timed out after {timeout} seconds. Your model or hardware may be too slow.")
        log_messages.append(f"💡 Solution: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value (e.g., 600 for 10 minutes)")
        return {"error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."}
    except httpx.TimeoutException as e:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"Ollama request timed out: {str(e)}")
        log_messages.append(f"⏱️ Ollama request timed out after {timeout} seconds: {str(e)}")
        log_messages.append(f"💡 Solution: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value")
        return {"error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."}
    except Exception as e:
        logger.exception("Error calling Ollama with tools")
        return {"error": f"Ollama error: {str(e)}"}


def execute_mcp_tool(tool_name: str, tool_args: Dict, ai_config: Dict) -> Dict:
    """Execute an MCP tool - 6 CORE TOOLS."""
    from tasks.mcp_server import (_artist_similarity_api_sync, _song_similarity_api_sync, 
                                    _database_genre_query_sync, _ai_brainstorm_sync, _song_alchemy_sync,
                                    _text_search_sync)
    
    try:
        if tool_name == "artist_similarity":
            return _artist_similarity_api_sync(
                tool_args['artist'],
                15,  # count - hardcoded
                tool_args.get('get_songs', 200)
            )
        elif tool_name == "text_search":
            # Guard: reject metadata-only queries that CLAP can't handle meaningfully
            desc = tool_args.get('description', '')
            import re as _re
            # Match queries that are purely year-based (e.g., "2026 songs", "1990 music", "songs from 2024")
            if _re.match(r'^(songs?\s+(from\s+)?)?(\d{4})\s*(songs?|music|tracks?)?$', desc.strip(), _re.IGNORECASE):
                return {"songs": [], "message": f"text_search rejected: '{desc}' is a metadata query (year), not an audio description. Use search_database with year_min/year_max instead."}
            return _text_search_sync(
                desc,
                tool_args.get('tempo_filter'),
                tool_args.get('energy_filter'),
                tool_args.get('get_songs', 200)
            )
        elif tool_name == "song_similarity":
            return _song_similarity_api_sync(
                tool_args['song_title'],
                tool_args['song_artist'],
                tool_args.get('get_songs', 200)
            )
        elif tool_name == "song_alchemy":
            # Handle both formats: ["artist1", "artist2"] or [{"type": "artist", "id": "artist1"}]
            add_items = tool_args.get('add_items', [])
            subtract_items = tool_args.get('subtract_items', [])
            
            # Normalize to proper format if AI sent simple strings
            def normalize_items(items):
                if not items:
                    return []
                normalized = []
                for item in items:
                    if isinstance(item, str):
                        # Simple string -> assume artist
                        normalized.append({"type": "artist", "id": item})
                    elif isinstance(item, dict):
                        # Already proper format
                        normalized.append(item)
                return normalized
            
            add_items = normalize_items(add_items)
            subtract_items = normalize_items(subtract_items)
            
            return _song_alchemy_sync(
                add_items,
                subtract_items,
                tool_args.get('get_songs', 200)
            )
        elif tool_name == "search_database":
            # Convert normalized energy (0-1) to raw energy scale
            # AI sees 0.0-1.0, raw DB range is ENERGY_MIN-ENERGY_MAX (e.g. 0.01-0.15)
            energy_min_raw = None
            energy_max_raw = None
            e_min = tool_args.get('energy_min')
            e_max = tool_args.get('energy_max')
            if e_min is not None:
                e_min = float(e_min)
                energy_min_raw = config.ENERGY_MIN + e_min * (config.ENERGY_MAX - config.ENERGY_MIN)
            if e_max is not None:
                e_max = float(e_max)
                energy_max_raw = config.ENERGY_MIN + e_max * (config.ENERGY_MAX - config.ENERGY_MIN)

            return _database_genre_query_sync(
                tool_args.get('genres'),
                tool_args.get('get_songs', 200),
                tool_args.get('moods'),
                tool_args.get('tempo_min'),
                tool_args.get('tempo_max'),
                energy_min_raw,
                energy_max_raw,
                tool_args.get('key'),
                tool_args.get('scale'),
                tool_args.get('year_min'),
                tool_args.get('year_max'),
                tool_args.get('min_rating'),
                tool_args.get('album'),
                tool_args.get('artist')
            )
        elif tool_name == "ai_brainstorm":
            return _ai_brainstorm_sync(
                tool_args['user_request'],
                ai_config,
                tool_args.get('get_songs', 200)
            )
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    except Exception as e:
        logger.exception(f"Error executing MCP tool {tool_name}")
        return {"error": f"Tool execution error: {str(e)}"}


def get_mcp_tools() -> List[Dict]:
    """Get the list of available MCP tools - 6 CORE TOOLS.

    ⚠️ CRITICAL: ALWAYS choose tools in THIS ORDER (most specific → most general):
    1. SONG_SIMILARITY - for specific song title + artist
    2. TEXT_SEARCH - for instruments, specific moods, descriptive queries (requires CLAP)
    3. ARTIST_SIMILARITY - for songs BY/FROM specific artist(s) (includes artist's own songs)
    4. SONG_ALCHEMY - for 'sounds LIKE' blending multiple artists/songs
    5. AI_BRAINSTORM - for world knowledge (trending, awards, songs NOT in library)
    6. SEARCH_DATABASE - for genre/mood/tempo filters (last resort)
    
    Never skip to a general tool when a specific tool can handle the request!
    
    CLAP Text Search: Check if available before using text_search tool.
    """
    from config import CLAP_ENABLED
    
    tools = [
        {
            "name": "song_similarity",
            "description": "🥇 PRIORITY #1: MOST SPECIFIC - Find songs similar to a specific song (requires exact title+artist). ✅ USE when user mentions a SPECIFIC SONG TITLE.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "song_title": {
                        "type": "string",
                        "description": "Song title"
                    },
                    "song_artist": {
                        "type": "string",
                        "description": "Artist name"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 200
                    }
                },
                "required": ["song_title", "song_artist"]
            }
        }
    ]
    
    # Add text_search only if CLAP is enabled
    if CLAP_ENABLED:
        tools.append({
            "name": "text_search",
            "description": "🥈 PRIORITY #2: HIGH PRIORITY - Natural language search using CLAP. ✅ USE for: INSTRUMENTS (piano, guitar, ukulele), SOUND DESCRIPTIONS (romantic, dreamy, chill vibes), DESCRIPTIVE QUERIES ('energetic workout'). Supports optional tempo/energy filters for hybrid search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description (e.g., 'piano music', 'romantic pop', 'ukulele songs', 'energetic guitar rock')"
                    },
                    "tempo_filter": {
                        "type": "string",
                        "enum": ["slow", "medium", "fast"],
                        "description": "Optional: Filter CLAP results by tempo (hybrid mode)"
                    },
                    "energy_filter": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Optional: Filter CLAP results by energy (hybrid mode)"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 200
                    }
                },
                "required": ["description"]
            }
        })
    
    tools.extend([
        {
            "name": "artist_similarity",
            "description": f"🥉 PRIORITY #{'5' if CLAP_ENABLED else '4'}: Find songs BY an artist AND similar artists. ✅ USE for: 'songs by/from/like Artist X' including the artist's own songs (call once per artist). ❌ DON'T USE for: 'sounds LIKE multiple artists blended' (use song_alchemy).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "artist": {
                        "type": "string",
                        "description": "Artist name"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 200
                    }
                },
                "required": ["artist"]
            }
        },
        {
            "name": "song_alchemy",
            "description": f"🏅 PRIORITY #{'6' if CLAP_ENABLED else '5'}: VECTOR ARITHMETIC - Blend or subtract MULTIPLE artists/songs. REQUIRES 2+ items. Keywords: 'meets', 'combined', 'blend', 'mix of', 'but not', 'without'. ✅ BEST for: 'play like A + B' ('play like Iron Maiden, Metallica, Deep Purple'), 'like X but NOT Y', 'Artist A meets Artist B', 'mix of A and B'. ❌ DON'T USE for: single artist (use artist_similarity), genre/mood (use search_database). Examples: 'play like Iron Maiden + Metallica + Deep Purple' = add all 3; 'Beatles but not ballads' = add Beatles, subtract ballads.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "add_items": {
                        "type": "array",
                        "description": "Items to ADD (blend into result). Each item: {type: 'song' or 'artist', id: 'artist_name' or 'song_title by artist'}",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["song", "artist"],
                                    "description": "Item type: 'song' or 'artist'"
                                },
                                "id": {
                                    "type": "string",
                                    "description": "For artist: 'Artist Name'; For song: 'Song Title by Artist Name'"
                                }
                            },
                            "required": ["type", "id"]
                        }
                    },
                    "subtract_items": {
                        "type": "array",
                        "description": "Items to SUBTRACT (remove from result). Same format as add_items.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["song", "artist"],
                                    "description": "Item type: 'song' or 'artist'"
                                },
                                "id": {
                                    "type": "string",
                                    "description": "For artist: 'Artist Name'; For song: 'Song Title by Artist Name'"
                                }
                            },
                            "required": ["type", "id"]
                        }
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 200
                    }
                },
                "required": ["add_items"]
            }
        },
        {
            "name": "ai_brainstorm",
            "description": f"🏅 PRIORITY #{'7' if CLAP_ENABLED else '6'}: AI world knowledge - Use ONLY when other tools CAN'T work. ✅ USE for: named events (Grammy, Billboard, festivals), cultural knowledge (trending, viral, classic hits), historical significance (best of decade, iconic albums), songs NOT in library. ❌ DON'T USE for: artist's own songs (use artist_similarity), 'sounds like' (use song_alchemy), genre/mood (use search_database), instruments/moods (use text_search if available).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "User's request"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 200
                    }
                },
                "required": ["user_request"]
            }
        },
        {
            "name": "search_database",
            "description": f"🎖️ PRIORITY #{'8' if CLAP_ENABLED else '7'}: MOST GENERAL (last resort) - Search by genre/mood/tempo/energy/year/rating/scale filters. ✅ USE for: genre/mood/tempo combinations when NO specific artists/songs mentioned AND text_search not available/suitable. ❌ DON'T USE if you can use other more specific tools. COMBINE all filters in ONE call! Use 1-3 SPECIFIC genres (not 'rock' which matches everything).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Genres (rock, pop, metal, jazz, etc.)"
                    },
                    "moods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Moods (danceable, aggressive, happy, party, relaxed, sad)"
                    },
                    "tempo_min": {
                        "type": "number",
                        "description": "Min BPM (40-200)"
                    },
                    "tempo_max": {
                        "type": "number",
                        "description": "Max BPM (40-200)"
                    },
                    "energy_min": {
                        "type": "number",
                        "description": "Min energy 0.0 (calm) to 1.0 (intense)"
                    },
                    "energy_max": {
                        "type": "number",
                        "description": "Max energy 0.0 (calm) to 1.0 (intense)"
                    },
                    "key": {
                        "type": "string",
                        "description": "Musical key (C, D, E, F, G, A, B with # or b)"
                    },
                    "scale": {
                        "type": "string",
                        "enum": ["major", "minor"],
                        "description": "Musical scale: major or minor"
                    },
                    "year_min": {
                        "type": "integer",
                        "description": "Earliest release year (e.g. 1990)"
                    },
                    "year_max": {
                        "type": "integer",
                        "description": "Latest release year (e.g. 1999)"
                    },
                    "min_rating": {
                        "type": "integer",
                        "description": "Minimum user rating 1-5"
                    },
                    "album": {
                        "type": "string",
                        "description": "Album name to filter by (e.g. 'Abbey Road', 'Thriller')"
                    },
                    "artist": {
                        "type": "string",
                        "description": "Artist name - returns ONLY songs BY this artist (e.g. 'Madonna', 'Blink-182')"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 200
                    }
                }
            }
        }
    ])
    
    return tools
