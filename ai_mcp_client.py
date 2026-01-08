"""
MCP Client for AudioMuse-AI
Handles MCP tool calling for different AI providers (Gemini, OpenAI, Mistral, Ollama)
"""
import json
import logging
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger(__name__)


def call_ai_with_mcp_tools(
    provider: str,
    user_message: str,
    tools: List[Dict],
    ai_config: Dict,
    log_messages: List[str]
) -> Dict:
    """
    Call AI provider with MCP tool definitions and handle tool calling flow.
    
    Args:
        provider: AI provider ('GEMINI', 'OPENAI', 'MISTRAL', 'OLLAMA')
        user_message: The user's natural language request
        tools: List of MCP tool definitions
        ai_config: Configuration dict with API keys, URLs, model names
        log_messages: List to append log messages to
    
    Returns:
        Dict with 'tool_calls' (list of tool calls) or 'error' (error message)
    """
    if provider == "GEMINI":
        return _call_gemini_with_tools(user_message, tools, ai_config, log_messages)
    elif provider == "OPENAI":
        return _call_openai_with_tools(user_message, tools, ai_config, log_messages)
    elif provider == "MISTRAL":
        return _call_mistral_with_tools(user_message, tools, ai_config, log_messages)
    elif provider == "OLLAMA":
        return _call_ollama_with_tools(user_message, tools, ai_config, log_messages)
    else:
        return {"error": f"Unsupported AI provider: {provider}"}


def _call_gemini_with_tools(user_message: str, tools: List[Dict], ai_config: Dict, log_messages: List[str]) -> Dict:
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
        
        # System instruction for playlist generation
        system_instruction = """You are an expert music playlist curator with access to a music database.

Your task is to analyze the user's request and determine which tools to call to build a great playlist.

IMPORTANT RULES:
1. Call tools to gather songs - you can call multiple tools
2. Each tool returns a list of songs with item_id, title, and artist
3. Combine results from multiple tool calls if needed
4. Return ONLY tool calls - do not provide text responses yet

Available strategies:
- For artist requests: Use artist_similarity or artist_hits
- For genre/mood: Use search_by_genre
- For energy/tempo: Use search_by_tempo_energy
- For vibe descriptions: Use vibe_match
- For specific songs: Use song_similarity
- To check what's available: Use explore_database first

Call the appropriate tools now to fulfill the user's request."""
        
        # Prepare tools for new API
        tools_list = [genai.types.Tool(function_declarations=function_declarations)]
        
        # Generate response with function calling using new API
        # Note: Using 'ANY' mode to force tool calling instead of text response
        response = client.models.generate_content(
            model=model_name,
            contents=f"{system_instruction}\n\nUser request: {user_message}",
            config=genai.types.GenerateContentConfig(
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


def _call_openai_with_tools(user_message: str, tools: List[Dict], ai_config: Dict, log_messages: List[str]) -> Dict:
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
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert music playlist curator with access to a music database.

Analyze the user's request and call the appropriate tools to build a playlist.

Rules:
1. Call one or more tools to gather songs
2. Each tool returns songs with item_id, title, and artist
3. Choose tools based on the request type:
   - Artist requests → artist_similarity or artist_hits
   - Genre/mood → search_by_genre
   - Energy/tempo → search_by_tempo_energy
   - Vibe descriptions → vibe_match
   - Specific songs → song_similarity
   - Check availability → explore_database

Call the tools needed to fulfill the request."""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "tools": functions,
            "tool_choice": "auto"
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


def _call_mistral_with_tools(user_message: str, tools: List[Dict], ai_config: Dict, log_messages: List[str]) -> Dict:
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
        
        # Call Mistral
        response = client.chat.complete(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert music playlist curator with access to a music database.

Analyze the user's request and call the appropriate tools to build a playlist.

Rules:
1. Call one or more tools to gather songs
2. Choose tools based on request type:
   - Artists → artist_similarity or artist_hits
   - Genres → search_by_genre
   - Energy/tempo → search_by_tempo_energy
   - Vibes → vibe_match

Call the tools now."""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            tools=mistral_tools,
            tool_choice="auto"
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


def _call_ollama_with_tools(user_message: str, tools: List[Dict], ai_config: Dict, log_messages: List[str]) -> Dict:
    """
    Call Ollama with tool definitions.
    Note: Ollama's tool calling support varies by model. This uses a prompt-based approach.
    """
    try:
        import httpx
        
        ollama_url = ai_config.get('ollama_url', 'http://localhost:11434/api/generate')
        model_name = ai_config.get('ollama_model', 'llama3.1:8b')
        
        # Build simpler tool list for Ollama
        tools_list = []
        has_text_search = False
        for tool in tools:
            if tool['name'] == 'text_search':
                has_text_search = True
            props = tool['inputSchema'].get('properties', {})
            required = tool['inputSchema'].get('required', [])
            params_desc = ", ".join([f"{k} ({v.get('type')})" for k, v in props.items()])
            tools_list.append(f"• {tool['name']}: {tool['description']}\n  Parameters: {params_desc}")
        
        tools_text = "\n".join(tools_list)
        
        # Build tool priority list dynamically
        tool_count = len(tools)
        tool_priorities = []
        tool_priorities.append("1. song_similarity - EXACT API: similar songs (needs title+artist)")
        if has_text_search:
            tool_priorities.append("2. text_search - CLAP SEARCH: natural language search for instruments, moods, descriptive queries")
            tool_priorities.append("3. artist_similarity - EXACT API: songs from similar artists (NOT artist's own songs)")
            tool_priorities.append("4. song_alchemy - VECTOR MATH: blend/subtract artists/songs")
            tool_priorities.append("5. ai_brainstorm - AI KNOWLEDGE: artist's own songs, trending, era, complex requests")
            tool_priorities.append("6. search_database - EXACT DB: filter by genre/mood/tempo/energy (LAST RESORT)")
        else:
            tool_priorities.append("2. artist_similarity - EXACT API: songs from similar artists (NOT artist's own songs)")
            tool_priorities.append("3. song_alchemy - VECTOR MATH: blend/subtract artists/songs")
            tool_priorities.append("4. ai_brainstorm - AI KNOWLEDGE: artist's own songs, trending, era, complex requests")
            tool_priorities.append("5. search_database - EXACT DB: filter by genre/mood/tempo/energy (LAST RESORT)")
        
        tool_priorities_text = "\n".join(tool_priorities)
        
        # Build decision tree dynamically
        decision_steps = []
        if has_text_search:
            decision_steps.append("- Specific song+artist mentioned? → song_similarity (exact API)")
            decision_steps.append("- ⚠️ INSTRUMENTS mentioned (piano, guitar, drums, violin, saxophone, etc.)? → text_search (CLAP) - NEVER use search_database for instruments!")
            decision_steps.append("- Descriptive/subjective moods (romantic, chill, melancholic, dreamy, uplifting)? → text_search (CLAP)")
            decision_steps.append("- 'songs like [ARTIST]' (similar artists)? → artist_similarity (exact API)")
            decision_steps.append("- 'sounds like [ARTIST1] + [ARTIST2]' or 'like X but NOT Y'? → song_alchemy (vector math)")
            decision_steps.append("- Artist's OWN songs, trending, era, complex? → ai_brainstorm (AI knowledge)")
            decision_steps.append("- Database genres/moods ONLY (rock, pop, metal, jazz - NO instruments)? → search_database (exact DB)")
        else:
            decision_steps.append("- Specific song+artist mentioned? → song_similarity (exact API)")
            decision_steps.append("- 'songs like [ARTIST]' (similar artists)? → artist_similarity (exact API)")
            decision_steps.append("- 'sounds like [ARTIST1] + [ARTIST2]' or 'like X but NOT Y'? → song_alchemy (vector math)")
            decision_steps.append("- Artist's OWN songs, trending, era, complex? → ai_brainstorm (AI knowledge)")
            decision_steps.append("- Genre/mood/tempo/energy filters only? → search_database (exact DB)")
        
        decision_steps_text = "\n".join(decision_steps)
        
        # Build examples dynamically
        examples = []
        examples.append("""
"Similar to By the Way by Red Hot Chili Peppers"
{{
  "tool_calls": [{{"name": "song_similarity", "arguments": {{"song_title": "By the Way", "song_artist": "Red Hot Chili Peppers", "get_songs": 100}}}}]
}}""")
        
        if has_text_search:
            examples.append("""
"calm piano song"
{{
  "tool_calls": [{{"name": "text_search", "arguments": {{"description": "calm piano", "get_songs": 100}}}}]
}}""")
            examples.append("""
"romantic acoustic guitar"
{{
  "tool_calls": [{{"name": "text_search", "arguments": {{"description": "romantic acoustic guitar", "get_songs": 100}}}}]
}}""")
            examples.append("""
"energetic ukulele songs"
{{
  "tool_calls": [{{"name": "text_search", "arguments": {{"description": "energetic ukulele", "energy_filter": "high", "get_songs": 100}}}}]
}}""")
        
        examples.append("""
"songs like blink-182" (similar artists, NOT blink-182's own)
{{
  "tool_calls": [{{"name": "artist_similarity", "arguments": {{"artist": "blink-182", "get_songs": 100}}}}]
}}""")
        
        examples.append("""
"blink-182 songs" (blink-182's OWN songs)
{{
  "tool_calls": [{{"name": "ai_brainstorm", "arguments": {{"user_request": "blink-182 songs", "get_songs": 100}}}}]
}}""")
        
        examples.append("""
"running 120 bpm"
{{
  "tool_calls": [{{"name": "search_database", "arguments": {{"tempo_min": 115, "tempo_max": 125, "energy_min": 0.08, "get_songs": 100}}}}]
}}""")
        
        examples_text = "\n".join(examples)
        
        prompt = f"""You are a music playlist curator. Analyze this request and decide which tools to call.

Request: "{user_message}"

Available tools:
{tools_text}

CRITICAL RULES:
1. Return ONLY valid JSON object (not an array)
2. Use this EXACT format:
{{
  "tool_calls": [
    {{"name": "tool_name", "arguments": {{"param": "value"}}}},
    {{"name": "tool_name2", "arguments": {{"param": "value"}}}}
  ]
}}

YOU HAVE {tool_count} TOOLS (in priority order):
{tool_priorities_text}

STEP 1 - ANALYZE INTENT:
What does the user want?
{decision_steps_text}

CRITICAL RULES:
1. song_similarity NEEDS title+artist - no empty titles!
2. ⚠️ INSTRUMENTS → text_search! If query mentions INSTRUMENTS (piano, guitar, drums, violin, saxophone, trumpet, flute, bass, ukulele, harmonica), you MUST use text_search, NOT search_database!
3. text_search is ALSO BEST for descriptive/subjective moods (romantic, chill, sad, melancholic, uplifting, dreamy)
4. artist_similarity returns SIMILAR artists, NOT artist's own songs
5. search_database = ONLY for database genres/moods listed below (NOT instruments!)
6. ai_brainstorm = DEFAULT for complex requests
7. Match ACTUAL user request - don't invent different requests!

⚠️ CRITICAL DISTINCTION:
- INSTRUMENTS (piano, guitar, drums) → text_search
- GENRES (rock, pop, metal, jazz) → search_database
- "piano" is NOT a genre! Use text_search for instruments!

VALID search_database VALUES (ONLY THESE):
GENRES: rock, pop, metal, jazz, electronic, dance, alternative, indie, punk, blues, hard rock, heavy metal, Hip-Hop, funk, country, 00s, 90s, 80s, 70s, 60s
MOODS: danceable, aggressive, happy, party, relaxed, sad
TEMPO: 40-200 BPM | ENERGY: 0.01-0.15
⚠️ NOTE: Instruments like "piano" are NOT valid genres! Use text_search instead!

KEY EXAMPLES:
{examples_text}

"energetic rock"
{{
  "tool_calls": [{{"name": "search_database", "arguments": {{"genres": ["rock"], "energy_min": 0.08, "moods": ["happy"], "get_songs": 100}}}}]
}}

"trending 2025"
{{
  "tool_calls": [{{"name": "ai_brainstorm", "arguments": {{"user_request": "trending 2025", "get_songs": 100}}}}]
}}

⚠️ WRONG EXAMPLES (DO NOT DO THIS):
❌ "piano songs" with search_database genres=["piano"] → WRONG! Piano is an instrument, not a genre. Use text_search instead.
❌ "guitar music" with search_database genres=["guitar"] → WRONG! Guitar is an instrument. Use text_search.
✅ "piano songs" → Use text_search with description="piano"
✅ "calm piano" → Use text_search with description="calm piano"

Now analyze this request and call tools:

Request: "{user_message}"

Return ONLY the JSON object with tool_calls array:"""
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
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
        
        # Try to extract JSON
        try:
            cleaned = response_text.strip()
            
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
            else:
                log_messages.append(f"⚠️ Unexpected JSON structure: {type(parsed)}, keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")
                return {"error": "Ollama response missing 'tool_calls' field"}
            
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            
            # Validate tool calls structure
            valid_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict) and 'name' in tc:
                    # Ensure arguments is a dict
                    if 'arguments' not in tc:
                        tc['arguments'] = {}
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
                tool_args.get('get_songs', 100)
            )
        elif tool_name == "text_search":
            return _text_search_sync(
                tool_args['description'],
                tool_args.get('tempo_filter'),
                tool_args.get('energy_filter'),
                tool_args.get('get_songs', 100)
            )
        elif tool_name == "song_similarity":
            return _song_similarity_api_sync(
                tool_args['song_title'],
                tool_args['song_artist'],
                tool_args.get('get_songs', 100)
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
                tool_args.get('get_songs', 100)
            )
        elif tool_name == "search_database":
            return _database_genre_query_sync(
                tool_args.get('genres'),
                tool_args.get('get_songs', 100),
                tool_args.get('moods'),
                tool_args.get('tempo_min'),
                tool_args.get('tempo_max'),
                tool_args.get('energy_min'),
                tool_args.get('energy_max'),
                tool_args.get('key')
            )
        elif tool_name == "ai_brainstorm":
            return _ai_brainstorm_sync(
                tool_args['user_request'],
                ai_config,
                tool_args.get('get_songs', 100)
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
    3. ARTIST_SIMILARITY - for songs FROM specific artist(s)
    4. SONG_ALCHEMY - for 'sounds LIKE' blending multiple artists/songs
    5. AI_BRAINSTORM - for world knowledge (artist's own songs, era, awards)
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
                        "default": 100
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
            "description": "🥈 PRIORITY #2: HIGH PRIORITY - Natural language search using CLAP. ✅ USE for: INSTRUMENTS (piano, guitar, ukulele), SPECIFIC MOODS (romantic, sad, happy), DESCRIPTIVE QUERIES ('chill vibes', 'energetic workout'). Supports optional tempo/energy filters for hybrid search.",
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
                        "default": 100
                    }
                },
                "required": ["description"]
            }
        })
    
    tools.extend([
        {
            "name": "artist_similarity",
            "description": f"🥉 PRIORITY #{'3' if CLAP_ENABLED else '2'}: Find songs FROM similar artists (NOT the artist's own songs). ✅ USE for: 'songs FROM Artist X, Artist Y' (call once per artist). ❌ DON'T USE for: 'sounds LIKE multiple artists' (use song_alchemy).",
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
                        "default": 100
                    }
                },
                "required": ["artist"]
            }
        },
        {
            "name": "song_alchemy",
            "description": f"🏅 PRIORITY #{'4' if CLAP_ENABLED else '3'}: VECTOR ARITHMETIC - Blend or subtract artists/songs using musical math. ✅ BEST for: 'SOUNDS LIKE / PLAY LIKE multiple artists' ('play like Iron Maiden, Metallica, Deep Purple'), 'like X but NOT Y', 'Artist A meets Artist B'. ❌ DON'T USE for: 'songs FROM artists' (use artist_similarity), single artist (use artist_similarity), genre/mood (use search_database). Examples: 'play like Iron Maiden + Metallica + Deep Purple' = add all 3; 'Beatles but not ballads' = add Beatles, subtract ballads.",
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
                        "default": 100
                    }
                },
                "required": ["add_items"]
            }
        },
        {
            "name": "ai_brainstorm",
            "description": f"🏅 PRIORITY #{'5' if CLAP_ENABLED else '4'}: AI world knowledge - Use ONLY when other tools CAN'T work. ✅ USE for: artist's OWN songs, specific era/year, trending songs, award winners, chart hits. ❌ DON'T USE for: 'sounds like' (use song_alchemy), artist similarity (use artist_similarity), genre/mood (use search_database), instruments/moods (use text_search if available).",
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
                        "default": 100
                    }
                },
                "required": ["user_request"]
            }
        },
        {
            "name": "search_database",
            "description": f"🎖️ PRIORITY #{'6' if CLAP_ENABLED else '5'}: MOST GENERAL (last resort) - Search by genre/mood/tempo/energy filters. ✅ USE for: genre/mood/tempo combinations when NO specific artists/songs mentioned AND text_search not available/suitable. ❌ DON'T USE if you can use other more specific tools. COMBINE all filters in ONE call!",
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
                        "description": "Min energy (0.01-0.15)"
                    },
                    "energy_max": {
                        "type": "number",
                        "description": "Max energy (0.01-0.15)"
                    },
                    "key": {
                        "type": "string",
                        "description": "Musical key (C, D, E, F, G, A, B with # or b)"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 100
                    }
                }
            }
        }
    ])
    
    return tools
