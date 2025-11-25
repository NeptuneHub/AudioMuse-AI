"""
MCP Client for AudioMuse-AI
Handles MCP tool calling for different AI providers (Gemini, OpenAI, Mistral, Ollama)
"""
import json
import logging
from typing import List, Dict, Any, Optional

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
        import google.generativeai as genai
        
        api_key = ai_config.get('gemini_key')
        model_name = ai_config.get('gemini_model', 'gemini-2.5-pro')
        
        if not api_key or api_key == "YOUR-GEMINI-API-KEY-HERE":
            return {"error": "Valid Gemini API key required"}
        
        genai.configure(api_key=api_key)
        
        # Convert MCP tools to Gemini function declarations
        function_declarations = []
        for tool in tools:
            func_decl = {
                "name": tool['name'],
                "description": tool['description'],
                "parameters": tool['inputSchema']
            }
            function_declarations.append(func_decl)
        
        # Create model with tools
        model = genai.GenerativeModel(
            model_name=model_name,
            tools=function_declarations
        )
        
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
        
        # Generate response with function calling
        chat = model.start_chat()
        response = chat.send_message(f"{system_instruction}\n\nUser request: {user_message}")
        
        log_messages.append(f"Gemini response type: {type(response)}")
        
        # Extract function calls
        tool_calls = []
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call'):
                    fc = part.function_call
                    tool_calls.append({
                        "name": fc.name,
                        "arguments": dict(fc.args)
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
        
        with httpx.Client(timeout=60.0) as client:
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
        for tool in tools:
            props = tool['inputSchema'].get('properties', {})
            required = tool['inputSchema'].get('required', [])
            params_desc = ", ".join([f"{k} ({v.get('type')})" for k, v in props.items()])
            tools_list.append(f"• {tool['name']}: {tool['description']}\n  Parameters: {params_desc}")
        
        tools_text = "\n".join(tools_list)
        
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

YOU HAVE 4 TOOLS:
1. song_similarity - EXACT API: similar songs (needs title+artist)
2. artist_similarity - EXACT API: songs from similar artists (NOT artist's own songs)
3. search_database - EXACT DB: filter by genre/mood/tempo/energy (COMBINE all in ONE call)
4. ai_brainstorm - AI KNOWLEDGE: everything else (artist's own songs, trending, era, complex requests)

STEP 1 - ANALYZE INTENT:
What does the user want?
- Specific song+artist mentioned? → song_similarity (exact API)
- "songs like [ARTIST]" (similar artists)? → artist_similarity (exact API)
- Genre/mood/tempo/energy filters? → search_database (exact DB)
- Artist's OWN songs, trending, era, complex? → ai_brainstorm (AI knowledge)

STEP 2 - SELECT TOOL:
"similar to [TITLE] by [ARTIST]" → song_similarity
"songs like [ARTIST]" → artist_similarity
"[GENRE]/[MOOD]/[TEMPO]/[ENERGY]" → search_database
"[ARTIST] songs", "trending", "era", "best of" → ai_brainstorm

CRITICAL RULES:
1. song_similarity NEEDS title+artist - no empty titles!
2. artist_similarity returns SIMILAR artists, NOT artist's own songs
3. search_database = genre/mood/tempo/energy ONLY (COMBINE all filters!)
4. ai_brainstorm = DEFAULT for complex requests
5. Match ACTUAL user request - don't invent different requests!

VALID search_database VALUES:
GENRES: rock, pop, metal, jazz, electronic, dance, alternative, indie, punk, blues, hard rock, heavy metal, Hip-Hop, funk, country, 00s, 90s, 80s, 70s, 60s
MOODS: danceable, aggressive, happy, party, relaxed, sad
TEMPO: 40-200 BPM | ENERGY: 0.01-0.15

KEY EXAMPLES:

"Similar to By the Way by Red Hot Chili Peppers"
{{
  "tool_calls": [{{"name": "song_similarity", "arguments": {{"song_title": "By the Way", "song_artist": "Red Hot Chili Peppers", "get_songs": 100}}}}]
}}

"songs like blink-182" (similar artists, NOT blink-182's own)
{{
  "tool_calls": [{{"name": "artist_similarity", "arguments": {{"artist": "blink-182", "get_songs": 100}}}}]
}}

"blink-182 songs" (blink-182's OWN songs)
{{
  "tool_calls": [{{"name": "ai_brainstorm", "arguments": {{"user_request": "blink-182 songs", "get_songs": 100}}}}]
}}

"running 120 bpm"
{{
  "tool_calls": [{{"name": "search_database", "arguments": {{"tempo_min": 115, "tempo_max": 125, "energy_min": 0.08, "get_songs": 100}}}}]
}}

"energetic rock"
{{
  "tool_calls": [{{"name": "search_database", "arguments": {{"genres": ["rock"], "energy_min": 0.08, "moods": ["happy"], "get_songs": 100}}}}]
}}

"post lunch" (relaxed mood, NOT "popular songs"!)
{{
  "tool_calls": [{{"name": "search_database", "arguments": {{"moods": ["relaxed"], "energy_min": 0.03, "energy_max": 0.08, "tempo_min": 80, "tempo_max": 110, "get_songs": 100}}}}]
}}

"trending 2025"
{{
  "tool_calls": [{{"name": "ai_brainstorm", "arguments": {{"user_request": "trending 2025", "get_songs": 100}}}}]
}}

"greatest hits Red Hot Chili Peppers"
{{
  "tool_calls": [{{"name": "ai_brainstorm", "arguments": {{"user_request": "greatest hits RHCP", "get_songs": 100}}}}]
}}

"Metal like AC/DC"
{{
  "tool_calls": [
    {{"name": "artist_similarity", "arguments": {{"artist": "AC/DC", "get_songs": 60}}}},
    {{"name": "search_database", "arguments": {{"genres": ["metal", "hard rock"], "energy_min": 0.08, "get_songs": 40}}}}
  ]
}}

Now analyze this request and call tools:

Request: "{user_message}"

Return ONLY the JSON object with tool_calls array:"""
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        with httpx.Client(timeout=120.0) as client:
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
    
    except Exception as e:
        logger.exception("Error calling Ollama with tools")
        return {"error": f"Ollama error: {str(e)}"}


def execute_mcp_tool(tool_name: str, tool_args: Dict, ai_config: Dict) -> Dict:
    """Execute an MCP tool - 4 CORE TOOLS."""
    from tasks.mcp_server import (_artist_similarity_api_sync, _song_similarity_api_sync, 
                                    _database_genre_query_sync, _ai_brainstorm_sync)
    
    try:
        if tool_name == "artist_similarity":
            return _artist_similarity_api_sync(
                tool_args['artist'],
                15,  # count - hardcoded
                tool_args.get('get_songs', 100)
            )
        elif tool_name == "song_similarity":
            return _song_similarity_api_sync(
                tool_args['song_title'],
                tool_args['song_artist'],
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
    """Get the list of available MCP tools - 4 CORE TOOLS."""
    return [
        {
            "name": "artist_similarity",
            "description": "EXACT API: Find songs from similar artists (NOT the artist's own songs).",
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
            "name": "song_similarity",
            "description": "EXACT API: Find songs similar to a specific song (requires title+artist).",
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
        },
        {
            "name": "search_database",
            "description": "EXACT DB: Search by genre/mood/tempo/energy filters. COMBINE all filters in ONE call!",
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
        },
        {
            "name": "ai_brainstorm",
            "description": "AI KNOWLEDGE: Suggest songs for complex requests (artist's own songs, trending, era, etc.).",
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
        }
    ]
