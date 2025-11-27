# app_chat.py
from flask import Blueprint, render_template, request, jsonify
from flasgger import swag_from # Import swag_from
from psycopg2.extras import DictCursor # To get results as dictionaries
import unicodedata # For ASCII normalization
import sqlglot # Import sqlglot
import json # For potential future use with more complex AI responses
import html # For unescaping HTML entities
import logging
import re # For regex-based quote escaping


logger = logging.getLogger(__name__)
# Import AI configuration from the main config.py
# This assumes config.py is in the same directory as app_chat.py or accessible via Python path.
from config import (
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME,
    OPENAI_SERVER_URL, OPENAI_MODEL_NAME, OPENAI_API_KEY, # Import OpenAI config
    GEMINI_MODEL_NAME, GEMINI_API_KEY, # Import GEMINI_API_KEY from config
    MISTRAL_MODEL_NAME, MISTRAL_API_KEY,
    AI_MODEL_PROVIDER, # Default AI provider
    AI_CHAT_DB_USER_NAME, AI_CHAT_DB_USER_PASSWORD, # Import new config
)
from ai import (
    get_gemini_playlist_name, get_ollama_playlist_name, get_mistral_playlist_name, 
    get_openai_compatible_playlist_name, call_ai_for_chat,
    chat_step1_understand_prompt, chat_step2_expand_prompt, chat_step3_explore_prompt
) # Import functions to call AI
from tasks.chat_manager import call_ai_step, explore_database_for_matches, execute_action

# Create a Blueprint for chat-related routes
chat_bp = Blueprint('chat_bp', __name__,
                    template_folder='templates', # Specifies where to look for templates like chat.html
                    static_folder='static')

ai_user_setup_done = False # Module-level flag to run setup once

def _ensure_ai_user_configured(db_conn):
    """
    Ensures the AI_USER exists and has SELECT ONLY privileges on public.score.
    This function should be called with a connection that has privileges to create users/roles and grant permissions.
    """
    global ai_user_setup_done
    if ai_user_setup_done:
        return

    try:
        with db_conn.cursor() as cur:
            # Check if role exists
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (AI_CHAT_DB_USER_NAME,))
            user_exists = cur.fetchone()

            if not user_exists:
                logger.info("Creating database user: %s", AI_CHAT_DB_USER_NAME)
                cur.execute(f"CREATE USER {AI_CHAT_DB_USER_NAME} WITH PASSWORD %s;", (AI_CHAT_DB_USER_PASSWORD,))
                logger.info("User %s created.", AI_CHAT_DB_USER_NAME)
            else:
                logger.info("User %s already exists.", AI_CHAT_DB_USER_NAME)

            # Grant necessary permissions
            cur.execute("SELECT current_database();")
            current_db_name = cur.fetchone()[0]

            logger.info("Granting CONNECT ON DATABASE %s TO %s", current_db_name, AI_CHAT_DB_USER_NAME)
            cur.execute(f"GRANT CONNECT ON DATABASE {current_db_name} TO {AI_CHAT_DB_USER_NAME};")

            logger.info("Granting USAGE ON SCHEMA public TO %s", AI_CHAT_DB_USER_NAME)
            cur.execute(f"GRANT USAGE ON SCHEMA public TO {AI_CHAT_DB_USER_NAME};")
            
            logger.info("Granting SELECT ON public.score TO %s", AI_CHAT_DB_USER_NAME)
            cur.execute(f"GRANT SELECT ON TABLE public.score TO {AI_CHAT_DB_USER_NAME};")
            
            # Revoke all other default privileges on public schema if necessary (more secure)
            # This is an advanced step and might be too restrictive if other public tables are needed by this user.
            # For now, we rely on explicit grants.
            # logger.info(f"Revoking ALL ON SCHEMA public FROM {AI_CHAT_DB_USER_NAME} (except USAGE already granted)")
            # cur.execute(f"REVOKE ALL ON SCHEMA public FROM {AI_CHAT_DB_USER_NAME};") # This revokes USAGE too
            # cur.execute(f"GRANT USAGE ON SCHEMA public TO {AI_CHAT_DB_USER_NAME};") # Re-grant USAGE

            db_conn.commit()
            logger.info("Permissions configured for user %s.", AI_CHAT_DB_USER_NAME)
            ai_user_setup_done = True
    except Exception as e:
        logger.error("Error during AI user setup. AI user might not be correctly configured.", exc_info=True)
        db_conn.rollback()
        # ai_user_setup_done remains False, so it might try again on next request.

def clean_and_validate_sql(raw_sql):
    """Cleans and performs basic validation on the SQL query."""
    if not raw_sql or not isinstance(raw_sql, str):
        return None, "Received empty or invalid SQL from AI."

    cleaned_sql = raw_sql.strip()
    if cleaned_sql.startswith("```sql"):
        cleaned_sql = cleaned_sql[len("```sql"):]
    if cleaned_sql.endswith("```"):
        cleaned_sql = cleaned_sql[:-len("```")]
    
    # Unescape HTML entities early (e.g., &gt; becomes >)
    cleaned_sql = html.unescape(cleaned_sql)

    # Further cleaning: find the first occurrence of SELECT (case-insensitive)
    # and take everything from there. This helps if there's leading text.
    select_pos = cleaned_sql.upper().find("SELECT")
    if select_pos == -1:
        return None, "Query does not contain SELECT."
    cleaned_sql = cleaned_sql[select_pos:] # Start the string from "SELECT"

    cleaned_sql = cleaned_sql.strip() # Strip again after taking the substring

    if not cleaned_sql.upper().startswith("SELECT"):
        # This case should ideally not be hit if the above find("SELECT") logic works,
        # but it's a good fallback.
        return None, "Cleaned query does not start with SELECT."

    # 1. Normalize Unicode characters (e.g., ’ -> ', è -> e) to standard ASCII representations.
    #    This helps standardize different types of quote characters to a simple apostrophe.
    try:
        cleaned_sql = unicodedata.normalize('NFKD', cleaned_sql).encode('ascii', 'ignore').decode('utf-8')
    except Exception as e_norm:
        logger.warning("Could not fully normalize SQL string to ASCII: %s", e_norm)

    # 2. Convert C-style escaped single quotes (e.g., \') to SQL standard double single quotes ('').
    #    This should be done after normalization, in case normalization affects the backslash,
    #    and before the regex, to correctly handle cases like "Player\'s".
    cleaned_sql = cleaned_sql.replace("\\'", "''")

    # 3. Fix unescaped single quotes *within* words that might remain after normalization
    #    and the \'-to-'' conversion.
    #    Example: "Player's" (from "Player’s" or direct output) becomes "Player''s".
    #    This regex looks for a word character, a single quote, and another word character.
    cleaned_sql = re.sub(r"(\w)'(\w)", r"\1''\2", cleaned_sql)

    try:
        # Parse the query using sqlglot, specifying PostgreSQL dialect
        parsed_expressions = sqlglot.parse(cleaned_sql, read='postgres')
        if not parsed_expressions: # Should not happen if it starts with SELECT but good check
            return None, "SQLglot could not parse the query."
        
        # Get the first (and should be only) expression
        expression = parsed_expressions[0]

        # Re-generate the SQL from the potentially modified structure.
        cleaned_sql = expression.sql(dialect='postgres', pretty=False).strip().rstrip(';')
    except sqlglot.errors.ParseError as e:
        # Log full parse exception server-side for diagnostics, but do not expose
        # parser internals to API clients.
        logger.exception("SQLglot parsing error while validating AI SQL")
        return None, "SQL parsing error"
    return cleaned_sql, None

@chat_bp.route('/')
@swag_from({
    'tags': ['Chat UI'],
    'summary': 'Serves the main chat interface HTML page.',
    'responses': {
        '200': {
            'description': 'HTML content of the chat page.',
            'content': {
                'text/html': {
                    'schema': {'type': 'string'}
                }
            }
        }
    }
})
def chat_home():
    """
    Serves the main chat page.
    """
    return render_template('chat.html', title = 'AudioMuse-AI - Instant Playlist', active='chat')

@chat_bp.route('/api/config_defaults', methods=['GET'])
@swag_from({
    'tags': ['Chat Configuration'],
    'summary': 'Get default AI configuration for the chat interface.',
    'responses': {
        '200': {
            'description': 'Default AI configuration.',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'default_ai_provider': {
                                'type': 'string', 'example': 'OLLAMA'
                            },
                            'default_ollama_model_name': {
                                'type': 'string', 'example': 'mistral:7b'
                            },
                            'ollama_server_url': {
                                'type': 'string', 'example': 'http://127.0.0.1:11434/api/generate'
                            },
                            'default_openai_model_name': {
                                'type': 'string', 'example': 'gpt-4'
                            },
                            'openai_server_url': {
                                'type': 'string', 'example': 'https://openrouter.ai/api/v1/chat/completions'
                            },
                            'default_gemini_model_name': {
                                'type': 'string', 'example': 'gemini-2.5-pro'
                            },
                            'default_mistral_model_name': {
                                'type': 'string', 'example': 'ministral-3b-latest'
                            },
                        }
                    }
                }
            }
        }
    }
})
def chat_config_defaults_api():
    """
    API endpoint to provide default configuration values for the chat interface.
    """
    # The default_gemini_api_key is no longer sent to the front end for security.
    return jsonify({
        "default_ai_provider": AI_MODEL_PROVIDER,
        "default_ollama_model_name": OLLAMA_MODEL_NAME,
        "ollama_server_url": OLLAMA_SERVER_URL, # Ollama server URL might be useful for display/info
        "default_openai_model_name": OPENAI_MODEL_NAME,
        "openai_server_url": OPENAI_SERVER_URL, # OpenAI server URL for display/info
        "default_gemini_model_name": GEMINI_MODEL_NAME,
        "default_mistral_model_name": MISTRAL_MODEL_NAME,
    }), 200

@chat_bp.route('/api/chatPlaylist', methods=['POST'])
@swag_from({
    'tags': ['Chat Interaction'],
    'summary': 'Process user chat input to generate a playlist idea using AI.',
    'requestBody': {
        'description': 'User input and AI configuration for generating a playlist.',
        'required': True,
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'required': ['userInput'],
                    'properties': {
                        'userInput': {
                            'type': 'string',
                            'description': "The user's natural language request for a playlist.",
                            'example': "Songs for a rainy afternoon"
                        },
                        'ai_provider': {
                            'type': 'string',
                            'description': 'The AI provider to use (OLLAMA, OPENAI, GEMINI, MISTRAL, NONE). Defaults to server config.',
                            'example': 'GEMINI',
                            'enum': ['OLLAMA', 'OPENAI', 'GEMINI', "MISTRAL", 'NONE']
                        },
                        'ai_model': {
                            'type': 'string',
                            'description': 'The specific AI model name to use. Defaults to server config for the provider.',
                            'example': 'gemini-2.5-pro'
                        },
                        'ollama_server_url': {
                            'type': 'string',
                            'description': 'Custom Ollama server URL (if ai_provider is OLLAMA).',
                            'example': 'http://localhost:11434/api/generate'
                        },
                        'openai_server_url': {
                            'type': 'string',
                            'description': 'Custom OpenAI/OpenRouter server URL (if ai_provider is OPENAI).',
                            'example': 'https://openrouter.ai/api/v1/chat/completions'
                        },
                        'openai_api_key': {
                            'type': 'string',
                            'description': 'OpenAI/OpenRouter API key (required if ai_provider is OPENAI).',
                        },
                        'gemini_api_key': {
                            'type': 'string',
                            'description': 'Custom Gemini API key (optional, defaults to server configuration).',
                        },
                        'mistral_api_key': {
                            'type': 'string',
                            'description': 'Custom Mistral API key (optional, defaults to server configuration).',
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': 'AI response containing the playlist idea, SQL query, and processing log.',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'response': {
                                'type': 'object',
                                'properties': {
                                    'message': {
                                        'type': 'string',
                                        'description': 'Log of AI interaction and processing.'
                                    },
                                    'original_request': {
                                        'type': 'string',
                                        'description': "The user's original input."
                                    },
                                    'ai_provider_used': {
                                        'type': 'string',
                                        'description': 'The AI provider that was used for the request.'
                                    },
                                    'ai_model_selected': {
                                        'type': 'string',
                                        'description': 'The specific AI model that was selected/used.'
                                    },
                                    'executed_query': {
                                        'type': 'string',
                                        'nullable': True,
                                        'description': 'The SQL query that was executed (or last attempted).'
                                    },
                                    'query_results': {
                                        'type': 'array',
                                        'nullable': True,
                                        'description': 'List of songs returned by the query.',
                                        'items': {
                                            'type': 'object',
                                            'properties': {
                                                'item_id': {'type': 'string'},
                                                'title': {'type': 'string'},
                                                'artist': {'type': 'string'}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        '400': {
            'description': 'Bad Request - Missing input or invalid parameters.',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'error': {'type': 'string'}
                        }
                    }
                }
            }
        }
    }
})
def chat_playlist_api():
    """
    Process user chat input to generate a playlist using AI with MCP tools.
    
    MCP TOOLS (4 CORE):
    1. artist_similarity - Songs from similar artists
    2. song_similarity - Songs similar to a specific song  
    3. search_database - Search by genre, mood, tempo, energy, key (ALL filters in ONE call)
    4. ai_brainstorm - AI suggests famous songs (trending, top hits, radio classics, etc.)
    
    AI analyzes request → calls tools → combines results → returns 100 songs
    """
    data = request.get_json()
    # Mask API key if present in the debug log
    data_for_log = dict(data) if data else {}
    if 'gemini_api_key' in data_for_log and data_for_log['gemini_api_key']:
        data_for_log['gemini_api_key'] = 'API-KEY'
    if 'mistral_api_key' in data_for_log and data_for_log['mistral_api_key']:
        data_for_log['mistral_api_key'] = 'API-KEY'
    if 'openai_api_key' in data_for_log and data_for_log['openai_api_key']:
        data_for_log['openai_api_key'] = 'API-KEY'
    logger.debug("chat_playlist_api called. Raw request data: %s", data_for_log)
    
    from app_helper import get_db
    from ai_mcp_client import call_ai_with_mcp_tools, execute_mcp_tool, get_mcp_tools
    
    if not data or 'userInput' not in data:
        return jsonify({"error": "Missing userInput in request"}), 400

    original_user_input = data.get('userInput')
    ai_provider = data.get('ai_provider', AI_MODEL_PROVIDER).upper()
    ai_model_from_request = data.get('ai_model')
    
    log_messages = []
    log_messages.append(f"🎵 NEW MCP-BASED PLAYLIST GENERATION")
    log_messages.append(f"Request: '{original_user_input}'")
    log_messages.append(f"AI Provider: {ai_provider}")
    
    # Check if AI provider is NONE
    if ai_provider == "NONE":
        return jsonify({
            "response": {
                "message": "No AI provider selected. Please configure an AI provider to use this feature.",
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": None,
                "executed_query": None,
                "query_results": None
            }
        }), 200
    
    # Build AI configuration object
    ai_config = {
        'provider': ai_provider,
        'ollama_url': data.get('ollama_server_url', OLLAMA_SERVER_URL),
        'ollama_model': ai_model_from_request or OLLAMA_MODEL_NAME,
        'openai_url': data.get('openai_server_url', OPENAI_SERVER_URL),
        'openai_model': ai_model_from_request or OPENAI_MODEL_NAME,
        'openai_key': data.get('openai_api_key') or OPENAI_API_KEY,
        'gemini_key': data.get('gemini_api_key') or GEMINI_API_KEY,
        'gemini_model': ai_model_from_request or GEMINI_MODEL_NAME,
        'mistral_key': data.get('mistral_api_key') or MISTRAL_API_KEY,
        'mistral_model': ai_model_from_request or MISTRAL_MODEL_NAME
    }
    
    # Validate API keys for cloud providers
    if ai_provider == "OPENAI" and not ai_config['openai_key']:
        error_msg = "Error: OpenAI API key is missing. Please provide a valid API key."
        log_messages.append(error_msg)
        return jsonify({"response": {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": ai_config.get('openai_model'),
            "executed_query": None,
            "query_results": None
        }}), 400
    
    if ai_provider == "GEMINI" and (not ai_config['gemini_key'] or ai_config['gemini_key'] == "YOUR-GEMINI-API-KEY-HERE"):
        error_msg = "Error: Gemini API key is missing. Please provide a valid API key."
        log_messages.append(error_msg)
        return jsonify({"response": {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": ai_config.get('gemini_model'),
            "executed_query": None,
            "query_results": None
        }}), 400
    
    if ai_provider == "MISTRAL" and (not ai_config['mistral_key'] or ai_config['mistral_key'] == "YOUR-MISTRAL-API-KEY-HERE"):
        error_msg = "Error: Mistral API key is missing. Please provide a valid API key."
        log_messages.append(error_msg)
        return jsonify({"response": {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": ai_config.get('mistral_model'),
            "executed_query": None,
            "query_results": None
        }}), 400
    
    # ====================
    # MCP AGENTIC WORKFLOW
    # ====================
    
    log_messages.append("\n🤖 Using MCP Agentic Workflow for playlist generation")
    log_messages.append("Target: 100 songs")
    
    # Get MCP tools
    mcp_tools = get_mcp_tools()
    log_messages.append(f"Available tools: {', '.join([t['name'] for t in mcp_tools])}")
    
    # Agentic workflow - AI iteratively calls tools until enough songs
    all_songs = []
    song_ids_seen = set()
    song_sources = {}  # Maps item_id -> tool_call_index to track which tool call added each song
    tool_execution_summary = []
    tools_used_history = []
    tool_call_counter = 0  # Track each tool call separately
    
    max_iterations = 5  # Prevent infinite loops
    target_song_count = 100
    
    for iteration in range(max_iterations):
        current_song_count = len(all_songs)
        
        log_messages.append(f"\n{'='*60}")
        log_messages.append(f"ITERATION {iteration + 1}/{max_iterations}")
        log_messages.append(f"Current progress: {current_song_count}/{target_song_count} songs")
        log_messages.append(f"{'='*60}")
        
        # Check if we have enough songs
        if current_song_count >= target_song_count:
            log_messages.append(f"✅ Target reached! Stopping iteration.")
            break
        
        # Build context for AI about current state
        if iteration == 0:
            ai_context = f"""Build a {target_song_count}-song playlist for: "{original_user_input}"

=== STEP 1: ANALYZE INTENT ===
First, understand what the user wants:
- Specific song + artist? → Use exact API lookup (song_similarity)
- Similar to an artist? → Use exact API lookup (artist_similarity)
- Genre/mood/tempo/energy? → Use exact DB search (search_database)
- Everything else? → Use AI knowledge (ai_brainstorm)

=== YOUR 4 TOOLS ===
1. song_similarity(song_title, artist, get_songs) - Exact API: find similar songs (NEEDS both title+artist)
2. artist_similarity(artist, get_songs) - Exact API: find songs from SIMILAR artists (NOT artist's own songs)
3. search_database(genres, moods, tempo_min, tempo_max, energy_min, energy_max, key, get_songs) - Exact DB: filter by attributes (COMBINE all filters in ONE call)
4. ai_brainstorm(user_request, get_songs) - AI knowledge: for ANYTHING else (artist's own songs, trending, era, complex requests)

=== DECISION RULES ===
"similar to [TITLE] by [ARTIST]" → song_similarity (exact API)
"songs like [ARTIST]" → artist_similarity (exact API)
"[GENRE]/[MOOD]/[TEMPO]/[ENERGY]" → search_database (exact DB search)
"[ARTIST] songs/hits", "trending", "era", etc. → ai_brainstorm (AI knowledge)

=== EXAMPLES ===
"Similar to Smells Like Teen Spirit by Nirvana" → song_similarity(song_title="Smells Like Teen Spirit", song_artist="Nirvana", get_songs=100)
"songs like AC/DC" → artist_similarity(artist="AC/DC", get_songs=100)
"AC/DC songs" → ai_brainstorm(user_request="AC/DC songs", get_songs=100)
"energetic rock music" → search_database(genres=["rock"], energy_min=0.08, moods=["happy"], get_songs=100)
"running 120 bpm" → search_database(tempo_min=115, tempo_max=125, energy_min=0.08, get_songs=100)
"post lunch" → search_database(moods=["relaxed"], energy_min=0.03, energy_max=0.08, tempo_min=80, tempo_max=110, get_songs=100)
"trending 2025" → ai_brainstorm(user_request="trending 2025", get_songs=100)
"greatest hits Red Hot Chili Peppers" → ai_brainstorm(user_request="greatest hits RHCP", get_songs=100)
"Metal like AC/DC + Metallica" → artist_similarity("AC/DC", 50) + artist_similarity("Metallica", 50)

VALID DB VALUES:
GENRES: rock, pop, metal, jazz, electronic, dance, alternative, indie, punk, blues, hard rock, heavy metal, Hip-Hop, funk, country, soul, 00s, 90s, 80s, 70s, 60s
MOODS: danceable, aggressive, happy, party, relaxed, sad
TEMPO: 40-200 BPM | ENERGY: 0.01-0.15

Now analyze the request and call tools:"""
        else:
            songs_needed = target_song_count - current_song_count
            previous_tools_str = ", ".join([f"{t['name']}({t.get('songs', 0)} songs)" for t in tools_used_history])
            
            ai_context = f"""User request: {original_user_input}
Goal: {target_song_count} songs total
Current: {current_song_count} songs
Needed: {songs_needed} MORE songs

Previous tools: {previous_tools_str}

Call 1-3 DIFFERENT tools or parameters to get {songs_needed} more diverse songs."""
        
        # AI decides which tools to call
        log_messages.append(f"\n--- AI Decision (Iteration {iteration + 1}) ---")
        tool_calling_result = call_ai_with_mcp_tools(
            provider=ai_provider,
            user_message=ai_context,
            tools=mcp_tools,
            ai_config=ai_config,
            log_messages=log_messages
        )
        
        if 'error' in tool_calling_result:
            error_msg = tool_calling_result['error']
            log_messages.append(f"❌ AI tool calling failed: {error_msg}")
            
            # Fallback based on iteration
            if iteration == 0:
                log_messages.append("\n🔄 Fallback: Trying genre search...")
                fallback_result = execute_mcp_tool('search_database', {'genres': ['pop', 'rock'], 'get_songs': 100}, ai_config)
                if 'songs' in fallback_result:
                    songs = fallback_result['songs']
                    for song in songs:
                        if song['item_id'] not in song_ids_seen:
                            all_songs.append(song)
                            song_ids_seen.add(song['item_id'])
                    tools_used_history.append({'name': 'search_database', 'songs': len(songs)})
                    log_messages.append(f"   Fallback added {len(songs)} songs")
            else:
                log_messages.append("   Stopping iteration due to AI error")
                break
            continue
        
        # Execute the tools AI selected
        tool_calls = tool_calling_result.get('tool_calls', [])
        
        if not tool_calls:
            log_messages.append("⚠️ AI returned no tool calls. Stopping iteration.")
            break
        
        log_messages.append(f"\n--- Executing {len(tool_calls)} Tool(s) ---")
        
        iteration_songs_added = 0
        
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('arguments', {})
            
            # Convert tool_args to dict if it's a protobuf object (for Gemini)
            # Need to recursively convert nested protobuf objects
            def convert_to_dict(obj):
                if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                    if hasattr(obj, 'items'):  # dict-like
                        return {k: convert_to_dict(v) for k, v in obj.items()}
                    else:  # list-like
                        return [convert_to_dict(item) for item in obj]
                return obj
            
            tool_args = convert_to_dict(tool_args)
            
            log_messages.append(f"\n🔧 Tool {i+1}/{len(tool_calls)}: {tool_name}")
            try:
                log_messages.append(f"   Arguments: {json.dumps(tool_args, indent=6)}")
            except TypeError:
                # If still not serializable, convert to string representation
                log_messages.append(f"   Arguments: {str(tool_args)}")
            
            # Execute the tool
            tool_result = execute_mcp_tool(tool_name, tool_args, ai_config)
            
            if 'error' in tool_result:
                log_messages.append(f"   ❌ Error: {tool_result['error']}")
                tools_used_history.append({'name': tool_name, 'args': tool_args, 'songs': 0, 'error': True, 'call_index': tool_call_counter})
                tool_call_counter += 1
                continue
            
            # Extract songs from result
            songs = tool_result.get('songs', [])
            log_messages.append(f"   ✅ Retrieved {len(songs)} songs from database")
            
            if tool_result.get('message'):
                for line in tool_result['message'].split('\n'):
                    if line.strip():
                        log_messages.append(f"   {line}")
            
            # Add to collection (deduplicate)
            new_songs = 0
            new_song_list = []
            for song in songs:
                if song['item_id'] not in song_ids_seen:
                    all_songs.append(song)
                    song_ids_seen.add(song['item_id'])
                    song_sources[song['item_id']] = tool_call_counter  # Track which tool CALL added this song
                    new_songs += 1
                    new_song_list.append(song)
            
            iteration_songs_added += new_songs
            log_messages.append(f"   📊 Added {new_songs} NEW unique songs")
            
            # Show first 5 songs added (reduced from 10)
            if new_song_list:
                preview_count = min(5, len(new_song_list))
                log_messages.append(f"   🎵 Sample songs: {preview_count}/{new_songs}")
                for j, song in enumerate(new_song_list[:preview_count]):
                    title = song.get('title', 'Unknown')
                    artist = song.get('artist', 'Unknown')
                    log_messages.append(f"      {j+1}. {title} - {artist}")
            
            # Track for summary (include arguments for visibility)
            tools_used_history.append({'name': tool_name, 'args': tool_args, 'songs': new_songs, 'call_index': tool_call_counter})
            tool_call_counter += 1
            
            # Format args for summary - show key parameters only
            args_summary = []
            if tool_name == "search_database":
                if 'genres' in tool_args and tool_args['genres']:
                    args_summary.append(f"genres={tool_args['genres']}")
                if 'moods' in tool_args and tool_args['moods']:
                    args_summary.append(f"moods={tool_args['moods']}")
                if 'tempo_min' in tool_args or 'tempo_max' in tool_args:
                    tempo_str = f"{tool_args.get('tempo_min', '')}..{tool_args.get('tempo_max', '')}"
                    args_summary.append(f"tempo={tempo_str}")
                if 'energy_min' in tool_args or 'energy_max' in tool_args:
                    energy_str = f"{tool_args.get('energy_min', '')}..{tool_args.get('energy_max', '')}"
                    args_summary.append(f"energy={energy_str}")
                if 'valence_min' in tool_args or 'valence_max' in tool_args:
                    valence_str = f"{tool_args.get('valence_min', '')}..{tool_args.get('valence_max', '')}"
                    args_summary.append(f"valence={valence_str}")
                if 'key' in tool_args:
                    args_summary.append(f"key={tool_args['key']}")
            elif tool_name in ["artist_similarity", "artist_hits"]:
                if 'artist' in tool_args or 'artist_name' in tool_args:
                    artist = tool_args.get('artist') or tool_args.get('artist_name')
                    args_summary.append(f"artist='{artist}'")
                if 'count' in tool_args:
                    args_summary.append(f"count={tool_args['count']}")
            elif tool_name == "song_similarity":
                if 'song_title' in tool_args:
                    args_summary.append(f"song='{tool_args['song_title']}'")
                if 'song_artist' in tool_args:
                    args_summary.append(f"artist='{tool_args['song_artist']}'")
            elif tool_name == "search_by_tempo_energy":
                if 'tempo_min' in tool_args or 'tempo_max' in tool_args:
                    tempo_str = f"{tool_args.get('tempo_min', '')}..{tool_args.get('tempo_max', '')}"
                    args_summary.append(f"tempo={tempo_str}")
                if 'energy_min' in tool_args or 'energy_max' in tool_args:
                    energy_str = f"{tool_args.get('energy_min', '')}..{tool_args.get('energy_max', '')}"
                    args_summary.append(f"energy={energy_str}")
            elif tool_name == "vibe_match":
                if 'vibe_description' in tool_args:
                    vibe = tool_args['vibe_description'][:30]
                    args_summary.append(f"vibe='{vibe}...'")
            elif tool_name == "ai_brainstorm":
                if 'user_request' in tool_args:
                    req = tool_args['user_request'][:35]
                    args_summary.append(f"req='{req}...'")
            elif tool_name == "popular_songs":
                if 'description' in tool_args:
                    desc = tool_args['description'][:30]
                    args_summary.append(f"desc='{desc}...'")
            
            args_str = ", ".join(args_summary) if args_summary else ""
            tool_summary = f"{tool_name}({args_str}, +{new_songs})" if args_str else f"{tool_name}(+{new_songs})"
            tool_execution_summary.append(tool_summary)
        
        log_messages.append(f"\n📈 Iteration {iteration + 1} Summary:")
        log_messages.append(f"   Songs added this iteration: {iteration_songs_added}")
        log_messages.append(f"   Total songs now: {len(all_songs)}/{target_song_count}")
        
        # If no new songs were added, stop iteration
        if iteration_songs_added == 0:
            log_messages.append("\n⚠️ No new songs added this iteration. Stopping.")
            break
    
    # Prepare final results
    if all_songs:
        # Proportional sampling to ensure representation from all tools
        if len(all_songs) <= target_song_count:
            # We have fewer songs than target, use all
            final_query_results_list = all_songs
        else:
            # We have more songs than target - sample proportionally from each tool CALL
            # Group songs by their source tool call (not just tool name!)
            songs_by_call = {}
            for song in all_songs:
                call_index = song_sources.get(song['item_id'], -1)
                if call_index not in songs_by_call:
                    songs_by_call[call_index] = []
                songs_by_call[call_index].append(song)
            
            # Calculate proportional allocation
            total_collected = len(all_songs)
            final_query_results_list = []
            
            for call_index, tool_songs in songs_by_call.items():
                # Proportional share: (tool_songs / total_collected) * target
                proportion = len(tool_songs) / total_collected
                allocated = int(proportion * target_song_count)
                
                # Ensure each tool call gets at least 1 song if it contributed any
                if allocated == 0 and len(tool_songs) > 0:
                    allocated = 1
                
                # Take allocated songs from this tool call
                selected = tool_songs[:allocated]
                final_query_results_list.extend(selected)
            
            # If we didn't reach target due to rounding, add remaining songs
            if len(final_query_results_list) < target_song_count:
                remaining_needed = target_song_count - len(final_query_results_list)
                remaining_songs = [s for s in all_songs if s not in final_query_results_list]
                final_query_results_list.extend(remaining_songs[:remaining_needed])
            
            # Truncate if we somehow went over (shouldn't happen)
            final_query_results_list = final_query_results_list[:target_song_count]
        
        final_executed_query_str = f"MCP Agentic ({len(tools_used_history)} tools, {iteration + 1} iterations): {' → '.join(tool_execution_summary)}"
        
        log_messages.append(f"\n✅ SUCCESS! Generated playlist with {len(final_query_results_list)} songs")
        log_messages.append(f"   Total songs collected: {len(all_songs)}")
        if len(all_songs) > target_song_count:
            log_messages.append(f"   ⚖️ Proportionally sampled {len(all_songs) - target_song_count} excess songs to meet target of {target_song_count}")
        log_messages.append(f"   Iterations used: {iteration + 1}/{max_iterations}")
        log_messages.append(f"   Tools called: {len(tools_used_history)}")
        
        # Show tool contribution breakdown (collected vs final)
        log_messages.append(f"\n📊 Tool Contribution (Collected → Final Playlist):")
        
        # Count songs in final playlist by tool call
        final_by_call = {}
        for song in final_query_results_list:
            call_index = song_sources.get(song['item_id'], -1)
            final_by_call[call_index] = final_by_call.get(call_index, 0) + 1
        
        for tool_info in tools_used_history:
            tool_name = tool_info['name']
            song_count = tool_info.get('songs', 0)
            args = tool_info.get('args', {})
            args_preview = []
            if 'artist' in args:
                args_preview.append(f"artist='{args['artist']}'")
            elif 'artist_name' in args:
                args_preview.append(f"artist='{args['artist_name']}'")
            if 'song_title' in args:
                args_preview.append(f"title='{args['song_title']}'")
            if 'genres' in args and args['genres']:
                args_preview.append(f"genres={args['genres'][:2]}")
            if 'moods' in args and args['moods']:
                args_preview.append(f"moods={args['moods'][:2]}")
            if 'user_request' in args:
                args_preview.append(f"request='{args['user_request'][:30]}...'")
            
            args_str = ", ".join(args_preview) if args_preview else "no filters"
            call_index = tool_info.get('call_index', -1)
            final_count = final_by_call.get(call_index, 0)
            if song_count != final_count:
                log_messages.append(f"   • {tool_name}({args_str}): {song_count} collected → {final_count} in final playlist")
            else:
                log_messages.append(f"   • {tool_name}({args_str}): {song_count} songs")
    else:
        log_messages.append("\n⚠️ No songs collected from agentic workflow")
        final_query_results_list = None
        final_executed_query_str = "MCP Agentic: No results"
    
    actual_model_used = ai_config.get(f'{ai_provider.lower()}_model')
    
    # Return final response
    return jsonify({
        "response": {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": actual_model_used,
            "executed_query": final_executed_query_str,
            "query_results": final_query_results_list
        }
    }), 200
    
    # Check if AI provider is NONE
    if ai_provider == "NONE":
        return jsonify({
            "response": {
                "message": "No AI provider selected. Please configure an AI provider to use this feature.",
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": None,
                "executed_query": None,
                "query_results": None
            }
        }), 200
    
    # Build AI configuration object
    ai_config = {
        'provider': ai_provider,
        'ollama_url': data.get('ollama_server_url', OLLAMA_SERVER_URL),
        'ollama_model': ai_model_from_request or OLLAMA_MODEL_NAME,
        'openai_url': data.get('openai_server_url', OPENAI_SERVER_URL),
        'openai_model': ai_model_from_request or OPENAI_MODEL_NAME,
        'openai_key': data.get('openai_api_key') or OPENAI_API_KEY,
        'gemini_key': data.get('gemini_api_key') or GEMINI_API_KEY,
        'gemini_model': ai_model_from_request or GEMINI_MODEL_NAME,
        'mistral_key': data.get('mistral_api_key') or MISTRAL_API_KEY,
        'mistral_model': ai_model_from_request or MISTRAL_MODEL_NAME
    }
    
    # Validate API keys for cloud providers
    if ai_provider == "OPENAI" and not ai_config['openai_key']:
        error_msg = "Error: OpenAI API key is missing. Please provide a valid API key or set it in the server configuration."
        log_messages.append(error_msg)
        return jsonify({"response": {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": ai_config.get('openai_model'),
            "executed_query": None,
            "query_results": None
        }}), 400
    
    if ai_provider == "GEMINI" and (not ai_config['gemini_key'] or ai_config['gemini_key'] == "YOUR-GEMINI-API-KEY-HERE"):
        error_msg = "Error: Gemini API key is missing. Please provide a valid API key or set it in the server configuration."
        log_messages.append(error_msg)
        return jsonify({"response": {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": ai_config.get('gemini_model'),
            "executed_query": None,
            "query_results": None
        }}), 400
    
    if ai_provider == "MISTRAL" and (not ai_config['mistral_key'] or ai_config['mistral_key'] == "YOUR-MISTRAL-API-KEY-HERE"):
        error_msg = "Error: Mistral API key is missing. Please provide a valid API key or set it in the server configuration."
        log_messages.append(error_msg)
        return jsonify({"response": {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": ai_config.get('mistral_model'),
            "executed_query": None,
            "query_results": None
        }}), 400


@chat_bp.route('/api/create_playlist', methods=['POST'])
@swag_from({
    'tags': ['Chat Interaction'],
    'summary': 'Create a playlist on the media server from a list of song item IDs.',
    'requestBody': {
        'description': 'Playlist name and song item IDs.',
        'required': True,
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'required': ['playlist_name', 'item_ids'],
                    'properties': {
                        'playlist_name': {
                            'type': 'string',
                            'description': 'The desired name for the playlist.',
                            'example': 'My Awesome Mix'
                        },
                        'item_ids': {
                            'type': 'array',
                            'description': 'A list of item IDs for the songs to include.',
                            'items': {'type': 'string'},
                            'example': ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"]
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': 'Playlist successfully created.',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'message': {'type': 'string'}
                        }
                    }
                }
            }
        },
        '400': {
            'description': 'Bad Request - Missing parameters or invalid input.'
        },
        '500': {
            'description': 'Server Error - Failed to create playlist.',
             'content': { # Added content for 400 and 500 for consistency
                'application/json': {
                    'schema': {'type': 'object', 'properties': {'message': {'type': 'string'}}}
                }
            }
        }
    }
})
def create_media_server_playlist_api():
    """
    API endpoint to create a playlist on the configured media server.
    """
    # Local import to break circular dependency at startup
    from tasks.mediaserver import create_instant_playlist

    data = request.get_json()
    if not data or 'playlist_name' not in data or 'item_ids' not in data:
        return jsonify({"message": "Error: Missing playlist_name or item_ids in request"}), 400

    user_playlist_name = data.get('playlist_name')
    item_ids = data.get('item_ids') # This will be a list of strings

    if not user_playlist_name.strip():
        return jsonify({"message": "Error: Playlist name cannot be empty."}), 400
    if not item_ids:
        return jsonify({"message": "Error: No songs provided to create the playlist."}), 400

    try:
        # MODIFIED: Call the simplified create_instant_playlist function
        created_playlist_info = create_instant_playlist(user_playlist_name, item_ids)
        
        if not created_playlist_info:
            raise Exception("Media server did not return playlist information after creation.")
            
        # The created_playlist_info is the full JSON response from the media server
        return jsonify({"message": f"Successfully created playlist '{user_playlist_name}' on the media server with ID: {created_playlist_info.get('Id')}"}), 200

    except Exception as e:
        # Log detailed error on the server
        error_details_for_server = f"Media Server API Request Exception: {str(e)}\n"
        if hasattr(e, 'response') and e.response is not None: # type: ignore
            try: error_details_for_server += f" - Media Server Response: {e.response.text}\n"
            except: pass # nosec
        logger.error("Error in create_media_server_playlist_api: %s", error_details_for_server, exc_info=True)
        # Return generic error to client
        return jsonify({"message": "An internal error occurred while creating the playlist."}), 500
