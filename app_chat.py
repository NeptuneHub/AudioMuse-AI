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
    Process user chat input to generate a playlist idea using AI with multi-step approach.
    Steps:
    1. Understand user request
    2. Expand with AI knowledge (similar artists, top songs, etc.)
    3. Explore database to find what actually exists
    4. Generate optimized SQL query based on real data
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
    
    if not data or 'userInput' not in data:
        return jsonify({"error": "Missing userInput in request"}), 400

    original_user_input = data.get('userInput')
    ai_provider = data.get('ai_provider', AI_MODEL_PROVIDER).upper()
    ai_model_from_request = data.get('ai_model')
    
    log_messages = []
    log_messages.append(f"Received request: '{original_user_input}'")
    log_messages.append(f"Using AI provider: {ai_provider}")
    
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
    
    final_query_results_list = None
    final_executed_query_str = None
    actual_model_used = ai_config.get(f'{ai_provider.lower()}_model')
    
    # Main processing with up to 3 total attempts
    max_attempts = 3
    for attempt in range(max_attempts):
        log_messages.append(f"\n========== ATTEMPT {attempt + 1} OF {max_attempts} ==========")
        
        # STEP 1: AI creates execution plan
        step1_prompt = chat_step1_understand_prompt.format(user_input=original_user_input)
        plan, error = call_ai_step("STEP 1: AI Planning Execution Strategy", step1_prompt, ai_config, log_messages)
        
        if error or not plan:
            if attempt < max_attempts - 1:
                log_messages.append("Step 1 failed, retrying...")
                continue
            else:
                log_messages.append("Failed to create execution plan after all attempts.")
                break
        
        intent = plan.get('intent', original_user_input)
        execution_plan = plan.get('execution_plan', [])
        strategy = plan.get('strategy', 'unknown')
        target_count = plan.get('target_count', 100)
        
        log_messages.append(f"\n📋 AI EXECUTION PLAN:")
        log_messages.append(f"   Intent: {intent}")
        log_messages.append(f"   Strategy: {strategy}")
        log_messages.append(f"   Target songs: {target_count}")
        log_messages.append(f"   Actions: {len(execution_plan)}")
        
        if not execution_plan:
            log_messages.append("   ⚠️ No actions in execution plan, falling back to AI approach")
            # Fallback to old AI approach
            execution_plan = [{
                "action": "ai_full_process",
                "params": {}
            }]
        
        # STEP 2: Execute the plan
        all_songs = []
        song_ids_seen = set()
        actions_executed = 0
        successful_actions = []  # Track which actions succeeded for potential expansion
        executed_queries = []  # Track actual SQL queries for transparency
        
        for i, action in enumerate(execution_plan):
            action_type = action.get('action')
            
            # Handle AI brainstorming titles action (for temporal queries)
            if action_type == "ai_brainstorm_titles":
                log_messages.append(f"\n🧠 ACTION {i+1}: AI Brainstorming song titles and artists...")
                
                # Call Step 2: AI expansion
                from ai import chat_step2_expand_prompt
                step2_prompt = chat_step2_expand_prompt.format(
                    intent=intent,
                    keywords=[],
                    artist_names=[],
                    temporal_context='recent years'
                )
                
                expansion, error = call_ai_step("STEP 2: AI Music Knowledge Expansion", step2_prompt, ai_config, log_messages)
                
                if error or not expansion:
                    log_messages.append("   ❌ AI brainstorming failed, falling back to genre query...")
                    # Fallback to pop genre query
                    fallback_action = {"action": "database_genre_query", "params": {"genre": "pop"}, "get_songs": 100}
                    songs = execute_action(fallback_action, get_db(), log_messages, ai_config)
                else:
                    expanded_artists = expansion.get('expanded_artists', [])
                    expanded_song_titles = expansion.get('expanded_song_titles', [])
                    expanded_song_artist_pairs = expansion.get('expanded_song_artist_pairs', [])
                    
                    log_messages.append(f"   ✓ AI suggested {len(expanded_artists)} artists")
                    log_messages.append(f"   ✓ AI suggested {len(expanded_song_titles)} song titles")
                    if expanded_song_artist_pairs:
                        log_messages.append(f"   ✓ AI suggested {len(expanded_song_artist_pairs)} song-artist pairs")
                    
                    # Search database for these titles and artists
                    db_results = explore_database_for_matches(
                        get_db(),
                        expanded_artists,
                        [],
                        expanded_song_titles,
                        log_messages,
                        expanded_song_artist_pairs  # Pass the paired data
                    )
                    
                    found_artists = db_results['found_artists']
                    found_song_titles = db_results['found_song_titles']
                    
                    # Build query from found matches
                    if found_song_titles or found_artists:
                        with get_db().cursor(cursor_factory=DictCursor) as cur:
                            # Query using found titles and artists
                            if found_song_titles:
                                # Query specific song titles
                                title_author_tuples = ', '.join([f"('{t[0].replace(chr(39), chr(39)+chr(39))}', '{t[1].replace(chr(39), chr(39)+chr(39))}')" for t in found_song_titles[:50]])
                                sql = f"""
                                    SELECT DISTINCT item_id, title, author FROM (
                                        SELECT item_id, title, author FROM public.score
                                        WHERE (title, author) IN ({title_author_tuples})
                                        ORDER BY RANDOM()
                                    ) AS randomized LIMIT %s
                                """
                                cur.execute(sql, [action.get('get_songs', 100)])
                                results = cur.fetchall()
                                songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
                                log_messages.append(f"   ✓ Found {len(songs)} songs by matching specific titles")
                            
                            # If not enough songs, add from found artists
                            if len(songs) < 50 and found_artists:
                                log_messages.append(f"   ⚠️ Only {len(songs)} from titles, adding from artists...")
                                placeholders = ','.join(['%s'] * len(found_artists[:20]))
                                sql = f"""
                                    SELECT DISTINCT item_id, title, author FROM (
                                        SELECT item_id, title, author FROM public.score
                                        WHERE author IN ({placeholders})
                                        ORDER BY RANDOM()
                                    ) AS randomized LIMIT %s
                                """
                                cur.execute(sql, found_artists[:20] + [action.get('get_songs', 100)])
                                results = cur.fetchall()
                                artist_songs = [{"item_id": r['item_id'], "title": r['title'], "artist": r['author']} for r in results]
                                
                                # Add only new songs
                                for s in artist_songs:
                                    if s['item_id'] not in [x['item_id'] for x in songs]:
                                        songs.append(s)
                                        if len(songs) >= action.get('get_songs', 100):
                                            break
                                
                                log_messages.append(f"   ✓ Added {len(artist_songs)} songs from popular artists (total: {len(songs)})")
                    else:
                        log_messages.append("   ❌ No matching titles/artists found in database, falling back to genre...")
                        fallback_action = {"action": "database_genre_query", "params": {"genre": "pop"}, "get_songs": 100}
                        songs = execute_action(fallback_action, get_db(), log_messages, ai_config)
                
                actions_executed += 1
                if songs:
                    successful_actions.append((action, songs))
                
                # Deduplicate and add to collection
                for song in songs:
                    if song['item_id'] not in song_ids_seen:
                        all_songs.append(song)
                        song_ids_seen.add(song['item_id'])
                
                log_messages.append(f"   📊 Progress: {len(all_songs)}/{target_count} songs collected")
                # Don't break - continue executing remaining actions
                    
            # Handle other AI brainstorming actions - skip for now
            elif action_type in ["ai_brainstorm_songs", "ai_brainstorm_artists", "database_custom_query", "ai_full_process"]:
                log_messages.append(f"\n⚠️ ACTION {i+1}: {action_type} not yet implemented, skipping...")
                continue
            
            # Execute API-based actions (artist_similarity_api, song_similarity_api, database_genre_query)
            else:
                songs = execute_action(action, get_db(), log_messages, ai_config)
                actions_executed += 1
                
                # Track successful actions for potential expansion
                if songs:
                    successful_actions.append((action, songs))
                
                # Deduplicate and add to collection
                for song in songs:
                    if song['item_id'] not in song_ids_seen:
                        all_songs.append(song)
                        song_ids_seen.add(song['item_id'])
                
                log_messages.append(f"   📊 Progress: {len(all_songs)}/{target_count} songs collected")
                # Don't break - continue executing remaining actions
        
        # Log completion of all planned actions
        log_messages.append(f"\n✅ Executed {actions_executed}/{len(execution_plan)} planned actions")
        log_messages.append(f"   Collected {len(all_songs)} unique songs total")
        
        # PROGRESSIVE REFINEMENT: If we don't have enough songs, keep expanding
        expansion_round = 1
        max_expansion_rounds = 5
        
        while len(all_songs) < target_count and successful_actions and expansion_round <= max_expansion_rounds:
            songs_needed = target_count - len(all_songs)
            log_messages.append(f"\n🔄 PROGRESSIVE REFINEMENT (Round {expansion_round}): Need {songs_needed} more songs")
            log_messages.append(f"   Expanding {len(successful_actions)} successful action(s)...")
            
            # Request MORE songs each round to overcome duplicates
            # Round 1: +50, Round 2: +100, Round 3: +150, etc.
            songs_per_action = (songs_needed // len(successful_actions)) + (50 * expansion_round)
            
            songs_added_this_round = 0
            
            for orig_action, orig_songs in successful_actions:
                if len(all_songs) >= target_count:
                    break
                    
                # Re-execute with progressively higher song count
                expanded_action = orig_action.copy()
                expanded_action['get_songs'] = songs_per_action
                
                log_messages.append(f"   Expanding {orig_action.get('action')} (requesting {songs_per_action} songs)...")
                additional_songs = execute_action(expanded_action, get_db(), log_messages, ai_config)
                
                # Add new unique songs
                new_count = 0
                for song in additional_songs:
                    if song['item_id'] not in song_ids_seen:
                        all_songs.append(song)
                        song_ids_seen.add(song['item_id'])
                        new_count += 1
                        songs_added_this_round += 1
                        if len(all_songs) >= target_count:
                            break
                
                if new_count > 0:
                    log_messages.append(f"   ✓ Added {new_count} new songs (total: {len(all_songs)})")
            
            # If we didn't add any songs this round, no point continuing
            if songs_added_this_round == 0:
                log_messages.append(f"   ⚠️ No new unique songs found in round {expansion_round}, stopping expansion")
                break
                
            expansion_round += 1
        
        # CRITICAL: Always return results if we have any
        if all_songs:
            # Limit to target count
            final_query_results_list = all_songs[:target_count]
            
            # Build query description with actual actions executed
            query_parts = []
            has_api_calls = False
            has_database_queries = False
            
            for action in execution_plan[:actions_executed]:
                action_type = action.get('action')
                if action_type == 'artist_hits_query':
                    artist = action.get('params', {}).get('artist')
                    query_parts.append(f"🧠 AI: Artist Hits ({artist})")
                    has_api_calls = True  # It uses AI knowledge
                elif action_type == 'artist_similarity_api':
                    artist = action.get('params', {}).get('artist')
                    query_parts.append(f"🔍 API: Artist Similarity ({artist})")
                    has_api_calls = True
                elif action_type == 'artist_similarity_filtered':
                    artist = action.get('params', {}).get('artist')
                    filters = action.get('params', {}).get('filters', {})
                    filter_desc = []
                    if filters.get('genre'):
                        filter_desc.append(f"genre={filters['genre']}")
                    if filters.get('tempo'):
                        filter_desc.append(f"tempo={filters['tempo']}")
                    if filters.get('energy'):
                        filter_desc.append(f"energy={filters['energy']}")
                    filter_str = f" [{', '.join(filter_desc)}]" if filter_desc else ""
                    query_parts.append(f"🔍 API: Artist Similarity ({artist}){filter_str}")
                    has_api_calls = True
                elif action_type == 'song_similarity_api':
                    title = action.get('params', {}).get('song_title')
                    artist = action.get('params', {}).get('artist')
                    query_parts.append(f"🔍 API: Song Similarity ('{title}' by {artist})")
                    has_api_calls = True
                elif action_type == 'database_genre_query':
                    genre = action.get('params', {}).get('genre') or 'all'
                    tempo = action.get('params', {}).get('tempo')
                    energy = action.get('params', {}).get('energy')
                    
                    # Build query description with filters
                    filter_parts = [f"mood_vector LIKE %{genre}%"]
                    if tempo:
                        tempo_cond = {'slow': 'tempo < 90', 'medium': 'tempo BETWEEN 90 AND 140', 'fast': 'tempo > 140', 'high': 'tempo > 140'}.get(tempo, 'tempo IS NOT NULL')
                        filter_parts.append(tempo_cond)
                    if energy:
                        energy_cond = {'low': 'energy < 0.05', 'medium': 'energy BETWEEN 0.05 AND 0.10', 'high': 'energy > 0.10'}.get(energy, 'energy IS NOT NULL')
                        filter_parts.append(energy_cond)
                    
                    query_parts.append(f"💾 SQL: Genre Query ({' AND '.join(filter_parts)})")
                    has_database_queries = True
                elif action_type == 'database_tempo_energy_query':
                    tempo = action.get('params', {}).get('tempo', 'medium')
                    energy = action.get('params', {}).get('energy', 'medium')
                    tempo_cond = {'slow': 'tempo < 90', 'medium': 'tempo BETWEEN 90 AND 140', 'fast': 'tempo > 140', 'high': 'tempo > 140'}.get(tempo, 'tempo IS NOT NULL')
                    energy_cond = {'low': 'energy < 0.05', 'medium': 'energy BETWEEN 0.05 AND 0.10', 'high': 'energy > 0.10'}.get(energy, 'energy IS NOT NULL')
                    query_parts.append(f"💾 SQL: Tempo/Energy Query (WHERE {tempo_cond} AND {energy_cond})")
                    has_database_queries = True
            
            # Build final query string with execution method indicator
            execution_method = ""
            if has_api_calls and has_database_queries:
                execution_method = "[HYBRID: API + SQL] "
            elif has_api_calls:
                execution_method = "[API ONLY] "
            elif has_database_queries:
                execution_method = "[SQL ONLY] "
            
            final_executed_query_str = f"{execution_method}AI Execution Plan ({strategy}): {' + '.join(query_parts)}"
            
            log_messages.append(f"\n✅ SUCCESS! Collected {len(final_query_results_list)} songs")
            log_messages.append(f"   Strategy: {strategy}")
            log_messages.append(f"   Actions executed: {actions_executed}/{len(execution_plan)}")
            break  # Success - exit attempt loop
        else:
            log_messages.append(f"\n⚠️ No songs collected from {actions_executed} actions")
            if attempt < max_attempts - 1:
                log_messages.append("Retrying with simpler approach...")
                continue
            else:
                log_messages.append("❌ Failed to collect songs after all attempts")
                break
    
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
