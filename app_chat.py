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
    GEMINI_MODEL_NAME, GEMINI_API_KEY, # Import GEMINI_API_KEY from config
    MISTRAL_MODEL_NAME, MISTRAL_API_KEY,
    AI_MODEL_PROVIDER, # Default AI provider
    AI_CHAT_DB_USER_NAME, AI_CHAT_DB_USER_PASSWORD, # Import new config
    OPENAI_API_KEY, OPENAI_MODEL_NAME, OPENAI_BASE_URL, OPENAI_API_TOKENS # Import OpenAI config
)
from ai import get_gemini_playlist_name, get_ollama_playlist_name, get_mistral_playlist_name, get_openai_playlist_name # Import functions to call AI

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
        return None, f"SQLglot parsing error: {str(e)}"
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
    return render_template('chat.html')

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
                            'default_gemini_model_name': {
                                'type': 'string', 'example': 'gemini-2.5-pro'
                            },
                            'default_mistral_model_name': {
                                'type': 'string', 'example': 'ministral-3b-latest'
                            },
                            'default_openai_model_name': {
                                'type': 'string', 'example': 'gpt-4o-mini'
                            },
                            'default_openai_base_url': {
                                'type': 'string', 'example': 'https://api.openai.com/v1/chat/completions'
                            },
                            'default_openai_api_tokens': {
                                'type': 'integer', 'example': 1000
                            }
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
        "default_gemini_model_name": GEMINI_MODEL_NAME,
        "default_mistral_model_name": MISTRAL_MODEL_NAME,
        "default_openai_model_name": OPENAI_MODEL_NAME,
        "default_openai_base_url": OPENAI_BASE_URL,
        "default_openai_api_tokens": OPENAI_API_TOKENS
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
                            'description': 'The AI provider to use (OLLAMA, GEMINI, MISTRAL, OPENAI, NONE). Defaults to server config.',
                            'example': 'GEMINI',
                            'enum': ['OLLAMA', 'GEMINI', "MISTRAL", "OPENAI", 'NONE']
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
                        'gemini_api_key': {
                            'type': 'string',
                            'description': 'Custom Gemini API key (optional, defaults to server configuration).',
                            'example': 'YOUR-GEMINI-API-KEY-HERE'
                        },
                        'mistral_api_key': {
                            'type': 'string',
                            'description': 'Custom Mistral API key (optional, defaults to server configuration).',
                            'example': 'YOUR-MISTRAL-API-KEY-HERE'
                        },
                        'openai_api_key': {
                            'type': 'string',
                            'description': 'Custom OpenAI API key (optional, defaults to server configuration).',
                            'example': 'sk-your-own-key'
                        },
                        'openai_base_url': {
                            'type': 'string',
                            'description': 'Custom OpenAI Base URL (optional, defaults to server configuration).',
                            'example': 'https://api.openai.com/v1/chat/completions'
                        },
                        'openai_api_tokens': {
                            'type': 'integer',
                            'description': 'Maximum tokens for OpenAI API responses (optional).',
                            'example': 1000
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
    Process user chat input to generate a playlist idea using AI.
    This is a synchronous endpoint.
    """
    data = request.get_json()
    # Mask API key if present in the debug log
    data_for_log = dict(data) if data else {}
    if 'gemini_api_key' in data_for_log and data_for_log['gemini_api_key']:
        data_for_log['gemini_api_key'] = 'API-KEY' # Masked
    if 'mistral_api_key' in data_for_log and data_for_log['mistral_api_key']:
        data_for_log['mistral_api_key'] = 'API-KEY' # Masked
    if 'openai_api_key' in data_for_log and data_for_log['openai_api_key']:
        data_for_log['openai_api_key'] = 'API-KEY' # Masked
    logger.debug("chat_playlist_api called. Raw request data: %s", data_for_log)
    from app_helper import get_db # Import get_db here, inside the function
    if not data or 'userInput' not in data:
        return jsonify({"error": "Missing userInput in request"}), 400

    original_user_input = data.get('userInput')
    # Use AI provider from request, or fallback to global config, then to "NONE"
    ai_provider = data.get('ai_provider', AI_MODEL_PROVIDER).upper() # Use the imported constant
    ai_model_from_request = data.get('ai_model') # Model selected by user on chat page

    ai_response_message = f"Received your request: '{original_user_input}'.\n"
    actual_model_used = None

    # Variables to hold the final state after potential retries
    final_query_results_list = None
    final_executed_query_str = None # The SQL string that was last attempted or successfully executed
    
    # Variables for retry logic
    last_raw_sql_from_ai = None
    last_error_for_retry = None

    # Define the prompt structure once, to be used by any provider that needs it.
    # The [USER INPUT] placeholder will be replaced dynamically.
    base_expert_playlist_creator_prompt = """
    You are a specialized AI with expert-level knowledge of music trends (Spotify charts, YouTube trending songs, MTV hits, radio top charts, film soundtracks, etc.) and proficiency in PostgreSQL.

    **Your Mission:**
    Convert the user's natural language playlist request into a precise, accurate, and optimized PostgreSQL SQL query for the `public.score` table.

    ### Step-by-Step Instructions:

    1. **Interpret the Request Thoughtfully:**

    * Clearly identify the user's intent: Are they asking for trending/top/popular hits, mood-based playlists, or specific themes?
    * Recall specific current hit song titles and artists matching the user's description.

    2. **SQL Query Requirements:**

    * Return ONLY the raw SQL query—no markdown, no comments, no explanations.
    * Always use:

        ```sql
        SELECT item_id, title, author
        FROM public.score
        ```
    * For general requests, conclude your query with:

        ```sql
        ORDER BY random()
        LIMIT 100
        ```
    * If the user explicitly requests ordered top/best/famous results, sort appropriately without randomization.

    3. **Critical Formatting Rules:**

    * To include single quotes (`'`) inside strings, use two single quotes (`''`). Example:

        ```sql
        'Don''t Stop Believin'''
        ```
    * When matching songs, always use precise matching criteria:

        ```sql
        WHERE (title = 'Exact Song Title' AND author ILIKE '%Artist Name%')
        ```

    4. **Mood and Feature Filtering:**

    * For `mood_vector` or `other_features` columns:

        * Extract numeric values using regex and CAST to float:

        ```sql
        CAST(regexp_replace(substring(mood_vector FROM 'rock:([0-9]*\.?[0-9]+)'), 'rock:', '') AS float) >= threshold
        ```
    * Recommended thresholds when asked for MEDIUM or HIGH:

        * `mood_vector`: ≥ 0.2 and < 1
        * `other_features`: ≥ 0.5 and < 1
        * `energy`: between ≥ 0.08 and 0.15
        * `tempo`: between ≥ 110 and 200

    5. **Database Structure Reference:**

    ```sql
    public.score
    (
        item_id,
        title,
        author,
        tempo numeric (40-200),
        key text,
        scale text,
        mood_vector text (e.g. 'pop:0.8,rock:0.3'),
        other_features text (e.g. 'danceable:0.7,party:0.6'),
        energy numeric (0-0.15)
    )
    ```

    6. **Available Labels:**

    * **MOOD\_LABELS:**

        ```
        rock, pop, alternative, indie, electronic, female vocalists, dance, 00s, alternative rock, jazz, beautiful, metal, chillout, male vocalists, classic rock, soul, indie rock, electronica, 80s, folk, 90s, chill, instrumental, punk, oldies, blues, hard rock, ambient, acoustic, experimental, female vocalist, guitar, Hip-Hop, 70s, party, country, funk, electro, heavy metal, Progressive rock, 60s, rnb, indie pop, sad, House, happy
        ```
    * **OTHER\_FEATURE\_LABELS:**

        ```
        danceable, aggressive, happy, party, relaxed, sad
        ```

    7. **PostgreSQL Syntax:**

    * When using UNION ALL, encapsulate it correctly:

        ```sql
        SELECT item_id, title, author
        FROM (
        SELECT item_id, title, author FROM public.score WHERE condition1
        UNION ALL
        SELECT item_id, title, author FROM public.score WHERE condition2
        ) AS combined_results
        ORDER BY random()
        LIMIT 100
        ```
    * Avoid aliasing individual SELECT statements within UNION ALL.

    **Your Task:**
    Generate a precise and contextually optimized SQL query for the user's request:
    "{user_input_placeholder}"
    """

    max_retries = 2 # Max 2 retries, so 3 attempts total
    for attempt_num in range(max_retries + 1):
        ai_response_message += f"\n--- Attempt {attempt_num + 1} of {max_retries + 1} ---\n"
        
        current_prompt_for_ai = ""
        retry_reason_for_prompt = last_error_for_retry # Capture before it's potentially overwritten

        if attempt_num > 0: # This is a retry
            ai_response_message += f"Retrying due to previous issue: {retry_reason_for_prompt}\n"
            if "no results" in str(retry_reason_for_prompt).lower():
                retry_prompt_text = f"""The user's original request was: '{original_user_input}'
Your previous SQL query attempt was:
```sql
{last_raw_sql_from_ai}
```
This query was valid but returned no songs.
Please make the query less stringent to find some matching songs. For example, you could broaden search terms, adjust thresholds, or simplify conditions.
Regenerate a SQL query based on the original instructions and user request, but aim for wider results.
Ensure all SQL rules are followed.
Return ONLY the new SQL query.
---
Original full prompt context (for reference):
{base_expert_playlist_creator_prompt.replace("{user_input_placeholder}", original_user_input)}
"""
            else: # SQL or DB Error
                retry_prompt_text = f"""The user's original request was: '{original_user_input}'
Your previous SQL query attempt was:
```sql
{last_raw_sql_from_ai}
```
This query resulted in the following error: '{retry_reason_for_prompt}'
Please carefully review the error and your previous SQL.
Then, regenerate a corrected SQL query based on the original instructions and user request.
Ensure all SQL rules are followed, especially for string escaping (e.g., 'Player''s Choice') and query structure.
Return ONLY the corrected SQL query.
---
Original full prompt context (for reference):
{base_expert_playlist_creator_prompt.replace("{user_input_placeholder}", original_user_input)}
"""
            current_prompt_for_ai = retry_prompt_text
        else: # First attempt
            current_prompt_for_ai = base_expert_playlist_creator_prompt.replace("{user_input_placeholder}", original_user_input)

        raw_sql_from_ai_this_attempt = None
        # --- Call AI (Ollama/Gemini/Mistral) ---
        if ai_provider == "OLLAMA":
            actual_model_used = ai_model_from_request or OLLAMA_MODEL_NAME
            ollama_url_from_request = data.get('ollama_server_url', OLLAMA_SERVER_URL)
            ai_response_message += f"Processing with OLLAMA model: {actual_model_used} (at {ollama_url_from_request}).\n"
            raw_sql_from_ai_this_attempt = get_ollama_playlist_name(ollama_url_from_request, actual_model_used, current_prompt_for_ai)
            if raw_sql_from_ai_this_attempt.startswith("Error:") or raw_sql_from_ai_this_attempt.startswith("An unexpected error occurred:"):
                ai_response_message += f"Ollama API Error: {raw_sql_from_ai_this_attempt}\n"
                last_error_for_retry = raw_sql_from_ai_this_attempt # Store error
                raw_sql_from_ai_this_attempt = None # Mark as failed AI call

        elif ai_provider == "GEMINI":
            actual_model_used = ai_model_from_request or GEMINI_MODEL_NAME
            # MODIFIED: Get API key from request, but fall back to server config if not provided.
            gemini_api_key_from_request = data.get('gemini_api_key') or GEMINI_API_KEY
            if not gemini_api_key_from_request or gemini_api_key_from_request == "YOUR-GEMINI-API-KEY-HERE":
                error_msg = "Error: Gemini API key is missing. Please provide a valid API key or set it in the server configuration."
                ai_response_message += error_msg + "\n"
                if attempt_num == 0:
                    return jsonify({"response": {"message": ai_response_message, "original_request": original_user_input, "ai_provider_used": ai_provider, "ai_model_selected": actual_model_used, "executed_query": None, "query_results": None}}), 400
                last_error_for_retry = error_msg
                break
            ai_response_message += f"Processing with GEMINI model: {actual_model_used}.\n"
            raw_sql_from_ai_this_attempt = get_gemini_playlist_name(gemini_api_key_from_request, actual_model_used, current_prompt_for_ai)
            if raw_sql_from_ai_this_attempt.startswith("Error:"):
                ai_response_message += f"Gemini API Error: {raw_sql_from_ai_this_attempt}\n"
                last_error_for_retry = raw_sql_from_ai_this_attempt
                raw_sql_from_ai_this_attempt = None

        elif ai_provider == "MISTRAL":
            actual_model_used = ai_model_from_request or MISTRAL_MODEL_NAME
            # MODIFIED: Get API key from request, but fall back to server config if not provided.
            mistral_api_key_from_request = data.get('mistral_api_key') or MISTRAL_API_KEY
            if not mistral_api_key_from_request or mistral_api_key_from_request == "YOUR-MISTRAL-API-KEY-HERE":
                error_msg = "Error: Mistral API key is missing. Please provide a valid API key or set it in the server configuration."
                ai_response_message += error_msg + "\n"
                if attempt_num == 0:
                    return jsonify({"response": {"message": ai_response_message, "original_request": original_user_input, "ai_provider_used": ai_provider, "ai_model_selected": actual_model_used, "executed_query": None, "query_results": None}}), 400
                last_error_for_retry = error_msg
                break
            ai_response_message += f"Processing with MISTRAL model: {actual_model_used}.\n"
            raw_sql_from_ai_this_attempt = get_mistral_playlist_name(mistral_api_key_from_request, actual_model_used, current_prompt_for_ai)
            if raw_sql_from_ai_this_attempt.startswith("Error:"):
                ai_response_message += f"Mistral API Error: {raw_sql_from_ai_this_attempt}\n"
                last_error_for_retry = raw_sql_from_ai_this_attempt
                raw_sql_from_ai_this_attempt = None
        
        elif ai_provider == "OPENAI":
            actual_model_used = ai_model_from_request or OPENAI_MODEL_NAME
            openai_api_key_from_request = data.get('openai_api_key') or OPENAI_API_KEY
            openai_base_url_from_request = data.get('openai_base_url') or OPENAI_BASE_URL
            openai_api_tokens_from_request = data.get('openai_api_tokens') or OPENAI_API_TOKENS
            ai_response_message += f"Processing with OPENAI model: {actual_model_used}.\n"
            raw_sql_from_ai_this_attempt = get_openai_playlist_name(current_prompt_for_ai, actual_model_used, openai_api_key_from_request, openai_base_url_from_request, openai_api_tokens_from_request)
            if raw_sql_from_ai_this_attempt.startswith("Error:"):
                ai_response_message += f"OpenAI API Error: {raw_sql_from_ai_this_attempt}\n"
                last_error_for_retry = raw_sql_from_ai_this_attempt
                raw_sql_from_ai_this_attempt = None

        elif ai_provider == "NONE":
            ai_response_message += "No AI provider selected. Input acknowledged."
            break 
        else:
            ai_response_message += f"AI Provider '{ai_provider}' is not recognized."
            break 

        last_raw_sql_from_ai = raw_sql_from_ai_this_attempt # Store for potential next retry

        if not raw_sql_from_ai_this_attempt: # If AI call failed
            if attempt_num >= max_retries: break 
            continue # Try next attempt if AI call itself failed and retries are left

        ai_response_message += f"AI raw response (SQL query attempt):\n{raw_sql_from_ai_this_attempt}\n"
        
        cleaned_sql_this_attempt, validation_err_msg = clean_and_validate_sql(raw_sql_from_ai_this_attempt)
        final_executed_query_str = cleaned_sql_this_attempt # Store the latest cleaned query for display

        if validation_err_msg:
            last_error_for_retry = validation_err_msg
            ai_response_message += f"SQL Validation Error: {validation_err_msg}\n"
            if attempt_num >= max_retries: break 
            continue 
        
        ai_response_message += "SQL query validated successfully. Attempting execution...\n"

        # Ensure AI user is configured (only if an AI provider was used)
        if ai_provider != "NONE":
            try:
                _ensure_ai_user_configured(get_db())
                if not ai_user_setup_done: 
                    raise Exception("AI user setup flag not set after configuration attempt.")
            except Exception as setup_err:
                # Log detailed error on the server
                logger.error("Error during AI user setup in chat_playlist_api", exc_info=True)
                ai_response_message += "Critical Error: Could not ensure AI user setup. Query will not be executed.\n" # Generic message for client
                last_error_for_retry = f"AI User setup failed: {setup_err}"
                break 

        # --- Execute Query ---
        if cleaned_sql_this_attempt and ai_user_setup_done :
            try:
                with get_db().cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(f"SET LOCAL ROLE {AI_CHAT_DB_USER_NAME};")
                    cur.execute(cleaned_sql_this_attempt)
                    results = cur.fetchall()
                    get_db().commit()
                
                if results:
                    final_query_results_list = [] 
                    for row in results:
                        final_query_results_list.append({
                            "item_id": row.get("item_id"), "title": row.get("title"), "artist": row.get("author")
                        })
                    ai_response_message += f"Successfully executed query. Found {len(final_query_results_list)} songs.\n"
                    final_executed_query_str = cleaned_sql_this_attempt # Store successful query
                    break # Success, exit retry loop
                else:
                    ai_response_message += "Query executed successfully, but found no matching songs.\n"
                    last_error_for_retry = "Query returned no results."
                    # final_executed_query_str is already set to cleaned_sql_this_attempt
                    if attempt_num >= max_retries: break
                    continue # Go to next retry for "no results"

            except Exception as db_exec_error:
                get_db().rollback()
                db_error_str = f"Database Error executing query: {str(db_exec_error)}"
                # Log detailed error on the server
                logger.error("Error executing AI generated query in chat_playlist_api: %s", db_error_str, exc_info=True)
                ai_response_message += "Database Error executing query. Please check server logs.\n" # Generic message for client
                last_error_for_retry = db_error_str
                if attempt_num >= max_retries: break
                continue
        elif cleaned_sql_this_attempt and not ai_user_setup_done:
             ai_response_message += "AI User setup was not completed successfully. Query was not executed for security reasons.\n"
             last_error_for_retry = "AI User setup failed prior to query execution."
             break # Cannot execute without user setup

    # --- After retry loop ---
    if not final_query_results_list and last_error_for_retry:
        ai_response_message += f"\nFailed to generate and execute a valid query that returns results after {attempt_num + 1} attempt(s). Last issue: {last_error_for_retry}\n"

    return jsonify({"response": {"message": ai_response_message, 
                                 "original_request": original_user_input, 
                                 "ai_provider_used": ai_provider, 
                                 "ai_model_selected": actual_model_used, 
                                 "executed_query": final_executed_query_str, # Show last attempted/successful query
                                 "query_results": final_query_results_list}}), 200

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
