# app_chat.py
from flask import Blueprint, render_template, request, jsonify
import requests # For Jellyfin API call
from psycopg2.extras import DictCursor # To get results as dictionaries
import unicodedata # For ASCII normalization
import sqlglot # Import sqlglot
import json # For potential future use with more complex AI responses
import re # For regex-based quote escaping

# Import AI configuration from the main config.py
# This assumes config.py is in the same directory as app_chat.py or accessible via Python path.
from config import (
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME,
    GEMINI_MODEL_NAME, GEMINI_API_KEY, # Import GEMINI_API_KEY from config
    AI_MODEL_PROVIDER, # Default AI provider
    AI_CHAT_DB_USER_NAME, AI_CHAT_DB_USER_PASSWORD, # Import new config
    JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN # For creating playlist
)
from ai import get_gemini_playlist_name, get_ollama_playlist_name # Import functions to call AI

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
                print(f"Creating database user: {AI_CHAT_DB_USER_NAME}")
                cur.execute(f"CREATE USER {AI_CHAT_DB_USER_NAME} WITH PASSWORD %s;", (AI_CHAT_DB_USER_PASSWORD,))
                print(f"User {AI_CHAT_DB_USER_NAME} created.")
            else:
                print(f"User {AI_CHAT_DB_USER_NAME} already exists.")

            # Grant necessary permissions
            cur.execute("SELECT current_database();")
            current_db_name = cur.fetchone()[0]

            print(f"Granting CONNECT ON DATABASE {current_db_name} TO {AI_CHAT_DB_USER_NAME}")
            cur.execute(f"GRANT CONNECT ON DATABASE {current_db_name} TO {AI_CHAT_DB_USER_NAME};")

            print(f"Granting USAGE ON SCHEMA public TO {AI_CHAT_DB_USER_NAME}")
            cur.execute(f"GRANT USAGE ON SCHEMA public TO {AI_CHAT_DB_USER_NAME};")
            
            print(f"Granting SELECT ON public.score TO {AI_CHAT_DB_USER_NAME}")
            cur.execute(f"GRANT SELECT ON TABLE public.score TO {AI_CHAT_DB_USER_NAME};")
            
            # Revoke all other default privileges on public schema if necessary (more secure)
            # This is an advanced step and might be too restrictive if other public tables are needed by this user.
            # For now, we rely on explicit grants.
            # print(f"Revoking ALL ON SCHEMA public FROM {AI_CHAT_DB_USER_NAME} (except USAGE already granted)")
            # cur.execute(f"REVOKE ALL ON SCHEMA public FROM {AI_CHAT_DB_USER_NAME};") # This revokes USAGE too
            # cur.execute(f"GRANT USAGE ON SCHEMA public TO {AI_CHAT_DB_USER_NAME};") # Re-grant USAGE

            db_conn.commit()
            print(f"Permissions configured for user {AI_CHAT_DB_USER_NAME}.")
            ai_user_setup_done = True
    except Exception as e:
        db_conn.rollback()
        print(f"Error during AI user setup: {e}. AI user might not be correctly configured.")
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

    # Attempt to fix unescaped single quotes within potential string literals
    # This looks for a word character, a single quote, and another word character (e.g., "L'isola")
    # and replaces the single quote with two single quotes ("L''isola").
    # This is a heuristic and might need refinement if it causes issues with valid SQL.
    cleaned_sql = re.sub(r"(\w)'(\w)", r"\1''\2", cleaned_sql)

    # Attempt to normalize to ASCII to remove problematic characters like 'è'
    # This might change characters like 'è' to 'e'.
    try:
        cleaned_sql = unicodedata.normalize('NFKD', cleaned_sql).encode('ascii', 'ignore').decode('utf-8')
    except Exception as e_norm:
        print(f"Warning: Could not fully normalize SQL string to ASCII: {e_norm}")
    try:
        # Parse the query using sqlglot, specifying PostgreSQL dialect
        parsed_expressions = sqlglot.parse(cleaned_sql, read='postgres')
        if not parsed_expressions: # Should not happen if it starts with SELECT but good check
            return None, "SQLglot could not parse the query."
        
        # Get the first (and should be only) expression
        expression = parsed_expressions[0]

        # Check and enforce LIMIT 25
        limit_node = expression.args.get("limit")
        if limit_node:
            # If limit exists, check its value. sqlglot parses limit value as an expression.
            # We expect a simple number here.
            if not (isinstance(limit_node.expression, sqlglot.exp.Literal) and limit_node.expression.this == '25'):
                print(f"Query had LIMIT {limit_node.expression.sql()}. Changing to LIMIT 25.")
                expression.args["limit"] = sqlglot.exp.Limit(this=sqlglot.exp.Literal.number(25))
        else:
            # If no limit, add LIMIT 25
            print("Query had no LIMIT clause. Adding LIMIT 25.")
            expression.args["limit"] = sqlglot.exp.Limit(this=sqlglot.exp.Literal.number(25))

        # Re-generate the SQL from the potentially modified structure.
        cleaned_sql = expression.sql(dialect='postgres', pretty=False).strip().rstrip(';')
    except sqlglot.errors.ParseError as e:
        return None, f"SQLglot parsing error: {str(e)}"
    return cleaned_sql, None

@chat_bp.route('/')
def chat_home():
    """
    Serves the main chat page.
    """
    return render_template('chat.html')

@chat_bp.route('/api/config_defaults', methods=['GET'])
def chat_config_defaults_api():
    """
    API endpoint to provide default configuration values for the chat interface.
    """
    # Ensure that the GEMINI_API_KEY from config is only sent if it's not the placeholder.
    # For security, it's often better not to send API keys to the client at all,
    # but per your setup, we'll make it available as a default suggestion.
    config_gemini_api_key = GEMINI_API_KEY
    if GEMINI_API_KEY == "YOUR-GEMINI-API-KEY-HERE": # Check against the placeholder
        config_gemini_api_key = "" # Send empty if it's the placeholder

    return jsonify({
        "default_ai_provider": AI_MODEL_PROVIDER,
        "default_ollama_model_name": OLLAMA_MODEL_NAME,
        "ollama_server_url": OLLAMA_SERVER_URL, # Ollama server URL might be useful for display/info
        "default_gemini_model_name": GEMINI_MODEL_NAME,
        "default_gemini_api_key": config_gemini_api_key # API key from config.py
    }), 200

@chat_bp.route('/api/chatPlaylist', methods=['POST'])
def chat_playlist_api():
    """
    API endpoint to handle chat input and (mock) AI interaction.
    This is a synchronous endpoint.
    """
    data = request.get_json()
    print(f"DEBUG: chat_playlist_api called. Raw request data: {data}") # Log raw request

    from app import get_db # Import get_db here, inside the function
    if not data or 'userInput' not in data:
        return jsonify({"error": "Missing userInput in request"}), 400

    user_input = data.get('userInput')
    # Use AI provider from request, or fallback to global config, then to "NONE"
    ai_provider = data.get('ai_provider', AI_MODEL_PROVIDER).upper() # Use the imported constant
    ai_model_from_request = data.get('ai_model') # Model selected by user on chat page

    ai_response_message = f"Received your request: '{user_input}'.\n"
    actual_model_used = None
    query_results_list = None
    executed_query_str = None

    # Define the prompt structure once, to be used by any provider that needs it.
    # The [USER INPUT] placeholder will be replaced dynamically.
    base_expert_playlist_creator_prompt = """
    You are both a music trends expert (with deep knowledge of current radio charts, MTV, Spotify, YouTube trending songs, and other popular music services as of 2024-2025) AND a PostgreSQL query writer.

    Your mission:
    Convert the user's natural language playlist request into the best possible SQL query for table public.score. Before writing SQL:
    - Think carefully: what are the most famous, top, trending, or best songs and artists for this request, based on your knowledge?
    - Use specific hit song titles (not just artist matches or generic mood filters) to build a smart query.

    SQL RULES:
    - Return ONLY the raw SQL query. No comments, no markdown, no explanations.
    - Always SELECT: item_id, title, author
    - Final outer SELECT must apply: ORDER BY random(), LIMIT 25 (unless the user asks for ordered top/best/famous results).

    WHEN USER ASKS FOR TOP / FAMOUS / BEST / TRENDING / RADIO / MTV / YOUTUBE SONGS / FILM SONGS:
    - Build a CASE WHEN in ORDER BY that prioritizes exact known hit titles.
    - Include 100 well-matched song titles and author based on your knowledge.
    - You need to add both title and artist ILIKE.

    AUTHOR / TITLE FILTERING:
    - Title matches: use title IN ('song1', 'song2', ...) where possible, or CASE WHEN for ordering.
    - Artist matches: use author ILIKE '%Artist%' patterns for secondary support.
    - Title and Artis matches: when both title and artist are avaiable, ALWAYS using an AND clause WHERE (title = 'Song Title' AND author ILIKE '%Artist Name%')
    - For mood_vector or other_features filtering, use CAST + regex where necessary.

    MOOD / FEATURE FILTERING:
    - mood_vector and other_features columns contain comma-separated label:score pairs (0-1).
    - Extract numeric values using regex and CAST as float, e.g.:
    CAST(regexp_replace(substring(mood_vector FROM 'rock:([0-9]*\\.?[0-9]+)'), 'rock:', '') AS float) >= threshold
    - Use provided MOOD_LABELS and OTHER_FEATURE_LABELS for filtering.

    DATABASE STRUCTURE:
    - Table: public.score
    - Columns: 
    - item_id
    - title
    - author
    - tempo (numeric, 40-200)
    - key (text)
    - scale (text)
    - mood_vector (text, comma-separated label:score pairs where each score is 0-1, e.g. 'pop:0.8,rock:0.3')
    - other_features (text, comma-separated label:score pairs where each score is 0-1, e.g. 'danceable:0.7,party:0.6')
    - energy (numeric, 0-0.15)

    VALUE NOTES:
    - tempo values are between 40 and 200
    - energy values are between 0 and 0.15
    - mood_vector scores between 0 and 1; 0.2+ is already a good match
    - other_features scores between 0 and 1; 0.5+ is already a good match

    MOOD_LABELS:
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'dance', '00s', 'alternative rock', 'jazz', 'beautiful', 'metal', 'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock', 'electronica', '80s', 'folk', '90s', 'chill', 'instrumental', 'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental', 'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'funk', 'electro', 'heavy metal', '60s', 'rnb', 'indie pop', 'House'

    OTHER_FEATURE_LABELS:
    'danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad'

    POSTGRESQL SYNTAX:
    - DOUBLE-CHECK all syntax. UNION ALL must be wrapped in FROM (...) AS combined_results.
    - Do not alias individual SELECTs inside UNION ALL.

    Your task: Generate a smart SQL query for:
    "{user_input_placeholder}"
    """





    # --- OLLAMA ---
    if ai_provider == "OLLAMA":
        actual_model_used = ai_model_from_request or OLLAMA_MODEL_NAME
        # Use Ollama server URL from request, or fallback to config
        ollama_url_from_request = data.get('ollama_server_url', OLLAMA_SERVER_URL)
        ai_response_message += f"Processing with OLLAMA model: {actual_model_used} (at {ollama_url_from_request}).\n"

        full_prompt_for_ollama = base_expert_playlist_creator_prompt.replace("{user_input_placeholder}", user_input)
        # Pass the determined Ollama URL to the AI call
        ollama_response_raw = get_ollama_playlist_name(ollama_url_from_request, actual_model_used, full_prompt_for_ollama)

        if ollama_response_raw.startswith("Error:") or ollama_response_raw.startswith("An unexpected error occurred:") :
            ai_response_message += f"Ollama API Error: {ollama_response_raw}\n"
        else:
            ai_response_message += f"Ollama raw response (SQL query attempt):\n{ollama_response_raw}\n"

            sql_query, validation_error = clean_and_validate_sql(ollama_response_raw)
            executed_query_str = sql_query # Store the cleaned query

            # Ensure AI user is configured using the main app's DB connection
            try:
                main_db_conn_for_setup = get_db()
                _ensure_ai_user_configured(main_db_conn_for_setup)
            except Exception as setup_err:
                ai_response_message += f"Critical Error: Could not ensure AI user setup: {setup_err}\nQuery will not be executed.\n"

            if validation_error:
                ai_response_message += f"SQL Validation Error: {validation_error}\n"
            elif sql_query and ai_user_setup_done: # Only execute if SQL is valid AND setup was successful
                try:
                    with get_db().cursor(cursor_factory=DictCursor) as cur:
                        cur.execute(f"SET LOCAL ROLE {AI_CHAT_DB_USER_NAME};")
                        cur.execute(sql_query)
                        results = cur.fetchall()
                        get_db().commit()
                    
                    query_results_list = []
                    if results:
                        for row in results:
                            query_results_list.append({
                                "item_id": row.get("item_id"),
                                "title": row.get("title"),
                                "artist": row.get("author")
                            })
                        ai_response_message += f"Successfully executed query. Found {len(query_results_list)} songs.\n"
                    else:
                        ai_response_message += "Query executed successfully, but found no matching songs.\n"
                except Exception as e:
                    get_db().rollback()
                    ai_response_message += f"Database Error executing query: {str(e)}\n"
            elif sql_query and not ai_user_setup_done:
                ai_response_message += "AI User setup was not completed successfully. Query was not executed for security reasons.\n"

    # --- GEMINI ---
    elif ai_provider == "GEMINI":
        actual_model_used = ai_model_from_request or GEMINI_MODEL_NAME
        gemini_api_key_from_request = data.get('gemini_api_key')

        if not gemini_api_key_from_request:
            error_message = "Error: Gemini API key was not provided in the request.\n"
            ai_response_message += error_message
            return jsonify({"response": {"message": ai_response_message, "original_request": user_input, "ai_provider_used": ai_provider, "ai_model_selected": actual_model_used, "query_results": None, "executed_query": None}}), 400

        ai_response_message += f"Processing with GEMINI model: {actual_model_used} (using API key from request).\n"

        # Prepare the full prompt by inserting the user's input
        full_prompt_for_gemini = base_expert_playlist_creator_prompt.replace("{user_input_placeholder}", user_input)

        gemini_response_raw = get_gemini_playlist_name(gemini_api_key_from_request, actual_model_used, full_prompt_for_gemini)

        if gemini_response_raw.startswith("Error:"):
            ai_response_message += f"Gemini API Error: {gemini_response_raw}\n"
        else:
            ai_response_message += f"Gemini raw response (SQL query attempt):\n{gemini_response_raw}\n"

            sql_query, validation_error = clean_and_validate_sql(gemini_response_raw)
            executed_query_str = sql_query # Store the cleaned query

            # Ensure AI user is configured using the main app's DB connection
            # This connection (get_db()) should have privileges to create roles/users and grant.
            try:
                main_db_conn_for_setup = get_db()
                _ensure_ai_user_configured(main_db_conn_for_setup)
            except Exception as setup_err:
                ai_response_message += f"Critical Error: Could not ensure AI user setup: {setup_err}\nQuery will not be executed.\n"
                # Proceed without executing query if setup fails

            if validation_error:
                ai_response_message += f"SQL Validation Error: {validation_error}\n"
            elif sql_query and ai_user_setup_done: # Only execute if SQL is valid AND setup was successful
                try:
                    with get_db().cursor(cursor_factory=DictCursor) as cur: # Use main connection context
                        cur.execute(f"SET LOCAL ROLE {AI_CHAT_DB_USER_NAME};") # Execute as AI_USER for this transaction
                        cur.execute(sql_query)
                        results = cur.fetchall()
                        # SET LOCAL ROLE is transaction-scoped, role resets on commit/rollback.
                        # SELECT doesn't strictly need a commit, but it finalizes the transaction.
                        get_db().commit()
                    
                    query_results_list = []
                    if results:
                        for row in results:
                            query_results_list.append({
                                "item_id": row.get("item_id"), # Ensure these columns are in your SELECT
                                "title": row.get("title"),
                                "artist": row.get("author") # Map author to artist for frontend
                            })
                        ai_response_message += f"Successfully executed query. Found {len(query_results_list)} songs.\n"
                    else:
                        ai_response_message += "Query executed successfully, but found no matching songs.\n"
                except Exception as e:
                    get_db().rollback() # Rollback on error
                    ai_response_message += f"Database Error executing query: {str(e)}\n"
                    # query_results_list remains None or empty
            elif sql_query and not ai_user_setup_done:
                ai_response_message += "AI User setup was not completed successfully. Query was not executed for security reasons.\n"

    elif ai_provider == "NONE":
        ai_response_message += "No AI provider selected. Input acknowledged."
    else:
        ai_response_message += f"AI Provider '{ai_provider}' is not recognized."

    return jsonify({"response": {"message": ai_response_message, "original_request": user_input, "ai_provider_used": ai_provider, "ai_model_selected": actual_model_used, "executed_query": executed_query_str, "query_results": query_results_list}}), 200

@chat_bp.route('/api/createJellyfinPlaylist', methods=['POST'])
def create_jellyfin_playlist_api():
    """
    API endpoint to create a playlist on Jellyfin.
    """
    data = request.get_json()
    if not data or 'playlist_name' not in data or 'item_ids' not in data:
        return jsonify({"message": "Error: Missing playlist_name or item_ids in request"}), 400

    user_playlist_name = data.get('playlist_name')
    item_ids = data.get('item_ids') # This will be a list of strings

    if not user_playlist_name.strip():
        return jsonify({"message": "Error: Playlist name cannot be empty."}), 400
    if not item_ids:
        return jsonify({"message": "Error: No songs provided to create the playlist."}), 400

    # Append _instant to the playlist name
    jellyfin_playlist_name = f"{user_playlist_name.strip()}_instant"

    headers = {"X-Emby-Token": JELLYFIN_TOKEN}
    playlist_creation_url = f"{JELLYFIN_URL}/Playlists"
    body = {
        "Name": jellyfin_playlist_name,
        "Ids": item_ids, # item_ids should be a list of strings
        "UserId": JELLYFIN_USER_ID
    }

    try:
        response = requests.post(playlist_creation_url, headers=headers, json=body, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # Jellyfin usually returns the created playlist object on success
        created_playlist_info = response.json()
        return jsonify({"message": f"Successfully created playlist '{jellyfin_playlist_name}' on Jellyfin with ID: {created_playlist_info.get('Id')}"}), 200

    except requests.exceptions.RequestException as e:
        error_message = f"Error creating playlist on Jellyfin: {str(e)}"
        if hasattr(e, 'response') and e.response is not None: # type: ignore
            try: error_message += f" - Server Response: {e.response.text}" # type: ignore
            except: pass
        return jsonify({"message": error_message}), 500
    except Exception as e: # Catch any other unexpected errors
        return jsonify({"message": f"An unexpected error occurred: {str(e)}"}), 500