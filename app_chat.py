# app_chat.py
from flask import Blueprint, render_template, request, jsonify
from flasgger import swag_from # Import swag_from
import json # For JSON serialization of tool arguments
import logging
import re


logger = logging.getLogger(__name__)
# Import config module - read attributes at call time so runtime updates take effect
import config

# Create a Blueprint for chat-related routes
chat_bp = Blueprint('chat_bp', __name__,
                    template_folder='templates', # Specifies where to look for templates like chat.html
                    static_folder='static')



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
    # Read from config module attributes (may be overridden by DB settings via apply_settings_to_config)
    import config as cfg
    return jsonify({
        "default_ai_provider": cfg.AI_MODEL_PROVIDER,
        "default_ollama_model_name": cfg.OLLAMA_MODEL_NAME,
        "ollama_server_url": cfg.OLLAMA_SERVER_URL,
        "default_openai_model_name": cfg.OPENAI_MODEL_NAME,
        "openai_server_url": cfg.OPENAI_SERVER_URL,
        "default_gemini_model_name": cfg.GEMINI_MODEL_NAME,
        "default_mistral_model_name": cfg.MISTRAL_MODEL_NAME,
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
    Process user chat input to generate a playlist using JointBERT routing.

    FLOW:
    1. JointBERT NLP router analyzes user input (intent + entity slots)
    2. High confidence (>= 0.7) → Execute tools directly
    3. Low confidence (< 0.7) → Fallback to AI brainstorm

    MCP TOOLS:
    1. song_similarity - Find songs similar to a specific song
    2. text_search - Natural language search (instruments, moods, descriptions)
    3. artist_similarity - Songs by an artist and similar artists
    4. song_alchemy - Blend/subtract multiple artists (vector arithmetic)
    5. search_database - Filter by genre, mood, tempo, energy, year, rating, scale
    6. lyrics_search - Search by lyrics content
    7. ai_brainstorm - AI suggests songs (fallback for uncertain queries)
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
    # Detect if user's request mentions ratings (guard against AI hallucinating rating filters)
    _user_wants_rating = bool(re.search(
        r'\b(rat(ed|ing|ings)|stars?|⭐|favorit|best[\s-]?rated|top[\s-]?rated|highly[\s-]?rated)\b',
        original_user_input, re.IGNORECASE
    ))
    ai_provider = data.get('ai_provider', config.AI_MODEL_PROVIDER).upper()
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
    
    # Build AI configuration object.
    # SECURITY: API keys come ONLY from server-side config (DB-overlaid).
    # Any *_api_key field in the client payload is ignored to prevent token
    # exfiltration via the chat endpoint -- the user explicitly may select a
    # provider/model/url from the client, but the secret token must already be
    # saved on the server.
    #
    # Secrets are kept in a SEPARATE dict (`ai_secrets`) so they never coexist
    # with loggable fields. This breaks CodeQL's clear-text-logging taint flow:
    # nothing logged below ever indexes into a dict that holds keys.
    ai_config = {
        'provider': ai_provider,
        'ollama_url': data.get('ollama_server_url', config.OLLAMA_SERVER_URL),
        'ollama_model': ai_model_from_request or config.OLLAMA_MODEL_NAME,
        'openai_url': data.get('openai_server_url', config.OPENAI_SERVER_URL),
        'openai_model': ai_model_from_request or config.OPENAI_MODEL_NAME,
        'gemini_model': ai_model_from_request or config.GEMINI_MODEL_NAME,
        'mistral_model': ai_model_from_request or config.MISTRAL_MODEL_NAME,
    }
    ai_secrets = {
        'openai_key': config.OPENAI_API_KEY,
        'gemini_key': config.GEMINI_API_KEY,
        'mistral_key': config.MISTRAL_API_KEY,
    }
    # The downstream AI layer expects a single merged dict.
    ai_config_with_secrets = {**ai_config, **ai_secrets}

    # Log the resolved AI target so it shows up in the flask log (without keys).
    _resolved_url = {
        "OLLAMA": ai_config['ollama_url'],
        "OPENAI": ai_config['openai_url'],
        "GEMINI": "(gemini-api)",
        "MISTRAL": "(mistral-api)",
    }.get(ai_provider, "(none)")
    _resolved_model = {
        "OLLAMA": ai_config['ollama_model'],
        "OPENAI": ai_config['openai_model'],
        "GEMINI": ai_config['gemini_model'],
        "MISTRAL": ai_config['mistral_model'],
    }.get(ai_provider, "(none)")
    logger.info(
        "chat_playlist_api -> provider=%s url=%s model=%s (default_provider=%s, client_override=%s)",
        ai_provider,
        _resolved_url,
        _resolved_model,
        config.AI_MODEL_PROVIDER,
        bool(data.get('ai_provider')),
    )

    # Validate API keys for cloud providers
    if ai_provider == "OPENAI" and not ai_secrets['openai_key']:
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
    
    if ai_provider == "GEMINI" and (not ai_secrets['gemini_key'] or ai_secrets['gemini_key'] == "YOUR-GEMINI-API-KEY-HERE"):
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
    
    if ai_provider == "MISTRAL" and (not ai_secrets['mistral_key'] or ai_secrets['mistral_key'] == "YOUR-MISTRAL-API-KEY-HERE"):
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
    
    # ========================
    # JOINBERT-FIRST ROUTING
    # ========================
    from tasks.playlist_engine import build_instant_playlist

    result = build_instant_playlist(original_user_input, ai_config_with_secrets)
    final_query_results_list = result.get("songs")
    message_text = result.get("message", "")
    final_executed_query_str = result.get("executed_query", "unknown")
    ai_used = result.get("ai_used", False)

    # Log messages are already in result["message"], but we'll parse them for display
    log_messages.append(message_text)

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
        created_playlist_info = create_instant_playlist(user_playlist_name, item_ids)

        if not created_playlist_info:
            raise Exception("Media server did not return playlist information after creation.")

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
