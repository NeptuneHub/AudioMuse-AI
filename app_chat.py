# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for the AI chat playlist generator.

Serves the chat UI (mounted at `/chat`) and turns a natural-language request
into a playlist by calling `tasks.ai.planner.plan_and_execute_once` with the
MCP tools from `tasks.ai.tools`, then materializes the result via
`app_server_context.create_instant_playlist_for_server`.

Main Features:
* Routes: `/` chat page, `/api/config_defaults`, `/api/chatPlaylist`,
  `/api/chatPlaylistStream` (Server-Sent Events), `/api/create_playlist`.
* Per-request AI provider/model override (Ollama/OpenAI/Gemini/Mistral) and
  optional `tasks.playlist_ordering.order_playlist` post-processing.
"""

from flask import Blueprint, render_template, request, jsonify, Response, stream_with_context
from flasgger import swag_from  # Import swag_from
import json  # For JSON serialization of tool arguments
import logging
import re
import time

import app_server_context
from error import error_manager
from error.error_dictionary import UNKNOWN_ERROR_CODE


logger = logging.getLogger(__name__)
# Import config module - read attributes at call time so runtime updates take effect
import config

_SSE_DATA_PREFIX = "data: "

# Create a Blueprint for chat-related routes
chat_bp = Blueprint(
    'chat_bp',
    __name__,
    template_folder='templates',  # Specifies where to look for templates like chat.html
    static_folder='static',
)


@chat_bp.route('/')
@swag_from(
    {
        'tags': ['Chat UI'],
        'summary': 'Serves the main chat interface HTML page.',
        'responses': {
            '200': {
                'description': 'HTML content of the chat page.',
                'content': {'text/html': {'schema': {'type': 'string'}}},
            }
        },
    }
)
def chat_home():
    """
    Serves the main chat page.
    """
    return render_template('chat.html', title='AudioMuse-AI - Instant Playlist', active='chat')


@chat_bp.route('/api/config_defaults', methods=['GET'])
@swag_from(
    {
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
                                'default_ai_provider': {'type': 'string', 'example': 'OLLAMA'},
                                'default_ollama_model_name': {
                                    'type': 'string',
                                    'example': 'mistral:7b',
                                },
                                'ollama_server_url': {
                                    'type': 'string',
                                    'example': 'http://127.0.0.1:11434/api/generate',
                                },
                                'default_openai_model_name': {'type': 'string', 'example': 'gpt-4'},
                                'openai_server_url': {
                                    'type': 'string',
                                    'example': 'https://openrouter.ai/api/v1/chat/completions',
                                },
                                'default_gemini_model_name': {
                                    'type': 'string',
                                    'example': 'gemini-2.5-pro',
                                },
                                'default_mistral_model_name': {
                                    'type': 'string',
                                    'example': 'ministral-3b-latest',
                                },
                            },
                        }
                    }
                },
            }
        },
    }
)
def chat_config_defaults_api():
    """
    API endpoint to provide default configuration values for the chat interface.
    """
    # Read from config module attributes (may be overridden by DB settings via apply_settings_to_config)
    import config as cfg

    return jsonify(
        {
            "default_ai_provider": cfg.AI_MODEL_PROVIDER,
            "default_ollama_model_name": cfg.OLLAMA_MODEL_NAME,
            "ollama_server_url": cfg.OLLAMA_SERVER_URL,
            "default_openai_model_name": cfg.OPENAI_MODEL_NAME,
            "openai_server_url": cfg.OPENAI_SERVER_URL,
            "default_gemini_model_name": cfg.GEMINI_MODEL_NAME,
            "default_mistral_model_name": cfg.MISTRAL_MODEL_NAME,
        }
    ), 200


def _reject_missing_user_input(data):
    # Shared guard for both chat endpoints: 400 on non-dict body or blank userInput.
    if (
        not isinstance(data, dict)
        or not isinstance(data.get('userInput'), str)
        or not data['userInput'].strip()
    ):
        return jsonify({"error": "Missing userInput in request"}), 400
    return None


@chat_bp.route('/api/chatPlaylist', methods=['POST'])
@swag_from(
    {
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
                                'example': "Songs for a rainy afternoon",
                            },
                            'ai_provider': {
                                'type': 'string',
                                'description': 'The AI provider to use (OLLAMA, OPENAI, GEMINI, MISTRAL, NONE). Defaults to server config.',
                                'example': 'GEMINI',
                                'enum': ['OLLAMA', 'OPENAI', 'GEMINI', "MISTRAL", 'NONE'],
                            },
                            'ai_model': {
                                'type': 'string',
                                'description': 'The specific AI model name to use. Defaults to server config for the provider.',
                                'example': 'gemini-2.5-pro',
                            },
                            'ollama_server_url': {
                                'type': 'string',
                                'description': 'Custom Ollama server URL (if ai_provider is OLLAMA).',
                                'example': 'http://localhost:11434/api/generate',
                            },
                            'openai_server_url': {
                                'type': 'string',
                                'description': 'Custom OpenAI/OpenRouter server URL (if ai_provider is OPENAI).',
                                'example': 'https://openrouter.ai/api/v1/chat/completions',
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
                            },
                        },
                    }
                }
            },
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
                                            'description': 'Log of AI interaction and processing.',
                                        },
                                        'original_request': {
                                            'type': 'string',
                                            'description': "The user's original input.",
                                        },
                                        'ai_provider_used': {
                                            'type': 'string',
                                            'description': 'The AI provider that was used for the request.',
                                        },
                                        'ai_model_selected': {
                                            'type': 'string',
                                            'description': 'The specific AI model that was selected/used.',
                                        },
                                        'executed_query': {
                                            'type': 'string',
                                            'nullable': True,
                                            'description': 'The SQL query that was executed (or last attempted).',
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
                                                    'artist': {'type': 'string'},
                                                },
                                            },
                                        },
                                    },
                                }
                            },
                        }
                    }
                },
            },
            '400': {
                'description': 'Bad Request - Missing input or invalid parameters.',
                'content': {
                    'application/json': {
                        'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
                    }
                },
            },
        },
    }
)
def chat_playlist_api():
    """
    Process user chat input to generate a playlist using AI with MCP tools.

    MCP TOOLS (4 CORE):
    1. seed_search - Songs similar to named seed songs/artists (union/alchemy/subtract)
    2. text_match - Semantic match on sound (CLAP) or lyric topics
    3. search_database - Filter by artist, album, genre, voice, mood, year, tempo, energy, key (ALL filters in ONE call)
    4. knowledge_lookup - Popularity/cultural requests turned into a grounded library recipe

    AI analyzes request -> calls tools -> combines results -> returns 100 songs

    Non-streaming variant: runs the whole pipeline then returns the full JSON.
    """
    data = request.get_json()
    err = _reject_missing_user_input(data)
    if err:
        return err
    try:
        app_server_context.resolve_request_server_id(data)
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return jsonify({'error': 'Invalid server selection.'}), 400
    log_messages = []
    resp_obj, status = _drain_pipeline(_run_chat_pipeline(data, log_messages))
    return jsonify({"response": resp_obj}), status


@chat_bp.route('/api/chatPlaylistStream', methods=['POST'])
def chat_playlist_stream_api():
    """
    Streaming variant of the playlist generator.

    Emits a Server-Sent-Event for every progress line as the pipeline produces it,
    then a final ``done`` event with the full response payload, so the frontend can
    show real, live progress with real per-step timing.

    Threadless: ``_run_chat_pipeline`` is a generator that ``yield``s a tick after
    each blocking step (LLM calls, the re-rank, DB queries). ``generate()`` runs it
    inline on the request thread and, after every tick, flushes any new
    ``log_messages`` lines as SSE ``log`` events; the generator's final ``return``
    value (the response object) is delivered as the ``done`` event.
    """
    data = request.get_json()
    err = _reject_missing_user_input(data)
    if err:
        return err
    try:
        app_server_context.resolve_request_server_id(data)
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return jsonify({'error': 'Invalid server selection.'}), 400

    @stream_with_context
    def generate():
        log_messages: list = []
        sent = 0

        def _flush():
            nonlocal sent
            out = ""
            while sent < len(log_messages):
                out += (
                    _SSE_DATA_PREFIX
                    + json.dumps({"type": "log", "line": log_messages[sent], "t": time.time()})
                    + "\n\n"
                )
                sent += 1
            return out

        # Emit a byte immediately so proxies/the browser open the pipe and don't
        # buffer while the first (slow) stage runs.
        yield ": stream-open\n\n"

        pipeline = _run_chat_pipeline(data, log_messages)
        resp_obj = None
        try:
            while True:
                try:
                    next(pipeline)
                except StopIteration as stop:
                    resp_obj = (stop.value or ({}, 200))[0]
                    break
                chunk = _flush()
                if chunk:
                    yield chunk
        except Exception:  # noqa: BLE001 - keep broad catch to protect streaming endpoint
            logger.exception("Streaming chat pipeline failed")
            yield (
                _SSE_DATA_PREFIX
                + json.dumps(
                    {"type": "error", "error": "An internal error has occurred.", "t": time.time()}
                )
                + "\n\n"
            )
            return

        trailing = _flush()
        if trailing:
            yield trailing
        yield (
            _SSE_DATA_PREFIX
            + json.dumps({"type": "done", "response": resp_obj, "t": time.time()})
            + "\n\n"
        )

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


def _drain_pipeline(pipeline):
    """Run a ``_run_chat_pipeline`` generator to completion, discarding the
    progress ticks, and return its final ``(response_obj, status)`` value."""
    try:
        while True:
            next(pipeline)
    except StopIteration as stop:
        return stop.value or ({}, 200)


def _run_chat_pipeline(data, log_messages):
    """Core chat-to-playlist pipeline, a GENERATOR. Appends progress to
    ``log_messages`` and ``yield``s a bare tick after each blocking step so the
    streaming endpoint can flush new lines live. Its final ``return`` value is
    ``(response_obj_dict, http_status)`` (read via ``StopIteration.value`` /
    ``_drain_pipeline``). Early ``return``s before the first ``yield`` still work --
    the function is a generator by virtue of the ``yield from`` below.
    """
    # Mask API key if present in the debug log
    data_for_log = dict(data) if data else {}
    if 'gemini_api_key' in data_for_log and data_for_log['gemini_api_key']:
        data_for_log['gemini_api_key'] = 'API-KEY'
    if 'mistral_api_key' in data_for_log and data_for_log['mistral_api_key']:
        data_for_log['mistral_api_key'] = 'API-KEY'
    if 'openai_api_key' in data_for_log and data_for_log['openai_api_key']:
        data_for_log['openai_api_key'] = 'API-KEY'
    logger.debug("chat_playlist_api called. Raw request data: %s", data_for_log)

    from tasks.ai.tools import get_mcp_tools
    from tasks.ai.planner import plan_and_execute_once

    original_user_input = data.get('userInput')
    # Detect if user's request mentions ratings (guard against AI hallucinating rating filters)
    _user_wants_rating = bool(
        re.search(
            r'\b(rat(ed|ing|ings)|stars?|⭐|favorit|best[\s-]?rated|top[\s-]?rated|highly[\s-]?rated)\b',
            original_user_input,
            re.IGNORECASE,
        )
    )
    ai_provider = data.get('ai_provider', config.AI_MODEL_PROVIDER).upper()
    ai_model_from_request = data.get('ai_model')

    log_messages.append("NEW MCP-BASED PLAYLIST GENERATION")
    log_messages.append(f"Request: '{original_user_input}'")
    log_messages.append(f"AI Provider: {ai_provider}")

    # Check if AI provider is NONE
    if ai_provider == "NONE":
        return (
            {
                "message": "No AI provider selected. Please configure an AI provider to use this feature.",
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": None,
                "executed_query": None,
                "query_results": None,
            },
            200,
        )

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
        return (
            {
                "message": "\n".join(log_messages),
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": ai_config.get('openai_model'),
                "executed_query": None,
                "query_results": None,
            },
            400,
        )

    if ai_provider == "GEMINI" and (
        not ai_secrets['gemini_key'] or ai_secrets['gemini_key'] == "YOUR-GEMINI-API-KEY-HERE"
    ):
        error_msg = "Error: Gemini API key is missing. Please provide a valid API key."
        log_messages.append(error_msg)
        return (
            {
                "message": "\n".join(log_messages),
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": ai_config.get('gemini_model'),
                "executed_query": None,
                "query_results": None,
            },
            400,
        )

    if ai_provider == "MISTRAL" and (
        not ai_secrets['mistral_key'] or ai_secrets['mistral_key'] == "YOUR-MISTRAL-API-KEY-HERE"
    ):
        error_msg = "Error: Mistral API key is missing. Please provide a valid API key."
        log_messages.append(error_msg)
        return (
            {
                "message": "\n".join(log_messages),
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": ai_config.get('mistral_model'),
                "executed_query": None,
                "query_results": None,
            },
            400,
        )

    # ====================
    # MCP AGENTIC WORKFLOW
    # ====================

    log_messages.append("\nUsing MCP Agentic Workflow for playlist generation")
    log_messages.append("Target: 100 songs")

    # Get MCP tools and library context
    mcp_tools = get_mcp_tools()
    log_messages.append(f"Available tools: {', '.join([t['name'] for t in mcp_tools])}")

    # Fetch library context for smarter AI prompting
    from tasks.mcp_helper import get_library_context

    library_context = get_library_context()
    if library_context.get('total_songs', 0) > 0:
        log_messages.append(
            f"Library: {library_context['total_songs']} songs, {library_context['unique_artists']} artists"
        )

    yield

    target_song_count = 100
    from config import MAX_SONGS_PER_ARTIST_PLAYLIST

    collection_cap = 1000

    plan_result = yield from plan_and_execute_once(
        user_message=f'Build a {target_song_count}-song playlist for: "{original_user_input}"',
        tools=mcp_tools,
        ai_config=ai_config_with_secrets,
        log_messages=log_messages,
        library_context=library_context,
        user_wants_rating=_user_wants_rating,
        collection_cap=collection_cap,
        target_song_count=target_song_count,
    )

    if 'error' in plan_result:
        # No fallback: do NOT invent an unrelated genre playlist. Return no
        # results so the user sees that the AI couldn't build a plan for this
        # request, rather than a made-up playlist.
        log_messages.append(f"AI planning failed: {plan_result['error']}")
        return (
            {
                "message": "\n".join(log_messages),
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": ai_config.get(f'{ai_provider.lower()}_model'),
                "executed_query": None,
                "query_results": None,
            },
            200,
        )

    all_songs = plan_result['songs']
    song_sources = plan_result['song_sources']
    tools_used_history = plan_result['tools_used_history']
    plan_notes = plan_result.get('plan_notes', [])
    executed_query_str = plan_result['executed_query_str']
    filter_applied = plan_result.get('filter_applied', False)

    # Keep canonical ids here: this pool is filtered for availability but stays
    # internal - it feeds playlist selection and create_instant_playlist_for_server,
    # which re-translates to the server's ids itself. Translating now would double it.
    scoped_pool = app_server_context.scope_results(
        all_songs, None, id_key='item_id', translate=False
    )
    if len(scoped_pool) != len(all_songs):
        log_messages.append(
            f"\nServer availability: removed {len(all_songs) - len(scoped_pool)} "
            "unavailable songs before playlist selection"
        )
    all_songs = scoped_pool

    log_messages.append(
        f"\nCollected {len(all_songs)} songs (target {target_song_count}, cap {collection_cap})"
    )

    yield

    # Prepare final results
    if all_songs:
        # NOTE: rating is NOT hard-filtered here. Like every other filter dim it
        # is applied as a SOFT re-rank inside planner._rerank_pool (rating/5
        # gradient), so high-rated songs float up but nothing is removed.

        # --- Phase 1: Artist Diversity Cap on full collected pool ---
        max_per_artist = MAX_SONGS_PER_ARTIST_PLAYLIST
        artist_song_counts = {}
        diversified_pool = []
        diversity_overflow = []
        for song in all_songs:
            artist = song.get('artist', 'Unknown')
            artist_song_counts[artist] = artist_song_counts.get(artist, 0) + 1
            if artist_song_counts[artist] <= max_per_artist:
                diversified_pool.append(song)
            else:
                diversity_overflow.append(song)

        diversity_removed = len(all_songs) - len(diversified_pool)
        if diversity_removed > 0:
            log_messages.append(
                f"\nArtist diversity: removed {diversity_removed} excess songs from pool (max {max_per_artist}/artist)"
            )

        # --- Phase 2: Proportional sampling from diversified pool ---
        if len(diversified_pool) <= target_song_count:
            # Not enough songs after diversity cap - use all, then backfill from overflow
            final_query_results_list = list(diversified_pool)
            if len(final_query_results_list) < target_song_count and diversity_overflow:
                # Progressive cap relaxation: raise per-artist cap until we hit target or exhaust overflow
                current_cap = max_per_artist
                while len(final_query_results_list) < target_song_count and diversity_overflow:
                    current_cap += 1
                    # Recount artists in current final list
                    diverse_artist_counts = {}
                    for s in final_query_results_list:
                        a = s.get('artist', 'Unknown')
                        diverse_artist_counts[a] = diverse_artist_counts.get(a, 0) + 1
                    # Try to add overflow songs that fit the raised cap
                    still_overflow = []
                    backfill_added = 0
                    for song in diversity_overflow:
                        if len(final_query_results_list) >= target_song_count:
                            still_overflow.append(song)
                            continue
                        artist = song.get('artist', 'Unknown')
                        if diverse_artist_counts.get(artist, 0) < current_cap:
                            final_query_results_list.append(song)
                            diverse_artist_counts[artist] = diverse_artist_counts.get(artist, 0) + 1
                            backfill_added += 1
                        else:
                            still_overflow.append(song)
                    diversity_overflow = still_overflow
                    if backfill_added == 0:
                        break  # No progress at this cap level, stop
                if current_cap > max_per_artist:
                    log_messages.append(
                        f"   Progressive cap relaxation: {max_per_artist} -> {current_cap}/artist to reach {len(final_query_results_list)} songs"
                    )
        else:
            # More diversified songs than target - sample proportionally by tool call
            songs_by_call = {}
            for song in diversified_pool:
                call_index = song_sources.get(song['item_id'], -1)
                if call_index not in songs_by_call:
                    songs_by_call[call_index] = []
                songs_by_call[call_index].append(song)

            total_in_pool = len(diversified_pool)
            final_query_results_list = []
            for call_index, tool_songs in songs_by_call.items():
                proportion = len(tool_songs) / total_in_pool
                allocated = int(proportion * target_song_count)
                if allocated == 0 and len(tool_songs) > 0:
                    allocated = 1
                final_query_results_list.extend(tool_songs[:allocated])

            # Round-up correction: fill remaining slots from diversified songs not yet selected
            if len(final_query_results_list) < target_song_count:
                selected_ids = {s['item_id'] for s in final_query_results_list}
                remaining = [s for s in diversified_pool if s['item_id'] not in selected_ids]
                needed = target_song_count - len(final_query_results_list)
                final_query_results_list.extend(remaining[:needed])

            final_query_results_list = final_query_results_list[:target_song_count]

        log_messages.append(
            f"\nPool: {len(all_songs)} collected -> {len(diversified_pool)} after diversity cap -> {len(final_query_results_list)} in final playlist"
        )

        # --- Song Ordering for Smooth Transitions (Phase 3A) ---
        # Only when NO filter drove the result. When a filter/score was applied
        # (e.g. "female vocalist", a genre, year, etc.), the songs are already in
        # the order the score produced -- matched songs on top, then the rest by
        # similarity. Re-sorting by tempo/energy/key here would scramble that and
        # bury the matched songs, so the scored order is preserved instead.
        if filter_applied:
            log_messages.append(
                "\nPlaylist kept in filter-ranked order (matched songs first); smooth-transition reorder skipped"
            )
        else:
            try:
                from tasks.playlist_ordering import order_playlist
                from config import PLAYLIST_ENERGY_ARC

                song_id_list = [s['item_id'] for s in final_query_results_list]
                ordered_ids = order_playlist(song_id_list, energy_arc=PLAYLIST_ENERGY_ARC)

                # Rebuild list in new order
                id_to_song = {s['item_id']: s for s in final_query_results_list}
                final_query_results_list = [
                    id_to_song[sid] for sid in ordered_ids if sid in id_to_song
                ]
                log_messages.append("\nPlaylist ordered for smooth transitions (tempo/energy/key)")
            except Exception:
                logger.warning("Playlist ordering failed (non-fatal)", exc_info=True)
                log_messages.append(
                    "\nPlaylist ordering skipped due to an internal processing issue"
                )

        final_executed_query_str = executed_query_str

        if plan_notes:
            log_messages.append("\nPlan notes:")
            for n in plan_notes:
                log_messages.append(f"   {n}")

        log_messages.append(
            f"\nOK SUCCESS! Generated playlist with {len(final_query_results_list)} songs"
        )
        log_messages.append(f"   Total songs collected: {len(all_songs)}")
        log_messages.append(f"   Tools called: {len(tools_used_history)}")

        # Show tool contribution breakdown (collected vs final)
        log_messages.append("\nTool Contribution (Collected -> Final Playlist):")

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
            if 'exclude_artists' in args and args['exclude_artists']:
                args_preview.append(f"exclude_artists={args['exclude_artists'][:2]}")
            if 'exclude_genres' in args and args['exclude_genres']:
                args_preview.append(f"exclude_genres={args['exclude_genres'][:2]}")
            if 'user_request' in args:
                args_preview.append(f"request='{args['user_request'][:30]}...'")

            args_str = ", ".join(args_preview) if args_preview else "no filters"
            call_index = tool_info.get('call_index', -1)
            final_count = final_by_call.get(call_index, 0)
            if song_count != final_count:
                log_messages.append(
                    f"   - {tool_name}({args_str}): {song_count} collected -> {final_count} in final playlist"
                )
            else:
                log_messages.append(f"   - {tool_name}({args_str}): {song_count} songs")
    else:
        log_messages.append("\nNo songs collected")
        if plan_notes:
            log_messages.append("\nPlan notes:")
            for n in plan_notes:
                log_messages.append(f"   {n}")
        log_messages.append(
            "\nNo matching songs were found in your library for this request "
            "(a corrective retry was already attempted). Try naming an artist, album or "
            "genre that exists in your library, or loosen the constraints."
        )
        final_query_results_list = None
        final_executed_query_str = executed_query_str or "MCP single-pass: No results"

    actual_model_used = ai_config.get(f'{ai_provider.lower()}_model')

    # The pool stayed canonical for internal selection/ordering; translate the
    # FINAL list to the selected server's provider ids so the response never emits
    # an internal fp_ id. /api/create_playlist resolves them back to canonical.
    if final_query_results_list:
        final_query_results_list = app_server_context.scope_results(
            final_query_results_list, None, id_key='item_id'
        )

    # Return final response object (caller wraps it for HTTP).
    return (
        {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": actual_model_used,
            "executed_query": final_executed_query_str,
            "query_results": final_query_results_list,
        },
        200,
    )


@chat_bp.route('/api/create_playlist', methods=['POST'])
@swag_from(
    {
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
                                'example': 'My Awesome Mix',
                            },
                            'item_ids': {
                                'type': 'array',
                                'description': 'A list of item IDs for the songs to include.',
                                'items': {'type': 'string'},
                                'example': [
                                    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                                    "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
                                ],
                            },
                        },
                    }
                }
            },
        },
        'responses': {
            '200': {
                'description': 'Playlist successfully created.',
                'content': {
                    'application/json': {
                        'schema': {'type': 'object', 'properties': {'message': {'type': 'string'}}}
                    }
                },
            },
            '400': {'description': 'Bad Request - Missing parameters or invalid input.'},
            '500': {
                'description': 'Server Error - Failed to create playlist.',
                'content': {  # Added content for 400 and 500 for consistency
                    'application/json': {
                        'schema': {'type': 'object', 'properties': {'message': {'type': 'string'}}}
                    }
                },
            },
        },
    }
)
def create_media_server_playlist_api():
    """
    API endpoint to create a playlist on the configured media server.
    """
    data = request.get_json()
    if not data or 'playlist_name' not in data or 'item_ids' not in data:
        return jsonify({"message": "Error: Missing playlist_name or item_ids in request"}), 400

    user_playlist_name = data.get('playlist_name')
    item_ids = data.get('item_ids')  # This will be a list of strings

    if not user_playlist_name or not str(user_playlist_name).strip():
        return jsonify({"message": "Error: Playlist name cannot be empty."}), 400
    if not item_ids:
        return jsonify({"message": "Error: No songs provided to create the playlist."}), 400

    try:
        server_id = app_server_context.resolve_request_server_id(data)
    except ValueError as exc:
        return jsonify({"message": f"Error: {exc}"}), 400

    # The client posts back the provider ids it got from /api/chatPlaylist;
    # canonicalize them so the dispatcher translates to the target server exactly
    # once. A canonical id passes through unchanged (older clients keep working).
    resolved = app_server_context.resolve_input_item_ids(item_ids, data)
    item_ids = [resolved.get(str(i), i) for i in item_ids]

    try:
        try:
            info = app_server_context.create_instant_playlist_for_server(
                user_playlist_name, item_ids, server_id
            )
        except ValueError as exc:
            return jsonify({"message": f"Error: {exc}"}), 400
        created_playlist_info = info['result']

        if not created_playlist_info:
            raise Exception("Media server did not return playlist information after creation.")

        return jsonify(
            {
                "message": f"Successfully created playlist '{user_playlist_name}' on the media server with ID: {created_playlist_info.get('Id')}"
            }
        ), 200

    except Exception as e:
        # Log detailed error on the server
        error_details_for_server = f"Media Server API Request Exception: {str(e)}\n"
        if hasattr(e, 'response') and e.response is not None:  # type: ignore[attr-defined]
            try:
                error_details_for_server += f" - Media Server Response: {e.response.text}\n"
            except Exception:
                pass  # nosec
        logger.exception(
            "Error in create_media_server_playlist_api: %s", error_details_for_server
        )
        # Return generic, structured error to client (traceback stays in the log only).
        code = error_manager.classify(e, UNKNOWN_ERROR_CODE)
        payload = error_manager.build(code)
        payload["message"] = "An internal error occurred while creating the playlist."
        payload["error"] = payload["message"]
        return jsonify(payload), error_manager.http_status_for_code(code)
