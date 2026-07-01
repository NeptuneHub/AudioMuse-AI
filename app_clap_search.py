# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for CLAP natural-language music search.

Serves the `/clap_search` UI and its API, delegating to
`tasks.clap_text_search` which matches a text query against tracks via CLAP
audio<->text embeddings and caches results.

Main Features:
* Routes: `/clap_search` page, `/api/clap/search`, warmup (`/api/clap/warmup`
  and its `/status`), `/api/clap/cache/refresh`, `/api/clap/stats`,
  `/api/clap/top_queries`.
* Task imports are deferred into the handlers so the blueprint loads even when
  CLAP is disabled.
"""

from flask import Blueprint, render_template, request, jsonify
import logging

logger = logging.getLogger(__name__)

clap_search_bp = Blueprint('clap_search_bp', __name__, template_folder='../templates')


@clap_search_bp.route('/clap_search', methods=['GET'])
def clap_search_page():
    """
    CLAP text search UI page.
    ---
    tags:
      - CLAP Search
    summary: HTML page for natural-language music search powered by CLAP audio<->text embeddings.
    responses:
      200:
        description: HTML page rendered.
    """
    from config import CLAP_ENABLED, APP_VERSION
    from tasks.clap_text_search import get_cache_stats

    cache_stats = get_cache_stats()

    return render_template(
        'clap_search.html',
        title='Text Search - AudioMuse-AI',
        active='clap_search',
        app_version=APP_VERSION,
        clap_enabled=CLAP_ENABLED,
        cache_stats=cache_stats,
    )


@clap_search_bp.route('/api/clap/search', methods=['POST'])
def clap_search_api():
    """
    Run a CLAP text-to-audio similarity search.
    ---
    tags:
      - CLAP Search
    summary: Return the top-N tracks whose CLAP audio embedding best matches a free-text query.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [query]
            properties:
              query:
                type: string
                minLength: 3
                example: "upbeat summer songs"
              limit:
                type: integer
                minimum: 1
                maximum: 500
                default: 100
    responses:
      200:
        description: Search results sorted by descending similarity.
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                count:
                  type: integer
                results:
                  type: array
                  items:
                    type: object
                    properties:
                      item_id:
                        type: string
                      title:
                        type: string
                      author:
                        type: string
                      similarity:
                        type: number
                        format: float
      400:
        description: CLAP disabled, missing query, or query too short.
      500:
        description: Internal error during search.
      503:
        description: CLAP cache not loaded yet (run analysis first).
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import search_by_text, is_clap_cache_loaded
    from app_helper import attach_song_features

    if not CLAP_ENABLED:
        return jsonify(
            {
                'error': 'CLAP text search is disabled. Set CLAP_ENABLED=true in config.',
                'results': [],
            }
        ), 400

    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" in request body'}), 400

        query = data['query'].strip()
        limit = data.get('limit', 100)

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        if len(query) < 1:
            return jsonify({'error': 'Query must be at least 1 character'}), 400

        # Validate limit
        limit = min(max(1, int(limit)), 500)  # Between 1 and 500

        # Check if cache is loaded
        if not is_clap_cache_loaded():
            return jsonify(
                {'error': 'CLAP cache not loaded. Please run song analysis first.', 'results': []}
            ), 503

        # Perform search
        results = search_by_text(query, limit=limit)
        attach_song_features(results)

        return jsonify({'query': query, 'results': results, 'count': len(results)})

    except ValueError as e:
        logger.warning(f"ValueError in DCLAP search API: {e}")
        return jsonify({'error': 'Invalid or missing request parameter.'}), 400
    except Exception as e:
        logger.exception(f"DCLAP search API error: {e}")
        return jsonify({'error': 'An internal server error occurred during DCLAP search.'}), 500


@clap_search_bp.route('/api/clap/warmup', methods=['POST'])
def warmup_model_api():
    """
    Warm up the CLAP text-search model.
    ---
    tags:
      - CLAP Search
    summary: Preload the DCLAP model and reset the 10-minute idle-eviction timer.
    description: Call this when the search page loads to ensure subsequent queries are fast.
    responses:
      200:
        description: Model loaded; cache timer reset.
        content:
          application/json:
            schema:
              type: object
              properties:
                loaded:
                  type: boolean
                expiry_seconds:
                  type: integer
                  example: 600
      400:
        description: CLAP search is disabled.
      500:
        description: Warmup failed.
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import warmup_text_search_model

    if not CLAP_ENABLED:
        return jsonify({'error': 'CLAP text search is disabled', 'loaded': False}), 400

    try:
        status = warmup_text_search_model()
        return jsonify(status)
    except Exception:
        logger.exception("Model warmup failed")
        return jsonify({'error': 'Warmup failed.', 'loaded': False}), 500


@clap_search_bp.route('/api/clap/warmup/status', methods=['GET'])
def warmup_status_api():
    """
    CLAP warmup status.
    ---
    tags:
      - CLAP Search
    summary: Return whether the warm cache is still active and how long until it expires.
    responses:
      200:
        description: Warm cache state.
        content:
          application/json:
            schema:
              type: object
              properties:
                active:
                  type: boolean
                seconds_remaining:
                  type: integer
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import get_warm_cache_status

    if not CLAP_ENABLED:
        return jsonify({'active': False, 'seconds_remaining': 0})

    try:
        status = get_warm_cache_status()
        return jsonify(status)
    except Exception as e:
        logger.exception(f"Failed to get warmup status: {e}")
        return jsonify({'active': False, 'seconds_remaining': 0})


@clap_search_bp.route('/api/clap/cache/refresh', methods=['POST'])
def refresh_cache_api():
    """
    Refresh the CLAP audio-embedding cache.
    ---
    tags:
      - CLAP Search
    summary: Reload the CLAP cache from the database (call after analysis completes).
    responses:
      200:
        description: Cache refreshed; updated stats returned.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                message:
                  type: string
                stats:
                  type: object
      400:
        description: CLAP disabled.
      500:
        description: Refresh failed.
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import refresh_clap_cache, get_cache_stats

    if not CLAP_ENABLED:
        return jsonify({'error': 'CLAP is disabled'}), 400

    try:
        success = refresh_clap_cache()
        stats = get_cache_stats()

        if success:
            return jsonify(
                {'success': True, 'message': 'CLAP cache refreshed successfully', 'stats': stats}
            )
        else:
            return jsonify(
                {'success': False, 'message': 'Failed to refresh CLAP cache', 'stats': stats}
            ), 500

    except Exception as e:
        logger.exception(f"Cache refresh failed: {e}")
        return jsonify(
            {'success': False, 'error': 'An internal error occurred. Please try again later.'}
        ), 500


@clap_search_bp.route('/api/clap/stats', methods=['GET'])
def cache_stats_api():
    """
    CLAP cache stats.
    ---
    tags:
      - CLAP Search
    summary: Return cache size, freshness, and the CLAP_ENABLED flag.
    responses:
      200:
        description: Cache statistics.
        content:
          application/json:
            schema:
              type: object
              properties:
                clap_enabled:
                  type: boolean
                num_embeddings:
                  type: integer
                last_refresh:
                  type: string
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import get_cache_stats

    stats = get_cache_stats()
    stats['clap_enabled'] = CLAP_ENABLED

    return jsonify(stats)


@clap_search_bp.route('/api/clap/top_queries', methods=['GET'])
def top_queries_api():
    """
    Top diverse CLAP queries.
    ---
    tags:
      - CLAP Search
    summary: Return the precomputed top-50 diverse search queries shown as suggestions.
    description: |
      Returns an empty list with `ready: false` if the background diversity
      computation hasn't finished yet.
    responses:
      200:
        description: Suggested queries.
        content:
          application/json:
            schema:
              type: object
              properties:
                queries:
                  type: array
                  items:
                    type: string
                ready:
                  type: boolean
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import get_cached_top_queries

    if not CLAP_ENABLED:
        return jsonify({'queries': [], 'ready': False, 'message': 'CLAP disabled'}), 200

    try:
        queries = get_cached_top_queries()
        return jsonify({'queries': queries, 'ready': len(queries) > 0}), 200
    except Exception:
        logger.exception("Failed to get top queries")
        return jsonify(
            {
                'error': 'An internal error occurred. Please try again later.',
                'queries': [],
                'ready': False,
            }
        ), 500
