"""
Lyrics Search Blueprint
Provides web interface and API for lyrics-based song search.

Two modes:
  * Axis search: target sliders over MUSIC_ANALYSIS_AXES labels (0..1).
  * Free-text search: e5-base-v2 embedding nearest-neighbor on the lyrics
    voyager index built from per-song lyrics embeddings.
"""

import logging

from flask import Blueprint, jsonify, render_template, request

from app_helper import attach_song_features

logger = logging.getLogger(__name__)

lyrics_search_bp = Blueprint('lyrics_search_bp', __name__, template_folder='../templates')


@lyrics_search_bp.route('/lyrics_search', methods=['GET'])
def lyrics_search_page():
    """
    Lyrics search UI page.
    ---
    tags:
      - Lyrics Search
    summary: HTML page for axis-based and free-text search over song lyrics embeddings.
    responses:
      200:
        description: HTML page rendered.
    """
    from config import APP_VERSION, LYRICS_ENABLED
    from tasks.lyrics_manager import get_axes_definition, get_cache_stats
    from tasks.sem_grove_manager import get_sem_grove_stats

    cache_stats      = get_cache_stats()
    axes             = get_axes_definition() if LYRICS_ENABLED else {}
    sem_grove_stats  = get_sem_grove_stats()

    return render_template(
        'lyrics_search.html',
        title='Lyrics Search - AudioMuse-AI',
        active='lyrics_search',
        app_version=APP_VERSION,
        lyrics_enabled=LYRICS_ENABLED,
        cache_stats=cache_stats,
        axes=axes,
        sem_grove_stats=sem_grove_stats,
    )


@lyrics_search_bp.route('/api/lyrics/search/axes', methods=['POST'])
def lyrics_search_axes_api():
    """
    Lyrics search by axis selections.
    ---
    tags:
      - Lyrics Search
    summary: Match lyrics against one chosen label per MUSIC_ANALYSIS_AXES axis.
    description: |
      Each axis may be omitted (no preference) or set to one of its label
      keys. Use `/api/lyrics/axes` to fetch the available axis/label catalog.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [targets]
            properties:
              targets:
                type: object
                additionalProperties:
                  type: string
                example:
                  AXIS_1_SETTING: URBAN
                  AXIS_3_EMOTIONAL_VALENCE: MELANCHOLIC
              limit:
                type: integer
                minimum: 1
                maximum: 500
                default: 50
    responses:
      200:
        description: Matching tracks.
        content:
          application/json:
            schema:
              type: object
              properties:
                results:
                  type: array
                  items:
                    type: object
                count:
                  type: integer
      400:
        description: Lyrics disabled, missing/invalid targets, or invalid limit.
      404:
        description: No matching lyrics.
      500:
        description: Internal error.
    """
    from config import LYRICS_ENABLED
    from tasks.lyrics_manager import search_by_axes

    if not LYRICS_ENABLED:
        return jsonify({'error': 'Lyrics search is disabled.', 'results': []}), 400

    try:
        data = request.get_json() or {}
        targets_raw = data.get('targets') or {}
        if not isinstance(targets_raw, dict) or not targets_raw:
            return jsonify({'error': 'Missing or empty "targets" object.'}), 400

        # Accept only {axis: label_str}; reject anything else.
        targets: dict = {}
        for axis_name, value in targets_raw.items():
            if isinstance(value, str) and value.strip():
                targets[axis_name] = value.strip()
        if not targets:
            return jsonify({'error': 'No valid axis selections supplied.'}), 400

        try:
            limit = int(data.get('limit', 50))
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid "limit" value.'}), 400
        limit = min(max(1, limit), 500)

        results = search_by_axes(targets, limit=limit)
        if not results:
            return jsonify({'error': 'No lyrics found.', 'results': []}), 404
        attach_song_features(results)
        return jsonify({'results': results, 'count': len(results)})
    except Exception:
        logger.exception("Lyrics axis search failed")
        return jsonify({'error': 'An internal error occurred.'}), 500


@lyrics_search_bp.route('/api/lyrics/search/text', methods=['POST'])
def lyrics_search_text_api():
    """
    Lyrics search by free-form text.
    ---
    tags:
      - Lyrics Search
    summary: Embed the query with e5-base-v2 and find nearest neighbors in the lyrics voyager index.
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
                example: "songs about heartbreak in the rain"
              limit:
                type: integer
                minimum: 1
                maximum: 500
                default: 50
    responses:
      200:
        description: Matching tracks.
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                results:
                  type: array
                  items:
                    type: object
                count:
                  type: integer
      400:
        description: Lyrics disabled, missing query, or query too short.
      404:
        description: No matching lyrics.
      500:
        description: Internal error.
    """
    from config import LYRICS_ENABLED
    from tasks.lyrics_manager import search_by_text

    if not LYRICS_ENABLED:
        return jsonify({'error': 'Lyrics search is disabled.', 'results': []}), 400

    try:
        data = request.get_json() or {}
        query = (data.get('query') or '').strip()
        if not query:
            return jsonify({'error': 'Missing "query".'}), 400
        if len(query) < 3:
            return jsonify({'error': 'Query must be at least 3 characters.'}), 400

        try:
            limit = int(data.get('limit', 50))
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid "limit" value.'}), 400
        limit = min(max(1, limit), 500)

        results = search_by_text(query, limit=limit)
        if not results:
            return jsonify({'error': 'No lyrics found.', 'query': query, 'results': []}), 404
        attach_song_features(results)
        return jsonify({'query': query, 'results': results, 'count': len(results)})
    except Exception:
        logger.exception("Lyrics text search failed")
        return jsonify({'error': 'An internal error occurred.'}), 500


@lyrics_search_bp.route('/api/lyrics/cache/refresh', methods=['POST'])
def lyrics_refresh_cache_api():
    """
    Refresh the lyrics voyager index.
    ---
    tags:
      - Lyrics Search
    summary: Rebuild the lyrics cache and index from the database.
    responses:
      200:
        description: Refresh attempted; updated stats returned.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                stats:
                  type: object
      400:
        description: Lyrics disabled.
      500:
        description: Internal error.
    """
    from config import LYRICS_ENABLED
    from tasks.lyrics_manager import get_cache_stats, refresh_lyrics_cache

    if not LYRICS_ENABLED:
        return jsonify({'error': 'Lyrics is disabled.'}), 400

    try:
        success = refresh_lyrics_cache()
        return jsonify({'success': success, 'stats': get_cache_stats()})
    except Exception:
        logger.exception("Lyrics cache refresh failed")
        return jsonify({'success': False, 'error': 'Internal error.'}), 500


@lyrics_search_bp.route('/api/lyrics/stats', methods=['GET'])
def lyrics_stats_api():
    """
    Lyrics cache stats.
    ---
    tags:
      - Lyrics Search
    summary: Return cache size, freshness, and the LYRICS_ENABLED flag.
    responses:
      200:
        description: Cache statistics.
        content:
          application/json:
            schema:
              type: object
              properties:
                lyrics_enabled:
                  type: boolean
                num_embeddings:
                  type: integer
                last_refresh:
                  type: string
    """
    from config import LYRICS_ENABLED
    from tasks.lyrics_manager import get_cache_stats

    stats = get_cache_stats()
    stats['lyrics_enabled'] = LYRICS_ENABLED
    return jsonify(stats)


@lyrics_search_bp.route('/api/lyrics/axes', methods=['GET'])
def lyrics_axes_api():
    """
    Lyrics axis catalog.
    ---
    tags:
      - Lyrics Search
    summary: Return MUSIC_ANALYSIS_AXES so the UI can build sliders dynamically.
    responses:
      200:
        description: Axis catalog (empty when lyrics features are disabled).
        content:
          application/json:
            schema:
              type: object
              properties:
                axes:
                  type: object
                  additionalProperties:
                    type: array
                    items:
                      type: string
    """
    from config import LYRICS_ENABLED
    from tasks.lyrics_manager import get_axes_definition

    if not LYRICS_ENABLED:
        return jsonify({'axes': {}})
    return jsonify({'axes': get_axes_definition()})
