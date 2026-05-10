"""
MuLan Text Search Blueprint
Provides web interface and API for natural language music search using MuLan.
"""

from flask import Blueprint, render_template, request, jsonify
import logging

logger = logging.getLogger(__name__)

mulan_search_bp = Blueprint('mulan_search_bp', __name__, template_folder='../templates')


@mulan_search_bp.route('/mulan_search', methods=['GET'])
def mulan_search_page():
    """
    MuLan text search UI page.
    ---
    tags:
      - MuLan Search
    summary: HTML page for natural-language music search powered by MuLan audio↔text embeddings.
    responses:
      200:
        description: HTML page rendered.
    """
    from config import MULAN_ENABLED, APP_VERSION
    from tasks.mulan_text_search import get_cache_stats
    
    cache_stats = get_cache_stats()
    
    return render_template(
        'mulan_search.html',
        title='MuLan Text Search - AudioMuse-AI',
        active='mulan_search',
        app_version=APP_VERSION,
        mulan_enabled=MULAN_ENABLED,
        cache_stats=cache_stats
    )


@mulan_search_bp.route('/api/mulan/search', methods=['POST'])
def mulan_search_api():
    """
    Run a MuLan text-to-audio similarity search.
    ---
    tags:
      - MuLan Search
    summary: Return the top-N tracks whose MuLan audio embedding best matches a free-text query.
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
        description: MuLan disabled, missing query, or query too short.
      500:
        description: Internal error during search.
      503:
        description: MuLan cache not loaded yet.
    """
    from config import MULAN_ENABLED
    from tasks.mulan_text_search import search_by_text, is_mulan_cache_loaded
    
    if not MULAN_ENABLED:
        return jsonify({
            'error': 'MuLan text search is disabled. Set MULAN_ENABLED=true in config.',
            'results': []
        }), 400
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" in request body'}), 400
        
        query = data['query'].strip()
        limit = data.get('limit', 100)
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if len(query) < 3:
            return jsonify({'error': 'Query must be at least 3 characters'}), 400
        
        # Validate limit
        limit = min(max(1, int(limit)), 500)  # Between 1 and 500
        
        # Check if cache is loaded
        if not is_mulan_cache_loaded():
            return jsonify({
                'error': 'MuLan cache not loaded. Please run song analysis first.',
                'results': []
            }), 503
        
        # Perform search
        results = search_by_text(query, limit=limit)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except ValueError as e:
        logger.warning(f"ValueError in MuLan search API: {e}")
        return jsonify({'error': 'Invalid or missing request parameter.'}), 400
    except Exception as e:
        logger.error(f"MuLan search API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred during MuLan search.'}), 500


@mulan_search_bp.route('/api/mulan/warmup', methods=['POST'])
def warmup_model_api():
    """
    Warm up the MuLan text-search model.
    ---
    tags:
      - MuLan Search
    summary: Preload the MuLan models and reset the idle-eviction timer.
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
        description: MuLan search is disabled.
      500:
        description: Warmup failed.
    """
    from config import MULAN_ENABLED
    from tasks.mulan_text_search import warmup_text_search_model
    
    if not MULAN_ENABLED:
        return jsonify({
            'error': 'MuLan text search is disabled',
            'loaded': False
        }), 400
    
    try:
        status = warmup_text_search_model()
        return jsonify(status)
    except Exception as e:
        logger.error(f"MuLan model warmup failed: {e}")
        return jsonify({
            'error': str(e),
            'loaded': False
        }), 500


@mulan_search_bp.route('/api/mulan/warmup/status', methods=['GET'])
def warmup_status_api():
    """
    MuLan warmup status.
    ---
    tags:
      - MuLan Search
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
    from config import MULAN_ENABLED
    from tasks.mulan_text_search import get_warm_cache_status
    
    if not MULAN_ENABLED:
        return jsonify({'active': False, 'seconds_remaining': 0})
    
    try:
        status = get_warm_cache_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Failed to get MuLan warmup status: {e}")
        return jsonify({'active': False, 'seconds_remaining': 0})


@mulan_search_bp.route('/api/mulan/cache/refresh', methods=['POST'])
def refresh_cache_api():
    """
    Refresh the MuLan audio-embedding cache.
    ---
    tags:
      - MuLan Search
    summary: Reload the MuLan cache from the database (call after analysis completes).
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
        description: MuLan disabled.
      500:
        description: Refresh failed.
    """
    from config import MULAN_ENABLED
    from tasks.mulan_text_search import refresh_mulan_cache, get_cache_stats
    
    if not MULAN_ENABLED:
        return jsonify({'error': 'MuLan is disabled'}), 400
    
    try:
        success = refresh_mulan_cache()
        stats = get_cache_stats()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'MuLan cache refreshed successfully',
                'stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to refresh MuLan cache',
                'stats': stats
            }), 500
            
    except Exception as e:
        logger.error(f"MuLan cache refresh failed: {e}")
        return jsonify({
            'success': False,
            'error': 'An internal error occurred. Please try again later.'
        }), 500


@mulan_search_bp.route('/api/mulan/stats', methods=['GET'])
def cache_stats_api():
    """
    MuLan cache stats.
    ---
    tags:
      - MuLan Search
    summary: Return cache size, freshness, and the MULAN_ENABLED flag.
    responses:
      200:
        description: Cache statistics.
        content:
          application/json:
            schema:
              type: object
              properties:
                mulan_enabled:
                  type: boolean
                num_embeddings:
                  type: integer
                last_refresh:
                  type: string
    """
    from config import MULAN_ENABLED
    from tasks.mulan_text_search import get_cache_stats
    
    stats = get_cache_stats()
    stats['mulan_enabled'] = MULAN_ENABLED
    
    return jsonify(stats)


@mulan_search_bp.route('/api/mulan/top_queries', methods=['GET'])
def top_queries_api():
    """
    Top diverse MuLan queries.
    ---
    tags:
      - MuLan Search
    summary: Return the precomputed top-50 diverse search queries shown as suggestions.
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
    from config import MULAN_ENABLED
    from tasks.mulan_text_search import get_cached_top_queries
    
    if not MULAN_ENABLED:
        return jsonify({'queries': [], 'ready': False, 'message': 'MuLan disabled'}), 200
    
    try:
        queries = get_cached_top_queries()
        return jsonify({
            'queries': queries,
            'ready': len(queries) > 0
        }), 200
    except Exception as e:
        logger.exception("Failed to get MuLan top queries")
        return jsonify({'error': 'An internal error occurred. Please try again later.', 'queries': [], 'ready': False}), 500
