"""
CLAP Text Search Blueprint
Provides web interface and API for natural language music search.
"""

from flask import Blueprint, render_template, request, jsonify
import logging

logger = logging.getLogger(__name__)

clap_search_bp = Blueprint('clap_search_bp', __name__, template_folder='../templates')


@clap_search_bp.route('/clap_search', methods=['GET'])
def clap_search_page():
    """Render CLAP text search page."""
    from config import CLAP_ENABLED, APP_VERSION
    from tasks.clap_text_search import get_cache_stats
    
    cache_stats = get_cache_stats()
    
    return render_template(
        'clap_search.html',
        title='Text Search - AudioMuse-AI',
        active='clap_search',
        app_version=APP_VERSION,
        clap_enabled=CLAP_ENABLED,
        cache_stats=cache_stats
    )


@clap_search_bp.route('/api/clap/search', methods=['POST'])
def clap_search_api():
    """
    API endpoint for CLAP text search.
    
    POST JSON:
    {
        "query": "upbeat summer songs",
        "limit": 100
    }
    
    Returns:
    {
        "query": "upbeat summer songs",
        "results": [
            {
                "item_id": "123",
                "title": "Song Title",
                "author": "Artist Name",
                "similarity": 0.85
            },
            ...
        ],
        "count": 100
    }
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import search_by_text, is_clap_cache_loaded
    
    if not CLAP_ENABLED:
        return jsonify({
            'error': 'CLAP text search is disabled. Set CLAP_ENABLED=true in config.',
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
        if not is_clap_cache_loaded():
            return jsonify({
                'error': 'CLAP cache not loaded. Please run song analysis first.',
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
        logger.warning(f"ValueError in CLAP search API: {e}")
        return jsonify({'error': 'Invalid or missing request parameter.'}), 400
    except Exception as e:
        logger.error(f"CLAP search API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred during CLAP search.'}), 500


@clap_search_bp.route('/api/clap/warmup', methods=['POST'])
def warmup_model_api():
    """
    API endpoint to preload CLAP model and start/reset 10-minute timer.
    Call this when the search page loads to ensure fast searches.
    
    Returns:
    {
        "loaded": true,
        "expiry_seconds": 600
    }
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import warmup_text_search_model
    
    if not CLAP_ENABLED:
        return jsonify({
            'error': 'CLAP text search is disabled',
            'loaded': False
        }), 400
    
    try:
        status = warmup_text_search_model()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        return jsonify({
            'error': str(e),
            'loaded': False
        }), 500


@clap_search_bp.route('/api/clap/warmup/status', methods=['GET'])
def warmup_status_api():
    """
    Get current warm cache status.
    
    Returns:
    {
        "active": true,
        "seconds_remaining": 423
    }
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import get_warm_cache_status
    
    if not CLAP_ENABLED:
        return jsonify({'active': False, 'seconds_remaining': 0})
    
    try:
        status = get_warm_cache_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Failed to get warmup status: {e}")
        return jsonify({'active': False, 'seconds_remaining': 0})


@clap_search_bp.route('/api/clap/cache/refresh', methods=['POST'])
def refresh_cache_api():
    """Refresh CLAP cache from database."""
    from config import CLAP_ENABLED
    from tasks.clap_text_search import refresh_clap_cache, get_cache_stats
    
    if not CLAP_ENABLED:
        return jsonify({'error': 'CLAP is disabled'}), 400
    
    try:
        success = refresh_clap_cache()
        stats = get_cache_stats()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'CLAP cache refreshed successfully',
                'stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to refresh CLAP cache',
                'stats': stats
            }), 500
            
    except Exception as e:
        logger.error(f"Cache refresh failed: {e}")
        return jsonify({
            'success': False,
            'error': 'An internal error occurred. Please try again later.'
        }), 500


@clap_search_bp.route('/api/clap/stats', methods=['GET'])
def cache_stats_api():
    """Get CLAP cache statistics."""
    from config import CLAP_ENABLED
    from tasks.clap_text_search import get_cache_stats
    
    stats = get_cache_stats()
    stats['clap_enabled'] = CLAP_ENABLED
    
    return jsonify(stats)


@clap_search_bp.route('/api/clap/top_queries', methods=['GET'])
def top_queries_api():
    """
    Return precomputed top 50 diverse queries.
    Returns empty array if not ready yet (still computing in background).
    """
    from config import CLAP_ENABLED
    from tasks.clap_text_search import get_cached_top_queries
    
    if not CLAP_ENABLED:
        return jsonify({'queries': [], 'ready': False, 'message': 'CLAP disabled'}), 200
    
    try:
        queries = get_cached_top_queries()
        return jsonify({
            'queries': queries,
            'ready': len(queries) > 0
        }), 200
    except Exception as e:
        logger.exception("Failed to get top queries")
        return jsonify({'error': 'An internal error occurred. Please try again later.', 'queries': [], 'ready': False}), 500

