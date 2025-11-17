from flask import Blueprint, jsonify, request, render_template
import logging

from tasks.song_alchemy import song_alchemy
import config

logger = logging.getLogger(__name__)

alchemy_bp = Blueprint('alchemy_bp', __name__, template_folder='../templates')


@alchemy_bp.route('/alchemy', methods=['GET'])
def alchemy_page():
    return render_template('alchemy.html', title = 'AudioMuse-AI - Song Alchemy', active='alchemy')


@alchemy_bp.route('/api/search_artists', methods=['GET'])
def search_artists():
    """Search for artists by name for autocomplete."""
    from tasks.artist_gmm_manager import search_artists_by_name
    
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 15))
    
    try:
        results = search_artists_by_name(query, limit)
        return jsonify(results)
    except Exception as e:
        logger.exception("Artist search failed")
        return jsonify([]), 200  # Return empty list on error


@alchemy_bp.route('/api/alchemy', methods=['POST'])
def alchemy_api():
    """POST payload: {"items": [{"id":"...","op":"ADD","type":"song/artist"}, ...], "n":100}
    Expect at least one ADD item (song or artist); SUBTRACT items are optional.
    """
    payload = request.get_json() or {}
    items = payload.get('items', [])
    n = payload.get('n', config.ALCHEMY_DEFAULT_N_RESULTS)
    # Temperature parameter for probabilistic sampling (softmax temperature)
    temperature = payload.get('temperature', config.ALCHEMY_TEMPERATURE)

    # Separate items by operation
    add_items = [{'type': i.get('type', 'song'), 'id': i['id']} for i in items if i.get('op', '').upper() == 'ADD' and i.get('id')]
    subtract_items = [{'type': i.get('type', 'song'), 'id': i['id']} for i in items if i.get('op', '').upper() == 'SUBTRACT' and i.get('id')]

    # Allow optional override for subtract distance (from frontend slider)
    subtract_distance = payload.get('subtract_distance')
    try:
        results = song_alchemy(add_items=add_items, subtract_items=subtract_items, n_results=n, subtract_distance=subtract_distance, temperature=temperature)
        # song_alchemy now returns a dict with results, filtered_out and centroid projections
        return jsonify(results)
    except ValueError as e:
        # Log the validation error server-side but do not expose internal error text to clients
        logger.exception("Alchemy validation failure")
        return jsonify({"error": "Invalid request"}), 400
    except Exception as e:
        logger.exception("Alchemy failure")
        return jsonify({"error": "Internal error"}), 500


@alchemy_bp.route('/api/artist_projections', methods=['GET'])
def artist_projections_api():
    """Return precomputed artist component projections.
    
    Response: JSON with:
    {
        "components": [
            {
                "artist_id": "...",
                "artist_name": "...",
                "component_idx": 0,
                "weight": 0.35,
                "projection": [x, y]
            },
            ...
        ],
        "count": N
    }
    """
    from app_helper import ARTIST_PROJECTION_CACHE
    
    try:
        if not ARTIST_PROJECTION_CACHE:
            return jsonify({'components': [], 'count': 0})
        
        component_map = ARTIST_PROJECTION_CACHE.get('component_map', [])
        projection = ARTIST_PROJECTION_CACHE.get('projection')
        
        if projection is None or len(component_map) == 0:
            return jsonify({'components': [], 'count': 0})
        
        # Build response with components and their 2D projections
        components = []
        for idx, comp_info in enumerate(component_map):
            if idx < len(projection):
                components.append({
                    'artist_id': comp_info['artist_id'],
                    'artist_name': comp_info.get('artist_name', comp_info['artist_id']),
                    'component_idx': comp_info['component_idx'],
                    'weight': comp_info['weight'],
                    'projection': [float(projection[idx][0]), float(projection[idx][1])]
                })
        
        return jsonify({
            'components': components,
            'count': len(components)
        })
    except Exception as e:
        logger.exception("Failed to retrieve artist projections")
        return jsonify({'components': [], 'count': 0, 'error': str(e)}), 500


@alchemy_bp.route('/api/build_artist_projection', methods=['POST'])
def build_artist_projection_endpoint():
    """Manually trigger artist component projection build.
    
    This is useful for rebuilding artist projections without running full analysis.
    Requires that artist GMM parameters already exist in the database.
    
    Response: JSON with status ('success' or 'error') and message
    """
    from app_helper import build_and_store_artist_projection
    
    try:
        success = build_and_store_artist_projection('artist_map')
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Artist component projection built and stored successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Artist projection build returned no data (no GMM parameters found?)'
            }), 400
    except Exception as e:
        logger.exception("Failed to build artist projection")
        return jsonify({
            'status': 'error',
            'message': f'Failed to build artist projection: {str(e)}'
        }), 500
