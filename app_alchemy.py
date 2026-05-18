from flask import Blueprint, jsonify, request, render_template
import logging

from tasks.song_alchemy import song_alchemy
import config

logger = logging.getLogger(__name__)

alchemy_bp = Blueprint('alchemy_bp', __name__, template_folder='../templates')


@alchemy_bp.route('/alchemy', methods=['GET'])
def alchemy_page():
    """
    Song Alchemy UI page.
    ---
    tags:
      - Alchemy
    summary: HTML page for blending songs/artists into a centroid-based recommendation set.
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template('alchemy.html', title = 'AudioMuse-AI - Song Alchemy', active='alchemy')


@alchemy_bp.route('/api/search_artists', methods=['GET'])
def search_artists():
    """
    Artist autocomplete.
    ---
    tags:
      - Alchemy
    summary: Search artists by partial name for autocomplete suggestions.
    parameters:
      - name: query
        in: query
        schema: { type: string }
        description: Partial artist name.
      - name: start
        in: query
        schema: { type: integer, default: 0 }
        description: 0-based pagination start.
      - name: end
        in: query
        schema: { type: integer }
        description: Exclusive pagination end. Default returns 20 items.
    responses:
      200:
        description: List of matching artists.
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
    """
    from tasks.artist_gmm_manager import search_artists_by_name
    
    query = request.args.get('query', '')

    # Pagination: start / end (0-based). Defaults to first 20 results.
    start = request.args.get('start', 0, type=int)
    end = request.args.get('end', None, type=int)
    if start < 0:
        start = 0
    if end is not None and end <= start:
        return jsonify([])
    limit = (end - start) if end is not None else 20
    offset = start

    try:
        results = search_artists_by_name(query, limit=limit, offset=offset)
        return jsonify(results)
    except Exception as e:
        logger.exception("Artist search failed")
        return jsonify([]), 200  # Return empty list on error


@alchemy_bp.route('/api/alchemy', methods=['POST'])
def alchemy_api():
    """
    Run a Song Alchemy blend.
    ---
    tags:
      - Alchemy
    summary: Combine ADD/SUBTRACT items into a centroid and return the nearest songs.
    description: |
      At least one ADD item (song or artist) is required. SUBTRACT items are
      optional and pull the centroid away from those songs/artists.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              items:
                type: array
                items:
                  type: object
                  required: [id, op]
                  properties:
                    id:
                      type: string
                    op:
                      type: string
                      enum: [ADD, SUBTRACT]
                    type:
                      type: string
                      enum: [song, artist]
                      default: song
              n:
                type: integer
                description: Number of results to return. Defaults to ALCHEMY_DEFAULT_N_RESULTS.
              temperature:
                type: number
                format: float
                description: Softmax temperature for probabilistic sampling. Defaults to ALCHEMY_TEMPERATURE.
              subtract_distance:
                type: number
                format: float
                description: Optional override for the SUBTRACT exclusion radius.
    responses:
      200:
        description: Recommendation results (each row contains the song and its centroid for save-as-anchor).
      400:
        description: Validation error (no ADD items, malformed payload).
      500:
        description: Internal error.
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
        # Keep full centroid in response for client-side save action, but not in anchor list endpoint.
        return jsonify(results)
    except ValueError as e:
        # Log the validation error server-side but do not expose internal error text to clients
        logger.exception("Alchemy validation failure")
        return jsonify({"error": "Invalid request"}), 400
    except Exception as e:
        logger.exception("Alchemy failure")
        return jsonify({"error": "Internal error"}), 500


@alchemy_bp.route('/api/anchors', methods=['GET'])
def list_anchors():
    """
    List saved alchemy anchors.
    ---
    tags:
      - Alchemy
    summary: Return id+name of every saved alchemy anchor (centroids omitted for size).
    responses:
      200:
        description: Anchor list.
        content:
          application/json:
            schema:
              type: object
              properties:
                anchors:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      name:
                        type: string
      500:
        description: Database error.
    """
    from app_helper import get_alchemy_anchors
    try:
        anchors = get_alchemy_anchors()
        # no centroid returned here (name-only list)
        return jsonify({'anchors': [{'id': a['id'], 'name': a['name']} for a in anchors]})
    except Exception:
        logger.exception('Failed to list anchors')
        return jsonify({'anchors': [], 'error': 'Unable to retrieve anchors at this time.'}), 500


@alchemy_bp.route('/api/anchors', methods=['POST'])
def create_anchor():
    """
    Save a new alchemy anchor.
    ---
    tags:
      - Alchemy
    summary: Persist an anchor (named centroid) for later reuse in path-finding or alchemy.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [name, centroid]
            properties:
              name:
                type: string
              centroid:
                type: array
                items:
                  type: number
                  format: float
                description: Embedding vector representing the anchor.
    responses:
      200:
        description: Anchor saved.
      400:
        description: Missing or invalid name/centroid.
      500:
        description: Database failure.
    """
    from app_helper import save_alchemy_anchor
    payload = request.get_json() or {}
    name = (payload.get('name') or '').strip()
    centroid = payload.get('centroid')
    if not name:
        return jsonify({'error': 'Anchor name is required'}), 400
    if not centroid or not isinstance(centroid, list):
        return jsonify({'error': 'Anchor centroid is required and must be a list'}), 400
    anchor = save_alchemy_anchor(name, centroid)
    if not anchor:
        return jsonify({'error': 'Failed to save anchor'}), 500
    return jsonify({'anchor': {'id': anchor['id'], 'name': anchor['name']}})


@alchemy_bp.route('/api/anchors/<int:anchor_id>', methods=['DELETE'])
def remove_anchor(anchor_id):
    """
    Delete an alchemy anchor.
    ---
    tags:
      - Alchemy
    summary: Remove a saved anchor by id.
    parameters:
      - name: anchor_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Anchor deleted.
      404:
        description: Anchor not found.
    """
    from app_helper import delete_alchemy_anchor
    ok = delete_alchemy_anchor(anchor_id)
    if not ok:
        return jsonify({'error': 'Anchor not found'}), 404
    return jsonify({'deleted': True})


@alchemy_bp.route('/api/anchors/<int:anchor_id>', methods=['PUT'])
def rename_anchor(anchor_id):
    """
    Rename an alchemy anchor.
    ---
    tags:
      - Alchemy
    summary: Update the display name of a saved anchor.
    parameters:
      - name: anchor_id
        in: path
        required: true
        schema: { type: integer }
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [name]
            properties:
              name:
                type: string
    responses:
      200:
        description: Anchor renamed.
      400:
        description: Empty name.
      404:
        description: Anchor not found.
    """
    from app_helper import update_alchemy_anchor_name
    payload = request.get_json() or {}
    name = (payload.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'Anchor name is required'}), 400
    anchor = update_alchemy_anchor_name(anchor_id, name)
    if not anchor:
        return jsonify({'error': 'Anchor not found or rename failed'}), 404
    return jsonify({'anchor': {'id': anchor['id'], 'name': anchor['name']}})


@alchemy_bp.route('/api/artist_projections', methods=['GET'])
def artist_projections_api():
    """
    Precomputed artist component projections.
    ---
    tags:
      - Alchemy
    summary: Return cached 2D projections of artist GMM components for the artist map.
    responses:
      200:
        description: Component list (empty if cache is cold).
        content:
          application/json:
            schema:
              type: object
              properties:
                components:
                  type: array
                  items:
                    type: object
                    properties:
                      artist_id:
                        type: string
                      artist_name:
                        type: string
                      component_idx:
                        type: integer
                      weight:
                        type: number
                        format: float
                      projection:
                        type: array
                        items:
                          type: number
                          format: float
                        description: 2D x/y projection.
                count:
                  type: integer
      500:
        description: Failure to read cache.
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
    except Exception:
        logger.exception("Failed to retrieve artist projections")
        return jsonify({'components': [], 'count': 0, 'error': 'Unable to retrieve artist projections at this time.'}), 500


@alchemy_bp.route('/api/build_artist_projection', methods=['POST'])
def build_artist_projection_endpoint():
    """
    Rebuild artist component projections.
    ---
    tags:
      - Alchemy
    summary: Manually compute and store artist projections (requires GMM params already in DB).
    description: |
      Useful for rebuilding the artist map without running a full analysis.
      Returns 400 if no artist GMM parameters are present.
    responses:
      200:
        description: Projection rebuilt and cached.
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  enum: [success]
                message:
                  type: string
      400:
        description: No GMM parameters available.
      500:
        description: Build failed.
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
    except Exception:
        logger.exception("Failed to build artist projection")
        return jsonify({
            'status': 'error',
            'message': 'Failed to build artist projection. Please try again later.'
        }), 500
