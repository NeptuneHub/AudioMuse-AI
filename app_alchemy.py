from flask import Blueprint, jsonify, request, render_template
import logging
import math

from tasks.song_alchemy import song_alchemy
from app_helper import attach_song_features
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
        attach_song_features(results.get('results'))
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
    from database import get_alchemy_anchors
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
    from database import save_alchemy_anchor
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
    from database import delete_alchemy_anchor
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
    from database import update_alchemy_anchor_name
    payload = request.get_json() or {}
    name = (payload.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'Anchor name is required'}), 400
    anchor = update_alchemy_anchor_name(anchor_id, name)
    if not anchor:
        return jsonify({'error': 'Anchor not found or rename failed'}), 404
    return jsonify({'anchor': {'id': anchor['id'], 'name': anchor['name']}})


def _parse_radio_settings(payload):
    temperature = payload.get('temperature')
    n_results = payload.get('n_results')
    if temperature is None:
        return None, None, 'Radio temperature is required'
    if n_results is None:
        return None, None, 'Radio number of results is required'
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        return None, None, 'Radio temperature must be a number'
    if not math.isfinite(temperature):
        return None, None, 'Radio temperature must be a finite number'
    try:
        n_results = int(n_results)
    except (TypeError, ValueError):
        return None, None, 'Radio number of results must be an integer'
    if temperature < 0:
        return None, None, 'Radio temperature must be 0 or greater'
    if n_results < 1 or n_results > config.ALCHEMY_MAX_N_RESULTS:
        return None, None, f'Radio number of results must be between 1 and {config.ALCHEMY_MAX_N_RESULTS}'
    return temperature, n_results, None


@alchemy_bp.route('/api/radios', methods=['GET'])
def list_radios():
    """
    List saved alchemy radios.
    ---
    tags:
      - Alchemy
    summary: Return every saved radio (anchor + temperature + number of results) with its enabled state.
    responses:
      200:
        description: Radio list.
        content:
          application/json:
            schema:
              type: object
              properties:
                radios:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      anchor_id:
                        type: integer
                      name:
                        type: string
                        description: Name of the underlying anchor (the radio shares it).
                      temperature:
                        type: number
                        format: float
                      n_results:
                        type: integer
                      enabled:
                        type: boolean
      500:
        description: Database error.
    """
    from database import get_alchemy_radios
    try:
        radios = get_alchemy_radios()
        return jsonify({'radios': [{
            'id': r['id'], 'anchor_id': r['anchor_id'], 'name': r['name'],
            'temperature': r['temperature'], 'n_results': r['n_results'], 'enabled': bool(r['enabled'])
        } for r in radios]})
    except Exception:
        logger.exception('Failed to list radios')
        return jsonify({'radios': [], 'error': 'Unable to retrieve radios at this time.'}), 500


@alchemy_bp.route('/api/radios', methods=['POST'])
def create_radio():
    """
    Save a new alchemy radio.
    ---
    tags:
      - Alchemy
    summary: Persist a radio (anchor + temperature + number of results) for batch playlist generation.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [anchor_id, temperature, n_results]
            properties:
              anchor_id:
                type: integer
                description: Saved anchor the radio is built on (one radio per anchor).
              temperature:
                type: number
                format: float
              n_results:
                type: integer
              enabled:
                type: boolean
                default: true
    responses:
      200:
        description: Radio saved.
      400:
        description: Missing or invalid anchor/temperature/number of results.
      500:
        description: Database failure.
    """
    from database import create_alchemy_radio
    payload = request.get_json() or {}
    anchor_id = payload.get('anchor_id')
    try:
        anchor_id = int(anchor_id)
    except (TypeError, ValueError):
        return jsonify({'error': 'Radio anchor is required'}), 400
    temperature, n_results, error = _parse_radio_settings(payload)
    if error:
        return jsonify({'error': error}), 400
    enabled = bool(payload.get('enabled', True))
    radio = create_alchemy_radio(anchor_id, temperature, n_results, enabled)
    if not radio:
        return jsonify({'error': 'Failed to save radio. Check that the anchor exists and has no radio yet.'}), 400
    return jsonify({'radio': radio})


@alchemy_bp.route('/api/radios/<int:radio_id>', methods=['PUT'])
def update_radio(radio_id):
    """
    Update an alchemy radio.
    ---
    tags:
      - Alchemy
    summary: Update temperature, number of results and enabled state of a saved radio.
    parameters:
      - name: radio_id
        in: path
        required: true
        schema: { type: integer }
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [temperature, n_results, enabled]
            properties:
              temperature:
                type: number
                format: float
              n_results:
                type: integer
              enabled:
                type: boolean
    responses:
      200:
        description: Radio updated.
      400:
        description: Invalid temperature/number of results.
      404:
        description: Radio not found.
    """
    from database import update_alchemy_radio
    payload = request.get_json() or {}
    temperature, n_results, error = _parse_radio_settings(payload)
    if error:
        return jsonify({'error': error}), 400
    enabled = bool(payload.get('enabled', True))
    radio = update_alchemy_radio(radio_id, temperature, n_results, enabled)
    if not radio:
        return jsonify({'error': 'Radio not found or update failed'}), 404
    return jsonify({'radio': radio})


@alchemy_bp.route('/api/radios/<int:radio_id>', methods=['DELETE'])
def remove_radio(radio_id):
    """
    Delete an alchemy radio.
    ---
    tags:
      - Alchemy
    summary: Remove a saved radio by id (the underlying anchor is kept).
    parameters:
      - name: radio_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Radio deleted.
      404:
        description: Radio not found.
    """
    from database import delete_alchemy_radio
    ok = delete_alchemy_radio(radio_id)
    if not ok:
        return jsonify({'error': 'Radio not found'}), 404
    return jsonify({'deleted': True})


@alchemy_bp.route('/api/radios/run', methods=['POST'])
def run_radio_playlists_endpoint():
    """
    Create playlists for all enabled radios.
    ---
    tags:
      - Alchemy
    summary: Upsert one playlist per enabled radio (reuses existing playlist by name, preserving its server-side ID).
    responses:
      200:
        description: Run summary.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                radios_enabled:
                  type: integer
                playlists_created:
                  type: integer
                failed:
                  type: array
                  items:
                    type: string
      500:
        description: Run failed.
    """
    from tasks.radio_manager import run_radio_playlists
    try:
        summary = run_radio_playlists()
        return jsonify(summary)
    except Exception:
        logger.exception('Radio playlist creation failed')
        return jsonify({'error': 'Failed to create radio playlists. Check container logs.'}), 500


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
    from database import ARTIST_PROJECTION_CACHE
    
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
