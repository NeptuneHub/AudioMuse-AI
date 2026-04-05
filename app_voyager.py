# app_voyager.py
from flask import Blueprint, jsonify, request, render_template
import logging
import json
import numpy as np

# Import the new config option
from config import SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT, SIMILARITY_RADIUS_DEFAULT, MOOD_CENTROIDS_FILE
from tasks.voyager_manager import (
    find_nearest_neighbors_by_id, 
    find_nearest_neighbors_by_vector,
    get_max_distance_for_id,
    create_playlist_from_ids,
    search_tracks_unified,
    get_item_id_by_title_and_artist
)

logger = logging.getLogger(__name__)

# --- Load mood centroids at module level ---
_MOOD_CENTROIDS_DATA = {}  # mood_name -> list of centroid dicts (with vectors)
_MOOD_CENTROIDS_META = {}  # mood_name -> list of {cluster_id, top_tags (top 3)} for API

def _load_mood_centroids_for_similarity():
    global _MOOD_CENTROIDS_DATA, _MOOD_CENTROIDS_META
    try:
        with open(MOOD_CENTROIDS_FILE) as f:
            data = json.load(f)
        for mood, info in data.items():
            centroids = info.get('centroids', [])
            _MOOD_CENTROIDS_DATA[mood] = centroids
            meta_list = []
            for i, c in enumerate(centroids):
                tags = c.get('top_tags', {})
                top5 = sorted(tags.items(), key=lambda x: -x[1])[:5]
                meta_list.append({
                    'index': i,
                    'top_tags': [t[0] for t in top5],
                    'n_songs': c.get('n_songs', 0),
                    'mood_score': c.get('mood_score', 0),
                    'cluster_id': c.get('cluster_id', i),
                })
            _MOOD_CENTROIDS_META[mood] = meta_list
        logger.info(f"Loaded mood centroids for similarity: {', '.join(f'{m}({len(cs)})' for m, cs in _MOOD_CENTROIDS_DATA.items())}")
    except Exception as e:
        logger.warning(f"Could not load mood centroids from {MOOD_CENTROIDS_FILE}: {e}")

_load_mood_centroids_for_similarity()

# Create a Blueprint for Voyager (similarity) related routes
voyager_bp = Blueprint('voyager_bp', __name__, template_folder='../templates')

@voyager_bp.route('/similarity', methods=['GET'])
def similarity_page():
    """
    Serves the frontend page for finding similar tracks.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the similarity page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('similarity.html', title = 'AudioMuse-AI - Playlist from Similar Song', active='similarity')

@voyager_bp.route('/api/search_tracks', methods=['GET'])
def search_tracks_endpoint():
    """
    Provides autocomplete suggestions for tracks based on title and artist.
    ---
    tags:
      - Similarity
    parameters:
      - name: search_query
        in: query
        description: Partial or full elements of songs' titles, artist or album names.
        schema:
          type: string
      - name: title
        in: query
        description: (Legacy) Partial or full title of the track. Used as fallback when search_query is absent.
        schema:
          type: string
      - name: artist
        in: query
        description: (Legacy) Partial or full name of the artist. Used as fallback when search_query is absent.
        schema:
          type: string
    responses:
      200:
        description: A list of matching tracks.
        content:
          application/json:
            schema:
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
                  album:
                    type: string
                    description: Album name or 'unknown' if missing
    """
    search_query = request.args.get('search_query', '', type=str)

    # Backward compatibility: support legacy 'title' and 'artist' params
    # so external apps using the old API continue to work.
    if not search_query:
        legacy_title = request.args.get('title', '', type=str).strip()
        legacy_artist = request.args.get('artist', '', type=str).strip()
        search_query = f"{legacy_artist} {legacy_title}".strip()

    if not search_query:
        return jsonify([])

    if len(search_query) < 3:
        return jsonify([])

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
        raw_results = search_tracks_unified(search_query, limit=limit, offset=offset)
        results = []
        for r in raw_results:
            # Be defensive in case the source returns non-dict entries
            if isinstance(r, dict):
                album = (r.get('album') or '').strip() or 'unknown'
                results.append({
                    'item_id': r.get('item_id'),
                    'title': r.get('title'),
                    'author': r.get('author'),
                    'album': album,
                    'album_artist': (r.get('album_artist') or '').strip() or 'unknown'
                })
            else:
                results.append({'item_id': None, 'title': None, 'author': None, 'album': 'unknown'})
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error during track search: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during search."}), 500


@voyager_bp.route('/api/mood_centroids', methods=['GET'])
def get_mood_centroids_endpoint():
    """
    Returns available mood categories and their centroids (top 3 tags only, no vectors).
    Optionally filter by mood name.
    ---
    tags:
      - Similarity
    parameters:
      - name: mood
        in: query
        description: Optional mood name to filter centroids for a specific mood.
        schema:
          type: string
    responses:
      200:
        description: Dictionary of mood names to lists of centroid metadata.
    """
    mood_filter = request.args.get('mood', '', type=str).strip().lower()
    if mood_filter:
        if mood_filter not in _MOOD_CENTROIDS_META:
            return jsonify({"error": f"Unknown mood '{mood_filter}'. Available: {list(_MOOD_CENTROIDS_META.keys())}"}), 400
        return jsonify({mood_filter: _MOOD_CENTROIDS_META[mood_filter]})
    return jsonify(_MOOD_CENTROIDS_META)


@voyager_bp.route('/api/similar_tracks', methods=['GET'])
def get_similar_tracks_endpoint():
    """
    Find similar tracks for a given track, identified either by item_id or title/artist.
    ---
    tags:
      - Similarity
    parameters:
      - name: item_id
        in: query
        description: The media server Item ID of the track. Use this OR title/artist.
        schema:
          type: string
      - name: title
        in: query
        description: The title of the track. Must be used with 'artist'.
        schema:
          type: string
      - name: artist
        in: query
        description: The artist of the track. Must be used with 'title'.
        schema:
          type: string
      - name: n
        in: query
        description: The number of similar tracks to return.
        schema:
          type: integer
          default: 10
      - name: eliminate_duplicates
        in: query
        description: If 'true', limits the number of songs per artist in the results. If 'false', this is disabled. If the parameter is omitted, the server's default behavior is used.
        schema:
          type: string
          enum: ['true', 'false']
      - name: mood_similarity
        in: query
        description: If 'true', filters results by mood similarity using stored mood features (danceability, aggressive, happy, party, relaxed, sad). If 'false', only acoustic similarity is used. Defaults to 'true' if omitted.
        schema:
          type: string
          enum: ['true', 'false']
    responses:
      200:
        description: A list of similar tracks with their details.
        content:
          application/json:
            schema:
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
                  album:
                    type: string
                    description: Album name or 'unknown' if missing
                  distance:
                    type: number
      400:
        description: Bad request, missing required parameters.
      404:
        description: Target track not found.
      500:
        description: Server error.
    """
    item_id = request.args.get('item_id')
    title = request.args.get('title')
    artist = request.args.get('artist')
    num_neighbors = request.args.get('n', 10, type=int)

    # Optional mood centroid parameters
    mood_param = request.args.get('mood', '', type=str).strip().lower()
    centroid_index_param = request.args.get('centroid_index', None, type=int)

    # Optional anchor parameter
    anchor_id_param = request.args.get('anchor_id', None, type=int)
    
    eliminate_duplicates_str = request.args.get('eliminate_duplicates')
    if eliminate_duplicates_str is None:
        eliminate_duplicates = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT
    else:
        eliminate_duplicates = eliminate_duplicates_str.lower() == 'true'

    radius_similarity_str = request.args.get('radius_similarity')
    if radius_similarity_str is None:
        # Use configured default when parameter is omitted
        radius_similarity = SIMILARITY_RADIUS_DEFAULT
    else:
        radius_similarity = radius_similarity_str.lower() == 'true'

    mood_similarity_str = request.args.get('mood_similarity')
    if mood_similarity_str is None:
        mood_similarity = None  # Respect config default when parameter is omitted
    else:
        mood_similarity = mood_similarity_str.lower() == 'true'

    # --- Mood centroid mode: use centroid vector instead of a song ---
    if mood_param and centroid_index_param is not None:
        if mood_param not in _MOOD_CENTROIDS_DATA:
            return jsonify({"error": f"Unknown mood '{mood_param}'. Available: {list(_MOOD_CENTROIDS_DATA.keys())}"}), 400
        centroids = _MOOD_CENTROIDS_DATA[mood_param]
        if centroid_index_param < 0 or centroid_index_param >= len(centroids):
            return jsonify({"error": f"Invalid centroid_index {centroid_index_param} for mood '{mood_param}' (0-{len(centroids)-1})."}), 400

        centroid_vector = np.array(centroids[centroid_index_param]['centroid'], dtype=np.float32)
        try:
            neighbor_results = find_nearest_neighbors_by_vector(
                centroid_vector,
                n=num_neighbors,
                eliminate_duplicates=eliminate_duplicates
            )
        except RuntimeError as e:
            logger.error(f"Runtime error finding neighbors for mood centroid: {e}", exc_info=True)
            return jsonify({"error": "The similarity search service is currently unavailable."}), 503
        except Exception as e:
            logger.error(f"Unexpected error finding neighbors for mood centroid: {e}", exc_info=True)
            return jsonify({"error": "An unexpected error occurred."}), 500

        if not neighbor_results:
            return jsonify({"error": "No similar tracks found for this mood centroid."}), 404

        from app import get_score_data_by_ids
        neighbor_ids = [n_item['item_id'] for n_item in neighbor_results]
        neighbor_details = get_score_data_by_ids(neighbor_ids)
        details_map = {d['item_id']: d for d in neighbor_details}
        distance_map = {n_item['item_id']: n_item['distance'] for n_item in neighbor_results}

        final_results = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in details_map:
                track_info = details_map[neighbor_id]
                final_results.append({
                    "item_id": track_info['item_id'],
                    "title": track_info['title'],
                    "author": track_info['author'],
                    "album": (track_info.get('album') or 'unknown'),
                    "album_artist": (track_info.get('album_artist') or 'unknown'),
                    "distance": distance_map[neighbor_id]
                })
        return jsonify(final_results)

    # --- Anchor mode: use anchor's centroid vector ---
    if anchor_id_param is not None:
        from app_helper import get_alchemy_anchor_by_id
        anchor = get_alchemy_anchor_by_id(anchor_id_param)
        if not anchor or not anchor.get('centroid'):
            return jsonify({"error": f"Anchor with id {anchor_id_param} not found or has no centroid."}), 404

        anchor_vector = np.array(anchor['centroid'], dtype=np.float32)
        try:
            neighbor_results = find_nearest_neighbors_by_vector(
                anchor_vector,
                n=num_neighbors,
                eliminate_duplicates=eliminate_duplicates
            )
        except RuntimeError as e:
            logger.error(f"Runtime error finding neighbors for anchor {anchor_id_param}: {e}", exc_info=True)
            return jsonify({"error": "The similarity search service is currently unavailable."}), 503
        except Exception as e:
            logger.error(f"Unexpected error finding neighbors for anchor {anchor_id_param}: {e}", exc_info=True)
            return jsonify({"error": "An unexpected error occurred."}), 500

        if not neighbor_results:
            return jsonify({"error": "No similar tracks found for this anchor."}), 404

        from app import get_score_data_by_ids
        neighbor_ids = [n_item['item_id'] for n_item in neighbor_results]
        neighbor_details = get_score_data_by_ids(neighbor_ids)
        details_map = {d['item_id']: d for d in neighbor_details}
        distance_map = {n_item['item_id']: n_item['distance'] for n_item in neighbor_results}

        final_results = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in details_map:
                track_info = details_map[neighbor_id]
                final_results.append({
                    "item_id": track_info['item_id'],
                    "title": track_info['title'],
                    "author": track_info['author'],
                    "album": (track_info.get('album') or 'unknown'),
                    "album_artist": (track_info.get('album_artist') or 'unknown'),
                    "distance": distance_map[neighbor_id]
                })
        return jsonify(final_results)

    # --- Standard song-based mode ---
    target_item_id = None

    if item_id:
        target_item_id = item_id
    elif title and artist:
        resolved_id = get_item_id_by_title_and_artist(title, artist)
        if not resolved_id:
            return jsonify({"error": f"Track '{title}' by '{artist}' not found in the database."}), 404
        target_item_id = resolved_id
    else:
        return jsonify({"error": "Request must include either 'item_id' or both 'title' and 'artist', or 'mood' and 'centroid_index'."}), 400

    try:
        neighbor_results = find_nearest_neighbors_by_id(
            target_item_id, 
            n=num_neighbors,
            eliminate_duplicates=eliminate_duplicates,
            mood_similarity=mood_similarity,
            radius_similarity=radius_similarity
        )
        if not neighbor_results:
            return jsonify({"error": "Target track not found in index or no similar tracks found."}), 404

        from app import get_score_data_by_ids

        neighbor_ids = [n['item_id'] for n in neighbor_results]
        neighbor_details = get_score_data_by_ids(neighbor_ids)

        details_map = {d['item_id']: d for d in neighbor_details}
        distance_map = {n['item_id']: n['distance'] for n in neighbor_results}

        final_results = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in details_map:
                track_info = details_map[neighbor_id]
                final_results.append({
                    "item_id": track_info['item_id'],
                    "title": track_info['title'],
                    "author": track_info['author'],
                    "album": (track_info.get('album') or 'unknown'),
                    "album_artist": (track_info.get('album_artist') or 'unknown'),
                    "distance": distance_map[neighbor_id]
                })

        return jsonify(final_results)
    except RuntimeError as e:
        logger.error(f"Runtime error finding neighbors for {target_item_id}: {e}", exc_info=True)
        return jsonify({"error": "The similarity search service is currently unavailable."}), 503
    except Exception as e:
        logger.error(f"Unexpected error finding neighbors for {target_item_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500


@voyager_bp.route('/api/max_distance', methods=['GET'])
def get_max_distance_endpoint():
  """
  Returns the exact maximum distance from the provided item_id to any other item in the index.
  Query param: item_id (required)
  Response: { "max_distance": float, "farthest_item_id": str | null }
  """
  item_id = request.args.get('item_id')
  if not item_id:
    return jsonify({"error": "Missing 'item_id' parameter."}), 400

  try:
    result = get_max_distance_for_id(item_id)
    if result is None:
      return jsonify({"error": f"Item '{item_id}' not found in index or index unavailable."}), 404
    return jsonify(result)
  except RuntimeError as e:
    logger.error(f"Runtime error computing max distance for {item_id}: {e}", exc_info=True)
    return jsonify({"error": "The similarity search service is currently unavailable."}), 503
  except Exception as e:
    logger.error(f"Unexpected error computing max distance for {item_id}: {e}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred."}), 500


@voyager_bp.route('/api/track', methods=['GET'])
def get_track_endpoint():
  """
  Fetch basic track metadata (title, author) for a given item_id.
  Query param: item_id (required)
  Response: { "item_id": str, "title": str, "author": str, "album": str } or 404
  """
  item_id = request.args.get('item_id')
  if not item_id:
    return jsonify({"error": "Missing 'item_id' parameter."}), 400

  try:
    from app import get_score_data_by_ids
    details = get_score_data_by_ids([item_id])
    if not details:
      return jsonify({"error": f"Item '{item_id}' not found."}), 404
    # Return only the basic fields
    d = details[0]
    return jsonify({
        "item_id": d.get('item_id'),
        "title": d.get('title'),
        "author": d.get('author'),
        "album": (d.get('album') or 'unknown'),
        "album_artist": (d.get('album_artist') or 'unknown')
    }), 200
  except Exception as e:
    logger.error(f"Unexpected error fetching track {item_id}: {e}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred."}), 500

@voyager_bp.route('/api/create_playlist', methods=['POST'])
def create_media_server_playlist():
    """
    Creates a new playlist in the configured media server with the provided tracks.
    ---
    tags:
      - Similarity
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              playlist_name:
                type: string
                description: The name for the new playlist.
              track_ids:
                type: array
                items:
                  type: string
                description: A list of track Item IDs to add to the playlist.
    responses:
      201:
        description: Playlist created successfully.
      400:
        description: Bad request, invalid payload.
      500:
        description: Server error during playlist creation.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Debug log incoming payload to help trace client/server mismatch
    try:
        logger.info(f"/api/create_playlist called with payload: {data}")
    except Exception:
        logger.info('/api/create_playlist called (unable to serialize payload)')

    playlist_name = data.get('playlist_name')
    track_ids_raw = data.get('track_ids', [])

    if not playlist_name:
        return jsonify({"error": "Missing 'playlist_name'"}), 400

    final_track_ids = []
    if isinstance(track_ids_raw, list):
        for item in track_ids_raw:
            item_id = None
            if isinstance(item, str):
                item_id = item
            elif isinstance(item, dict) and 'item_id' in item:
                item_id = item['item_id']

            if item_id and item_id not in final_track_ids:
                final_track_ids.append(item_id)

    if not final_track_ids:
        return jsonify({"error": "No valid track IDs were provided to create the playlist"}), 400

    # Optional user credentials may be provided by the client (e.g., from the Sonic Fingerprint UI)
    user_creds = data.get('user_creds') if isinstance(data, dict) else None

    try:
        new_playlist_id = create_playlist_from_ids(playlist_name, final_track_ids, user_creds=user_creds)

        logger.info(f"Successfully created playlist '{playlist_name}' with ID {new_playlist_id}.")

        return jsonify({
            "message": f"Playlist '{playlist_name}' created successfully!",
            "playlist_id": new_playlist_id
        }), 201

    except Exception as e:
        logger.error(f"Failed to create media server playlist '{playlist_name}': {e}", exc_info=True)
        return jsonify({"error": "An error occurred while creating the playlist on the media server."}), 500
