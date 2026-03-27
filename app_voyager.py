# app_voyager.py
from flask import Blueprint, jsonify, request, render_template, g
import logging

# Import the new config option
from config import SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT, SIMILARITY_RADIUS_DEFAULT
from tasks.voyager_manager import (
    find_nearest_neighbors_by_id,
    get_max_distance_for_id,
    create_playlist_from_ids,
    search_tracks_unified,
    get_item_id_by_title_and_artist
)

logger = logging.getLogger(__name__)

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
                  file_path:
                    type: string
                    description: File path of the track (null if not available)
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
                    'album_artist': (r.get('album_artist') or '').strip() or 'unknown',
                    'file_path': r.get('file_path')
                })
            else:
                results.append({'item_id': None, 'title': None, 'author': None, 'album': 'unknown'})
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error during track search: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during search."}), 500


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
                  file_path:
                    type: string
                    description: File path of the track (null if not available)
                  distance:
                    type: number
      400:
        description: Bad request, missing required parameters.
      404:
        description: Target track not found.
      500:
        description: Server error.
    """
    num_neighbors = request.args.get('n', 10, type=int)

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

    # Use track_id resolved by before_request middleware (from item_id or id query param)
    track_id = g.get('track_id')
    if not track_id:
        # Fallback: try title+artist lookup
        title = request.args.get('title')
        artist = request.args.get('artist')
        if title and artist:
            track_id = get_item_id_by_title_and_artist(title, artist)
        if not track_id:
            return jsonify({"error": "Request must include either 'item_id' or both 'title' and 'artist'."}), 400

    try:
        neighbor_results = find_nearest_neighbors_by_id(
            track_id,
            n=num_neighbors,
            eliminate_duplicates=eliminate_duplicates,
            mood_similarity=mood_similarity,
            radius_similarity=radius_similarity
        )
        if not neighbor_results:
            return jsonify({"error": "Target track not found in index or no similar tracks found."}), 404

        from app_helper import get_score_data_by_ids

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
                    "file_path": track_info.get('file_path'),
                    "distance": distance_map[neighbor_id]
                })

        return jsonify(final_results)
    except RuntimeError as e:
        logger.error(f"Runtime error finding neighbors for {track_id}: {e}", exc_info=True)
        return jsonify({"error": "The similarity search service is currently unavailable."}), 503
    except Exception as e:
        logger.error(f"Unexpected error finding neighbors for {track_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500


@voyager_bp.route('/api/max_distance', methods=['GET'])
def get_max_distance_endpoint():
  """
  Returns the exact maximum distance from the provided item_id to any other item in the index.
  Query param: item_id (required)
  Response: { "max_distance": float, "farthest_item_id": str | null }
  """
  # Use track_id resolved by before_request middleware
  track_id = g.get('track_id')
  if not track_id:
    return jsonify({"error": "Missing or unresolvable 'item_id' parameter."}), 400

  try:
    result = get_max_distance_for_id(track_id)
    if result is None:
      return jsonify({"error": f"Track '{track_id}' not found in index or index unavailable."}), 404
    # Ensure farthest_item_id is a string for frontend compatibility
    if result.get('farthest_item_id') is not None:
      result['farthest_item_id'] = str(result['farthest_item_id'])
    return jsonify(result)
  except RuntimeError as e:
    logger.error(f"Runtime error computing max distance for {track_id}: {e}", exc_info=True)
    return jsonify({"error": "The similarity search service is currently unavailable."}), 503
  except Exception as e:
    logger.error(f"Unexpected error computing max distance for {track_id}: {e}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred."}), 500


@voyager_bp.route('/api/track', methods=['GET'])
def get_track_endpoint():
  """
  Fetch basic track metadata (title, author) for a given item_id.
  Query param: item_id (required)
  Response: { "item_id": str, "title": str, "author": str, "album": str } or 404
  """
  # Use track_id resolved by before_request middleware
  track_id = g.get('track_id')
  if not track_id:
    return jsonify({"error": "Missing or unresolvable 'item_id' parameter."}), 400

  try:
    from app_helper import get_score_data_by_ids
    details = get_score_data_by_ids([track_id])
    if not details:
      return jsonify({"error": f"Track '{track_id}' not found."}), 404
    # Return only the basic fields; item_id is already str(track_id) from get_score_data_by_ids
    d = details[0]
    return jsonify({
        "item_id": d.get('item_id'),
        "title": d.get('title'),
        "author": d.get('author'),
        "album": (d.get('album') or 'unknown'),
        "album_artist": (d.get('album_artist') or 'unknown'),
        "file_path": d.get('file_path')
    }), 200
  except Exception as e:
    logger.error(f"Unexpected error fetching track {track_id}: {e}", exc_info=True)
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
              provider_ids:
                description: Provider(s) to create playlist on. Can be 'all', a single ID, or array of IDs.
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
    provider_ids = data.get('provider_ids')  # Can be 'all', int, or list of ints

    if not playlist_name:
        return jsonify({"error": "Missing 'playlist_name'"}), 400

    # Use track_ids resolved by before_request middleware (from track_ids or item_ids in JSON body)
    final_track_ids = g.get('track_ids', [])

    if not final_track_ids:
        return jsonify({"error": "No valid track IDs were provided to create the playlist"}), 400

    # Optional user credentials may be provided by the client (e.g., from the Sonic Fingerprint UI)
    user_creds = data.get('user_creds') if isinstance(data, dict) else None

    try:
        result = create_playlist_from_ids(playlist_name, final_track_ids, user_creds=user_creds, provider_ids=provider_ids)

        # Handle multi-provider result (dict) vs single provider result (string)
        if isinstance(result, dict):
            # Multi-provider response
            success_count = sum(1 for r in result.values() if r.get('success'))
            total_count = len(result)
            logger.info(f"Created playlist '{playlist_name}' on {success_count}/{total_count} providers.")
            return jsonify({
                "message": f"Playlist '{playlist_name}' created on {success_count}/{total_count} provider(s).",
                "results": result
            }), 201
        else:
            # Single provider response (backward compatible)
            logger.info(f"Successfully created playlist '{playlist_name}' with ID {result}.")
            return jsonify({
                "message": f"Playlist '{playlist_name}' created successfully!",
                "playlist_id": result
            }), 201

    except Exception as e:
        logger.error(f"Failed to create media server playlist '{playlist_name}': {e}", exc_info=True)
        return jsonify({"error": "An error occurred while creating the playlist on the media server."}), 500


@voyager_bp.route('/api/providers/enabled', methods=['GET'])
def get_enabled_providers():
    """
    Get list of enabled providers for playlist creation dropdown.
    ---
    tags:
      - Providers
    responses:
      200:
        description: List of enabled providers
    """
    try:
        from tasks.mediaserver import get_enabled_providers_for_playlists
        providers = get_enabled_providers_for_playlists()
        return jsonify(providers), 200
    except Exception as e:
        logger.error(f"Failed to get enabled providers: {e}", exc_info=True)
        return jsonify({'error': 'Failed to load providers'}), 500


@voyager_bp.route('/api/track_by_path', methods=['GET'])
def get_track_by_path_endpoint():
    """
    Look up a track by its file path.
    ---
    tags:
      - Tracks
    parameters:
      - name: path
        in: query
        required: true
        description: The file path to look up (exact or normalized match).
        schema:
          type: string
    responses:
      200:
        description: Track metadata for the matching file path.
      400:
        description: Missing path parameter.
      404:
        description: No track found for the given path.
    """
    file_path = request.args.get('path', '').strip()
    if not file_path:
        return jsonify({"error": "Missing 'path' query parameter."}), 400

    try:
        from app_helper import get_db, normalize_provider_path
        from psycopg2.extras import DictCursor

        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)

        # Try both original and normalized path
        normalized = normalize_provider_path(file_path)
        paths_to_check = [file_path]
        if normalized and normalized != file_path:
            paths_to_check.append(normalized)

        try:
            cur.execute("""
                SELECT track_id, title, author, album, album_artist, tempo, key, scale,
                       mood_vector, energy, other_features, year, rating, file_path
                FROM score
                WHERE file_path = ANY(%s)
                LIMIT 1
            """, (paths_to_check,))
            row = cur.fetchone()
        finally:
            cur.close()

        if not row:
            return jsonify({"error": f"No track found for path: {file_path}"}), 404

        d = dict(row)
        return jsonify({
            "item_id": str(d['track_id']),
            "track_id": d['track_id'],
            "title": d.get('title'),
            "author": d.get('author'),
            "album": (d.get('album') or 'unknown'),
            "album_artist": (d.get('album_artist') or 'unknown'),
            "tempo": d.get('tempo'),
            "key": d.get('key'),
            "scale": d.get('scale'),
            "mood_vector": d.get('mood_vector'),
            "energy": d.get('energy'),
            "other_features": d.get('other_features'),
            "year": d.get('year'),
            "rating": d.get('rating'),
            "file_path": d.get('file_path')
        }), 200

    except Exception as e:
        logger.error(f"Error looking up track by path '{file_path}': {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500


@voyager_bp.route('/api/tracks_by_paths', methods=['POST'])
def get_tracks_by_paths_endpoint():
    """
    Batch look up tracks by file paths.
    ---
    tags:
      - Tracks
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              paths:
                type: array
                items:
                  type: string
                description: List of file paths to look up.
    responses:
      200:
        description: Object mapping each requested path to its track data (or null if not found).
      400:
        description: Bad request.
    """
    data = request.get_json()
    if not data or not isinstance(data.get('paths'), list):
        return jsonify({"error": "Request body must contain a 'paths' array."}), 400

    paths = [p.strip() for p in data['paths'] if isinstance(p, str) and p.strip()]
    if not paths:
        return jsonify({}), 200

    # Cap batch size to prevent abuse
    MAX_BATCH = 500
    if len(paths) > MAX_BATCH:
        return jsonify({"error": f"Too many paths. Maximum is {MAX_BATCH}."}), 400

    try:
        from app_helper import get_db, normalize_provider_path
        from psycopg2.extras import DictCursor

        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)

        # Build lookup set: original + normalized for each path
        all_lookup_paths = []
        path_to_original = {}  # maps normalized/original DB path back to the user's requested path
        for p in paths:
            all_lookup_paths.append(p)
            path_to_original[p.lower()] = p
            normalized = normalize_provider_path(p)
            if normalized and normalized != p:
                all_lookup_paths.append(normalized)
                path_to_original[normalized.lower()] = p

        try:
            cur.execute("""
                SELECT track_id, title, author, album, album_artist, tempo, key, scale,
                       mood_vector, energy, other_features, year, rating, file_path
                FROM score
                WHERE file_path = ANY(%s)
            """, (all_lookup_paths,))
            rows = cur.fetchall()
        finally:
            cur.close()

        # Build result keyed by the original requested path
        result = {p: None for p in paths}
        for row in rows:
            d = dict(row)
            db_path = (d.get('file_path') or '').lower()
            original_key = path_to_original.get(db_path)
            if original_key and original_key in result:
                result[original_key] = {
                    "item_id": str(d['track_id']),
                    "track_id": d['track_id'],
                    "title": d.get('title'),
                    "author": d.get('author'),
                    "album": (d.get('album') or 'unknown'),
                    "album_artist": (d.get('album_artist') or 'unknown'),
                    "tempo": d.get('tempo'),
                    "key": d.get('key'),
                    "scale": d.get('scale'),
                    "mood_vector": d.get('mood_vector'),
                    "energy": d.get('energy'),
                    "other_features": d.get('other_features'),
                    "year": d.get('year'),
                    "rating": d.get('rating'),
                    "file_path": d.get('file_path')
                }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in batch track lookup by paths: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500
