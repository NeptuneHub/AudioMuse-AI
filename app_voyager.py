# app_voyager.py
from flask import Blueprint, jsonify, request, render_template
import logging

# Import the new config option
from config import SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT
from tasks.voyager_manager import (
    find_nearest_neighbors_by_id, 
    create_playlist_from_ids,
    search_tracks_by_title_and_artist,
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
    return render_template('similarity.html')

@voyager_bp.route('/api/search_tracks', methods=['GET'])
def search_tracks_endpoint():
    """
    Provides autocomplete suggestions for tracks based on title and artist.
    ---
    tags:
      - Similarity
    parameters:
      - name: title
        in: query
        description: Partial or full title of the track.
        schema:
          type: string
      - name: artist
        in: query
        description: Partial or full name of the artist.
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
    """
    title_query = request.args.get('title', '', type=str)
    artist_query = request.args.get('artist', '', type=str)

    if not title_query and not artist_query:
        return jsonify([])

    if len(title_query) < 3 and len(artist_query) < 3:
        return jsonify([])

    try:
        results = search_tracks_by_title_and_artist(title_query, artist_query)
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
    
    eliminate_duplicates_str = request.args.get('eliminate_duplicates')
    if eliminate_duplicates_str is None:
        eliminate_duplicates = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT
    else:
        eliminate_duplicates = eliminate_duplicates_str.lower() == 'true'

    radius_similarity_str = request.args.get('radius_similarity')
    radius_similarity = radius_similarity_str and radius_similarity_str.lower() == 'true'

    mood_similarity_str = request.args.get('mood_similarity')
    if mood_similarity_str is None:
        mood_similarity = True  # Default to True when parameter is not provided
    else:
        mood_similarity = mood_similarity_str.lower() == 'true'

    target_item_id = None

    if item_id:
        target_item_id = item_id
    elif title and artist:
        resolved_id = get_item_id_by_title_and_artist(title, artist)
        if not resolved_id:
            return jsonify({"error": f"Track '{title}' by '{artist}' not found in the database."}), 404
        target_item_id = resolved_id
    else:
        return jsonify({"error": "Request must include either 'item_id' or both 'title' and 'artist'."}), 400

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
                    "distance": distance_map[neighbor_id]
                })

        return jsonify(final_results)
    except RuntimeError as e:
        logger.error(f"Runtime error finding neighbors for {target_item_id}: {e}", exc_info=True)
        return jsonify({"error": "The similarity search service is currently unavailable."}), 503
    except Exception as e:
        logger.error(f"Unexpected error finding neighbors for {target_item_id}: {e}", exc_info=True)
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
