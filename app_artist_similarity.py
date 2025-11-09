# app_artist_similarity.py
"""
Flask Blueprint for Artist Similarity functionality.

Provides endpoints for:
- Searching for artists (autocomplete)
- Finding similar artists using GMM-based HNSW index
- Getting all tracks for an artist
- Creating playlists from artist tracks
"""

from flask import Blueprint, jsonify, request, render_template
import logging

from tasks.artist_gmm_manager import (
    find_similar_artists,
    search_artists_by_name,
    get_artist_tracks
)

logger = logging.getLogger(__name__)

# Create Blueprint
artist_similarity_bp = Blueprint('artist_similarity_bp', __name__, template_folder='templates')


@artist_similarity_bp.route('/artist_similarity', methods=['GET'])
def artist_similarity_page():
    """
    Serves the frontend page for finding similar artists.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the artist similarity page.
    """
    return render_template('artist_similarity.html', title='AudioMuse-AI - Artist Similarity', active='artist_similarity')


@artist_similarity_bp.route('/api/search_artists', methods=['GET'])
def search_artists_endpoint():
    """
    Provides autocomplete suggestions for artists based on name.
    ---
    tags:
      - Artist Similarity
    parameters:
      - name: query
        in: query
        description: Partial or full name of the artist.
        schema:
          type: string
    responses:
      200:
        description: A list of matching artists.
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
                properties:
                  artist:
                    type: string
                  track_count:
                    type: integer
    """
    query = request.args.get('query', '', type=str)
    
    if not query or len(query) < 2:
        return jsonify([])
    
    try:
        results = search_artists_by_name(query)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error during artist search: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during search."}), 500


@artist_similarity_bp.route('/api/similar_artists', methods=['GET'])
def get_similar_artists_endpoint():
    """
    Find similar artists for a given artist using GMM-based similarity.
    Accepts either artist name or artist_id.
    ---
    tags:
      - Artist Similarity
    parameters:
      - name: artist
        in: query
        description: The name of the artist.
        schema:
          type: string
      - name: artist_id
        in: query
        description: The ID of the artist from the media server.
        schema:
          type: string
      - name: n
        in: query
        description: The number of similar artists to return.
        schema:
          type: integer
          default: 10
      - name: ef_search
        in: query
        description: HNSW search parameter (higher = more accurate but slower).
        schema:
          type: integer
    responses:
      200:
        description: A list of similar artists with divergence scores, artist names and IDs.
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
                properties:
                  artist:
                    type: string
                  artist_id:
                    type: string
                    nullable: true
                  divergence:
                    type: number
      400:
        description: Bad request, missing artist parameter.
      404:
        description: Artist not found.
      503:
        description: Artist similarity service unavailable.
    """
    artist = request.args.get('artist')
    artist_id = request.args.get('artist_id')
    n = request.args.get('n', 10, type=int)
    ef_search = request.args.get('ef_search', type=int)
    
    # Accept either artist name or artist_id
    query_artist = artist or artist_id
    
    if not query_artist:
        return jsonify({"error": "Missing 'artist' or 'artist_id' parameter"}), 400
    
    try:
        similar_artists = find_similar_artists(query_artist, n=n, ef_search=ef_search)
        
        if not similar_artists:
            return jsonify({"error": f"Artist '{query_artist}' not found in index or no similar artists found."}), 404
        
        return jsonify(similar_artists)
        
    except RuntimeError as e:
        logger.error(f"Runtime error finding similar artists for '{query_artist}': {e}", exc_info=True)
        return jsonify({"error": "The artist similarity search service is currently unavailable."}), 503
    except Exception as e:
        logger.error(f"Unexpected error finding similar artists for '{query_artist}': {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500


@artist_similarity_bp.route('/api/artist_tracks', methods=['GET'])
def get_artist_tracks_endpoint():
    """
    Get all tracks for a given artist (by name or ID).
    ---
    tags:
      - Artist Similarity
    parameters:
      - name: artist
        in: query
        description: The name of the artist.
        schema:
          type: string
      - name: artist_id
        in: query
        description: The ID of the artist from the media server.
        schema:
          type: string
    responses:
      200:
        description: A list of tracks by the artist.
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
      400:
        description: Bad request, missing artist parameter.
    """
    artist = request.args.get('artist')
    artist_id = request.args.get('artist_id')
    
    # Accept either artist name or artist_id
    query_artist = artist or artist_id
    
    if not query_artist:
        return jsonify({"error": "Missing 'artist' or 'artist_id' parameter"}), 400
    
    try:
        tracks = get_artist_tracks(query_artist)
        return jsonify(tracks)
    except Exception as e:
        logger.error(f"Error getting tracks for artist '{query_artist}': {e}", exc_info=True)
        return jsonify({"error": "An error occurred while fetching tracks."}), 500
