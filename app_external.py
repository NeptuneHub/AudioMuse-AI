# app_external.py

from flask import Blueprint, jsonify, request
from psycopg2.extras import DictCursor
import numpy as np
import logging

# Import voyager_manager functions for track lookups
from tasks.voyager_manager import search_tracks_unified
# NOTE: The import of 'get_db' has been moved inside each function to prevent circular imports.

logger = logging.getLogger(__name__)

# Create a Blueprint for external API routes
external_bp = Blueprint('external_bp', __name__)

@external_bp.route('/get_score', methods=['GET'])
def get_score_endpoint():
    """
    Get all content from the score database for a given id.
    ---
    tags:
      - External
    parameters:
      - name: id
        in: query
        required: true
        description: The Item ID of the track.
        schema:
          type: string
    responses:
      200:
        description: Score data for the track.
        content:
          application/json:
            schema:
              type: object
      400:
        description: Missing id parameter.
      404:
        description: Score not found for the given id.
      500:
        description: Internal server error.
    """
    # Local import to prevent circular dependency
    from app_helper import get_db

    item_id = request.args.get('id')
    if not item_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400

    try:
        db = get_db()
        cur = db.cursor(cursor_factory=DictCursor)
        cur.execute("SELECT * FROM score WHERE item_id = %s", (item_id,))
        score_data = cur.fetchone()
        cur.close()

        if score_data:
            # Convert DictRow to a standard dictionary for consistent JSON output
            return jsonify(dict(score_data))
        else:
            return jsonify({"error": f"Score not found for id: {item_id}"}), 404
    except Exception as e:
        logger.error(f"Error fetching score for id {item_id}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred"}), 500


@external_bp.route('/get_embedding', methods=['GET'])
def get_embedding_endpoint():
    """
    Get the embedding vector from the database for a given id.
    ---
    tags:
      - External
    parameters:
      - name: id
        in: query
        required: true
        description: The Item ID of the track.
        schema:
          type: string
    responses:
      200:
        description: Embedding data for the track, with the vector as a list of floats.
      400:
        description: Missing id parameter.
      404:
        description: Embedding not found for the given id.
      500:
        description: Internal server error.
    """
    # Local import to prevent circular dependency
    from app_helper import get_db

    item_id = request.args.get('id')
    if not item_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400
    
    try:
        db = get_db()
        cur = db.cursor(cursor_factory=DictCursor)
        cur.execute("SELECT * FROM embedding WHERE item_id = %s", (item_id,))
        embedding_data = cur.fetchone()
        cur.close()

        if embedding_data:
            embedding_dict = dict(embedding_data)
            if embedding_dict.get('embedding'):
                # The embedding is stored as BYTEA, convert it back to a list of floats
                embedding_vector = np.frombuffer(embedding_dict['embedding'], dtype=np.float32)
                embedding_dict['embedding'] = embedding_vector.tolist()
            return jsonify(embedding_dict)
        else:
            return jsonify({"error": f"Embedding not found for id: {item_id}"}), 404
    except Exception as e:
        logger.error(f"Error fetching embedding for id {item_id}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred"}), 500


@external_bp.route('/search', methods=['GET'])
def search_tracks_endpoint():
    """
    Provides autocomplete suggestions for tracks based on title and/or artist.
    A query for either title or artist must be at least 3 characters long.
    ---
    tags:
      - External
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
      400:
        description: Query string too short.
      500:
        description: Internal server error.
    """
    title_query = request.args.get('title', '', type=str)
    artist_query = request.args.get('artist', '', type=str)

    # Return empty list if both queries are empty
    if not title_query and not artist_query:
        return jsonify([])

    # Enforce minimum length constraint
    if len(title_query) < 3 and len(artist_query) < 3:
        return jsonify({"error": "Query for title or artist must be at least 3 characters long"}), 400

    try:
        # Reuse the existing search logic
        results = search_tracks_unified(title_query + artist_query) # TODO convert to search_query
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error during external track search: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during search."}), 500
