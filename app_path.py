# app_path.py
from flask import Blueprint, jsonify, request, render_template
import logging
from tasks.path_manager import find_path_between_songs
from config import PATH_DEFAULT_LENGTH
import numpy as np
import math # Import the math module

logger = logging.getLogger(__name__)

# Create a Blueprint for the path finding routes
path_bp = Blueprint('path_bp', __name__, template_folder='../templates')

@path_bp.route('/path', methods=['GET'])
def path_page():
    """
    Serves the frontend page for finding a path between songs.
    """
    return render_template('path.html')

@path_bp.route('/api/find_path', methods=['GET'])
def find_path_endpoint():
    """
    Finds a path of similar songs between a start and end track.
    """
    start_song_id = request.args.get('start_song_id')
    end_song_id = request.args.get('end_song_id')
    # Use the default from config if max_steps is not provided in the request
    max_steps = request.args.get('max_steps', PATH_DEFAULT_LENGTH, type=int)
    # Optional temperature parameter for similarity sampling (float). If omitted, config default will be used.
    temperature_raw = request.args.get('temperature')
    temperature = None
    if temperature_raw is not None:
        try:
            temperature = float(temperature_raw)
        except Exception:
            temperature = None

    if not start_song_id or not end_song_id:
        return jsonify({"error": "Both a start and end song must be provided."}), 400

    if start_song_id == end_song_id:
        return jsonify({"error": "Start and end songs cannot be the same."}), 400

    try:
        path, total_distance = find_path_between_songs(start_song_id, end_song_id, max_steps, temperature=temperature)

        if not path:
            return jsonify({"error": f"No path found between the selected songs within {max_steps} steps."}), 404

        # --- CHANGED: Process embedding vectors for JSON response ---
        for song in path:
            # The raw 'embedding' is a memoryview/bytes object and is not JSON serializable.
            # We remove it as the frontend will use 'embedding_vector'.
            if 'embedding' in song:
                del song['embedding']

            # Convert numpy array 'embedding_vector' to a plain list if it exists
            if 'embedding_vector' in song and isinstance(song['embedding_vector'], np.ndarray):
                song['embedding_vector'] = song['embedding_vector'].tolist()
            else:
                # Ensure the key exists even if there's no vector, for frontend consistency
                song['embedding_vector'] = []

        # --- FIX: Convert total_distance from numpy.float32 to a standard Python float ---
        final_distance = float(total_distance) if total_distance is not None and math.isfinite(total_distance) else 0.0

        return jsonify({
            "path": path,
            "total_distance": final_distance
        })

    except Exception as e:
        logger.error(f"Error finding path between {start_song_id} and {end_song_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while finding the path."}), 500
