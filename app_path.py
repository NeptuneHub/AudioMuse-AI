# app_path.py
from flask import Blueprint, jsonify, request, render_template
import logging
from tasks.path_manager import find_path_between_songs
from config import PATH_DEFAULT_LENGTH, PATH_FIX_SIZE
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
    # Pass the server default for path_fix_size so the UI checkbox reflects config/env
    return render_template('path.html', path_fix_size=PATH_FIX_SIZE, title = 'AudioMuse-AI - Song Path', active='path')

@path_bp.route('/api/find_path', methods=['GET'])
def find_path_endpoint():
    """
    Finds a path of similar songs between a start and end track.
    """
    start_song_id = request.args.get('start_song_id')
    end_song_id = request.args.get('end_song_id')
    # Use the default from config if max_steps is not provided in the request
    max_steps = request.args.get('max_steps', PATH_DEFAULT_LENGTH, type=int)

    if not start_song_id or not end_song_id:
        return jsonify({"error": "Both a start and end song must be provided."}), 400

    if start_song_id == end_song_id:
        return jsonify({"error": "Start and end songs cannot be the same."}), 400

    try:
        # parse optional path_fix_size override from request (query param)
        pfs = request.args.get('path_fix_size')
        if pfs is None:
            path_fix_size = PATH_FIX_SIZE
        else:
            path_fix_size = str(pfs).lower() in ('1', 'true', 'yes', 'y')

        path, total_distance = find_path_between_songs(start_song_id, end_song_id, max_steps, path_fix_size=path_fix_size)

        if not path:
            return jsonify({"error": f"No path found between the selected songs within {max_steps} steps."}), 404

        # --- CHANGED: Process embedding vectors for JSON response ---
        for song in path:
            # The raw 'embedding' is a memoryview/bytes object and is not JSON serializable.
            if 'embedding' in song:
                del song['embedding']

            # Convert numpy array 'embedding_vector' to a plain list if it exists
            if 'embedding_vector' in song and isinstance(song['embedding_vector'], np.ndarray):
                song['embedding_vector'] = song['embedding_vector'].tolist()
            else:
                song['embedding_vector'] = []

            # Ensure album field is present (for frontend)
            if 'album' not in song:
                song['album'] = song.get('album', '')

        # --- FIX: Convert total_distance from numpy.float32 to a standard Python float ---
        final_distance = float(total_distance) if total_distance is not None and math.isfinite(total_distance) else 0.0

        return jsonify({
            "path": path,
            "total_distance": final_distance
        })

    except Exception as e:
        logger.error(f"Error finding path between {start_song_id} and {end_song_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while finding the path."}), 500
