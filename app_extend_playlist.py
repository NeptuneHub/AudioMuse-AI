from flask import Blueprint, jsonify, request, render_template
import logging
import numpy as np
from collections import defaultdict

from tasks.voyager_manager import find_nearest_neighbors_by_vector, get_vector_by_id
from app_helper import get_db
from psycopg2.extras import DictCursor
import config

logger = logging.getLogger(__name__)

extend_playlist_bp = Blueprint('extend_playlist_bp', __name__, template_folder='../templates')


@extend_playlist_bp.route('/extend_playlist', methods=['GET'])
def extend_playlist_page():
    """Render the Extend Playlist page."""
    return render_template('extend_playlist.html', title='AudioMuse-AI - Extend Playlist', active='extend_playlist')


def _compute_centroid_from_ids(ids: list) -> np.ndarray:
    """Fetch vectors by id and compute their centroid (mean)."""
    vectors = []
    for item_id in ids:
        vec = get_vector_by_id(item_id)
        if vec is not None:
            vectors.append(np.array(vec, dtype=float))
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


@extend_playlist_bp.route('/api/extend_playlist', methods=['POST'])
def extend_playlist_api():
    """
    Extend a playlist by finding similar songs.

    POST payload: {
        "playlist_name": "My Playlist",
        "max_songs": 50,
        "similarity_threshold": 0.5,
        "accepted_ids": [],  # Songs that have been accepted (to include in centroid)
        "excluded_ids": []   # Songs that have been declined (to exclude from results)
    }

    Returns: {
        "results": [
            {"item_id": str, "title": str, "author": str, "distance": float},
            ...
        ]
    }
    """
    payload = request.get_json() or {}

    playlist_name = payload.get('playlist_name')
    max_songs = payload.get('max_songs', 50)
    similarity_threshold = payload.get('similarity_threshold', 0.5)
    accepted_ids = payload.get('accepted_ids', [])
    excluded_ids = payload.get('excluded_ids', [])

    if not playlist_name:
        return jsonify({"error": "Missing 'playlist_name'"}), 400

    try:
        # Get all songs from the playlist
        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute("SELECT item_id FROM playlist WHERE playlist_name = %s", (playlist_name,))
        rows = cur.fetchall()
        cur.close()

        playlist_ids = [row['item_id'] for row in rows]

        if not playlist_ids:
            return jsonify({"error": f"Playlist '{playlist_name}' not found or is empty"}), 404

        # Combine original playlist songs with accepted songs for centroid calculation
        all_ids_for_centroid = list(set(playlist_ids + accepted_ids))

        # Compute centroid of all songs
        centroid = _compute_centroid_from_ids(all_ids_for_centroid)

        if centroid is None:
            return jsonify({"error": "Failed to compute playlist centroid"}), 500

        # Find similar songs
        # Request more songs than needed to account for filtering
        n_candidates = max_songs * 3

        neighbor_results = find_nearest_neighbors_by_vector(
            centroid,
            n=n_candidates,
            eliminate_duplicates=True
        )

        # Filter results
        filtered_results = []
        already_included_ids = set(playlist_ids + accepted_ids + excluded_ids)

        for result in neighbor_results:
            item_id = result.get('item_id')
            distance = result.get('distance', 0)

            # Skip if already in playlist, accepted, or excluded
            if item_id in already_included_ids:
                continue

            # Filter by similarity threshold (lower distance = more similar)
            if distance <= similarity_threshold:
                filtered_results.append(result)

            # Stop if we have enough results
            if len(filtered_results) >= max_songs:
                break

        return jsonify({
            "results": filtered_results,
            "playlist_song_count": len(playlist_ids),
            "accepted_count": len(accepted_ids),
            "excluded_count": len(excluded_ids)
        })

    except Exception as e:
        logger.exception("Extend playlist failed")
        return jsonify({"error": "Internal error"}), 500


@extend_playlist_bp.route('/api/save_extended_playlist', methods=['POST'])
def save_extended_playlist():
    """
    Save an extended playlist to the media server.

    POST payload: {
        "original_playlist_name": "My Playlist",
        "new_playlist_name": "My Extended Playlist",
        "accepted_ids": []  # Songs that were accepted
    }
    """
    from tasks.voyager_manager import create_playlist_from_ids

    payload = request.get_json() or {}

    original_playlist_name = payload.get('original_playlist_name')
    new_playlist_name = payload.get('new_playlist_name')
    accepted_ids = payload.get('accepted_ids', [])

    if not original_playlist_name:
        return jsonify({"error": "Missing 'original_playlist_name'"}), 400

    if not new_playlist_name:
        return jsonify({"error": "Missing 'new_playlist_name'"}), 400

    try:
        # Get all songs from the original playlist
        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute("SELECT item_id FROM playlist WHERE playlist_name = %s", (original_playlist_name,))
        rows = cur.fetchall()
        cur.close()

        original_ids = [row['item_id'] for row in rows]

        if not original_ids:
            return jsonify({"error": f"Original playlist '{original_playlist_name}' not found or is empty"}), 404

        # Combine original playlist with accepted songs
        all_track_ids = original_ids + accepted_ids

        # Remove duplicates while preserving order
        seen = set()
        final_track_ids = []
        for track_id in all_track_ids:
            if track_id not in seen:
                seen.add(track_id)
                final_track_ids.append(track_id)

        if not final_track_ids:
            return jsonify({"error": "No valid track IDs were provided"}), 400

        # Create playlist on media server
        new_playlist_id = create_playlist_from_ids(new_playlist_name, final_track_ids)

        return jsonify({
            "message": f"Playlist '{new_playlist_name}' created successfully with {len(final_track_ids)} songs!",
            "playlist_id": new_playlist_id,
            "total_songs": len(final_track_ids),
            "original_songs": len(original_ids),
            "new_songs": len(accepted_ids)
        }), 201

    except Exception as e:
        logger.exception("Save extended playlist failed")
        return jsonify({"error": "Internal error"}), 500
