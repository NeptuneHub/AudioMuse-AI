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
        "included_ids": [],  # Songs that have been included (to include in centroid)
        "excluded_ids": []   # Songs that have been excluded (to exclude from results)
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
    included_ids = payload.get('included_ids', [])
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

        # Combine original playlist songs with included songs for positive centroid calculation
        all_ids_for_centroid = list(set(playlist_ids + included_ids))

        # Compute positive centroid
        positive_centroid = _compute_centroid_from_ids(all_ids_for_centroid)

        if positive_centroid is None:
            return jsonify({"error": "Failed to compute playlist centroid"}), 500

        # Compute excluded centroid if there are excluded songs
        excluded_centroid = None
        if excluded_ids:
            excluded_centroid = _compute_centroid_from_ids(excluded_ids)

        # Adjust query vector: Subtract excluded centroid from positive centroid
        # We weight the exclusion to avoid pushing it too far, but enough to be effective.
        # Using a heuristic weight of 0.5 for now.
        query_vector = positive_centroid
        if excluded_centroid is not None:
            # Normalize vectors to ensure consistent subtraction magnitude?
            # For now, simple subtraction.
            query_vector = positive_centroid - (excluded_centroid * 0.5)

        # Find similar songs
        # Request more songs than needed to account for filtering
        n_candidates = max_songs * 5  # Increased buffer for exclusion filtering

        neighbor_results = find_nearest_neighbors_by_vector(
            query_vector,
            n=n_candidates,
            eliminate_duplicates=True
        )

        # Filter results
        filtered_results = []
        already_included_ids = set(playlist_ids + included_ids + excluded_ids)

        # Determine filtering threshold based on distance metric
        subtract_threshold = config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR if config.PATH_DISTANCE_METRIC == 'angular' else config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN

        for result in neighbor_results:
            item_id = result.get('item_id')
            distance = result.get('distance', 0)

            # Skip if already in playlist, included, or excluded
            if item_id in already_included_ids:
                continue

            # Active Exclusion Filtering: Check distance to excluded centroid
            if excluded_centroid is not None:
                vec = get_vector_by_id(item_id)
                if vec is not None:
                    v_cand = np.array(vec, dtype=float)
                    
                    if config.PATH_DISTANCE_METRIC == 'angular':
                        # Angular distance
                        v1 = excluded_centroid / (np.linalg.norm(excluded_centroid) or 1.0)
                        v2 = v_cand / (np.linalg.norm(v_cand) or 1.0)
                        cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
                        dist_to_excluded = np.arccos(cosine) / np.pi
                        
                        if dist_to_excluded < subtract_threshold:
                            continue # Too close to excluded songs
                    else:
                        # Euclidean distance
                        dist_to_excluded = np.linalg.norm(excluded_centroid - v_cand)
                        if dist_to_excluded < subtract_threshold:
                            continue # Too close to excluded songs

            # Filter by similarity threshold (lower distance = more similar)
            if distance <= similarity_threshold:
                filtered_results.append(result)

            # Stop if we have enough results
            if len(filtered_results) >= max_songs:
                break

        return jsonify({
            "results": filtered_results,
            "playlist_song_count": len(playlist_ids),
            "included_count": len(included_ids),
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
        "included_ids": []  # Songs that were included
    }
    """
    from tasks.voyager_manager import create_playlist_from_ids

    payload = request.get_json() or {}

    original_playlist_name = payload.get('original_playlist_name')
    new_playlist_name = payload.get('new_playlist_name')
    included_ids = payload.get('included_ids', [])

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

        # Combine original playlist with included songs
        all_track_ids = original_ids + included_ids

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
            "new_songs": len(included_ids)
        }), 201

    except Exception as e:
        logger.exception("Save extended playlist failed")
        return jsonify({"error": "Internal error"}), 500
