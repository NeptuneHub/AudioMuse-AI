from flask import Blueprint, jsonify, request, render_template, redirect, url_for
import logging
import numpy as np
from collections import defaultdict

from tasks.voyager_manager import find_nearest_neighbors_by_vector, get_vector_by_id
from app_helper import get_db
from psycopg2.extras import DictCursor
import config

logger = logging.getLogger(__name__)

extend_playlist_bp = Blueprint('extend_playlist_bp', __name__, template_folder='../templates')


@extend_playlist_bp.route('/playlist_builder', methods=['GET'])
def playlist_builder_page():
    """Render the Playlist Builder page."""
    return render_template('extend_playlist.html', title='AudioMuse-AI - Playlist Builder', active='playlist_builder')


@extend_playlist_bp.route('/extend_playlist', methods=['GET'])
def extend_playlist_redirect():
    """Redirect old URL to Playlist Builder with Extend tab active."""
    return redirect(url_for('extend_playlist_bp.playlist_builder_page') + '?tab=extend')


def _compute_centroid_from_ids(ids: list, weights: dict = None) -> np.ndarray:
    """
    Fetch vectors by id and compute their weighted centroid.

    Args:
        ids: List of track item_ids (strings)
        weights: Optional dict mapping item_id (str) -> weight (int, 1-1024)
                 If None or missing key, defaults to weight=1

    Returns:
        Weighted mean vector: sum(w_i * v_i) / sum(w_i)
        Returns None if no valid vectors found
    """
    if weights is None:
        weights = {}

    vectors = []
    weight_values = []

    for item_id in ids:
        vec = get_vector_by_id(item_id)
        if vec is not None:
            vectors.append(np.array(vec, dtype=float))
            # Get weight, default to 1, ensure positive (min 1)
            w = weights.get(str(item_id), 1)
            weight_values.append(max(1, w))

    if not vectors:
        return None

    vectors_array = np.array(vectors)
    weights_array = np.array(weight_values, dtype=float)

    weights_sum = np.sum(weights_array)
    # weights_sum guaranteed > 0 since all weights >= 1

    # Weighted mean: sum(w_i * v_i) / sum(w_i)
    return np.sum(vectors_array * weights_array[:, np.newaxis], axis=0) / weights_sum


def _get_stream_url(item_id):
    """Constructs a stream URL for the given item based on configuration."""
    # Use Download endpoint with API key for authenticated playback
    if config.MEDIASERVER_TYPE == 'jellyfin' or config.MEDIASERVER_TYPE == 'emby':
        token = config.JELLYFIN_TOKEN if config.MEDIASERVER_TYPE == 'jellyfin' else config.EMBY_TOKEN
        base_url = config.JELLYFIN_URL if config.MEDIASERVER_TYPE == 'jellyfin' else config.EMBY_URL
        return f"{base_url}/Items/{item_id}/Download?api_key={token}"
    elif config.MEDIASERVER_TYPE == 'navidrome':
        # Navidrome/Subsonic stream url
        return f"{config.NAVIDROME_URL}/rest/stream?id={item_id}&u={config.NAVIDROME_USER}&p={config.NAVIDROME_PASSWORD}&v=1.16.1&c=AudioMuse"
    # Fallback or other servers might need different logic
    return ""


@extend_playlist_bp.route('/api/filter_options', methods=['GET'])
def get_filter_options():
    """Returns available filter options for Smart Filter dropdowns."""
    # Query unique moods from database
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("""
            SELECT DISTINCT TRIM(SPLIT_PART(mood, ':', 1)) as mood_label
            FROM (
                SELECT UNNEST(STRING_TO_ARRAY(mood_vector, ',')) as mood
                FROM score WHERE mood_vector IS NOT NULL AND mood_vector != ''
            ) t
            ORDER BY mood_label
        """)
        unique_moods = [row[0] for row in cur.fetchall() if row[0]]
    except Exception as e:
        logger.warning(f"Failed to query unique moods: {e}")
        unique_moods = []
    finally:
        cur.close()

    return jsonify({
        "keys": ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
        "scales": ['major', 'minor'],
        "moods": unique_moods,
        "bpm_ranges": [
            {"value": "0-80", "label": "Slow (< 80 BPM)"},
            {"value": "80-100", "label": "Moderate (80-100 BPM)"},
            {"value": "100-120", "label": "Medium (100-120 BPM)"},
            {"value": "120-140", "label": "Fast (120-140 BPM)"},
            {"value": "140-160", "label": "Very Fast (140-160 BPM)"},
            {"value": "160-999", "label": "Extremely Fast (160+ BPM)"}
        ],
        "energy_ranges": [
            {"value": "0-0.33", "label": "Low Energy"},
            {"value": "0.33-0.66", "label": "Medium Energy"},
            {"value": "0.66-1", "label": "High Energy"}
        ]
    })


def _build_filter_query(filters, match_mode='all'):
    """
    Builds a SQL WHERE clause from smart filters.
    filters: list of dicts {field, operator, value}
    match_mode: 'all' (AND) or 'any' (OR)
    """
    if not filters:
        return "1=1", []

    clauses = []
    params = []

    # Map UI fields to DB columns
    field_map = {
        'album': 'album',
        'artist': 'author', # Search both author and song_artist? For now author is main artist.
        'title': 'title',
        'bpm': 'tempo',
        'energy': 'energy',
        'key': 'key',
        'scale': 'scale',
        'mood': 'mood_vector'
    }

    for f in filters:
        field = f.get('field')
        operator = f.get('operator')
        value = f.get('value')

        db_col = field_map.get(field)
        if not db_col:
            continue

        # Handle range-based values for BPM and Energy (e.g., "80-100", "0.33-0.66")
        if field in ['bpm', 'energy'] and '-' in str(value):
            try:
                parts = value.split('-')
                min_val, max_val = float(parts[0]), float(parts[1])
                clauses.append(f"({db_col} >= %s AND {db_col} <= %s)")
                params.extend([min_val, max_val])
                continue
            except (ValueError, IndexError):
                pass  # Fall through to standard handling

        if operator == 'contains':
            clauses.append(f"{db_col} ILIKE %s")
            params.append(f"%{value}%")
        elif operator == 'does_not_contain':
            clauses.append(f"{db_col} NOT ILIKE %s")
            params.append(f"%{value}%")
        elif operator == 'is':
            clauses.append(f"{db_col} = %s")
            params.append(value)
        elif operator == 'is_not':
            clauses.append(f"{db_col} != %s")
            params.append(value)
        elif operator == 'greater_than':
            try:
                val = float(value)
                clauses.append(f"{db_col} > %s")
                params.append(val)
            except ValueError:
                continue
        elif operator == 'less_than':
            try:
                val = float(value)
                clauses.append(f"{db_col} < %s")
                params.append(val)
            except ValueError:
                continue

    if not clauses:
        return "1=1", []

    join_op = " AND " if match_mode == 'all' else " OR "
    return f"({join_op.join(clauses)})", params


@extend_playlist_bp.route('/api/extend_playlist', methods=['POST'])
def extend_playlist_api():
    """
    Extend a playlist by finding similar songs.

    POST payload: {
        "playlist_name": "My Playlist", # Optional if filters are provided
        "filters": [ ... ],             # Optional, used if playlist_name is not provided
        "match_mode": "all",            # 'all' or 'any' for filters
        "max_songs": 50,
        "similarity_threshold": 0.5,
        "included_ids": [],  # Songs that have been included (to include in centroid)
        "excluded_ids": []   # Songs that have been excluded (to exclude from results)
    }

    Returns: {
        "results": [
            {"item_id": str, "title": str, "author": str, "song_artist": str, "distance": float, "stream_url": str},
            ...
        ]
    }
    """
    payload = request.get_json() or {}

    playlist_name = payload.get('playlist_name')
    filters = payload.get('filters')
    match_mode = payload.get('match_mode', 'all')

    max_songs = payload.get('max_songs', 50)
    similarity_threshold = payload.get('similarity_threshold', 0.5)
    included_ids = payload.get('included_ids', [])
    excluded_ids = payload.get('excluded_ids', [])

    search_only = payload.get('search_only', False)
    source_ids = payload.get('source_ids', [])  # Direct source IDs from Smart Filter

    # Track weights for centroid calculation
    source_weights = payload.get('source_weights', {})
    included_weights = payload.get('included_weights', {})

    # Validate weights - ensure valid integers from allowed set
    VALID_WEIGHTS = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}

    def sanitize_weights(weights_dict):
        """Ensure weights are valid integers from allowed set."""
        sanitized = {}
        for k, v in weights_dict.items():
            try:
                w = int(v)
                sanitized[str(k)] = w if w in VALID_WEIGHTS else 1
            except (ValueError, TypeError):
                sanitized[str(k)] = 1
        return sanitized

    source_weights = sanitize_weights(source_weights)
    included_weights = sanitize_weights(included_weights)

    if not playlist_name and not filters and not source_ids:
        return jsonify({"error": "Missing 'playlist_name', 'filters', or 'source_ids'"}), 400

    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)
        playlist_ids = []

        if playlist_name:
            # Get all songs from the playlist
            cur.execute("SELECT item_id FROM playlist WHERE playlist_name = %s", (playlist_name,))
            rows = cur.fetchall()
            playlist_ids = [row['item_id'] for row in rows]

            if not playlist_ids:
                return jsonify({"error": f"Playlist '{playlist_name}' not found or is empty"}), 404

        elif filters:
            # Build query from filters
            where_clause, params = _build_filter_query(filters, match_mode)
            query = f"SELECT item_id FROM score WHERE {where_clause}"
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            playlist_ids = [row['item_id'] for row in rows]

            if not playlist_ids:
                 return jsonify({"error": "No songs found matching the filters"}), 404

        elif source_ids:
            # Use provided source IDs directly (from Smart Filter results)
            playlist_ids = source_ids

        cur.close()

        # --- SEARCH ONLY MODE ---
        if search_only:
            # If search_only is True, we just return the metadata for the found playlist_ids (which came from filters)
            # We don't do centroid calculation or neighbor search.
            
            # Fetch metadata
            from app_helper import get_score_data_by_ids
            metadata_list = get_score_data_by_ids(playlist_ids)
            
            results = []
            for meta in metadata_list:
                # Enrich with stream url
                item_id = meta['item_id']
                meta['stream_url'] = _get_stream_url(item_id)
                # Ensure song_artist is present (fallback to author)
                meta['song_artist'] = meta.get('song_artist') or meta.get('author')
                # Distance is 0 for exact matches
                meta['distance'] = 0.0
                results.append(meta)
                
            return jsonify({
                "results": results,
                "playlist_song_count": len(results),
                "included_count": 0,
                "excluded_count": 0
            })

        # --- EXTEND MODE (Default) ---

        # Combine original playlist songs with included songs for positive centroid calculation
        all_ids_for_centroid = list(set(playlist_ids + included_ids))

        # Build combined weights: source_weights for playlist tracks, included_weights for included tracks
        combined_weights = {}
        for pid in playlist_ids:
            combined_weights[str(pid)] = source_weights.get(str(pid), 1)
        for inc_id in included_ids:
            combined_weights[str(inc_id)] = included_weights.get(str(inc_id), 1)

        # Compute positive centroid with weights
        positive_centroid = _compute_centroid_from_ids(all_ids_for_centroid, combined_weights)

        if positive_centroid is None:
            return jsonify({"error": "Failed to compute playlist centroid"}), 500

        # Compute excluded centroid if there are excluded songs (unweighted - intentional)
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

        # Gather item IDs to fetch additional metadata
        candidate_ids = [r.get('item_id') for r in neighbor_results]

        # Fetch metadata for candidates
        from app_helper import get_score_data_by_ids
        metadata_list = get_score_data_by_ids(candidate_ids)
        metadata_map = {m['item_id']: m for m in metadata_list}

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
                # Enrich with song_artist and stream_url
                meta = metadata_map.get(item_id, {})
                # Use song_artist if available, fallback to author (Album Artist usually)
                result['song_artist'] = meta.get('song_artist') or meta.get('author')
                result['album'] = meta.get('album')
                result['album_artist'] = meta.get('album_artist')

                # Ensure title is present (sometimes missing in vector result)
                if not result.get('title'):
                    result['title'] = meta.get('title')
                if not result.get('author'):
                    result['author'] = meta.get('author')

                result['stream_url'] = _get_stream_url(item_id)

                filtered_results.append(result)

            # Stop if we have enough results
            if len(filtered_results) >= max_songs:
                break

        # Return source tracks metadata for drawer display (only in extend mode)
        source_tracks_meta = []
        if playlist_ids:
            source_meta_list = get_score_data_by_ids(playlist_ids)
            for meta in source_meta_list:
                meta['stream_url'] = _get_stream_url(meta['item_id'])
                meta['song_artist'] = meta.get('song_artist') or meta.get('author')
                source_tracks_meta.append(meta)

        return jsonify({
            "results": filtered_results,
            "playlist_song_count": len(playlist_ids),
            "included_count": len(included_ids),
            "excluded_count": len(excluded_ids),
            "source_tracks": source_tracks_meta
        })

    except Exception as e:
        logger.exception("Extend playlist failed")
        return jsonify({"error": "Internal error"}), 500


@extend_playlist_bp.route('/api/save_extended_playlist', methods=['POST'])
def save_extended_playlist():
    """
    Save an extended playlist to the media server.

    POST payload: {
        "original_playlist_name": "My Playlist", # Optional if filters are provided
        "filters": [...],                        # Optional if original_playlist_name is provided
        "match_mode": "all",
        "new_playlist_name": "My Extended Playlist",
        "included_ids": []  # Songs that were included
    }
    """
    from tasks.voyager_manager import create_playlist_from_ids

    payload = request.get_json() or {}

    original_playlist_name = payload.get('original_playlist_name')
    filters = payload.get('filters')
    source_ids = payload.get('source_ids')  # Direct source IDs from Smart Filter → Extend flow
    match_mode = payload.get('match_mode', 'all')
    new_playlist_name = payload.get('new_playlist_name')
    included_ids = payload.get('included_ids', [])

    if not original_playlist_name and not filters and not source_ids:
        return jsonify({"error": "Missing source (original_playlist_name, filters, or source_ids)"}), 400

    if not new_playlist_name:
        return jsonify({"error": "Missing 'new_playlist_name'"}), 400

    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)
        original_ids = []

        if original_playlist_name:
            # Get all songs from the original playlist
            cur.execute("SELECT item_id FROM playlist WHERE playlist_name = %s", (original_playlist_name,))
            rows = cur.fetchall()
            original_ids = [row['item_id'] for row in rows]

            if not original_ids:
                cur.close()
                return jsonify({"error": f"Original playlist '{original_playlist_name}' not found or is empty"}), 404

        elif filters:
            # Get songs matching filters
            where_clause, params = _build_filter_query(filters, match_mode)
            query = f"SELECT item_id FROM score WHERE {where_clause}"
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            original_ids = [row['item_id'] for row in rows]

            if not original_ids:
                cur.close()
                return jsonify({"error": "No songs found matching the filters to save"}), 404

        elif source_ids:
            # Use provided source IDs directly (from Smart Filter → Extend flow)
            original_ids = source_ids

        cur.close()

        # Combine original source songs with included songs
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

        # Create playlist on media server (use exact name, no _instant suffix)
        new_playlist_id = create_playlist_from_ids(new_playlist_name, final_track_ids, add_instant_suffix=False)

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
