from flask import Blueprint, jsonify, request, render_template, Response
import logging
import os
import re
import numpy as np
import requests as http_requests
from psycopg2.extras import DictCursor

from tasks.voyager_manager import find_nearest_neighbors_by_vector, get_vector_by_id
from app_helper import get_db, get_score_data_by_ids
import config

logger = logging.getLogger(__name__)

playlist_curator_bp = Blueprint('playlist_curator_bp', __name__, template_folder='templates')

INFLUENCE_LEVELS = {
    0: 0.0,    # x1 — equal weight
    1: 0.05,   # Boost — ~5% of centroid
    2: 0.15,   # Strong — ~15% of centroid
    3: 0.30,   # Focus — ~30% of centroid
}

VALID_LEVELS = set(INFLUENCE_LEVELS.keys())


def _sanitize_levels(levels_dict):
    """Ensure influence levels are valid (0-3)."""
    sanitized = {}
    for k, v in levels_dict.items():
        try:
            lvl = int(v)
            sanitized[str(k)] = lvl if lvl in VALID_LEVELS else 0
        except (ValueError, TypeError):
            sanitized[str(k)] = 0
    return sanitized


def _levels_to_weights(levels_dict, total_tracks):
    """Convert influence levels to actual weights based on playlist size.

    For a track with target influence pct in a playlist of N tracks:
        weight = (pct * (N - 1)) / (1 - pct)
    Minimum weight is 1.
    """
    weights = {}
    for item_id, level in levels_dict.items():
        pct = INFLUENCE_LEVELS.get(level, 0.0)
        if pct <= 0 or total_tracks <= 1:
            weights[item_id] = 1
        else:
            weights[item_id] = max(1, round(pct * (total_tracks - 1) / (1 - pct)))
    return weights


def _compute_centroid_from_ids(ids, weights=None):
    """
    Fetch vectors by item_id and compute their weighted centroid.

    Args:
        ids: List of item_ids (strings)
        weights: Optional dict mapping str(item_id) -> weight (1-1024)

    Returns:
        Weighted mean vector, or None if no valid vectors found.
    """
    if weights is None:
        weights = {}

    vectors = []
    weight_values = []

    for item_id in ids:
        vec = get_vector_by_id(str(item_id))
        if vec is not None:
            vectors.append(np.array(vec, dtype=float))
            w = weights.get(str(item_id), 1)
            weight_values.append(max(1, w))

    if not vectors:
        return None

    vectors_array = np.array(vectors)
    weights_array = np.array(weight_values, dtype=float)
    return np.sum(vectors_array * weights_array[:, np.newaxis], axis=0) / np.sum(weights_array)


def _build_filter_query(filters, match_mode='all'):
    """
    Builds a SQL WHERE clause from smart search filters.
    Returns (where_clause_string, params_list).
    """
    if not filters:
        return "1=1", []

    clauses = []
    params = []

    field_map = {
        'album': 'album',
        'artist': 'author',
        'album_artist': 'album_artist',
        'title': 'title',
        'bpm': 'tempo',
        'energy': 'energy',
        'key': 'key',
        'scale': 'scale',
        'mood': 'mood_vector',
        'genre': 'mood_vector',
        'year': 'year',
        'decade': 'year',
        'rating': 'rating',
        'features': 'other_features',
    }

    for f in filters:
        field = f.get('field')
        operator = f.get('operator')
        value = f.get('value')
        db_col = field_map.get(field)
        if not db_col:
            continue

        # Range-based values for BPM, Energy, Year, Rating
        if field in ['bpm', 'energy', 'year', 'decade', 'rating'] and '-' in str(value):
            try:
                parts = value.split('-')
                min_val, max_val = float(parts[0]), float(parts[1])

                # Energy: convert normalized 0-1 range to raw DB range
                if field == 'energy':
                    e_min = config.ENERGY_MIN
                    e_max = config.ENERGY_MAX
                    e_span = e_max - e_min
                    min_val = e_min + min_val * e_span
                    max_val = e_min + max_val * e_span

                clauses.append(f"({db_col} >= %s AND {db_col} <= %s)")
                params.extend([min_val, max_val])
                continue
            except (ValueError, IndexError):
                pass

        if operator == 'contains':
            clauses.append(f"{db_col} ILIKE %s")
            params.append(f"%{value}%")
        elif operator == 'does_not_contain':
            clauses.append(f"{db_col} NOT ILIKE %s")
            params.append(f"%{value}%")
        elif operator == 'is':
            if field in ('mood', 'genre'):
                # Use regex to match genre label within comma-separated mood_vector
                clauses.append(f"{db_col} ~ %s")
                params.append(f"(^|,)\\s*{re.escape(value)}:")
            else:
                clauses.append(f"{db_col} = %s")
                params.append(value)
        elif operator == 'is_not':
            if field in ('mood', 'genre'):
                clauses.append(f"{db_col} !~ %s")
                params.append(f"(^|,)\\s*{re.escape(value)}:")
            else:
                clauses.append(f"{db_col} != %s")
                params.append(value)
        elif operator in ('greater_than', 'less_than'):
            if field in ('features', 'genre', 'mood') and ':' in str(value):
                # Score-aware filter: value is "label:threshold"
                parts = value.rsplit(':', 1)
                label = parts[0].strip()
                try:
                    threshold = float(parts[1])
                except (ValueError, IndexError):
                    continue
                op_sym = '>=' if operator == 'greater_than' else '<='
                clauses.append(f"""EXISTS (
                    SELECT 1 FROM UNNEST(STRING_TO_ARRAY({db_col}, ',')) AS f
                    WHERE TRIM(SPLIT_PART(f, ':', 1)) = %s
                    AND CAST(SPLIT_PART(f, ':', 2) AS FLOAT) {op_sym} %s
                )""")
                params.extend([label, threshold])
            else:
                try:
                    fval = float(value)
                except ValueError:
                    continue
                op_sym = '>' if operator == 'greater_than' else '<'
                clauses.append(f"{db_col} {op_sym} %s")
                params.append(fval)

    if not clauses:
        return "1=1", []

    join_op = " AND " if match_mode == 'all' else " OR "
    return f"({join_op.join(clauses)})", params


def _find_duplicate_groups(item_ids, threshold=0.015):
    """
    Find duplicate groups in a set of tracks using embedding cosine distance.

    Args:
        item_ids: List of item_id strings
        threshold: Cosine distance threshold (0.01=strict, 0.15=loose)

    Returns:
        Dict with 'groups', 'total_groups', 'total_duplicate_tracks'
    """
    from collections import defaultdict

    valid_ids = []
    vectors = []
    for iid in item_ids:
        vec = get_vector_by_id(str(iid))
        if vec is not None:
            valid_ids.append(str(iid))
            vectors.append(np.array(vec, dtype=np.float32))

    if len(vectors) < 2:
        return {"groups": [], "total_groups": 0, "total_duplicate_tracks": 0}

    V = np.vstack(vectors)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    V_normed = V / norms
    similarity_matrix = V_normed @ V_normed.T
    np.clip(similarity_matrix, -1.0, 1.0, out=similarity_matrix)
    distance_matrix = 1.0 - similarity_matrix

    n = len(valid_ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < threshold:
                union(i, j)

    clusters = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    duplicate_groups = [indices for indices in clusters.values() if len(indices) >= 2]
    if not duplicate_groups:
        return {"groups": [], "total_groups": 0, "total_duplicate_tracks": 0}

    all_dup_ids = []
    for indices in duplicate_groups:
        for idx in indices:
            all_dup_ids.append(valid_ids[idx])

    metadata_list = get_score_data_by_ids(all_dup_ids)
    metadata_map = {m['item_id']: m for m in metadata_list}

    position_map = {iid: pos for pos, iid in enumerate(item_ids)}
    total_tracks = len(item_ids)

    groups = []
    total_duplicate_tracks = 0

    for indices in duplicate_groups:
        group_tracks = []
        for idx in indices:
            iid = valid_ids[idx]
            meta = metadata_map.get(iid, {})

            rating_score = ((meta.get('rating') or 0) / 5.0) * 3.0
            completeness = sum(1 for f in ['album', 'year', 'album_artist'] if meta.get(f) is not None)
            completeness_score = (completeness / 3.0) * 2.0
            year = meta.get('year')
            year_score = ((2050 - year) / 100.0 * 3.0) if year and year > 1900 else 0.0
            pos = position_map.get(iid, total_tracks)
            position_score = (1.0 - (pos / max(total_tracks, 1))) * 0.1

            score = round(rating_score + completeness_score + year_score + position_score, 2)

            group_tracks.append({
                'item_id': iid,
                'title': meta.get('title'),
                'author': meta.get('author'),
                'album': meta.get('album'),
                'album_artist': meta.get('album_artist'),
                'year': meta.get('year'),
                'rating': meta.get('rating'),
                'score': score
            })

        group_tracks.sort(key=lambda t: t['score'], reverse=True)
        groups.append({'tracks': group_tracks})
        total_duplicate_tracks += len(group_tracks)

    return {
        "groups": groups,
        "total_groups": len(groups),
        "total_duplicate_tracks": total_duplicate_tracks
    }


# --- Routes -----------------------------------------------------------------

@playlist_curator_bp.route('/playlist_curator', methods=['GET'])
def playlist_curator_page():
    return render_template('playlist_curator.html',
                           title='AudioMuse-AI - Playlist Curator',
                           active='playlist_curator')


@playlist_curator_bp.route('/api/curator/filter_options', methods=['GET'])
def get_filter_options():
    """Returns available filter options for Smart Search dropdowns."""
    db = get_db()
    cur = db.cursor()
    unique_moods = []
    unique_features = []
    year_min = None
    year_max = None
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

        cur.execute("""
            SELECT DISTINCT TRIM(SPLIT_PART(feature, ':', 1)) as feature_label
            FROM (
                SELECT UNNEST(STRING_TO_ARRAY(other_features, ',')) as feature
                FROM score WHERE other_features IS NOT NULL AND other_features != ''
            ) t
            WHERE TRIM(SPLIT_PART(feature, ':', 1)) != ''
            ORDER BY feature_label
        """)
        unique_features = [row[0] for row in cur.fetchall() if row[0]]

        cur.execute("SELECT MIN(year) AS ymin, MAX(year) AS ymax FROM score WHERE year IS NOT NULL AND year > 0")
        row = cur.fetchone()
        if row:
            year_min = row[0]
            year_max = row[1]
    except Exception as e:
        logger.warning(f"Failed to query filter options: {e}")
    finally:
        cur.close()

    return jsonify({
        "keys": ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
        "scales": ['major', 'minor'],
        "moods": unique_moods,
        "features": unique_features,
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
        ],
        "year_ranges": [
            {"value": "0-1969", "label": "Before 1970"},
            {"value": "1970-1979", "label": "1970s"},
            {"value": "1980-1989", "label": "1980s"},
            {"value": "1990-1999", "label": "1990s"},
            {"value": "2000-2009", "label": "2000s"},
            {"value": "2010-2019", "label": "2010s"},
            {"value": "2020-2029", "label": "2020s"}
        ],
        "rating_ranges": [
            {"value": "1-5", "label": "Any Rating (1-5)"},
            {"value": "3-5", "label": "Good (3-5)"},
            {"value": "4-5", "label": "Great (4-5)"},
            {"value": "5-5", "label": "Favorites (5)"}
        ],
        "year_min": year_min,
        "year_max": year_max
    })


@playlist_curator_bp.route('/api/curator/search', methods=['POST'])
def search_api():
    """
    Main search/extend endpoint.
    search_only=true  -> Smart Search (returns filter matches)
    search_only=false -> Extend mode (weighted centroid + neighbors)
    """
    payload = request.get_json() or {}

    playlist_name = payload.get('playlist_name')
    filters = payload.get('filters')
    match_mode = payload.get('match_mode', 'all')
    try:
        max_songs = min(max(1, int(payload.get('max_songs', 50))), 500)
    except (TypeError, ValueError):
        max_songs = 50
    similarity_threshold = payload.get('similarity_threshold', 0.5)
    included_ids = [str(i) for i in payload.get('included_ids', [])]
    excluded_ids = [str(i) for i in payload.get('excluded_ids', [])]
    min_rating = payload.get('min_rating')
    year_min = payload.get('year_min')
    year_max = payload.get('year_max')
    search_only = payload.get('search_only', False)
    source_ids = [str(s) for s in payload.get('source_ids', [])]

    try:
        duplicate_threshold = max(0.005, min(float(payload.get('duplicate_threshold', 0.01)), 0.3))
    except (TypeError, ValueError):
        duplicate_threshold = 0.01

    source_levels = _sanitize_levels(payload.get('source_weights', {}))
    included_levels = _sanitize_levels(payload.get('included_weights', {}))

    if not playlist_name and not filters and not source_ids:
        return jsonify({"error": "Missing 'playlist_name', 'filters', or 'source_ids'"}), 400

    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)
        playlist_ids = []

        if playlist_name:
            cur.execute("SELECT item_id FROM playlist WHERE playlist_name = %s", (playlist_name,))
            rows = cur.fetchall()
            playlist_ids = [row['item_id'] for row in rows]
            if not playlist_ids:
                cur.close()
                return jsonify({"error": f"Playlist '{playlist_name}' not found or is empty"}), 404

        elif filters:
            where_clause, params = _build_filter_query(filters, match_mode)
            cur.execute(f"SELECT item_id FROM score WHERE {where_clause}", tuple(params))
            rows = cur.fetchall()
            playlist_ids = [row['item_id'] for row in rows]
            if not playlist_ids:
                cur.close()
                return jsonify({"error": "No songs found matching the filters"}), 404

        elif source_ids:
            playlist_ids = list(source_ids)

        cur.close()

        # -- SEARCH ONLY MODE --------------------------------------------------
        if search_only:
            metadata_list = get_score_data_by_ids(playlist_ids)
            for meta in metadata_list:
                meta['distance'] = 0.0
            return jsonify({
                "results": metadata_list,
                "playlist_song_count": len(metadata_list),
                "included_count": 0,
                "excluded_count": 0
            })

        # -- EXTEND MODE -------------------------------------------------------

        # Combine source + included for positive centroid
        all_ids_for_centroid = list(set(list(playlist_ids) + list(included_ids)))

        # Convert influence levels to actual weights based on total track count
        total_tracks = len(all_ids_for_centroid)
        combined_levels = {}
        for pid in playlist_ids:
            combined_levels[str(pid)] = source_levels.get(str(pid), 0)
        for inc_id in included_ids:
            combined_levels[str(inc_id)] = included_levels.get(str(inc_id), 0)
        combined_weights = _levels_to_weights(combined_levels, total_tracks)

        positive_centroid = _compute_centroid_from_ids(all_ids_for_centroid, combined_weights)
        if positive_centroid is None:
            return jsonify({"error": "Failed to compute playlist centroid - no valid embeddings found"}), 500

        # Excluded centroid (unweighted)
        excluded_centroid = None
        if excluded_ids:
            excluded_centroid = _compute_centroid_from_ids(list(excluded_ids))

        # Adjust query vector
        query_vector = positive_centroid
        if excluded_centroid is not None:
            query_vector = positive_centroid - (excluded_centroid * 0.5)

        # Find similar songs - request more candidates to account for source track filtering
        source_count = len(playlist_ids) + len(included_ids)
        n_candidates = max(max_songs * 5, source_count * 3, 500)
        neighbor_results = find_nearest_neighbors_by_vector(query_vector, n=n_candidates, eliminate_duplicates=True)
        logger.info(f"Extend: requested {n_candidates} candidates, got {len(neighbor_results)}, source_count={source_count}")

        # Filter results
        already_seen = set(playlist_ids) | set(included_ids) | set(excluded_ids)

        subtract_threshold = (config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
                              if config.PATH_DISTANCE_METRIC == 'angular'
                              else config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN)

        candidate_ids = [r['item_id'] for r in neighbor_results]
        metadata_list = get_score_data_by_ids(candidate_ids) if candidate_ids else []
        metadata_map = {m['item_id']: m for m in metadata_list}

        filtered_results = []
        for result in neighbor_results:
            item_id = result['item_id']
            distance = result.get('distance', 0)

            if item_id in already_seen:
                continue

            # Rating filter
            if min_rating is not None:
                meta = metadata_map.get(item_id, {})
                track_rating = meta.get('rating')
                if track_rating is None or track_rating < min_rating:
                    continue

            # Year range filter
            if year_min is not None or year_max is not None:
                meta = metadata_map.get(item_id, {})
                track_year = meta.get('year')
                if track_year is None or track_year <= 0:
                    continue
                if year_min is not None and track_year < year_min:
                    continue
                if year_max is not None and track_year > year_max:
                    continue

            # Excluded centroid proximity filter
            if excluded_centroid is not None:
                vec = get_vector_by_id(item_id)
                if vec is not None:
                    v_cand = np.array(vec, dtype=float)
                    if config.PATH_DISTANCE_METRIC == 'angular':
                        v1 = excluded_centroid / (np.linalg.norm(excluded_centroid) or 1.0)
                        v2 = v_cand / (np.linalg.norm(v_cand) or 1.0)
                        cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
                        dist_to_excluded = float(np.arccos(cosine) / np.pi)
                    else:
                        dist_to_excluded = float(np.linalg.norm(excluded_centroid - v_cand))

                    if dist_to_excluded < subtract_threshold:
                        continue

            if distance <= similarity_threshold:
                meta = metadata_map.get(item_id, {})
                result['album'] = meta.get('album')
                result['album_artist'] = meta.get('album_artist')
                result['year'] = meta.get('year')
                if not result.get('title'):
                    result['title'] = meta.get('title')
                if not result.get('author'):
                    result['author'] = meta.get('author')
                filtered_results.append(result)

            if len(filtered_results) >= max_songs:
                break

        # Source tracks metadata for drawer display
        source_tracks_meta = get_score_data_by_ids(playlist_ids) if playlist_ids else []

        # Annotate results with duplicate warnings against source tracks
        source_vectors = {}
        for sid in playlist_ids:
            vec = get_vector_by_id(str(sid))
            if vec is not None:
                v = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(v)
                source_vectors[sid] = v / norm if norm > 0 else v

        if source_vectors:
            source_meta_map = {m['item_id']: m for m in source_tracks_meta}
            for result in filtered_results:
                cand_vec = get_vector_by_id(result['item_id'])
                if cand_vec is None:
                    continue
                v_cand = np.array(cand_vec, dtype=np.float32)
                norm_cand = np.linalg.norm(v_cand)
                if norm_cand > 0:
                    v_cand = v_cand / norm_cand

                best_dist = float('inf')
                best_source_id = None
                for sid, v_src in source_vectors.items():
                    cosine = np.clip(np.dot(v_src, v_cand), -1.0, 1.0)
                    dist = float(1.0 - cosine)
                    if dist < best_dist:
                        best_dist = dist
                        best_source_id = sid

                if best_dist < duplicate_threshold and best_source_id is not None:
                    src_meta = source_meta_map.get(best_source_id, {})
                    result['duplicate_of'] = {
                        'item_id': best_source_id,
                        'title': src_meta.get('title'),
                        'author': src_meta.get('author'),
                        'album': src_meta.get('album'),
                        'distance': round(best_dist, 4)
                    }

        return jsonify({
            "results": filtered_results,
            "playlist_song_count": len(playlist_ids),
            "included_count": len(included_ids),
            "excluded_count": len(excluded_ids),
            "source_tracks": source_tracks_meta
        })

    except Exception as e:
        logger.exception("Playlist curator search failed")
        return jsonify({"error": "Internal error"}), 500


@playlist_curator_bp.route('/api/curator/save_playlist', methods=['POST'])
def save_playlist_api():
    """Save an extended/curated playlist to the configured media server."""
    from tasks.voyager_manager import create_playlist_from_ids

    payload = request.get_json() or {}
    new_playlist_name = payload.get('new_playlist_name')
    track_ids = payload.get('track_ids', [])

    if not new_playlist_name:
        return jsonify({"error": "Missing 'new_playlist_name'"}), 400
    if not track_ids:
        return jsonify({"error": "No tracks to save"}), 400

    try:
        str_ids = [str(tid) for tid in track_ids]

        # Deduplicate preserving order
        seen = set()
        final_ids = []
        for tid in str_ids:
            if tid not in seen:
                seen.add(tid)
                final_ids.append(tid)

        playlist_id = create_playlist_from_ids(new_playlist_name, final_ids)

        return jsonify({
            "message": f"Playlist '{new_playlist_name}' created with {len(final_ids)} songs!",
            "playlist_id": playlist_id,
            "total_songs": len(final_ids)
        }), 201

    except Exception as e:
        logger.exception("Save curator playlist failed")
        return jsonify({"error": "Internal error"}), 500


@playlist_curator_bp.route('/api/curator/server_playlists', methods=['GET'])
def server_playlists_api():
    """List playlists from the configured media server."""
    try:
        raw_playlists = _fetch_server_playlists()
        normalized = []
        for pl in (raw_playlists or []):
            pl_id = pl.get('Id') or pl.get('id', '')
            pl_name = pl.get('Name') or pl.get('name', 'Unknown')
            song_count = pl.get('songCount') or pl.get('ChildCount') or 0
            normalized.append({
                'playlist_id': str(pl_id),
                'playlist_name': pl_name,
                'song_count': int(song_count) if song_count else 0
            })
        return jsonify(normalized)
    except Exception as e:
        logger.exception("Failed to fetch server playlists")
        return jsonify({"error": "Internal error"}), 500


def _fetch_server_playlists():
    """Fetch playlists from the single configured media server."""
    try:
        mstype = config.MEDIASERVER_TYPE
        if mstype == 'jellyfin':
            from tasks.mediaserver_jellyfin import get_all_playlists
            return get_all_playlists()
        elif mstype == 'emby':
            from tasks.mediaserver_emby import get_all_playlists
            return get_all_playlists()
        elif mstype == 'navidrome':
            from tasks.mediaserver_navidrome import get_all_playlists
            raw = get_all_playlists() or []
            return [{'Id': p.get('id'), 'Name': p.get('name'), 'songCount': p.get('songCount', 0)} for p in raw]
        elif mstype == 'lyrion':
            from tasks.mediaserver_lyrion import get_all_playlists
            return get_all_playlists()
        elif mstype == 'mpd':
            return []  # MPD intentionally unsupported
        return []
    except Exception as e:
        logger.warning(f"_fetch_server_playlists failed for {config.MEDIASERVER_TYPE}: {e}")
        return []


@playlist_curator_bp.route('/api/curator/server_playlist_tracks', methods=['POST'])
def server_playlist_tracks_api():
    """Get tracks from a specific media-server playlist (analyzed tracks only)."""
    if config.MEDIASERVER_TYPE == 'mpd':
        return jsonify({"error": "MPD is not supported by the playlist curator"}), 501

    payload = request.get_json() or {}
    playlist_id = payload.get('playlist_id')

    if not playlist_id:
        return jsonify({"error": "Missing playlist_id"}), 400

    try:
        server_item_ids = _fetch_server_playlist_item_ids(playlist_id)
        if server_item_ids is None:
            return jsonify({"error": "Failed to fetch playlist tracks from server"}), 500
        if not server_item_ids:
            return jsonify({"error": "Playlist is empty"}), 404

        # Intersect with the score table - drop tracks not yet analyzed.
        # Main doesn't need provider_track resolution: the server's item_id IS score.item_id.
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT item_id FROM score WHERE item_id = ANY(%s)", (list(server_item_ids),))
        rows = cur.fetchall()
        cur.close()
        analyzed_set = {r[0] for r in rows}
        resolved_ids = [iid for iid in server_item_ids if iid in analyzed_set]

        if not resolved_ids:
            return jsonify({"error": "No tracks in this playlist have been analyzed yet"}), 404

        metadata_list = get_score_data_by_ids(resolved_ids)

        return jsonify({
            "tracks": metadata_list,
            "total_provider_tracks": len(server_item_ids),
            "resolved_tracks": len(resolved_ids),
            "unresolved_tracks": len(server_item_ids) - len(resolved_ids)
        })

    except Exception as e:
        logger.exception("Failed to fetch server playlist tracks")
        return jsonify({"error": "Internal error"}), 500


def _fetch_server_playlist_item_ids(playlist_id):
    """Fetch track item_ids from a playlist on the configured server.

    Returns list[str] on success, None on error.
    """
    try:
        mstype = config.MEDIASERVER_TYPE

        if mstype == 'jellyfin':
            base_url = config.JELLYFIN_URL.rstrip('/')
            url = f"{base_url}/Users/{config.JELLYFIN_USER_ID}/Items?ParentId={playlist_id}&IncludeItemTypes=Audio&Fields=Path"
            resp = http_requests.get(url, headers=config.HEADERS, timeout=30)
            if resp.status_code == 200:
                return [str(item['Id']) for item in resp.json().get('Items', [])]

        elif mstype == 'emby':
            base_url = config.EMBY_URL.rstrip('/')
            url = f"{base_url}/Users/{config.EMBY_USER_ID}/Items?ParentId={playlist_id}&IncludeItemTypes=Audio"
            headers = {'X-Emby-Token': config.EMBY_TOKEN}
            resp = http_requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return [str(item['Id']) for item in resp.json().get('Items', [])]

        elif mstype == 'navidrome':
            base_url = config.NAVIDROME_URL.rstrip('/')
            hex_pass = config.NAVIDROME_PASSWORD.encode('utf-8').hex() if config.NAVIDROME_PASSWORD else ''
            params = {
                "u": config.NAVIDROME_USER, "p": f"enc:{hex_pass}",
                "v": "1.16.1", "c": "AudioMuse-AI", "f": "json", "id": playlist_id
            }
            resp = http_requests.get(f"{base_url}/rest/getPlaylist.view", params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get('subsonic-response', {})
                if data.get('status') == 'ok' and 'playlist' in data:
                    entries = data['playlist'].get('entry', [])
                    return [str(e.get('id')) for e in entries if e.get('id')]

        elif mstype == 'lyrion':
            base_url = config.LYRION_URL.rstrip('/')
            payload = {"id": 1, "method": "slim.request",
                       "params": ["", ["playlists", "tracks", "0", "999999", f"playlist_id:{playlist_id}"]]}
            resp = http_requests.post(f"{base_url}/jsonrpc.js", json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json().get('result', {})
                if result and "playlisttracks_loop" in result:
                    return [str(t.get('id')) for t in result["playlisttracks_loop"] if t.get('id')]

        return None
    except Exception as e:
        logger.warning(f"Failed to fetch playlist tracks for {config.MEDIASERVER_TYPE}: {e}")
        return None


@playlist_curator_bp.route('/api/curator/stream/<path:item_id>', methods=['GET'])
def stream_track(item_id):
    """Redirect to media-server stream URL for the configured server."""
    from flask import redirect as flask_redirect
    try:
        mstype = config.MEDIASERVER_TYPE

        if mstype == 'jellyfin':
            base_url = config.JELLYFIN_URL.rstrip('/')
            return flask_redirect(f"{base_url}/Items/{item_id}/Download?api_key={config.JELLYFIN_TOKEN}")

        elif mstype == 'emby':
            base_url = config.EMBY_URL.rstrip('/')
            return flask_redirect(f"{base_url}/Items/{item_id}/Download?api_key={config.EMBY_TOKEN}")

        elif mstype == 'navidrome':
            base_url = config.NAVIDROME_URL.rstrip('/')
            hex_pass = config.NAVIDROME_PASSWORD.encode('utf-8').hex() if config.NAVIDROME_PASSWORD else ''
            return flask_redirect(
                f"{base_url}/rest/stream?id={item_id}&u={config.NAVIDROME_USER}"
                f"&p=enc:{hex_pass}&v=1.16.1&c=AudioMuse-AI&f=json"
            )

        elif mstype == 'lyrion':
            base_url = config.LYRION_URL.rstrip('/')
            return flask_redirect(f"{base_url}/music/{item_id}/download")

        elif mstype == 'mpd':
            return jsonify({"error": "MPD streaming is not supported by the playlist curator"}), 501

        return jsonify({"error": "Stream not supported for this media server type"}), 501

    except Exception as e:
        logger.exception(f"Stream failed for item_id={item_id}")
        return jsonify({"error": "Stream error"}), 500


@playlist_curator_bp.route('/api/curator/find_duplicates', methods=['POST'])
def find_duplicates_api():
    """Find duplicate tracks in a set using embedding similarity."""
    payload = request.get_json() or {}
    track_ids = payload.get('track_ids', [])
    threshold = payload.get('threshold', 0.05)

    if not track_ids:
        return jsonify({"error": "No track_ids provided"}), 400
    if len(track_ids) > 2000:
        return jsonify({"error": "Too many tracks (max 2000)"}), 400

    try:
        threshold = max(0.005, min(float(threshold), 0.3))
    except (TypeError, ValueError):
        threshold = 0.05

    str_ids = [str(tid) for tid in track_ids if tid is not None]

    result = _find_duplicate_groups(str_ids, threshold=threshold)
    return jsonify(result)
