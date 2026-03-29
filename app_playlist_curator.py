from flask import Blueprint, jsonify, request, render_template, Response
import logging
import os
import re
import numpy as np
import requests as http_requests
from psycopg2.extras import DictCursor

from tasks.voyager_manager import find_nearest_neighbors_by_vector, get_vector_by_id
from app_helper import get_db, get_score_data_by_ids, get_primary_provider_id, get_item_id_for_provider
import config

logger = logging.getLogger(__name__)

playlist_curator_bp = Blueprint('playlist_curator_bp', __name__, template_folder='templates')

VALID_WEIGHTS = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}


def _sanitize_weights(weights_dict):
    """Ensure weights are valid integers from allowed set."""
    sanitized = {}
    for k, v in weights_dict.items():
        try:
            w = int(v)
            sanitized[str(k)] = w if w in VALID_WEIGHTS else 1
        except (ValueError, TypeError):
            sanitized[str(k)] = 1
    return sanitized


def _compute_centroid_from_ids(ids, weights=None):
    """
    Fetch vectors by track_id and compute their weighted centroid.

    Args:
        ids: List of track_ids (ints or strings)
        weights: Optional dict mapping str(track_id) -> weight (1-1024)

    Returns:
        Weighted mean vector, or None if no valid vectors found.
    """
    if weights is None:
        weights = {}

    vectors = []
    weight_values = []

    for track_id in ids:
        vec = get_vector_by_id(int(track_id))
        if vec is not None:
            vectors.append(np.array(vec, dtype=float))
            w = weights.get(str(track_id), 1)
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
            elif field == 'features':
                # other_features is comma-separated: "danceable, aggressive, happy"
                clauses.append(f"{db_col} ILIKE %s")
                params.append(f"%{value}%")
            else:
                clauses.append(f"{db_col} = %s")
                params.append(value)
        elif operator == 'is_not':
            if field in ('mood', 'genre'):
                clauses.append(f"{db_col} !~ %s")
                params.append(f"(^|,)\\s*{re.escape(value)}:")
            elif field == 'features':
                clauses.append(f"{db_col} NOT ILIKE %s")
                params.append(f"%{value}%")
            else:
                clauses.append(f"{db_col} != %s")
                params.append(value)
        elif operator in ('greater_than', 'less_than'):
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


def _find_duplicate_groups(track_ids, threshold=0.015):
    """
    Find duplicate groups in a set of tracks using embedding cosine distance.

    Args:
        track_ids: List of canonical track_id integers
        threshold: Cosine distance threshold (0.01=strict, 0.15=loose)

    Returns:
        Dict with 'groups', 'total_groups', 'total_duplicate_tracks'
    """
    from collections import defaultdict

    # Fetch vectors, skip tracks without embeddings
    valid_ids = []
    vectors = []
    for tid in track_ids:
        vec = get_vector_by_id(int(tid))
        if vec is not None:
            valid_ids.append(int(tid))
            vectors.append(np.array(vec, dtype=np.float32))

    if len(vectors) < 2:
        return {"groups": [], "total_groups": 0, "total_duplicate_tracks": 0}

    # Build matrix and compute pairwise cosine distances
    V = np.vstack(vectors)  # (N, 200)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
    V_normed = V / norms
    similarity_matrix = V_normed @ V_normed.T
    np.clip(similarity_matrix, -1.0, 1.0, out=similarity_matrix)
    distance_matrix = 1.0 - similarity_matrix

    # Union-find to cluster duplicates
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

    # Group by root
    clusters = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    # Filter to groups with 2+ members
    duplicate_groups = [indices for indices in clusters.values() if len(indices) >= 2]

    if not duplicate_groups:
        return {"groups": [], "total_groups": 0, "total_duplicate_tracks": 0}

    # Fetch metadata for all tracks in duplicate groups
    all_dup_ids = []
    for indices in duplicate_groups:
        for idx in indices:
            all_dup_ids.append(valid_ids[idx])

    metadata_list = get_score_data_by_ids(all_dup_ids)
    metadata_map = {m['track_id']: m for m in metadata_list}

    # Build position map for playlist-order scoring
    position_map = {tid: pos for pos, tid in enumerate(track_ids)}
    total_tracks = len(track_ids)

    # Score and build response groups
    groups = []
    total_duplicate_tracks = 0

    for indices in duplicate_groups:
        group_tracks = []
        for idx in indices:
            tid = valid_ids[idx]
            meta = metadata_map.get(tid, {})

            # Composite score: rating(w3) + completeness(w2) + oldest_year(w3) + position(w0.1)
            # Older albums strongly preferred — newer are often compilations
            rating_score = ((meta.get('rating') or 0) / 5.0) * 3.0
            completeness = sum(1 for f in ['album', 'year', 'album_artist'] if meta.get(f) is not None)
            completeness_score = (completeness / 3.0) * 2.0
            year = meta.get('year')
            year_score = ((2050 - year) / 100.0 * 3.0) if year and year > 1900 else 0.0
            pos = position_map.get(tid, total_tracks)
            position_score = (1.0 - (pos / max(total_tracks, 1))) * 0.1

            score = round(rating_score + completeness_score + year_score + position_score, 2)

            group_tracks.append({
                'item_id': str(tid),
                'title': meta.get('title'),
                'author': meta.get('author'),
                'album': meta.get('album'),
                'album_artist': meta.get('album_artist'),
                'year': meta.get('year'),
                'rating': meta.get('rating'),
                'score': score
            })

        # Sort by score descending — first is suggested keeper
        group_tracks.sort(key=lambda t: t['score'], reverse=True)
        groups.append({'tracks': group_tracks})
        total_duplicate_tracks += len(group_tracks)

    return {
        "groups": groups,
        "total_groups": len(groups),
        "total_duplicate_tracks": total_duplicate_tracks
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

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
            SELECT DISTINCT TRIM(feature) as feature_label
            FROM (
                SELECT UNNEST(STRING_TO_ARRAY(other_features, ',')) as feature
                FROM score WHERE other_features IS NOT NULL AND other_features != ''
            ) t
            WHERE TRIM(feature) != ''
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
    search_only=true  → Smart Search (returns filter matches)
    search_only=false → Extend mode (weighted centroid + neighbors)
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
    included_ids = payload.get('included_ids', [])
    excluded_ids = payload.get('excluded_ids', [])
    min_rating = payload.get('min_rating')
    year_min = payload.get('year_min')
    year_max = payload.get('year_max')
    search_only = payload.get('search_only', False)
    source_ids = payload.get('source_ids', [])

    source_weights = _sanitize_weights(payload.get('source_weights', {}))
    included_weights = _sanitize_weights(payload.get('included_weights', {}))

    if not playlist_name and not filters and not source_ids:
        return jsonify({"error": "Missing 'playlist_name', 'filters', or 'source_ids'"}), 400

    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)
        playlist_ids = []

        if playlist_name:
            cur.execute("SELECT track_id FROM playlist WHERE playlist_name = %s", (playlist_name,))
            rows = cur.fetchall()
            playlist_ids = [row['track_id'] for row in rows]
            if not playlist_ids:
                cur.close()
                return jsonify({"error": f"Playlist '{playlist_name}' not found or is empty"}), 404

        elif filters:
            where_clause, params = _build_filter_query(filters, match_mode)
            cur.execute(f"SELECT track_id FROM score WHERE {where_clause}", tuple(params))
            rows = cur.fetchall()
            playlist_ids = [row['track_id'] for row in rows]
            if not playlist_ids:
                cur.close()
                return jsonify({"error": "No songs found matching the filters"}), 404

        elif source_ids:
            playlist_ids = [int(sid) for sid in source_ids]

        cur.close()

        # ── SEARCH ONLY MODE ────────────────────────────────────────────
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

        # ── EXTEND MODE ─────────────────────────────────────────────────

        # Combine source + included for positive centroid
        all_ids_for_centroid = list(set(playlist_ids + [int(i) for i in included_ids]))

        combined_weights = {}
        for pid in playlist_ids:
            combined_weights[str(pid)] = source_weights.get(str(pid), 1)
        for inc_id in included_ids:
            combined_weights[str(inc_id)] = included_weights.get(str(inc_id), 1)

        positive_centroid = _compute_centroid_from_ids(all_ids_for_centroid, combined_weights)
        if positive_centroid is None:
            return jsonify({"error": "Failed to compute playlist centroid — no valid embeddings found"}), 500

        # Excluded centroid (unweighted)
        excluded_centroid = None
        if excluded_ids:
            excluded_centroid = _compute_centroid_from_ids([int(i) for i in excluded_ids])

        # Adjust query vector
        query_vector = positive_centroid
        if excluded_centroid is not None:
            query_vector = positive_centroid - (excluded_centroid * 0.5)

        # Find similar songs — request more candidates to account for source track filtering
        source_count = len(playlist_ids) + len(included_ids)
        n_candidates = max(max_songs * 5, source_count * 3, 500)
        neighbor_results = find_nearest_neighbors_by_vector(query_vector, n=n_candidates, eliminate_duplicates=True)
        logger.info(f"Extend: requested {n_candidates} candidates, got {len(neighbor_results)}, source_count={source_count}")

        # Filter results
        already_seen = set(str(tid) for tid in playlist_ids) | set(str(i) for i in included_ids) | set(str(i) for i in excluded_ids)

        subtract_threshold = (config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
                              if config.PATH_DISTANCE_METRIC == 'angular'
                              else config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN)

        candidate_ids = [r['track_id'] for r in neighbor_results]
        metadata_list = get_score_data_by_ids(candidate_ids)
        metadata_map = {m['track_id']: m for m in metadata_list}

        filtered_results = []
        for result in neighbor_results:
            track_id = result['track_id']
            item_id = result['item_id']  # str(track_id)
            distance = result.get('distance', 0)

            if item_id in already_seen:
                continue

            # Rating filter
            if min_rating is not None:
                meta = metadata_map.get(track_id, {})
                track_rating = meta.get('rating')
                if track_rating is None or track_rating < min_rating:
                    continue

            # Year range filter
            if year_min is not None or year_max is not None:
                meta = metadata_map.get(track_id, {})
                track_year = meta.get('year')
                if track_year is None or track_year <= 0:
                    continue
                if year_min is not None and track_year < year_min:
                    continue
                if year_max is not None and track_year > year_max:
                    continue

            # Excluded centroid proximity filter
            if excluded_centroid is not None:
                vec = get_vector_by_id(track_id)
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
                meta = metadata_map.get(track_id, {})
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
    """Save an extended/curated playlist to media server(s)."""
    from tasks.voyager_manager import create_playlist_from_ids

    payload = request.get_json() or {}
    new_playlist_name = payload.get('new_playlist_name')
    track_ids = payload.get('track_ids', [])
    provider_ids = payload.get('provider_ids')

    if not new_playlist_name:
        return jsonify({"error": "Missing 'new_playlist_name'"}), 400
    if not track_ids:
        return jsonify({"error": "No tracks to save"}), 400

    try:
        int_track_ids = [int(tid) for tid in track_ids]

        # Deduplicate preserving order
        seen = set()
        final_ids = []
        for tid in int_track_ids:
            if tid not in seen:
                seen.add(tid)
                final_ids.append(tid)

        result = create_playlist_from_ids(new_playlist_name, final_ids, provider_ids=provider_ids)

        return jsonify({
            "message": f"Playlist '{new_playlist_name}' created with {len(final_ids)} songs!",
            "playlist_id": result if isinstance(result, str) else None,
            "provider_results": result if isinstance(result, dict) else None,
            "total_songs": len(final_ids)
        }), 201

    except Exception as e:
        logger.exception("Save curator playlist failed")
        return jsonify({"error": "Internal error"}), 500


@playlist_curator_bp.route('/api/curator/provider_playlists', methods=['GET'])
def provider_playlists_api():
    """List playlists from a specific provider or all enabled providers."""
    from app_setup import get_providers_raw, get_provider_by_id

    provider_id = request.args.get('provider_id')

    try:
        if provider_id:
            prov = get_provider_by_id(int(provider_id))
            if not prov or not prov.get('enabled'):
                return jsonify({"error": "Provider not found or disabled"}), 404
            providers = [prov]
        else:
            providers = get_providers_raw(enabled_only=True)

        all_playlists = []

        for prov in providers:
            p_id = prov['id']
            p_type = prov['provider_type']
            p_name = prov.get('name', p_type)
            cfg = prov.get('config', {}) or {}

            try:
                raw_playlists = _fetch_provider_playlists(p_type, cfg)
                for pl in (raw_playlists or []):
                    pl_id = pl.get('Id') or pl.get('id', '')
                    pl_name = pl.get('Name') or pl.get('name', 'Unknown')
                    song_count = pl.get('songCount') or pl.get('MediaSourceCount') or pl.get('ChildCount') or 0
                    all_playlists.append({
                        'provider_id': p_id,
                        'provider_name': p_name,
                        'provider_type': p_type,
                        'playlist_id': str(pl_id),
                        'playlist_name': pl_name,
                        'song_count': int(song_count) if song_count else 0
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch playlists from provider {p_name}: {e}")
                continue

        return jsonify(all_playlists)

    except Exception as e:
        logger.exception("Failed to fetch provider playlists")
        return jsonify({"error": "Internal error"}), 500


def _fetch_provider_playlists(provider_type, provider_config):
    """Fetch playlist list from a provider using DB-stored credentials."""
    try:
        if provider_type in ('jellyfin', 'emby'):
            base_url = provider_config.get('url') or (config.JELLYFIN_URL if provider_type == 'jellyfin' else config.EMBY_URL)
            token = provider_config.get('token') or provider_config.get('api_key') or (config.JELLYFIN_TOKEN if provider_type == 'jellyfin' else config.EMBY_TOKEN)
            user_id = provider_config.get('user_id') or (config.JELLYFIN_USER_ID if provider_type == 'jellyfin' else config.EMBY_USER_ID)
            url = f"{base_url.rstrip('/')}/Users/{user_id}/Items"
            params = {"IncludeItemTypes": "Playlist", "Recursive": True}
            headers = {'X-Emby-Token': token} if provider_type == 'emby' else {'Authorization': f'MediaBrowser Token="{token}"'}
            resp = http_requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get('Items', [])

        elif provider_type == 'navidrome':
            base_url = provider_config.get('url') or config.NAVIDROME_URL
            user = provider_config.get('user') or provider_config.get('username') or config.NAVIDROME_USER
            password = provider_config.get('password') or config.NAVIDROME_PASSWORD
            if not user or not password:
                logger.warning("Navidrome credentials not available for playlist fetch")
                return []
            hex_pass = password.encode('utf-8').hex()
            params = {"u": user, "p": f"enc:{hex_pass}", "v": "1.16.1", "c": "AudioMuse-AI", "f": "json"}
            resp = http_requests.get(f"{base_url.rstrip('/')}/rest/getPlaylists.view", params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get('subsonic-response', {})
                if data.get('status') == 'ok' and 'playlists' in data:
                    pls = data['playlists'].get('playlist', [])
                    return [{'Id': p.get('id'), 'Name': p.get('name'), 'songCount': p.get('songCount', 0)} for p in pls]

        elif provider_type == 'lyrion':
            base_url = provider_config.get('url') or config.LYRION_URL
            payload = {"id": 1, "method": "slim.request", "params": ["", ["playlists", 0, 999999]]}
            resp = http_requests.post(f"{base_url.rstrip('/')}/jsonrpc.js", json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json().get('result', {})
                return [{'Id': p.get('id'), 'Name': p.get('playlist')} for p in result.get('playlists_loop', [])]

        elif provider_type == 'localfiles':
            from tasks.mediaserver_localfiles import get_all_playlists
            return get_all_playlists()

        return []
    except Exception as e:
        logger.warning(f"_fetch_provider_playlists failed for {provider_type}: {e}")
        return []


@playlist_curator_bp.route('/api/curator/provider_playlist_tracks', methods=['POST'])
def provider_playlist_tracks_api():
    """Get tracks from a specific provider playlist, resolved to canonical track_ids."""
    payload = request.get_json() or {}
    provider_id = payload.get('provider_id')
    playlist_id = payload.get('playlist_id')

    if not provider_id or not playlist_id:
        return jsonify({"error": "Missing provider_id or playlist_id"}), 400

    try:
        from app_setup import get_provider_by_id
        provider = get_provider_by_id(int(provider_id))
        if not provider:
            return jsonify({"error": "Provider not found"}), 404

        p_type = provider['provider_type']
        cfg = provider.get('config', {}) or {}

        # Fetch playlist tracks from provider
        item_ids = _fetch_playlist_item_ids(p_type, playlist_id, cfg)
        if item_ids is None:
            return jsonify({"error": "Failed to fetch playlist tracks from provider"}), 500
        if not item_ids:
            return jsonify({"error": "Playlist is empty"}), 404

        # Resolve provider item_ids to canonical track_ids via provider_track table
        db = get_db()
        cur = db.cursor()
        cur.execute("""
            SELECT item_id, track_id FROM provider_track
            WHERE provider_id = %s AND item_id = ANY(%s)
        """, (int(provider_id), list(item_ids)))
        rows = cur.fetchall()
        cur.close()

        item_to_track = {row[0]: row[1] for row in rows}
        resolved_track_ids = [item_to_track[iid] for iid in item_ids if iid in item_to_track]

        if not resolved_track_ids:
            return jsonify({"error": "No tracks in this playlist have been analyzed yet"}), 404

        # Fetch metadata for resolved tracks
        metadata_list = get_score_data_by_ids(resolved_track_ids)

        return jsonify({
            "tracks": metadata_list,
            "total_provider_tracks": len(item_ids),
            "resolved_tracks": len(resolved_track_ids),
            "unresolved_tracks": len(item_ids) - len(resolved_track_ids)
        })

    except Exception as e:
        logger.exception("Failed to fetch provider playlist tracks")
        return jsonify({"error": "Internal error"}), 500


def _fetch_playlist_item_ids(provider_type, playlist_id, provider_config):
    """Fetch track item_ids from a provider playlist. Returns list of str item_ids or None on error."""
    try:
        if provider_type in ('jellyfin', 'emby'):
            base_url = provider_config.get('url') or (config.JELLYFIN_URL if provider_type == 'jellyfin' else config.EMBY_URL)
            token = provider_config.get('token') or provider_config.get('api_key') or (config.JELLYFIN_TOKEN if provider_type == 'jellyfin' else config.EMBY_TOKEN)
            user_id = provider_config.get('user_id') or (config.JELLYFIN_USER_ID if provider_type == 'jellyfin' else config.EMBY_USER_ID)
            url = f"{base_url.rstrip('/')}/Users/{user_id}/Items?ParentId={playlist_id}&IncludeItemTypes=Audio&Fields=Path"
            headers = {'X-Emby-Token': token} if provider_type == 'emby' else {'Authorization': f'MediaBrowser Token="{token}"'}
            resp = http_requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                items = resp.json().get('Items', [])
                return [str(item['Id']) for item in items]

        elif provider_type == 'navidrome':
            base_url = provider_config.get('url') or config.NAVIDROME_URL
            user = provider_config.get('user') or provider_config.get('username') or config.NAVIDROME_USER
            password = provider_config.get('password') or config.NAVIDROME_PASSWORD
            if not user or not password:
                return None
            hex_pass = password.encode('utf-8').hex()
            params = {"u": user, "p": f"enc:{hex_pass}", "v": "1.16.1", "c": "AudioMuse-AI", "f": "json", "id": playlist_id}
            resp = http_requests.get(f"{base_url.rstrip('/')}/rest/getPlaylist.view", params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json().get('subsonic-response', {})
                if data.get('status') == 'ok' and 'playlist' in data:
                    entries = data['playlist'].get('entry', [])
                    return [str(e.get('id')) for e in entries if e.get('id')]

        elif provider_type == 'lyrion':
            base_url = provider_config.get('url') or config.LYRION_URL
            payload = {"id": 1, "method": "slim.request", "params": ["", ["playlists", "tracks", "0", "999999", f"playlist_id:{playlist_id}"]]}
            resp = http_requests.post(f"{base_url.rstrip('/')}/jsonrpc.js", json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json().get('result', {})
            if result and "playlisttracks_loop" in result:
                return [str(t.get('id')) for t in result["playlisttracks_loop"] if t.get('id')]

        elif provider_type == 'localfiles':
            from tasks.mediaserver_localfiles import get_all_playlists
            playlists = get_all_playlists()
            for pl in playlists:
                if pl.get('Id') == playlist_id or pl.get('Name') == playlist_id:
                    # Parse M3U to get file paths, then resolve to item_ids
                    path = pl.get('Path')
                    if path and os.path.isfile(path):
                        import os
                        tracks = []
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    tracks.append(line)
                        return tracks

        return None
    except Exception as e:
        logger.warning(f"Failed to fetch playlist tracks from {provider_type}: {e}")
        return None


@playlist_curator_bp.route('/api/curator/stream/<int:track_id>', methods=['GET'])
def stream_track(track_id):
    """Redirect to direct media server stream URL for instant playback.
    For API-based providers, redirects to the provider's download/stream endpoint.
    For localfiles, serves directly from disk.
    """
    from flask import redirect as flask_redirect
    try:
        primary_provider_id = get_primary_provider_id()
        provider_item_id = get_item_id_for_provider(track_id, primary_provider_id) if primary_provider_id else None
        if not provider_item_id:
            provider_item_id = str(track_id)

        from app_setup import get_provider_by_id
        provider = get_provider_by_id(primary_provider_id) if primary_provider_id else None

        if provider:
            p_type = provider['provider_type']
            cfg = provider.get('config', {}) or {}
        else:
            p_type = config.MEDIASERVER_TYPE
            cfg = {}

        if p_type in ('jellyfin', 'emby'):
            base_url = cfg.get('url') or config.JELLYFIN_URL or config.EMBY_URL
            token = cfg.get('token') or cfg.get('api_key') or config.JELLYFIN_TOKEN or config.EMBY_TOKEN
            return flask_redirect(f"{base_url.rstrip('/')}/Items/{provider_item_id}/Download?api_key={token}")

        elif p_type == 'navidrome':
            base_url = cfg.get('url') or config.NAVIDROME_URL
            user = cfg.get('user') or cfg.get('username') or config.NAVIDROME_USER
            password = cfg.get('password') or config.NAVIDROME_PASSWORD
            hex_pass = password.encode('utf-8').hex() if password else ''
            return flask_redirect(f"{base_url.rstrip('/')}/rest/stream?id={provider_item_id}&u={user}&p=enc:{hex_pass}&v=1.16.1&c=AudioMuse-AI&f=json")

        elif p_type == 'lyrion':
            base_url = cfg.get('url') or config.LYRION_URL
            return flask_redirect(f"{base_url.rstrip('/')}/music/{provider_item_id}/download")

        elif p_type == 'localfiles':
            db = get_db()
            cur = db.cursor()
            cur.execute("SELECT file_path FROM score WHERE track_id = %s", (track_id,))
            row = cur.fetchone()
            cur.close()
            if row and row[0]:
                file_path = row[0]
                if os.path.isfile(file_path):
                    import mimetypes
                    mime = mimetypes.guess_type(file_path)[0] or 'audio/mpeg'
                    file_size = os.path.getsize(file_path)

                    def generate():
                        with open(file_path, 'rb') as f:
                            while True:
                                chunk = f.read(65536)
                                if not chunk:
                                    break
                                yield chunk

                    return Response(generate(), mimetype=mime,
                                    headers={'Content-Length': str(file_size),
                                             'Accept-Ranges': 'bytes'})
            return jsonify({"error": "File not found"}), 404

        return jsonify({"error": "Could not construct stream URL for provider"}), 500

    except Exception as e:
        logger.exception(f"Stream failed for track_id={track_id}")
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

    int_ids = []
    for tid in track_ids:
        try:
            int_ids.append(int(tid))
        except (TypeError, ValueError):
            continue

    result = _find_duplicate_groups(int_ids, threshold=threshold)
    return jsonify(result)
