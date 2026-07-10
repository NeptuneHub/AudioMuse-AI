# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for playlist curation and smart-search workflows.

Provides pages and API endpoints for searching analyzed tracks, extending
playlists from embedding centroids, reviewing duplicates, and saving playlists
through supported media servers.

Main Features:
* Smart-search filters over score metadata and analysis features.
* Weighted embedding-centroid playlist extension and duplicate detection.
* Media-server playlist loading, preview, and persistence routes.
"""

from flask import Blueprint, jsonify, request, render_template, Response, stream_with_context, redirect, url_for
import logging
import re
import numpy as np
import requests as http_requests
from psycopg2.extras import DictCursor

from tasks.ivf_manager import find_nearest_neighbors_by_vector, get_vector_by_id, get_vectors_by_ids
from app_helper import get_db, get_score_data_by_ids, get_score_data_lite_by_ids
import config

logger = logging.getLogger(__name__)

playlist_curator_bp = Blueprint('playlist_curator_bp', __name__, template_folder='templates')
INTERNAL_ERROR_MESSAGE = "Internal error"

INFLUENCE_LEVELS = {
    0: 0.0,    # x1 - equal weight
    1: 0.05,   # Boost - ~5% of centroid
    2: 0.15,   # Strong - ~15% of centroid
    3: 0.30,   # Focus - ~30% of centroid
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


def _compute_centroid_from_ids(ids, weights=None, vector_cache=None):
    """
    Fetch vectors by item_id and compute their weighted centroid.

    Args:
        ids: List of item_ids (strings)
        weights: Optional dict mapping str(item_id) -> weight (1-1024)
        vector_cache: Optional dict mapping str(item_id) -> np.ndarray for
            pre-fetched vectors. When provided, no Voyager calls are made.

    Returns:
        Weighted mean vector, or None if no valid vectors found.
    """
    if weights is None:
        weights = {}
    if vector_cache is None:
        vector_cache = get_vectors_by_ids([str(i) for i in ids])

    vectors = []
    weight_values = []

    for item_id in ids:
        sid = str(item_id)
        vec = vector_cache.get(sid)
        if vec is not None:
            vectors.append(np.array(vec, dtype=float))
            w = weights.get(sid, 1)
            weight_values.append(max(1, w))

    if not vectors:
        return None

    vectors_array = np.array(vectors)
    weights_array = np.array(weight_values, dtype=float)
    return np.sum(vectors_array * weights_array[:, np.newaxis], axis=0) / np.sum(weights_array)


_FILTER_FIELD_MAP = {
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
_RANGE_FILTER_FIELDS = {'bpm', 'energy', 'year', 'decade', 'rating'}
_LABEL_FILTER_FIELDS = {'features', 'genre', 'mood'}


def _range_filter_clause(field, db_column, value):
    """Build a bounded numeric range clause when the value contains a range."""
    if field not in _RANGE_FILTER_FIELDS or '-' not in str(value):
        return None
    try:
        minimum, maximum = (float(part) for part in value.split('-', 1))
    except (TypeError, ValueError):
        return None
    if field == 'energy':
        energy_span = config.ENERGY_MAX - config.ENERGY_MIN
        minimum = config.ENERGY_MIN + minimum * energy_span
        maximum = config.ENERGY_MIN + maximum * energy_span
    return f"({db_column} >= %s AND {db_column} <= %s)", [minimum, maximum]


def _identity_filter_clause(field, db_column, operator, value):
    """Build equality or inequality clauses, including label-aware regex matching."""
    symbol = '~' if operator == 'is' else '!~'
    if field in ('mood', 'genre'):
        return f"{db_column} {symbol} %s", [f"(^|,)\\s*{re.escape(value)}:"]
    symbol = '=' if operator == 'is' else '!='
    return f"{db_column} {symbol} %s", [value]


def _comparison_filter_clause(field, db_column, operator, value):
    """Build numeric or label-score comparison clauses."""
    is_greater = operator == 'greater_than'
    if field in _LABEL_FILTER_FIELDS and ':' in str(value):
        label, raw_threshold = value.rsplit(':', 1)
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            return None
        symbol = '>=' if is_greater else '<='
        clause = f"""EXISTS (
                    SELECT 1 FROM UNNEST(STRING_TO_ARRAY({db_column}, ',')) AS f
                    WHERE TRIM(SPLIT_PART(f, ':', 1)) = %s
                    AND CAST(SPLIT_PART(f, ':', 2) AS FLOAT) {symbol} %s
                )"""
        return clause, [label.strip(), threshold]
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    symbol = '>' if is_greater else '<'
    return f"{db_column} {symbol} %s", [numeric_value]


def _build_filter_clause(filter_item):
    """Translate one allowlisted smart-search filter into SQL and parameters."""
    field = filter_item.get('field')
    operator = filter_item.get('operator')
    value = filter_item.get('value')
    db_column = _FILTER_FIELD_MAP.get(field)
    if not db_column:
        return None

    range_clause = _range_filter_clause(field, db_column, value)
    if range_clause:
        return range_clause
    if operator in ('contains', 'does_not_contain'):
        sql_operator = 'ILIKE' if operator == 'contains' else 'NOT ILIKE'
        return f"{db_column} {sql_operator} %s", [f"%{value}%"]
    if operator in ('is', 'is_not'):
        return _identity_filter_clause(field, db_column, operator, value)
    if operator in ('greater_than', 'less_than'):
        return _comparison_filter_clause(field, db_column, operator, value)
    return None


def _build_filter_query(filters, match_mode='all'):
    """Build a parameterized SQL WHERE clause from smart-search filters."""
    built_clauses = [_build_filter_clause(item) for item in (filters or [])]
    valid_clauses = [item for item in built_clauses if item]
    if not valid_clauses:
        return "1=1", []
    clauses = [clause for clause, _ in valid_clauses]
    params = [param for _, values in valid_clauses for param in values]
    join_operator = " AND " if match_mode == 'all' else " OR "
    return f"({join_operator.join(clauses)})", params


def _normalized_duplicate_vectors(item_ids):
    """Return ids and normalized vectors for tracks with embeddings."""
    string_ids = [str(item_id) for item_id in item_ids]
    vector_cache = get_vectors_by_ids(string_ids)
    valid_ids = []
    vectors = []
    for item_id in string_ids:
        vector = vector_cache.get(item_id)
        if vector is not None:
            valid_ids.append(item_id)
            vectors.append(np.array(vector, dtype=np.float32))
    if len(vectors) < 2:
        return valid_ids, None
    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return valid_ids, matrix / np.where(norms == 0, 1.0, norms)


def _duplicate_index_groups(normalized_vectors, threshold):
    """Cluster vector indices whose cosine distance falls below the threshold."""
    from collections import defaultdict

    count = len(normalized_vectors)
    parents = list(range(count))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[left_root] = right_root

    distances = 1.0 - np.clip(normalized_vectors @ normalized_vectors.T, -1.0, 1.0)
    for left in range(count):
        for right in range(left + 1, count):
            if distances[left, right] < threshold:
                union(left, right)

    clusters = defaultdict(list)
    for index in range(count):
        clusters[find(index)].append(index)
    return [indices for indices in clusters.values() if len(indices) >= 2]


def _duplicate_track_score(metadata, position, total_tracks):
    """Score which copy should be retained within one duplicate group."""
    rating_score = ((metadata.get('rating') or 0) / 5.0) * 3.0
    complete_fields = sum(
        metadata.get(field) is not None for field in ('album', 'year', 'album_artist')
    )
    completeness_score = (complete_fields / 3.0) * 2.0
    year = metadata.get('year')
    year_score = ((2050 - year) / 100.0 * 3.0) if year and year > 1900 else 0.0
    position_score = (1.0 - (position / max(total_tracks, 1))) * 0.1
    return round(rating_score + completeness_score + year_score + position_score, 2)


def _duplicate_track(item_id, metadata, position_map, total_tracks):
    """Create the response model for one duplicate track."""
    return {
        'item_id': item_id,
        'title': metadata.get('title'),
        'author': metadata.get('author'),
        'album': metadata.get('album'),
        'album_artist': metadata.get('album_artist'),
        'year': metadata.get('year'),
        'rating': metadata.get('rating'),
        'score': _duplicate_track_score(
            metadata, position_map.get(item_id, total_tracks), total_tracks
        ),
    }


def _duplicate_response_groups(index_groups, valid_ids, item_ids):
    """Load metadata, rank each duplicate group, and build the API response."""
    duplicate_ids = [valid_ids[index] for group in index_groups for index in group]
    metadata_map = {
        metadata['item_id']: metadata for metadata in get_score_data_by_ids(duplicate_ids)
    }
    position_map = {item_id: position for position, item_id in enumerate(item_ids)}
    groups = []
    for index_group in index_groups:
        tracks = [
            _duplicate_track(
                valid_ids[index],
                metadata_map.get(valid_ids[index], {}),
                position_map,
                len(item_ids),
            )
            for index in index_group
        ]
        tracks.sort(key=lambda track: track['score'], reverse=True)
        groups.append({'tracks': tracks})
    return groups


def _find_duplicate_groups(item_ids, threshold=0.015):
    """Find duplicate track groups using embedding cosine distance."""
    valid_ids, normalized_vectors = _normalized_duplicate_vectors(item_ids)
    if normalized_vectors is None:
        groups = []
    else:
        index_groups = _duplicate_index_groups(normalized_vectors, threshold)
        groups = (
            _duplicate_response_groups(index_groups, valid_ids, item_ids)
            if index_groups
            else []
        )
    return {
        "groups": groups,
        "total_groups": len(groups),
        "total_duplicate_tracks": sum(len(group['tracks']) for group in groups),
    }

# --- Routes -----------------------------------------------------------------

@playlist_curator_bp.route('/playlist_curator', methods=['GET'])
def playlist_curator_page():
    """Backwards-compatible redirect: legacy /playlist_curator URL now lands on Smart Search."""
    return redirect(url_for('playlist_curator_bp.smart_search_page'), code=302)


@playlist_curator_bp.route('/playlist_curator/search', methods=['GET'])
def smart_search_page():
    return render_template('playlist_curator_search.html',
                           title='AudioMuse-AI - Smart Search',
                           active='smart_search',
                           active_tool='search')


@playlist_curator_bp.route('/playlist_curator/extender', methods=['GET'])
def playlist_extender_page():
    return render_template('playlist_curator_extender.html',
                           title='AudioMuse-AI - Playlist Extender',
                           active='playlist_extender',
                           active_tool='extender')


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


def _bounded_integer(value, default, minimum, maximum):
    """Coerce a request integer and clamp it to the accepted range."""
    try:
        return min(max(minimum, int(value)), maximum)
    except (TypeError, ValueError):
        return default


def _search_duplicate_threshold(value):
    """Normalize duplicate detection threshold; out-of-range values disable it."""
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.01
    if threshold <= 0 or threshold >= 1.0:
        return 0
    return max(0.005, min(threshold, 0.3))


def _parse_search_options(payload):
    """Normalize Playlist Curator search and extend request options."""
    return {
        'playlist_name': payload.get('playlist_name'),
        'filters': payload.get('filters'),
        'match_mode': payload.get('match_mode', 'all'),
        'max_songs': _bounded_integer(payload.get('max_songs', 50), 50, 1, 500),
        'similarity_threshold': payload.get('similarity_threshold', 0.5),
        'included_ids': [str(item) for item in payload.get('included_ids', [])],
        'excluded_ids': [str(item) for item in payload.get('excluded_ids', [])],
        'min_rating': payload.get('min_rating'),
        'year_min': payload.get('year_min'),
        'year_max': payload.get('year_max'),
        'search_only': payload.get('search_only', False),
        'source_ids': [str(item) for item in payload.get('source_ids', [])],
        'page': _bounded_integer(payload.get('page', 1), 1, 1, 2 ** 31 - 1),
        'per_page': _bounded_integer(payload.get('per_page', 500), 500, 1, 2000),
        'duplicate_threshold': _search_duplicate_threshold(
            payload.get('duplicate_threshold', 0.01)
        ),
        'source_levels': _sanitize_levels(payload.get('source_weights', {})),
        'included_levels': _sanitize_levels(payload.get('included_weights', {})),
    }


def _playlist_seed_ids(cursor, playlist_name):
    """Load ids for one local playlist seed."""
    cursor.execute(
        "SELECT item_id FROM playlist WHERE playlist_name = %s", (playlist_name,)
    )
    return [row['item_id'] for row in cursor.fetchall()]


def _filter_seed_ids(cursor, options):
    """Load filter seed ids, using SQL pagination for search-only requests."""
    # _build_filter_query emits only allowlisted SQL identifiers and operators;
    # every request value remains in the psycopg2 parameter tuples below.
    where_clause, params = _build_filter_query(
        options['filters'], options['match_mode']
    )
    if not options['search_only']:
        query = f"SELECT item_id FROM score WHERE {where_clause}"  # nosec B608
        cursor.execute(query, tuple(params))
        return [row['item_id'] for row in cursor.fetchall()], None

    offset = (options['page'] - 1) * options['per_page']
    count_query = f"SELECT COUNT(*) AS total FROM score WHERE {where_clause}"  # nosec B608
    cursor.execute(count_query, tuple(params))
    count_row = cursor.fetchone()
    total = int(count_row['total'] if count_row else 0)
    page_query = (
        f"SELECT item_id FROM score WHERE {where_clause} "  # nosec B608
        "ORDER BY item_id LIMIT %s OFFSET %s"
    )
    cursor.execute(page_query, tuple(params + [options['per_page'], offset]))
    return [row['item_id'] for row in cursor.fetchall()], total


def _load_search_seed_ids(options):
    """Resolve playlist, filter, or explicit track seeds for one request."""
    connection = get_db()
    cursor = connection.cursor(cursor_factory=DictCursor)
    try:
        if options['playlist_name']:
            playlist_ids = _playlist_seed_ids(cursor, options['playlist_name'])
            if not playlist_ids:
                message = (
                    f"Playlist '{options['playlist_name']}' not found or is empty"
                )
                return [], None, (message, 404)
            return playlist_ids, None, None
        if options['filters']:
            playlist_ids, filter_total = _filter_seed_ids(cursor, options)
            if not playlist_ids and (filter_total is None or filter_total == 0):
                return [], filter_total, (
                    "No songs found matching the filters",
                    404,
                )
            return playlist_ids, filter_total, None
        return list(options['source_ids']), None, None
    finally:
        cursor.close()


def _search_only_result(playlist_ids, filter_total, options):
    """Build one paged Smart Search response while retaining database id order."""
    offset = (options['page'] - 1) * options['per_page']
    if filter_total is None:
        total = len(playlist_ids)
        page_ids = playlist_ids[offset:offset + options['per_page']]
    else:
        total = filter_total
        page_ids = playlist_ids
    metadata = get_score_data_lite_by_ids(page_ids) if page_ids else []
    metadata_by_id = {item['item_id']: item for item in metadata}
    ordered = []
    for item_id in page_ids:
        item = metadata_by_id.get(item_id)
        if item is not None:
            item['distance'] = 0.0
            ordered.append(item)
    return {
        "results": ordered,
        "total": total,
        "page": options['page'],
        "per_page": options['per_page'],
        "has_more": (offset + len(ordered)) < total,
        "playlist_song_count": total,
        "included_count": 0,
        "excluded_count": 0,
    }


def _extend_centroid_context(playlist_ids, options):
    """Prepare positive/excluded centroids and cached source vectors."""
    included_ids = options['included_ids']
    centroid_ids = list(set(list(playlist_ids) + list(included_ids)))
    combined_levels = {
        str(item_id): options['source_levels'].get(str(item_id), 0)
        for item_id in playlist_ids
    }
    combined_levels.update({
        str(item_id): options['included_levels'].get(str(item_id), 0)
        for item_id in included_ids
    })
    weights = _levels_to_weights(combined_levels, len(centroid_ids))
    upfront_ids = [str(item) for item in centroid_ids + options['excluded_ids']]
    vector_cache = get_vectors_by_ids(upfront_ids)
    positive_centroid = _compute_centroid_from_ids(
        centroid_ids, weights, vector_cache=vector_cache
    )
    if positive_centroid is None:
        return None
    excluded_centroid = None
    if options['excluded_ids']:
        excluded_centroid = _compute_centroid_from_ids(
            options['excluded_ids'], vector_cache=vector_cache
        )
    query_vector = positive_centroid
    if excluded_centroid is not None:
        query_vector = positive_centroid - (excluded_centroid * 0.5)
    return {
        'query_vector': query_vector,
        'excluded_centroid': excluded_centroid,
        'vector_cache': vector_cache,
    }


def _find_extend_candidates(query_vector, source_count, max_songs):
    """Fetch a bounded neighbor pool sized for downstream filtering attrition."""
    candidate_count = min(max(max_songs * 10, 500) + source_count // 5, 1500)
    results = find_nearest_neighbors_by_vector(
        query_vector, n=candidate_count, eliminate_duplicates=True
    )
    logger.info(
        f"Extend: requested {candidate_count} candidates, got {len(results)}, "
        f"source_count={source_count}"
    )
    return results


def _candidate_metadata_allowed(metadata, options):
    """Apply optional rating and year bounds to one candidate."""
    if options['min_rating'] is not None:
        rating = metadata.get('rating')
        if rating is None or rating < options['min_rating']:
            return False
    if options['year_min'] is None and options['year_max'] is None:
        return True
    year = metadata.get('year')
    if year is None or year <= 0:
        return False
    if options['year_min'] is not None and year < options['year_min']:
        return False
    return options['year_max'] is None or year <= options['year_max']


def _candidate_near_excluded(item_id, excluded_centroid, subtract_threshold):
    """Report whether one candidate falls within the excluded-centroid radius."""
    vector = get_vector_by_id(str(item_id))
    if vector is None:
        return False
    candidate = np.array(vector, dtype=float)
    if config.PATH_DISTANCE_METRIC != 'angular':
        distance = float(np.linalg.norm(excluded_centroid - candidate))
        return distance < subtract_threshold
    excluded_unit = excluded_centroid / (np.linalg.norm(excluded_centroid) or 1.0)
    candidate_unit = candidate / (np.linalg.norm(candidate) or 1.0)
    cosine = np.clip(np.dot(excluded_unit, candidate_unit), -1.0, 1.0)
    distance = float(np.arccos(cosine) / np.pi)
    return distance < subtract_threshold


def _enrich_candidate(candidate, metadata):
    """Attach database metadata without overwriting provider-populated labels."""
    candidate['album'] = metadata.get('album')
    candidate['album_artist'] = metadata.get('album_artist')
    candidate['year'] = metadata.get('year')
    if not candidate.get('title'):
        candidate['title'] = metadata.get('title')
    if not candidate.get('author'):
        candidate['author'] = metadata.get('author')


def _filter_extend_candidates(neighbor_results, playlist_ids, options, context):
    """Filter, enrich, and cap the candidate neighbor results."""
    seen_ids = set(playlist_ids) | set(options['included_ids']) | set(options['excluded_ids'])
    candidate_ids = [result['item_id'] for result in neighbor_results]
    metadata = get_score_data_by_ids(candidate_ids) if candidate_ids else []
    metadata_by_id = {item['item_id']: item for item in metadata}
    subtract_threshold = (
        config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
        if config.PATH_DISTANCE_METRIC == 'angular'
        else config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN
    )
    filtered = []
    for result in neighbor_results:
        item_id = result['item_id']
        item_metadata = metadata_by_id.get(item_id, {})
        if item_id in seen_ids or not _candidate_metadata_allowed(item_metadata, options):
            continue
        if context['excluded_centroid'] is not None and _candidate_near_excluded(
            item_id, context['excluded_centroid'], subtract_threshold
        ):
            continue
        if result.get('distance', 0) <= options['similarity_threshold']:
            _enrich_candidate(result, item_metadata)
            filtered.append(result)
        if len(filtered) >= options['max_songs']:
            break
    return filtered


def _normalized_source_vectors(playlist_ids, vector_cache):
    """Normalize source vectors for duplicate annotation."""
    normalized = {}
    for item_id in playlist_ids:
        vector = vector_cache.get(str(item_id))
        if vector is not None:
            source = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(source)
            normalized[item_id] = source / norm if norm > 0 else source
    return normalized


def _nearest_duplicate_source(candidate_vector, source_vectors):
    """Find the closest source vector and cosine distance for one candidate."""
    best_distance = float('inf')
    best_source_id = None
    for source_id, source_vector in source_vectors.items():
        cosine = np.clip(np.dot(source_vector, candidate_vector), -1.0, 1.0)
        distance = float(1.0 - cosine)
        if distance < best_distance:
            best_distance = distance
            best_source_id = source_id
    return best_source_id, best_distance


def _annotate_duplicate_candidates(results, source_tracks, source_vectors, threshold):
    """Annotate result tracks whose embeddings duplicate a source track."""
    if not source_vectors:
        return
    result_ids = [str(result['item_id']) for result in results]
    result_vectors = get_vectors_by_ids(result_ids) if result_ids else {}
    source_metadata = {item['item_id']: item for item in source_tracks}
    for result in results:
        vector = result_vectors.get(str(result['item_id']))
        if vector is None:
            continue
        candidate = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(candidate)
        if norm > 0:
            candidate = candidate / norm
        source_id, distance = _nearest_duplicate_source(candidate, source_vectors)
        if source_id is not None and distance < threshold:
            metadata = source_metadata.get(source_id, {})
            result['duplicate_of'] = {
                'item_id': source_id,
                'title': metadata.get('title'),
                'author': metadata.get('author'),
                'album': metadata.get('album'),
                'distance': round(distance, 4),
            }


def _extend_search_result(playlist_ids, options):
    """Run centroid neighbor search and build the extender response."""
    context = _extend_centroid_context(playlist_ids, options)
    if context is None:
        return None, (
            "Failed to compute playlist centroid - no valid embeddings found",
            500,
        )
    source_count = len(playlist_ids) + len(options['included_ids'])
    neighbors = _find_extend_candidates(
        context['query_vector'], source_count, options['max_songs']
    )
    results = _filter_extend_candidates(neighbors, playlist_ids, options, context)
    source_tracks = get_score_data_by_ids(playlist_ids) if playlist_ids else []
    if options['duplicate_threshold'] > 0:
        source_vectors = _normalized_source_vectors(
            playlist_ids, context['vector_cache']
        )
        _annotate_duplicate_candidates(
            results, source_tracks, source_vectors, options['duplicate_threshold']
        )
    return {
        "results": results,
        "playlist_song_count": len(playlist_ids),
        "included_count": len(options['included_ids']),
        "excluded_count": len(options['excluded_ids']),
        "source_tracks": source_tracks,
    }, None


@playlist_curator_bp.route('/api/curator/search', methods=['POST'])
def search_api():
    """Search the library or extend a weighted playlist seed."""
    payload = request.get_json() or {}
    options = _parse_search_options(payload)
    if not (
        options['playlist_name'] or options['filters'] or options['source_ids']
    ):
        return jsonify({"error": "Missing 'playlist_name', 'filters', or 'source_ids'"}), 400
    try:
        playlist_ids, filter_total, seed_error = _load_search_seed_ids(options)
        if seed_error:
            message, status = seed_error
            return jsonify({"error": message}), status
        if options['search_only']:
            return jsonify(_search_only_result(playlist_ids, filter_total, options))
        result, extend_error = _extend_search_result(playlist_ids, options)
        if extend_error:
            message, status = extend_error
            return jsonify({"error": message}), status
        return jsonify(result)
    except Exception:
        logger.exception("Playlist curator search failed")
        return jsonify({"error": INTERNAL_ERROR_MESSAGE}), 500

def _parse_save_playlist_request(payload):
    """Validate and normalize a create-new or replace-existing save request."""
    if not isinstance(payload, dict):
        return None, ("JSON body must be an object", 400)
    has_new_name = 'new_playlist_name' in payload
    has_replace_name = 'replace_playlist_name' in payload
    if has_new_name == has_replace_name:
        return None, ("Provide exactly one playlist save action", 400)

    name_key = 'replace_playlist_name' if has_replace_name else 'new_playlist_name'
    raw_name = payload.get(name_key)
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, ("Playlist name must be a non-empty string", 400)
    track_ids = payload.get('track_ids')
    if not isinstance(track_ids, list) or not track_ids:
        return None, ("Track IDs must be a non-empty list", 400)
    return {
        'action': 'replaced' if has_replace_name else 'created',
        'playlist_name': raw_name.strip(),
        'track_ids': list(dict.fromkeys(str(track_id) for track_id in track_ids)),
    }, None


def _replace_saved_playlist(playlist_name, track_ids, replace_playlist):
    """Replace an exact-name server playlist and validate the provider response."""
    existing_names = {
        str(playlist.get('Name') or playlist.get('name') or '').strip()
        for playlist in (_fetch_server_playlists() or [])
    }
    if playlist_name not in existing_names:
        return None, None, (f"Playlist '{playlist_name}' no longer exists", 404)
    try:
        replaced = replace_playlist(playlist_name, track_ids)
    except NotImplementedError:
        return None, None, (
            "Replacing playlists is not supported by this media server",
            501,
        )
    if not replaced:
        return None, None, ("Media server failed to replace playlist", 502)
    playlist_id = replaced.get('Id') or replaced.get('id')
    if not playlist_id:
        return None, None, ("Media server replacement returned no playlist ID", 502)
    message = f"Playlist '{playlist_name}' replaced with {len(track_ids)} songs!"
    return playlist_id, message, None


def _execute_playlist_save(save_request, create_playlist, replace_playlist):
    """Execute one normalized playlist save request."""
    playlist_name = save_request['playlist_name']
    track_ids = save_request['track_ids']
    if save_request['action'] == 'replaced':
        playlist_id, message, error = _replace_saved_playlist(
            playlist_name, track_ids, replace_playlist
        )
        return playlist_id, message, 200, error
    playlist_id = create_playlist(playlist_name, track_ids)
    message = f"Playlist '{playlist_name}' created with {len(track_ids)} songs!"
    return playlist_id, message, 201, None


@playlist_curator_bp.route('/api/curator/save_playlist', methods=['POST'])
def save_playlist_api():
    """Create a new curator playlist or replace the named server-playlist seed."""
    from tasks.ivf_manager import create_playlist_from_ids
    from tasks.mediaserver import create_or_replace_playlist

    save_request, validation_error = _parse_save_playlist_request(
        request.get_json(silent=True)
    )
    if validation_error:
        message, status = validation_error
        return jsonify({"error": message}), status
    try:
        playlist_id, message, status, save_error = _execute_playlist_save(
            save_request, create_playlist_from_ids, create_or_replace_playlist
        )
        if save_error:
            error_message, error_status = save_error
            return jsonify({"error": error_message}), error_status
        return jsonify({
            "action": save_request['action'],
            "message": message,
            "playlist_id": playlist_id,
            "playlist_name": save_request['playlist_name'],
            "total_songs": len(save_request['track_ids']),
        }), status
    except Exception:
        logger.exception("Save curator playlist failed")
        return jsonify({"error": INTERNAL_ERROR_MESSAGE}), 500

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
    except Exception:
        logger.exception("Failed to fetch server playlists")
        return jsonify({"error": INTERNAL_ERROR_MESSAGE}), 500


def _fetch_server_playlists():
    """Fetch playlists from the single configured media server."""
    try:
        if config.MEDIASERVER_TYPE == 'mpd':
            return []  # MPD intentionally unsupported
        from tasks.mediaserver import get_all_playlists
        return get_all_playlists()
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

    except Exception:
        logger.exception("Failed to fetch server playlist tracks")
        return jsonify({"error": INTERNAL_ERROR_MESSAGE}), 500


def _fetch_server_playlist_item_ids(playlist_id):
    """Fetch track item_ids from a playlist on the configured server.

    Returns list[str] on success, None on error.
    """
    try:
        from tasks.mediaserver import get_playlist_track_ids
        return get_playlist_track_ids(playlist_id)
    except Exception as e:
        logger.warning(f"Failed to fetch playlist tracks for {config.MEDIASERVER_TYPE}: {e}")
        return None


_ITEM_ID_RE = re.compile(r'[A-Za-z0-9_\-]{1,128}')


def _resolve_stream_target(item_id, media_server_type):
    """Resolve one media-server track into an authenticated upstream request."""
    if media_server_type == 'jellyfin':
        return (
            f"{config.JELLYFIN_URL.rstrip('/')}/Items/{item_id}/Download",
            {"X-Emby-Token": config.JELLYFIN_TOKEN},
            None,
        ), None
    if media_server_type == 'emby':
        return (
            f"{config.EMBY_URL.rstrip('/')}/Items/{item_id}/Download",
            {"X-Emby-Token": config.EMBY_TOKEN},
            None,
        ), None
    if media_server_type == 'plex':
        from tasks.mediaserver.plex import _resolve_part

        part_key, _ = _resolve_part(item_id)
        if not part_key:
            return None, ("Track stream not found", 404)
        return (
            f"{config.PLEX_URL.rstrip('/')}{part_key}",
            {"X-Plex-Token": config.PLEX_TOKEN},
            None,
        ), None
    if media_server_type == 'navidrome':
        from tasks.mediaserver.navidrome import get_navidrome_auth_params

        auth_params = get_navidrome_auth_params()
        if not auth_params:
            return None, ("Navidrome credentials not configured", 500)
        return (
            f"{config.NAVIDROME_URL.rstrip('/')}/rest/stream.view",
            {},
            {"id": item_id, **auth_params},
        ), None
    if media_server_type == 'lyrion':
        return (
            f"{config.LYRION_URL.rstrip('/')}/music/{item_id}/download",
            {},
            None,
        ), None
    if media_server_type == 'mpd':
        return None, ("MPD streaming is not supported by the playlist curator", 501)
    return None, ("Stream not supported for this media server type", 501)


def _fetch_stream_upstream(item_id, media_server_type, target):
    """Open the upstream media response and convert expected failures to API errors."""
    upstream_url, upstream_headers, params = target
    client_range = request.headers.get('Range')
    if client_range:
        upstream_headers['Range'] = client_range
    try:
        upstream = http_requests.get(
            upstream_url,
            params=params,
            headers=upstream_headers,
            stream=True,
            timeout=(10, 60),
            allow_redirects=True,
        )
    except http_requests.exceptions.RequestException as exc:
        logger.warning(
            f"Upstream connection failed for item_id={item_id} "
            f"backend={media_server_type} error_type={type(exc).__name__}"
        )
        return None, ("Upstream stream error", 502)
    if upstream.status_code < 400:
        return upstream, None
    logger.warning(
        f"Upstream stream request failed for item_id={item_id} "
        f"backend={media_server_type} status={upstream.status_code}"
    )
    upstream.close()
    return None, ("Upstream stream error", 502)


def _stream_response_headers(upstream):
    """Copy safe response headers from the media server."""
    passthrough = (
        'Content-Type', 'Content-Length', 'Content-Range',
        'Accept-Ranges', 'Last-Modified', 'ETag',
    )
    headers = {
        name: upstream.headers[name]
        for name in passthrough
        if upstream.headers.get(name) is not None
    }
    headers.setdefault('Content-Type', 'audio/mpeg')
    headers.setdefault('Accept-Ranges', 'bytes')
    return headers


def _build_stream_response(upstream):
    """Create the streaming Flask response and guarantee upstream closure."""
    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=64 * 1024):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    response = Response(
        stream_with_context(generate()),
        status=upstream.status_code,
        headers=_stream_response_headers(upstream),
    )
    response.call_on_close(upstream.close)
    return response


@playlist_curator_bp.route('/api/curator/stream/<path:item_id>', methods=['GET'])
def stream_track(item_id):
    """Proxy an audio stream without exposing media-server credentials."""
    try:
        if not _ITEM_ID_RE.fullmatch(item_id):
            return jsonify({"error": "Invalid item id"}), 400
        media_server_type = config.MEDIASERVER_TYPE
        target, target_error = _resolve_stream_target(item_id, media_server_type)
        if target_error:
            message, status = target_error
            return jsonify({"error": message}), status
        upstream, upstream_error = _fetch_stream_upstream(
            item_id, media_server_type, target
        )
        if upstream_error:
            message, status = upstream_error
            return jsonify({"error": message}), status
        return _build_stream_response(upstream)
    except Exception:
        logger.exception(
            f"Stream failed for item_id={item_id} backend={config.MEDIASERVER_TYPE}"
        )
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
