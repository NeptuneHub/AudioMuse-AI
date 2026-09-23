# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Blend and subtract musical anchors to generate an alchemy playlist.

Powers the song-alchemy feature and the radios built on it: callers add and
subtract songs, artists, moods, playlists or saved anchors and get back tracks
ranked around the blended centroid.

Main Features:
* Gathers anchor points across types (song, artist GMM, mood, playlist, saved
  anchor), forms add and subtract centroids and multi-queries the index.
* Temperature controls exploration, with a zero-temperature single-song shortcut
  to plain nearest-neighbours; subtracted regions are filtered by distance and a
  2D projection of the centroid goes to the UI.
* Returns the subtract vectors with their exclusion radius as `exclusions` and
  every ADD point with its weight as `inclusions` so a saved anchor persists
  both; an anchor used as input contributes each stored include point (its
  centroid when it has none) on either side, and an ADD-ed anchor re-applies
  its stored exclusions, so anchor re-runs and radios equal the original run.
* Stored song seeds of an anchor are kept out of the results by a same-embedding
  test (cosine within twice the int8 rounding error of the index) and by their
  stored [title, artist] signature, as a live run drops its input songs.
* Every exported point carries the index of its run input; past
  ALCHEMY_MAX_ANCHOR_POINTS, inputs are ranked by total weight, then the stored
  groups inside an anchor, each taking its heaviest point before the rest fill
  by point weight, so a re-run of a saved anchor queries the same points.
* Every run input gets an equal share of the candidate pool; with the artist cap
  on, each query over-fetches 5x and the quotas are filled after the cap.
* Anchors are loaded once per run; one whose centroid size differs from the
  embedding dimension, or whose include points are stamped with another model
  file (SHA-256 prefix) or dimension, is ignored for the run with a warning;
  an unreadable model file yields no fingerprint and only the dimension counts.
* Governed by config: ALCHEMY_DEFAULT_N_RESULTS (50) when the caller names no
  count, ALCHEMY_TEMPERATURE (1.0), ALCHEMY_SUBTRACT_DISTANCE_ANGULAR (0.2) /
  _EUCLIDEAN (5.0), and MAX_SONGS_PER_ARTIST, applied when
  SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT is on. n_results has no upper bound
  here: ALCHEMY_MAX_N_RESULTS only caps the page's input box.
"""

import hashlib
import json
import logging
import math
import threading
from typing import List, Tuple
import numpy as np

from app_logging import sanitize_log_value

from .ivf_manager import (
    multi_query_ids,
    find_nearest_neighbors_by_id,
    get_vector_by_id,
    _filter_by_distance,
)
from .ivf_quant import I8_SCALE
from .search_shaping import apply_artist_cap
from .alchemy_projections import (
    _project_to_2d,
    _project_with_discriminant,
)
from database import get_score_data_by_ids, load_map_projection
import config

logger = logging.getLogger(__name__)


def _normalize_artist_name(s: str) -> str:
    return (
        s.lower()
        .replace(' ', '')
        .replace('-', '')
        .replace('\u2010', '')
        .replace('/', '')
        .replace("'", '')
    )


def _fuzzy_match_gmm(artist_name, artist_gmm_params, reverse_artist_map):
    query_norm = _normalize_artist_name(artist_name)
    for gmm_artist in reverse_artist_map:
        if _normalize_artist_name(gmm_artist) != query_norm:
            continue
        gmm = artist_gmm_params.get(gmm_artist)
        if gmm:
            logger.info(f"Fuzzy GMM match: '{artist_name}' -> '{gmm_artist}'")
            return gmm, gmm_artist
    return None, artist_name


def _get_artist_gmm_vectors_and_weights(
    artist_identifier: str,
) -> Tuple[List[np.ndarray], List[float]]:
    from tasks.artist_gmm_manager import (
        artist_gmm_params,
        load_artist_index_for_querying,
        reverse_artist_map,
    )
    from tasks.mediaserver import registry, context as ms_context

    if artist_gmm_params is None:
        load_artist_index_for_querying()

    if artist_gmm_params is None:
        logger.warning(f"Artist GMM index not available for {artist_identifier}")
        return [], []

    artist_name = artist_identifier
    resolved_name = registry.artist_names_for_ids(
        [artist_identifier], ms_context.active_server_id()
    ).get(str(artist_identifier))
    if resolved_name:
        artist_name = resolved_name

    gmm = artist_gmm_params.get(artist_name)

    if not gmm and reverse_artist_map:
        gmm, artist_name = _fuzzy_match_gmm(
            artist_name, artist_gmm_params, reverse_artist_map
        )

    if not gmm:
        logger.warning(f"No GMM found for artist '{artist_name}'")
        return [], []

    means = np.array(gmm['means'])
    weights = np.array(gmm['weights'])

    if gmm.get('is_single_track', False):
        logger.info(f"Loaded single-track artist '{artist_name}' with 1 component")

    return [means[i] for i in range(len(means))], weights.tolist()


_mood_centroids_cache = None
_mood_centroids_lock = threading.Lock()


def _load_mood_centroids_data():
    global _mood_centroids_cache
    if _mood_centroids_cache is None:
        with _mood_centroids_lock:
            if _mood_centroids_cache is None:
                with open(config.MOOD_CENTROIDS_FILE, encoding='utf-8') as _f:
                    _mood_centroids_cache = json.load(_f)
    return _mood_centroids_cache


def _get_mood_centroid_vector(item_id: str):
    parts = str(item_id).split(':', 1)
    if len(parts) != 2:
        return None
    mood_name, idx_str = parts[0].strip().lower(), parts[1].strip()
    try:
        cidx = int(idx_str)
        _mcdata = _load_mood_centroids_data()
        centroids_list = _mcdata.get(mood_name, {}).get('centroids', [])
        if 0 <= cidx < len(centroids_list):
            vec = centroids_list[cidx].get('centroid')
            if vec:
                return np.array(vec, dtype=float)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning(f"Failed to load mood centroid from '{item_id}': {exc}")
    return None


def _get_mood_label(item_id: str) -> str:
    parts = str(item_id).split(':', 1)
    if len(parts) != 2:
        return str(item_id)
    mood_name = parts[0].strip()
    return f"{mood_name.capitalize()} #{parts[1].strip()}"


def _get_playlist_components(playlist_id: str) -> Tuple[List[np.ndarray], List[float]]:
    import random
    from tasks.mediaserver import context as ms_context, get_playlist_track_ids
    from tasks.mediaserver.registry import canonical_input_ids
    from .ivf_manager import get_cell_groups_for_items

    track_ids = get_playlist_track_ids(playlist_id)
    if not track_ids:
        logger.warning(f"Playlist '{playlist_id}' returned no tracks")
        return [], []
    track_ids = list(
        dict.fromkeys(
            canonical_input_ids(track_ids, ms_context.active_server_id()).values()
        )
    )

    total = len(track_ids)
    if total > config.ALCHEMY_PLAYLIST_MAX_SONGS:
        track_ids = random.sample(track_ids, config.ALCHEMY_PLAYLIST_MAX_SONGS)

    groups = get_cell_groups_for_items(track_ids)
    if not groups:
        logger.warning(
            f"Playlist '{playlist_id}': none of {total} tracks are in the index; no anchor points"
        )
        return [], []

    if len(groups) > config.ALCHEMY_PLAYLIST_MAX_CENTROIDS:
        groups = _select_spread_centroids(groups, config.ALCHEMY_PLAYLIST_MAX_CENTROIDS)

    counts = np.array([count for _, count in groups], dtype=float)
    weights = (counts / counts.sum()).tolist()
    centroids = [np.array(vec, dtype=float) for vec, _ in groups]
    logger.info(f"Playlist '{playlist_id}': {total} tracks -> {len(centroids)} IVF-cell centroids")
    return centroids, weights


def _select_spread_centroids(groups, k):
    vecs = [np.array(vec, dtype=float) for vec, _ in groups]
    selected = [0]
    remaining = set(range(1, len(vecs)))
    while len(selected) < k and remaining:
        far_idx, far_dist = None, -1.0
        for i in remaining:
            nearest = min(_metric_distance(vecs[i], vecs[s]) for s in selected)
            if nearest > far_dist:
                far_dist, far_idx = nearest, i
        selected.append(far_idx)
        remaining.discard(far_idx)
    return [groups[i] for i in selected]


def _metric_distance(v_query: np.ndarray, v_cand: np.ndarray) -> float:
    a = np.asarray(v_query, dtype=float)
    b = np.asarray(v_cand, dtype=float)
    if config.PATH_DISTANCE_METRIC == 'angular':
        a = a / (np.linalg.norm(a) or 1.0)
        b = b / (np.linalg.norm(b) or 1.0)
        cosine = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(cosine) / np.pi)
    return float(np.linalg.norm(a - b))


_DISTANCE_BLOCK_ROWS = 1024


def _metric_distance_blocks(vectors, points):
    angular = config.PATH_DISTANCE_METRIC == 'angular'
    columns = np.asarray(points, dtype=float)
    if angular:
        columns = _unit_rows(columns)
    for start in range(0, len(vectors), _DISTANCE_BLOCK_ROWS):
        rows = np.asarray(vectors[start:start + _DISTANCE_BLOCK_ROWS], dtype=float)
        if angular:
            yield np.arccos(np.clip(_unit_rows(rows) @ columns.T, -1.0, 1.0)) / np.pi
        else:
            yield np.vstack([np.linalg.norm(row - columns, axis=1) for row in rows])


def _song_anchor_points(item_id) -> List[dict]:
    vec = get_vector_by_id(item_id)
    if vec is None:
        return []
    return [
        {
            'vector': np.array(vec, dtype=float),
            'weight': 1.0,
            'source_type': 'song',
            'source_id': item_id,
            'comp_idx': 0,
            'label': None,
            'seed': True,
        }
    ]


def _artist_anchor_points(item_id) -> List[dict]:
    gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(item_id)
    return [
        {
            'vector': np.array(vec, dtype=float),
            'weight': float(weight),
            'source_type': 'artist',
            'source_id': item_id,
            'comp_idx': idx,
            'label': None,
        }
        for idx, (vec, weight) in enumerate(zip(gmm_vecs, gmm_weights))
    ]


_embedding_fingerprints: dict = {}


def anchor_embedding_tag() -> dict:
    path = str(config.EMBEDDING_MODEL_PATH)
    fingerprint = _embedding_fingerprints.get(path)
    if fingerprint is None:
        try:
            digest = hashlib.sha256()
            with open(path, 'rb') as model_file:
                for chunk in iter(lambda: model_file.read(1 << 20), b''):
                    digest.update(chunk)
            fingerprint = digest.hexdigest()[:16]
            _embedding_fingerprints[path] = fingerprint
        except OSError as exc:
            logger.warning(
                "Cannot read the embedding model %s to fingerprint anchors (%s); anchors are "
                "matched on the embedding dimension only until it is readable.",
                path, exc,
            )
    return {'embedding_model_sha256': fingerprint, 'dimension': int(config.EMBEDDING_DIMENSION)}


def embedding_tags_match(saved, current) -> bool:
    if not isinstance(saved, dict) or saved.get('dimension') != current.get('dimension'):
        return False
    saved_model = saved.get('embedding_model_sha256')
    current_model = current.get('embedding_model_sha256')
    return saved_model is None or current_model is None or saved_model == current_model


def anchor_embedding_problem(anchor) -> str | None:
    dimension = int(config.EMBEDDING_DIMENSION)
    centroid = anchor.get('centroid')
    if not isinstance(centroid, list) or len(centroid) != dimension:
        size = len(centroid) if isinstance(centroid, list) else 0
        return f"its centroid has {size} values but the embedding has {dimension}"
    stored = anchor.get('inclusions')
    if isinstance(stored, dict):
        current = anchor_embedding_tag()
        saved = {key: stored.get(key) for key in current}
        if not embedding_tags_match(saved, current):
            return f"it was saved with embedding {saved} but the library now uses {current}"
    return None


_IGNORED_ANCHORS_KEY = '__ignored_anchors__'


def _load_usable_anchor(item_id, anchor_cache) -> dict | None:
    key = str(item_id)
    if key not in anchor_cache:
        from database import get_alchemy_anchor_by_id

        anchor = get_alchemy_anchor_by_id(item_id)
        problem = anchor_embedding_problem(anchor) if anchor else None
        if problem:
            logger.warning(
                "Ignoring anchor '%s' (id %s): %s. Run the alchemy again and re-save the anchor.",
                sanitize_log_value(str(anchor.get('name'))),
                sanitize_log_value(str(item_id)),
                sanitize_log_value(problem),
            )
            anchor_cache.setdefault(_IGNORED_ANCHORS_KEY, []).append(
                {'id': item_id, 'name': anchor.get('name'), 'problem': problem}
            )
            anchor = None
        anchor_cache[key] = anchor
    return anchor_cache[key]


def _stored_vector(entry) -> np.ndarray | None:
    if not isinstance(entry, dict) or not isinstance(entry.get('vector'), list):
        return None
    try:
        vector = np.array(entry['vector'], dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (int(config.EMBEDDING_DIMENSION),) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _stored_inclusion_points(stored) -> List[dict]:
    entries = stored.get('points') if isinstance(stored, dict) else None
    points = []
    for entry in entries if isinstance(entries, list) else []:
        vector = _stored_vector(entry)
        if vector is None:
            continue
        try:
            weight = float(entry.get('weight', 1.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(weight) or weight < 0:
            continue
        group = entry.get('group')
        signature = entry.get('signature')
        points.append(
            {
                'vector': vector,
                'weight': weight,
                'seed': entry.get('seed') is True,
                'group': group if isinstance(group, int) and not isinstance(group, bool) else None,
                'signature': tuple(signature)
                if isinstance(signature, list)
                and len(signature) == 2
                and all(isinstance(part, str) for part in signature)
                else None,
            }
        )
    return points


def _anchor_anchor_points(item_id, anchor_cache=None) -> List[dict]:
    anchor = _load_usable_anchor(item_id, {} if anchor_cache is None else anchor_cache)
    if anchor is None:
        return []
    points = _stored_inclusion_points(anchor.get('inclusions'))
    if not points:
        points = [
            {
                'vector': np.array(anchor['centroid'], dtype=float),
                'weight': 1.0,
                'seed': False,
                'group': None,
                'signature': None,
            }
        ]
    total = sum(p['weight'] for p in points)
    return [
        {
            'vector': p['vector'],
            'weight': p['weight'] / total if total > 0 else 1.0 / len(points),
            'source_type': 'anchor',
            'source_id': item_id,
            'comp_idx': idx,
            'label': anchor.get('name', 'Anchor'),
            'seed': p['seed'],
            'group': p['group'],
            'signature': p['signature'],
        }
        for idx, p in enumerate(points)
    ]


def _mood_anchor_points(item_id) -> List[dict]:
    vec = _get_mood_centroid_vector(item_id)
    if vec is None:
        return []
    return [
        {
            'vector': vec,
            'weight': 1.0,
            'source_type': 'mood',
            'source_id': item_id,
            'comp_idx': 0,
            'label': _get_mood_label(item_id),
        }
    ]


def _playlist_anchor_points(item_id) -> List[dict]:
    pl_vecs, pl_weights = _get_playlist_components(item_id)
    return [
        {
            'vector': np.array(vec, dtype=float),
            'weight': float(weight),
            'source_type': 'playlist',
            'source_id': item_id,
            'comp_idx': idx,
            'label': f'Cluster {idx + 1} (w={float(weight):.2f})',
        }
        for idx, (vec, weight) in enumerate(zip(pl_vecs, pl_weights))
    ]


def _anchor_exclusion_points(items: List[dict], anchor_cache=None) -> List[dict]:
    anchor_cache = {} if anchor_cache is None else anchor_cache
    points = []
    for item in items or []:
        if (item.get('type') or '').lower() != 'anchor' or not item.get('id'):
            continue
        anchor = _load_usable_anchor(item['id'], anchor_cache)
        for entry in (anchor or {}).get('exclusions') or []:
            vector = _stored_vector(entry)
            if vector is None:
                continue
            try:
                distance = entry.get('distance')
                if distance is not None:
                    distance = float(distance)
            except (TypeError, ValueError):
                continue
            points.append({'vector': vector, 'distance': distance})
    return points


_ANCHOR_POINT_HANDLERS = {
    'song': _song_anchor_points,
    'artist': _artist_anchor_points,
    'mood': _mood_anchor_points,
    'playlist': _playlist_anchor_points,
}


def _gather_anchor_points(items: List[dict], anchor_cache=None) -> List[dict]:
    anchor_cache = {} if anchor_cache is None else anchor_cache
    points = []
    for item in items or []:
        item_id = item.get('id')
        if not item_id:
            continue
        item_type = item.get('type', 'song').lower()
        if item_type == 'anchor':
            points.extend(_anchor_anchor_points(item_id, anchor_cache))
            continue
        handler = _ANCHOR_POINT_HANDLERS.get(item_type)
        if handler:
            points.extend(handler(item_id))
    return points


def _export_inclusions(points: List[dict], signature_by_id: dict) -> List[dict]:
    groups: dict = {}
    exported = []
    for point in points:
        group = groups.setdefault(_group_key(point), len(groups))
        if point.get('source_type') == 'song':
            signature = signature_by_id.get(point.get('source_id'))
        else:
            signature = point.get('signature')
        exported.append(
            {
                'vector': np.asarray(point['vector'], dtype=float).tolist(),
                'weight': float(point['weight']),
                'seed': bool(point.get('seed')),
                'group': group,
                'signature': list(signature) if signature else None,
            }
        )
    return exported


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _drop_stored_seeds(candidate_ids: List[str], seed_vectors: List[np.ndarray], vector_of) -> List[str]:
    positions, rows = [], []
    for position, cid in enumerate(candidate_ids):
        vector = vector_of(cid)
        if vector is not None:
            positions.append(position)
            rows.append(np.asarray(vector, dtype=np.float64))
    if not rows:
        return candidate_ids
    seeds = _unit_rows(np.vstack(seed_vectors).astype(np.float64))
    candidates = _unit_rows(np.vstack(rows))
    same_embedding_cosine = 1.0 - (0.5 * math.sqrt(seeds.shape[1]) / float(I8_SCALE)) ** 2
    is_seed = (candidates @ seeds.T).max(axis=1) >= same_embedding_cosine
    dropped = {positions[row] for row in np.flatnonzero(is_seed)}
    return [cid for position, cid in enumerate(candidate_ids) if position not in dropped]


def _compute_centroid_from_points(points: List[dict]) -> np.ndarray:
    if not points:
        return None
    vectors_array = np.array([p['vector'] for p in points])
    weights_array = np.array([p['weight'] for p in points], dtype=float)
    total = np.sum(weights_array)
    if total <= 0:
        weights_array = np.ones(len(weights_array)) / len(weights_array)
    else:
        weights_array = weights_array / total
    return np.sum(vectors_array * weights_array[:, np.newaxis], axis=0)


def _input_key(point) -> tuple:
    return (point.get('source_type'), str(point.get('source_id')))


def _group_key(point) -> tuple:
    return (point.get('source_type'), str(point.get('source_id')), point.get('group'))


def _select_query_points(points: List[dict], max_points: int) -> List[dict]:
    if len(points) <= max_points:
        return points
    chosen = set()
    for key_of in (_input_key, _group_key):
        totals, leaders = {}, {}
        for position, point in enumerate(points):
            key = key_of(point)
            totals[key] = totals.get(key, 0.0) + point['weight']
            if key not in leaders or point['weight'] > points[leaders[key]]['weight']:
                leaders[key] = position
        for key in sorted(totals, key=lambda k: totals[k], reverse=True):
            if len(chosen) >= max_points:
                break
            chosen.add(leaders[key])
    for position in sorted(range(len(points)), key=lambda i: points[i]['weight'], reverse=True):
        if len(chosen) >= max_points:
            break
        chosen.add(position)
    return sorted((points[i] for i in sorted(chosen)), key=lambda p: p['weight'], reverse=True)


def _shares(weights: dict) -> dict:
    total = sum(weights.values())
    if total > 0:
        return {key: weight / total for key, weight in weights.items()}
    return {key: 1.0 / len(weights) for key in weights}


def _query_quotas(points: List[dict], query_points: List[dict], target: int) -> List[int]:
    group_totals: dict = {}
    input_groups: dict = {}
    for point in points:
        key = _group_key(point)
        group_totals[key] = group_totals.get(key, 0.0) + point['weight']
        input_groups.setdefault(_input_key(point), {})[key] = None
    group_shares: dict = {}
    for groups in input_groups.values():
        group_shares.update(_shares({key: group_totals[key] for key in groups}))
    members: dict = {}
    for position, point in enumerate(query_points):
        members.setdefault(_group_key(point), {})[position] = point['weight']
    shares = [0.0] * len(query_points)
    for key, weights in members.items():
        for position, share in _shares(weights).items():
            shares[position] = group_shares[key] * share
    total = sum(shares)
    if total <= 0:
        shares, total = [1.0] * len(shares), float(len(shares))
    return [max(1, math.ceil(round(target * share / total, 6))) for share in shares]


_CAPPED_POOL_FETCH_FACTOR = 5


def _multi_query_candidates(points: List[dict], n_results: int, fetch_factor: int = 1):
    query_points = _select_query_points(points, config.ALCHEMY_MAX_ANCHOR_POINTS)
    if not query_points:
        return [], []
    quotas = _query_quotas(points, query_points, n_results * 3)
    ranked = [
        multi_query_ids([point['vector']], quota * fetch_factor)
        for point, quota in zip(query_points, quotas)
    ]
    return ranked, quotas


def _fill_quotas(ranked: List[List[str]], quotas: List[int], keep) -> List[str]:
    pool: dict = {}
    for ids, quota in zip(ranked, quotas):
        taken = 0
        for cid in ids:
            if taken >= quota:
                break
            if cid in keep and cid not in pool:
                pool[cid] = None
                taken += 1
    return list(pool)


def _fill_album_defaults(row):
    if 'album' not in row or not row['album']:
        row['album'] = 'Unknown'
    if 'album_artist' not in row or not row['album_artist']:
        row['album_artist'] = 'Unknown'
    return row


def song_alchemy(
    add_items=None,
    subtract_items=None,
    add_ids=None,
    subtract_ids=None,
    n_results: int | None = None,
    subtract_distance: float | None = None,
    temperature: float | None = None,
) -> dict:
    from tasks.mediaserver import registry, context as ms_context

    if n_results is None:
        n_results = config.ALCHEMY_DEFAULT_N_RESULTS

    if add_items is None and add_ids is not None:
        add_items = [{'type': 'song', 'id': aid} for aid in add_ids]
    if subtract_items is None and subtract_ids is not None:
        subtract_items = [{'type': 'song', 'id': sid} for sid in subtract_ids]

    if not add_items or len(add_items) < 1:
        raise ValueError("At least one item must be in the ADD set")

    anchor_cache: dict = {}

    def _empty_outcome():
        return {
            "results": [],
            "filtered_out": [],
            "centroid_2d": None,
            "ignored_anchors": anchor_cache.get(_IGNORED_ANCHORS_KEY, []),
        }

    add_anchor_points = _gather_anchor_points(add_items, anchor_cache)
    if not add_anchor_points:
        return _empty_outcome()
    sub_anchor_points = (
        _gather_anchor_points(subtract_items, anchor_cache) if subtract_items else []
    )

    add_centroid = _compute_centroid_from_points(add_anchor_points)
    subtract_centroid = (
        _compute_centroid_from_points(sub_anchor_points) if sub_anchor_points else None
    )

    try:
        if temperature is None:
            temperature = float(config.ALCHEMY_TEMPERATURE)
        else:
            temperature = float(temperature)
    except Exception:
        logger.warning(
            f"Invalid temperature value passed to song_alchemy: {temperature!r}; falling back to config default"
        )
        try:
            temperature = float(config.ALCHEMY_TEMPERATURE)
        except Exception:
            temperature = 1.0

    capping = bool(config.SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT) and (
        config.MAX_SONGS_PER_ARTIST or 0
    ) > 0
    fetch_factor = _CAPPED_POOL_FETCH_FACTOR if capping else 1
    if (
        temperature is not None
        and math.isclose(float(temperature), 0.0)
        and add_items
        and len(add_items) == 1
        and add_items[0].get('type') == 'song'
    ):
        try:
            neighbors = find_nearest_neighbors_by_id(add_items[0]['id'], n=n_results)
            ranked = [[n['item_id'] for n in neighbors]]
            quotas = [len(ranked[0])]
        except Exception:
            ranked, quotas = _multi_query_candidates(add_anchor_points, n_results, fetch_factor)
    else:
        ranked, quotas = _multi_query_candidates(add_anchor_points, n_results, fetch_factor)
    candidate_ids = list(dict.fromkeys(cid for ids in ranked for cid in ids))
    if not candidate_ids:
        return _empty_outcome()

    vec_cache: dict = {}

    def _vec(cid):
        if cid not in vec_cache:
            vec_cache[cid] = get_vector_by_id(cid)
        return vec_cache[cid]

    add_song_ids = [
        item['id'] for item in add_items if item.get('type') == 'song' and item.get('id')
    ]
    subtract_song_ids = [
        item['id']
        for item in (subtract_items or [])
        if item.get('type') == 'song' and item.get('id')
    ]

    if add_song_ids:
        add_set = set(add_song_ids)
        candidate_ids = [cid for cid in candidate_ids if cid not in add_set]
    if subtract_song_ids:
        sub_set = set(subtract_song_ids)
        candidate_ids = [cid for cid in candidate_ids if cid not in sub_set]

    anchor_seed_points = [
        p for p in add_anchor_points + sub_anchor_points
        if p['source_type'] == 'anchor' and p.get('seed')
    ]
    anchor_seed_vectors = [p['vector'] for p in anchor_seed_points]
    if anchor_seed_vectors:
        candidate_ids = _drop_stored_seeds(candidate_ids, anchor_seed_vectors, _vec)

    add_vecs = [p['vector'] for p in add_anchor_points]
    distances: dict = {}

    def _measure(ids):
        measured = [cid for cid in ids if cid not in distances and _vec(cid) is not None]
        nearest = [
            float(value)
            for block in _metric_distance_blocks([_vec(cid) for cid in measured], add_vecs)
            for value in block.min(axis=1)
        ]
        distances.update(zip(measured, nearest))

    detail_cache: dict = {}

    def _details(ids):
        missing = [cid for cid in ids if cid not in detail_cache]
        if missing:
            detail_cache.update(dict.fromkeys(missing))
            detail_cache.update((d['item_id'], d) for d in get_score_data_by_ids(missing))
        return {cid: detail_cache[cid] for cid in ids if detail_cache.get(cid) is not None}

    if capping:
        _measure(candidate_ids)
        pool_details = _details(candidate_ids)
        capped = apply_artist_cap(
            [{'item_id': cid} for cid in sorted(candidate_ids, key=lambda c: distances.get(c, math.inf))],
            lambda song: (pool_details.get(song['item_id']) or {}).get('author'),
        )
        candidate_ids = _fill_quotas(ranked, quotas, {song['item_id'] for song in capped})

    if subtract_distance is None:
        if config.PATH_DISTANCE_METRIC == 'angular':
            threshold = config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
        else:
            threshold = config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN
    else:
        threshold = subtract_distance

    anchor_exclusions = _anchor_exclusion_points(add_items, anchor_cache)
    exclusion_checks = [(p['vector'], threshold) for p in sub_anchor_points]
    exclusion_checks.extend(
        (p['vector'], threshold if p['distance'] is None else p['distance'])
        for p in anchor_exclusions
    )

    filtered_out = []
    filtered = candidate_ids
    if exclusion_checks:
        present = [cid for cid in candidate_ids if _vec(cid) is not None]
        limits = np.array([limit for _, limit in exclusion_checks], dtype=float)
        excluded = [
            flag
            for block in _metric_distance_blocks(
                [_vec(cid) for cid in present], [s for s, _ in exclusion_checks]
            )
            for flag in (block < limits).any(axis=1)
        ]
        filtered = [cid for cid, out in zip(present, excluded) if not out]
        filtered_out = [cid for cid, out in zip(present, excluded) if out]

    candidate_ids = filtered

    candidate_ids = candidate_ids[: max(n_results * 3, n_results)]

    from database import get_db

    candidate_ids = [
        r['item_id']
        for r in _filter_by_distance([{'item_id': cid} for cid in candidate_ids], get_db())
    ]

    proj_vectors = []
    proj_ids = []
    playlist_vec_by_marker = {}

    def _collect_playlist_components(anchor_points, marker_prefix, meta):
        for p in anchor_points:
            if p['source_type'] != 'playlist':
                continue
            marker = f"{marker_prefix}{p['source_id']}_c{p['comp_idx']}"
            vec = np.array(p['vector'], dtype=float)
            proj_vectors.append(vec)
            proj_ids.append(marker)
            playlist_vec_by_marker[marker] = vec
            meta.append(
                {
                    'item_id': f"{p['source_id']}_c{p['comp_idx']}",
                    'title': p['label'],
                    'author': 'Playlist',
                    'is_playlist_component': True,
                    'weight': p['weight'],
                }
            )

    def _collect_side(items, anchor_points, marker, label, meta):
        if not items:
            return
        song_items = [item for item in items if item.get('type') == 'song']
        if song_items:
            detail_map = {
                d['item_id']: d
                for d in get_score_data_by_ids([item['id'] for item in song_items])
            }
            for item in song_items:
                sid = item['id']
                vec = get_vector_by_id(sid)
                if vec is not None:
                    proj_vectors.append(np.array(vec, dtype=float))
                    proj_ids.append(f'__{marker}_id__{sid}')
                    meta.append(
                        {
                            'item_id': sid,
                            'title': detail_map.get(sid, {}).get('title'),
                            'author': detail_map.get(sid, {}).get('author'),
                            'type': 'song',
                        }
                    )

        anchor_items = [item for item in items if item.get('type') == 'anchor']
        if anchor_items:
            for item in anchor_items:
                anchor_id = item['id']
                anchor = _load_usable_anchor(anchor_id, anchor_cache)
                if anchor is not None:
                    proj_vectors.append(np.array(anchor['centroid'], dtype=float))
                    proj_ids.append(f'__{marker}_anchor__{anchor_id}')
                    meta.append(
                        {
                            'item_id': anchor_id,
                            'title': anchor.get('name', 'Anchor'),
                            'author': '',
                            'type': 'anchor',
                        }
                    )

        for item in [i for i in items if i.get('type') == 'mood']:
            mood_id = item['id']
            vec = _get_mood_centroid_vector(mood_id)
            if vec is not None:
                proj_vectors.append(vec)
                proj_ids.append(f'__{marker}_mood__{mood_id}')
                meta.append(
                    {
                        'item_id': mood_id,
                        'title': _get_mood_label(mood_id),
                        'author': '',
                        'type': 'mood',
                    }
                )

        for item in [i for i in items if i.get('type') == 'artist']:
            artist_id = item['id']
            safe_artist_id = sanitize_log_value(artist_id)
            logger.info("Processing %s artist: %s", label, safe_artist_id)
            gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(artist_id)
            logger.info(
                "Retrieved %d GMM components for artist %s", len(gmm_vecs), safe_artist_id
            )
            for comp_idx, (_vec, weight) in enumerate(zip(gmm_vecs, gmm_weights)):
                artist_name = artist_id
                resolved = registry.artist_names_for_ids(
                    [artist_id], ms_context.active_server_id()
                ).get(str(artist_id))
                if resolved:
                    artist_name = resolved
                logger.info(
                    f"Added {label} artist component {comp_idx}: {artist_name} (weight={weight:.2f})"
                )
                meta.append(
                    {
                        'item_id': f'{artist_id}_comp{comp_idx}',
                        'title': f'Component {comp_idx + 1} (w={weight:.2f})',
                        'author': artist_name,
                        'is_artist_component': True,
                        'weight': weight,
                    }
                )

        _collect_playlist_components(anchor_points, f'__{marker}_playlist__', meta)

    add_meta = []
    _collect_side(add_items, add_anchor_points, 'add', 'ADD', add_meta)

    sub_meta = []
    _collect_side(subtract_items, sub_anchor_points, 'sub', 'SUBTRACT', sub_meta)

    if add_centroid is not None:
        proj_vectors.append(add_centroid)
        proj_ids.append('__add_centroid__')
    if subtract_centroid is not None:
        proj_vectors.append(subtract_centroid)
        proj_ids.append('__subtract_centroid__')

    for cid in candidate_ids:
        vec = get_vector_by_id(cid)
        if vec is None:
            continue
        proj_vectors.append(np.array(vec, dtype=float))
        proj_ids.append(cid)
    for fid in filtered_out:
        vec = get_vector_by_id(fid)
        if vec is None:
            continue
        proj_vectors.append(np.array(vec, dtype=float))
        proj_ids.append(fid)

    projection_used = 'none'
    proj_map = {}

    try:
        id_map, precomp_proj = load_map_projection('main_map')
    except Exception:
        id_map, precomp_proj = None, None

    wanted_coords = {
        str(item.get('id'))
        for item in (add_items or []) + (subtract_items or [])
        if item.get('type') in ('song', 'anchor')
    }
    for pid in proj_ids:
        if isinstance(pid, str) and pid.startswith(('__add_id__', '__sub_id__')):
            wanted_coords.add(str(pid.replace('__add_id__', '').replace('__sub_id__', '')))
        elif pid not in ('__add_centroid__', '__subtract_centroid__'):
            wanted_coords.add(str(pid))

    id_to_coord = {}
    if id_map is not None and precomp_proj is not None:
        try:
            for position, iid in zip(range(len(precomp_proj)), id_map):
                key = str(iid)
                if key in wanted_coords:
                    coord = precomp_proj[position]
                    id_to_coord[key] = (float(coord[0]), float(coord[1]))
        except Exception:
            id_to_coord = {}

    artist_comp_to_coord = {}
    try:
        from database import ARTIST_PROJECTION_CACHE

        if ARTIST_PROJECTION_CACHE:
            component_map = ARTIST_PROJECTION_CACHE.get('component_map', [])
            projection = ARTIST_PROJECTION_CACHE.get('projection')
            if projection is not None and len(component_map) > 0:
                for idx, comp_info in enumerate(component_map):
                    if idx < len(projection):
                        comp_idx = comp_info['component_idx']
                        coord = (float(projection[idx][0]), float(projection[idx][1]))
                        artist_comp_to_coord[f"{comp_info['artist_id']}_{comp_idx}"] = coord
                        artist_name = comp_info.get('artist_name')
                        if artist_name:
                            artist_comp_to_coord.setdefault(f"{artist_name}_{comp_idx}", coord)
                logger.info(
                    f"Loaded {min(len(component_map), len(projection))} precomputed artist component projections"
                )
    except Exception as e:
        logger.warning(f"Failed to load artist projection cache: {e}")

    missing_ids = []
    missing_vectors = []
    for pid in proj_ids:
        if isinstance(pid, str) and pid.startswith(('__add_id__', '__sub_id__')):
            item_id = pid.replace('__add_id__', '').replace('__sub_id__', '')
            coord = id_to_coord.get(str(item_id))
            if coord is not None:
                proj_map[pid] = coord
        elif pid in ('__add_centroid__', '__subtract_centroid__'):
            continue
        else:
            coord = id_to_coord.get(str(pid))
            if coord is not None:
                proj_map[pid] = coord

    for meta, side in ((add_meta, 'add'), (sub_meta, 'sub')):
        for m in meta:
            if not m.get('is_artist_component'):
                continue
            item_id_parts = m['item_id'].split('_comp')
            if len(item_id_parts) != 2:
                continue
            artist_id = item_id_parts[0]
            comp_idx = int(item_id_parts[1])
            key = f"{artist_id}_{comp_idx}"
            coord = artist_comp_to_coord.get(key)
            if coord is None:
                coord = artist_comp_to_coord.get(f"{m.get('author')}_{comp_idx}")
            if coord is not None:
                pid = f"__{side}_artist_comp__{artist_id}_{comp_idx}"
                proj_map[pid] = coord
                logger.debug(
                    f"Added {side.upper()} artist component to proj_map: key={key}, pid={pid}, coord={coord}"
                )
            else:
                logger.warning(
                    f"No precomputed projection for {side.upper()} artist component: key={key}, available keys={list(artist_comp_to_coord.keys())[:5]}"
                )

    def _centroid_from_member_coords(items, is_add=True):
        coords = []
        weights = []

        for item in items:
            if item.get('type') in ('song', 'anchor'):
                mid = item['id']
                c = id_to_coord.get(str(mid))
                if c is not None:
                    coords.append(np.array(c, dtype=float))
                    weights.append(1.0)
            elif item.get('type') == 'mood':
                prefix = '__add_mood__' if is_add else '__sub_mood__'
                c = proj_map.get(f"{prefix}{item['id']}")
                if c is not None:
                    coords.append(np.array(c, dtype=float))
                    weights.append(1.0)

        for item in items:
            if item.get('type') == 'artist':
                artist_id = item['id']
                _gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(artist_id)
                artist_name = registry.artist_names_for_ids(
                    [artist_id], ms_context.active_server_id()
                ).get(str(artist_id)) if gmm_weights else None
                for comp_idx, weight in enumerate(gmm_weights):
                    key = f"{artist_id}_{comp_idx}"
                    c = artist_comp_to_coord.get(key)
                    if c is None and artist_name:
                        c = artist_comp_to_coord.get(f"{artist_name}_{comp_idx}")
                    if c is not None:
                        coords.append(np.array(c, dtype=float))
                        weights.append(weight)

        member_points = add_anchor_points if is_add else sub_anchor_points
        prefix = '__add_playlist__' if is_add else '__sub_playlist__'
        for item in items:
            if item.get('type') == 'playlist':
                for p in member_points:
                    if p['source_type'] != 'playlist' or p['source_id'] != item['id']:
                        continue
                    c = proj_map.get(f"{prefix}{p['source_id']}_c{p['comp_idx']}")
                    if c is not None:
                        coords.append(np.array(c, dtype=float))
                        weights.append(p['weight'])

        if not coords:
            return None

        coords_array = np.vstack(coords)
        weights_array = np.array(weights)
        weights_array = weights_array / np.sum(weights_array)

        weighted_mean = np.sum(coords_array * weights_array[:, np.newaxis], axis=0)
        return (float(weighted_mean[0]), float(weighted_mean[1]))

    for pid in proj_ids:
        if pid in proj_map:
            continue
        if pid in ('__add_centroid__', '__subtract_centroid__'):
            continue

        vec = None

        if isinstance(pid, str) and pid.startswith(('__add_id__', '__sub_id__')):
            item_id = pid.replace('__add_id__', '').replace('__sub_id__', '')
            vec = get_vector_by_id(item_id)
        elif isinstance(pid, str) and pid.startswith(('__add_anchor__', '__sub_anchor__')):
            anchor_id = pid.replace('__add_anchor__', '').replace('__sub_anchor__', '')
            anchor = _load_usable_anchor(anchor_id, anchor_cache)
            vec = np.array(anchor['centroid'], dtype=float) if anchor is not None else None
        elif isinstance(pid, str) and pid.startswith(('__add_mood__', '__sub_mood__')):
            mood_id = pid.split('__', 3)[-1]
            vec = _get_mood_centroid_vector(mood_id)
        elif isinstance(pid, str) and pid.startswith(('__add_playlist__', '__sub_playlist__')):
            vec = playlist_vec_by_marker.get(pid)
        else:
            vec = get_vector_by_id(pid)

        if vec is None:
            continue
        missing_ids.append(pid)
        missing_vectors.append(np.array(vec, dtype=float))

    if missing_vectors:
        try:
            local_projections = None

            if len(missing_vectors) >= 4:
                try:
                    local_add_vecs = []
                    local_sub_vecs = []

                    for pid in missing_ids:
                        idx = missing_ids.index(pid)
                        vec = missing_vectors[idx]
                        if pid.startswith(
                            ('__add_id__', '__add_artist_comp__', '__add_playlist__')
                        ):
                            local_add_vecs.append(vec)
                        elif pid.startswith(
                            ('__sub_id__', '__sub_artist_comp__', '__sub_playlist__')
                        ):
                            local_sub_vecs.append(vec)

                    if local_add_vecs and local_sub_vecs and _project_with_discriminant is not None:
                        local_projections = _project_with_discriminant(
                            local_add_vecs, local_sub_vecs, missing_vectors
                        )
                        projection_used = 'discriminant'
                except Exception:
                    local_projections = None

            if local_projections is None:
                try:
                    local_projections = _project_to_2d(missing_vectors)
                    projection_used = 'pca'
                except Exception:
                    local_projections = [(0.0, 0.0) for _ in missing_vectors]

            for pid, coord in zip(missing_ids, local_projections):
                proj_map[pid] = (float(coord[0]), float(coord[1]))
        except Exception as e:
            logger.warning(f"Failed to compute local projections for missing ids: {e}")

    for pid in proj_ids:
        if pid not in proj_map:
            proj_map[pid] = (0.0, 0.0)

    add_centroid_2d_db = None
    subtract_centroid_2d_db = None
    try:
        if add_items:
            add_centroid_2d_db = _centroid_from_member_coords(add_items, is_add=True)
        if subtract_items:
            subtract_centroid_2d_db = _centroid_from_member_coords(subtract_items, is_add=False)
        if add_centroid_2d_db is not None:
            proj_map['__add_centroid__'] = add_centroid_2d_db
            logger.info(f"ADD centroid 2D computed from members: {add_centroid_2d_db}")
        if subtract_centroid_2d_db is not None:
            proj_map['__subtract_centroid__'] = subtract_centroid_2d_db
            logger.info(f"SUBTRACT centroid 2D computed from members: {subtract_centroid_2d_db}")
    except Exception as e:
        logger.warning(f"Failed to compute centroid from member coords: {e}")

    _measure(candidate_ids)
    details_map = _details(candidate_ids)

    for d in details_map.values():
        _fill_album_defaults(d)

    seed_song_ids = [sid for sid in (add_song_ids + subtract_song_ids) if sid]
    seen_signatures = set()
    signature_by_id = {}
    if seed_song_ids:
        for sd in get_score_data_by_ids(seed_song_ids):
            signature = (
                (sd.get('title') or '').strip().lower(),
                (sd.get('author') or '').strip().lower(),
            )
            seen_signatures.add(signature)
            signature_by_id[sd.get('item_id')] = signature
    seen_signatures.update(p['signature'] for p in anchor_seed_points if p.get('signature'))
    deduped_ids = []
    for cid in candidate_ids:
        d = details_map.get(cid)
        if not d:
            continue
        signature = (
            (d.get('title') or '').strip().lower(),
            (d.get('author') or '').strip().lower(),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_ids.append(cid)
    candidate_ids = deduped_ids

    scored_candidates = []
    for cid in candidate_ids:
        if cid in details_map and cid in distances:
            scored_candidates.append((cid, distances[cid]))

    if temperature is None:
        try:
            from config import ALCHEMY_TEMPERATURE as _cfg_temp

            temperature = float(_cfg_temp)
        except Exception:
            temperature = 1.0

    logger.info(
        f"Song Alchemy: Using temperature={temperature} for probabilistic sampling of {len(scored_candidates)} candidates"
    )

    import random

    ids = [c[0] for c in scored_candidates]
    raw_scores = [-float(c[1]) for c in scored_candidates]

    ordered = []

    def _append_ordered(cid):
        item = _fill_album_defaults(details_map.get(cid, {}))
        item['distance'] = distances.get(cid)
        item['embedding_2d'] = proj_map.get(cid)
        ordered.append(item)

    if ids:
        try:
            if temperature is not None and math.isclose(float(temperature), 0.0):
                ids_sorted = sorted(ids, key=lambda x: distances.get(x, float('inf')))
                for cid in ids_sorted[:n_results]:
                    _append_ordered(cid)
            else:
                temps = [s / temperature for s in raw_scores]
                max_t = max(temps) if temps else 0.0
                exps = [math.exp(t - max_t) for t in temps]
                total = sum(exps)
                if total <= 0:
                    probs = [1.0 / len(exps)] * len(exps)
                else:
                    probs = [e / total for e in exps]

                if probs:
                    max_prob = max(probs)
                    min_prob = min(probs)
                    mean_prob = sum(probs) / len(probs)
                    logger.info(
                        f"Temperature={temperature}: Probability distribution - max={max_prob:.4f}, min={min_prob:.6f}, mean={mean_prob:.4f}, entropy={(-sum(p * math.log(p) if p > 0 else 0 for p in probs)):.3f}"
                    )

                chosen = []
                avail_ids = ids.copy()
                avail_probs = probs.copy()
                k = min(n_results, len(avail_ids))
                for _ in range(k):
                    s = sum(avail_probs)
                    if s <= 0:
                        idx = random.randrange(len(avail_ids))
                    else:
                        r = random.random() * s
                        acc = 0.0
                        idx = 0
                        for j, p in enumerate(avail_probs):
                            acc += p
                            if r <= acc:
                                idx = j
                                break
                    chosen_id = avail_ids.pop(idx)
                    avail_probs.pop(idx)
                    chosen.append(chosen_id)

                for cid in chosen:
                    _append_ordered(cid)
        except Exception as e:
            logger.warning(f"Sampling failed, falling back to deterministic selection: {e}")
            ids_sorted = sorted(ids, key=lambda x: distances.get(x, float('inf')))
            for i in ids_sorted[:n_results]:
                _append_ordered(i)

    filtered_details = []
    if filtered_out:
        details_f = get_score_data_by_ids(filtered_out)
        details_f_map = {d['item_id']: d for d in details_f}
        for fid in filtered_out:
            if fid in details_f_map:
                fd = details_f_map[fid]
                fd['embedding_2d'] = proj_map.get(fid)
                _fill_album_defaults(fd)
                filtered_details.append(fd)

    centroid_2d = proj_map.get('__add_centroid__')
    subtract_centroid_2d = proj_map.get('__subtract_centroid__')

    def _projection_points(meta, side):
        points = []
        for m in meta:
            if m.get('is_artist_component'):
                pid = f"__{side}_artist_comp__{m['item_id'].rsplit('_comp', 1)[0]}_{m['item_id'].split('_comp')[1]}"
                logger.debug(
                    f"Looking for {side.upper()} artist component: item_id={m['item_id']}, pid={pid}, found={pid in proj_map}"
                )
            elif m.get('is_playlist_component'):
                pid = f"__{side}_playlist__{m['item_id']}"
            elif m.get('type') == 'anchor':
                pid = f"__{side}_anchor__{m['item_id']}"
            elif m.get('type') == 'mood':
                pid = f"__{side}_mood__{m['item_id']}"
            else:
                pid = f"__{side}_id__{m['item_id']}"
            points.append({**m, 'embedding_2d': proj_map.get(pid)})
        return points

    add_points = _projection_points(add_meta, 'add')
    sub_points = _projection_points(sub_meta, 'sub')

    logger.info(f"Returning {len(add_points)} add_points and {len(sub_points)} sub_points")
    logger.info(
        f"add_points artist components: {sum(1 for p in add_points if p.get('is_artist_component'))}"
    )
    logger.info(
        f"sub_points artist components: {sum(1 for p in sub_points if p.get('is_artist_component'))}"
    )

    return {
        'results': ordered,
        'filtered_out': filtered_details,
        'centroid_2d': centroid_2d,
        'add_centroid_2d': centroid_2d,
        'subtract_centroid_2d': subtract_centroid_2d,
        'add_centroid_vector': add_centroid.tolist() if add_centroid is not None else None,
        'subtract_centroid_vector': subtract_centroid.tolist()
        if subtract_centroid is not None
        else None,
        'add_points': add_points,
        'sub_points': sub_points,
        'exclusions': [
            {'vector': np.asarray(vec, dtype=float).tolist(), 'distance': float(limit)}
            for vec, limit in exclusion_checks
        ],
        'inclusions': _export_inclusions(add_anchor_points, signature_by_id),
        'inclusions_embedding': anchor_embedding_tag(),
        'ignored_anchors': anchor_cache.get(_IGNORED_ANCHORS_KEY, []),
        'projection': projection_used,
    }
