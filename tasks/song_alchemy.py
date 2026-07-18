# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Blend and subtract musical anchors to generate an alchemy playlist.

Powers the song-alchemy feature (and the radios built on it): callers add and
subtract songs, artists, moods, playlists, or saved anchors and get back a
ranked set of tracks near the blended centroid.

Main Features:
* Gathers anchor points across item types (song/artist GMM/mood centroid/
  playlist/saved anchor), forms add and subtract centroids, and multi-queries
  the similarity index around them.
* Temperature controls exploration, with a zero-temperature single-song shortcut
  to plain nearest-neighbours; subtracted regions are filtered by distance and a
  2D projection of the centroid is returned for the UI.
* Governed by config: ALCHEMY_DEFAULT_N_RESULTS (100) capped by ALCHEMY_MAX_N_RESULTS
  (200), ALCHEMY_TEMPERATURE (1.0), and the metric-dependent subtract cutoffs
  ALCHEMY_SUBTRACT_DISTANCE_ANGULAR (0.2) / _EUCLIDEAN (5.0).
"""

import json
import logging
import math
import threading
from typing import List, Tuple
import numpy as np

from .ivf_manager import (
    multi_query_ids,
    find_nearest_neighbors_by_id,
    get_vector_by_id,
    _filter_by_distance,
)
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
    # Duplicate provider files in the playlist resolve to one canonical id; keeping
    # both would weight that song's IVF cell twice in the anchor.
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


def _anchor_anchor_points(item_id) -> List[dict]:
    from database import get_alchemy_anchor_by_id

    anchor = get_alchemy_anchor_by_id(item_id)
    if not (anchor and anchor.get('centroid') and isinstance(anchor.get('centroid'), list)):
        return []
    return [
        {
            'vector': np.array(anchor['centroid'], dtype=float),
            'weight': 1.0,
            'source_type': 'anchor',
            'source_id': item_id,
            'comp_idx': 0,
            'label': anchor.get('name', 'Anchor'),
        }
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


_ANCHOR_POINT_HANDLERS = {
    'song': _song_anchor_points,
    'artist': _artist_anchor_points,
    'anchor': _anchor_anchor_points,
    'mood': _mood_anchor_points,
    'playlist': _playlist_anchor_points,
}


def _gather_anchor_points(items: List[dict]) -> List[dict]:
    points = []
    for item in items or []:
        item_id = item.get('id')
        if not item_id:
            continue
        handler = _ANCHOR_POINT_HANDLERS.get(item.get('type', 'song').lower())
        if handler:
            points.extend(handler(item_id))
    return points


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


def _compute_centroid_from_items(items: List[dict]) -> np.ndarray:
    return _compute_centroid_from_points(_gather_anchor_points(items))


def _select_query_points(points: List[dict], max_points: int) -> List[dict]:
    if len(points) <= max_points:
        return points
    return sorted(points, key=lambda p: p['weight'], reverse=True)[:max_points]


def _multi_query_candidates(points: List[dict], n_results: int) -> List[str]:
    query_points = _select_query_points(points, config.ALCHEMY_MAX_ANCHOR_POINTS)
    p = len(query_points)
    if p == 0:
        return []
    target = n_results * 3
    if p == 1:
        per_point_n = target
    else:
        per_point_n = max(n_results // 4, (target + p - 1) // p)
    return multi_query_ids([pt['vector'] for pt in query_points], per_point_n)


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
    n_results = min(n_results, config.ALCHEMY_MAX_N_RESULTS)

    if add_items is None and add_ids is not None:
        add_items = [{'type': 'song', 'id': aid} for aid in add_ids]
    if subtract_items is None and subtract_ids is not None:
        subtract_items = [{'type': 'song', 'id': sid} for sid in subtract_ids]

    if not add_items or len(add_items) < 1:
        raise ValueError("At least one item must be in the ADD set")

    add_anchor_points = _gather_anchor_points(add_items)
    if not add_anchor_points:
        return {"results": [], "filtered_out": [], "centroid_2d": None}
    sub_anchor_points = _gather_anchor_points(subtract_items) if subtract_items else []

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

    if (
        temperature is not None
        and math.isclose(float(temperature), 0.0)
        and add_items
        and len(add_items) == 1
        and add_items[0].get('type') == 'song'
    ):
        try:
            neighbors = find_nearest_neighbors_by_id(add_items[0]['id'], n=n_results)
            candidate_ids = [n['item_id'] for n in neighbors]
        except Exception:
            candidate_ids = _multi_query_candidates(add_anchor_points, n_results)
    else:
        candidate_ids = _multi_query_candidates(add_anchor_points, n_results)
    if not candidate_ids:
        return {"results": [], "filtered_out": [], "centroid_2d": None}

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

    filtered_out = []
    filtered = candidate_ids
    if sub_anchor_points:
        filtered = []
        if subtract_distance is None:
            if config.PATH_DISTANCE_METRIC == 'angular':
                threshold = config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
            else:
                threshold = config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN
        else:
            threshold = subtract_distance

        sub_vecs = [p['vector'] for p in sub_anchor_points]
        for cid in candidate_ids:
            vec = _vec(cid)
            if vec is None:
                continue
            v_sub = np.array(vec, dtype=float)
            min_sub = min(_metric_distance(s, v_sub) for s in sub_vecs)
            if min_sub >= threshold:
                filtered.append(cid)
            else:
                filtered_out.append(cid)

    candidate_ids = filtered

    candidate_ids = candidate_ids[: max(n_results * 3, n_results)]

    from app_helper import get_db

    candidate_ids = [
        r['item_id']
        for r in _filter_by_distance([{'item_id': cid} for cid in candidate_ids], get_db())
    ]

    proj_vectors = []
    proj_ids = []
    playlist_vec_by_marker = {}
    add_meta = []
    if add_items:
        add_song_items = [item for item in add_items if item.get('type') == 'song']
        if add_song_items:
            add_song_ids = [item['id'] for item in add_song_items]
            add_details = get_score_data_by_ids(add_song_ids)
            add_map = {d['item_id']: d for d in add_details}
            for item in add_song_items:
                aid = item['id']
                vec = get_vector_by_id(aid)
                if vec is not None:
                    proj_vectors.append(np.array(vec, dtype=float))
                    proj_ids.append(f'__add_id__{aid}')
                    add_meta.append(
                        {
                            'item_id': aid,
                            'title': add_map.get(aid, {}).get('title'),
                            'author': add_map.get(aid, {}).get('author'),
                            'type': 'song',
                        }
                    )

        add_anchor_items = [item for item in add_items if item.get('type') == 'anchor']
        if add_anchor_items:
            from database import get_alchemy_anchor_by_id

            for item in add_anchor_items:
                anchor_id = item['id']
                anchor = get_alchemy_anchor_by_id(anchor_id)
                if anchor and anchor.get('centroid') and isinstance(anchor['centroid'], list):
                    vec = np.array(anchor['centroid'], dtype=float)
                    proj_vectors.append(vec)
                    proj_ids.append(f'__add_anchor__{anchor_id}')
                    add_meta.append(
                        {
                            'item_id': anchor_id,
                            'title': anchor.get('name', 'Anchor'),
                            'author': '',
                            'type': 'anchor',
                        }
                    )

        add_mood_items = [item for item in add_items if item.get('type') == 'mood']
        for item in add_mood_items:
            mood_id = item['id']
            vec = _get_mood_centroid_vector(mood_id)
            if vec is not None:
                proj_vectors.append(vec)
                proj_ids.append(f'__add_mood__{mood_id}')
                add_meta.append(
                    {
                        'item_id': mood_id,
                        'title': _get_mood_label(mood_id),
                        'author': '',
                        'type': 'mood',
                    }
                )

        add_artist_items = [item for item in add_items if item.get('type') == 'artist']
        for item in add_artist_items:
            artist_id = item['id']
            logger.info(f"Processing ADD artist: {artist_id}")
            gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(artist_id)
            logger.info(f"Retrieved {len(gmm_vecs)} GMM components for artist {artist_id}")
            for comp_idx, (vec, weight) in enumerate(zip(gmm_vecs, gmm_weights)):
                artist_name = artist_id
                resolved = registry.artist_names_for_ids(
                    [artist_id], ms_context.active_server_id()
                ).get(str(artist_id))
                if resolved:
                    artist_name = resolved
                logger.info(
                    f"Added ADD artist component {comp_idx}: {artist_name} (weight={weight:.2f})"
                )
                add_meta.append(
                    {
                        'item_id': f'{artist_id}_comp{comp_idx}',
                        'title': f'Component {comp_idx + 1} (w={weight:.2f})',
                        'author': artist_name,
                        'is_artist_component': True,
                        'weight': weight,
                    }
                )

        for p in add_anchor_points:
            if p['source_type'] != 'playlist':
                continue
            marker = f"__add_playlist__{p['source_id']}_c{p['comp_idx']}"
            vec = np.array(p['vector'], dtype=float)
            proj_vectors.append(vec)
            proj_ids.append(marker)
            playlist_vec_by_marker[marker] = vec
            add_meta.append(
                {
                    'item_id': f"{p['source_id']}_c{p['comp_idx']}",
                    'title': p['label'],
                    'author': 'Playlist',
                    'is_playlist_component': True,
                    'weight': p['weight'],
                }
            )

    sub_meta = []
    if subtract_items:
        subtract_song_items = [item for item in subtract_items if item.get('type') == 'song']
        if subtract_song_items:
            subtract_song_ids = [item['id'] for item in subtract_song_items]
            sub_details = get_score_data_by_ids(subtract_song_ids)
            sub_map = {d['item_id']: d for d in sub_details}
            for item in subtract_song_items:
                sid = item['id']
                vec = get_vector_by_id(sid)
                if vec is not None:
                    proj_vectors.append(np.array(vec, dtype=float))
                    proj_ids.append(f'__sub_id__{sid}')
                    sub_meta.append(
                        {
                            'item_id': sid,
                            'title': sub_map.get(sid, {}).get('title'),
                            'author': sub_map.get(sid, {}).get('author'),
                            'type': 'song',
                        }
                    )

        subtract_anchor_items = [item for item in subtract_items if item.get('type') == 'anchor']
        if subtract_anchor_items:
            from database import get_alchemy_anchor_by_id

            for item in subtract_anchor_items:
                anchor_id = item['id']
                anchor = get_alchemy_anchor_by_id(anchor_id)
                if anchor and anchor.get('centroid') and isinstance(anchor['centroid'], list):
                    vec = np.array(anchor['centroid'], dtype=float)
                    proj_vectors.append(vec)
                    proj_ids.append(f'__sub_anchor__{anchor_id}')
                    sub_meta.append(
                        {
                            'item_id': anchor_id,
                            'title': anchor.get('name', 'Anchor'),
                            'author': '',
                            'type': 'anchor',
                        }
                    )

        subtract_mood_items = [item for item in subtract_items if item.get('type') == 'mood']
        for item in subtract_mood_items:
            mood_id = item['id']
            vec = _get_mood_centroid_vector(mood_id)
            if vec is not None:
                proj_vectors.append(vec)
                proj_ids.append(f'__sub_mood__{mood_id}')
                sub_meta.append(
                    {
                        'item_id': mood_id,
                        'title': _get_mood_label(mood_id),
                        'author': '',
                        'type': 'mood',
                    }
                )

        subtract_artist_items = [item for item in subtract_items if item.get('type') == 'artist']
        for item in subtract_artist_items:
            artist_id = item['id']
            logger.info(f"Processing SUBTRACT artist: {artist_id}")
            gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(artist_id)
            logger.info(f"Retrieved {len(gmm_vecs)} GMM components for artist {artist_id}")
            for comp_idx, (vec, weight) in enumerate(zip(gmm_vecs, gmm_weights)):
                artist_name = artist_id
                resolved = registry.artist_names_for_ids(
                    [artist_id], ms_context.active_server_id()
                ).get(str(artist_id))
                if resolved:
                    artist_name = resolved
                logger.info(
                    f"Added SUBTRACT artist component {comp_idx}: {artist_name} (weight={weight:.2f})"
                )
                sub_meta.append(
                    {
                        'item_id': f'{artist_id}_comp{comp_idx}',
                        'title': f'Component {comp_idx + 1} (w={weight:.2f})',
                        'author': artist_name,
                        'is_artist_component': True,
                        'weight': weight,
                    }
                )

        for p in sub_anchor_points:
            if p['source_type'] != 'playlist':
                continue
            marker = f"__sub_playlist__{p['source_id']}_c{p['comp_idx']}"
            vec = np.array(p['vector'], dtype=float)
            proj_vectors.append(vec)
            proj_ids.append(marker)
            playlist_vec_by_marker[marker] = vec
            sub_meta.append(
                {
                    'item_id': f"{p['source_id']}_c{p['comp_idx']}",
                    'title': p['label'],
                    'author': 'Playlist',
                    'is_playlist_component': True,
                    'weight': p['weight'],
                }
            )

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

    id_to_coord = {}
    if id_map is not None and precomp_proj is not None:
        try:
            for iid, coord in zip(id_map, precomp_proj.tolist()):
                id_to_coord[str(iid)] = (float(coord[0]), float(coord[1]))
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
                        artist_id = comp_info['artist_id']
                        comp_idx = comp_info['component_idx']
                        key = f"{artist_id}_{comp_idx}"
                        artist_comp_to_coord[key] = (
                            float(projection[idx][0]),
                            float(projection[idx][1]),
                        )
                logger.info(
                    f"Loaded {len(artist_comp_to_coord)} precomputed artist component projections"
                )
    except Exception as e:
        logger.warning(f"Failed to load artist projection cache: {e}")

    missing_ids = []
    missing_vectors = []
    for pid in proj_ids:
        if isinstance(pid, str) and pid.startswith('__add_id__'):
            item_id = pid.replace('__add_id__', '')
            coord = id_to_coord.get(str(item_id))
            if coord is not None:
                proj_map[pid] = coord
        elif isinstance(pid, str) and pid.startswith('__sub_id__'):
            item_id = pid.replace('__sub_id__', '')
            coord = id_to_coord.get(str(item_id))
            if coord is not None:
                proj_map[pid] = coord
        elif pid in ('__add_centroid__', '__subtract_centroid__'):
            continue
        else:
            coord = id_to_coord.get(str(pid))
            if coord is not None:
                proj_map[pid] = coord

    for m in add_meta:
        if m.get('is_artist_component'):
            item_id_parts = m['item_id'].split('_comp')
            if len(item_id_parts) == 2:
                artist_id = item_id_parts[0]
                comp_idx = int(item_id_parts[1])
                key = f"{artist_id}_{comp_idx}"
                coord = artist_comp_to_coord.get(key)
                if coord is not None:
                    pid = f"__add_artist_comp__{artist_id}_{comp_idx}"
                    proj_map[pid] = coord
                    logger.debug(
                        f"Added ADD artist component to proj_map: key={key}, pid={pid}, coord={coord}"
                    )
                else:
                    logger.warning(
                        f"No precomputed projection for ADD artist component: key={key}, available keys={list(artist_comp_to_coord.keys())[:5]}"
                    )

    for m in sub_meta:
        if m.get('is_artist_component'):
            item_id_parts = m['item_id'].split('_comp')
            if len(item_id_parts) == 2:
                artist_id = item_id_parts[0]
                comp_idx = int(item_id_parts[1])
                key = f"{artist_id}_{comp_idx}"
                coord = artist_comp_to_coord.get(key)
                if coord is not None:
                    pid = f"__sub_artist_comp__{artist_id}_{comp_idx}"
                    proj_map[pid] = coord
                    logger.debug(
                        f"Added SUB artist component to proj_map: key={key}, pid={pid}, coord={coord}"
                    )
                else:
                    logger.warning(
                        f"No precomputed projection for SUB artist component: key={key}, available keys={list(artist_comp_to_coord.keys())[:5]}"
                    )

    def _centroid_from_member_coords(items, is_add=True):
        coords = []
        weights = []

        for item in items:
            if item.get('type') == 'song':
                mid = item['id']
                c = id_to_coord.get(str(mid))
                if c is not None:
                    coords.append(np.array(c, dtype=float))
                    weights.append(1.0)
            elif item.get('type') == 'anchor':
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
                gmm_vecs, gmm_weights = _get_artist_gmm_vectors_and_weights(artist_id)
                for comp_idx, weight in enumerate(gmm_weights):
                    key = f"{artist_id}_{comp_idx}"
                    c = artist_comp_to_coord.get(key)
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

        if isinstance(pid, str) and pid.startswith('__add_id__'):
            item_id = pid.replace('__add_id__', '')
            vec = get_vector_by_id(item_id)
        elif isinstance(pid, str) and pid.startswith('__sub_id__'):
            item_id = pid.replace('__sub_id__', '')
            vec = get_vector_by_id(item_id)
        elif isinstance(pid, str) and pid.startswith('__add_anchor__'):
            anchor_id = pid.replace('__add_anchor__', '')
            from database import get_alchemy_anchor_by_id

            anchor = get_alchemy_anchor_by_id(anchor_id)
            if anchor and anchor.get('centroid') and isinstance(anchor['centroid'], list):
                vec = np.array(anchor['centroid'], dtype=float)
            else:
                vec = None
        elif isinstance(pid, str) and pid.startswith('__sub_anchor__'):
            anchor_id = pid.replace('__sub_anchor__', '')
            from database import get_alchemy_anchor_by_id

            anchor = get_alchemy_anchor_by_id(anchor_id)
            if anchor and anchor.get('centroid') and isinstance(anchor['centroid'], list):
                vec = np.array(anchor['centroid'], dtype=float)
            else:
                vec = None
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

    distances = {}
    add_vecs = [p['vector'] for p in add_anchor_points]
    for cid in candidate_ids:
        vec = _vec(cid)
        if vec is None:
            continue
        v = np.array(vec, dtype=float)
        distances[cid] = min(_metric_distance(a, v) for a in add_vecs)

    details = get_score_data_by_ids(candidate_ids)
    details_map = {d['item_id']: d for d in details}

    for d in details_map.values():
        if 'album' not in d or not d['album']:
            d['album'] = 'Unknown'
        if 'album_artist' not in d or not d['album_artist']:
            d['album_artist'] = 'Unknown'

    seed_song_ids = [sid for sid in (add_song_ids + subtract_song_ids) if sid]
    seen_signatures = set()
    if seed_song_ids:
        for sd in get_score_data_by_ids(seed_song_ids):
            seen_signatures.add(
                ((sd.get('title') or '').strip().lower(), (sd.get('author') or '').strip().lower())
            )
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
    if ids:
        try:
            if temperature is not None and math.isclose(float(temperature), 0.0):
                ids_sorted = sorted(ids, key=lambda x: distances.get(x, float('inf')))
                for cid in ids_sorted[:n_results]:
                    item = details_map.get(cid, {})
                    item['distance'] = distances.get(cid)
                    item['embedding_2d'] = proj_map.get(cid)
                    if 'album' not in item or not item['album']:
                        item['album'] = 'Unknown'
                    if 'album_artist' not in item or not item['album_artist']:
                        item['album_artist'] = 'Unknown'
                    ordered.append(item)
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
                    item = details_map.get(cid, {})
                    item['distance'] = distances.get(cid)
                    item['embedding_2d'] = proj_map.get(cid)
                    if 'album' not in item or not item['album']:
                        item['album'] = 'Unknown'
                    if 'album_artist' not in item or not item['album_artist']:
                        item['album_artist'] = 'Unknown'
                    ordered.append(item)
        except Exception as e:
            logger.warning(f"Sampling failed, falling back to deterministic selection: {e}")
            ids_sorted = sorted(ids, key=lambda x: distances.get(x, float('inf')))
            for i in ids_sorted[:n_results]:
                item = details_map.get(i, {})
                item['distance'] = distances.get(i)
                item['embedding_2d'] = proj_map.get(i)
                if 'album' not in item or not item['album']:
                    item['album'] = 'Unknown'
                if 'album_artist' not in item or not item['album_artist']:
                    item['album_artist'] = 'Unknown'
                ordered.append(item)

    filtered_details = []
    if filtered_out:
        details_f = get_score_data_by_ids(filtered_out)
        details_f_map = {d['item_id']: d for d in details_f}
        for fid in filtered_out:
            if fid in details_f_map:
                fd = details_f_map[fid]
                fd['embedding_2d'] = proj_map.get(fid)
                if 'album' not in fd or not fd['album']:
                    fd['album'] = 'Unknown'
                if 'album_artist' not in fd or not fd['album_artist']:
                    fd['album_artist'] = 'Unknown'
                filtered_details.append(fd)

    centroid_2d = proj_map.get('__add_centroid__')
    subtract_centroid_2d = proj_map.get('__subtract_centroid__')

    add_points = []
    for m in add_meta:
        if m.get('is_artist_component'):
            pid = f"__add_artist_comp__{m['item_id'].rsplit('_comp', 1)[0]}_{m['item_id'].split('_comp')[1]}"
            logger.debug(
                f"Looking for ADD artist component: item_id={m['item_id']}, pid={pid}, found={pid in proj_map}"
            )
        elif m.get('is_playlist_component'):
            pid = f"__add_playlist__{m['item_id']}"
        elif m.get('type') == 'anchor':
            pid = f"__add_anchor__{m['item_id']}"
        elif m.get('type') == 'mood':
            pid = f"__add_mood__{m['item_id']}"
        else:
            pid = f"__add_id__{m['item_id']}"
        coord = proj_map.get(pid)
        add_points.append({**m, 'embedding_2d': coord})

    sub_points = []
    for m in sub_meta:
        if m.get('is_artist_component'):
            pid = f"__sub_artist_comp__{m['item_id'].rsplit('_comp', 1)[0]}_{m['item_id'].split('_comp')[1]}"
            logger.debug(
                f"Looking for SUB artist component: item_id={m['item_id']}, pid={pid}, found={pid in proj_map}"
            )
        elif m.get('is_playlist_component'):
            pid = f"__sub_playlist__{m['item_id']}"
        elif m.get('type') == 'anchor':
            pid = f"__sub_anchor__{m['item_id']}"
        elif m.get('type') == 'mood':
            pid = f"__sub_mood__{m['item_id']}"
        else:
            pid = f"__sub_id__{m['item_id']}"
        coord = proj_map.get(pid)
        sub_points.append({**m, 'embedding_2d': coord})

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
        'projection': projection_used,
    }
