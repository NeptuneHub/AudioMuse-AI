# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Vibe Sketch manager: turn a hand-drawn curve into an ordered playlist.

A user sketches a freehand path across the 2D music map; this module resamples
that polyline into evenly spaced waypoints and snaps each waypoint to the
nearest song in the precomputed projection, so the playlist follows the drawn
arc through sonic space. Nearest-song lookup runs against an in-memory
scipy cKDTree built once over ``database.load_map_projection``, which keeps a
million-song library at O(log N) per waypoint instead of scanning the whole
catalogue.

Main Features:
* Lazy, cached cKDTree over the precomputed 2D projection.
* Arc-length polyline resampling into the requested number of waypoints.
* Greedy nearest-song assignment with deduplication, an optional variety knob
  that trades exactness for surprise, and an availability filter supplied by
  the caller so one mapping call scopes the whole run to a single server.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_PROJECTION_INDEX = 'main_map'
_MIN_LENGTH = 1
_MAX_LENGTH = 500
_MAX_POINTS = 500
_MIN_VARIETY = 0.0
_MAX_VARIETY = 1.0

_TREE_CACHE = {}
_TREE_LOCK = threading.Lock()


def _tree_entry():
    from database import load_map_projection

    id_map, proj = load_map_projection(_PROJECTION_INDEX)
    if not id_map or proj is None or len(id_map) == 0:
        return None
    key = (len(id_map), id(proj))
    with _TREE_LOCK:
        entry = _TREE_CACHE.get('entry')
        if entry is None or entry.get('key') != key:
            from scipy.spatial import cKDTree

            coords = np.ascontiguousarray(np.asarray(proj))
            if coords.ndim != 2 or coords.shape != (len(id_map), 2):
                return None
            entry = {
                'key': key,
                'id_map': [str(i) for i in id_map],
                'coords': coords,
                'tree': cKDTree(coords),
            }
            _TREE_CACHE['entry'] = entry
        return entry


def _resample_polyline(points, length):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2))
    if pts.shape[0] == 1:
        return np.repeat(pts, length, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total <= 0.0:
        return np.repeat(pts[:1], length, axis=0)
    ts = np.linspace(0.0, total, length)
    idx = np.clip(np.searchsorted(cum, ts, side='right') - 1, 0, pts.shape[0] - 2)
    seg_len = cum[idx + 1] - cum[idx]
    frac = np.divide(
        ts - cum[idx], seg_len, out=np.zeros_like(ts), where=seg_len > 0
    )
    return pts[idx] + frac[:, None] * (pts[idx + 1] - pts[idx])


def _candidate_rank_order(candidate_k, pool, rng):
    start = 0
    if pool > 1:
        start = int(rng.integers(0, min(pool, candidate_k)))
    return list(range(start, candidate_k)) + list(range(0, start))


def sketch_playlist(points, length, available=None, variety=0.0, seed=None):
    entry = _tree_entry()
    if entry is None:
        raise RuntimeError(
            "No music map projection is available yet; run an analysis first."
        )

    length = int(max(_MIN_LENGTH, min(_MAX_LENGTH, int(length))))
    variety = float(max(_MIN_VARIETY, min(_MAX_VARIETY, float(variety))))
    rng = np.random.default_rng(seed)

    id_map = entry['id_map']
    coords = entry['coords']
    tree = entry['tree']
    waypoints = _resample_polyline(points, length)

    candidate_k = min(
        len(id_map),
        max(4, int(round(8 + 20 * variety))),
    )
    if candidate_k < 1:
        candidate_k = 1

    dists, indices = tree.query(np.ascontiguousarray(waypoints), k=candidate_k)
    if length == 1:
        dists = dists.reshape(1, -1)
        indices = indices.reshape(1, -1)

    flat_ids = list(dict.fromkeys(str(id_map[int(i)]) for row in indices for i in row))
    mapping = available(flat_ids) if available is not None else {i: i for i in flat_ids}
    mapping = mapping or {}
    available_set = {str(c) for c in mapping}

    pool = 1
    if variety > 0.0:
        pool = max(2, int(round(variety * candidate_k)))

    used = set()
    results = []
    for wi in range(length):
        order = _candidate_rank_order(candidate_k, pool, rng)
        chosen = None
        for rank in order:
            kd_index = int(indices[wi][rank])
            cid = id_map[kd_index]
            if cid in used or cid not in available_set:
                continue
            chosen = (cid, kd_index, float(dists[wi][rank]))
            break
        if chosen is None:
            continue
        cid, kd_index, distance = chosen
        used.add(cid)
        results.append(
            {
                'item_id': cid,
                'distance': distance,
                'x': float(coords[kd_index][0]),
                'y': float(coords[kd_index][1]),
                'waypoint': [float(waypoints[wi][0]), float(waypoints[wi][1])],
            }
        )
    return {'results': results, 'mapping': mapping, 'sampled': length}
