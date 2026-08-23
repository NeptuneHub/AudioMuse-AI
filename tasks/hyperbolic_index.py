# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Disk-paged Poincare IVF index for the Hyperbolic Explorer.

A nearest-neighbour index over the Poincare-projected embeddings, built per
music server and persisted as segmented blobs in ivf_dir (the same storage
pattern the tree cache and the other indexes use). Only the cell directory and
the coarse centroids are held in memory; the projected vectors live on disk and
are decoded cell by cell on demand under a process-wide byte cap, so the full
projected catalogue never stays resident.

Main Features:
* The catalogue is partitioned by HYPERBOLIC k-means (tasks.hyperbolic_geometry
  .poincare_kmeans): k-means++ seeding, assignment and the Frechet-mean centroid
  update all run in the exact Poincare metric, so no Euclidean or cosine step
  enters the partition. Cell count follows the same rule as the other IVF
  indexes, 8*sqrt(n) capped at IVF_NLIST_MAX, and training runs on an
  IVF_TRAIN_POINTS_PER_CELL sample before every track is assigned to its
  nearest centroid.
* This REPLACED a radial-band split, which measured out as useless: radius
  alone cannot separate points in 200 dimensions, because almost all of the
  distance between two tracks comes from direction. On a 200k catalogue the
  band layout still scanned 63-95% of the library per query and adding bands
  barely moved it (8 bands 97.2%, 384 bands 93.8%). A centroid encodes
  direction as well as radius, which is what makes IVF_NPROBE cells out of
  8*sqrt(n) an actual reduction rather than a full scan with extra steps.
* Querying takes the exact Poincare distance to every coarse centroid, probes
  the nearest IVF_NPROBE cells, and heaps the top-k over their members. Like
  every IVF this is approximate: a neighbour parked in an unprobed cell is
  missed, and IVF_NPROBE is the recall/latency knob. Probing N cells would be N
  Postgres round trips, so _prefetch_cells pulls every uncached probed cell in
  ONE ANY() read and hands the arrays straight to the scan. It returns them
  rather than relying on the cell cache to still hold them, because a probe set
  larger than HYPERBOLIC_INDEX_CACHE_MB would otherwise evict its own earlier
  cells during the prefetch and the scan would re-read them one at a time,
  which is the exact round-trip storm the bulk read exists to prevent.
* IVF_NPROBE and _RERANK_OVERFETCH are NOT redundant: IVF_NPROBE decides which
  cells are looked at at all, while the overfetch decides how many of the
  scanned candidates survive i8 ranking error into the exact re-rank. Neither
  covers the other's failure mode.
* Cell vectors are quantized through tasks.ivf_quant on config.IVF_STORAGE_DTYPE
  (default i8), the same knob and the same codec every other index uses, so the
  projected catalogue on disk costs 1 byte per dimension instead of 4. This
  path deliberately does NOT go through ivf_quant.effective_code: that helper
  downgrades i8 to f16 for any non-angular metric, and the Poincare metric is
  non-angular, but the storage dtype here is taken literally so i8 means i8.
  The coarse centroids stay float32, exactly as they do for the paged IVF.
* i8 makes the cell scan coarser here than it would for an angular index,
  because the Poincare metric divides by (1 - ||u||^2), which for a track near
  the ball boundary is ~1e-6 while the i8 grid of 1/127 moves a radius by
  ~1e-2. Two things absorb that. Decoded cells are pushed back inside the ball
  with clip_into_ball, so a quantized point can never land on or past the
  boundary and blow the denominator up. And the scan overfetches
  _RERANK_OVERFETCH-fold (capped at _RERANK_SCAN_CAP) before hyperbolic_nearest
  re-ranks those candidates against the exact float32 poincare_embedding rows,
  so the distances and ordering it returns are exact for everything the probed
  cells reached. hyperbolic_nearest_multi skips the re-rank on purpose: it is a
  candidate generator whose caller (the geodesic journey) already re-ranks on
  the exact vectors itself.
* hyperbolic_nearest / hyperbolic_nearest_multi return item ids ranked by exact
  Poincare distance, or None when no index is built yet; callers surface that
  as a "run analysis to build it" error instead of scanning the catalogue.
* Build targets mirror the Hyperbolic Explorer tree: one index per configured
  server plus the default server, keyed the same way so a request scoped to a
  server reads only that server's index. A rebuild also sweeps the blobs of the
  retired radial-band layout, which nothing names any more.
"""

import gzip
import heapq
import json
import logging
import re
import threading
from collections import OrderedDict

import numpy as np

import config
from tasks import ivf_quant as quant

logger = logging.getLogger(__name__)

_TABLE = "ivf_dir"
_DIR_PREFIX = "hyperbolic_index_dir"
_CELL_PREFIX = "hyperbolic_index_cell"
_CENTROID_PREFIX = "hyperbolic_index_centroids"
_LEGACY_BAND_PREFIX = "hyperbolic_index_band"
_VERSION = 3
_DEFAULT_SERVER_KEY = "default"
_METRIC = "poincare"
_RERANK_OVERFETCH = 32
_RERANK_SCAN_CAP = 4096
_SCAN_CHUNK = 4096
_TRAIN_ITERATIONS = 10

_INDEX_CACHE = {"loaded": False, "servers": {}}
_INDEX_CACHE_LOCK = threading.RLock()

_CELL_CACHE = OrderedDict()
_CELL_CACHE_BYTES = 0
_CELL_CACHE_LOCK = threading.Lock()


def _storage_code():
    return quant.dtype_code(config.IVF_STORAGE_DTYPE)


def _scoped_name(prefix, server_key):
    key = server_key or _DEFAULT_SERVER_KEY
    return f"{prefix}__{key}"


def _dir_name(server_key):
    return _scoped_name(_DIR_PREFIX, server_key)


def _cell_name(server_key, cell):
    return _scoped_name(_CELL_PREFIX, server_key) + f"__{cell}"


def _centroids_name(server_key):
    return _scoped_name(_CENTROID_PREFIX, server_key)


def _cell_count(n_items):
    base = int(round(8.0 * np.sqrt(max(1, int(n_items)))))
    return max(1, min(int(config.IVF_NLIST_MAX), base, int(n_items)))


def _partition_into_cells(vectors):
    from tasks.hyperbolic_geometry import nearest_centroid, poincare_kmeans

    n = vectors.shape[0]
    n_cells = _cell_count(n)
    sample_n = min(n, int(config.IVF_TRAIN_POINTS_PER_CELL) * n_cells)
    if sample_n >= n:
        centroids, labels = poincare_kmeans(vectors, n_cells, iterations=_TRAIN_ITERATIONS)
        return centroids, nearest_centroid(vectors, centroids)
    picks = np.random.RandomState(0).choice(n, sample_n, replace=False)
    centroids, _labels = poincare_kmeans(
        vectors[picks], n_cells, iterations=_TRAIN_ITERATIONS
    )
    return centroids, nearest_centroid(vectors, centroids)


def _resolve_default_server_id():
    from tasks.mediaserver import registry

    try:
        return registry.get_default_server_id()
    except Exception:
        return None


def _build_targets():
    from tasks.mediaserver import registry

    try:
        servers = registry.list_servers()
    except Exception:
        servers = []
    if not servers:
        return [(_DEFAULT_SERVER_KEY, None, True)]
    default_id = _resolve_default_server_id()
    targets = [(_DEFAULT_SERVER_KEY, default_id, True)]
    for server in servers:
        if server["server_id"] != default_id:
            targets.append((server["server_id"], server["server_id"], False))
    return targets


def _delete_index(db_conn, server_key):
    key = server_key or _DEFAULT_SERVER_KEY
    dir_name = _scoped_name(_DIR_PREFIX, key)
    exact = [dir_name, _centroids_name(key)]
    patterns = [dir_name.replace("_", r"\_") + r"\_%\_%"]
    for prefix in (_CELL_PREFIX, _CENTROID_PREFIX, _LEGACY_BAND_PREFIX):
        patterns.append(_scoped_name(prefix, key).replace("_", r"\_") + r"%")
    clause = " OR ".join(["name = %s"] * len(exact) + ["name LIKE %s ESCAPE '\\'"] * len(patterns))
    with db_conn.cursor() as cur:
        cur.execute(f"DELETE FROM ivf_dir WHERE {clause}", tuple(exact + patterns))


def build_and_store_hyperbolic_index(db_conn=None):
    from database import get_db
    from tasks.hyperbolic_manager import fetch_all_poincare_rows
    from tasks.index_build_helpers import store_segmented_blob

    if db_conn is None:
        db_conn = get_db()
    code = _storage_code()
    try:
        for server_key, server_id, is_default in _build_targets():
            _delete_index(db_conn, server_key)
            rows = fetch_all_poincare_rows(
                server_id=server_id, include_legacy_default=is_default
            )
            if not rows:
                continue
            item_ids = list(rows.keys())
            vectors = np.stack([rows[i][0] for i in item_ids]).astype(np.float64)
            centroids, assigned = _partition_into_cells(vectors)
            n_cells = centroids.shape[0]

            cell_item_ids = [[] for _ in range(n_cells)]
            cell_rows = [[] for _ in range(n_cells)]
            for position, (item_id, cell) in enumerate(zip(item_ids, assigned)):
                cell_item_ids[int(cell)].append(item_id)
                cell_rows[int(cell)].append(position)

            cells = []
            for cell in range(n_cells):
                members = cell_item_ids[cell]
                blob = _cell_name(server_key, cell)
                if members:
                    matrix = vectors[cell_rows[cell]].astype(np.float32, copy=False)
                    store_segmented_blob(
                        db_conn, _TABLE, blob, quant.encode_vectors(matrix, code).tobytes()
                    )
                cells.append(
                    {"blob": blob, "count": len(members), "item_ids": members}
                )

            store_segmented_blob(
                db_conn, _TABLE, _centroids_name(server_key),
                centroids.astype(np.float32).tobytes(),
            )
            logger.info(
                "Hyperbolic Poincare index '%s': %d tracks into %d hyperbolic k-means cells "
                "(%d tracks/cell), stored as %s.",
                server_key, len(item_ids), n_cells,
                len(item_ids) // max(n_cells, 1), quant.dtype_name(code),
            )

            directory = {
                "version": _VERSION,
                "dim": int(vectors.shape[1]),
                "dtype": quant.dtype_name(code),
                "centroids_blob": _centroids_name(server_key),
                "cells": cells,
            }
            raw = gzip.compress(
                json.dumps(directory, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            store_segmented_blob(db_conn, _TABLE, _dir_name(server_key), raw)
        db_conn.commit()
    except Exception:
        logger.exception("Hyperbolic Poincare index build failed")
        try:
            db_conn.rollback()
        except Exception:
            pass


def _load_directory(db_conn, name):
    from tasks.index_build_helpers import load_segmented_blob

    try:
        raw = load_segmented_blob(db_conn, _TABLE, name)
    except Exception:
        logger.exception("Hyperbolic Poincare index directory load failed")
        return None
    if raw is None:
        return None
    try:
        directory = json.loads(gzip.decompress(raw).decode("utf-8"))
    except Exception:
        logger.exception("Hyperbolic Poincare index directory is malformed")
        return None
    if directory.get("version") != _VERSION:
        return None
    return directory


def _scan_index_names():
    from database import get_db

    prefix = _DIR_PREFIX + "__"
    like = prefix.replace("_", r"\_") + "%"
    out = set()
    try:
        db_conn = get_db()
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT name FROM ivf_dir WHERE name LIKE %s ESCAPE '\\'",
                (like,),
            )
            for (raw,) in cur.fetchall():
                base = re.sub(r"_\d+_\d+$", "", raw)
                if base.startswith(prefix):
                    out.add(base[len(prefix):])
    except Exception:
        logger.exception("Hyperbolic Poincare index name scan failed")
    return sorted(out)


def _clear_cell_cache():
    global _CELL_CACHE_BYTES
    with _CELL_CACHE_LOCK:
        _CELL_CACHE.clear()
        _CELL_CACHE_BYTES = 0


def _load_centroids(db_conn, blob_name, dim):
    from tasks.hyperbolic_geometry import clip_into_ball
    from tasks.index_build_helpers import load_segmented_blob

    if not blob_name:
        return None
    try:
        raw = load_segmented_blob(db_conn, _TABLE, blob_name)
    except Exception:
        logger.exception("Hyperbolic Poincare index centroid load failed")
        return None
    if raw is None:
        return None
    matrix = np.frombuffer(raw, dtype=np.float32).reshape(-1, dim)
    if matrix.shape[0] == 0:
        return None
    return clip_into_ball(matrix.astype(np.float64))


def load_hyperbolic_index(force_reload=False):
    from database import get_db

    if not force_reload and _INDEX_CACHE["loaded"]:
        return len(_INDEX_CACHE["servers"])
    with _INDEX_CACHE_LOCK:
        _clear_cell_cache()
        db_conn = get_db()
        servers = {}
        default_id = _resolve_default_server_id()
        for name in _scan_index_names():
            directory = _load_directory(db_conn, _DIR_PREFIX + "__" + name)
            if directory is None:
                continue
            dim = int(directory["dim"])
            centroids = _load_centroids(db_conn, directory.get("centroids_blob"), dim)
            if centroids is None:
                logger.warning(
                    "Hyperbolic Poincare index '%s' has no centroid blob; skipping it.", name
                )
                continue
            servers[name] = {
                "server_key": name,
                "dim": dim,
                "code": quant.dtype_code(directory.get("dtype")),
                "centroids": centroids,
                "centroid_norms2": np.sum(centroids * centroids, axis=1),
                "cells": directory["cells"],
            }
        default_entry = servers.get(_DEFAULT_SERVER_KEY)
        if default_entry is not None and default_id:
            servers[default_id] = default_entry
        _INDEX_CACHE["servers"] = servers
        _INDEX_CACHE["loaded"] = True
        return len(servers)


def get_hyperbolic_index_stats():
    servers = _INDEX_CACHE.get("servers", {})
    unique = {entry["server_key"]: entry for entry in servers.values()}
    song_count = sum(
        sum(int(cell["count"]) for cell in entry["cells"]) for entry in unique.values()
    )
    cell_count = sum(len(entry["cells"]) for entry in unique.values())
    return {
        "server_count": len(unique),
        "song_count": song_count,
        "cell_count": cell_count,
    }


def reset_hyperbolic_index():
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE["loaded"] = False
        _INDEX_CACHE["servers"] = {}
    _clear_cell_cache()


def ensure_hyperbolic_index_loaded():
    if _INDEX_CACHE["loaded"]:
        return True
    try:
        load_hyperbolic_index()
    except Exception:
        logger.exception("On-demand hyperbolic Poincare index load failed")
    return _INDEX_CACHE["loaded"]


def _index_for(server_id):
    key = server_id or _DEFAULT_SERVER_KEY
    return _INDEX_CACHE.get("servers", {}).get(key)


def _cache_cell(key, vectors, item_ids):
    global _CELL_CACHE_BYTES
    nbytes = int(vectors.nbytes) + sum(len(i) for i in item_ids)
    cap = int(config.HYPERBOLIC_INDEX_CACHE_MB) * 1024 * 1024
    if nbytes > cap:
        return
    with _CELL_CACHE_LOCK:
        while _CELL_CACHE and _CELL_CACHE_BYTES + nbytes > cap:
            _, (old_vecs, old_ids) = _CELL_CACHE.popitem(last=False)
            _CELL_CACHE_BYTES -= int(old_vecs.nbytes) + sum(len(i) for i in old_ids)
        _CELL_CACHE[key] = (vectors, item_ids)
        _CELL_CACHE_BYTES += nbytes


def _prefetch_cells(cells, index):
    from database import get_db
    from tasks.index_build_helpers import load_segmented_blob

    server_key = index["server_key"]
    wanted = []
    seen = set()
    with _CELL_CACHE_LOCK:
        for cell in cells:
            cell = int(cell)
            if cell in seen:
                continue
            seen.add(cell)
            if (server_key, cell) not in _CELL_CACHE:
                wanted.append(cell)
    if not wanted:
        return {}
    names = {index["cells"][c]["blob"]: c for c in wanted}
    db_conn = get_db()
    found = {}
    with db_conn.cursor() as cur:
        cur.execute(
            f"SELECT name, blob_data FROM {_TABLE} WHERE name = ANY(%s)",
            (list(names),),
        )
        for name, data in cur.fetchall():
            if data is not None:
                found[name] = bytes(data)
    stored_dtype = quant.np_dtype(index["code"])
    loaded = {}
    for name, cell in names.items():
        data = found.get(name)
        if data is None:
            data = load_segmented_blob(db_conn, _TABLE, name)
        meta = index["cells"][cell]
        if data is None:
            vectors = np.empty((0, index["dim"]), dtype=stored_dtype)
        else:
            vectors = np.frombuffer(data, dtype=stored_dtype).reshape(-1, index["dim"])
        _cache_cell((server_key, cell), vectors, meta["item_ids"])
        loaded[cell] = (vectors, meta["item_ids"])
    return loaded


def _load_cell(cell, index):
    key = (index["server_key"], cell)
    with _CELL_CACHE_LOCK:
        entry = _CELL_CACHE.get(key)
        if entry is not None:
            _CELL_CACHE.move_to_end(key)
            return entry

    from database import get_db
    from tasks.index_build_helpers import load_segmented_blob

    meta = index["cells"][cell]
    data = load_segmented_blob(get_db(), _TABLE, meta["blob"])
    stored_dtype = quant.np_dtype(index["code"])
    if data is None:
        vectors = np.empty((0, index["dim"]), dtype=stored_dtype)
    else:
        vectors = np.frombuffer(data, dtype=stored_dtype).reshape(-1, index["dim"])
    item_ids = meta["item_ids"]
    _cache_cell(key, vectors, item_ids)
    return vectors, item_ids


def _decode_cell(vectors, code):
    from tasks.hyperbolic_geometry import clip_into_ball

    if code == quant.DTYPE_F32:
        return np.asarray(vectors, dtype=np.float64)
    return clip_into_ball(quant.decode_row(vectors, code).astype(np.float64))


def _cell_distances(vec, vectors, code):
    from tasks.hyperbolic_geometry import hyperbolic_distances_to

    n = vectors.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, _SCAN_CHUNK):
        stop = start + _SCAN_CHUNK
        out[start:stop] = hyperbolic_distances_to(
            vec, _decode_cell(vectors[start:stop], code)
        )
    return out


def _probe_order(vec, index):
    from tasks.hyperbolic_geometry import hyperbolic_distances_to

    centroids = index["centroids"]
    distances = hyperbolic_distances_to(vec, centroids)
    nprobe = max(1, min(int(config.IVF_NPROBE), centroids.shape[0]))
    if nprobe >= centroids.shape[0]:
        return np.argsort(distances)
    picked = np.argpartition(distances, nprobe - 1)[:nprobe]
    return picked[np.argsort(distances[picked])]


def _nearest(vector, k, index, exclude):
    k = max(1, int(k))
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    probe = _probe_order(vec, index)
    try:
        prefetched = _prefetch_cells(probe, index)
    except Exception:
        logger.exception("Hyperbolic Poincare cell prefetch failed; falling back to per-cell reads")
        prefetched = {}
    heap = []
    for cell in probe:
        cell = int(cell)
        vectors, item_ids = prefetched.get(cell) or _load_cell(cell, index)
        if vectors.shape[0] == 0:
            continue
        distances = _cell_distances(vec, vectors, index["code"])
        for item_id, distance in zip(item_ids, distances):
            if item_id in exclude:
                continue
            dist = float(distance)
            if len(heap) < k:
                heapq.heappush(heap, (-dist, item_id))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, item_id))
    ranked = sorted(((-neg, item_id) for neg, item_id in heap), key=lambda pair: pair[0])
    return [(item_id, dist) for dist, item_id in ranked]


def _scan_width(k, code):
    if code == quant.DTYPE_F32:
        return k
    return min(max(k * _RERANK_OVERFETCH, k), max(k, _RERANK_SCAN_CAP))


def _rerank_exact(vector, candidates, k, code):
    if code == quant.DTYPE_F32 or not candidates:
        return candidates[:k]
    from tasks.hyperbolic_geometry import hyperbolic_distances_to
    from tasks.hyperbolic_manager import fetch_poincare_rows

    ids = [item_id for item_id, _distance in candidates]
    rows = fetch_poincare_rows(ids)
    exact_ids = [item_id for item_id in ids if item_id in rows]
    if not exact_ids:
        return candidates[:k]
    vectors = np.stack([rows[item_id][0] for item_id in exact_ids]).astype(np.float64)
    distances = hyperbolic_distances_to(np.asarray(vector, dtype=np.float64), vectors)
    order = np.argsort(distances)
    return [(exact_ids[i], float(distances[i])) for i in order[:k]]


def hyperbolic_nearest(vector, k, server_id=None, exclude=frozenset()):
    if not ensure_hyperbolic_index_loaded():
        return None
    index = _index_for(server_id)
    if index is None:
        return None
    k = max(1, int(k))
    code = index["code"]
    return _rerank_exact(vector, _nearest(vector, _scan_width(k, code), index, exclude), k, code)


def hyperbolic_nearest_multi(vectors, k, server_id=None, exclude=frozenset()):
    if not ensure_hyperbolic_index_loaded():
        return None
    index = _index_for(server_id)
    if index is None:
        return None
    k = max(1, int(k))
    scan = _scan_width(k, index["code"])
    seen = {}
    for vector in vectors:
        for item_id, _distance in _nearest(vector, scan, index, exclude):
            seen.setdefault(item_id, None)
    return list(seen)
