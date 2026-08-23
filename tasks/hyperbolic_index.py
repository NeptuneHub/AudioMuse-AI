# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Disk-paged exact Poincare index for the Hyperbolic Explorer.

An exact nearest-neighbour index over the Poincare-projected embeddings, built
per music server and persisted as segmented blobs in ivf_dir (the same storage
pattern the tree cache and the other indexes use). Only the band directory is
held in memory; the projected vectors live on disk and are decoded band by band
on demand under a process-wide byte cap, so the full projected catalogue never
stays resident.

Main Features:
* The catalogue is split into radial bands from the observed radius
  distribution. The directory stores the band edges, the per-band item ids and
  the blob names; the vectors for each band are stored separately.
* Querying ranks bands by an exact radial lower bound of the Poincare distance
  and probes them in that order, stopping as soon as the next band's lower
  bound reaches the current k-th distance, so top-k results are exact without
  scanning every band.
* hyperbolic_nearest / hyperbolic_nearest_multi return item ids ranked by exact
  Poincare distance, or None when no index is built yet so callers can fall
  back to a full catalogue scan.
* Build targets mirror the Hyperbolic Explorer tree: one index per configured
  server plus the default server, keyed the same way so a request scoped to a
  server reads only that server's index.
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

logger = logging.getLogger(__name__)

_TABLE = "ivf_dir"
_DIR_PREFIX = "hyperbolic_index_dir"
_BAND_PREFIX = "hyperbolic_index_band"
_VERSION = 1
_DEFAULT_SERVER_KEY = "default"

_INDEX_CACHE = {"loaded": False, "servers": {}}
_INDEX_CACHE_LOCK = threading.RLock()

_BAND_CACHE = OrderedDict()
_BAND_CACHE_BYTES = 0
_BAND_CACHE_LOCK = threading.Lock()


def _scoped_name(prefix, server_key):
    key = server_key or _DEFAULT_SERVER_KEY
    return f"{prefix}__{key}"


def _dir_name(server_key):
    return _scoped_name(_DIR_PREFIX, server_key)


def _band_name(server_key, band):
    return _scoped_name(_BAND_PREFIX, server_key) + f"__{band}"


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


def _band_edges(radii, n_bands):
    n = max(1, int(n_bands))
    quantiles = np.quantile(radii, np.linspace(0.0, 1.0, n + 1))
    edges = np.unique(quantiles).astype(np.float64)
    if edges.size < 2:
        return np.array([0.0, 1.0], dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    return edges


def _delete_index(db_conn, server_key):
    key = server_key or _DEFAULT_SERVER_KEY
    dir_name = _scoped_name(_DIR_PREFIX, key)
    band_prefix = _scoped_name(_BAND_PREFIX, key)
    dir_like = dir_name.replace("_", r"\_") + r"\_%\_%"
    band_like = band_prefix.replace("_", r"\_") + r"\_\_%"
    with db_conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ivf_dir WHERE name = %s OR name LIKE %s ESCAPE '\\' "
            "OR name LIKE %s ESCAPE '\\'",
            (dir_name, dir_like, band_like),
        )


def build_and_store_hyperbolic_index(db_conn=None):
    from database import get_db
    from tasks.hyperbolic_manager import fetch_all_poincare_rows
    from tasks.index_build_helpers import store_segmented_blob

    if db_conn is None:
        db_conn = get_db()
    try:
        for server_key, server_id, is_default in _build_targets():
            _delete_index(db_conn, server_key)
            rows = fetch_all_poincare_rows(
                server_id=server_id, include_legacy_default=is_default
            )
            if not rows:
                continue
            item_ids = list(rows.keys())
            vectors = np.stack([rows[i][0] for i in item_ids]).astype(np.float32)
            radii = np.array([rows[i][1] for i in item_ids], dtype=np.float64)
            edges = _band_edges(radii, int(config.HYPERBOLIC_INDEX_BANDS))
            n_bands = edges.size - 1
            assigned = np.clip(
                np.searchsorted(edges, radii, side="right") - 1, 0, n_bands - 1
            )

            band_item_ids = [[] for _ in range(n_bands)]
            band_vectors = [[] for _ in range(n_bands)]
            for item_id, vector, band in zip(item_ids, vectors, assigned):
                band_item_ids[int(band)].append(item_id)
                band_vectors[int(band)].append(vector)

            bands = []
            for band in range(n_bands):
                members = band_item_ids[band]
                blob = _band_name(server_key, band)
                if members:
                    matrix = np.stack(band_vectors[band]).astype(np.float32)
                    store_segmented_blob(db_conn, _TABLE, blob, matrix.tobytes())
                bands.append(
                    {"blob": blob, "count": len(members), "item_ids": members}
                )

            directory = {
                "version": _VERSION,
                "dim": int(config.EMBEDDING_DIMENSION),
                "band_edges": [float(e) for e in edges],
                "bands": bands,
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


def _clear_band_cache():
    global _BAND_CACHE_BYTES
    with _BAND_CACHE_LOCK:
        _BAND_CACHE.clear()
        _BAND_CACHE_BYTES = 0


def load_hyperbolic_index(force_reload=False):
    from database import get_db

    if not force_reload and _INDEX_CACHE["loaded"]:
        return len(_INDEX_CACHE["servers"])
    with _INDEX_CACHE_LOCK:
        _clear_band_cache()
        db_conn = get_db()
        servers = {}
        default_id = _resolve_default_server_id()
        for name in _scan_index_names():
            directory = _load_directory(db_conn, _DIR_PREFIX + "__" + name)
            if directory is None:
                continue
            servers[name] = {
                "server_key": name,
                "dim": int(directory["dim"]),
                "band_edges": np.asarray(directory["band_edges"], dtype=np.float64),
                "bands": directory["bands"],
            }
        default_entry = servers.get(_DEFAULT_SERVER_KEY)
        if default_entry is not None and default_id:
            servers[default_id] = default_entry
        _INDEX_CACHE["servers"] = servers
        _INDEX_CACHE["loaded"] = True
        return len(servers)


def reset_hyperbolic_index():
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE["loaded"] = False
        _INDEX_CACHE["servers"] = {}
    _clear_band_cache()


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


def _radial_lower_bound(rp, lo, hi):
    if lo <= rp <= hi:
        return 0.0
    r_star = lo if rp < lo else hi
    diff2 = (rp - r_star) * (rp - r_star)
    denom = max((1.0 - rp * rp) * (1.0 - r_star * r_star), 1e-12)
    return float(np.arccosh(max(1.0 + 2.0 * diff2 / denom, 1.0)))


def _cache_band(key, vectors, item_ids):
    global _BAND_CACHE_BYTES
    nbytes = int(vectors.nbytes) + sum(len(i) for i in item_ids)
    cap = int(config.HYPERBOLIC_INDEX_CACHE_MB) * 1024 * 1024
    if nbytes > cap:
        return
    with _BAND_CACHE_LOCK:
        while _BAND_CACHE and _BAND_CACHE_BYTES + nbytes > cap:
            _, (old_vecs, old_ids) = _BAND_CACHE.popitem(last=False)
            _BAND_CACHE_BYTES -= int(old_vecs.nbytes) + sum(len(i) for i in old_ids)
        _BAND_CACHE[key] = (vectors, item_ids)
        _BAND_CACHE_BYTES += nbytes


def _load_band(band, index):
    key = (index["server_key"], band)
    with _BAND_CACHE_LOCK:
        entry = _BAND_CACHE.get(key)
        if entry is not None:
            _BAND_CACHE.move_to_end(key)
            return entry

    from database import get_db
    from tasks.index_build_helpers import load_segmented_blob

    meta = index["bands"][band]
    data = load_segmented_blob(get_db(), _TABLE, meta["blob"])
    if data is None:
        vectors = np.empty((0, index["dim"]), dtype=np.float32)
    else:
        vectors = np.frombuffer(data, dtype=np.float32).reshape(-1, index["dim"])
    item_ids = meta["item_ids"]
    _cache_band(key, vectors, item_ids)
    return vectors, item_ids


def _nearest(vector, k, index, exclude):
    from tasks.hyperbolic_geometry import hyperbolic_distances_to

    k = max(1, int(k))
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    rp = float(np.linalg.norm(vec))
    edges = index["band_edges"]
    bands = index["bands"]
    n_bands = len(bands)
    lower = [
        _radial_lower_bound(rp, float(edges[b]), float(edges[b + 1]))
        for b in range(n_bands)
    ]
    order = sorted(range(n_bands), key=lambda b: lower[b])
    heap = []
    threshold = float("inf")
    for b in order:
        if len(heap) >= k and lower[b] >= threshold:
            break
        vectors, item_ids = _load_band(b, index)
        if vectors.shape[0] == 0:
            continue
        distances = hyperbolic_distances_to(vec, vectors)
        for item_id, distance in zip(item_ids, distances):
            if item_id in exclude:
                continue
            dist = float(distance)
            if len(heap) < k:
                heapq.heappush(heap, (-dist, item_id))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, item_id))
        if heap:
            threshold = -heap[0][0]
    ranked = sorted(((-neg, item_id) for neg, item_id in heap), key=lambda pair: pair[0])
    return [(item_id, dist) for dist, item_id in ranked]


def hyperbolic_nearest(vector, k, server_id=None, exclude=frozenset()):
    if not ensure_hyperbolic_index_loaded():
        return None
    index = _index_for(server_id)
    if index is None:
        return None
    return _nearest(vector, int(k), index, exclude)


def hyperbolic_nearest_multi(vectors, k, server_id=None, exclude=frozenset()):
    if not ensure_hyperbolic_index_loaded():
        return None
    index = _index_for(server_id)
    if index is None:
        return None
    seen = {}
    for vector in vectors:
        for item_id, _distance in _nearest(vector, int(k), index, exclude):
            seen.setdefault(item_id, None)
    return list(seen)
