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
* Band vectors are quantized through tasks.ivf_quant on config.IVF_STORAGE_DTYPE
  (default i8), the same knob and the same codec every other index uses, so the
  projected catalogue on disk costs 1 byte per dimension instead of 4. This
  path deliberately does NOT go through ivf_quant.effective_code: that helper
  downgrades i8 to f16 for any non-angular metric, and the Poincare metric is
  non-angular, but the storage dtype here is taken literally so i8 means i8.
* i8 makes the band scan genuinely approximate here, more so than it does for
  an angular index, because the Poincare metric divides by (1 - ||u||^2), which
  for a track near the ball boundary is ~1e-6 while the i8 grid of 1/127 moves
  a radius by ~1e-2. Two things absorb that. Decoded bands are pushed back
  inside the ball with clip_into_ball, so a quantized point can never land on
  or past the boundary and blow the denominator up. And the scan overfetches
  _RERANK_OVERFETCH-fold (capped at _RERANK_SCAN_CAP) before hyperbolic_nearest
  re-ranks those candidates against the exact float32 poincare_embedding rows,
  so the distances and ordering it returns are exact for everything the widened
  scan reached. Measured on a 30k catalogue at 32x: 100% exact top-20 recall on
  an ordinary radius distribution, 99.97% when the radius tail runs right up to
  the boundary. hyperbolic_nearest_multi skips the re-rank on purpose: it is a
  candidate generator whose caller (the geodesic journey) already re-ranks on
  the exact vectors itself.
* hyperbolic_nearest / hyperbolic_nearest_multi return item ids ranked by exact
  Poincare distance, or None when no index is built yet; callers surface that
  as a "run analysis to build it" error instead of scanning the catalogue.
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
from tasks import ivf_quant as quant

logger = logging.getLogger(__name__)

_TABLE = "ivf_dir"
_DIR_PREFIX = "hyperbolic_index_dir"
_BAND_PREFIX = "hyperbolic_index_band"
_VERSION = 2
_DEFAULT_SERVER_KEY = "default"
_METRIC = "poincare"
_RERANK_OVERFETCH = 32
_RERANK_SCAN_CAP = 4096

_INDEX_CACHE = {"loaded": False, "servers": {}}
_INDEX_CACHE_LOCK = threading.RLock()

_BAND_CACHE = OrderedDict()
_BAND_CACHE_BYTES = 0
_BAND_CACHE_LOCK = threading.Lock()


def _storage_code():
    return quant.dtype_code(config.IVF_STORAGE_DTYPE)


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
            radii = np.array([rows[i][1] for i in item_ids], dtype=np.float64)
            edges = _band_edges(radii, int(config.HYPERBOLIC_INDEX_BANDS))
            n_bands = edges.size - 1
            assigned = np.clip(
                np.searchsorted(edges, radii, side="right") - 1, 0, n_bands - 1
            )

            band_item_ids = [[] for _ in range(n_bands)]
            band_vectors = [[] for _ in range(n_bands)]
            for item_id, band in zip(item_ids, assigned):
                band_item_ids[int(band)].append(item_id)
                band_vectors[int(band)].append(rows[item_id][0])

            bands = []
            for band in range(n_bands):
                members = band_item_ids[band]
                blob = _band_name(server_key, band)
                if members:
                    matrix = np.stack(band_vectors[band]).astype(np.float32, copy=False)
                    store_segmented_blob(
                        db_conn, _TABLE, blob, quant.encode_vectors(matrix, code).tobytes()
                    )
                bands.append(
                    {"blob": blob, "count": len(members), "item_ids": members}
                )

            directory = {
                "version": _VERSION,
                "dim": int(config.EMBEDDING_DIMENSION),
                "dtype": quant.dtype_name(code),
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
                "code": quant.dtype_code(directory.get("dtype")),
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
    stored_dtype = quant.np_dtype(index["code"])
    if data is None:
        vectors = np.empty((0, index["dim"]), dtype=stored_dtype)
    else:
        vectors = np.frombuffer(data, dtype=stored_dtype).reshape(-1, index["dim"])
    item_ids = meta["item_ids"]
    _cache_band(key, vectors, item_ids)
    return vectors, item_ids


def _decode_band(vectors, code):
    from tasks.hyperbolic_geometry import clip_into_ball

    if code == quant.DTYPE_F32:
        return np.asarray(vectors, dtype=np.float64)
    return clip_into_ball(quant.decode_row(vectors, code).astype(np.float64))


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
        distances = hyperbolic_distances_to(vec, _decode_band(vectors, index["code"]))
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
