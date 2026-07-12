# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Disk-paged IVF index: on-disk format, cell caches and query path.

The storage and read engine shared by all six similarity indexes. Vectors are
grouped into IVF cells persisted as segmented blobs in Postgres, exported to a
local cell file, and mmap-paged at query time so only IVF_NPROBE cells are read
per request. Callers (tasks.ivf_manager, tasks.lyrics_manager) build via
build_and_store_paged_ivf and query via the PagedIvfIndex returned by
load_paged_ivf_index; distance math is delegated to tasks.ivf_quant.

Main Features:
* PagedIvfIndex: nprobe cell selection with single-round-trip ANY() reads over
  the mmap'd cell file; angular vectors are stored normalized (header flag) so
  scans skip renormalizing.
* Process-wide L2 cell cache (IVF_GLOBAL_CACHE_MB) layered over a per-request L1,
  with opt-in IVF_PRELOAD_ALL; STORAGE EXTERNAL blobs avoid TOAST compression.
* Idle mmap page-dropping (Windows and POSIX) and idle callbacks that return
  freed heap to the OS without touching glibc arena tuning.
"""

from __future__ import annotations

import glob
import hashlib
import io
import logging
import mmap
import os
import platform
import struct
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import psycopg2

import config

from . import ivf_quant as quant

logger = logging.getLogger(__name__)

_warned_numkong_missing = False


def _warn_numkong_missing_once(dtype_name: str) -> None:
    global _warned_numkong_missing
    if _warned_numkong_missing:
        return
    _warned_numkong_missing = True
    logger.warning(
        "NumKong native kernels unavailable (%s); %s IVF cells fall back to the NumPy "
        "distance path (correct results, slower per-scan compute). Install the numkong wheel "
        "for this platform, or set IVF_STORAGE_DTYPE=f32 to skip quantization.",
        quant.NUMKONG_IMPORT_ERROR or "unknown import failure",
        dtype_name,
    )


_MAGIC = b"AMIV"
_VERSION = 1
_HEADER_FMT = "<4sIBBBxIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_METRIC_TO_CODE = {"angular": 0, "euclidean": 1, "dot": 2}
_CODE_TO_METRIC = {v: k for k, v in _METRIC_TO_CODE.items()}

IVF_DIR_TABLE = "ivf_dir"
IVF_CELL_TABLE = "ivf_cell"


def _metric_code(metric: str) -> int:
    return _METRIC_TO_CODE.get((metric or "angular").lower(), 0)


def _normalize_rows(mat: np.ndarray, inplace: bool = False) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0.0] = np.float32(1.0)
    if inplace:
        mat /= norms
        return mat
    return (mat / norms).astype(np.float32, copy=False)


def pack_directory(
    centroids: np.ndarray,
    id2cell: np.ndarray,
    item_ids: List[str],
    dim: int,
    metric: str,
    normalized: bool = False,
    storage_dtype: int = 0,
) -> bytes:
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    id2cell = np.ascontiguousarray(id2cell, dtype=np.uint32)
    nlist = centroids.shape[0]
    n_items = len(item_ids)
    if id2cell.shape[0] != n_items:
        raise ValueError(f"id2cell length {id2cell.shape[0]} != n_items {n_items}")
    if centroids.shape[1] != dim:
        raise ValueError(f"centroid dim {centroids.shape[1]} != dim {dim}")

    buf = io.BytesIO()
    buf.write(
        struct.pack(
            _HEADER_FMT,
            _MAGIC,
            _VERSION,
            _metric_code(metric),
            1 if normalized else 0,
            int(storage_dtype),
            dim,
            nlist,
            n_items,
        )
    )
    buf.write(centroids.tobytes())
    buf.write(id2cell.tobytes())
    id_blob = io.BytesIO()
    for item_id in item_ids:
        raw = item_id.encode("utf-8")
        if len(raw) > 0xFFFF:
            raise ValueError(f"item_id too long for uint16 length prefix: {len(raw)} bytes")
        id_blob.write(struct.pack("<H", len(raw)))
        id_blob.write(raw)
    buf.write(id_blob.getvalue())
    return buf.getvalue()


def unpack_directory(blob: bytes) -> Tuple[np.ndarray, np.ndarray, List[str], int, str, bool, int]:
    if len(blob) < _HEADER_SIZE:
        raise ValueError(f"directory blob too short ({len(blob)} bytes)")
    magic, version, metric_code, normalized, storage_dtype, dim, nlist, n_items = (
        struct.unpack_from(_HEADER_FMT, blob, 0)
    )
    if magic != _MAGIC:
        raise ValueError(f"directory magic mismatch: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported directory version: {version}")
    pos = _HEADER_SIZE
    cent_count = nlist * dim
    centroids = np.frombuffer(blob, dtype=np.float32, count=cent_count, offset=pos).reshape(
        nlist, dim
    )
    pos += cent_count * 4
    id2cell = np.frombuffer(blob, dtype=np.uint32, count=n_items, offset=pos).copy()
    pos += n_items * 4
    item_ids: List[str] = []
    for _ in range(n_items):
        (slen,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        item_ids.append(blob[pos : pos + slen].decode("utf-8"))
        pos += slen
    return (
        centroids.copy(),
        id2cell,
        item_ids,
        int(dim),
        _CODE_TO_METRIC.get(metric_code, "angular"),
        bool(normalized),
        int(storage_dtype),
    )


def pack_cell(int_ids: np.ndarray, vecs: np.ndarray, storage_dtype: int = 0) -> bytes:
    int_ids = np.ascontiguousarray(int_ids, dtype=np.int32)
    enc = np.ascontiguousarray(quant.encode_vectors(vecs, storage_dtype))
    return int_ids.tobytes() + enc.tobytes()


def unpack_cell(blob: bytes, dim: int, storage_dtype: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    record = 4 + dim * quant.elem_size(storage_dtype)
    if record <= 0 or len(blob) % record != 0:
        raise ValueError(f"cell blob size {len(blob)} not a multiple of record size {record}")
    n = len(blob) // record
    ids = np.frombuffer(blob, dtype=np.int32, count=n, offset=0)
    vecs = np.frombuffer(
        blob, dtype=quant.np_dtype(storage_dtype), count=n * dim, offset=n * 4
    ).reshape(n, dim)
    return ids, vecs


_CELLFILE_MAGIC = b"AMVF"
_CELLFILE_VERSION = 2
_CELLFILE_HEADER_FMT_V1 = "<4sIIII"
_CELLFILE_HEADER_FMT = "<4sIIIII"
_CELLFILE_HEADER_SIZE = struct.calcsize(_CELLFILE_HEADER_FMT)
_CELLFILE_ROW_FMT = "<IQQ"
_CELLFILE_ROW_SIZE = struct.calcsize(_CELLFILE_ROW_FMT)
_IVF_FILE_SWAP_LOCK = threading.Lock()


def _cell_file_path(cache_dir: str, index_name: str, build_id: str) -> str:
    return os.path.join(cache_dir, f"{index_name}.{build_id}.amivf")


def _prune_old_cell_files(cache_dir: str, index_name: str, keep_path: str) -> None:
    keep = os.path.abspath(keep_path)
    for pat in (f"{index_name}.*.amivf", f"{index_name}.*.amivf.tmp"):
        for p in glob.glob(os.path.join(cache_dir, pat)):
            if os.path.abspath(p) == keep:
                continue
            try:
                os.remove(p)
            except OSError as e:
                logger.info(
                    "IVF index '%s': stale cell file %s not deleted yet (%s); will retry on next load.",
                    index_name,
                    p,
                    e,
                )


def _export_cells_to_file(
    db_conn, index_name: str, dim: int, metric: str, storage_dtype: int, path: str
) -> int:
    record = 4 + int(dim) * quant.elem_size(storage_dtype)
    with db_conn.cursor() as cur:
        cur.execute(
            f"SELECT cell_id, octet_length(cell_data) FROM {IVF_CELL_TABLE} "
            f"WHERE index_name = %s ORDER BY cell_id",
            (index_name,),
        )
        sizes = [(int(cid), int(ln)) for cid, ln in cur.fetchall() if ln and int(ln) > 0]

    n_cells = len(sizes)
    offset = _CELLFILE_HEADER_SIZE + n_cells * _CELLFILE_ROW_SIZE
    table = []
    exp_len = {}
    for cid, ln in sizes:
        table.append((cid, offset, ln))
        exp_len[cid] = ln
        offset += ln
    order = [cid for cid, _ln in sizes]

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(
            struct.pack(
                _CELLFILE_HEADER_FMT,
                _CELLFILE_MAGIC,
                _CELLFILE_VERSION,
                int(dim),
                n_cells,
                _metric_code(metric),
                int(storage_dtype),
            )
        )
        for cid, off, ln in table:
            f.write(struct.pack(_CELLFILE_ROW_FMT, cid, off, ln))
        chunk = 64
        with db_conn.cursor() as cur:
            for start in range(0, n_cells, chunk):
                ids_chunk = order[start : start + chunk]
                cur.execute(
                    f"SELECT cell_id, cell_data FROM {IVF_CELL_TABLE} "
                    f"WHERE index_name = %s AND cell_id = ANY(%s)",
                    (index_name, ids_chunk),
                )
                blobs = {int(c): bytes(b) for c, b in cur.fetchall()}
                for cid in ids_chunk:
                    b = blobs.get(cid)
                    if b is None or len(b) != exp_len[cid] or len(b) % record != 0:
                        raise ValueError(
                            f"cell {cid} of '{index_name}' changed or malformed during export"
                        )
                    f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return n_cells


def _open_cell_file(path: str):
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    magic, version = struct.unpack_from("<4sI", bytes(mm[:8]), 0)
    if magic != _CELLFILE_MAGIC:
        raise ValueError(f"cell file magic mismatch: {magic!r}")
    if version == 1:
        hsize = struct.calcsize(_CELLFILE_HEADER_FMT_V1)
        _m, _v, dim, n_cells, _metric_code_v = struct.unpack(
            _CELLFILE_HEADER_FMT_V1, bytes(mm[:hsize])
        )
        storage_dtype = quant.DTYPE_F32
    elif version == 2:
        hsize = _CELLFILE_HEADER_SIZE
        _m, _v, dim, n_cells, _metric_code_v, storage_dtype = struct.unpack(
            _CELLFILE_HEADER_FMT, bytes(mm[:hsize])
        )
    else:
        raise ValueError(f"unsupported cell file version: {version}")
    table = bytes(mm[hsize : hsize + n_cells * _CELLFILE_ROW_SIZE])
    offsets = {}
    for i in range(n_cells):
        cid, off, ln = struct.unpack_from(_CELLFILE_ROW_FMT, table, i * _CELLFILE_ROW_SIZE)
        offsets[int(cid)] = (int(off), int(ln))
    return mm, int(dim), offsets, int(storage_dtype)


def _vec_in_cell(ids: np.ndarray, vecs: np.ndarray, int_id: int) -> Optional[np.ndarray]:
    if ids.size == 0:
        return None
    pos = int(np.searchsorted(ids, int_id)) if ids[0] <= int_id else -1
    if 0 <= pos < ids.shape[0] and int(ids[pos]) == int(int_id):
        return vecs[pos]
    match = np.where(ids == int_id)[0]
    if match.size:
        return vecs[int(match[0])]
    return None


class _CellLruCache:
    def __init__(self, record_size: int, max_bytes: int):
        self._record_size = record_size
        self._max_bytes = max(max_bytes, record_size)
        self._cells: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._bytes = 0

    def _evict_until_fits(self, incoming: int) -> None:
        while self._bytes + incoming > self._max_bytes and self._cells:
            _old_id, (ids, _vecs) = self._cells.popitem(last=False)
            self._bytes -= int(ids.shape[0]) * self._record_size

    def add_cell(self, cell_id: int, ids: np.ndarray, vecs: np.ndarray) -> None:
        if cell_id in self._cells:
            self._cells.move_to_end(cell_id)
            return
        size = int(ids.shape[0]) * self._record_size
        self._evict_until_fits(size)
        self._cells[cell_id] = (ids, vecs)
        self._bytes += size

    def has_cell(self, cell_id: int) -> bool:
        return cell_id in self._cells

    def get_cell(self, cell_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        entry = self._cells.get(cell_id)
        if entry is not None:
            self._cells.move_to_end(cell_id)
        return entry

    def resident_bytes(self) -> int:
        return self._bytes


class _GlobalCellCache:
    def __init__(self, max_bytes: int, idle_seconds: int = 0):
        self._max_bytes = int(max_bytes)
        self._idle_seconds = int(idle_seconds)
        self._cells: "OrderedDict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._last_access = time.monotonic()
        self._timer_thread: Optional[threading.Thread] = None

    @staticmethod
    def _entry_bytes(ids: np.ndarray, vecs: np.ndarray) -> int:
        return int(ids.nbytes) + int(vecs.nbytes)

    @property
    def enabled(self) -> bool:
        return self._max_bytes > 0

    def _touch_locked(self) -> None:
        self._last_access = time.monotonic()
        if self._idle_seconds > 0 and (
            self._timer_thread is None or not self._timer_thread.is_alive()
        ):
            t = threading.Thread(target=self._idle_worker, name="ivf-l2-idle", daemon=True)
            self._timer_thread = t
            t.start()

    def _idle_worker(self) -> None:
        while True:
            dropped = None
            sleep_for = 1.0
            with self._lock:
                if self._idle_seconds <= 0 or not self._cells:
                    self._timer_thread = None
                    return
                idle = time.monotonic() - self._last_access
                if idle >= self._idle_seconds:
                    dropped = len(self._cells)
                    self._cells.clear()
                    self._bytes = 0
                    self._timer_thread = None
                else:
                    sleep_for = self._idle_seconds - idle
            if dropped is not None:
                logger.info(
                    "IVF global cell cache idle for %ds; dropped %d cells to free RAM.",
                    self._idle_seconds,
                    dropped,
                )
                _run_idle_callbacks()
                _return_freed_heap_to_os()
                return
            time.sleep(min(max(sleep_for, 1.0), 30.0))

    def get_cell(self, index_name: str, cell_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._max_bytes <= 0:
            return None
        key = (index_name, int(cell_id))
        with self._lock:
            entry = self._cells.get(key)
            if entry is None:
                return None
            self._cells.move_to_end(key)
            self._touch_locked()
            return entry

    def put_cell(self, index_name: str, cell_id: int, ids: np.ndarray, vecs: np.ndarray) -> None:
        if self._max_bytes <= 0:
            return
        key = (index_name, int(cell_id))
        size = self._entry_bytes(ids, vecs)
        if size > self._max_bytes:
            return
        with self._lock:
            if key in self._cells:
                self._cells.move_to_end(key)
                self._touch_locked()
                return
            while self._bytes + size > self._max_bytes and self._cells:
                _old_key, (old_ids, old_vecs) = self._cells.popitem(last=False)
                self._bytes -= self._entry_bytes(old_ids, old_vecs)
            self._cells[key] = (ids, vecs)
            self._bytes += size
            self._touch_locked()

    def invalidate_index(self, index_name: str) -> None:
        with self._lock:
            stale = [k for k in self._cells if k[0] == index_name]
            for k in stale:
                old_ids, old_vecs = self._cells.pop(k)
                self._bytes -= self._entry_bytes(old_ids, old_vecs)

    def clear(self) -> None:
        with self._lock:
            self._cells.clear()
            self._bytes = 0

    def resident_bytes(self) -> int:
        with self._lock:
            return self._bytes


def _return_freed_heap_to_os() -> None:
    try:
        from .memory_utils import release_memory_to_os

        release_memory_to_os()
    except Exception:
        pass


_GLOBAL_CELL_CACHE: Optional[_GlobalCellCache] = None
_GLOBAL_CELL_CACHE_LOCK = threading.Lock()


def get_global_cell_cache() -> _GlobalCellCache:
    global _GLOBAL_CELL_CACHE
    if _GLOBAL_CELL_CACHE is None:
        with _GLOBAL_CELL_CACHE_LOCK:
            if _GLOBAL_CELL_CACHE is None:
                idle_seconds = 0 if config.IVF_PRELOAD_ALL else config.IVF_GLOBAL_CACHE_IDLE_SECONDS
                _GLOBAL_CELL_CACHE = _GlobalCellCache(
                    config.IVF_GLOBAL_CACHE_MB * 1024 * 1024,
                    idle_seconds=idle_seconds,
                )
    return _GLOBAL_CELL_CACHE


def invalidate_global_cell_cache(index_name: str) -> None:
    cache = _GLOBAL_CELL_CACHE
    if cache is not None:
        cache.invalidate_index(index_name)


def begin_query(index) -> None:
    if index is not None and hasattr(index, "begin_request"):
        index.begin_request()


_QUERY_THREAD_POOL = None
_QUERY_THREAD_POOL_LOCK = threading.Lock()
_QUERY_THREAD_PREFIX = "ivf-query"


def _query_worker_count() -> int:
    try:
        cpu = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu = os.cpu_count() or 1
    return max(cpu // 2, 1)


def _query_thread_pool():
    if _query_worker_count() <= 1:
        return None
    global _QUERY_THREAD_POOL
    if _QUERY_THREAD_POOL is None:
        with _QUERY_THREAD_POOL_LOCK:
            if _QUERY_THREAD_POOL is None:
                from concurrent.futures import ThreadPoolExecutor

                _pin_blas_single_thread()
                _QUERY_THREAD_POOL = ThreadPoolExecutor(
                    max_workers=_query_worker_count(),
                    thread_name_prefix=_QUERY_THREAD_PREFIX,
                )
    return _QUERY_THREAD_POOL


def _in_query_pool_thread() -> bool:
    return threading.current_thread().name.startswith(_QUERY_THREAD_PREFIX)


def shutdown_query_pool() -> None:
    global _QUERY_THREAD_POOL
    with _QUERY_THREAD_POOL_LOCK:
        pool = _QUERY_THREAD_POOL
        _QUERY_THREAD_POOL = None
    if pool is not None:
        pool.shutdown(wait=True)


_BLAS_LIMITER = None
_BLAS_PIN_DONE = False


def _pin_blas_single_thread() -> None:
    global _BLAS_LIMITER, _BLAS_PIN_DONE
    if _BLAS_PIN_DONE:
        return
    _BLAS_PIN_DONE = True
    try:
        from threadpoolctl import threadpool_limits

        _BLAS_LIMITER = threadpool_limits(limits=1, user_api="blas")
    except Exception:
        _BLAS_LIMITER = None


_LIVE_INDEXES: "weakref.WeakSet[PagedIvfIndex]" = weakref.WeakSet()
_AVAILABILITY_CACHE = {}
_AVAILABILITY_CACHE_LOCK = threading.Lock()
_AVAILABILITY_CACHE_TTL = 30.0


def invalidate_availability_cache(server_id=None):
    """Invalidate cached per-server index masks after mapping changes."""
    with _AVAILABILITY_CACHE_LOCK:
        if server_id is None:
            _AVAILABILITY_CACHE.clear()
            return
        sid = str(server_id)
        for key in [key for key in _AVAILABILITY_CACHE if key[1] == sid]:
            _AVAILABILITY_CACHE.pop(key, None)


def active_availability_scope():
    """Return the current request/job server, or None for union/background scope.

    An unknown or disabled requested server maps to a fail-closed sentinel scope;
    any other resolution error fails open to None (union scope).
    """
    try:
        from tasks.mediaserver import context

        active = context.active_server_id()
        if active:
            return str(active)
    except Exception:
        pass
    try:
        from flask import has_request_context

        if not has_request_context():
            return None
        from app_server_context import resolve_request_server_id
        from tasks.mediaserver import registry

        requested = resolve_request_server_id()
        return str(requested or registry.get_default_server_id() or '') or None
    except ValueError:
        return '__invalid_server__'
    except Exception:
        logger.exception("Could not resolve request availability scope")
        return None


def end_all_requests() -> None:
    for idx in list(_LIVE_INDEXES):
        try:
            idx.end_request()
        except Exception:
            pass


def _close_live_indexes(index_name: str) -> None:
    for idx in list(_LIVE_INDEXES):
        try:
            if getattr(idx, "_index_name", None) == index_name:
                idx.close()
        except Exception:
            pass


_IDLE_CALLBACKS: "List[Callable[[], None]]" = []
_IDLE_CALLBACKS_LOCK = threading.Lock()


def register_idle_callback(fn: Callable[[], None]) -> None:
    with _IDLE_CALLBACKS_LOCK:
        if fn not in _IDLE_CALLBACKS:
            _IDLE_CALLBACKS.append(fn)


def unregister_idle_callback(fn: Callable[[], None]) -> None:
    with _IDLE_CALLBACKS_LOCK:
        if fn in _IDLE_CALLBACKS:
            _IDLE_CALLBACKS.remove(fn)


def _run_idle_callbacks() -> None:
    with _IDLE_CALLBACKS_LOCK:
        callbacks = list(_IDLE_CALLBACKS)
    for fn in callbacks:
        try:
            fn()
        except Exception:
            logger.debug("IVF idle callback %r failed.", fn, exc_info=True)


_MMAP_IDLE_LOCK = threading.Lock()
_MMAP_IDLE_THREAD: Optional[threading.Thread] = None


def _note_mmap_activity(index=None) -> None:
    if config.IVF_DISK_CACHE_IDLE_SECONDS <= 0:
        return
    global _MMAP_IDLE_THREAD
    with _MMAP_IDLE_LOCK:
        if index is not None:
            index._last_mmap_access = time.monotonic()
            index._mmap_pages_dropped = False
        if _MMAP_IDLE_THREAD is None or not _MMAP_IDLE_THREAD.is_alive():
            t = threading.Thread(target=_mmap_idle_worker, name="ivf-mmap-idle", daemon=True)
            _MMAP_IDLE_THREAD = t
            t.start()


_WIN_VIRTUAL_UNLOCK = None
_WIN_ERROR_NOT_LOCKED = 158


def _win_virtual_unlock():
    global _WIN_VIRTUAL_UNLOCK
    if _WIN_VIRTUAL_UNLOCK is None:
        import ctypes
        from ctypes import wintypes

        fn = ctypes.WinDLL("kernel32", use_last_error=True).VirtualUnlock
        fn.argtypes = [wintypes.LPVOID, ctypes.c_size_t]
        fn.restype = wintypes.BOOL
        _WIN_VIRTUAL_UNLOCK = (fn, ctypes.get_last_error)
    return _WIN_VIRTUAL_UNLOCK


def _drop_pages_windows(targets) -> int:
    try:
        unlock, last_error = _win_virtual_unlock()
    except Exception as e:
        logger.debug("IVF idle page-drop: VirtualUnlock unavailable (%s).", e)
        return 0
    dropped = 0
    for idx in targets:
        mm = getattr(idx, "_mmap", None)
        if mm is None:
            continue
        try:
            addr = int(mm.ctypes.data)
            size = int(mm.nbytes)
        except Exception:
            continue
        if not addr or size <= 0:
            continue
        ok = unlock(addr, size)
        err = last_error()
        if ok or err == _WIN_ERROR_NOT_LOCKED:
            dropped += 1
        else:
            logger.debug("IVF idle page-drop: VirtualUnlock failed (err=%s).", err)
    return dropped


def _drop_pages_posix(targets) -> int:
    advise = getattr(mmap, "MADV_DONTNEED", None)
    if advise is None:
        return 0
    dropped = 0
    skipped = 0
    for idx in targets:
        mm = getattr(idx, "_mmap", None)
        if mm is None:
            continue
        buf = getattr(mm, "_mmap", None)
        if buf is None or not hasattr(buf, "madvise"):
            skipped += 1
            continue
        try:
            buf.madvise(advise)
            dropped += 1
        except (OSError, ValueError) as e:
            skipped += 1
            logger.debug("IVF idle page-drop could not advise a mapping (%s).", e)
    if skipped:
        logger.debug("IVF idle page-drop skipped %d live mapping(s) it could not reach.", skipped)
    return dropped


def _drop_resident_mmap_pages(indexes=None) -> int:
    targets = list(_LIVE_INDEXES) if indexes is None else list(indexes)
    if not targets:
        return 0
    if platform.system() == "Windows":
        return _drop_pages_windows(targets)
    return _drop_pages_posix(targets)


def _collect_idle_mmap_indexes(now: float, idle_seconds: float):
    to_drop = []
    next_due = None
    for idx in list(_LIVE_INDEXES):
        if getattr(idx, "_mmap", None) is None or getattr(idx, "_mmap_pages_dropped", False):
            continue
        idle = now - getattr(idx, "_last_mmap_access", now)
        if idle >= idle_seconds:
            to_drop.append(idx)
        else:
            remaining = idle_seconds - idle
            next_due = remaining if next_due is None else min(next_due, remaining)
    return to_drop, next_due


def _mmap_idle_worker() -> None:
    global _MMAP_IDLE_THREAD
    while True:
        sleep_for = 30.0
        finished = False
        dropped = 0
        with _MMAP_IDLE_LOCK:
            idle_seconds = config.IVF_DISK_CACHE_IDLE_SECONDS
            if idle_seconds <= 0:
                _MMAP_IDLE_THREAD = None
                return
            to_drop, next_due = _collect_idle_mmap_indexes(time.monotonic(), idle_seconds)
            dropped = _drop_resident_mmap_pages(to_drop) if to_drop else 0
            for idx in to_drop:
                idx._mmap_pages_dropped = True
            if next_due is None:
                _MMAP_IDLE_THREAD = None
                finished = True
            else:
                sleep_for = next_due
        if dropped:
            logger.info(
                "IVF disk cache: dropped resident pages of %d idle index file(s) to free RAM.",
                dropped,
            )
        if finished:
            _run_idle_callbacks()
            _return_freed_heap_to_os()
            return
        time.sleep(min(max(sleep_for, 1.0), 30.0))


class PagedIvfIndex:
    def __init__(
        self,
        centroids: np.ndarray,
        id2cell: np.ndarray,
        item_ids: List[str],
        dim: int,
        metric: str,
        index_name: str,
        conn_factory: Callable[[], "psycopg2.extensions.connection"],
        nprobe: Optional[int] = None,
        query_cache_bytes: Optional[int] = None,
        read_batch_cells: Optional[int] = None,
        normalized: bool = False,
        storage_dtype: int = 0,
        mmap_obj=None,
        cell_offsets: Optional[Dict[int, Tuple[int, int]]] = None,
        track_scoped: bool = True,
    ):
        self._dim = int(dim)
        self._track_scoped = bool(track_scoped)
        self._metric = (metric or "angular").lower()
        self._normalized = bool(normalized)
        self._storage_dtype = int(storage_dtype)
        self._np_vec_dtype = quant.np_dtype(self._storage_dtype)
        self._index_name = index_name
        self._conn_factory = conn_factory
        self._mmap = mmap_obj
        self._cell_offsets = cell_offsets or {}
        self._n_items = len(item_ids)
        self._item_ids = list(item_ids)
        self._record_size = 4 + self._dim * quant.elem_size(self._storage_dtype)
        self._id2cell = np.ascontiguousarray(id2cell, dtype=np.uint32)
        self._nprobe = int(nprobe if nprobe is not None else config.IVF_NPROBE)
        _query_cache_bytes = int(
            query_cache_bytes
            if query_cache_bytes is not None
            else config.IVF_QUERY_CACHE_MB * 1024 * 1024
        )
        _max_cell_bytes = min(config.IVF_MAX_CELL_MB, config.IVF_MAX_PART_SIZE_MB) * 1024 * 1024
        self._cache_bytes = max(_query_cache_bytes, _max_cell_bytes)
        self._read_batch = int(
            read_batch_cells if read_batch_cells is not None else config.IVF_READ_BATCH_CELLS
        )
        if self._metric == "angular":
            self._centroids = _normalize_rows(np.ascontiguousarray(centroids, dtype=np.float32))
        else:
            self._centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        self._num_cells = int(self._centroids.shape[0])
        self._generation = uuid.uuid4().hex
        self._tl = threading.local()
        self._last_mmap_access = time.monotonic()
        self._mmap_pages_dropped = False
        _LIVE_INDEXES.add(self)

    def _availability_mask(self):
        if not self._track_scoped:
            return None
        server_id = active_availability_scope()
        if server_id is None:
            return None
        try:
            from tasks.mediaserver import registry

            if server_id == str(
                registry.get_default_server_id() or ''
            ) and not registry.has_secondary_servers():
                return None
        except Exception:
            logger.debug("Single-server availability fast path failed.", exc_info=True)
        key = (self._index_name, server_id, self._generation)
        now = time.monotonic()
        with _AVAILABILITY_CACHE_LOCK:
            cached = _AVAILABILITY_CACHE.get(key)
            if cached is not None and now - cached[0] < _AVAILABILITY_CACHE_TTL:
                return cached[1]
        conn = self._conn_factory()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT item_id FROM track_server_map WHERE server_id = %s",
                (server_id,),
            )
            available = {str(row[0]) for row in cur.fetchall()}
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM music_servers "
                "WHERE server_id = %s AND is_default)",
                (server_id,),
            )
            is_default = bool(cur.fetchone()[0])
        if is_default:
            from tasks.audio_fingerprint import is_fingerprint_id

            available.update(
                item_id for item_id in self._item_ids
                if not is_fingerprint_id(item_id)
            )
        mask = np.fromiter(
            (item_id in available for item_id in self._item_ids),
            dtype=np.bool_,
            count=self._n_items,
        )
        with _AVAILABILITY_CACHE_LOCK:
            stale_keys = [
                cached_key for cached_key, cached_value in _AVAILABILITY_CACHE.items()
                if (cached_key[0] == self._index_name and cached_key[2] != self._generation)
                or now - cached_value[0] >= _AVAILABILITY_CACHE_TTL
            ]
            for stale_key in stale_keys:
                _AVAILABILITY_CACHE.pop(stale_key, None)
            _AVAILABILITY_CACHE[key] = (now, mask)
        return mask

    def __len__(self) -> int:
        return self._n_items

    def close(self) -> None:
        self._mmap = None

    @property
    def num_elements(self) -> int:
        return self._n_items

    def begin_request(self) -> None:
        self._tl.cache = _CellLruCache(self._record_size, self._cache_bytes)

    def end_request(self) -> None:
        self._tl.cache = None

    def _cache(self) -> _CellLruCache:
        cache = getattr(self._tl, "cache", None)
        if cache is None:
            cache = _CellLruCache(self._record_size, self._cache_bytes)
            self._tl.cache = cache
        return cache

    def _cell_scores(self, q: np.ndarray) -> np.ndarray:
        if self._metric == "euclidean":
            diffs = self._centroids - q[None, :]
            return np.einsum("ij,ij->i", diffs, diffs)
        if self._metric == "dot":
            return -(self._centroids @ q)
        qn = q / (float(np.linalg.norm(q)) + 1e-12)
        return -(self._centroids @ qn)

    def _rank_cells(self, q: np.ndarray) -> np.ndarray:
        scores = self._cell_scores(q)
        n = scores.shape[0]
        topn = max(1, self._nprobe)
        if topn >= n:
            return np.argsort(scores)
        part = np.argpartition(scores, topn - 1)[:topn]
        return part[np.argsort(scores[part])]

    def _farthest_cells(self, q: np.ndarray, k: int) -> np.ndarray:
        scores = self._cell_scores(q)
        n = scores.shape[0]
        if k >= n:
            return np.arange(n)
        return np.argpartition(scores, n - k)[n - k :]

    def _cell_from_mmap(self, mm, offsets, cell_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        rec = offsets.get(int(cell_id))
        if rec is None:
            return None
        off, ln = rec
        sub = mm[off : off + ln]
        n = ln // self._record_size
        if n == 0:
            return None
        ids = sub[: 4 * n].view(np.int32)
        vecs = sub[4 * n :].view(self._np_vec_dtype).reshape(n, self._dim)
        return ids, vecs

    def _iter_db_cells(self, cur, cell_ids):
        cur.execute(
            f"SELECT cell_id, cell_data FROM {IVF_CELL_TABLE} "
            f"WHERE index_name = %s AND cell_id = ANY(%s)",
            (self._index_name, list(cell_ids)),
        )
        for cell_id, blob in cur.fetchall():
            ids, vecs = unpack_cell(blob, self._dim, self._storage_dtype)
            yield int(cell_id), ids, vecs

    def _read_cells_mmap(
        self, mm, offsets, cell_ids: List[int]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        _note_mmap_activity(self)
        out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        ordered = sorted(cell_ids, key=lambda c: offsets.get(int(c), (1 << 62, 0))[0])
        for c in ordered:
            cid = int(c)
            cell = self._cell_from_mmap(mm, offsets, cid)
            if cell is not None:
                out[cid] = cell
        return out

    def _lookup_cached_cell(self, cid: int, cache: _CellLruCache, gcache):
        entry = cache.get_cell(cid)
        if entry is None:
            entry = gcache.get_cell(self._index_name, cid)
            if entry is not None:
                cache.add_cell(cid, entry[0], entry[1])
        return entry

    def _read_cells(
        self, cell_ids: List[int], cache: _CellLruCache
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        mm = self._mmap
        offsets = self._cell_offsets
        if mm is not None:
            return self._read_cells_mmap(mm, offsets, cell_ids)

        out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        gcache = get_global_cell_cache()
        db_needed: List[int] = []
        for raw in cell_ids:
            cid = int(raw)
            if cid in out:
                continue
            entry = self._lookup_cached_cell(cid, cache, gcache)
            if entry is not None:
                out[cid] = entry
            else:
                db_needed.append(cid)
        if db_needed:
            conn = self._conn_factory()
            with conn.cursor() as cur:
                for cid, ids, vecs in self._iter_db_cells(cur, db_needed):
                    cache.add_cell(cid, ids, vecs)
                    gcache.put_cell(self._index_name, cid, ids, vecs)
                    out[cid] = (ids, vecs)
        return out

    def _prep_query(self, q: np.ndarray) -> np.ndarray:
        return quant.prepare_query(q, self._storage_dtype, self._metric)

    def _cell_distances(self, qp: np.ndarray, vecs: np.ndarray) -> np.ndarray:
        return quant.cell_distances(self._metric, self._storage_dtype, qp, vecs, self._normalized)

    def _distances(self, q: np.ndarray, vecs: np.ndarray) -> np.ndarray:
        return self._cell_distances(self._prep_query(q), vecs)

    def _distances_over_cells(self, q: np.ndarray, vecs_list: List[np.ndarray]) -> List[np.ndarray]:
        n_cells = len(vecs_list)
        qp = self._prep_query(q)
        if n_cells <= 1:
            return [self._cell_distances(qp, v) for v in vecs_list]
        total = 0
        for v in vecs_list:
            total += int(v.shape[0])
        pool = None
        if total >= config.IVF_QUERY_PARALLEL_MIN_VECTORS and not _in_query_pool_thread():
            pool = _query_thread_pool()
        if pool is None:
            return [self._cell_distances(qp, v) for v in vecs_list]

        n_groups = min(_query_worker_count(), n_cells)
        bounds = [(i * n_cells) // n_groups for i in range(n_groups)] + [n_cells]

        def _score_slice(lo: int, hi: int) -> List[np.ndarray]:
            return [self._cell_distances(qp, vecs_list[j]) for j in range(lo, hi)]

        try:
            futures = [pool.submit(_score_slice, bounds[i], bounds[i + 1]) for i in range(n_groups)]
            out: List[np.ndarray] = []
            for f in futures:
                out.extend(f.result())
            return out
        except Exception as e:
            logger.warning(
                "IVF '%s' parallel cell scan failed (%s); using serial scan.", self._index_name, e
            )
            return [self._cell_distances(qp, v) for v in vecs_list]

    def query(self, vector, k: int):
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        order = self._rank_cells(q)
        cache = self._cache()
        probe = order[: max(1, self._nprobe)]
        cells = self._read_cells(probe, cache)
        allowed = self._availability_mask()
        cand_ids: List[np.ndarray] = []
        cand_vecs: List[np.ndarray] = []
        for cell_id in probe:
            cell = cells.get(cell_id)
            if cell is None:
                continue
            ids, vecs = cell
            if ids.shape[0] == 0:
                continue
            if allowed is not None:
                keep = allowed[ids]
                if not keep.any():
                    continue
                ids = ids[keep]
                vecs = vecs[keep]
            cand_ids.append(ids)
            cand_vecs.append(vecs)
        if not cand_ids:
            return [], []
        cand_dist = self._distances_over_cells(q, cand_vecs)
        all_ids = np.concatenate(cand_ids)
        all_dist = np.concatenate(cand_dist)
        kk = min(int(k), all_dist.shape[0])
        if kk <= 0:
            return [], []
        part = np.argpartition(all_dist, kk - 1)[:kk]
        part = part[np.argsort(all_dist[part])]
        return all_ids[part].astype(np.int64).tolist(), all_dist[part].astype(float).tolist()

    def distance_to_similarity(self, distance: float) -> float:
        d = float(distance)
        if self._metric == "euclidean":
            return 1.0 / (1.0 + d)
        if self._metric == "dot":
            return -d
        return 1.0 - d

    def get_vectors(self, int_ids) -> Dict[int, np.ndarray]:
        cache = self._cache()
        out: Dict[int, np.ndarray] = {}
        need_cells: Dict[int, List[int]] = {}
        for raw in int_ids:
            vid = int(raw)
            if vid < 0 or vid >= self._n_items:
                continue
            cell_id = int(self._id2cell[vid])
            entry = cache.get_cell(cell_id)
            v = _vec_in_cell(entry[0], entry[1], vid) if entry is not None else None
            if v is not None:
                out[vid] = quant.decode_row(v, self._storage_dtype)
            else:
                need_cells.setdefault(cell_id, []).append(vid)
        if need_cells:
            cells = self._read_cells(list(need_cells.keys()), cache)
            for cell_id, vids in need_cells.items():
                cell = cells.get(cell_id)
                if cell is None:
                    continue
                ids, vecs = cell
                for vid in vids:
                    v = _vec_in_cell(ids, vecs, vid)
                    if v is not None:
                        out[vid] = quant.decode_row(v, self._storage_dtype)
        return out

    def get_vector(self, int_id):
        return self.get_vectors([int(int_id)]).get(int(int_id))

    def cell_groups(self, int_ids):
        counts: Dict[int, int] = {}
        for raw in int_ids:
            vid = int(raw)
            if 0 <= vid < self._n_items:
                cell_id = int(self._id2cell[vid])
                counts[cell_id] = counts.get(cell_id, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (np.array(self._centroids[cell_id], dtype=np.float32), count)
            for cell_id, count in ordered
            if 0 <= cell_id < self._num_cells
        ]

    def _max_distance_cell_ids(self, q: np.ndarray, k: int) -> List[int]:
        if k <= 0 or k >= self._num_cells:
            return [int(c) for c in np.unique(self._id2cell)]
        return [int(c) for c in self._farthest_cells(q, k)]

    def _scan_cells_mmap(self, mm, cell_ids: List[int], consume) -> None:
        _note_mmap_activity(self)
        offsets = self._cell_offsets
        for cid in cell_ids:
            cell = self._cell_from_mmap(mm, offsets, cid)
            if cell is not None:
                consume(cell[0], cell[1])

    def _scan_cells_db(self, cell_ids: List[int], consume) -> None:
        gcache = get_global_cell_cache()
        db_needed: List[int] = []
        for cid in cell_ids:
            entry = gcache.get_cell(self._index_name, cid)
            if entry is not None:
                consume(entry[0], entry[1])
            else:
                db_needed.append(cid)
        if not db_needed:
            return
        conn = self._conn_factory()
        with conn.cursor() as cur:
            for start in range(0, len(db_needed), self._read_batch):
                chunk = db_needed[start : start + self._read_batch]
                for _cid, ids, vecs in self._iter_db_cells(cur, chunk):
                    consume(ids, vecs)

    def get_max_distance(
        self, int_id, nprobe: Optional[int] = None
    ) -> Tuple[Optional[float], Optional[int]]:
        anchor = self.get_vector(int_id)
        if anchor is None:
            return None, None
        q = np.asarray(anchor, dtype=np.float32).reshape(-1)
        k = config.IVF_MAX_DISTANCE_NPROBE if nprobe is None else int(nprobe)
        cell_ids = self._max_distance_cell_ids(q, k)
        state = {"max_d": float("-inf"), "far_id": None}
        qp = self._prep_query(q)
        allowed = self._availability_mask()

        def _consume(ids: np.ndarray, vecs: np.ndarray) -> None:
            if vecs.shape[0] == 0:
                return
            if allowed is not None:
                keep = allowed[ids]
                if not keep.any():
                    return
                ids = ids[keep]
                vecs = vecs[keep]
            dists = self._cell_distances(qp, vecs)
            mask = ids != int(int_id)
            if not mask.any():
                return
            masked = np.where(mask, dists, -np.inf)
            midx = int(np.argmax(masked))
            cell_max = float(masked[midx])
            if cell_max > state["max_d"]:
                state["max_d"] = cell_max
                state["far_id"] = int(ids[midx])

        mm = self._mmap
        if mm is not None:
            self._scan_cells_mmap(mm, cell_ids, _consume)
        else:
            self._scan_cells_db(cell_ids, _consume)
        if state["max_d"] == float("-inf"):
            return 0.0, None
        return state["max_d"], state["far_id"]

    def preload_all(self, db_conn=None) -> int:
        if self._mmap is not None:
            return 0
        gcache = get_global_cell_cache()
        if not gcache.enabled:
            return 0
        conn = db_conn if db_conn is not None else self._conn_factory()
        cell_ids = [int(c) for c in np.unique(self._id2cell)]
        loaded = 0
        with conn.cursor() as cur:
            for start in range(0, len(cell_ids), self._read_batch):
                chunk = cell_ids[start : start + self._read_batch]
                for cid, ids, vecs in self._iter_db_cells(cur, chunk):
                    gcache.put_cell(self._index_name, cid, ids, vecs)
                    loaded += 1
        return loaded


def _split_cells_over_cap(
    centroids: np.ndarray,
    id2cell: np.ndarray,
    cells: List[Tuple[int, np.ndarray, np.ndarray]],
    dim: int,
    cap_bytes: int,
    elem_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray, np.ndarray]]]:
    record_size = 4 + dim * elem_size
    cap_records = max(1, cap_bytes // record_size)
    if all(int(ids.shape[0]) <= cap_records for _cid, ids, _vecs in cells):
        return centroids, id2cell, cells

    id2cell = np.array(id2cell, dtype=np.uint32, copy=True)
    next_cell_id = int(centroids.shape[0])
    extra_centroids: List[np.ndarray] = []
    out_cells: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for cell_id, ids, vecs in cells:
        n = int(ids.shape[0])
        if n <= cap_records:
            out_cells.append((cell_id, ids, vecs))
            continue
        for start in range(0, n, cap_records):
            chunk_ids = ids[start : start + cap_records]
            chunk_vecs = vecs[start : start + cap_records]
            if start == 0:
                out_cells.append((cell_id, chunk_ids, chunk_vecs))
            else:
                out_cells.append((next_cell_id, chunk_ids, chunk_vecs))
                extra_centroids.append(chunk_vecs.mean(axis=0).astype(np.float32))
                id2cell[chunk_ids] = next_cell_id
                next_cell_id += 1
    if extra_centroids:
        centroids = np.ascontiguousarray(np.vstack([centroids] + extra_centroids), dtype=np.float32)
    return centroids, id2cell, out_cells


def store_paged_ivf(
    db_conn,
    index_name: str,
    centroids: np.ndarray,
    id2cell: np.ndarray,
    item_ids: List[str],
    cells: List[Tuple[int, np.ndarray, np.ndarray]],
    dim: int,
    metric: str,
    max_part_size_mb: Optional[int] = None,
    normalized: bool = False,
    storage_dtype: int = 0,
) -> None:
    from .index_build_helpers import store_segmented_blob

    part_mb = config.IVF_MAX_PART_SIZE_MB if max_part_size_mb is None else int(max_part_size_mb)
    part_bytes = part_mb * 1024 * 1024

    esize = quant.elem_size(storage_dtype)
    centroids, id2cell, cells = _split_cells_over_cap(
        centroids, id2cell, cells, dim, part_bytes, esize
    )
    dir_blob = pack_directory(
        centroids,
        id2cell,
        item_ids,
        dim,
        metric,
        normalized=normalized,
        storage_dtype=storage_dtype,
    )
    with db_conn.cursor() as cur:
        cur.execute(f"DELETE FROM {IVF_CELL_TABLE} WHERE index_name = %s", (index_name,))
        for cell_id, ids, vecs in cells:
            if ids.shape[0] == 0:
                continue
            packed = pack_cell(ids, vecs, storage_dtype)
            cur.execute(
                f"INSERT INTO {IVF_CELL_TABLE} (index_name, cell_id, cell_data) VALUES (%s, %s, %s)",
                (index_name, int(cell_id), psycopg2.Binary(packed)),
            )
    store_segmented_blob(
        db_conn, IVF_DIR_TABLE, f"{index_name}__ivf_dir", dir_blob, max_part_size_mb=part_mb
    )
    invalidate_global_cell_cache(index_name)


def _bounded_cell_groups(
    members: np.ndarray,
    member_vecs: np.ndarray,
    base_centroid: np.ndarray,
    max_records: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.cluster import MiniBatchKMeans

    if members.shape[0] <= max_records:
        return [(members, base_centroid)]

    n_sub = int(np.ceil(members.shape[0] / max_records))
    sub = MiniBatchKMeans(n_clusters=n_sub, batch_size=10000, n_init=1, max_iter=15, random_state=0)
    sub_labels = sub.fit_predict(member_vecs)
    groups: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in range(n_sub):
        mask = sub_labels == s
        grp = members[mask]
        if grp.shape[0] == 0:
            continue
        if grp.shape[0] <= max_records:
            groups.append((grp, sub.cluster_centers_[s].astype(np.float32)))
            continue
        grp_vecs = member_vecs[mask]
        for start in range(0, grp.shape[0], max_records):
            chunk = grp[start : start + max_records]
            chunk_centroid = grp_vecs[start : start + max_records].mean(axis=0).astype(np.float32)
            groups.append((chunk, chunk_centroid))
    return groups


def build_and_store_paged_ivf(
    db_conn,
    index_name: str,
    vectors: np.ndarray,
    item_ids: List[str],
    dim: int,
    metric: str,
    consume_vectors: bool = False,
) -> bool:
    from sklearn.cluster import MiniBatchKMeans

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n_items = vectors.shape[0]
    if n_items == 0 or len(item_ids) != n_items:
        logger.warning(
            "IVF build '%s': empty or mismatched input (n=%d ids=%d).",
            index_name,
            n_items,
            len(item_ids),
        )
        return False
    if vectors.shape[1] != dim:
        raise ValueError(f"IVF build '{index_name}': matrix dim {vectors.shape[1]} != {dim}")

    metric = (metric or "angular").lower()
    normalized = metric == "angular"
    storage_dtype = quant.effective_code(quant.dtype_code(config.IVF_STORAGE_DTYPE), metric)
    if normalized:
        train_mat = _normalize_rows(
            vectors if consume_vectors else vectors.copy(), inplace=True
        )
    else:
        train_mat = vectors

    base_nlist = int(round(8.0 * np.sqrt(max(1, n_items))))
    nlist = max(1, min(config.IVF_NLIST_MAX, base_nlist, n_items))

    sample_n = min(
        n_items,
        config.IVF_TRAIN_POINTS_PER_CELL * nlist,
        max(1, config.IVF_TRAIN_SAMPLE_MAX),
    )
    if sample_n < n_items:
        sample_keys = np.fromiter(
            (
                int.from_bytes(
                    hashlib.blake2b(
                        str(iid).encode("utf-8"), digest_size=8, usedforsecurity=False
                    ).digest(),
                    "big",
                )
                for iid in item_ids
            ),
            dtype=np.uint64,
            count=n_items,
        )
        sample_idx = np.sort(np.argpartition(sample_keys, sample_n - 1)[:sample_n])
        sample = train_mat[sample_idx]
    else:
        sample = train_mat

    logger.info(
        "IVF build '%s': training %d cells on %d sampled vectors (N=%d, dim=%d).",
        index_name,
        nlist,
        sample_n,
        n_items,
        dim,
    )
    km = MiniBatchKMeans(n_clusters=nlist, batch_size=10000, n_init=1, max_iter=25, random_state=0)
    km.fit(sample)
    centroids = km.cluster_centers_.astype(np.float32)
    del sample

    labels = np.empty(n_items, dtype=np.int64)
    for start in range(0, n_items, 20000):
        labels[start : start + 20000] = km.predict(train_mat[start : start + 20000])

    max_cell_bytes = min(config.IVF_MAX_CELL_MB, config.IVF_MAX_PART_SIZE_MB) * 1024 * 1024
    max_cell_records = max(1, max_cell_bytes // (4 + dim * quant.elem_size(storage_dtype)))
    int_ids = np.arange(n_items, dtype=np.int32)

    id2cell = np.empty(n_items, dtype=np.uint32)
    centroid_list: List[np.ndarray] = [centroids[c] for c in range(nlist)]
    next_cell_id = nlist
    with db_conn.cursor() as cur:
        cur.execute(f"DELETE FROM {IVF_CELL_TABLE} WHERE index_name = %s", (index_name,))
        for c in range(nlist):
            members = np.where(labels == c)[0]
            if members.shape[0] == 0:
                continue
            reused_c = False
            member_vecs = train_mat[members]
            for grp, centroid in _bounded_cell_groups(
                members, member_vecs, centroids[c], max_cell_records
            ):
                if not reused_c:
                    assigned_cell = c
                    centroid_list[c] = centroid
                    reused_c = True
                else:
                    assigned_cell = next_cell_id
                    centroid_list.append(centroid)
                    next_cell_id += 1
                cell_ids = int_ids[grp]
                cell_vecs = train_mat[grp]
                id2cell[grp] = assigned_cell
                packed = pack_cell(cell_ids, cell_vecs, storage_dtype)
                cur.execute(
                    f"INSERT INTO {IVF_CELL_TABLE} "
                    "(index_name, cell_id, cell_data) VALUES (%s, %s, %s)",
                    (index_name, int(assigned_cell), psycopg2.Binary(packed)),
                )
                del cell_ids, cell_vecs, packed
            del member_vecs

    final_centroids = np.ascontiguousarray(np.vstack(centroid_list), dtype=np.float32)
    logger.info(
        "IVF build '%s': %d cells after splitting (max_cell_records=%d, storage=%s).",
        index_name,
        len(centroid_list),
        max_cell_records,
        quant.dtype_name(storage_dtype),
    )
    from .index_build_helpers import store_segmented_blob

    dir_blob = pack_directory(
        final_centroids, id2cell, list(item_ids), dim, metric,
        normalized=normalized, storage_dtype=storage_dtype,
    )
    store_segmented_blob(
        db_conn, IVF_DIR_TABLE, f"{index_name}__ivf_dir", dir_blob,
        max_part_size_mb=config.IVF_MAX_PART_SIZE_MB,
    )
    invalidate_global_cell_cache(index_name)
    return True


def has_paged_ivf(db_conn, index_name: str) -> bool:
    from .index_build_helpers import load_segmented_blob

    try:
        blob = load_segmented_blob(db_conn, IVF_DIR_TABLE, f"{index_name}__ivf_dir")
        return blob is not None and len(blob) >= _HEADER_SIZE
    except Exception:
        return False


def _setup_disk_cell_file(
    db_conn, index_name: str, dim: int, metric: str, storage_dtype: int, dir_blob: bytes, label: str
):
    if not config.IVF_DISK_CACHE_ENABLED:
        return None, None
    try:
        cache_dir = config.IVF_DISK_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        build_id = hashlib.sha1(dir_blob, usedforsecurity=False).hexdigest()[:16]
        path = _cell_file_path(cache_dir, index_name, build_id)
        with _IVF_FILE_SWAP_LOCK:
            if not os.path.exists(path):
                n = _export_cells_to_file(db_conn, index_name, dim, metric, storage_dtype, path)
                logger.info("IVF index '%s' exported %d cells to %s.", label, n, path)
            _close_live_indexes(index_name)
            _prune_old_cell_files(cache_dir, index_name, path)
        mm, _dim, offsets, _file_dtype = _open_cell_file(path)
        return mm, offsets
    except Exception as e:
        logger.warning(
            "IVF index '%s' disk cell cache unavailable (%s); reading cells from Postgres.",
            label,
            e,
        )
        return None, None


def load_paged_ivf_index(
    db_conn,
    index_name: str,
    expected_dim: Optional[int],
    metric: str,
    conn_factory: Optional[Callable[[], "psycopg2.extensions.connection"]] = None,
    label: Optional[str] = None,
    track_scoped: bool = True,
):
    from .index_build_helpers import load_segmented_blob

    label = label or index_name
    invalidate_global_cell_cache(index_name)
    blob = load_segmented_blob(db_conn, IVF_DIR_TABLE, f"{index_name}__ivf_dir")
    if not blob:
        return None
    centroids, id2cell, item_ids, dim, stored_metric, normalized, storage_dtype = unpack_directory(
        bytes(blob)
    )
    if expected_dim is not None and dim != expected_dim:
        logger.error("IVF '%s': dimension mismatch db=%s expected=%s", label, dim, expected_dim)
        return None

    if stored_metric and metric and str(stored_metric).lower() != str(metric).lower():
        logger.warning(
            "IVF index '%s' stored with metric '%s' but config now uses '%s'; treating as not built so it rebuilds.",
            label,
            stored_metric,
            metric,
        )
        return None

    expected_storage_dtype = quant.effective_code(
        quant.dtype_code(config.IVF_STORAGE_DTYPE), stored_metric
    )
    if int(storage_dtype) != int(expected_storage_dtype):
        logger.warning(
            "IVF index '%s' stored as %s but config now builds %s; treating as not built so it rebuilds.",
            label,
            quant.dtype_name(storage_dtype),
            quant.dtype_name(expected_storage_dtype),
        )
        return None

    if conn_factory is None:
        from app_helper import get_db

        conn_factory = get_db

    mmap_obj, cell_offsets = _setup_disk_cell_file(
        db_conn, index_name, dim, stored_metric or metric, storage_dtype, bytes(blob), label
    )

    index = PagedIvfIndex(
        centroids=centroids,
        id2cell=id2cell,
        item_ids=item_ids,
        dim=dim,
        metric=stored_metric or metric,
        index_name=index_name,
        conn_factory=conn_factory,
        normalized=normalized,
        storage_dtype=storage_dtype,
        mmap_obj=mmap_obj,
        cell_offsets=cell_offsets,
        track_scoped=track_scoped,
    )
    id_map = {i: item_id for i, item_id in enumerate(item_ids)}
    reverse_id_map = {item_id: i for i, item_id in id_map.items()}
    logger.info(
        "IVF index '%s' loaded: %d items, %d cells, dim=%d, normalized=%s, storage=%s, disk_mmap=%s.",
        label,
        len(item_ids),
        centroids.shape[0],
        dim,
        normalized,
        quant.dtype_name(storage_dtype),
        mmap_obj is not None,
    )
    if storage_dtype != quant.DTYPE_F32 and not quant.HAVE_NUMKONG:
        _warn_numkong_missing_once(quant.dtype_name(storage_dtype))
    if config.IVF_PRELOAD_ALL:
        try:
            loaded = index.preload_all(db_conn)
            logger.info("IVF index '%s' preloaded %d cells into the global cache.", label, loaded)
        except Exception as e:
            logger.warning("IVF index '%s' preload failed (continuing lazily): %s", label, e)
    return index, id_map, reverse_id_map


def load_index_auto(
    db_conn,
    index_name: str,
    expected_dim: Optional[int],
    metric: str,
    label: Optional[str] = None,
):
    return load_paged_ivf_index(db_conn, index_name, expected_dim, metric, label=label)
