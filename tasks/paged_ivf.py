"""
Disk-paged inverted-file (IVF) vector index.

This backend keeps full-precision float32 vectors out of the Flask container's
resident heap. Vectors are partitioned by a coarse k-means quantizer into
``nlist`` cells. Postgres (``ivf_cell``, one row per cell) is the source of
truth, written at build time. The only always-resident state is the directory
(centroids + id maps), tiny relative to the vectors.

PRIMARY read path -- local mmap (default, IVF_DISK_CACHE_ENABLED): at index load
each index's cells are exported once from Postgres into a single local file
under IVF_DISK_CACHE_DIR (``<index>.<dirhash>.amivf``), and queries read cells
zero-copy from that file via ``np.memmap``. Residency is owned by the OS page
cache (file-backed, reclaimable, does not grow the Flask heap), and Postgres is
never touched on the query hot path. The filename embeds a hash of the directory
blob, so an unchanged index reuses its file across restarts and a rebuild writes
a new versioned file (safe to swap even on Windows, where a mapped file cannot be
overwritten). If the cache dir is unwritable / disabled, it falls back to the
Postgres read path below.

FALLBACK read path -- Postgres + caches: a query ranks the in-RAM centroids,
reads the nearest ``nprobe`` cells from Postgres, and caches decoded cells in a
per-request ``_CellLruCache`` (L1, thread-local) in front of a process-wide
``_GlobalCellCache`` (L2, byte-bounded by IVF_GLOBAL_CACHE_MB, idle-dropped).

No quantization is used: stored vectors are full-precision float32
(unit-normalized when the metric is angular, so queries skip per-call
renormalization). Decoded cells (mmap views or ``unpack_cell`` output) are
read-only, so the same arrays are shared by reference across threads without
copying; callers that mutate a vector copy it first.

Format is self-describing and versioned (directory magic ``AMIV``, cell-file
magic ``AMVF``).

Format is self-describing and versioned (magic ``AMIV``). The index lives in
dedicated tables (``ivf_dir`` for the directory blob, ``ivf_cell`` for cell
rows), separate from the legacy ivf ``*_index_data`` tables, so the two
backends coexist and a deployment falls back to ivf until the next rebuild
writes IVF rows.
"""

from __future__ import annotations

import glob
import hashlib
import io
import logging
import os
import struct
import threading
import time
import weakref
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import psycopg2

import config

logger = logging.getLogger(__name__)

_MAGIC = b"AMIV"
_VERSION = 1
_HEADER_FMT = "<4sIBBxxIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_METRIC_TO_CODE = {"angular": 0, "euclidean": 1, "dot": 2}
_CODE_TO_METRIC = {v: k for k, v in _METRIC_TO_CODE.items()}

IVF_DIR_TABLE = "ivf_dir"
IVF_CELL_TABLE = "ivf_cell"


def _metric_code(metric: str) -> int:
    return _METRIC_TO_CODE.get((metric or "angular").lower(), 0)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32, copy=False)


def pack_directory(
    centroids: np.ndarray,
    id2cell: np.ndarray,
    item_ids: List[str],
    dim: int,
    metric: str,
    normalized: bool = False,
) -> bytes:
    """Serialize the resident directory (centroids + id maps) into one blob."""
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    id2cell = np.ascontiguousarray(id2cell, dtype=np.uint32)
    nlist = centroids.shape[0]
    n_items = len(item_ids)
    if id2cell.shape[0] != n_items:
        raise ValueError(f"id2cell length {id2cell.shape[0]} != n_items {n_items}")
    if centroids.shape[1] != dim:
        raise ValueError(f"centroid dim {centroids.shape[1]} != dim {dim}")

    buf = io.BytesIO()
    buf.write(struct.pack(_HEADER_FMT, _MAGIC, _VERSION, _metric_code(metric), 1 if normalized else 0, dim, nlist, n_items))
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


def unpack_directory(blob: bytes) -> Tuple[np.ndarray, np.ndarray, List[str], int, str, bool]:
    """Inverse of :func:`pack_directory`.

    Returns ``(centroids, id2cell, item_ids, dim, metric, normalized)``. Blobs
    written before the normalized flag existed carry a zero in that byte and so
    report ``normalized=False``, keeping old indexes correct.
    """
    if len(blob) < _HEADER_SIZE:
        raise ValueError(f"directory blob too short ({len(blob)} bytes)")
    magic, version, metric_code, normalized, dim, nlist, n_items = struct.unpack_from(_HEADER_FMT, blob, 0)
    if magic != _MAGIC:
        raise ValueError(f"directory magic mismatch: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported directory version: {version}")
    pos = _HEADER_SIZE
    cent_count = nlist * dim
    centroids = np.frombuffer(blob, dtype=np.float32, count=cent_count, offset=pos).reshape(nlist, dim)
    pos += cent_count * 4
    id2cell = np.frombuffer(blob, dtype=np.uint32, count=n_items, offset=pos).copy()
    pos += n_items * 4
    item_ids: List[str] = []
    for _ in range(n_items):
        (slen,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        item_ids.append(blob[pos:pos + slen].decode("utf-8"))
        pos += slen
    return centroids.copy(), id2cell, item_ids, int(dim), _CODE_TO_METRIC.get(metric_code, "angular"), bool(normalized)


def pack_cell(int_ids: np.ndarray, vecs: np.ndarray) -> bytes:
    """Pack a cell as ``[n int32 ids][n*dim float32 vectors]``."""
    int_ids = np.ascontiguousarray(int_ids, dtype=np.int32)
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    return int_ids.tobytes() + vecs.tobytes()


def unpack_cell(blob: bytes, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_cell`."""
    record = 4 + dim * 4
    if record <= 0 or len(blob) % record != 0:
        raise ValueError(f"cell blob size {len(blob)} not a multiple of record size {record}")
    n = len(blob) // record
    ids = np.frombuffer(blob, dtype=np.int32, count=n, offset=0)
    vecs = np.frombuffer(blob, dtype=np.float32, count=n * dim, offset=n * 4).reshape(n, dim)
    return ids, vecs


_CELLFILE_MAGIC = b"AMVF"
_CELLFILE_VERSION = 1
_CELLFILE_HEADER_FMT = "<4sIIII"
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
            except OSError:
                pass


def _export_cells_to_file(db_conn, index_name: str, dim: int, metric: str, path: str) -> int:
    """Stream every cell of ``index_name`` from Postgres into a local mmap file.

    Two passes keep RAM bounded: pass 1 reads only ``octet_length`` to lay out the
    offset table; pass 2 streams the blobs in cell_id order in small chunks. Writes
    to ``path + '.tmp'`` then atomically renames. Returns the number of cells.
    """
    record = 4 + int(dim) * 4
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
        f.write(struct.pack(_CELLFILE_HEADER_FMT, _CELLFILE_MAGIC, _CELLFILE_VERSION, int(dim), n_cells, _metric_code(metric)))
        for cid, off, ln in table:
            f.write(struct.pack(_CELLFILE_ROW_FMT, cid, off, ln))
        chunk = 64
        with db_conn.cursor() as cur:
            for start in range(0, n_cells, chunk):
                ids_chunk = order[start:start + chunk]
                cur.execute(
                    f"SELECT cell_id, cell_data FROM {IVF_CELL_TABLE} "
                    f"WHERE index_name = %s AND cell_id = ANY(%s)",
                    (index_name, ids_chunk),
                )
                blobs = {int(c): bytes(b) for c, b in cur.fetchall()}
                for cid in ids_chunk:
                    b = blobs.get(cid)
                    if b is None or len(b) != exp_len[cid] or len(b) % record != 0:
                        raise ValueError(f"cell {cid} of '{index_name}' changed or malformed during export")
                    f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return n_cells


def _open_cell_file(path: str):
    """Open a cell file as a read-only memmap; return ``(mmap, dim, {cell_id:(off,len)})``."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    head = bytes(mm[:_CELLFILE_HEADER_SIZE])
    magic, version, dim, n_cells, _metric_code_v = struct.unpack(_CELLFILE_HEADER_FMT, head)
    if magic != _CELLFILE_MAGIC:
        raise ValueError(f"cell file magic mismatch: {magic!r}")
    if version != _CELLFILE_VERSION:
        raise ValueError(f"unsupported cell file version: {version}")
    table = bytes(mm[_CELLFILE_HEADER_SIZE:_CELLFILE_HEADER_SIZE + n_cells * _CELLFILE_ROW_SIZE])
    offsets = {}
    for i in range(n_cells):
        cid, off, ln = struct.unpack_from(_CELLFILE_ROW_FMT, table, i * _CELLFILE_ROW_SIZE)
        offsets[int(cid)] = (int(off), int(ln))
    return mm, int(dim), offsets


def _vec_in_cell(ids: np.ndarray, vecs: np.ndarray, int_id: int) -> Optional[np.ndarray]:
    """Return the row of ``vecs`` whose id is ``int_id`` (cell ids are sorted), or None."""
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
    """Byte-bounded LRU cache of decoded cells for a single request.

    Holds ``cell_id -> (ids, vecs)``. ``add_cell`` evicts least-recently-used
    cells before inserting, so ``resident_bytes() <= max_bytes`` holds whenever
    no single cell exceeds ``max_bytes`` (the index floors the cap to at least
    one full cell). Used on a single thread per request (IVF mode runs vector
    post-filtering on the request thread), so no lock is required.

    Point lookups go through :meth:`get_cell`: the caller already knows an item's
    cell from the index-level ``id2cell`` map, so the cache never has to maintain
    its own ``int_id -> cell_id`` index.
    """

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
    """Process-wide byte-bounded LRU cache of decoded cells (L2).

    Shared by every ``PagedIvfIndex`` in the process and keyed by
    ``(index_name, cell_id)`` so all indexes draw from one bounded pool. A single
    RLock guards the dict ops; it is never held while reading from Postgres, so
    concurrent DB fetches are not serialized. Stored ``(ids, vecs)`` tuples are
    handed out by reference and must stay read-only (see the module docstring).
    Size is accounted with ``ndarray.nbytes`` (correct for 2-D arrays, unlike
    ``sys.getsizeof``). ``max_bytes <= 0`` turns the cache into a no-op.

    When ``idle_seconds > 0`` a background daemon drops the whole cache after that
    many seconds with no access, so an idle process releases the RAM (mirrors the
    CLAP/lyrics model warm-cache timers). Every get/put resets the idle clock; the
    timer thread re-checks under the lock so a fresh access cancels a pending drop.
    Idle-dropping is disabled (``idle_seconds=0``) when ``IVF_PRELOAD_ALL`` is on,
    so a preloaded working set is not silently evicted and left un-rewarmed.
    """

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
        if self._idle_seconds > 0 and (self._timer_thread is None or not self._timer_thread.is_alive()):
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
                logger.info("IVF global cell cache idle for %ds; dropped %d cells to free RAM.", self._idle_seconds, dropped)
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
    """Best-effort: hand freed heap back to the OS so RSS drops after a big free.

    glibc keeps freed allocations in its arenas, so dropping the cache frees the
    Python objects but does not lower RSS until the heap is trimmed. Delegates to
    the shared malloc_trim helper (Linux-only; no-op elsewhere).
    """
    try:
        from .memory_utils import release_memory_to_os
        release_memory_to_os()
    except Exception:
        pass


_GLOBAL_CELL_CACHE: Optional[_GlobalCellCache] = None
_GLOBAL_CELL_CACHE_LOCK = threading.Lock()


def get_global_cell_cache() -> _GlobalCellCache:
    """Return the lazily-built process-wide L2 cell cache (sized from config)."""
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
    """Drop every L2 entry for ``index_name`` (call on rebuild/reload)."""
    cache = _GLOBAL_CELL_CACHE
    if cache is not None:
        cache.invalidate_index(index_name)


def begin_query(index) -> None:
    """Start a fresh per-request L1 cell cache on ``index`` before a query.

    Shared by every query entry point so the per-request reset lives in one
    place; a no-op when ``index`` is None or does not expose ``begin_request``
    (e.g. a test double).
    """
    if index is not None and hasattr(index, "begin_request"):
        index.begin_request()


_LIVE_INDEXES: "weakref.WeakSet[PagedIvfIndex]" = weakref.WeakSet()


def end_all_requests() -> None:
    """Free the per-request L1 cache of every loaded index on the current thread.

    The L1 cache is thread-local, so begin_request() leaves it resident on the
    worker thread until the next request overwrites it. Calling this from a
    per-request teardown drops it immediately so an idle worker holds no L1.
    """
    for idx in list(_LIVE_INDEXES):
        try:
            idx.end_request()
        except Exception:
            pass


class PagedIvfIndex:
    """IVF-compatible query surface backed by Postgres-resident IVF cells.

    Exposes the subset of the ivf API the call sites rely on:
    ``query``, ``get_vector``, ``get_vectors``, ``get_max_distance``,
    ``__len__``, ``num_elements`` and a no-op ``ef`` setter. Cell reads use a
    connection from ``conn_factory`` (the request-scoped Flask DB connection),
    so all methods must be called on the request thread.
    """

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
        mmap_obj=None,
        cell_offsets: Optional[Dict[int, Tuple[int, int]]] = None,
    ):
        self._dim = int(dim)
        self._metric = (metric or "angular").lower()
        self._normalized = bool(normalized)
        self._index_name = index_name
        self._conn_factory = conn_factory
        self._mmap = mmap_obj
        self._cell_offsets = cell_offsets or {}
        self._n_items = len(item_ids)
        self._record_size = 4 + self._dim * 4
        self._id2cell = np.ascontiguousarray(id2cell, dtype=np.uint32)
        self._nprobe = int(nprobe if nprobe is not None else config.IVF_NPROBE)
        _query_cache_bytes = int(
            query_cache_bytes if query_cache_bytes is not None else config.IVF_QUERY_CACHE_MB * 1024 * 1024
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
        self._tl = threading.local()
        _LIVE_INDEXES.add(self)

    def __len__(self) -> int:
        return self._n_items

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
        """Per-centroid score where SMALLER means nearer (larger means farther)."""
        if self._metric == "euclidean":
            diffs = self._centroids - q[None, :]
            return np.einsum("ij,ij->i", diffs, diffs)
        if self._metric == "dot":
            return -(self._centroids @ q)
        qn = q / (np.linalg.norm(q) + 1e-12)
        return -(self._centroids @ qn)

    def _rank_cells(self, q: np.ndarray) -> np.ndarray:
        """Return the nearest ``nprobe`` cell ids, ordered nearest first.

        Uses argpartition to pull the top ``nprobe`` in O(nlist) and sorts only
        those, instead of a full argsort over every centroid. Cell selection and
        order are identical to a full sort, so recall is unchanged.
        """
        scores = self._cell_scores(q)
        n = scores.shape[0]
        topn = max(1, self._nprobe)
        if topn >= n:
            return np.argsort(scores)
        part = np.argpartition(scores, topn - 1)[:topn]
        return part[np.argsort(scores[part])]

    def _farthest_cells(self, q: np.ndarray, k: int) -> np.ndarray:
        """Return the ``k`` cell ids whose centroids are FARTHEST from ``q``."""
        scores = self._cell_scores(q)
        n = scores.shape[0]
        if k >= n:
            return np.arange(n)
        return np.argpartition(scores, n - k)[n - k:]

    def _cell_from_mmap(self, cell_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Zero-copy read of one cell from the memmap, or None if absent."""
        rec = self._cell_offsets.get(int(cell_id))
        if rec is None:
            return None
        off, ln = rec
        sub = self._mmap[off:off + ln]
        n = ln // self._record_size
        if n == 0:
            return None
        ids = sub[:4 * n].view(np.int32)
        vecs = sub[4 * n:].view(np.float32).reshape(n, self._dim)
        return ids, vecs

    def _read_cells(self, cell_ids: List[int], cache: _CellLruCache) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Decode ``cell_ids`` and return ``{cell_id: (ids, vecs)}``.

        When a local mmap cell file is loaded, cells are read zero-copy from it
        (OS page cache manages residency; Postgres is not touched). Otherwise it
        falls back to L1 -> L2 -> Postgres, fetching all DB misses in a single
        round-trip. The returned dict holds direct references, so callers do not
        depend on L1 retaining every cell under its byte cap.
        """
        out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if self._mmap is not None:
            for raw in cell_ids:
                cid = int(raw)
                if cid in out:
                    continue
                cell = self._cell_from_mmap(cid)
                if cell is not None:
                    out[cid] = cell
            return out
        gcache = get_global_cell_cache()
        db_needed: List[int] = []
        for raw in cell_ids:
            cid = int(raw)
            if cid in out:
                continue
            entry = cache.get_cell(cid)
            if entry is None:
                entry = gcache.get_cell(self._index_name, cid)
                if entry is not None:
                    cache.add_cell(cid, entry[0], entry[1])
            if entry is not None:
                out[cid] = entry
            else:
                db_needed.append(cid)
        if db_needed:
            conn = self._conn_factory()
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT cell_id, cell_data FROM {IVF_CELL_TABLE} "
                    f"WHERE index_name = %s AND cell_id = ANY(%s)",
                    (self._index_name, db_needed),
                )
                for cell_id, blob in cur.fetchall():
                    cid = int(cell_id)
                    ids, vecs = unpack_cell(bytes(blob), self._dim)
                    cache.add_cell(cid, ids, vecs)
                    gcache.put_cell(self._index_name, cid, ids, vecs)
                    out[cid] = (ids, vecs)
        return out

    def _distances(self, q: np.ndarray, vecs: np.ndarray) -> np.ndarray:
        if self._metric == "euclidean":
            diffs = vecs - q[None, :]
            return np.sqrt(np.einsum("ij,ij->i", diffs, diffs)).astype(np.float32)
        if self._metric == "dot":
            return (-(vecs @ q)).astype(np.float32)
        qn = q / (np.linalg.norm(q) + 1e-12)
        if self._normalized:
            return (1.0 - np.clip(vecs @ qn, -1.0, 1.0)).astype(np.float32)
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        return (1.0 - np.clip(vn @ qn, -1.0, 1.0)).astype(np.float32)

    def query(self, vector, k: int):
        """Return ``(ids, distances)`` for the ``k`` nearest items."""
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        order = self._rank_cells(q)
        cache = self._cache()
        probe = [int(c) for c in order[:max(1, self._nprobe)]]
        cells = self._read_cells(probe, cache)
        cand_ids: List[np.ndarray] = []
        cand_dist: List[np.ndarray] = []
        for cell_id in probe:
            cell = cells.get(cell_id)
            if cell is None:
                continue
            ids, vecs = cell
            if ids.shape[0] == 0:
                continue
            cand_ids.append(ids)
            cand_dist.append(self._distances(q, vecs))
        if not cand_ids:
            return [], []
        all_ids = np.concatenate(cand_ids)
        all_dist = np.concatenate(cand_dist)
        kk = min(int(k), all_dist.shape[0])
        if kk <= 0:
            return [], []
        part = np.argpartition(all_dist, kk - 1)[:kk]
        part = part[np.argsort(all_dist[part])]
        return all_ids[part].astype(np.int64).tolist(), all_dist[part].astype(float).tolist()

    def distance_to_similarity(self, distance: float) -> float:
        """Map a query distance to a higher-is-more-similar score for this metric.

        Angular distance is ``1 - cosine`` so similarity is ``1 - distance``;
        euclidean uses ``1 / (1 + distance)``; dot stores ``-(v . q)`` so
        similarity is ``-distance``. Keeping this on the index means callers stay
        correct when ``IVF_METRIC`` is not angular instead of hardcoding 1 - d.
        """
        d = float(distance)
        if self._metric == "euclidean":
            return 1.0 / (1.0 + d)
        if self._metric == "dot":
            return -d
        return 1.0 - d

    def get_vectors(self, int_ids) -> Dict[int, np.ndarray]:
        """Return ``{int_id: vector}`` for the given ids, reading missing cells."""
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
                out[vid] = np.asarray(v, dtype=np.float32)
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
                        out[vid] = np.asarray(v, dtype=np.float32)
        return out

    def get_vector(self, int_id):
        """Return the stored vector for ``int_id`` or None."""
        return self.get_vectors([int(int_id)]).get(int(int_id))

    def get_max_distance(self, int_id, nprobe: Optional[int] = None) -> Tuple[Optional[float], Optional[int]]:
        """Maximum distance from ``int_id`` to any other item.

        Returns ``(max_distance, farthest_int_id)``. By default this is an
        APPROXIMATE farthest-neighbor: it ranks centroids and scans only the
        ``IVF_MAX_DISTANCE_NPROBE`` cells whose centroids are farthest from the
        anchor (the farthest point almost always lives in one of them), which is
        far cheaper than a full scan and is intended for the UI "max distance"
        reference value. Pass ``nprobe=0`` (or ``IVF_MAX_DISTANCE_NPROBE=0``, or
        any value >= the cell count) for the EXACT full scan.

        Cells are read zero-copy from the local mmap when present; otherwise from
        the L2 cache / Postgres in ``read_batch`` chunks (that fallback scan does
        NOT populate L2, to avoid evicting the hot working set with cold cells).
        ``(None, None)`` if ``int_id`` is unknown; ``(0.0, None)`` for a
        single-item index.
        """
        anchor = self.get_vector(int_id)
        if anchor is None:
            return None, None
        q = np.asarray(anchor, dtype=np.float32).reshape(-1)
        k = config.IVF_MAX_DISTANCE_NPROBE if nprobe is None else int(nprobe)
        if k <= 0 or k >= self._num_cells:
            cell_ids = [int(c) for c in np.unique(self._id2cell)]
        else:
            cell_ids = [int(c) for c in self._farthest_cells(q, k)]
        state = {"max_d": float("-inf"), "far_id": None}

        def _consume(ids: np.ndarray, vecs: np.ndarray) -> None:
            if vecs.shape[0] == 0:
                return
            dists = self._distances(q, vecs)
            mask = ids != int(int_id)
            if not mask.any():
                return
            masked = np.where(mask, dists, -np.inf)
            midx = int(np.argmax(masked))
            cell_max = float(masked[midx])
            if cell_max > state["max_d"]:
                state["max_d"] = cell_max
                state["far_id"] = int(ids[midx])

        if self._mmap is not None:
            for cid in cell_ids:
                cell = self._cell_from_mmap(cid)
                if cell is not None:
                    _consume(cell[0], cell[1])
        else:
            gcache = get_global_cell_cache()
            db_needed: List[int] = []
            for cid in cell_ids:
                entry = gcache.get_cell(self._index_name, cid)
                if entry is not None:
                    _consume(entry[0], entry[1])
                else:
                    db_needed.append(cid)
            if db_needed:
                conn = self._conn_factory()
                with conn.cursor() as cur:
                    for start in range(0, len(db_needed), self._read_batch):
                        chunk = db_needed[start:start + self._read_batch]
                        cur.execute(
                            f"SELECT cell_data FROM {IVF_CELL_TABLE} "
                            f"WHERE index_name = %s AND cell_id = ANY(%s)",
                            (self._index_name, chunk),
                        )
                        for (blob,) in cur.fetchall():
                            ids, vecs = unpack_cell(bytes(blob), self._dim)
                            _consume(ids, vecs)
        if state["max_d"] == float("-inf"):
            return 0.0, None
        return state["max_d"], state["far_id"]

    def preload_all(self, db_conn=None) -> int:
        """Stream every cell into the global L2 cache (opt-in in-memory mode).

        No-op when a local mmap is active (the OS page cache already owns
        residency) or when the global cache is disabled. Otherwise reads in
        ``read_batch`` chunks bounded by the L2 byte cap.
        """
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
                chunk = cell_ids[start:start + self._read_batch]
                cur.execute(
                    f"SELECT cell_id, cell_data FROM {IVF_CELL_TABLE} "
                    f"WHERE index_name = %s AND cell_id = ANY(%s)",
                    (self._index_name, chunk),
                )
                for cell_id, blob in cur.fetchall():
                    ids, vecs = unpack_cell(bytes(blob), self._dim)
                    gcache.put_cell(self._index_name, int(cell_id), ids, vecs)
                    loaded += 1
        return loaded


def _split_cells_over_cap(
    centroids: np.ndarray,
    id2cell: np.ndarray,
    cells: List[Tuple[int, np.ndarray, np.ndarray]],
    dim: int,
    cap_bytes: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray, np.ndarray]]]:
    """Guarantee every cell packs to at most ``cap_bytes`` by splitting oversized ones.

    A cell with more than ``cap_bytes // record_size`` records is sliced by id into
    fixed-size chunks: the first chunk keeps the original cell id and centroid, each
    extra chunk becomes a new cell with its own id and a centroid equal to the chunk
    mean, and ``id2cell`` is rewritten for every moved record. Returns the (possibly
    extended) ``(centroids, id2cell, cells)``. This is the structural backstop that
    makes it impossible to write a cell BYTEA over the cap, no matter what the build
    produced -- the build's smaller per-cell target normally keeps it from firing.
    A cell packs to exactly ``record_size`` bytes per record (no header), so a chunk
    of ``cap_bytes // record_size`` records is always at most ``cap_bytes``.
    """
    record_size = 4 + dim * 4
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
            chunk_ids = ids[start:start + cap_records]
            chunk_vecs = vecs[start:start + cap_records]
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
) -> None:
    """Persist a built IVF index: directory blob in ``ivf_dir``, cells in ``ivf_cell``.

    Replaces any existing rows for ``index_name`` in both tables in the caller's
    transaction. Every stored BYTEA value -- each cell row and each directory
    part -- is guaranteed to be at most ``IVF_MAX_PART_SIZE_MB``: the directory
    blob is segmented across part rows, and any cell over the cap is split into
    additional cells by :func:`_split_cells_over_cap` before writing (so storing
    never fails on an oversized value, it splits). This keeps every value far below
    PostgreSQL's 1 GB field limit at any library size. ``normalized`` records
    whether the stored cell vectors are unit-normalized. The process-wide L2 cell
    cache is invalidated for ``index_name`` so a rebuild never leaves stale vectors
    resident.
    """
    from .index_build_helpers import store_segmented_blob

    part_mb = config.IVF_MAX_PART_SIZE_MB if max_part_size_mb is None else int(max_part_size_mb)
    part_bytes = part_mb * 1024 * 1024

    centroids, id2cell, cells = _split_cells_over_cap(centroids, id2cell, cells, dim, part_bytes)
    dir_blob = pack_directory(centroids, id2cell, item_ids, dim, metric, normalized=normalized)
    with db_conn.cursor() as cur:
        cur.execute(f"DELETE FROM {IVF_CELL_TABLE} WHERE index_name = %s", (index_name,))
        for cell_id, ids, vecs in cells:
            if ids.shape[0] == 0:
                continue
            packed = pack_cell(ids, vecs)
            cur.execute(
                f"INSERT INTO {IVF_CELL_TABLE} (index_name, cell_id, cell_data) VALUES (%s, %s, %s)",
                (index_name, int(cell_id), psycopg2.Binary(packed)),
            )
    store_segmented_blob(db_conn, IVF_DIR_TABLE, f"{index_name}__ivf_dir", dir_blob, max_part_size_mb=part_mb)
    invalidate_global_cell_cache(index_name)


def _bounded_cell_groups(
    members: np.ndarray,
    member_vecs: np.ndarray,
    base_centroid: np.ndarray,
    max_records: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split one coarse cell into ``(member_indices, centroid)`` groups under the cap.

    A cell within ``max_records`` is kept whole. An oversized cell is first
    sub-clustered with k-means for locality; any sub-cluster k-means cannot shrink
    below the cap -- typically a block of identical vectors that always collapses
    into a single cluster (e.g. instrumental tracks sharing one lyrics embedding) --
    is then hard-split into fixed-size chunks by id, so the size bound always holds
    regardless of vector distribution. Chunk centroids are the chunk mean (the
    shared point itself when the rows are identical).
    """
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
            chunk = grp[start:start + max_records]
            chunk_centroid = grp_vecs[start:start + max_records].mean(axis=0).astype(np.float32)
            groups.append((chunk, chunk_centroid))
    return groups


def build_and_store_paged_ivf(
    db_conn,
    index_name: str,
    vectors: np.ndarray,
    item_ids: List[str],
    dim: int,
    metric: str,
) -> bool:
    """Build a disk-paged IVF index from a full ``(N, dim)`` float32 matrix.

    Trains a coarse k-means quantizer on a bounded sample, assigns every vector
    to a cell, splits oversized cells so none exceeds ``IVF_MAX_CELL_MB``, packs
    each cell, and persists via :func:`store_paged_ivf`. Returns False (without
    writing) when the matrix is empty.
    """
    from sklearn.cluster import MiniBatchKMeans

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n_items = vectors.shape[0]
    if n_items == 0 or len(item_ids) != n_items:
        logger.warning("IVF build '%s': empty or mismatched input (n=%d ids=%d).", index_name, n_items, len(item_ids))
        return False
    if vectors.shape[1] != dim:
        raise ValueError(f"IVF build '{index_name}': matrix dim {vectors.shape[1]} != {dim}")

    metric = (metric or "angular").lower()
    normalized = metric == "angular"
    train_mat = _normalize_rows(vectors) if normalized else vectors

    base_nlist = int(round(8.0 * np.sqrt(max(1, n_items))))
    nlist = max(1, min(config.IVF_NLIST_MAX, base_nlist, n_items))

    sample_n = min(n_items, config.IVF_TRAIN_SAMPLE_MAX)
    if sample_n < n_items:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_items, size=sample_n, replace=False)
        sample = train_mat[sample_idx]
    else:
        sample = train_mat

    logger.info("IVF build '%s': training %d cells on %d sampled vectors (N=%d, dim=%d).", index_name, nlist, sample_n, n_items, dim)
    km = MiniBatchKMeans(n_clusters=nlist, batch_size=10000, n_init=1, max_iter=25, random_state=0)
    km.fit(sample)
    centroids = km.cluster_centers_.astype(np.float32)

    labels = np.empty(n_items, dtype=np.int64)
    for start in range(0, n_items, 20000):
        labels[start:start + 20000] = km.predict(train_mat[start:start + 20000])

    max_cell_bytes = min(config.IVF_MAX_CELL_MB, config.IVF_MAX_PART_SIZE_MB) * 1024 * 1024
    max_cell_records = max(1, max_cell_bytes // (4 + dim * 4))
    int_ids = np.arange(n_items, dtype=np.int32)

    cells: List[Tuple[int, np.ndarray, np.ndarray]] = []
    id2cell = np.empty(n_items, dtype=np.uint32)
    centroid_list: List[np.ndarray] = [centroids[c] for c in range(nlist)]
    next_cell_id = nlist

    for c in range(nlist):
        members = np.where(labels == c)[0]
        if members.shape[0] == 0:
            cells.append((c, np.empty(0, dtype=np.int32), np.empty((0, dim), dtype=np.float32)))
            continue
        reused_c = False
        for grp, centroid in _bounded_cell_groups(members, train_mat[members], centroids[c], max_cell_records):
            if not reused_c:
                assigned_cell = c
                centroid_list[c] = centroid
                reused_c = True
            else:
                assigned_cell = next_cell_id
                centroid_list.append(centroid)
                next_cell_id += 1
            cells.append((assigned_cell, int_ids[grp], train_mat[grp]))
            id2cell[grp] = assigned_cell

    final_centroids = np.ascontiguousarray(np.vstack(centroid_list), dtype=np.float32)
    logger.info("IVF build '%s': %d cells after splitting (max_cell_records=%d).", index_name, len(centroid_list), max_cell_records)
    store_paged_ivf(db_conn, index_name, final_centroids, id2cell, list(item_ids), cells, dim, metric,
                    max_part_size_mb=config.IVF_MAX_PART_SIZE_MB, normalized=normalized)
    return True


def has_paged_ivf(db_conn, index_name: str) -> bool:
    """True if a built IVF directory exists for ``index_name``."""
    from .index_build_helpers import load_segmented_blob

    try:
        blob = load_segmented_blob(db_conn, IVF_DIR_TABLE, f"{index_name}__ivf_dir")
        return blob is not None and len(blob) >= _HEADER_SIZE
    except Exception:
        return False


def _setup_disk_cell_file(db_conn, index_name: str, dim: int, metric: str, dir_blob: bytes, label: str):
    """Export this index's cells to a local mmap file and open it.

    The filename embeds a hash of the directory blob, so an unchanged index
    reuses the existing file across restarts (no re-export) while a rebuild
    produces a new file (versioned name = Windows-safe swap). Returns
    ``(mmap, offsets)`` or ``(None, None)`` to fall back to Postgres reads when
    the cache is disabled or the dir is not writable.
    """
    if not config.IVF_DISK_CACHE_ENABLED:
        return None, None
    try:
        cache_dir = config.IVF_DISK_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        build_id = hashlib.sha1(dir_blob).hexdigest()[:16]
        path = _cell_file_path(cache_dir, index_name, build_id)
        with _IVF_FILE_SWAP_LOCK:
            if not os.path.exists(path):
                n = _export_cells_to_file(db_conn, index_name, dim, metric, path)
                logger.info("IVF index '%s' exported %d cells to %s.", label, n, path)
            _prune_old_cell_files(cache_dir, index_name, path)
        mm, _dim, offsets = _open_cell_file(path)
        return mm, offsets
    except Exception as e:
        logger.warning("IVF index '%s' disk cell cache unavailable (%s); reading cells from Postgres.", label, e)
        return None, None


def load_paged_ivf_index(
    db_conn,
    index_name: str,
    expected_dim: int,
    metric: str,
    conn_factory: Optional[Callable[[], "psycopg2.extensions.connection"]] = None,
    label: Optional[str] = None,
):
    """Load a persisted IVF index into a ``PagedIvfIndex``.

    Returns ``(index, id_map, reverse_id_map)`` mirroring
    :func:`tasks.index_build_helpers.load_ivf_index_from_db`, or ``None`` if
    no IVF directory is present.
    """
    from .index_build_helpers import load_segmented_blob

    label = label or index_name
    invalidate_global_cell_cache(index_name)
    blob = load_segmented_blob(db_conn, IVF_DIR_TABLE, f"{index_name}__ivf_dir")
    if not blob:
        return None
    centroids, id2cell, item_ids, dim, stored_metric, normalized = unpack_directory(bytes(blob))
    if expected_dim is not None and dim != expected_dim:
        logger.error("IVF '%s': dimension mismatch db=%s expected=%s", label, dim, expected_dim)
        return None

    if conn_factory is None:
        from app_helper import get_db
        conn_factory = get_db

    mmap_obj, cell_offsets = _setup_disk_cell_file(db_conn, index_name, dim, stored_metric or metric, bytes(blob), label)

    index = PagedIvfIndex(
        centroids=centroids,
        id2cell=id2cell,
        item_ids=item_ids,
        dim=dim,
        metric=stored_metric or metric,
        index_name=index_name,
        conn_factory=conn_factory,
        normalized=normalized,
        mmap_obj=mmap_obj,
        cell_offsets=cell_offsets,
    )
    id_map = {i: item_id for i, item_id in enumerate(item_ids)}
    reverse_id_map = {item_id: i for i, item_id in id_map.items()}
    logger.info("IVF index '%s' loaded: %d items, %d cells, dim=%d, normalized=%s, disk_mmap=%s.", label, len(item_ids), centroids.shape[0], dim, normalized, mmap_obj is not None)
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
    expected_dim: int,
    metric: str,
    label: Optional[str] = None,
):
    """Load a disk-paged IVF index.

    Returns ``(index, id_map, reverse_id_map)`` or ``None`` if the index has not
    been built yet.
    """
    return load_paged_ivf_index(db_conn, index_name, expected_dim, metric, label=label)
