# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neural fingerprint index: built by the worker into ivf_dir and ivf_cell, paged by Flask.

Every analysed track stores one 32-byte code per half second in
embedding.neural_fingerprint (tasks.neural_fingerprint). This index follows the
lifecycle and the storage of the other similarity indexes: the worker builds
it at the rebuild points of the analysis run, in one transaction it commits
itself, stores the directory in ivf_dir and the cells in ivf_cell, and
publishes the index-reload event; the web process loads the directory at
startup (a few megabytes, seconds at any library size, blocking like every
other index load) and reads the cells a query probes from ivf_cell on demand,
keeping them in a RAM cache bounded by NEURAL_FINGERPRINT_CACHE_MB that is
dropped when the recording search has been idle. Nothing is copied to local
disk and no step on either side holds more than one part or one training
sample in memory.

Main Features:
* index rows: every NEURAL_FINGERPRINT_INDEX_STRIDE-th stored row of a track,
  at most _MAX_TRACK_ROWS of them (index_rows); the blobs keep every row and
  the alignment check reads them, so the stride only trades index size and
  query work against recall on degraded clips
* two-level quantizer (train_quantizer): about sqrt(rows) cells, at most
  _MAX_CELLS, trained on at least _TRAIN_ROWS_PER_CELL rows per cell sampled
  100 per track from random tracks; past _SINGLE_LEVEL_CELLS the cells are
  grouped under sqrt(cells) coarse centroids and a row is assigned to the
  nearest cell of its nearest group, so assigning a million tracks costs
  minutes instead of hours; a query still ranks the flat cell list exactly
* build_and_store_neural_fingerprint_index: full build or append, in parts of
  at most _PART_ROWS rows cut at track boundaries (label_tracks yields them
  with their codes). Every part writes one ivf_cell row per cell under the
  index name neural_fingerprint_index/p<part>: the cell's codes, then the
  track index, the offset in the track and the inverse norm of the decoded
  vector, 40 bytes per row (pack_cell / unpack_cell); a slice larger than
  IVF_MAX_PART_SIZE_MB is split over rows p<part>.1, p<part>.2 and so on, so
  every stored value stays under the cap like the other indexes' cells, and
  the reader concatenates whatever rows a cell has. A full build retrains
  the quantizer and replaces every part; an append keeps the quantizer,
  labels only the tracks fingerprinted since the last build and adds parts.
  The quantizer is retrained when the library has grown
  NEURAL_FINGERPRINT_RETRAIN_GROWTH times since it was trained, when more than
  a tenth of the indexed tracks are gone, when the largest cell holds more
  than ten times the average, or when the codebook, the stride or the layout
  changed
* directory blob (ivf_dir, neural_fingerprint_index__ivf_dir): build id,
  codebook id, stride, the quantizer, ids and lengths in build order, per-cell
  sizes, part count, the library size the quantizer was trained on; a tiny
  neural_fingerprint_index__build row carries the build id alone so a
  freshness check costs one small read
* Flask side: load_at_startup and ensure_loaded read the directory into ONE
  immutable Pack swapped by a single reference assignment; reload_from_db
  swaps to a new build on the index-reload event and drops the old build's
  cached cells; a directory of an older layout or another codebook is
  reported as IndexUnavailable with a message that is safe to show
* Search: each pass first lists the cells its segments probe, fetches the
  ones not cached in one query (every part of each cell, concatenated), then
  scores the segments in parallel threads (NEURAL_FINGERPRINT_QUERY_THREADS,
  0 = one per core) through the codebook lookup table; every third segment
  votes first and, when one track already leads the runner-up by
  _EARLY_EXIT_RATIO with at least _EARLY_EXIT_VOTES, the rest is skipped.
  Every neighbour votes for (track, offset) with its similarity, votes within
  one hop are pooled, and the best candidates are verified by the mean cosine
  between the whole clip and the track at that offset, read from the
  candidates' blobs in one query; a track deleted since the build has no blob
  and therefore never comes back. Rows carry item_id, score, votes,
  offset_seconds, identified and lead, the last two set by flag_identified
  over the full candidate list (the search-by-song merge reuses it). identify
  embeds a clip first; identify_vectors takes a fingerprint sequence directly
  and can leave given tracks out, which is how a stored song is searched for
  its other recordings without finding itself
* IndexUnavailable is the one exception whose text may reach a user: it
  carries the curated "not built yet", "being prepared", "older version" and
  similar messages, and any other failure of a load is wrapped into it with a
  generic text, so an internal id or a library error never leaves the log
* Per-server scope like every other index: the index holds the union of all
  servers, and a request scoped to a server votes only over that server's
  tracks through the shared availability mask (tasks.index_availability),
  cached per server and build for 30 s and dropped by
  invalidate_availability_cache when the mappings change
* picker_where gives the song picker of the Search by Song tab a server-side
  filter (songs that have a fingerprint) once the index is loaded, and None
  before, so the picker never ships a list of every indexed id per keystroke
"""

import io
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

import config
from tasks import ivf_quant
from tasks.index_availability import active_availability_scope, build_availability_mask
from tasks.index_build_helpers import load_segmented_blob, store_segmented_blob
from tasks.neural_fingerprint import (
    CODE_BYTES, DIM, HEADER_BYTES, HOP_SAMPLES, HOP_SECONDS, PQ_CENTROIDS, PQ_SUBDIM, PQ_SUBSPACES, codebook,
    decode_blob, decode_codes, fingerprint_audio, is_available, is_enabled,
)

logger = logging.getLogger(__name__)

INDEX_NAME = 'neural_fingerprint_index'
DIR_TABLE = 'ivf_dir'
CELL_TABLE = 'ivf_cell'
_DIR_NAME = f'{INDEX_NAME}__ivf_dir'
_BUILD_NAME = f'{INDEX_NAME}__build'
_CELL_NAMESPACE = f'{INDEX_NAME}/p'
FORMAT = 5
_MAX_CELLS = 65536
_SINGLE_LEVEL_CELLS = 64
_MIN_TRAIN_ROWS = 2000
_TRAIN_ROWS_PER_CELL = 20
_TRAIN_ROWS_PER_TRACK = 100
_MAX_TRACK_ROWS = 65535
_ASSIGN_BATCH = 65536
_BLOB_CHUNK = 500
_PART_ROWS = 1 << 23
_NORM_CHUNK_ROWS = 1 << 18
_CELL_WRITE_BATCH = 256
_CELL_FETCH_BATCH = 512
_ROW_BYTES = CODE_BYTES + 4 + 2 + 2
_TOP_K = 8
_VERIFY = 20
_FIRST_PASS_EVERY = 3
_EARLY_EXIT_RATIO = 4.0
_EARLY_EXIT_VOTES = 3.0
_MAX_QUERY_THREADS = 8
_REMOVED_FRACTION = 0.1
_IMBALANCE = 10.0
_SUBSPACE_OFFSETS = (np.arange(PQ_SUBSPACES, dtype=np.int32) * PQ_CENTROIDS)[None, :]
PICKER_WHERE = (
    'EXISTS (SELECT 1 FROM embedding e WHERE e.item_id = score.item_id AND e.neural_fingerprint IS NOT NULL)',
    (),
)
OLDER_LAYOUT = 'The neural fingerprint index was built by an older version; the next analysis run rebuilds it.'
LOAD_FAILED = 'The neural fingerprint index could not be loaded; check the container logs.'
_AVAILABILITY_CACHE = {}
_AVAILABILITY_CACHE_LOCK = threading.Lock()
_AVAILABILITY_CACHE_TTL = 30.0
_CANONICAL = {}
_LOCK = threading.RLock()
_STATE = {'pack': None, 'building': False, 'error': None, 'gpu': None, 'executor': None}


class IndexUnavailable(RuntimeError):
    pass


class Quantizer(NamedTuple):
    coarse: np.ndarray
    cells: np.ndarray
    offsets: np.ndarray

    @property
    def n_cells(self):
        return int(self.cells.shape[0])


class Pack(NamedTuple):
    build_id: str
    ids: np.ndarray
    lengths: np.ndarray
    centroids: np.ndarray
    cell_bounds: np.ndarray
    stride: int
    parts: int
    codebook_id: int

    @property
    def live_tracks(self):
        return int(self.ids.size)

    @property
    def n_cells(self):
        return int(self.cell_bounds.size - 1)


class _CellCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._bytes = 0

    def _limit(self):
        return max(0, int(config.NEURAL_FINGERPRINT_CACHE_MB)) * 1024 * 1024

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key, arrays):
        size = sum(int(array.nbytes) for array in arrays)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._entries[key] = (arrays, size)
            self._bytes += size
            limit = self._limit()
            while self._bytes > limit and self._entries:
                _key, (_arrays, dropped) = self._entries.popitem(last=False)
                self._bytes -= dropped

    def drop(self, build_id=None):
        with self._lock:
            if build_id is None:
                self._entries.clear()
                self._bytes = 0
                return
            for key in [key for key in self._entries if key[0] != build_id]:
                _arrays, size = self._entries.pop(key)
                self._bytes -= size

    def resident_mb(self):
        with self._lock:
            return round(self._bytes / (1024 * 1024), 1)

    def __len__(self):
        with self._lock:
            return len(self._entries)


_CELLS = _CellCache()


def _stride():
    return max(1, int(config.NEURAL_FINGERPRINT_INDEX_STRIDE))


def index_rows(codes):
    if codes is None:
        return None
    return np.ascontiguousarray(codes[::_stride()][:_MAX_TRACK_ROWS])


def _cell_count(n_rows):
    return int(max(1, min(_MAX_CELLS, round(math.sqrt(max(1, n_rows))))))


def _unit_rows(vectors):
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)


def train_centroids(sample, n_cells):
    from sklearn.cluster import MiniBatchKMeans

    if n_cells <= 1 or sample.shape[0] <= n_cells:
        return _unit_rows(np.ascontiguousarray(sample[:max(1, n_cells)], dtype=np.float32))
    km = MiniBatchKMeans(n_clusters=n_cells, batch_size=10000, n_init=1, max_iter=25, random_state=0, init='random')
    km.fit(sample)
    return _unit_rows(km.cluster_centers_.astype(np.float32))


def train_quantizer(sample, n_cells):
    sample = np.ascontiguousarray(sample, dtype=np.float32)
    if n_cells <= _SINGLE_LEVEL_CELLS or sample.shape[0] <= n_cells:
        cells = train_centroids(sample, n_cells)
        coarse = _unit_rows(cells.mean(axis=0, keepdims=True))
        return Quantizer(coarse, cells, np.array([0, cells.shape[0]], dtype=np.int64))
    coarse = train_centroids(sample, int(math.ceil(math.sqrt(n_cells))))
    groups = _nearest_centroids(sample, coarse)
    per_group = int(math.ceil(n_cells / coarse.shape[0]))
    cells, offsets = [], [0]
    for g in range(coarse.shape[0]):
        rows = sample[groups == g]
        fine = coarse[g:g + 1] if rows.shape[0] == 0 else train_centroids(rows, min(per_group, rows.shape[0]))
        cells.append(fine)
        offsets.append(offsets[-1] + fine.shape[0])
    return Quantizer(coarse, np.concatenate(cells), np.asarray(offsets, dtype=np.int64))


def _gpu_array_module():
    if _STATE['gpu'] is None:
        try:
            import cupy

            cupy.zeros(1).sum()
            _STATE['gpu'] = cupy
            logger.info('Neural fingerprint index assigns rows on the GPU through cupy')
        except Exception:
            _STATE['gpu'] = False
    return _STATE['gpu'] or None


def _nearest_centroids(block, centroids):
    gpu = _gpu_array_module()
    if gpu is not None:
        try:
            scores = gpu.asarray(block) @ gpu.asarray(centroids).T
            return gpu.asnumpy(gpu.argmax(scores, axis=1)).astype(np.int32)
        except Exception:
            logger.exception('GPU assignment failed; the rest of this build runs on the CPU')
            _STATE['gpu'] = False
    return np.argmax(block @ centroids.T, axis=1).astype(np.int32)


def _assign_block(block, quantizer):
    if quantizer.coarse.shape[0] == 1:
        return _nearest_centroids(block, quantizer.cells)
    groups = _nearest_centroids(block, quantizer.coarse)
    order = np.argsort(groups, kind='stable')
    edges = np.searchsorted(groups[order], np.arange(quantizer.coarse.shape[0] + 1))
    labels = np.empty(block.shape[0], dtype=np.int32)
    for g in range(quantizer.coarse.shape[0]):
        a, b = int(edges[g]), int(edges[g + 1])
        if a == b:
            continue
        base = int(quantizer.offsets[g])
        picked = order[a:b]
        labels[picked] = base + _nearest_centroids(block[picked], quantizer.cells[base:int(quantizer.offsets[g + 1])])
    return labels


def assign_cells(codes, quantizer):
    labels = np.empty(codes.shape[0], dtype=np.int32)
    for start in range(0, codes.shape[0], _ASSIGN_BATCH):
        block = decode_codes(codes[start:start + _ASSIGN_BATCH])
        labels[start:start + block.shape[0]] = _assign_block(block, quantizer)
    return labels


def _squared_norm_table():
    return (2.0 * codebook()[2]).astype(np.float32).ravel()


def _inverse_norms(block, squared):
    out = np.empty(block.shape[0], dtype=np.float32)
    for a in range(0, block.shape[0], _NORM_CHUNK_ROWS):
        piece = block[a:a + _NORM_CHUNK_ROWS]
        total = np.take(squared, piece.astype(np.int32) + _SUBSPACE_OFFSETS).sum(axis=1)
        out[a:a + piece.shape[0]] = 1.0 / (np.sqrt(total) + 1e-9)
    return out


def pack_cell(codes, tracks, offsets, norms):
    return (
        np.ascontiguousarray(codes, dtype=np.uint8).tobytes()
        + np.ascontiguousarray(tracks, dtype='<u4').tobytes()
        + np.ascontiguousarray(offsets, dtype='<u2').tobytes()
        + np.ascontiguousarray(norms, dtype='<f2').tobytes()
    )


def unpack_cell(blob):
    raw = bytes(blob)
    n, rest = divmod(len(raw), _ROW_BYTES)
    if rest:
        raise ValueError('not a neural fingerprint cell')
    codes = np.frombuffer(raw, dtype=np.uint8, count=n * CODE_BYTES).reshape(n, CODE_BYTES)
    tracks = np.frombuffer(raw, dtype='<u4', count=n, offset=n * CODE_BYTES)
    offsets = np.frombuffer(raw, dtype='<u2', count=n, offset=n * (CODE_BYTES + 4))
    norms = np.frombuffer(raw, dtype='<f2', count=n, offset=n * (CODE_BYTES + 6))
    return codes, tracks, offsets, norms


def _part_name(part, piece=0):
    return f'{_CELL_NAMESPACE}{part}' if piece == 0 else f'{_CELL_NAMESPACE}{part}.{piece}'


def _rows_per_cell_row():
    return max(1, (int(config.IVF_MAX_PART_SIZE_MB) * 1024 * 1024) // _ROW_BYTES)


def part_cells(part, labels, codes, tracks, offsets, n_cells):
    norms = _inverse_norms(codes, _squared_norm_table())
    order = np.argsort(labels, kind='stable')
    counts = np.bincount(labels, minlength=n_cells).astype(np.int64)
    bounds = np.concatenate(([0], np.cumsum(counts)))
    cap = _rows_per_cell_row()
    for cell in np.flatnonzero(counts):
        rows = order[bounds[cell]:bounds[cell + 1]]
        for piece, start in enumerate(range(0, rows.size, cap)):
            picked = rows[start:start + cap]
            yield _part_name(part, piece), int(cell), pack_cell(codes[picked], tracks[picked], offsets[picked], norms[picked])
    return counts


def _write_cell_rows(conn, rows):
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {CELL_TABLE} (index_name, cell_id, cell_data) VALUES %s "
            "ON CONFLICT (index_name, cell_id) DO UPDATE SET cell_data = EXCLUDED.cell_data",
            [(name, cell, psycopg2.Binary(blob)) for name, cell, blob in rows],
            page_size=len(rows),
        )


def _store_part(conn, part, labels, codes, tracks, offsets, n_cells):
    cells = part_cells(part, labels, codes, tracks, offsets, n_cells)
    pending = []
    while True:
        try:
            pending.append(next(cells))
        except StopIteration as done:
            if pending:
                _write_cell_rows(conn, pending)
            return done.value
        if len(pending) >= _CELL_WRITE_BATCH:
            _write_cell_rows(conn, pending)
            pending = []


def _delete_cells(conn):
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {CELL_TABLE} WHERE index_name LIKE %s ESCAPE '\\'",
            (_CELL_NAMESPACE.replace('_', r'\_') + '%',),
        )


def _quantizer_blob_fields(quantizer):
    return {
        'coarse': ivf_quant.encode_vectors(np.asarray(quantizer.coarse, dtype=np.float32), ivf_quant.DTYPE_I8),
        'centroids': ivf_quant.encode_vectors(np.asarray(quantizer.cells, dtype=np.float32), ivf_quant.DTYPE_I8),
        'group_offsets': np.asarray(quantizer.offsets, dtype=np.int64),
    }


def pack_directory(build_id, codebook_id, quantizer, ids, lengths, cell_sizes, parts, trained_tracks):
    buffer = io.BytesIO()
    np.savez(
        buffer, format=np.int64(FORMAT), build_id=np.asarray(str(build_id)), codebook_id=np.uint32(codebook_id),
        stride=np.int64(_stride()), ids=np.asarray(list(ids), dtype=str), lengths=np.asarray(lengths, dtype=np.int64),
        cell_sizes=np.asarray(cell_sizes, dtype=np.int64), parts=np.int64(parts),
        trained_tracks=np.int64(trained_tracks), **_quantizer_blob_fields(quantizer),
    )
    return buffer.getvalue()


def _decode_i8(vectors):
    return _unit_rows(vectors.astype(np.float32) / ivf_quant.I8_SCALE)


def unpack_directory(blob):
    with np.load(io.BytesIO(bytes(blob)), allow_pickle=False) as data:
        layout = int(data['format']) if 'format' in data.files else 0
        if layout != FORMAT:
            return {'format': layout, 'build_id': str(data['build_id']) if 'build_id' in data.files else ''}
        quantizer = Quantizer(
            _decode_i8(data['coarse']), _decode_i8(data['centroids']), data['group_offsets'].astype(np.int64),
        )
        return {
            'format': layout, 'build_id': str(data['build_id']), 'codebook_id': int(data['codebook_id']),
            'stride': int(data['stride']), 'quantizer': quantizer, 'centroids': quantizer.cells,
            'ids': [str(i) for i in data['ids']], 'lengths': data['lengths'].astype(np.int64),
            'cell_sizes': data['cell_sizes'].astype(np.int64), 'parts': int(data['parts']),
            'trained_tracks': int(data['trained_tracks']),
        }


def needs_full_build(directory, fingerprinted_ids, codebook_id):
    if directory is None:
        return 'no index yet'
    if directory['format'] != FORMAT:
        return 'the index layout changed'
    if directory['codebook_id'] != codebook_id:
        return 'the codebook changed'
    if directory['stride'] != _stride():
        return 'the index stride changed'
    known = set(directory['ids'])
    current = set(fingerprinted_ids)
    removed = len(known - current)
    if known and removed > _REMOVED_FRACTION * len(known):
        return f'{removed} of {len(known)} indexed tracks are gone'
    growth = float(config.NEURAL_FINGERPRINT_RETRAIN_GROWTH)
    if len(current) >= growth * max(1, directory['trained_tracks']):
        return f'the library grew from {directory["trained_tracks"]} to {len(current)} tracks since the centroids were trained'
    sizes = directory['cell_sizes']
    if sizes.size and sizes.max() > _IMBALANCE * max(1.0, float(sizes.mean())):
        return f'the largest cell holds {int(sizes.max())} rows against an average of {sizes.mean():.0f}'
    return None


def _fingerprinted_tracks(conn):
    cur = conn.cursor()
    try:
        cur.execute('SELECT item_id FROM embedding WHERE neural_fingerprint IS NOT NULL ORDER BY item_id')
        return [row[0] for row in cur.fetchall()]
    finally:
        cur.close()


def _estimated_rows(conn):
    cur = conn.cursor()
    try:
        cur.execute(
            'SELECT coalesce(sum((octet_length(neural_fingerprint) - %s) / %s), 0) '
            'FROM embedding WHERE neural_fingerprint IS NOT NULL',
            (int(HEADER_BYTES), int(CODE_BYTES)),
        )
        return int(cur.fetchone()[0] or 0) // _stride()
    finally:
        cur.close()


def iter_codes(conn, ids):
    cur = conn.cursor()
    try:
        for start in range(0, len(ids), _BLOB_CHUNK):
            chunk = list(ids[start:start + _BLOB_CHUNK])
            cur.execute('SELECT item_id, neural_fingerprint FROM embedding WHERE item_id = ANY(%s)', (chunk,))
            found = {row[0]: row[1] for row in cur.fetchall()}
            for item_id in chunk:
                blob = found.get(item_id)
                yield item_id, (decode_blob(bytes(blob)) if blob is not None else None)
    finally:
        cur.close()


def training_sample(codes_iter, cap, rng):
    pieces, total = [], 0
    for _item_id, codes in codes_iter:
        codes = index_rows(codes)
        if codes is None or not codes.shape[0]:
            continue
        take = min(_TRAIN_ROWS_PER_TRACK, codes.shape[0])
        pick = np.sort(rng.choice(codes.shape[0], take, replace=False))
        pieces.append(decode_codes(codes[pick]))
        total += take
        if total >= cap:
            break
    if not pieces:
        return np.zeros((0, DIM), dtype=np.float32)
    return np.concatenate(pieces)[:cap]


def label_tracks(codes_iter, quantizer, part_rows=_PART_ROWS):
    ids, lengths, labels, kept, pending = [], [], [], [], []
    pending_rows = chunk_rows = 0

    def flush():
        nonlocal pending, pending_rows
        if pending:
            block = np.concatenate(pending)
            labels.append(assign_cells(block, quantizer))
            kept.append(block)
            pending, pending_rows = [], 0

    for item_id, codes in codes_iter:
        codes = index_rows(codes)
        if codes is None or not codes.shape[0]:
            continue
        if chunk_rows and chunk_rows + codes.shape[0] > part_rows:
            flush()
            yield ids, np.asarray(lengths, dtype=np.int64), np.concatenate(labels), np.concatenate(kept)
            ids, lengths, labels, kept, chunk_rows = [], [], [], [], 0
        ids.append(item_id)
        lengths.append(int(codes.shape[0]))
        pending.append(codes)
        pending_rows += int(codes.shape[0])
        chunk_rows += int(codes.shape[0])
        if pending_rows >= _ASSIGN_BATCH:
            flush()
    flush()
    if ids:
        yield ids, np.asarray(lengths, dtype=np.int64), np.concatenate(labels), np.concatenate(kept)


def _load_directory(conn):
    blob = load_segmented_blob(conn, DIR_TABLE, _DIR_NAME)
    return unpack_directory(blob) if blob else None


def _stored_build_id(conn):
    blob = load_segmented_blob(conn, DIR_TABLE, _BUILD_NAME)
    return bytes(blob).decode('ascii') if blob else None


def _store_directory(conn, directory_blob, build_id):
    store_segmented_blob(conn, DIR_TABLE, _DIR_NAME, directory_blob)
    store_segmented_blob(conn, DIR_TABLE, _BUILD_NAME, build_id.encode('ascii'))


def _store_labelled_parts(conn, chunks, n_cells, first_part, first_track):
    ids, lengths, cell_sizes, part, track = [], [], np.zeros(n_cells, dtype=np.int64), first_part, first_track
    for chunk_ids, chunk_lengths, labels, codes in chunks:
        tracks = np.repeat(np.arange(track, track + len(chunk_ids), dtype=np.uint32), chunk_lengths)
        offsets = np.concatenate([np.arange(n, dtype=np.uint16) for n in chunk_lengths])
        cell_sizes += _store_part(conn, part, labels, codes, tracks, offsets, n_cells)
        ids.extend(chunk_ids)
        lengths.append(chunk_lengths)
        part += 1
        track += len(chunk_ids)
    joined = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
    return ids, joined, cell_sizes, part


def _full_build(conn, ids, codebook_id, reason):
    started = time.time()
    rng = np.random.default_rng(0)
    n_cells = _cell_count(_estimated_rows(conn))
    cap = max(_MIN_TRAIN_ROWS, int(config.NEURAL_FINGERPRINT_TRAIN_ROWS), _TRAIN_ROWS_PER_CELL * n_cells)
    sample_tracks = min(len(ids), max(1, math.ceil(cap / _TRAIN_ROWS_PER_TRACK)))
    chosen = [ids[i] for i in np.sort(rng.choice(len(ids), sample_tracks, replace=False))]
    sample = training_sample(iter_codes(conn, chosen), cap, rng)
    quantizer = train_quantizer(sample, n_cells)
    sample_rows = int(sample.shape[0])
    del sample
    trained = time.time()
    _delete_cells(conn)
    kept_ids, lengths, cell_sizes, parts = _store_labelled_parts(
        conn, label_tracks(iter_codes(conn, ids), quantizer), quantizer.n_cells, 0, 0,
    )
    if not parts:
        raise RuntimeError('no track carries a usable neural fingerprint')
    build_id = uuid.uuid4().hex
    _store_directory(
        conn, pack_directory(build_id, codebook_id, quantizer, kept_ids, lengths, cell_sizes, parts, len(kept_ids)),
        build_id,
    )
    logger.info(
        'Neural fingerprint index built from scratch (%s): %d tracks, %d rows, %d cells in %d groups, %d parts; '
        'quantizer from %d rows in %.0fs, rows assigned and stored in %.0fs',
        reason, len(kept_ids), int(lengths.sum()), quantizer.n_cells, int(quantizer.coarse.shape[0]), parts,
        sample_rows, trained - started, time.time() - trained,
    )


def _incremental_build(conn, directory, ids):
    started = time.time()
    known = set(directory['ids'])
    new_ids = [item_id for item_id in ids if item_id not in known]
    if not new_ids:
        logger.info('Neural fingerprint index is up to date (%d tracks)', len(directory['ids']))
        return
    quantizer = directory['quantizer']
    kept_ids, lengths, cell_sizes, parts = _store_labelled_parts(
        conn, label_tracks(iter_codes(conn, new_ids), quantizer), quantizer.n_cells, directory['parts'],
        len(directory['ids']),
    )
    if not kept_ids:
        logger.info('Neural fingerprint index: the %d new tracks carry no usable fingerprint', len(new_ids))
        return
    build_id = uuid.uuid4().hex
    _store_directory(
        conn,
        pack_directory(
            build_id, directory['codebook_id'], quantizer, list(directory['ids']) + kept_ids,
            np.concatenate([directory['lengths'], lengths]), directory['cell_sizes'] + cell_sizes, parts,
            directory['trained_tracks'],
        ),
        build_id,
    )
    logger.info(
        'Neural fingerprint index appended: %d new tracks, %d rows, parts %d to %d, %.0fs (%d tracks indexed)',
        len(kept_ids), int(lengths.sum()), directory['parts'], parts - 1, time.time() - started,
        len(directory['ids']) + len(kept_ids),
    )


def build_and_store_neural_fingerprint_index(db_conn, force_full=False):
    if not is_enabled():
        logger.info('Neural fingerprint index skipped: NEURAL_FINGERPRINT_ENABLED is false')
        return False
    if not is_available():
        logger.info('Neural fingerprint index skipped: the model or the codebook is missing')
        return False
    codebook_id = codebook()[1]
    try:
        ids = _fingerprinted_tracks(db_conn)
        if not ids:
            logger.info('Neural fingerprint index skipped: no track has a fingerprint yet')
            return False
        directory = _load_directory(db_conn)
        reason = 'forced' if force_full else needs_full_build(directory, ids, codebook_id)
        if reason:
            _full_build(db_conn, ids, codebook_id, reason)
        else:
            _incremental_build(db_conn, directory, ids)
        db_conn.commit()
        return True
    except Exception:
        try:
            db_conn.rollback()
        except Exception:
            logger.debug('Rollback after a failed neural fingerprint index build failed', exc_info=True)
        raise


def _pack_from(directory):
    bounds = np.concatenate(([0], np.cumsum(directory['cell_sizes']))).astype(np.int64)
    return Pack(
        build_id=directory['build_id'], ids=np.asarray(directory['ids'], dtype=str), lengths=directory['lengths'],
        centroids=np.ascontiguousarray(directory['centroids'], dtype=np.float32), cell_bounds=bounds,
        stride=int(directory['stride']), parts=int(directory['parts']), codebook_id=int(directory['codebook_id']),
    )


def _current_layout(directory, build_id):
    if directory is None or directory['build_id'] != build_id:
        raise IndexUnavailable('The neural fingerprint index is being written; try again in a minute.')
    if directory['format'] != FORMAT:
        raise IndexUnavailable(OLDER_LAYOUT)
    if directory['codebook_id'] != codebook()[1]:
        raise IndexUnavailable('The neural fingerprint index was built with another codebook; rebuild the indexes.')
    return directory


def _read_pack(conn):
    build_id = _stored_build_id(conn)
    if build_id is None:
        raise IndexUnavailable('No neural fingerprint index is built yet. Run the analysis, which builds it.')
    return _pack_from(_current_layout(_load_directory(conn), build_id))


def _swap_pack(pack):
    with _LOCK:
        _STATE['pack'] = pack
    _CELLS.drop(pack.build_id)
    invalidate_availability_cache()


def is_loaded():
    return _STATE['pack'] is not None


def _current_pack():
    pack = _STATE['pack']
    if pack is None:
        raise IndexUnavailable('The neural fingerprint index is not loaded.')
    return pack


def picker_where():
    return PICKER_WHERE if is_loaded() else None


def _record_load_error(message):
    with _LOCK:
        _STATE['error'] = message


def ensure_loaded():
    from database import connect_raw

    with _LOCK:
        if is_loaded():
            return True
        if _STATE['building']:
            raise IndexUnavailable('The neural fingerprint index is being prepared; try again in a minute.')
        _STATE['building'] = True
        _STATE['error'] = None
    try:
        conn = connect_raw(application_name='neural_fingerprint_index')
        try:
            pack = _read_pack(conn)
        finally:
            conn.close()
        _swap_pack(pack)
        logger.info(
            'Neural fingerprint index loaded: %d tracks, %d cells, %d parts (build %s)',
            pack.live_tracks, pack.n_cells, pack.parts, pack.build_id,
        )
        return True
    except IndexUnavailable as exc:
        _record_load_error(str(exc))
        logger.warning('Neural fingerprint index not loaded: %s', exc)
        raise
    except Exception as exc:
        _record_load_error(LOAD_FAILED)
        logger.exception('Neural fingerprint index could not be loaded')
        raise IndexUnavailable(LOAD_FAILED) from exc
    finally:
        with _LOCK:
            _STATE['building'] = False


def reload_from_db():
    from database import connect_raw

    if not is_enabled():
        logger.info('Neural fingerprint index reload skipped: NEURAL_FINGERPRINT_ENABLED is false')
        return False
    conn = connect_raw(application_name='neural_fingerprint_index')
    try:
        build_id = _stored_build_id(conn)
        if build_id is None:
            return False
        with _LOCK:
            if is_loaded() and _STATE['pack'].build_id == build_id:
                return True
        try:
            pack = _read_pack(conn)
        except IndexUnavailable as exc:
            logger.info('Neural fingerprint index not reloaded: %s', exc)
            return False
    finally:
        conn.close()
    _swap_pack(pack)
    logger.info('Neural fingerprint index swapped to build %s (%d tracks)', pack.build_id, pack.live_tracks)
    return True


def load_at_startup():
    if not is_enabled():
        logger.info('Neural fingerprint index not loaded at startup: NEURAL_FINGERPRINT_ENABLED is false')
        return 0
    if not is_available():
        logger.info('Neural fingerprint index not loaded at startup: the model or the codebook is missing')
        return 0
    try:
        ensure_loaded()
    except IndexUnavailable:
        return 0
    with _LOCK:
        return _STATE['pack'].live_tracks if is_loaded() else 0


def start_background_load():
    with _LOCK:
        if is_loaded() or _STATE['building']:
            return None

    def _run():
        try:
            ensure_loaded()
        except Exception:
            logger.warning('Background neural fingerprint load failed; the next request retries')

    thread = threading.Thread(target=_run, name='neural-fingerprint-load', daemon=True)
    thread.start()
    return thread


def unload():
    with _LOCK:
        was_loaded = is_loaded()
        _STATE['pack'] = None
    _CELLS.drop()
    return was_loaded


def drop_cell_cache():
    _CELLS.drop()


def get_status():
    with _LOCK:
        pack = _STATE['pack']
        return {
            'available': bool(is_available()),
            'loaded': pack is not None,
            'building': bool(_STATE['building']),
            'tracks': pack.live_tracks if pack is not None else 0,
            'cells': pack.n_cells if pack is not None else 0,
            'parts': pack.parts if pack is not None else 0,
            'cached_cells': len(_CELLS),
            'cache_mb': _CELLS.resident_mb(),
            'error': _STATE['error'],
        }


def _db_connection():
    from flask import has_app_context

    if has_app_context():
        from database import get_db

        return get_db(), False
    from database import connect_raw

    return connect_raw(application_name='neural_fingerprint_query'), True


def _read_cell_rows(pack, cell_ids):
    pattern = _CELL_NAMESPACE.replace('_', r'\_') + '%'
    conn, owned = _db_connection()
    try:
        rows = []
        with conn.cursor() as cur:
            for start in range(0, len(cell_ids), _CELL_FETCH_BATCH):
                cur.execute(
                    f"SELECT index_name, cell_id, cell_data FROM {CELL_TABLE} "
                    "WHERE index_name LIKE %s ESCAPE '\\' AND cell_id = ANY(%s)",
                    (pattern, list(cell_ids[start:start + _CELL_FETCH_BATCH])),
                )
                rows.extend(cur.fetchall())
        return rows
    finally:
        if owned:
            conn.close()


def _empty_cell():
    return (
        np.zeros((0, CODE_BYTES), dtype=np.uint8), np.zeros(0, dtype=np.uint32),
        np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.float16),
    )


def _join_rows(blobs):
    pieces = [unpack_cell(blob) for blob in blobs]
    if len(pieces) == 1:
        return tuple(np.ascontiguousarray(array) for array in pieces[0])
    return tuple(np.concatenate([piece[i] for piece in pieces]) for i in range(4))


def _cells_for(pack, cell_ids):
    found, missing = {}, []
    for cell in cell_ids:
        arrays = _CELLS.get((pack.build_id, cell))
        if arrays is None:
            missing.append(cell)
        else:
            found[cell] = arrays
    if not missing:
        return found
    grouped = {cell: [] for cell in missing}
    for _name, cell, blob in _read_cell_rows(pack, missing):
        grouped[int(cell)].append(blob)
    for cell, blobs in grouped.items():
        arrays = _join_rows(blobs) if blobs else _empty_cell()
        _CELLS.put((pack.build_id, cell), arrays)
        found[cell] = arrays
    return found


def _query_threads():
    wanted = int(config.NEURAL_FINGERPRINT_QUERY_THREADS)
    if wanted <= 0:
        wanted = os.cpu_count() or 1
    return max(1, min(_MAX_QUERY_THREADS, wanted))


def _executor():
    with _LOCK:
        if _STATE['executor'] is None:
            _STATE['executor'] = ThreadPoolExecutor(max_workers=_MAX_QUERY_THREADS, thread_name_prefix='neural-vote')
        return _STATE['executor']


def _probe_cells(pack, query, nprobe):
    scores = pack.centroids @ query
    if nprobe >= scores.size:
        return np.arange(scores.size)
    return np.argpartition(-scores, nprobe)[:nprobe]


def _lookup_table(book, query):
    return np.einsum('sd,skd->sk', query.reshape(PQ_SUBSPACES, PQ_SUBDIM), book).astype(np.float32).ravel()


def _cell_scores(arrays, table, allowed):
    codes, tracks, offsets, norms = arrays
    if not codes.shape[0]:
        return None
    if allowed is not None:
        keep = allowed[tracks]
        if not keep.any():
            return None
        if not keep.all():
            codes, tracks, offsets, norms = codes[keep], tracks[keep], offsets[keep], norms[keep]
    sims = np.take(table, codes.astype(np.int32) + _SUBSPACE_OFFSETS).sum(axis=1) * norms.astype(np.float32)
    return sims, tracks, offsets


def _segment_votes(pack, book, query_vectors, qi, probed, cells, allowed):
    table = _lookup_table(book, query_vectors[qi])
    pieces = []
    for cell in probed:
        piece = _cell_scores(cells[int(cell)], table, allowed)
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        return []
    sims = np.concatenate([piece[0] for piece in pieces])
    tracks = np.concatenate([piece[1] for piece in pieces])
    offsets = np.concatenate([piece[2] for piece in pieces])
    top = np.argpartition(-sims, _TOP_K)[:_TOP_K] if sims.size > _TOP_K else np.arange(sims.size)
    shift = qi // pack.stride
    return [(int(tracks[j]), int(offsets[j]) - shift, float(sims[j])) for j in top]


def _score_segments(pack, book, query_vectors, segments, nprobe, allowed):
    probed = {qi: _probe_cells(pack, query_vectors[qi], nprobe) for qi in segments}
    cells = _cells_for(pack, sorted({int(cell) for cells in probed.values() for cell in cells}))
    if len(segments) > 1 and _query_threads() > 1:
        return list(_executor().map(
            lambda qi: _segment_votes(pack, book, query_vectors, qi, probed[qi], cells, allowed), segments,
        ))
    return [_segment_votes(pack, book, query_vectors, qi, probed[qi], cells, allowed) for qi in segments]


def _pool(votes):
    pooled = {}
    for (track, offset), value in votes.items():
        total = value + votes.get((track, offset - 1), 0.0) + votes.get((track, offset + 1), 0.0)
        if total > pooled.get(track, (0.0, 0))[0]:
            pooled[track] = (total, offset)
    return pooled


def _decided(pooled):
    if not pooled:
        return False
    totals = sorted((total for total, _offset in pooled.values()), reverse=True)
    runner_up = totals[1] if len(totals) > 1 else 0.0
    return totals[0] >= _EARLY_EXIT_VOTES and totals[0] >= _EARLY_EXIT_RATIO * runner_up


def _vote(pack, book, query_vectors, nprobe, allowed=None):
    n = query_vectors.shape[0]
    first = list(range(0, n, _FIRST_PASS_EVERY))
    rest = [qi for qi in range(n) if qi % _FIRST_PASS_EVERY]
    votes = {}
    for pass_segments in (first, rest):
        if not pass_segments:
            continue
        for segment_votes in _score_segments(pack, book, query_vectors, pass_segments, nprobe, allowed):
            for track, offset, sim in segment_votes:
                votes[(track, offset)] = votes.get((track, offset), 0.0) + sim
        pooled = _pool(votes)
        if _decided(pooled):
            return pooled, len(pass_segments)
    return _pool(votes), n


def _candidate_codes(item_ids):
    conn, owned = _db_connection()
    try:
        return {item_id: codes for item_id, codes in iter_codes(conn, list(item_ids)) if codes is not None}
    finally:
        if owned:
            conn.close()


def _verify(codes, query_vectors, offset_index, stride):
    length = int(codes.shape[0])
    base = int(offset_index) * stride
    best = (-1.0, base)
    for candidate in range(base - stride, base + stride + 1):
        lo = max(0, -candidate)
        hi = min(query_vectors.shape[0], length - candidate)
        if hi - lo < 1:
            continue
        rows = decode_codes(codes[candidate + lo: candidate + hi])
        sims = np.einsum('ij,ij->i', rows, query_vectors[lo:hi])
        score = float(sims.mean()) * (hi - lo) / query_vectors.shape[0]
        if score > best[0]:
            best = (score, candidate)
    return best


def identify(audio, sr, n_results):
    if not is_available():
        raise IndexUnavailable('The neural fingerprint model is not available here.')
    ensure_loaded()
    query_vectors = fingerprint_audio(audio, sr, HOP_SAMPLES)
    if query_vectors is None or query_vectors.shape[0] < 2:
        raise ValueError('The clip is too short to fingerprint: at least two seconds are needed.')
    return identify_vectors(query_vectors, n_results)


def invalidate_availability_cache(server_id=None):
    with _AVAILABILITY_CACHE_LOCK:
        if server_id is None:
            _AVAILABILITY_CACHE.clear()
            return
        for key in [key for key in _AVAILABILITY_CACHE if key[0] == str(server_id)]:
            _AVAILABILITY_CACHE.pop(key, None)


def _has_canonical_ids(pack):
    from tasks.simhash import is_fingerprint_id

    if pack.build_id not in _CANONICAL:
        _CANONICAL.clear()
        _CANONICAL[pack.build_id] = any(is_fingerprint_id(str(item_id)) for item_id in pack.ids)
    return _CANONICAL[pack.build_id]


def _mask_unneeded(pack, server_id):
    try:
        from tasks.mediaserver import registry

        return bool(
            server_id == str(registry.get_default_server_id() or '')
            and not registry.has_secondary_servers()
            and not _has_canonical_ids(pack)
        )
    except Exception:
        logger.debug('Single-server availability fast path failed.', exc_info=True)
        return False


def _availability_mask(pack):
    server_id = active_availability_scope()
    if server_id is None or _mask_unneeded(pack, server_id):
        return None
    from database import get_db

    key = (server_id, pack.build_id)
    now = time.monotonic()
    with _AVAILABILITY_CACHE_LOCK:
        cached = _AVAILABILITY_CACHE.get(key)
        if cached is not None and now - cached[0] < _AVAILABILITY_CACHE_TTL:
            return cached[1]
    mask = build_availability_mask(server_id, pack.ids, get_db)
    with _AVAILABILITY_CACHE_LOCK:
        stale = [k for k, v in _AVAILABILITY_CACHE.items() if k[1] != key[1] or now - v[0] >= _AVAILABILITY_CACHE_TTL]
        for old in stale:
            _AVAILABILITY_CACHE.pop(old, None)
        _AVAILABILITY_CACHE[key] = (now, mask)
    return mask


def _allowed_tracks(pack, exclude_ids):
    allowed = _availability_mask(pack)
    if not exclude_ids:
        return allowed
    allowed = np.ones(pack.ids.size, dtype=np.bool_) if allowed is None else allowed.copy()
    allowed[np.isin(pack.ids, list(exclude_ids))] = False
    return allowed


def flag_identified(rows):
    for row in rows:
        row['identified'] = False
        row['lead'] = None
    if not rows:
        return rows
    lead = rows[0]['score'] - rows[1]['score'] if len(rows) > 1 else float('inf')
    rows[0]['identified'] = bool(
        rows[0]['score'] >= float(config.NEURAL_FINGERPRINT_MIN_SCORE)
        and lead >= float(config.NEURAL_FINGERPRINT_MIN_LEAD)
    )
    rows[0]['lead'] = round(float(lead), 3) if np.isfinite(lead) else None
    return rows


def identify_vectors(query_vectors, n_results, exclude_ids=()):
    ensure_loaded()
    pack = _current_pack()
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    if query_vectors.ndim != 2 or query_vectors.shape[0] < 2:
        raise ValueError('The query is too short to identify: at least two segments are needed.')
    started = time.time()
    nprobe = max(1, int(config.NEURAL_FINGERPRINT_NPROBE))
    pooled, scored_segments = _vote(pack, codebook()[0], query_vectors, nprobe, _allowed_tracks(pack, exclude_ids))
    ranked = sorted(pooled.items(), key=lambda kv: -kv[1][0])[:max(int(n_results), _VERIFY)]
    candidates = [(str(pack.ids[track]), votes, offset) for track, (votes, offset) in ranked]
    fetched = _candidate_codes([item_id for item_id, _votes, _offset in candidates])
    scored = []
    for item_id, votes, offset in candidates:
        codes = fetched.get(item_id)
        if codes is None or not codes.shape[0]:
            continue
        score, aligned = _verify(codes, query_vectors, offset, pack.stride)
        scored.append((item_id, score, votes, aligned))
    scored.sort(key=lambda row: -row[1])
    rows = flag_identified([
        {
            'item_id': item_id,
            'score': round(float(score), 3),
            'votes': round(float(votes), 2),
            'offset_seconds': round(float(aligned) * HOP_SECONDS, 1),
        }
        for item_id, score, votes, aligned in scored
    ])
    logger.info(
        'Neural fingerprint identify: %d of %d query segments voted, %d candidates, %.1fs, best score %.3f lead %s '
        'identified %s',
        scored_segments, query_vectors.shape[0], len(pooled), time.time() - started, rows[0]['score'] if rows else 0.0,
        rows[0]['lead'] if rows else None, bool(rows and rows[0]['identified']),
    )
    return rows[: int(n_results)]
