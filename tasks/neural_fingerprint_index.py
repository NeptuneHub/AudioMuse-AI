# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neural fingerprint index: built by the worker, stored in ivf_dir, synced by Flask.

Every analysed track stores one 32-byte code per half second in
embedding.neural_fingerprint (tasks.neural_fingerprint). This index follows the
lifecycle of the other similarity indexes: the worker builds it at the rebuild
points of the analysis run, stores it in the ivf_dir table in one transaction it
commits itself, and publishes the index-reload event; the web process never
builds anything, it only syncs local files from what the worker stored and
memory-maps them. Only the structure goes to the table (about 4 bytes per row:
the cell centroids as int8, the track order and lengths, the rows of each cell);
the codes stay in their blobs. Everything is sized for a library of millions of
tracks: no step on either side holds more than one part (at most _PART_ROWS
rows) or one training sample in memory at a time.

Main Features:
* index rows: every NEURAL_FINGERPRINT_INDEX_STRIDE-th stored row of a track,
  at most _MAX_TRACK_ROWS of them (index_rows); the blobs keep every row and
  the alignment check reads them, so the stride only trades pack size and
  query work against recall on degraded clips
* two-level quantizer (train_quantizer): about sqrt(rows) cells, at most
  _MAX_CELLS, trained on at least _TRAIN_ROWS_PER_CELL rows per cell sampled
  100 per track from random tracks; past _SINGLE_LEVEL_CELLS the cells are
  grouped under sqrt(cells) coarse centroids and a row is assigned to the
  nearest cell of its nearest group, so assigning a million tracks costs
  minutes instead of the hours a flat search over tens of thousands of cells
  would take; a query still ranks the flat cell list exactly
* build_and_store_neural_fingerprint_index: full build or append, written as
  parts of at most _PART_ROWS rows (label_tracks yields them at track
  boundaries), each stored as it is labelled. A full build retrains the
  quantizer and rewrites every part; an append keeps the quantizer, labels
  only the tracks fingerprinted since the last build and adds parts. The
  quantizer is retrained when the library has grown
  NEURAL_FINGERPRINT_RETRAIN_GROWTH times since it was trained, when more than
  a tenth of the indexed tracks are gone, when the largest cell holds more
  than ten times the average, or when the codebook, the stride or the layout
  changed
* directory blob (ivf_dir, neural_fingerprint_index__ivf_dir): build id,
  codebook id, stride, the quantizer, ids and lengths in build order, per-cell
  sizes, part count, the library size the quantizer was trained on; a tiny
  neural_fingerprint_index__build row carries the build id alone so a
  freshness check costs one small read
* part blobs (neural_fingerprint_index__part<n>): the rows of that part
  grouped by cell, as local indices plus the part's row offset
* Flask side, the local pack (ephemeral, under IVF_DISK_CACHE_DIR, re-synced
  from Postgres at every start, never persisted): the codes ordered BY CELL
  (slab.u8) with, per slab position, the track (track.u32), the offset in the
  track (offset.u16) and the inverse norm of the decoded vector (norm.f16),
  40 bytes per indexed row. A query segment therefore reads its
  NEURAL_FINGERPRINT_NPROBE cells as contiguous runs and scores them with one
  table lookup per byte and a multiply. write_local_pack processes one part
  at a time: the part's positions are derived from its cell runs and the
  cells already filled, its tracks' blobs stream in 2M-row chunks that are
  sorted by position and written as contiguous runs, so RAM stays in the
  hundreds of megabytes whatever the library size. When the previous local
  pack is a prefix of the new build under the same quantizer (an append),
  every cell's old run is copied block-wise and only the new parts' blobs are
  fetched
* load_at_startup starts the sync and the mapping in the background when
  Flask boots (a large library's pack must never hold the web server's
  start); ensure_loaded syncs when the stored build id differs from the local
  one and memory-maps the files into ONE immutable Pack swapped by a single
  reference assignment, so a query that started before a reload keeps its own
  consistent pack
* a track whose fingerprint row is gone since the build (cleaning deleted it)
  is written into the local pack as dead rows and masked out of every vote
  and the track count; the next full build drops it for good
* Search: the clip's segments are scored in parallel threads
  (NEURAL_FINGERPRINT_QUERY_THREADS, 0 = one per core); every third segment
  votes first and, when one track already leads the runner-up by
  _EARLY_EXIT_RATIO with at least _EARLY_EXIT_VOTES, the rest of the voting
  is skipped. Every neighbour votes for (track, offset) with its similarity,
  votes within one hop are pooled, and the best candidates are verified by the
  mean cosine between the whole clip and the track at that offset, read from
  the candidates' blobs in one query so the pack needs no row-to-position
  map. Rows carry item_id, score, votes, offset_seconds, identified and lead,
  the last two set by flag_identified over the full candidate list (the
  search-by-song merge reuses it). identify embeds a clip first;
  identify_vectors takes a fingerprint sequence directly and can leave given
  tracks out, which is how a stored song is searched for its other recordings
  without finding itself
* Per-server scope like every other index: the index holds the union of all
  servers, and a request scoped to a server votes only over that server's
  tracks through the shared availability mask (tasks.index_availability),
  cached per server and build for 30 s and dropped by
  invalidate_availability_cache when the mappings change
* picker_where gives the song picker of the Search by Song tab a server-side
  filter (songs that have a fingerprint) once the pack is loaded, and None
  before, so the picker never ships a list of every indexed id per keystroke
"""

import glob
import io
import logging
import math
import os
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np

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
_DIR_NAME = f'{INDEX_NAME}__ivf_dir'
_BUILD_NAME = f'{INDEX_NAME}__build'
_PART_PREFIX = f'{INDEX_NAME}__part'
_FILE_PREFIX = 'neural_fingerprint'
FORMAT = 4
_PACK_FORMAT = 5
_MAX_CELLS = 65536
_SINGLE_LEVEL_CELLS = 64
_MIN_TRAIN_ROWS = 2000
_TRAIN_ROWS_PER_CELL = 20
_TRAIN_ROWS_PER_TRACK = 100
_MAX_TRACK_ROWS = 65535
_ASSIGN_BATCH = 65536
_BLOB_CHUNK = 500
_PART_ROWS = 1 << 23
_SYNC_CHUNK_ROWS = 1 << 21
_NORM_CHUNK_ROWS = 1 << 18
_TOP_K = 8
_VERIFY = 20
_FIRST_PASS_EVERY = 3
_EARLY_EXIT_RATIO = 4.0
_EARLY_EXIT_VOTES = 3.0
_MAX_QUERY_THREADS = 8
_REMOVED_FRACTION = 0.1
_IMBALANCE = 10.0
_PART_MAGIC = b'NFPP'
_PART_HEADER = struct.Struct('<4sHIQQ')
_PACK_FILES = {'slab': 'slab.u8', 'track': 'track.u32', 'offset': 'offset.u16', 'norm': 'norm.f16', 'meta': 'meta.npz'}
_SUBSPACE_OFFSETS = (np.arange(PQ_SUBSPACES, dtype=np.int32) * PQ_CENTROIDS)[None, :]
PICKER_WHERE = (
    'EXISTS (SELECT 1 FROM embedding e WHERE e.item_id = score.item_id AND e.neural_fingerprint IS NOT NULL)',
    (),
)
_AVAILABILITY_CACHE = {}
_AVAILABILITY_CACHE_LOCK = threading.Lock()
_AVAILABILITY_CACHE_TTL = 30.0
_CANONICAL = {}
_LOCK = threading.RLock()
_STATE = {'pack': None, 'building': False, 'error': None, 'gpu': None, 'executor': None}


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
    dead: np.ndarray
    stride: int
    slab: np.ndarray
    track: np.ndarray
    offset: np.ndarray
    norm: np.ndarray

    @property
    def live_tracks(self):
        return int(self.ids.size - self.dead.size)


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


def pack_part(labels, n_cells, row_offset):
    counts = np.bincount(labels, minlength=n_cells).astype(np.int64)
    bounds = np.concatenate(([0], np.cumsum(counts))).astype('<i8')
    order = np.argsort(labels, kind='stable').astype('<i4')
    header = _PART_HEADER.pack(_PART_MAGIC, 1, int(n_cells), int(labels.size), int(row_offset))
    return header + bounds.tobytes() + order.tobytes(), counts


def unpack_part(blob):
    magic, version, n_cells, n_rows, row_offset = _PART_HEADER.unpack_from(blob)
    if magic != _PART_MAGIC or version != 1:
        raise ValueError('not a neural fingerprint index part')
    offset = _PART_HEADER.size
    bounds = np.frombuffer(blob, dtype='<i8', count=n_cells + 1, offset=offset)
    offset += bounds.nbytes
    order = np.frombuffer(blob, dtype='<i4', count=n_rows, offset=offset)
    return {'n_cells': int(n_cells), 'n_rows': int(n_rows), 'row_offset': int(row_offset), 'bounds': bounds, 'order': order}


def part_positions(part, bounds, filled):
    counts = np.diff(part['bounds'])
    labels = np.repeat(np.arange(counts.size), counts)
    dest = bounds[labels] + filled[labels] + (np.arange(part['n_rows'], dtype=np.int64) - part['bounds'][labels])
    positions = np.empty(part['n_rows'], dtype=np.int64)
    positions[part['order']] = dest
    return positions


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
        quantizer = Quantizer(
            _decode_i8(data['coarse']), _decode_i8(data['centroids']), data['group_offsets'].astype(np.int64),
        )
        return {
            'format': int(data['format']), 'build_id': str(data['build_id']), 'codebook_id': int(data['codebook_id']),
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


def iter_index_rows(conn, ids):
    for item_id, codes in iter_codes(conn, ids):
        yield item_id, index_rows(codes)


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
    ids, lengths, labels, pending = [], [], [], []
    pending_rows = chunk_rows = 0

    def flush():
        nonlocal pending, pending_rows
        if pending:
            labels.append(assign_cells(np.concatenate(pending), quantizer))
            pending, pending_rows = [], 0

    for item_id, codes in codes_iter:
        codes = index_rows(codes)
        if codes is None or not codes.shape[0]:
            continue
        if chunk_rows and chunk_rows + codes.shape[0] > part_rows:
            flush()
            yield ids, np.asarray(lengths, dtype=np.int64), np.concatenate(labels)
            ids, lengths, labels, chunk_rows = [], [], [], 0
        ids.append(item_id)
        lengths.append(int(codes.shape[0]))
        pending.append(codes)
        pending_rows += int(codes.shape[0])
        chunk_rows += int(codes.shape[0])
        if pending_rows >= _ASSIGN_BATCH:
            flush()
    flush()
    if ids:
        yield ids, np.asarray(lengths, dtype=np.int64), np.concatenate(labels)


def _load_directory(conn):
    blob = load_segmented_blob(conn, DIR_TABLE, _DIR_NAME)
    return unpack_directory(blob) if blob else None


def _stored_build_id(conn):
    blob = load_segmented_blob(conn, DIR_TABLE, _BUILD_NAME)
    return bytes(blob).decode('ascii') if blob else None


def _store_directory(conn, directory_blob, build_id):
    store_segmented_blob(conn, DIR_TABLE, _DIR_NAME, directory_blob)
    store_segmented_blob(conn, DIR_TABLE, _BUILD_NAME, build_id.encode('ascii'))


def _store_part(conn, index, blob):
    store_segmented_blob(conn, DIR_TABLE, f'{_PART_PREFIX}{index}', blob)


def _delete_parts(conn):
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM ivf_dir WHERE name LIKE %s ESCAPE '\\'",
            (_PART_PREFIX.replace('_', r'\_') + '%',),
        )
    finally:
        cur.close()


def _load_part(conn, index):
    blob = load_segmented_blob(conn, DIR_TABLE, f'{_PART_PREFIX}{index}')
    if blob is None:
        raise RuntimeError(f'Part {index} of the neural fingerprint index is missing; rebuild the indexes.')
    return unpack_part(bytes(blob))


def _store_labelled_parts(conn, chunks, n_cells, first_part, row_offset):
    ids, lengths, cell_sizes, parts = [], [], np.zeros(n_cells, dtype=np.int64), first_part
    for chunk_ids, chunk_lengths, labels in chunks:
        part_blob, counts = pack_part(labels, n_cells, row_offset)
        _store_part(conn, parts, part_blob)
        ids.extend(chunk_ids)
        lengths.append(chunk_lengths)
        cell_sizes += counts
        row_offset += int(labels.size)
        parts += 1
    joined = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
    return ids, joined, cell_sizes, parts


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
    _delete_parts(conn)
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
        int(directory['lengths'].sum()),
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


def _paths(build_id):
    directory = config.IVF_DISK_CACHE_DIR
    return {key: os.path.join(directory, f'{_FILE_PREFIX}.{build_id}.{name}') for key, name in _PACK_FILES.items()}


def _local_builds():
    pattern = os.path.join(config.IVF_DISK_CACHE_DIR, f'{_FILE_PREFIX}.*.{_PACK_FILES["meta"]}')
    found = []
    for path in glob.glob(pattern):
        try:
            with np.load(path, allow_pickle=False) as meta:
                if int(meta['format']) == _PACK_FORMAT:
                    found.append((os.path.getmtime(path), str(meta['build_id'])))
        except Exception:
            logger.exception('Unreadable neural fingerprint pack metadata %s', path)
    return [build_id for _mtime, build_id in sorted(found)]


def _dead_from(meta):
    if 'dead' not in meta.files:
        return np.zeros(0, dtype=np.int64)
    return meta['dead'].astype(np.int64)


def _local_meta(build_id):
    paths = _paths(build_id)
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    with np.load(paths['meta'], allow_pickle=False) as meta:
        return {
            'ids': [str(i) for i in meta['ids']], 'lengths': meta['lengths'].astype(np.int64),
            'dead': _dead_from(meta), 'centroids': meta['centroids'], 'cell_bounds': meta['cell_bounds'].astype(np.int64),
            'stride': int(meta['stride']), 'parts': int(meta['parts']),
        }


def _prune_local(keep_build_id):
    for build_id in _local_builds():
        if build_id == keep_build_id:
            continue
        for path in glob.glob(os.path.join(config.IVF_DISK_CACHE_DIR, f'{_FILE_PREFIX}.{build_id}.*')):
            try:
                os.remove(path)
            except OSError:
                logger.debug('Could not remove %s yet', path)


def _reusable_parts(previous, directory, ids, lengths):
    if previous is None or previous['stride'] != directory['stride']:
        return 0
    n = len(previous['ids'])
    if n > len(ids) or previous['ids'] != ids[:n] or not np.array_equal(previous['lengths'], lengths[:n]):
        return 0
    if not np.array_equal(previous['centroids'], directory['centroids']):
        return 0
    if previous['cell_bounds'].size != directory['cell_sizes'].size + 1:
        return 0
    return int(previous['parts'])


def _copy_cells(previous_paths, previous, bounds, files):
    old_bounds = previous['cell_bounds']
    kept_rows = int(previous['lengths'].sum())
    old = _open_files(previous_paths, kept_rows, 'r')
    for c in range(bounds.size - 1):
        a, b = int(old_bounds[c]), int(old_bounds[c + 1])
        if a == b:
            continue
        start = int(bounds[c])
        for key in ('slab', 'track', 'offset', 'norm'):
            files[key][start:start + (b - a)] = old[key][a:b]


def _open_files(paths, n_rows, mode):
    return {
        'slab': np.memmap(paths['slab'], dtype=np.uint8, mode=mode, shape=(n_rows, CODE_BYTES)),
        'track': np.memmap(paths['track'], dtype=np.uint32, mode=mode, shape=(n_rows,)),
        'offset': np.memmap(paths['offset'], dtype=np.uint16, mode=mode, shape=(n_rows,)),
        'norm': np.memmap(paths['norm'], dtype=np.float16, mode=mode, shape=(n_rows,)),
    }


def _squared_norm_table():
    return (2.0 * codebook()[2]).astype(np.float32).ravel()


def _inverse_norms(block, squared):
    out = np.empty(block.shape[0], dtype=np.float32)
    for a in range(0, block.shape[0], _NORM_CHUNK_ROWS):
        piece = block[a:a + _NORM_CHUNK_ROWS]
        total = np.take(squared, piece.astype(np.int32) + _SUBSPACE_OFFSETS).sum(axis=1)
        out[a:a + piece.shape[0]] = 1.0 / (np.sqrt(total) + 1e-9)
    return out


def _scatter(files, positions, block, tracks, offsets, squared):
    order = np.argsort(positions, kind='stable')
    p_sorted = positions[order]
    values = {
        'slab': block[order], 'track': tracks[order], 'offset': offsets[order],
        'norm': _inverse_norms(block, squared)[order],
    }
    breaks = np.flatnonzero(np.diff(p_sorted) != 1) + 1
    for a, b in zip(np.concatenate(([0], breaks)), np.concatenate((breaks, [p_sorted.size]))):
        start = int(p_sorted[a])
        for key, array in values.items():
            files[key][start:start + (b - a)] = array[a:b]


class _PartWriter:
    def __init__(self, files, positions, first_track, squared):
        self.files, self.positions, self.squared = files, positions, squared
        self.track = first_track
        self.local = 0
        self.pending, self.pending_tracks, self.pending_rows = [], [], 0
        self.dead = []

    def add(self, item_id, codes, expected):
        if codes is None:
            codes = np.zeros((expected, CODE_BYTES), dtype=np.uint8)
            self.dead.append(self.track)
        elif codes.shape[0] != expected:
            raise RuntimeError(
                f'The fingerprint of {item_id} no longer matches the index ({expected} rows expected); rebuild the indexes.'
            )
        self.pending.append(np.ascontiguousarray(codes, dtype=np.uint8))
        self.pending_tracks.append((self.track, expected))
        self.pending_rows += expected
        self.track += 1
        if self.pending_rows >= _SYNC_CHUNK_ROWS:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        tracks = np.concatenate([np.full(n, t, dtype=np.uint32) for t, n in self.pending_tracks])
        offsets = np.concatenate([np.arange(n, dtype=np.uint16) for _t, n in self.pending_tracks])
        block = np.concatenate(self.pending)
        _scatter(self.files, self.positions[self.local:self.local + block.shape[0]], block, tracks, offsets, self.squared)
        self.local += block.shape[0]
        self.pending, self.pending_tracks, self.pending_rows = [], [], 0


def _tracks_of_part(lengths, first_track, n_rows):
    total, track = 0, first_track
    while total < n_rows:
        if track >= lengths.size:
            raise RuntimeError('The neural fingerprint index parts hold more rows than the directory lists; rebuild the indexes.')
        total += int(lengths[track])
        track += 1
    if total != n_rows:
        raise RuntimeError('A neural fingerprint index part does not end on a track boundary; rebuild the indexes.')
    return track


def _fill_pack(files, directory, ids, lengths, bounds, part_at, codes_iter, keep_parts, previous_paths, previous):
    n_rows = int(lengths.sum())
    filled = np.zeros(bounds.size - 1, dtype=np.int64)
    dead, track, next_row = [], 0, 0
    if keep_parts:
        _copy_cells(previous_paths, previous, bounds, files)
        filled += np.diff(previous['cell_bounds'])
        track, next_row = len(previous['ids']), int(previous['lengths'].sum())
        dead = [int(index) for index in previous['dead']]
    squared = _squared_norm_table()
    for index in range(keep_parts, int(directory['parts'])):
        part = part_at(index)
        if part['row_offset'] != next_row or part['n_cells'] != filled.size:
            raise RuntimeError(f'Part {index} of the neural fingerprint index does not follow the previous one; rebuild the indexes.')
        last_track = _tracks_of_part(lengths, track, part['n_rows'])
        writer = _PartWriter(files, part_positions(part, bounds, filled), track, squared)
        for item_id, codes in zip(ids[track:last_track], (codes for _id, codes in codes_iter(ids[track:last_track]))):
            writer.add(item_id, index_rows(codes), int(lengths[writer.track]))
        writer.flush()
        if writer.local != part['n_rows'] or writer.track != last_track:
            raise RuntimeError(f'The blobs delivered {writer.local} rows for part {index} of {part["n_rows"]}; rebuild the indexes.')
        dead.extend(writer.dead)
        filled += np.diff(part['bounds'])
        track, next_row = last_track, next_row + part['n_rows']
    if track != len(ids) or next_row != n_rows:
        raise RuntimeError(f'The neural fingerprint index parts cover {next_row} rows of {n_rows}; rebuild the indexes.')
    return dead


def _release(files):
    for handle in list(files.values()):
        handle.flush()
    files.clear()


def write_local_pack(paths, directory, parts, codes_iter, previous_paths=None, previous=None):
    os.makedirs(os.path.dirname(paths['slab']), exist_ok=True)
    started = time.time()
    part_at = parts.__getitem__ if isinstance(parts, (list, tuple)) else parts
    ids, lengths = list(directory['ids']), directory['lengths'].astype(np.int64)
    n_rows = int(lengths.sum())
    if n_rows <= 0:
        raise RuntimeError('The neural fingerprint index is empty; rebuild the indexes.')
    bounds = np.concatenate(([0], np.cumsum(directory['cell_sizes']))).astype(np.int64)
    if int(bounds[-1]) != n_rows:
        raise RuntimeError(f'The neural fingerprint index cells hold {int(bounds[-1])} rows for {n_rows} stored; rebuild the indexes.')
    tmp = {key: path + '.tmp' for key, path in paths.items()}
    tmp['meta'] = paths['meta'] + '.tmp.npz'
    keep_parts = _reusable_parts(previous, directory, ids, lengths) if previous_paths else 0
    files = _open_files(tmp, n_rows, 'w+')
    try:
        dead = _fill_pack(files, directory, ids, lengths, bounds, part_at, codes_iter, keep_parts, previous_paths, previous)
    finally:
        _release(files)
    np.savez(
        tmp['meta'], lengths=lengths, ids=np.asarray(ids, dtype=str), centroids=directory['centroids'],
        cell_bounds=bounds, build_id=np.asarray(directory['build_id']), codebook_id=np.uint32(directory['codebook_id']),
        format=np.int64(_PACK_FORMAT), stride=np.int64(directory['stride']), parts=np.int64(directory['parts']),
        dead=np.asarray(dead, dtype=np.int64),
    )
    for key in _PACK_FILES:
        os.replace(tmp[key], paths[key])
    logger.info(
        'Neural fingerprint pack synced: %d tracks, %d rows, %d cells, %d parts reused of %d, '
        '%d tracks gone since the build (masked until the next full build), %.0fs',
        len(ids), n_rows, bounds.size - 1, keep_parts, int(directory['parts']), len(dead), time.time() - started,
    )


def _sync_from_db(conn, directory):
    paths = _paths(directory['build_id'])
    previous_id = next(iter(reversed(_local_builds())), None)
    previous_paths = _paths(previous_id) if previous_id else None
    previous = _local_meta(previous_id) if previous_id else None
    write_local_pack(
        paths, directory, lambda index: _load_part(conn, index), lambda ids: iter_codes(conn, ids),
        previous_paths, previous,
    )
    return paths


def _open_pack(paths):
    with np.load(paths['meta'], allow_pickle=False) as meta:
        lengths = meta['lengths'].astype(np.int64)
        files = _open_files(paths, int(lengths.sum()), 'r')
        pack = Pack(
            build_id=str(meta['build_id']), ids=meta['ids'], lengths=lengths, centroids=meta['centroids'],
            cell_bounds=meta['cell_bounds'].astype(np.int64), dead=_dead_from(meta), stride=int(meta['stride']),
            slab=files['slab'], track=files['track'], offset=files['offset'], norm=files['norm'],
        )
    _STATE['pack'] = pack
    return pack


def is_loaded():
    return _STATE['pack'] is not None


def _current_pack():
    pack = _STATE['pack']
    if pack is None:
        raise RuntimeError('The neural fingerprint index is not loaded.')
    return pack


def picker_where():
    return PICKER_WHERE if is_loaded() else None


def _local_pack_for(conn):
    build_id = _stored_build_id(conn)
    if build_id is None:
        raise RuntimeError('No neural fingerprint index is built yet. Run the analysis, which builds it.')
    if _local_meta(build_id) is not None:
        return _paths(build_id)
    directory = _load_directory(conn)
    if directory is None or directory['build_id'] != build_id:
        raise RuntimeError('The neural fingerprint index is being written; try again in a minute.')
    if directory['codebook_id'] != codebook()[1]:
        raise RuntimeError('The neural fingerprint index was built with another codebook; rebuild the indexes.')
    return _sync_from_db(conn, directory)


def ensure_loaded():
    from database import connect_raw

    with _LOCK:
        if is_loaded():
            return True
        if _STATE['building']:
            raise RuntimeError('The neural fingerprint index is being prepared; try again in a minute.')
        _STATE['building'] = True
        _STATE['error'] = None
    try:
        conn = connect_raw(application_name='neural_fingerprint_index')
        try:
            paths = _local_pack_for(conn)
        finally:
            conn.close()
        with _LOCK:
            pack = _open_pack(paths)
        _prune_local(pack.build_id)
        logger.info('Neural fingerprint pack loaded: %d tracks (build %s)', pack.live_tracks, pack.build_id)
        return True
    except Exception as exc:
        with _LOCK:
            _STATE['error'] = str(exc)
        logger.exception('Neural fingerprint pack could not be loaded')
        raise
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
        if _local_meta(build_id) is None:
            directory = _load_directory(conn)
            if directory is None or directory['build_id'] != build_id:
                return False
            _sync_from_db(conn, directory)
        with _LOCK:
            if is_loaded():
                _open_pack(_paths(build_id))
                logger.info('Neural fingerprint pack swapped to build %s', build_id)
        _prune_local(build_id)
        return True
    finally:
        conn.close()


def load_at_startup():
    from database import connect_raw

    if not is_enabled():
        logger.info('Neural fingerprint index not loaded at startup: NEURAL_FINGERPRINT_ENABLED is false')
        return None
    if not is_available():
        logger.info('Neural fingerprint index not loaded at startup: the model or the codebook is missing')
        return None
    conn = connect_raw(application_name='neural_fingerprint_index')
    try:
        if _stored_build_id(conn) is None:
            return None
    finally:
        conn.close()
    return start_background_load()


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
    return was_loaded


def get_status():
    with _LOCK:
        pack = _STATE['pack']
        return {
            'available': bool(is_available()),
            'loaded': pack is not None,
            'synced': pack is not None or bool(_local_builds()),
            'building': bool(_STATE['building']),
            'tracks': pack.live_tracks if pack is not None else 0,
            'error': _STATE['error'],
        }


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


def _cell_scores(pack, table, b0, b1, allowed):
    codes = pack.slab[b0:b1]
    tracks = np.asarray(pack.track[b0:b1])
    offsets = np.asarray(pack.offset[b0:b1])
    norms = np.asarray(pack.norm[b0:b1])
    if allowed is not None:
        keep = allowed[tracks]
        if not keep.any():
            return None
        if not keep.all():
            codes, tracks, offsets, norms = codes[keep], tracks[keep], offsets[keep], norms[keep]
    sims = np.take(table, codes.astype(np.int32) + _SUBSPACE_OFFSETS).sum(axis=1) * norms.astype(np.float32)
    return sims, tracks, offsets


def _segment_votes(pack, book, query_vectors, qi, nprobe, allowed):
    table = _lookup_table(book, query_vectors[qi])
    bounds = pack.cell_bounds
    pieces = []
    for cell in _probe_cells(pack, query_vectors[qi], nprobe):
        piece = _cell_scores(pack, table, int(bounds[cell]), int(bounds[cell + 1]), allowed)
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
    if len(segments) > 1 and _query_threads() > 1:
        return list(_executor().map(lambda qi: _segment_votes(pack, book, query_vectors, qi, nprobe, allowed), segments))
    return [_segment_votes(pack, book, query_vectors, qi, nprobe, allowed) for qi in segments]


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


def _db_connection():
    from flask import has_app_context

    if has_app_context():
        from database import get_db

        return get_db(), False
    from database import connect_raw

    return connect_raw(application_name='neural_fingerprint_verify'), True


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
        raise RuntimeError('The neural fingerprint model is not available here.')
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
    if not exclude_ids and not pack.dead.size:
        return allowed
    allowed = np.ones(pack.ids.size, dtype=np.bool_) if allowed is None else allowed.copy()
    allowed[pack.dead] = False
    if exclude_ids:
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
