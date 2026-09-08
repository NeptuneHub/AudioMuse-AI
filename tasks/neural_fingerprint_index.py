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
points of the analysis run, stores it in the ivf_dir table, and publishes the
index-reload event; the web process never builds anything, it only syncs local
files from what the worker stored and memory-maps them. Only the structure
goes to the table: the centroids (int8), the track order and lengths, and the
cell lists (one int32 per row); the codes stay in their blobs, so the table
holds about 4 bytes per row instead of a second copy of the fingerprints.

Main Features:
* build_and_store_neural_fingerprint_index: full build or append. A full
  build trains about sqrt(rows) centroids (at most 8192) on at most
  NEURAL_FINGERPRINT_TRAIN_ROWS rows sampled 100 per track from random tracks,
  assigns every row and writes part 0. An append keeps the centroids, reads
  only the tracks fingerprinted since the last build, assigns their rows and
  writes one more part; fingerprints never change and the library only grows,
  so this is exact, not an approximation. The centroids are retrained (a full
  build) when the library has grown NEURAL_FINGERPRINT_RETRAIN_GROWTH times
  since they were trained, when more than a tenth of the indexed tracks are
  gone, when the largest cell holds more than ten times the average, or when
  the codebook or the layout changed
* directory blob (ivf_dir, neural_fingerprint_index__ivf_dir): build id,
  codebook id, centroids, ids and lengths in build order, per-cell sizes,
  part count, the library size the centroids were trained on; a tiny
  neural_fingerprint_index__build row carries the build id alone so a
  freshness check costs one small read
* part blobs (neural_fingerprint_index__part<n>): the rows of that build
  grouped by cell, as local indices plus the part's row offset, split by
  store_segmented_blob under IVF_MAX_PART_SIZE_MB
* the build is one transaction committed by the builder itself (like the
  audio IVF builder), so a web process syncing at any moment sees either the
  previous build complete or the new one complete, never a directory whose
  parts are half written
* Flask side, the local pack (ephemeral, under IVF_DISK_CACHE_DIR, re-synced
  from Postgres at every start, never persisted): write_local_pack streams the
  blobs in track order and scatters every row to its place in a slab ordered
  BY CELL (slab.u8), so a query reads NEURAL_FINGERPRINT_NPROBE contiguous
  runs instead of hundreds of thousands of scattered 32-byte rows, which is
  what keeps a cold query at a million tracks in seconds rather than hours;
  next to it, per slab position, the track (track.i32), the row id (row.i32)
  and the inverse norm of the decoded vector (norm.f16, so scoring is one
  table lookup per byte and a multiply), plus the slab position of every row
  (pos.i32) for the alignment check. The scatter works on 2M-row chunks of
  the stream: sorted by position, the rows of one cell form one contiguous
  run, so writes are large and sequential and RAM stays bounded whatever the
  library size. When the previous local pack is a prefix of the new build
  under the same centroids (an append), every cell's old run is copied
  block-wise into the new slab and only the new tracks' blobs are fetched
* load_at_startup starts the sync and the mapping in the background when
  Flask boots (a large library's pack must never hold the web server's
  start; it says when no build exists yet); ensure_loaded syncs when the
  stored build id differs from the local one and memory-maps the files into
  ONE immutable Pack that is swapped by a single reference assignment, so a
  query that started before a reload keeps its own consistent pack; the pack
  then stays mapped for the life of the process (the recording search's idle
  timer releases only the encoder session)
* a track whose fingerprint row is gone since the build (cleaning deleted it)
  is written into the local pack as dead rows and masked out of every vote
  and the track count, instead of failing the whole sync; the next full
  build (past the removed-fraction rule) drops it for good
* Search: each query segment scores the rows of its nearest cells through
  the codebook lookup table, every neighbour votes for (track, offset) with
  its similarity, votes within one hop are pooled, and the best candidates
  are verified by the mean cosine between the whole clip and the track at
  that offset; rows carry item_id, score, votes, offset_seconds, identified
  and lead, the last two set by flag_identified over the full candidate list
  (the search-by-song merge reuses it). identify embeds a clip first;
  identify_vectors takes a fingerprint sequence directly and can leave given
  tracks out, which is how a stored song is searched for its other
  recordings without finding itself
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
FORMAT = 3
_MAX_CELLS = 8192
_MIN_TRAIN_ROWS = 2000
_TRAIN_ROWS_PER_TRACK = 100
_ASSIGN_BATCH = 65536
_BLOB_CHUNK = 500
_TOP_K = 8
_VERIFY = 20
_REMOVED_FRACTION = 0.1
_IMBALANCE = 10.0
_PART_MAGIC = b'NFPP'
_PART_HEADER = struct.Struct('<4sHIQQ')
_PACK_FORMAT = 4
_PACK_FILES = {
    'slab': 'slab.u8', 'track': 'track.i32', 'row': 'row.i32', 'norm': 'norm.f16', 'pos': 'pos.i32', 'meta': 'meta.npz',
}
_SYNC_CHUNK_ROWS = 1 << 21
_NORM_CHUNK_ROWS = 1 << 18
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
_STATE = {'pack': None, 'building': False, 'error': None, 'gpu': None}


class Pack(NamedTuple):
    build_id: str
    ids: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    centroids: np.ndarray
    cell_bounds: np.ndarray
    dead: np.ndarray
    slab: np.ndarray
    track: np.ndarray
    row: np.ndarray
    norm: np.ndarray
    pos: np.ndarray

    @property
    def live_tracks(self):
        return int(self.ids.size - self.dead.size)


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


def assign_cells(codes, centroids):
    labels = np.empty(codes.shape[0], dtype=np.int32)
    for start in range(0, codes.shape[0], _ASSIGN_BATCH):
        block = decode_codes(codes[start:start + _ASSIGN_BATCH])
        labels[start:start + _ASSIGN_BATCH] = _nearest_centroids(block, centroids)
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


def merge_parts(parts, n_cells):
    counts = np.zeros(n_cells, dtype=np.int64)
    for part in parts:
        counts += np.diff(part['bounds'])
    bounds = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    cell_rows = np.empty(int(counts.sum()), dtype=np.int32)
    filled = np.zeros(n_cells, dtype=np.int64)
    for part in parts:
        part_counts = np.diff(part['bounds'])
        labels = np.repeat(np.arange(n_cells), part_counts)
        rank = np.arange(part['n_rows'], dtype=np.int64) - part['bounds'][labels]
        dest = bounds[labels] + filled[labels] + rank
        cell_rows[dest] = part['order'].astype(np.int64) + part['row_offset']
        filled += part_counts
    return cell_rows, bounds


def pack_directory(build_id, codebook_id, centroids, ids, lengths, cell_sizes, parts, trained_tracks):
    buffer = io.BytesIO()
    np.savez(
        buffer, format=np.int64(FORMAT), build_id=np.asarray(str(build_id)), codebook_id=np.uint32(codebook_id),
        centroids=ivf_quant.encode_vectors(np.asarray(centroids, dtype=np.float32), ivf_quant.DTYPE_I8),
        ids=np.asarray(list(ids), dtype=str), lengths=np.asarray(lengths, dtype=np.int32),
        cell_sizes=np.asarray(cell_sizes, dtype=np.int64), parts=np.int64(parts),
        trained_tracks=np.int64(trained_tracks),
    )
    return buffer.getvalue()


def unpack_directory(blob):
    with np.load(io.BytesIO(bytes(blob)), allow_pickle=False) as data:
        centroids = _unit_rows(data['centroids'].astype(np.float32) / ivf_quant.I8_SCALE)
        return {
            'format': int(data['format']), 'build_id': str(data['build_id']), 'codebook_id': int(data['codebook_id']),
            'centroids': centroids, 'ids': [str(i) for i in data['ids']], 'lengths': data['lengths'].astype(np.int64),
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
        return int(cur.fetchone()[0] or 0)
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


def label_tracks(codes_iter, centroids):
    ids, lengths, labels, pending, pending_rows = [], [], [], [], 0
    for item_id, codes in codes_iter:
        if codes is None or not codes.shape[0]:
            continue
        ids.append(item_id)
        lengths.append(int(codes.shape[0]))
        pending.append(codes)
        pending_rows += int(codes.shape[0])
        if pending_rows >= _ASSIGN_BATCH:
            labels.append(assign_cells(np.concatenate(pending), centroids))
            pending, pending_rows = [], 0
    if pending:
        labels.append(assign_cells(np.concatenate(pending), centroids))
    joined = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int32)
    return ids, np.asarray(lengths, dtype=np.int64), joined


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


def _load_parts(conn, count):
    parts = []
    for index in range(count):
        blob = load_segmented_blob(conn, DIR_TABLE, f'{_PART_PREFIX}{index}')
        if blob is None:
            raise RuntimeError(f'Part {index} of the neural fingerprint index is missing; rebuild the indexes.')
        parts.append(unpack_part(bytes(blob)))
    return parts


def _full_build(conn, ids, codebook_id, reason):
    started = time.time()
    rng = np.random.default_rng(0)
    cap = max(_MIN_TRAIN_ROWS, int(config.NEURAL_FINGERPRINT_TRAIN_ROWS))
    n_cells = _cell_count(_estimated_rows(conn))
    sample_tracks = min(len(ids), max(1, math.ceil(cap / _TRAIN_ROWS_PER_TRACK)))
    chosen = [ids[i] for i in np.sort(rng.choice(len(ids), sample_tracks, replace=False))]
    sample = training_sample(iter_codes(conn, chosen), cap, rng)
    centroids = train_centroids(sample, n_cells)
    n_cells = int(centroids.shape[0])
    trained = time.time()
    kept_ids, lengths, labels = label_tracks(iter_codes(conn, ids), centroids)
    assigned = time.time()
    part_blob, counts = pack_part(labels, n_cells, 0)
    build_id = uuid.uuid4().hex
    _delete_parts(conn)
    _store_part(conn, 0, part_blob)
    _store_directory(
        conn, pack_directory(build_id, codebook_id, centroids, kept_ids, lengths, counts, 1, len(kept_ids)), build_id,
    )
    logger.info(
        'Neural fingerprint index built from scratch (%s): %d tracks, %d rows, %d cells; '
        'centroids from %d rows in %.0fs, rows assigned in %.0fs, stored in %.0fs',
        reason, len(kept_ids), int(labels.size), n_cells, int(sample.shape[0]), trained - started,
        assigned - trained, time.time() - assigned,
    )


def _incremental_build(conn, directory, ids):
    started = time.time()
    known = set(directory['ids'])
    new_ids = [item_id for item_id in ids if item_id not in known]
    if not new_ids:
        logger.info('Neural fingerprint index is up to date (%d tracks)', len(directory['ids']))
        return
    n_cells = int(directory['centroids'].shape[0])
    kept_ids, lengths, labels = label_tracks(iter_codes(conn, new_ids), directory['centroids'])
    if not kept_ids:
        logger.info('Neural fingerprint index: the %d new tracks carry no usable fingerprint', len(new_ids))
        return
    row_offset = int(directory['lengths'].sum())
    part_blob, counts = pack_part(labels, n_cells, row_offset)
    build_id = uuid.uuid4().hex
    _store_part(conn, directory['parts'], part_blob)
    _store_directory(
        conn,
        pack_directory(
            build_id, directory['codebook_id'], directory['centroids'], list(directory['ids']) + kept_ids,
            np.concatenate([directory['lengths'], lengths]), directory['cell_sizes'] + counts,
            directory['parts'] + 1, directory['trained_tracks'],
        ),
        build_id,
    )
    logger.info(
        'Neural fingerprint index appended: %d new tracks, %d rows, part %d, %.0fs (%d tracks indexed)',
        len(kept_ids), int(labels.size), directory['parts'], time.time() - started, len(directory['ids']) + len(kept_ids),
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


def _reusable_prefix(previous, ids, lengths):
    if previous is None:
        return 0
    n = min(len(previous['ids']), len(ids))
    if previous['ids'][:n] != ids[:n] or not np.array_equal(previous['lengths'][:n], lengths[:n]):
        return 0
    return n


def _slab_reusable(previous, previous_paths, directory, keep, pos, bounds):
    if keep != len(previous['ids']) or not np.array_equal(previous['centroids'], directory['centroids']):
        return False
    old_bounds = previous['cell_bounds']
    if old_bounds.size != bounds.size:
        return False
    kept_rows = int(previous['lengths'].sum())
    old_row = np.memmap(previous_paths['row'], dtype=np.int32, mode='r', shape=(kept_rows,))
    for c in range(bounds.size - 1):
        a, b = int(old_bounds[c]), int(old_bounds[c + 1])
        if a == b:
            continue
        if pos[old_row[a]] != bounds[c] or pos[old_row[b - 1]] != bounds[c] + (b - a) - 1:
            return False
    return True


def _copy_cells(previous_paths, previous, bounds, slab, norm):
    old_bounds = previous['cell_bounds']
    kept_rows = int(previous['lengths'].sum())
    old_slab = np.memmap(previous_paths['slab'], dtype=np.uint8, mode='r', shape=(kept_rows, CODE_BYTES))
    old_norm = np.memmap(previous_paths['norm'], dtype=np.float16, mode='r', shape=(kept_rows,))
    for c in range(bounds.size - 1):
        a, b = int(old_bounds[c]), int(old_bounds[c + 1])
        if a == b:
            continue
        start = int(bounds[c])
        slab[start:start + (b - a)] = old_slab[a:b]
        norm[start:start + (b - a)] = old_norm[a:b]


def _write_tracks(path, cell_rows, starts):
    track = np.memmap(path, dtype=np.int32, mode='w+', shape=(cell_rows.size,))
    for p0 in range(0, cell_rows.size, _SYNC_CHUNK_ROWS):
        p1 = min(cell_rows.size, p0 + _SYNC_CHUNK_ROWS)
        track[p0:p1] = np.searchsorted(starts, cell_rows[p0:p1], side='right') - 1
    track.flush()
    del track


def _squared_norm_table():
    return (2.0 * codebook()[2]).astype(np.float32).ravel()


def _inverse_norms(block, squared):
    out = np.empty(block.shape[0], dtype=np.float32)
    for a in range(0, block.shape[0], _NORM_CHUNK_ROWS):
        piece = block[a:a + _NORM_CHUNK_ROWS]
        total = np.take(squared, piece.astype(np.int32) + _SUBSPACE_OFFSETS).sum(axis=1)
        out[a:a + piece.shape[0]] = 1.0 / (np.sqrt(total) + 1e-9)
    return out


def _scatter(slab, norm, pos, first_row, block, squared):
    p = pos[first_row:first_row + block.shape[0]]
    order = np.argsort(p, kind='stable')
    p_sorted = p[order]
    block = block[order]
    inverse = _inverse_norms(block, squared)
    breaks = np.flatnonzero(np.diff(p_sorted) != 1) + 1
    for a, b in zip(np.concatenate(([0], breaks)), np.concatenate((breaks, [p_sorted.size]))):
        start = int(p_sorted[a])
        slab[start:start + (b - a)] = block[a:b]
        norm[start:start + (b - a)] = inverse[a:b]


def write_local_pack(paths, directory, parts, codes_iter, previous_paths=None, previous=None):
    os.makedirs(os.path.dirname(paths['slab']), exist_ok=True)
    started = time.time()
    ids, lengths = list(directory['ids']), directory['lengths'].astype(np.int64)
    n_rows = int(lengths.sum())
    starts = np.zeros(lengths.size, dtype=np.int64)
    if lengths.size:
        starts[1:] = np.cumsum(lengths)[:-1]
    n_cells = int(directory['centroids'].shape[0])
    cell_rows, bounds = merge_parts(parts, n_cells)
    if cell_rows.size != n_rows:
        raise RuntimeError(f'The neural fingerprint index parts hold {cell_rows.size} rows for {n_rows} stored; rebuild the indexes.')
    tmp = {key: path + '.tmp' for key, path in paths.items()}
    tmp['meta'] = paths['meta'] + '.tmp.npz'
    pos = np.empty(n_rows, dtype=np.int32)
    pos[cell_rows] = np.arange(n_rows, dtype=np.int32)
    keep = _reusable_prefix(previous, ids, lengths) if previous_paths else 0
    if keep and not _slab_reusable(previous, previous_paths, directory, keep, pos, bounds):
        keep = 0
    dead = [int(index) for index in previous['dead'] if index < keep] if keep else []
    pos.tofile(tmp['pos'])
    cell_rows.astype(np.int32).tofile(tmp['row'])
    _write_tracks(tmp['track'], cell_rows, starts)
    del cell_rows
    slab = np.memmap(tmp['slab'], dtype=np.uint8, mode='w+', shape=(n_rows, CODE_BYTES))
    norm = np.memmap(tmp['norm'], dtype=np.float16, mode='w+', shape=(n_rows,))
    if keep:
        _copy_cells(previous_paths, previous, bounds, slab, norm)
    squared = _squared_norm_table()
    next_row = int(lengths[:keep].sum())
    pending, pending_rows = [], 0
    for index, (item_id, codes) in enumerate(codes_iter(ids[keep:]), start=keep):
        expected = int(lengths[index])
        if codes is None:
            codes = np.zeros((expected, CODE_BYTES), dtype=np.uint8)
            dead.append(index)
        elif codes.shape[0] != expected:
            raise RuntimeError(
                f'The fingerprint of {item_id} no longer matches the index ({expected} rows expected); rebuild the indexes.'
            )
        pending.append(np.ascontiguousarray(codes, dtype=np.uint8))
        pending_rows += expected
        if pending_rows >= _SYNC_CHUNK_ROWS:
            _scatter(slab, norm, pos, next_row, np.concatenate(pending), squared)
            next_row += pending_rows
            pending, pending_rows = [], 0
    if pending:
        _scatter(slab, norm, pos, next_row, np.concatenate(pending), squared)
        next_row += pending_rows
    if next_row != n_rows:
        raise RuntimeError(f'The fingerprint blobs delivered {next_row} rows for {n_rows} in the index; rebuild the indexes.')
    slab.flush()
    norm.flush()
    del slab, norm, pos
    np.savez(
        tmp['meta'], starts=starts, lengths=lengths, ids=np.asarray(ids, dtype=str),
        centroids=directory['centroids'], cell_bounds=bounds, build_id=np.asarray(directory['build_id']),
        codebook_id=np.uint32(directory['codebook_id']), format=np.int64(_PACK_FORMAT),
        dead=np.asarray(dead, dtype=np.int64),
    )
    for key in _PACK_FILES:
        os.replace(tmp[key], paths[key])
    logger.info(
        'Neural fingerprint pack synced: %d tracks, %d rows, %d cells, %d tracks reused, '
        '%d tracks gone since the build (masked until the next full build), %.0fs',
        len(ids), n_rows, n_cells, keep, len(dead), time.time() - started,
    )


def _sync_from_db(conn, directory):
    paths = _paths(directory['build_id'])
    previous_id = next(iter(reversed(_local_builds())), None)
    previous_paths = _paths(previous_id) if previous_id else None
    previous = _local_meta(previous_id) if previous_id else None
    parts = _load_parts(conn, directory['parts'])
    write_local_pack(paths, directory, parts, lambda ids: iter_codes(conn, ids), previous_paths, previous)
    return paths


def _open_pack(paths):
    with np.load(paths['meta'], allow_pickle=False) as meta:
        lengths = meta['lengths'].astype(np.int64)
        n_rows = int(lengths.sum())
        pack = Pack(
            build_id=str(meta['build_id']), ids=meta['ids'], starts=meta['starts'].astype(np.int64), lengths=lengths,
            centroids=meta['centroids'], cell_bounds=meta['cell_bounds'].astype(np.int64), dead=_dead_from(meta),
            slab=np.memmap(paths['slab'], dtype=np.uint8, mode='r', shape=(n_rows, CODE_BYTES)),
            track=np.memmap(paths['track'], dtype=np.int32, mode='r', shape=(n_rows,)),
            row=np.memmap(paths['row'], dtype=np.int32, mode='r', shape=(n_rows,)),
            norm=np.memmap(paths['norm'], dtype=np.float16, mode='r', shape=(n_rows,)),
            pos=np.memmap(paths['pos'], dtype=np.int32, mode='r', shape=(n_rows,)),
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
    rows = np.asarray(pack.row[b0:b1])
    norms = np.asarray(pack.norm[b0:b1])
    if allowed is not None:
        keep = allowed[tracks]
        if not keep.any():
            return None
        if not keep.all():
            codes, tracks, rows, norms = codes[keep], tracks[keep], rows[keep], norms[keep]
    sims = np.take(table, codes.astype(np.int32) + _SUBSPACE_OFFSETS).sum(axis=1) * norms.astype(np.float32)
    return sims, tracks, rows


def _vote(pack, book, query_vectors, nprobe, allowed=None):
    starts, bounds = pack.starts, pack.cell_bounds
    votes = {}
    for qi, query in enumerate(query_vectors):
        table = _lookup_table(book, query)
        pieces = []
        for cell in _probe_cells(pack, query, nprobe):
            piece = _cell_scores(pack, table, int(bounds[cell]), int(bounds[cell + 1]), allowed)
            if piece is not None:
                pieces.append(piece)
        if not pieces:
            continue
        sims = np.concatenate([piece[0] for piece in pieces])
        tracks = np.concatenate([piece[1] for piece in pieces])
        rows = np.concatenate([piece[2] for piece in pieces])
        top = np.argpartition(-sims, _TOP_K)[:_TOP_K] if sims.size > _TOP_K else np.arange(sims.size)
        for j in top:
            track = int(tracks[j])
            key = (track, int(rows[j] - starts[track]) - qi)
            votes[key] = votes.get(key, 0.0) + float(sims[j])
    pooled = {}
    for (track, offset), value in votes.items():
        total = value + votes.get((track, offset - 1), 0.0) + votes.get((track, offset + 1), 0.0)
        if total > pooled.get(track, (0.0, 0))[0]:
            pooled[track] = (total, offset)
    return pooled


def _verify(pack, query_vectors, track, offset):
    length = int(pack.lengths[track])
    base = int(pack.starts[track])
    best = (-1.0, offset)
    for candidate in (offset - 1, offset, offset + 1):
        lo = max(0, -candidate)
        hi = min(query_vectors.shape[0], length - candidate)
        if hi - lo < 1:
            continue
        positions = np.asarray(pack.pos[base + candidate + lo: base + candidate + hi])
        rows = decode_codes(pack.slab[positions])
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
    pooled = _vote(pack, codebook()[0], query_vectors, nprobe, _allowed_tracks(pack, exclude_ids))
    ranked = sorted(pooled.items(), key=lambda kv: -kv[1][0])[:max(int(n_results), _VERIFY)]
    scored = []
    for track, (votes, offset) in ranked:
        score, aligned = _verify(pack, query_vectors, track, offset)
        scored.append((track, score, votes, aligned))
    scored.sort(key=lambda row: -row[1])
    rows = flag_identified([
        {
            'item_id': str(pack.ids[track]),
            'score': round(float(score), 3),
            'votes': round(float(votes), 2),
            'offset_seconds': round(float(aligned) * HOP_SECONDS, 1),
        }
        for track, score, votes, aligned in scored
    ])
    logger.info(
        'Neural fingerprint identify: %d query segments, %d candidates, %.1fs, best score %.3f lead %s identified %s',
        query_vectors.shape[0], len(pooled), time.time() - started, rows[0]['score'] if rows else 0.0,
        rows[0]['lead'] if rows else None, bool(rows and rows[0]['identified']),
    )
    return rows[: int(n_results)]
