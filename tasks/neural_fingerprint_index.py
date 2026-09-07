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
* Flask side: load_at_startup maps the stored build when Flask boots (and
  says when none exists yet), ensure_loaded reads the build id, syncs the
  local files when it differs (codes appended for new tracks when the
  previous local order is a prefix of the new one, rewritten otherwise; parts
  merged into one cell list) and memory-maps them under a build-id-keyed
  name, so a reload on the index-reload event prepares the next build while
  the current one serves and swaps at the end; the pack then stays mapped for
  the life of the process like the other indexes (the recording search's
  idle timer releases only the encoder session)
* Search: each query segment fetches its nearest rows from
  NEURAL_FINGERPRINT_NPROBE cells, every neighbour votes for (track, offset)
  with its similarity, votes within one hop are pooled, and the best
  candidates are verified by the mean cosine between the whole clip and the
  track at that offset; rows carry item_id, score, votes, offset_seconds,
  identified and lead. identify embeds a clip first; identify_vectors takes
  a fingerprint sequence directly and can leave given tracks out, which is
  how a stored song is searched for its other recordings without finding
  itself
* Per-server scope like every other index: the index holds the union of all
  servers, and a request scoped to a server votes only over that server's
  tracks through the shared availability mask (tasks.index_availability),
  cached per server and build for 30 s and dropped by
  invalidate_availability_cache when the mappings change
* get_indexed_item_ids lists the tracks of the loaded build for the song
  picker of the Search by Song tab, the way the SemGrove index feeds the
  lyrics picker, so only songs the index knows are offered
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

import numpy as np

import config
from tasks import ivf_quant
from tasks.index_availability import active_availability_scope, build_availability_mask
from tasks.neural_fingerprint import (
    CODE_BYTES, DIM, HOP_SAMPLES, HOP_SECONDS, PQ_SUBDIM, PQ_SUBSPACES, codebook, decode_blob, decode_codes,
    fingerprint_audio, is_available,
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
_SUBSPACE_INDEX = np.arange(PQ_SUBSPACES)[None, :]
_AVAILABILITY_CACHE = {}
_AVAILABILITY_CACHE_LOCK = threading.Lock()
_AVAILABILITY_CACHE_TTL = 30.0
_CANONICAL = {}
_LOCK = threading.RLock()
_STATE = {
    'codes': None, 'starts': None, 'lengths': None, 'ids': None, 'centroids': None,
    'cell_rows': None, 'cell_bounds': None, 'build_id': None, 'building': False, 'error': None, 'gpu': None,
}


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
            'SELECT coalesce(sum(CASE WHEN substring(neural_fingerprint from 1 for 4) = %s '
            'THEN (octet_length(neural_fingerprint) - 15) / 32 ELSE (octet_length(neural_fingerprint) - 11) / 128 END), 0) '
            'FROM embedding WHERE neural_fingerprint IS NOT NULL',
            (b'NFP2',),
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
    from tasks.index_build_helpers import load_segmented_blob

    blob = load_segmented_blob(conn, DIR_TABLE, _DIR_NAME)
    return unpack_directory(blob) if blob else None


def _stored_build_id(conn):
    from tasks.index_build_helpers import load_segmented_blob

    blob = load_segmented_blob(conn, DIR_TABLE, _BUILD_NAME)
    return bytes(blob).decode('ascii') if blob else None


def _store_directory(conn, directory_blob, build_id):
    from tasks.index_build_helpers import store_segmented_blob

    store_segmented_blob(conn, DIR_TABLE, _DIR_NAME, directory_blob)
    store_segmented_blob(conn, DIR_TABLE, _BUILD_NAME, build_id.encode('ascii'))


def _store_part(conn, index, blob):
    from tasks.index_build_helpers import store_segmented_blob

    store_segmented_blob(conn, DIR_TABLE, f'{_PART_PREFIX}{index}', blob)


def _delete_parts(conn):
    cur = conn.cursor()
    try:
        cur.execute(
            f"DELETE FROM {DIR_TABLE} WHERE name LIKE %s ESCAPE '\\'",
            (_PART_PREFIX.replace('_', r'\_') + '%',),
        )
    finally:
        cur.close()


def _load_parts(conn, count):
    from tasks.index_build_helpers import load_segmented_blob

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
    if not is_available():
        logger.info('Neural fingerprint index skipped: the model or the codebook is missing')
        return False
    codebook_id = codebook()[1]
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
    return True


def _paths(build_id):
    directory = config.IVF_DISK_CACHE_DIR
    return {
        'codes': os.path.join(directory, f'{_FILE_PREFIX}.{build_id}.rows.u8'),
        'cells': os.path.join(directory, f'{_FILE_PREFIX}.{build_id}.cells.i32'),
        'meta': os.path.join(directory, f'{_FILE_PREFIX}.{build_id}.meta.npz'),
    }


def _local_builds():
    pattern = os.path.join(config.IVF_DISK_CACHE_DIR, f'{_FILE_PREFIX}.*.meta.npz')
    found = []
    for path in glob.glob(pattern):
        try:
            with np.load(path, allow_pickle=False) as meta:
                if int(meta['format']) == FORMAT:
                    found.append((os.path.getmtime(path), str(meta['build_id'])))
        except Exception:
            logger.exception('Unreadable neural fingerprint pack metadata %s', path)
    return [build_id for _mtime, build_id in sorted(found)]


def _local_meta(build_id):
    paths = _paths(build_id)
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    with np.load(paths['meta'], allow_pickle=False) as meta:
        return {'ids': [str(i) for i in meta['ids']], 'lengths': meta['lengths'].astype(np.int64)}


def _prune_local(keep_build_id):
    for build_id in _local_builds():
        if build_id == keep_build_id:
            continue
        for path in _paths(build_id).values():
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


def write_local_pack(paths, directory, parts, codes_iter, previous_paths=None, previous=None):
    os.makedirs(os.path.dirname(paths['codes']), exist_ok=True)
    started = time.time()
    ids, lengths = list(directory['ids']), directory['lengths']
    keep = _reusable_prefix(previous, ids, lengths) if previous_paths else 0
    tmp_rows = paths['codes'] + '.tmp'
    with open(tmp_rows, 'wb') as out:
        if keep:
            remaining = int(lengths[:keep].sum()) * CODE_BYTES
            with open(previous_paths['codes'], 'rb') as source:
                while remaining > 0:
                    chunk = source.read(min(remaining, 1 << 24))
                    if not chunk:
                        raise RuntimeError('the previous neural fingerprint pack is shorter than its metadata says')
                    out.write(chunk)
                    remaining -= len(chunk)
        for index, (item_id, codes) in enumerate(codes_iter(ids[keep:]), start=keep):
            expected = int(lengths[index])
            if codes is None or codes.shape[0] != expected:
                raise RuntimeError(
                    f'The fingerprint of {item_id} no longer matches the index ({expected} rows expected); rebuild the indexes.'
                )
            out.write(np.ascontiguousarray(codes).tobytes())
    n_cells = int(directory['centroids'].shape[0])
    cell_rows, bounds = merge_parts(parts, n_cells)
    n_rows = int(lengths.sum())
    if cell_rows.size != n_rows:
        raise RuntimeError(f'The neural fingerprint index parts hold {cell_rows.size} rows for {n_rows} stored; rebuild the indexes.')
    starts = np.zeros(lengths.size, dtype=np.int64)
    if lengths.size:
        starts[1:] = np.cumsum(lengths)[:-1]
    cell_rows.tofile(paths['cells'] + '.tmp')
    np.savez(
        paths['meta'] + '.tmp.npz', starts=starts, lengths=lengths, ids=np.asarray(ids, dtype=str),
        centroids=directory['centroids'], cell_bounds=bounds, build_id=np.asarray(directory['build_id']),
        codebook_id=np.uint32(directory['codebook_id']), format=np.int64(FORMAT),
    )
    os.replace(tmp_rows, paths['codes'])
    os.replace(paths['cells'] + '.tmp', paths['cells'])
    os.replace(paths['meta'] + '.tmp.npz', paths['meta'])
    logger.info(
        'Neural fingerprint pack synced: %d tracks, %d rows, %d cells, %d tracks reused, %.0fs',
        len(ids), n_rows, n_cells, keep, time.time() - started,
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
        n_rows = int(meta['lengths'].sum())
        _STATE['starts'] = meta['starts']
        _STATE['lengths'] = meta['lengths']
        _STATE['ids'] = meta['ids']
        _STATE['centroids'] = meta['centroids']
        _STATE['cell_bounds'] = meta['cell_bounds']
        _STATE['build_id'] = str(meta['build_id'])
    _STATE['codes'] = np.memmap(paths['codes'], dtype=np.uint8, mode='r', shape=(n_rows, CODE_BYTES))
    _STATE['cell_rows'] = np.memmap(paths['cells'], dtype=np.int32, mode='r', shape=(n_rows,))


def is_loaded():
    return _STATE['codes'] is not None


def get_indexed_item_ids():
    with _LOCK:
        if not is_loaded():
            return set()
        return set(_STATE['ids'].tolist())


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
            _open_pack(paths)
        _prune_local(_STATE['build_id'])
        logger.info('Neural fingerprint pack loaded: %d tracks (build %s)', int(_STATE['ids'].size), _STATE['build_id'])
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

    conn = connect_raw(application_name='neural_fingerprint_index')
    try:
        build_id = _stored_build_id(conn)
        if build_id is None:
            return False
        with _LOCK:
            if is_loaded() and _STATE['build_id'] == build_id:
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

    if not is_available():
        logger.info('Neural fingerprint index not loaded at startup: the model or the codebook is missing')
        return 0
    conn = connect_raw(application_name='neural_fingerprint_index')
    try:
        if _stored_build_id(conn) is None:
            return 0
    finally:
        conn.close()
    ensure_loaded()
    with _LOCK:
        return int(_STATE['ids'].size) if is_loaded() else 0


def start_background_load():
    with _LOCK:
        if is_loaded() or _STATE['building']:
            return False

    def _run():
        try:
            ensure_loaded()
        except Exception:
            logger.warning('Background neural fingerprint load failed; the next request retries')

    threading.Thread(target=_run, name='neural-fingerprint-load', daemon=True).start()
    return True


def unload():
    with _LOCK:
        was_loaded = is_loaded()
        for key in ('codes', 'starts', 'lengths', 'ids', 'centroids', 'cell_rows', 'cell_bounds', 'build_id'):
            _STATE[key] = None
    return was_loaded


def get_status():
    with _LOCK:
        return {
            'available': bool(is_available()),
            'loaded': is_loaded(),
            'synced': is_loaded() or bool(_local_builds()),
            'building': bool(_STATE['building']),
            'tracks': int(_STATE['ids'].size) if is_loaded() else 0,
            'error': _STATE['error'],
        }


def _probe_rows(query, nprobe):
    centroids = _STATE['centroids']
    cells = np.argsort(-(centroids @ query))[:nprobe]
    bounds = _STATE['cell_bounds']
    pieces = [_STATE['cell_rows'][bounds[c]:bounds[c + 1]] for c in cells]
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)


def _row_scores(query, rows):
    book = codebook()[0]
    table = np.einsum('sd,skd->sk', query.reshape(PQ_SUBSPACES, PQ_SUBDIM), book)
    norms = np.einsum('skd,skd->sk', book, book)
    picked = _STATE['codes'][rows]
    dots = table[_SUBSPACE_INDEX, picked].sum(axis=1)
    return dots / (np.sqrt(norms[_SUBSPACE_INDEX, picked].sum(axis=1)) + 1e-9)


def _vote(query_vectors, nprobe, allowed=None):
    starts = _STATE['starts']
    votes = {}
    for qi, query in enumerate(query_vectors):
        rows = np.sort(_probe_rows(query, nprobe))
        if allowed is not None:
            rows = rows[allowed[np.searchsorted(starts, rows, side='right') - 1]]
        if not rows.size:
            continue
        sims = _row_scores(query, rows)
        top = np.argpartition(-sims, min(_TOP_K, sims.size - 1))[:_TOP_K] if sims.size > _TOP_K else np.arange(sims.size)
        tracks = np.searchsorted(starts, rows[top], side='right') - 1
        for j, track in zip(top, tracks):
            offset = int(rows[j] - starts[track]) - qi
            key = (int(track), offset)
            votes[key] = votes.get(key, 0.0) + float(sims[j])
    pooled = {}
    for (track, offset), value in votes.items():
        total = value + votes.get((track, offset - 1), 0.0) + votes.get((track, offset + 1), 0.0)
        if total > pooled.get(track, (0.0, 0))[0]:
            pooled[track] = (total, offset)
    return pooled


def _verify(query_vectors, track, offset):
    starts, lengths = _STATE['starts'], _STATE['lengths']
    length = int(lengths[track])
    best = (-1.0, offset)
    for candidate in (offset - 1, offset, offset + 1):
        lo = max(0, -candidate)
        hi = min(query_vectors.shape[0], length - candidate)
        if hi - lo < 1:
            continue
        rows = decode_codes(_STATE['codes'][starts[track] + candidate + lo: starts[track] + candidate + hi])
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


def _has_canonical_ids():
    from tasks.simhash import is_fingerprint_id

    build_id = _STATE['build_id']
    if build_id not in _CANONICAL:
        _CANONICAL.clear()
        _CANONICAL[build_id] = any(is_fingerprint_id(str(item_id)) for item_id in _STATE['ids'])
    return _CANONICAL[build_id]


def _mask_unneeded(server_id):
    try:
        from tasks.mediaserver import registry

        return bool(
            server_id == str(registry.get_default_server_id() or '')
            and not registry.has_secondary_servers()
            and not _has_canonical_ids()
        )
    except Exception:
        logger.debug('Single-server availability fast path failed.', exc_info=True)
        return False


def _availability_mask():
    server_id = active_availability_scope()
    if server_id is None or _mask_unneeded(server_id):
        return None
    from database import get_db

    key = (server_id, _STATE['build_id'])
    now = time.monotonic()
    with _AVAILABILITY_CACHE_LOCK:
        cached = _AVAILABILITY_CACHE.get(key)
        if cached is not None and now - cached[0] < _AVAILABILITY_CACHE_TTL:
            return cached[1]
    mask = build_availability_mask(server_id, _STATE['ids'], get_db)
    with _AVAILABILITY_CACHE_LOCK:
        stale = [k for k, v in _AVAILABILITY_CACHE.items() if k[1] != key[1] or now - v[0] >= _AVAILABILITY_CACHE_TTL]
        for old in stale:
            _AVAILABILITY_CACHE.pop(old, None)
        _AVAILABILITY_CACHE[key] = (now, mask)
    return mask


def _allowed_tracks(exclude_ids):
    allowed = _availability_mask()
    if not exclude_ids:
        return allowed
    allowed = np.ones(_STATE['ids'].size, dtype=np.bool_) if allowed is None else allowed.copy()
    allowed[np.isin(_STATE['ids'], list(exclude_ids))] = False
    return allowed


def identify_vectors(query_vectors, n_results, exclude_ids=()):
    ensure_loaded()
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    if query_vectors.ndim != 2 or query_vectors.shape[0] < 2:
        raise ValueError('The query is too short to identify: at least two segments are needed.')
    started = time.time()
    nprobe = max(1, int(config.NEURAL_FINGERPRINT_NPROBE))
    pooled = _vote(query_vectors, nprobe, _allowed_tracks(exclude_ids))
    ranked = sorted(pooled.items(), key=lambda kv: -kv[1][0])[:max(int(n_results), _VERIFY)]
    scored = []
    for track, (votes, offset) in ranked:
        score, aligned = _verify(query_vectors, track, offset)
        scored.append((track, score, votes, aligned))
    scored.sort(key=lambda row: -row[1])
    ids = _STATE['ids']
    lead = scored[0][1] - scored[1][1] if len(scored) > 1 else float('inf')
    identified = bool(
        scored and scored[0][1] >= float(config.NEURAL_FINGERPRINT_MIN_SCORE)
        and lead >= float(config.NEURAL_FINGERPRINT_MIN_LEAD)
    )
    logger.info(
        'Neural fingerprint identify: %d query segments, %d candidates, %.1fs, best score %.3f lead %.3f identified %s',
        query_vectors.shape[0], len(pooled), time.time() - started, scored[0][1] if scored else 0.0,
        lead if np.isfinite(lead) else -1.0, identified,
    )
    rows = []
    for rank, (track, score, votes, aligned) in enumerate(scored[: int(n_results)]):
        rows.append(
            {
                'item_id': str(ids[track]),
                'score': round(float(score), 3),
                'votes': round(float(votes), 2),
                'offset_seconds': round(float(aligned) * HOP_SECONDS, 1),
                'identified': bool(identified and rank == 0),
                'lead': round(float(lead), 3) if rank == 0 and np.isfinite(lead) else None,
            }
        )
    return rows
