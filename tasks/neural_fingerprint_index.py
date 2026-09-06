# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Find which track, and where in it, a clip's neural fingerprint aligns with.

Every analysed track stores one 32-byte code per half second, a 128-number
vector product-quantised against the shipped codebook
(tasks.neural_fingerprint). This module packs all of them into raw files under
IVF_DISK_CACHE_DIR, groups them into coarse cells so a query touches a small
part of the library, and turns per-segment neighbours into a track answer by
asking the one question a fingerprint can answer: do the clip's seconds land
on consecutive seconds of the same track.

Main Features:
* Pack: the stored blobs are read from the embedding table (legacy int8 blobs
  are re-encoded on the way), kept in track order in one file of codes
  (memory-mapped at query time and decoded through the codebook only for the
  rows a query touches), with per-track row ranges and item ids; it is
  rebuilt when the number of fingerprinted tracks changes or the pack layout
  is older, and released by the idle timer of the recording search
* Cells: k-means centroids on a sample of rows, every row assigned to its
  nearest centroid, rows listed per cell so a query vector reads only
  NEURAL_FINGERPRINT_NPROBE cells
* Search: each query segment fetches its nearest rows, every neighbour votes
  for (track, offset = its position minus the query position) with its
  similarity, votes within one hop of each other are pooled, and the best
  offset per track is kept; the top candidates are then verified by scoring
  the whole clip against the track at that offset (mean cosine over the
  overlap), which is what the rows report as score
* Rows carry item_id, score, votes, offset_seconds, identified and lead: the
  best track is identified when its score clears NEURAL_FINGERPRINT_MIN_SCORE
  and leads the next different track by NEURAL_FINGERPRINT_MIN_LEAD
"""

import logging
import math
import os
import threading
import time

import numpy as np

import config
from tasks.idle_unload import IdleUnloadTimer
from tasks.neural_fingerprint import CODE_BYTES, DIM, HOP_SAMPLES, HOP_SECONDS, decode_blob, decode_codes, fingerprint_audio

logger = logging.getLogger(__name__)

_FILE_PREFIX = 'neural_fingerprint'
_PACK_FORMAT = 2
_MIN_TRAIN_ROWS = 2000
_MAX_TRAIN_ROWS = 400000
_MAX_CELLS = 8192
_ASSIGN_BATCH = 65536
_TOP_K = 8
_VERIFY = 20
_TIMER = IdleUnloadTimer()
_LOCK = threading.RLock()
_STATE = {
    'codes': None, 'starts': None, 'lengths': None, 'ids': None, 'count': None,
    'centroids': None, 'cell_rows': None, 'cell_bounds': None, 'row_track': None,
    'building': False, 'error': None,
}


def _paths():
    directory = config.IVF_DISK_CACHE_DIR
    return {
        'codes': os.path.join(directory, f'{_FILE_PREFIX}_rows.u8'),
        'cells': os.path.join(directory, f'{_FILE_PREFIX}_cells.i32'),
        'meta': os.path.join(directory, f'{_FILE_PREFIX}_meta.npz'),
    }


def _db_count(conn):
    cur = conn.cursor()
    try:
        cur.execute('SELECT count(*) FROM embedding WHERE neural_fingerprint IS NOT NULL')
        return int(cur.fetchone()[0])
    finally:
        cur.close()


def _cell_count(n_rows):
    return int(max(1, min(_MAX_CELLS, round(math.sqrt(max(1, n_rows))))))


def train_centroids(sample, n_cells):
    from sklearn.cluster import MiniBatchKMeans

    if n_cells <= 1 or sample.shape[0] <= n_cells:
        return np.ascontiguousarray(sample[:max(1, n_cells)], dtype=np.float32)
    km = MiniBatchKMeans(n_clusters=n_cells, batch_size=10000, n_init=1, max_iter=25, random_state=0, init='random')
    km.fit(sample)
    centroids = km.cluster_centers_.astype(np.float32)
    return centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)


def assign_cells(codes, centroids):
    labels = np.empty(codes.shape[0], dtype=np.int32)
    for start in range(0, codes.shape[0], _ASSIGN_BATCH):
        block = decode_codes(codes[start:start + _ASSIGN_BATCH])
        labels[start:start + _ASSIGN_BATCH] = np.argmax(block @ centroids.T, axis=1)
    return labels


def _stored_rows(conn):
    cur = conn.cursor(name='neural_fingerprint_pack')
    cur.itersize = 500
    cur.execute('SELECT item_id, neural_fingerprint FROM embedding WHERE neural_fingerprint IS NOT NULL ORDER BY item_id')
    try:
        for item_id, blob in cur:
            yield item_id, blob
    finally:
        cur.close()


def _build_pack(conn, paths, count):
    legacy = os.path.join(os.path.dirname(paths['codes']), f'{_FILE_PREFIX}_rows.i8')
    if os.path.exists(legacy):
        os.remove(legacy)
    build_pack_from_rows(_stored_rows(conn), paths, count)


def build_pack_from_rows(rows_iter, paths, count):
    os.makedirs(os.path.dirname(paths['codes']), exist_ok=True)
    started = time.time()
    ids, lengths = [], []
    tmp_rows = paths['codes'] + '.tmp'
    with open(tmp_rows, 'wb') as out:
        for item_id, blob in rows_iter:
            codes = decode_blob(blob)
            if codes is None or codes.shape[0] == 0:
                continue
            out.write(np.ascontiguousarray(codes).tobytes())
            ids.append(item_id)
            lengths.append(int(codes.shape[0]))
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    starts = np.zeros_like(lengths_arr)
    if lengths_arr.size:
        starts[1:] = np.cumsum(lengths_arr)[:-1]
    n_rows = int(lengths_arr.sum())
    rows = np.memmap(tmp_rows, dtype=np.uint8, mode='r', shape=(n_rows, CODE_BYTES)) if n_rows else np.zeros((0, CODE_BYTES), np.uint8)
    n_cells = _cell_count(n_rows)
    rng = np.random.default_rng(0)
    take = min(n_rows, max(_MIN_TRAIN_ROWS, min(_MAX_TRAIN_ROWS, 50 * n_cells)))
    sample_idx = np.sort(rng.choice(n_rows, size=take, replace=False)) if take < n_rows else np.arange(n_rows)
    sample = decode_codes(rows[sample_idx]) if n_rows else np.zeros((0, DIM), np.float32)
    centroids = train_centroids(sample, n_cells) if n_rows else np.zeros((1, DIM), np.float32)
    labels = assign_cells(rows, centroids) if n_rows else np.zeros(0, np.int32)
    order = np.argsort(labels, kind='stable').astype(np.int32)
    bounds = np.concatenate(([0], np.cumsum(np.bincount(labels, minlength=centroids.shape[0])))).astype(np.int64)
    row_track = np.repeat(np.arange(lengths_arr.size, dtype=np.int32), lengths_arr)
    del rows
    order.tofile(paths['cells'] + '.tmp')
    np.savez(
        paths['meta'] + '.tmp.npz', starts=starts, lengths=lengths_arr, ids=np.asarray(ids, dtype=str),
        count=np.int64(count), centroids=centroids, cell_bounds=bounds, row_track=row_track,
        format=np.int64(_PACK_FORMAT),
    )
    os.replace(tmp_rows, paths['codes'])
    os.replace(paths['cells'] + '.tmp', paths['cells'])
    os.replace(paths['meta'] + '.tmp.npz', paths['meta'])
    logger.info(
        'Neural fingerprint pack built: %d tracks, %d rows, %d cells, %.0fs',
        len(ids), n_rows, centroids.shape[0], time.time() - started,
    )


def _open_pack(paths):
    meta = np.load(paths['meta'], allow_pickle=False)
    n_rows = int(meta['lengths'].sum())
    _STATE['starts'] = meta['starts']
    _STATE['lengths'] = meta['lengths']
    _STATE['ids'] = meta['ids']
    _STATE['count'] = int(meta['count'])
    _STATE['centroids'] = meta['centroids']
    _STATE['cell_bounds'] = meta['cell_bounds']
    _STATE['row_track'] = meta['row_track']
    _STATE['codes'] = np.memmap(paths['codes'], dtype=np.uint8, mode='r', shape=(n_rows, CODE_BYTES))
    _STATE['cell_rows'] = np.memmap(paths['cells'], dtype=np.int32, mode='r', shape=(n_rows,))


def _cached_count(paths):
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    try:
        with np.load(paths['meta'], allow_pickle=False) as meta:
            if 'format' not in meta.files or int(meta['format']) != _PACK_FORMAT:
                return None
            return int(meta['count'])
    except Exception:
        logger.exception('Neural fingerprint pack metadata unreadable; rebuilding')
        return None


def is_loaded():
    return _STATE['codes'] is not None


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
        paths = _paths()
        conn = connect_raw(application_name='neural_fingerprint_index')
        try:
            count = _db_count(conn)
            if count == 0:
                raise RuntimeError('No neural fingerprints are stored yet. Run analysis first.')
            if _cached_count(paths) != count:
                logger.info('Neural fingerprint pack missing or stale; building from %d tracks', count)
                _build_pack(conn, paths, count)
        finally:
            conn.close()
        with _LOCK:
            _open_pack(paths)
        logger.info('Neural fingerprint pack loaded: %d tracks', int(_STATE['ids'].size))
        return True
    except Exception as exc:
        with _LOCK:
            _STATE['error'] = str(exc)
        logger.exception('Neural fingerprint pack could not be loaded')
        raise
    finally:
        with _LOCK:
            _STATE['building'] = False


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
        for key in ('codes', 'starts', 'lengths', 'ids', 'centroids', 'cell_rows', 'cell_bounds', 'row_track'):
            _STATE[key] = None
    return was_loaded


def get_status():
    from tasks.neural_fingerprint import is_available

    with _LOCK:
        return {
            'available': bool(is_available()),
            'loaded': is_loaded(),
            'building': bool(_STATE['building']),
            'tracks': int(_STATE['ids'].size) if is_loaded() else 0,
            'error': _STATE['error'],
        }


def _arm_idle_unload():
    if _TIMER.arm(config.RECORDING_SEARCH_WARMUP_DURATION, unload):
        logger.info('Neural fingerprint pack in use; idle unload in %ss', config.RECORDING_SEARCH_WARMUP_DURATION)


def _probe_rows(query, nprobe):
    centroids = _STATE['centroids']
    cells = np.argsort(-(centroids @ query))[:nprobe]
    bounds = _STATE['cell_bounds']
    pieces = [_STATE['cell_rows'][bounds[c]:bounds[c + 1]] for c in cells]
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)


def _vote(query_vectors, nprobe):
    codes = _STATE['codes']
    row_track = _STATE['row_track']
    starts = _STATE['starts']
    votes = {}
    for qi, query in enumerate(query_vectors):
        rows = _probe_rows(query, nprobe)
        if not rows.size:
            continue
        rows = np.sort(rows)
        sims = decode_codes(codes[rows]) @ query
        top = np.argpartition(-sims, min(_TOP_K, sims.size - 1))[:_TOP_K] if sims.size > _TOP_K else np.arange(sims.size)
        for j in top:
            track = int(row_track[rows[j]])
            offset = int(rows[j] - starts[track]) - qi
            key = (track, offset)
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
    from tasks.neural_fingerprint import is_available

    if not is_available():
        raise RuntimeError('The neural fingerprint model is not available here.')
    ensure_loaded()
    query_vectors = fingerprint_audio(audio, sr, HOP_SAMPLES)
    if query_vectors is None or query_vectors.shape[0] < 2:
        raise ValueError('The clip is too short to fingerprint: at least two seconds are needed.')
    with _TIMER.lock():
        _arm_idle_unload()
    started = time.time()
    nprobe = max(1, int(config.NEURAL_FINGERPRINT_NPROBE))
    pooled = _vote(query_vectors, nprobe)
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
