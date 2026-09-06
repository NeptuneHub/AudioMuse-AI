# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Identify the exact recording behind a clip from the chromaprints already stored.

Nothing new is analysed. The `chromaprint` table, written for the duplicate
detector, already holds an fpcalc fingerprint of the first 120 seconds of every
analysed track. This module reads it back, packs it, and slides the clip's own
fpcalc fingerprint across every track to find where it aligns, the way a
phone-recording identifier does, but on data the library already has.

Main Features:
* Library pack: every stored fingerprint is split into two uint16 planes and
  kept, with its item id, in raw files under IVF_DISK_CACHE_DIR. The pack is rebuilt when the table's fingerprint count
  changes and memory-mapped at query time, so nothing stays resident when idle;
  the build runs in a background thread started by the page warmup.
* Score: a weighted bit error rate over every alignment offset. The 32 per-bit
  weights are log-likelihood ratios measured on 80 tracks degraded to the
  profile of a real phone recording (coarse Gray bits flip 24%, fine bits 36%,
  filter 15 carries almost nothing). Two 65536-entry tables turn the weighted
  popcount of a 16-bit XOR into one gather, and chunks of equal-length tracks
  run on a thread pool while numpy releases the GIL.
* Noise frames: stationary noise fingerprints to a tight family of values (98
  percent of a pure-noise clip's frames sit within 8 bits of the clip's
  bitwise-majority value, 3 to 12 percent of a music clip's), and library
  tracks with long noise-like passages (sound effects, ambient intros) hold
  the same family, so a noise-dominated recording was confidently matched to
  them. Query frames within _FLAT_BITS of the majority value are left out of
  every scan; a clip with too few frames left is refused as too noisy.
* Two-stage scan: every track is first scored with every fourth query frame
  (a quarter of the work; on the reference phone recording the true song still
  ranked inside the top 600 of 200k), the top 2 percent become the pool, and
  the pool plus a random sample of 1000 tracks get the exact score. The sample
  gives an unbiased null mean and deviation for the z statistic. The best
  candidate counts as identified when its z clears the extreme a random
  library of the scanned size would produce (sqrt(2 ln N)) by
  RECORDING_SEARCH_IDENTIFY_MARGIN AND it leads the next different recording
  by RECORDING_SEARCH_IDENTIFY_LEAD: noise-like matches arrive in dense
  groups, a genuine one stands alone. "Different" is decided on the stored
  windows the two candidates aligned to the clip (weighted bit error above
  _DUPLICATE_BER), so a duplicate of the best track shares its flag and its
  lead instead of blocking them.
* Variant selection: the clip's frames sit at a random fraction of one
  chromaprint hop (0.124 s) from the reference frames, and a source played one
  percent fast or slow (radio, TV, a re-encoded stream) triples the error rate
  of a clean clip, so the top RECORDING_SEARCH_IDENTIFY_RERANK candidates and
  the null sample are re-scored with the clip fingerprinted at four phases of
  a hop and at the RECORDING_SEARCH_IDENTIFY_SPEEDS factors, and ONE variant is
  kept for everybody: the one whose best candidate is most extreme against its
  own null. Keeping each candidate's best variant instead was measured to help
  random tracks more than the true one (rank 1 became rank 8). When a speed
  wins, its own four phases are tried as well: the reference phone recording,
  measured against the studio original, runs 1.1 percent fast and not even
  uniformly, and the half-percent speed steps plus that refinement took its
  lead over the runner-up from 0.5 to 1.2 sigma.
* What was measured NOT to help, so nobody retries it: adding MusiCNN or DCLAP
  similarity to the chromaprint score (rank-1 rate drops on every degradation
  profile of a 43-song set), per-class chroma gain or noise corrections (worse
  even when fitted on the original), channel-specific bit weights (the per-bit
  flip pattern of a real recording is unstable: its two halves correlate
  0.19), soft-decision bits (no better than unweighted bits, half the rank-1
  rate of the learned weights on the phone profile) and piecewise alignment
  with 2 percent drift slack per piece (never beats the speed variants).
* Rows carry ber (at the recorded speed and phase), z (of the chosen variant),
  offset_seconds, identified, lead (on the best recording and its duplicates,
  None elsewhere) and the chosen variant's name.
* identify returns rows with item_id, ber, z, offset_seconds and identified;
  the caller attaches titles. Measured on the owner's phone recording of a song
  whose embeddings ranked it 90,803rd: rank 1 of 200,453 tracks at z 5.4. The
  stored fingerprints cover the first 120 s of a track, so a clip taken later
  in a song cannot be identified from the data the library holds.
"""

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import config
from cpu_budget import usable_cpu_count
from tasks.idle_unload import IdleUnloadTimer

logger = logging.getLogger(__name__)

HOP_SECONDS = 0.1238
_WINDOW_SECONDS = 2.6
_MIN_QUERY_FRAMES = 16
PHASES = 4
_CHUNK_TRACKS = 96
_MAX_THREADS = 4
_STAGE1_STRIDE = 4
_POOL_FRACTION = 0.02
_POOL_MIN = 2000
_NULL_SAMPLE = 1000
_MIN_NULL_SAMPLE = 200
_FILE_PREFIX = 'chromaprint_identify'
_LUT_SCALE = 256.0
_FLAT_BITS = 7
_DUPLICATE_BER = 0.2
_POPCOUNT16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)
_BIT_WEIGHTS = np.array(
    [
        0.691, 1.114, 0.798, 1.286, 0.509, 1.139, 0.683, 1.166, 0.803, 1.244, 0.551, 1.006,
        0.613, 1.089, 0.590, 1.277, 0.473, 1.131, 0.657, 0.737, 0.628, 1.451, 0.706, 1.259,
        0.655, 1.275, 0.524, 1.842, 0.462, 1.688, 0.077, 0.099,
    ],
    dtype=np.float32,
)

_TIMER = IdleUnloadTimer()
_LOCK = threading.RLock()
_STATE = {
    'lo': None, 'hi': None, 'starts': None, 'lengths': None, 'ids': None,
    'count': None, 'building': False, 'error': None,
}
_LUTS = {}


def split_planes(fingerprint):
    arr = np.asarray(fingerprint, dtype=np.uint32)
    return (arr & np.uint32(0xFFFF)).astype(np.uint16), (arr >> np.uint32(16)).astype(np.uint16)


def weight_luts():
    if not _LUTS:
        values = np.arange(65536, dtype=np.uint32)
        for name, offset in (('lo', 0), ('hi', 16)):
            table = np.zeros(65536, dtype=np.float32)
            for bit in range(16):
                table += ((values >> np.uint32(bit)) & 1).astype(np.float32) * _BIT_WEIGHTS[offset + bit]
            _LUTS[name] = np.round(table * _LUT_SCALE).astype(np.uint16)
        _LUTS['total'] = float(_BIT_WEIGHTS.sum()) * _LUT_SCALE
    return _LUTS


def informative_positions(query):
    values = np.asarray(query, dtype=np.uint32)
    bits = (values[:, None] >> np.arange(32, dtype=np.uint32)) & np.uint32(1)
    majority = (bits.mean(axis=0) >= 0.5).astype(np.uint32)
    centre = np.uint32((majority << np.arange(32, dtype=np.uint32)).sum())
    lo, hi = split_planes(values ^ centre)
    distance = _POPCOUNT16[lo].astype(np.int64) + _POPCOUNT16[hi]
    return np.nonzero(distance > _FLAT_BITS)[0]


def score_rows(lo_rows, hi_rows, q_lo, q_hi, stride=1, positions=None):
    luts = weight_luts()
    qlen = q_lo.size
    if positions is None:
        positions = np.arange(qlen)
    positions = np.asarray(positions)[::stride]
    used = positions.size
    windows = sliding_window_view(lo_rows, qlen, axis=1)[:, :, positions]
    total = luts['lo'][windows ^ q_lo[None, None, positions]].sum(axis=2, dtype=np.int64)
    windows = sliding_window_view(hi_rows, qlen, axis=1)[:, :, positions]
    total += luts['hi'][windows ^ q_hi[None, None, positions]].sum(axis=2, dtype=np.int64)
    ber = total / (used * luts['total'])
    best = np.argmin(ber, axis=1)
    return ber[np.arange(best.size), best].astype(np.float32), best.astype(np.int64)


def expected_extreme_z(scanned):
    return math.sqrt(2.0 * math.log(max(2, int(scanned))))


def _cache_paths():
    directory = config.IVF_DISK_CACHE_DIR
    return {
        'lo': os.path.join(directory, f'{_FILE_PREFIX}_lo.bin'),
        'hi': os.path.join(directory, f'{_FILE_PREFIX}_hi.bin'),
        'meta': os.path.join(directory, f'{_FILE_PREFIX}_meta.npz'),
    }


def _db_fingerprint_count(conn):
    cur = conn.cursor()
    try:
        cur.execute('SELECT count(*) FROM chromaprint WHERE fingerprint IS NOT NULL')
        return int(cur.fetchone()[0])
    finally:
        cur.close()


def _build_pack(conn, paths, count):
    from tasks.chromaprint import decode_fingerprint

    os.makedirs(config.IVF_DISK_CACHE_DIR, exist_ok=True)
    cur = conn.cursor(name='chromaprint_identify_build')
    cur.itersize = 2000
    cur.execute(
        'SELECT m.item_id, c.fingerprint FROM chromaprint c '
        'JOIN track_server_map m ON m.server_id = c.server_id '
        'AND m.provider_track_id = c.provider_track_id WHERE c.fingerprint IS NOT NULL'
    )
    seen = set()
    ids, lengths = [], []
    tmp_lo, tmp_hi = paths['lo'] + '.tmp', paths['hi'] + '.tmp'
    started = time.time()
    try:
        with open(tmp_lo, 'wb') as out_lo, open(tmp_hi, 'wb') as out_hi:
            for item_id, blob in cur:
                if item_id in seen:
                    continue
                fingerprint = decode_fingerprint(blob)
                if fingerprint is None:
                    continue
                seen.add(item_id)
                lo, hi = split_planes(fingerprint)
                out_lo.write(lo.tobytes())
                out_hi.write(hi.tobytes())
                ids.append(item_id)
                lengths.append(lo.size)
    finally:
        cur.close()
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    starts = np.zeros_like(lengths_arr)
    if lengths_arr.size:
        starts[1:] = np.cumsum(lengths_arr)[:-1]
    np.savez(
        paths['meta'] + '.tmp.npz', starts=starts, lengths=lengths_arr,
        ids=np.asarray(ids, dtype=str), count=np.int64(count),
    )
    os.replace(tmp_lo, paths['lo'])
    os.replace(tmp_hi, paths['hi'])
    os.replace(paths['meta'] + '.tmp.npz', paths['meta'])
    logger.info(
        'Chromaprint identify pack built: %d tracks, %d sub-fingerprints, %.0fs',
        len(ids), int(lengths_arr.sum()), time.time() - started,
    )


def _open_pack(paths):
    meta = np.load(paths['meta'], allow_pickle=False)
    _STATE['starts'] = meta['starts']
    _STATE['lengths'] = meta['lengths']
    _STATE['ids'] = meta['ids']
    _STATE['count'] = int(meta['count'])
    _STATE['lo'] = np.memmap(paths['lo'], dtype=np.uint16, mode='r')
    _STATE['hi'] = np.memmap(paths['hi'], dtype=np.uint16, mode='r')


def _cached_count(paths):
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    try:
        return int(np.load(paths['meta'], allow_pickle=False)['count'])
    except Exception:
        logger.exception('Chromaprint identify pack metadata unreadable; rebuilding')
        return None


def is_loaded():
    return _STATE['lo'] is not None


def ensure_loaded():
    from database import connect_raw

    with _LOCK:
        if is_loaded():
            return True
        if _STATE['building']:
            raise RuntimeError('The identification index is being prepared; try again in a minute.')
        _STATE['building'] = True
        _STATE['error'] = None
    try:
        paths = _cache_paths()
        conn = connect_raw(application_name='chromaprint_identify')
        try:
            count = _db_fingerprint_count(conn)
            if count == 0:
                raise RuntimeError('No chromaprint fingerprints are stored yet. Run analysis first.')
            if _cached_count(paths) != count:
                logger.info('Chromaprint identify pack missing or stale; building from %d fingerprints', count)
                _build_pack(conn, paths, count)
        finally:
            conn.close()
        with _LOCK:
            _open_pack(paths)
        logger.info('Chromaprint identify pack loaded: %d tracks', int(_STATE['ids'].size))
        return True
    except Exception as exc:
        with _LOCK:
            _STATE['error'] = str(exc)
        logger.exception('Chromaprint identify pack could not be loaded')
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
            logger.warning('Background chromaprint identify load failed; the next request retries')

    threading.Thread(target=_run, name='chromaprint-identify-load', daemon=True).start()
    return True


def unload():
    with _LOCK:
        was_loaded = is_loaded()
        for key in ('lo', 'hi', 'starts', 'lengths', 'ids'):
            _STATE[key] = None
    return was_loaded


def get_status():
    from tasks.chromaprint import is_available

    with _LOCK:
        return {
            'available': bool(is_available()),
            'loaded': is_loaded(),
            'building': bool(_STATE['building']),
            'tracks': int(_STATE['ids'].size) if is_loaded() else 0,
            'error': _STATE['error'],
        }


def _thread_count():
    configured = int(config.RECORDING_SEARCH_IDENTIFY_THREADS)
    if configured > 0:
        return configured
    return max(1, min(_MAX_THREADS, int(usable_cpu_count() or _MAX_THREADS)))


def _chunks(indices, lengths, qlen):
    ordered = indices[np.argsort(lengths[indices], kind='stable')]
    i = 0
    while i < ordered.size:
        n = lengths[ordered[i]]
        j = i
        while j < ordered.size and lengths[ordered[j]] == n and j - i < _CHUNK_TRACKS:
            j += 1
        if n >= qlen:
            yield ordered[i:j]
        i = j


def _rows_for(idx):
    starts, lengths = _STATE['starts'], _STATE['lengths']
    n = int(lengths[idx[0]])
    lo = np.stack([_STATE['lo'][starts[k]:starts[k] + n] for k in idx])
    hi = np.stack([_STATE['hi'][starts[k]:starts[k] + n] for k in idx])
    return lo, hi


def _score_chunk(idx, q_lo, q_hi, stride, positions):
    lo, hi = _rows_for(idx)
    scores, offsets = score_rows(lo, hi, q_lo, q_hi, stride, positions)
    return idx, scores, offsets


def _scan(indices, q_lo, q_hi, threads, stride=1, positions=None):
    n_tracks = int(_STATE['ids'].size)
    scores = np.full(n_tracks, np.inf, dtype=np.float32)
    offsets = np.full(n_tracks, -1, dtype=np.int64)
    chunks = list(_chunks(indices, _STATE['lengths'], q_lo.size))
    if not chunks:
        return scores, offsets, 0
    scanned = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for idx, chunk_scores, chunk_offsets in pool.map(
            lambda c: _score_chunk(c, q_lo, q_hi, stride, positions), chunks
        ):
            scores[idx] = chunk_scores
            offsets[idx] = chunk_offsets
            scanned += idx.size
    return scores, offsets, scanned


def _null_stats(scores):
    finite = scores[np.isfinite(scores)]
    if finite.size < 2:
        return float('nan'), float('nan')
    return float(finite.mean()), float(finite.std())


def _candidate_pool(stage1_scores, n_tracks):
    finite = np.isfinite(stage1_scores)
    keep = min(int(finite.sum()), max(_POOL_MIN, int(_POOL_FRACTION * n_tracks)))
    pool = np.argsort(stage1_scores)[:keep]
    return pool[finite[pool]]


def _null_sample(n_tracks, exclude, rng):
    candidates = np.setdiff1d(np.arange(n_tracks), exclude, assume_unique=False)
    if candidates.size <= _NULL_SAMPLE:
        return candidates
    return rng.choice(candidates, size=_NULL_SAMPLE, replace=False)


def _arm_idle_unload():
    if _TIMER.arm(config.RECORDING_SEARCH_WARMUP_DURATION, unload):
        logger.info('Chromaprint identify pack in use; idle unload in %ss', config.RECORDING_SEARCH_WARMUP_DURATION)


def _speed_factors():
    factors = []
    for token in str(config.RECORDING_SEARCH_IDENTIFY_SPEEDS).split(','):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            logger.warning('Ignoring RECORDING_SEARCH_IDENTIFY_SPEEDS entry %r', token)
            continue
        if 0.5 < value < 2.0 and abs(value - 1.0) > 1e-6:
            factors.append(value)
    return factors


def _phase_shifts(clip, sr):
    for phase in range(1, PHASES):
        yield f'phase {phase}/{PHASES}', clip[int(round(phase * HOP_SECONDS / PHASES * sr)):]


def _query_variants(audio, sr):
    from tasks.analysis import resample_audio

    clip = np.asarray(audio, dtype=np.float32)
    yield from _phase_shifts(clip, sr)
    for factor in _speed_factors():
        yield f'speed {factor}', resample_audio(clip, sr, int(round(sr * factor)))


def _select_variant(audio, sr, top, sample, scores, offsets, min_len, threads):
    from tasks.chromaprint import fingerprint_audio

    subset = np.concatenate([top, sample])
    chosen = None
    speed_audio = {}

    def consider(label, variant):
        nonlocal chosen
        if variant is None:
            s, o = scores, offsets
        else:
            query = fingerprint_audio(variant, sr)
            if query is None or query.size < min_len:
                return
            positions = informative_positions(query)
            if positions.size < min_len:
                return
            q_lo, q_hi = split_planes(query)
            s, o, _n = _scan(subset, q_lo, q_hi, threads, positions=positions)
        mu, sd = _null_stats(s[sample])
        if not np.isfinite(sd) or sd <= 0:
            return
        z = (mu - s[top]) / sd
        z = np.where(np.isfinite(z), z, -np.inf)
        top1 = float(z.max()) if z.size else -np.inf
        if chosen is None or top1 > chosen[0]:
            chosen = (top1, label, z, o[top])

    consider('as recorded', None)
    for label, variant in _query_variants(audio, sr):
        if label.startswith('speed'):
            speed_audio[label] = variant
        consider(label, variant)
    if chosen is not None and chosen[1] in speed_audio:
        speed_label = chosen[1]
        for phase_label, shifted in _phase_shifts(speed_audio[speed_label], sr):
            consider(f'{speed_label}, {phase_label}', shifted)
    return chosen


def _window_distance(a, offset_a, b, offset_b, length):
    luts = weight_luts()
    starts, lengths = _STATE['starts'], _STATE['lengths']
    n = min(int(length), int(lengths[a]) - offset_a, int(lengths[b]) - offset_b)
    if n <= 0 or offset_a < 0 or offset_b < 0:
        return 1.0
    lo = _STATE['lo'][starts[a] + offset_a: starts[a] + offset_a + n] ^ _STATE['lo'][starts[b] + offset_b: starts[b] + offset_b + n]
    hi = _STATE['hi'][starts[a] + offset_a: starts[a] + offset_a + n] ^ _STATE['hi'][starts[b] + offset_b: starts[b] + offset_b + n]
    return float((luts['lo'][lo].sum(dtype=np.int64) + luts['hi'][hi].sum(dtype=np.int64)) / (n * luts['total']))


def _lead_over_distinct(top, top_offsets, z, order, length):
    first = int(order[0])
    same = {first}
    for j in order[1:]:
        j = int(j)
        if not np.isfinite(z[j]):
            break
        if _window_distance(int(top[first]), int(top_offsets[first]), int(top[j]), int(top_offsets[j]), length) < _DUPLICATE_BER:
            same.add(j)
            continue
        return float(z[first] - z[j]), same
    return float('inf'), same


def identify(audio, sr, n_results):
    from tasks.chromaprint import fingerprint_audio, is_available

    if not is_available():
        raise RuntimeError('fpcalc is not available here, so a clip cannot be identified.')
    ensure_loaded()
    too_short = ValueError(
        f'The clip is too short to identify: at least {config.RECORDING_SEARCH_IDENTIFY_MIN_SECONDS:.0f} seconds are needed.'
    )
    if np.asarray(audio).shape[0] < config.RECORDING_SEARCH_IDENTIFY_MIN_SECONDS * sr:
        raise too_short
    min_len = max(_MIN_QUERY_FRAMES, int((config.RECORDING_SEARCH_IDENTIFY_MIN_SECONDS - _WINDOW_SECONDS) / HOP_SECONDS))
    query = fingerprint_audio(audio, sr)
    if query is None or query.size < min_len:
        raise too_short
    positions = informative_positions(query)
    if positions.size < min_len:
        raise ValueError(
            f'The clip is too noisy or too uniform to identify: only {positions.size} of its {query.size} '
            'fingerprint frames carry usable detail. Record closer to the speaker.'
        )
    with _TIMER.lock():
        _arm_idle_unload()
    q_lo, q_hi = split_planes(query)
    ids = _STATE['ids']
    n_tracks = int(ids.size)
    threads = _thread_count()
    started = time.time()
    stage1, _offsets1, scanned = _scan(np.arange(n_tracks), q_lo, q_hi, threads, _STAGE1_STRIDE, positions)
    pool = _candidate_pool(stage1, n_tracks)
    if not pool.size:
        return []
    sample = _null_sample(n_tracks, pool, np.random.default_rng(int(query[0])))
    stage2, offsets, _n = _scan(np.concatenate([pool, sample]), q_lo, q_hi, threads, positions=positions)
    keep = max(int(n_results), int(config.RECORDING_SEARCH_IDENTIFY_RERANK))
    ranked_pool = pool[np.argsort(stage2[pool])]
    ranked_pool = ranked_pool[np.isfinite(stage2[ranked_pool])]
    if not ranked_pool.size:
        return []
    if sample.size < _MIN_NULL_SAMPLE:
        sample = ranked_pool
    threshold = expected_extreme_z(scanned) + config.RECORDING_SEARCH_IDENTIFY_MARGIN
    top = ranked_pool[:keep]
    chosen = _select_variant(audio, sr, top, sample, stage2, offsets, min_len, threads)
    if chosen is None:
        return []
    top1, label, z, top_offsets = chosen
    order = np.argsort(-z)
    lead, same = _lead_over_distinct(top, top_offsets, z, order, query.size)
    identified = bool(top1 >= threshold and lead >= float(config.RECORDING_SEARCH_IDENTIFY_LEAD))
    logger.info(
        'Chromaprint identify: %d tracks, pool %d, %d of %d frames used, %.1fs with %d threads, variant %s, '
        'best z %.2f (threshold %.2f), lead %.2f over the next distinct recording, identified %s',
        scanned, pool.size, positions.size, query.size, time.time() - started, threads, label, top1, threshold, lead, identified,
    )
    rows = []
    for j in order[: int(n_results)]:
        j = int(j)
        k = int(top[j])
        rows.append(
            {
                'item_id': str(ids[k]),
                'ber': round(float(stage2[k]), 4),
                'z': round(float(z[j]), 2),
                'offset_seconds': round(float(top_offsets[j]) * HOP_SECONDS, 1),
                'identified': bool(identified and j in same),
                'lead': round(lead, 2) if j in same and np.isfinite(lead) else None,
                'variant': label,
            }
        )
    return rows
