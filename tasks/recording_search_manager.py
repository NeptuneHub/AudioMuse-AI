# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Identify a recorded or uploaded clip against the library's neural fingerprints.

A short clip (a phone recording, a file) is matched against the neural
fingerprint sequence the analysis stores for every track
(tasks.neural_fingerprint_index): which song, and where in it, the clip aligns
with. The chromaprint, lyrics and similarity-embedding modes this page once
had were dropped after measurement; the neural fingerprint is the one that
identifies a phone recording from any part of a song.

Main Features:
* decode_clip streams the upload (bytes or a file-like object) to a temporary
  file, refusing it past RECORDING_SEARCH_MAX_UPLOAD_MB while copying so a big
  file never sits in RAM, decodes whatever container PyAV understands (the
  browser's webm/opus recording, mp3, flac, m4a, a video's sound track)
  through the analysis loader at the clip's native rate, and cuts it to
  RECORDING_SEARCH_MAX_CLIP_SECONDS. The loader itself never decodes more than
  AUDIO_LOAD_TIMEOUT seconds, so a 1 GB file costs disk and transfer, not memory.
* normalize_level brings the clip to RECORDING_SEARCH_TARGET_LEVEL_DB RMS,
  refusing a silent clip, and the level before the gain is reported with the
  results.
* search_neural aligns the clip's fingerprint sequence on the stored sequences
  and returns rows with score, votes, offset_seconds, identified and lead plus
  the track's title, author and album.
* run_recording_search is the entry point for a clip. ValueError means the
  clip is at fault, RuntimeError means the index or the model is unavailable.
* search_by_track is the entry point for a library song: its stored
  fingerprint is cut into up to three 20-second windows (a fifth, half and
  four fifths of the way in), each is aligned on every other track, and the
  best score per song is kept, so the other recordings of that song (copies,
  remasters, compilations) come out first without the song finding itself.
  No audio is decoded and no model runs.
* warmup_recording_models maps the index when it is not (normally a no-op,
  Flask maps it at startup and keeps it), preloads the encoder session and
  arms the idle timer that releases that session, and only it, after
  RECORDING_SEARCH_WARMUP_DURATION seconds without a query.
"""

import logging
import os
import re
import tempfile

import numpy as np

import config
from tasks.idle_unload import IdleUnloadTimer

logger = logging.getLogger(__name__)

_SILENCE_RMS = 1e-5
_COPY_CHUNK_BYTES = 1024 * 1024
_SUFFIXES = {
    ext: ext for ext in (
        '.webm', '.weba', '.ogg', '.oga', '.opus', '.mp3', '.m4a', '.mp4', '.mov', '.aac', '.wav', '.flac',
        '.wma', '.caf', '.aiff', '.aif', '.3gp', '.amr', '.mkv', '.mka',
    )
}

_TIMER = IdleUnloadTimer()


def _safe_suffix(filename):
    extension = re.sub(r'[^a-z0-9.]', '', os.path.splitext(filename or '')[1].lower())
    return _SUFFIXES.get(extension, '.bin')


def _write_clip(clip, handle):
    limit = config.RECORDING_SEARCH_MAX_UPLOAD_MB * 1024 * 1024
    too_big = f'The clip is larger than {config.RECORDING_SEARCH_MAX_UPLOAD_MB} MB.'
    if isinstance(clip, (bytes, bytearray, memoryview)):
        if len(clip) > limit:
            raise ValueError(too_big)
        handle.write(clip)
        return len(clip)
    written = 0
    while True:
        chunk = clip.read(_COPY_CHUNK_BYTES)
        if not chunk:
            return written
        written += len(chunk)
        if written > limit:
            raise ValueError(too_big)
        handle.write(chunk)


def decode_clip(clip, filename):
    from tasks.analysis import decode_audio_once

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_safe_suffix(filename))
    try:
        written = _write_clip(clip, tmp)
        tmp.close()
        if written == 0:
            raise ValueError('The clip is empty.')
        audio, sr = decode_audio_once(tmp.name)
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            logger.warning('Could not remove the temporary clip %s', tmp.name)
    if audio is None or not sr or audio.size == 0:
        raise ValueError('The clip could not be decoded as audio.')
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    max_samples = int(config.RECORDING_SEARCH_MAX_CLIP_SECONDS * sr)
    if max_samples > 0 and audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    return audio, int(sr)


def clip_level_db(audio):
    clip = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(clip)))) if clip.size else 0.0
    if rms < _SILENCE_RMS:
        raise ValueError('The clip is silent.')
    return 20.0 * np.log10(rms)


def normalize_level(audio, target_db=None):
    if target_db is None:
        target_db = config.RECORDING_SEARCH_TARGET_LEVEL_DB
    clip = np.asarray(audio, dtype=np.float32)
    gain = 10.0 ** ((target_db - clip_level_db(clip)) / 20.0)
    return np.clip(clip * gain, -1.0, 1.0).astype(np.float32)


def _unload_expired():
    from tasks import neural_fingerprint

    with _TIMER.lock():
        neural_fingerprint.unload_session()
    logger.info(
        'Recording search encoder unloaded after %ss idle', config.RECORDING_SEARCH_WARMUP_DURATION
    )


def _arm_idle_unload():
    duration = config.RECORDING_SEARCH_WARMUP_DURATION
    if _TIMER.arm(duration, _unload_expired):
        logger.info('Recording search encoder loaded; idle unload in %ss', duration)


def _fetch_track_metadata(item_ids):
    from database import get_db
    from psycopg2.extras import DictCursor

    if not item_ids:
        return {}
    cur = get_db().cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            'SELECT item_id, title, author, album FROM score WHERE item_id = ANY(%s)',
            (list(item_ids),),
        )
        return {row['item_id']: dict(row) for row in cur.fetchall()}
    finally:
        cur.close()


def _rows_with_metadata(hits):
    metadata = _fetch_track_metadata([hit['item_id'] for hit in hits])
    rows = []
    for hit in hits:
        info = metadata.get(hit['item_id'])
        if not info:
            continue
        row = dict(hit)
        row.update(
            {
                'title': info.get('title') or '',
                'author': info.get('author') or '',
                'album': info.get('album') or '',
            }
        )
        rows.append(row)
    return rows


def search_neural(audio, sr, n_results):
    from tasks import neural_fingerprint_index

    with _TIMER.lock():
        _arm_idle_unload()
    return _rows_with_metadata(neural_fingerprint_index.identify(audio, sr, n_results))


_WINDOW_ROWS = 40
_WINDOW_POSITIONS = (0.2, 0.5, 0.8)


def _stored_fingerprint(item_id):
    from database import get_db

    cur = get_db().cursor()
    try:
        cur.execute('SELECT neural_fingerprint FROM embedding WHERE item_id = %s', (item_id,))
        row = cur.fetchone()
        return bytes(row[0]) if row and row[0] is not None else None
    finally:
        cur.close()


def query_windows(vectors):
    n = int(vectors.shape[0])
    if n <= _WINDOW_ROWS:
        return [vectors]
    starts = sorted({min(n - _WINDOW_ROWS, max(0, int(round(position * n)) - _WINDOW_ROWS // 2)) for position in _WINDOW_POSITIONS})
    return [vectors[start:start + _WINDOW_ROWS] for start in starts]


def search_by_track(item_id, n_results=None):
    from tasks import neural_fingerprint, neural_fingerprint_index

    if n_results is None:
        n_results = config.RECORDING_SEARCH_DEFAULT_N_RESULTS
    n_results = max(1, int(n_results))
    blob = _stored_fingerprint(item_id)
    if blob is None:
        raise ValueError('This song has no neural fingerprint yet; the next analysis run computes it.')
    codes = neural_fingerprint.decode_blob(blob)
    if codes is None or codes.shape[0] < 2:
        raise ValueError('The stored fingerprint of this song is unreadable; re-analyse it.')
    vectors = neural_fingerprint.decode_codes(codes)
    best = {}
    for window in query_windows(vectors):
        for row in neural_fingerprint_index.identify_vectors(window, n_results, exclude_ids=(item_id,)):
            current = best.get(row['item_id'])
            if current is None or row['score'] > current['score']:
                best[row['item_id']] = dict(row)
    ranked = neural_fingerprint_index.flag_identified(sorted(best.values(), key=lambda row: -row['score']))
    rows = _rows_with_metadata(ranked[:n_results])
    return {'item_id': item_id, 'results': rows, 'count': len(rows)}


def run_recording_search(clip, filename, n_results=None):
    if n_results is None:
        n_results = config.RECORDING_SEARCH_DEFAULT_N_RESULTS
    n_results = max(1, int(n_results))
    audio, sr = decode_clip(clip, filename)
    level_db = clip_level_db(audio)
    audio = normalize_level(audio)
    payload = {
        'clip_seconds': round(audio.shape[0] / sr, 1),
        'clip_level_db': round(float(level_db), 1),
    }
    rows = search_neural(audio, sr, n_results)
    payload['results'] = rows
    payload['count'] = len(rows)
    return payload


def warmup_recording_models():
    from tasks import neural_fingerprint, neural_fingerprint_index

    if not neural_fingerprint.is_enabled():
        return {'loaded': False, 'models': {'neural': False, 'encoder': False}, 'expiry_seconds': config.RECORDING_SEARCH_WARMUP_DURATION}
    with _TIMER.lock():
        status = neural_fingerprint_index.get_status()
        if status['available'] and not status['loaded']:
            neural_fingerprint_index.start_background_load()
        try:
            encoder = neural_fingerprint.warm_session()
        except Exception:
            logger.exception('Neural fingerprint encoder warmup failed')
            encoder = False
        _arm_idle_unload()
    return {
        'loaded': bool(status['loaded']),
        'models': {'neural': bool(status['loaded']), 'encoder': bool(encoder)},
        'expiry_seconds': config.RECORDING_SEARCH_WARMUP_DURATION,
    }


def _pack_state(status, missing_reason):
    if not status['available']:
        return missing_reason
    if status['loaded'] or status.get('synced'):
        return 'ready'
    if status['building']:
        return 'loading'
    if status.get('error'):
        return status['error']
    return 'not loaded yet'


def get_index_status():
    from tasks import neural_fingerprint, neural_fingerprint_index

    if not neural_fingerprint.is_enabled():
        return {'neural': 'disabled (NEURAL_FINGERPRINT_ENABLED=false)'}
    return {'neural': _pack_state(neural_fingerprint_index.get_status(), 'needs the model file')}
