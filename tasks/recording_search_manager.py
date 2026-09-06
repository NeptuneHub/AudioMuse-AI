# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Identify a recorded or uploaded clip against the library.

A short clip (a phone recording, a file) is matched against what the analysis
already stores for every track: the chromaprint of its first two minutes, its
neural fingerprint sequence, or the words Whisper hears in it. One mode per
tab; the similarity embeddings (MusiCNN, DCLAP) were dropped from this page
after measurement showed a phone recording lands nowhere near its own song in
those spaces.

Main Features:
* decode_clip streams the upload (bytes or a file-like object) to a temporary
  file, refusing it past RECORDING_SEARCH_MAX_UPLOAD_MB while copying so a big
  file never sits in RAM, decodes whatever container PyAV understands (the
  browser's webm/opus recording, mp3, flac, m4a, a video) through the analysis
  loader at the clip's native rate, and cuts it to
  RECORDING_SEARCH_MAX_CLIP_SECONDS. The loader itself never decodes more than
  AUDIO_LOAD_TIMEOUT seconds, so a 1 GB file costs disk and transfer, not memory.
* normalize_level brings the clip to RECORDING_SEARCH_TARGET_LEVEL_DB RMS and
  quiet_clip_warning flags a clip below RECORDING_SEARCH_QUIET_LEVEL_DB before
  the gain is applied; a recording made too far from the speaker cannot be
  rescued by gain, so the page says to record closer.
* search_identify: the clip's fpcalc fingerprint slid across the chromaprints
  the duplicate detector already stored (tasks.chromaprint_identify); only the
  first 120 s of each track exist there.
* search_neural: the clip's neural fingerprint sequence aligned on the stored
  sequences of every track (tasks.neural_fingerprint_index), any part of a
  song; rows carry score, votes, offset_seconds, identified and lead.
* search_lyrics transcribes the clip with Whisper and searches the GTE lyrics
  index with the words.
* run_recording_search is the entry point, one mode per tab. ValueError means
  the clip is at fault, RuntimeError means an index or model is unavailable.
* warmup_recording_models starts the chromaprint and neural packs loading in
  the web process and arms the shared idle timer that releases them, with
  Whisper and the neural session, after RECORDING_SEARCH_WARMUP_DURATION
  seconds without a query.
"""

import logging
import os
import re
import tempfile

import numpy as np

import config
from tasks.idle_unload import IdleUnloadTimer

logger = logging.getLogger(__name__)

MODES = ('identify', 'neural', 'lyrics')
MODE_LABELS = {
    'identify': 'Chromaprint', 'neural': 'Neural fingerprint', 'lyrics': 'Lyrics (GTE)',
}

_SILENCE_RMS = 1e-5
_COPY_CHUNK_BYTES = 1024 * 1024
_SUFFIX_RE = re.compile(r'[^A-Za-z0-9.]')

_TIMER = IdleUnloadTimer()


def _safe_suffix(filename):
    suffix = _SUFFIX_RE.sub('', os.path.splitext(filename or '')[1])[:8]
    return suffix if suffix.startswith('.') and len(suffix) > 1 else '.bin'


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


def quiet_clip_warning(level_db):
    if level_db >= config.RECORDING_SEARCH_QUIET_LEVEL_DB:
        return None
    return (
        f'The clip is very quiet ({level_db:.0f} dBFS): the music is probably below the '
        'microphone noise, so the results will not be reliable. Hold the phone close to the '
        'speaker, raise the volume and record again.'
    )


def normalize_level(audio, target_db=None):
    if target_db is None:
        target_db = config.RECORDING_SEARCH_TARGET_LEVEL_DB
    clip = np.asarray(audio, dtype=np.float32)
    gain = 10.0 ** ((target_db - clip_level_db(clip)) / 20.0)
    return np.clip(clip * gain, -1.0, 1.0).astype(np.float32)


def _unload_expired():
    from lyrics import whisper_onnx
    from tasks import chromaprint_identify, neural_fingerprint, neural_fingerprint_index

    with _TIMER.lock():
        whisper_onnx.unload()
        chromaprint_identify.unload()
        neural_fingerprint_index.unload()
        neural_fingerprint.unload_session()
    logger.info(
        'Recording search models unloaded after %ss idle', config.RECORDING_SEARCH_WARMUP_DURATION
    )


def _arm_idle_unload():
    duration = config.RECORDING_SEARCH_WARMUP_DURATION
    if _TIMER.arm(duration, _unload_expired):
        logger.info('Recording search models loaded; idle unload in %ss', duration)


def transcribe_clip(audio, sr):
    from lyrics.lyrics_transcriber import transcribe_clip as _transcribe_clip

    with _TIMER.lock():
        _arm_idle_unload()
        return _transcribe_clip(audio, sr)


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


def search_lyrics(audio, sr, n_results):
    from tasks.lyrics_manager import get_cache_stats, search_by_text

    if not config.LYRICS_ENABLED:
        raise RuntimeError('Lyrics search is disabled (LYRICS_ENABLED=false).')
    if not get_cache_stats().get('index_loaded'):
        raise RuntimeError('The lyrics index is not loaded. Run analysis first.')
    text = transcribe_clip(audio, sr)
    if not text:
        raise ValueError('No words were recognised in the clip.')
    return search_by_text(text, limit=n_results), text


def search_identify(audio, sr, n_results):
    from tasks import chromaprint_identify

    return _rows_with_metadata(chromaprint_identify.identify(audio, sr, n_results)), None


def search_neural(audio, sr, n_results):
    from tasks import neural_fingerprint_index

    with _TIMER.lock():
        _arm_idle_unload()
    return _rows_with_metadata(neural_fingerprint_index.identify(audio, sr, n_results)), None


_SEARCHERS = {'identify': search_identify, 'neural': search_neural, 'lyrics': search_lyrics}


def run_recording_search(clip, filename, mode, n_results=None):
    if mode not in MODES:
        raise ValueError(f'Unknown mode {mode!r}; expected one of {", ".join(MODES)}.')
    if n_results is None:
        n_results = config.RECORDING_SEARCH_DEFAULT_N_RESULTS
    n_results = max(1, int(n_results))
    audio, sr = decode_clip(clip, filename)
    level_db = clip_level_db(audio)
    audio = normalize_level(audio)
    payload = {
        'mode': mode,
        'clip_seconds': round(audio.shape[0] / sr, 1),
        'clip_level_db': round(float(level_db), 1),
        'transcript': None,
        'warnings': [],
    }
    quiet = quiet_clip_warning(level_db)
    if quiet:
        logger.warning('Recording search: %s', quiet)
        payload['warnings'].append(quiet)
    rows, transcript = _SEARCHERS[mode](audio, sr, n_results)
    if transcript:
        payload['transcript'] = transcript
    payload['results'] = rows
    payload['count'] = len(rows)
    return payload


def warmup_recording_models(include_lyrics=False):
    from tasks import chromaprint_identify, neural_fingerprint_index

    loaded = {}
    with _TIMER.lock():
        identify_status = chromaprint_identify.get_status()
        if identify_status['available'] and not identify_status['loaded']:
            chromaprint_identify.start_background_load()
        loaded['identify'] = bool(identify_status['loaded'])
        neural_status = neural_fingerprint_index.get_status()
        if neural_status['available'] and not neural_status['loaded']:
            neural_fingerprint_index.start_background_load()
        loaded['neural'] = bool(neural_status['loaded'])
        if include_lyrics and config.LYRICS_ENABLED:
            from lyrics import whisper_onnx

            try:
                whisper_onnx.load_whisper_model()
                loaded['whisper'] = True
            except Exception:
                logger.exception('Whisper warmup for recording search failed')
                loaded['whisper'] = False
        _arm_idle_unload()
    return {
        'loaded': any(loaded.values()),
        'models': loaded,
        'expiry_seconds': config.RECORDING_SEARCH_WARMUP_DURATION,
    }


def _pack_state(status, missing_reason):
    if not status['available']:
        return missing_reason
    if status['loaded']:
        return 'ready'
    if status['building']:
        return 'building'
    return 'not built yet'


def get_index_status():
    from tasks import chromaprint_identify, neural_fingerprint_index
    from tasks.lyrics_manager import get_cache_stats as lyrics_cache_stats

    return {
        'identify': _pack_state(chromaprint_identify.get_status(), 'needs fpcalc'),
        'neural': _pack_state(neural_fingerprint_index.get_status(), 'needs the model file'),
        'lyrics': bool(config.LYRICS_ENABLED and lyrics_cache_stats().get('index_loaded')),
    }
