# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Search the library from a recorded or uploaded audio clip.

A short clip (a phone recording, a file) is turned into the same vectors the
analysis stores for every track and queried against the MusiCNN, DCLAP and
lyrics indexes; the combined mode fuses the three rankings by rank agreement.

Main Features:
* decode_clip streams the upload (bytes or a file-like object) to a temporary
  file, refusing it past RECORDING_SEARCH_MAX_UPLOAD_MB while copying so a big
  file never sits in RAM, decodes whatever container PyAV understands (the
  browser's webm/opus recording, mp3, flac, m4a, a video) through the analysis
  loader at the clip's native rate, and cuts it to
  RECORDING_SEARCH_MAX_CLIP_SECONDS. The loader itself never decodes more than
  AUDIO_LOAD_TIMEOUT seconds, so a 1 GB file costs disk and transfer, not memory.
* normalize_level brings the clip to RECORDING_SEARCH_TARGET_LEVEL_DB RMS. The
  mel front ends carry no per-clip normalisation, so a quiet recording lands far
  from its own song: -12 dB alone cost two thirds of the neighbourhood overlap
  when measured on 50 songs, and level is the one degradation a query can undo.
* quiet_clip_warning flags a clip below RECORDING_SEARCH_QUIET_LEVEL_DB before
  the gain is applied. Gain cannot help a recording made too far from the
  source: on a real -45 dBFS phone recording the music sat below the
  microphone's own noise from 200 Hz up, the true song ranked 90,803rd and
  Whisper heard no speech, and every denoiser or channel correction tried made
  it worse. The only fix is to record closer and louder, so the page says so.
* search_musicnn / search_dclap / search_lyrics give one ranked list per index.
  The lyrics list transcribes the clip with Whisper and searches the GTE lyrics
  index with the words, so it answers "what is sung" rather than "how it sounds".
* search_identify answers "which exact recording is this": the clip's fpcalc
  fingerprint slid across the chromaprints the duplicate detector already stored
  (tasks.chromaprint_identify).
  It is the one source that survives a phone recording whose music sits below
  the microphone noise, where every embedding fails.
* In combined mode the identify list weighs twice in the fusion, its ber, z,
  offset and identified flag are copied onto the fused rows, and rows the
  identifier is confident about are pinned to the top.
* fuse_by_rank is reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR
  2009): score = sum over the lists holding the track of 1 / (k + rank). Two
  lists agreeing on rank 20 (2/80) beat one list's rank 1 (1/61), which is the
  agreement rule the combined tab wants, and no score from one embedding space
  is ever added to a score from another.
* run_recording_search is the entry point, one mode per tab; in combined mode an
  index that cannot answer is reported as a warning instead of failing the
  request. ValueError means the clip is at fault, RuntimeError means an index or
  model is unavailable.
* warmup_recording_models loads the MusiCNN sessions and the DCLAP audio tower
  in the web process and arms the shared idle timer that unloads them, together
  with Whisper, after RECORDING_SEARCH_WARMUP_DURATION seconds without a query.
"""

import logging
import os
import re
import tempfile

import numpy as np

import config
from tasks.idle_unload import IdleUnloadTimer

logger = logging.getLogger(__name__)

MODES = ('musicnn', 'dclap', 'lyrics', 'identify', 'combined')
SOURCES = ('identify', 'musicnn', 'dclap', 'lyrics')
SOURCE_LABELS = {
    'identify': 'Identify (Chromaprint)', 'musicnn': 'MusiCNN', 'dclap': 'DCLAP', 'lyrics': 'Lyrics (GTE)',
}
_RRF_SOURCE_WEIGHTS = {'identify': 2.0}
_IDENTIFY_FIELDS = ('ber', 'z', 'offset_seconds', 'identified', 'lead')

_MUSICNN_SAMPLE_RATE = 16000
_SILENCE_RMS = 1e-5
_COPY_CHUNK_BYTES = 1024 * 1024
_SUFFIX_RE = re.compile(r'[^A-Za-z0-9.]')

_TIMER = IdleUnloadTimer()
_STATE = {'musicnn_sessions': None}


def _model_paths():
    return {'embedding': config.EMBEDDING_MODEL_PATH, 'prediction': config.PREDICTION_MODEL_PATH}


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
    from tasks import chromaprint_identify
    from tasks.analysis.song import cleanup_musicnn_sessions
    from tasks.clap_analyzer import unload_clap_audio_only

    with _TIMER.lock():
        sessions = _STATE['musicnn_sessions']
        _STATE['musicnn_sessions'] = None
        cleanup_musicnn_sessions(sessions, context='recording search idle')
        unload_clap_audio_only()
        whisper_onnx.unload()
        chromaprint_identify.unload()
    logger.info(
        'Recording search models unloaded after %ss idle', config.RECORDING_SEARCH_WARMUP_DURATION
    )


def _arm_idle_unload():
    duration = config.RECORDING_SEARCH_WARMUP_DURATION
    if _TIMER.arm(duration, _unload_expired):
        logger.info('Recording search models loaded; idle unload in %ss', duration)


def _musicnn_sessions():
    from tasks.analysis.song import load_musicnn_sessions

    if _STATE['musicnn_sessions'] is None:
        _STATE['musicnn_sessions'] = load_musicnn_sessions(_model_paths())
    return _STATE['musicnn_sessions']


def musicnn_embedding(audio, sr):
    from tasks.analysis import musicnn_embedding_for_audio, resample_audio

    clip = resample_audio(audio, sr, _MUSICNN_SAMPLE_RATE)
    with _TIMER.lock():
        sessions = _musicnn_sessions()
        if sessions is None:
            raise RuntimeError('The MusiCNN model could not be loaded.')
        _arm_idle_unload()
        vector = musicnn_embedding_for_audio(
            clip, _MUSICNN_SAMPLE_RATE, config.MOOD_LABELS, _model_paths(), sessions, 'recording'
        )
    if vector is None:
        raise ValueError(
            'The clip is too short: the MusiCNN model needs at least three seconds of audio.'
        )
    return np.asarray(vector, dtype=np.float32)


def dclap_embedding(audio, sr):
    from tasks.clap_analyzer import analyze_audio_file, initialize_clap_audio_model

    with _TIMER.lock():
        if not initialize_clap_audio_model():
            raise RuntimeError('The DCLAP audio model could not be loaded.')
        _arm_idle_unload()
        vector, _seconds, segments = analyze_audio_file(None, native_audio=audio, native_sr=sr)
    if vector is None or not segments:
        raise RuntimeError('The DCLAP model could not embed the clip.')
    return np.asarray(vector, dtype=np.float32)


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


def search_musicnn(audio, sr, n_results):
    from tasks import ivf_manager

    vector = musicnn_embedding(audio, sr)
    neighbors = ivf_manager.find_nearest_neighbors_by_vector(vector, n=n_results)
    metadata = _fetch_track_metadata([neighbor['item_id'] for neighbor in neighbors])
    index = ivf_manager.ivf_index
    rows = []
    for neighbor in neighbors:
        info = metadata.get(neighbor['item_id'])
        if not info:
            continue
        distance = float(neighbor['distance'])
        rows.append(
            {
                'item_id': neighbor['item_id'],
                'title': info.get('title') or '',
                'author': info.get('author') or '',
                'album': info.get('album') or '',
                'distance': distance,
                'similarity': (
                    float(index.distance_to_similarity(distance)) if index is not None else None
                ),
            }
        )
    return rows, None


def search_dclap(audio, sr, n_results):
    from tasks.clap_text_search import is_clap_cache_loaded, search_by_embedding

    if not config.CLAP_ENABLED:
        raise RuntimeError('DCLAP search is disabled (CLAP_ENABLED=false).')
    if not is_clap_cache_loaded():
        raise RuntimeError('The DCLAP index is not loaded. Run analysis first.')
    return search_by_embedding(dclap_embedding(audio, sr), limit=n_results), None


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

    hits = chromaprint_identify.identify(audio, sr, n_results)
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
    return rows, None


_SEARCHERS = {
    'identify': search_identify, 'musicnn': search_musicnn, 'dclap': search_dclap, 'lyrics': search_lyrics,
}


def _annotate_identified(results, identify_rows):
    by_id = {row['item_id']: row for row in identify_rows}
    for entry in results:
        hit = by_id.get(entry.get('item_id'))
        if hit:
            for field in _IDENTIFY_FIELDS:
                entry[field] = hit.get(field)
    pinned = [entry for entry in results if entry.get('identified')]
    if not pinned:
        return results
    pinned_ids = {entry['item_id'] for entry in pinned}
    return pinned + [entry for entry in results if entry['item_id'] not in pinned_ids]


def fuse_by_rank(ranked_lists, k=None, limit=None, weights=None):
    if k is None:
        k = config.RECORDING_SEARCH_RRF_K
    weights = weights or {}
    fused = {}
    for source, rows in ranked_lists.items():
        weight = float(weights.get(source, 1.0))
        for rank, row in enumerate(rows, start=1):
            item_id = row.get('item_id')
            if not item_id:
                continue
            entry = fused.get(item_id)
            if entry is None:
                entry = {
                    key: value
                    for key, value in row.items()
                    if key not in ('similarity', 'distance') and key not in _IDENTIFY_FIELDS
                }
                entry['ranks'] = {}
                entry['similarity_by_source'] = {}
                entry['rrf_score'] = 0.0
                fused[item_id] = entry
            if source in entry['ranks']:
                continue
            entry['ranks'][source] = rank
            entry['similarity_by_source'][source] = row.get('similarity')
            entry['rrf_score'] += weight / (k + rank)
    ordered = sorted(
        fused.values(),
        key=lambda e: (
            -e['rrf_score'],
            -len(e['ranks']),
            min(e['ranks'].values()),
            str(e['item_id']),
        ),
    )
    for entry in ordered:
        entry['agreement'] = len(entry['ranks'])
        entry['rrf_score'] = round(entry['rrf_score'], 6)
    return ordered[:limit] if limit else ordered


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
        'sources': [],
    }
    quiet = quiet_clip_warning(level_db)
    if quiet:
        logger.warning('Recording search: %s', quiet)
        payload['warnings'].append(quiet)
    lists = {}
    for source in (SOURCES if mode == 'combined' else (mode,)):
        try:
            rows, transcript = _SEARCHERS[source](audio, sr, n_results)
        except (ValueError, RuntimeError) as exc:
            if mode != 'combined':
                raise
            logger.warning('Recording search: %s skipped: %s', SOURCE_LABELS[source], exc)
            payload['warnings'].append(f'{SOURCE_LABELS[source]} skipped: {exc}')
            continue
        lists[source] = rows
        if transcript:
            payload['transcript'] = transcript
    if mode == 'combined':
        if not lists:
            raise RuntimeError('No index could answer this clip. ' + ' '.join(payload['warnings']))
        results = fuse_by_rank(lists, limit=n_results, weights=_RRF_SOURCE_WEIGHTS)
        if 'identify' in lists:
            results = _annotate_identified(results, lists['identify'])
    else:
        results = lists.get(mode, [])
    payload['sources'] = sorted(lists)
    payload['results'] = results
    payload['count'] = len(results)
    return payload


def warmup_recording_models(include_lyrics=False):
    from tasks.clap_analyzer import initialize_clap_audio_model

    from tasks import chromaprint_identify

    loaded = {}
    with _TIMER.lock():
        loaded['musicnn'] = _musicnn_sessions() is not None
        loaded['dclap'] = bool(config.CLAP_ENABLED and initialize_clap_audio_model())
        identify_status = chromaprint_identify.get_status()
        if identify_status['available'] and not identify_status['loaded']:
            chromaprint_identify.start_background_load()
        loaded['identify'] = bool(identify_status['loaded'])
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


def get_index_status():
    from tasks import chromaprint_identify, ivf_manager
    from tasks.clap_text_search import get_cache_stats as clap_cache_stats
    from tasks.lyrics_manager import get_cache_stats as lyrics_cache_stats

    identify = chromaprint_identify.get_status()
    if not identify['available']:
        identify_state = 'needs fpcalc'
    elif identify['loaded']:
        identify_state = 'ready'
    elif identify['building']:
        identify_state = 'building'
    else:
        identify_state = 'not built yet'
    return {
        'musicnn': ivf_manager.ivf_index is not None,
        'dclap': bool(config.CLAP_ENABLED and clap_cache_stats().get('loaded')),
        'lyrics': bool(config.LYRICS_ENABLED and lyrics_cache_stats().get('index_loaded')),
        'identify': identify_state,
    }
