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
  clip is at fault, neural_fingerprint_index.IndexUnavailable means the index
  or the model is not ready (its text is safe to show).
* search_by_track is the entry point for a library song: its stored
  fingerprint is cut into up to three 20-second windows (a fifth, half and
  four fifths of the way in), each is aligned on every other track, and the
  best score per song is kept. Like similar song it then drops repeats of the
  song or a kept result (same title and author, or a fingerprint twin: mean
  cosine plus a 20 s alignment) and caps MAX_SONGS_PER_ARTIST, with an index
  margin so the count holds. No audio is decoded and no model runs.
* smart_fade_order: one path by end/start match, BPM, intensity, MusiCNN (30 s edges, no silence) or a kept order
* warmup_recording_models maps the index if Flask has not, preloads the encoder
  and arms the idle timer (every search re-arms it) that frees the session and
  the cell cache after RECORDING_SEARCH_WARMUP_DURATION seconds without a query.
"""

import logging
import os
import re
import tempfile

import numpy as np

import config
from tasks.album_creation_manager import INTENSITY_WEIGHTS, arousal_of, centered_moods, folded_tempo, zscores
from tasks.idle_unload import IdleUnloadTimer
from tasks.memory_utils import release_memory_to_os
from tasks.search_shaping import name_key_for
from tasks.smart_fade_mix import EDGE_SECONDS, tempos_match

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
    from tasks import neural_fingerprint, neural_fingerprint_index

    with _TIMER.lock():
        neural_fingerprint.unload_session()
        neural_fingerprint_index.drop_cell_cache()
    release_memory_to_os()
    logger.info(
        'Recording search encoder and cell cache released after %ss idle', config.RECORDING_SEARCH_WARMUP_DURATION
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
_FADE_WINDOW_ROWS = 8
_FADE_STEP_ROWS = 2
_FADE_SILENCE_COSINE = 0.95
_FADE_TEMPO_PENALTY = 0.5
_FADE_INTENSITY_WEIGHT = 0.04
_FADE_SOUND_WEIGHT = 0.1
_FADE_WEAK_WEIGHT = 1.0
_FADE_SEGMENT = 3
_FADE_MOVES_PER_SONG = 2
_FADE_MIN_GAIN = 1e-6
_TWIN_ALIGN_SCORE = 0.5


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


def fade_edges(rows, decode=None):
    from tasks.neural_fingerprint import HOP_SECONDS

    count = int(rows.shape[0])
    edge = max(1, min(int(EDGE_SECONDS / HOP_SECONDS), count // 2))
    decode = decode or (lambda part: part)
    return decode(rows[:edge]), decode(rows[count - edge:]), count


def _stored_edges(item_ids):
    from database import get_db
    from tasks import neural_fingerprint
    from tasks.neural_fingerprint_index import iter_codes

    found = {}
    for item_id, codes in iter_codes(get_db(), list(dict.fromkeys(item_ids))):
        if codes is not None and codes.shape[0]:
            found[item_id] = fade_edges(codes, neural_fingerprint.decode_codes)
    return found


def _stored_twins(item_ids):
    from database import get_db
    from tasks import neural_fingerprint
    from tasks.neural_fingerprint_index import iter_codes

    found = {}
    for item_id, codes in iter_codes(get_db(), list(dict.fromkeys(item_ids))):
        if codes is not None and codes.shape[0]:
            mean = neural_fingerprint.decode_codes(codes).mean(axis=0)
            found[item_id] = {'mean': mean / max(float(np.linalg.norm(mean)), 1e-9), 'codes': codes}
    return found


def _alignment(rows, other):
    best = -1.0
    for window in query_windows(rows):
        width = window.shape[0]
        if other.shape[0] >= width:
            sims = window @ other.T
            span = other.shape[0] - width + 1
            best = max(best, float(sum(sims[k, k:k + span] for k in range(width)).max()) / width)
    return best


def _without_repeats(item_id, ranked, limit):
    from tasks import neural_fingerprint

    threshold = float(config.DUPLICATE_DISTANCE_THRESHOLD_NEURAL_FINGERPRINT or 0.0)
    cap = int(config.MAX_SONGS_PER_ARTIST or 0)
    ids = [item_id] + [row['item_id'] for row in ranked]
    metadata = _fetch_track_metadata(ids)
    songs = _stored_twins(ids) if threshold > 0 else {}
    decoded = {}

    def rows(song_id):
        if song_id not in decoded:
            decoded[song_id] = neural_fingerprint.decode_codes(songs[song_id]['codes'])
        return decoded[song_id]

    def twins(a, b):
        if a not in songs or b not in songs or float(songs[a]['mean'] @ songs[b]['mean']) < 1.0 - threshold:
            return False
        return max(_alignment(rows(a), rows(b)), _alignment(rows(b), rows(a))) >= _TWIN_ALIGN_SCORE

    seed = metadata.get(item_id) or {}
    names = {name_key_for(seed.get('title'), seed.get('author'))} - {None}
    claimed, kept, artists, dropped = [item_id], [], {}, 0
    for row in ranked:
        if len(kept) >= limit:
            break
        info = metadata.get(row['item_id'])
        if not info:
            continue
        name = name_key_for(info.get('title'), info.get('author'))
        artist = (info.get('author') or '').strip().lower()
        repeat = name is not None and name in names
        capped = bool(artist) and 0 < cap <= artists.get(artist, 0)
        if repeat or capped or any(twins(row['item_id'], other) for other in claimed):
            dropped += 1
            continue
        kept.append(row)
        claimed.append(row['item_id'])
        if name is not None:
            names.add(name)
        if artist:
            artists[artist] = artists.get(artist, 0) + 1
    if dropped:
        logger.info('Search by song left out %d songs: repeats, fingerprint twins or past the per-artist cap', dropped)
    return kept


def _stored_tempos(item_ids):
    from database import get_db

    cur = get_db().cursor()
    try:
        cur.execute('SELECT item_id, tempo FROM score WHERE item_id = ANY(%s)', (list(dict.fromkeys(item_ids)),))
        return dict(cur.fetchall())
    finally:
        cur.close()


def _tempo_penalty(tempos):
    values = np.array([float(tempo or 0.0) for tempo in tempos])
    known = values > 0.0
    values = np.where(known, values, 1.0)
    clash = ~tempos_match(values[:, None], values[None, :]) & known[:, None] & known[None, :]
    return _FADE_TEMPO_PENALTY * clash.astype(np.float32)


def _fade_windows(vectors, width):
    rows = np.asarray(vectors, dtype=np.float32)
    rows = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
    still = np.einsum('ij,ij->i', rows[1:], rows[:-1]) >= _FADE_SILENCE_COSINE
    silent = np.zeros(rows.shape[0], dtype=np.int32)
    silent[1:] |= still
    silent[:-1] |= still
    silent_sums = np.concatenate([[0], np.cumsum(silent)])
    sums = np.cumsum(np.vstack([np.zeros((1, rows.shape[1]), dtype=np.float32), rows]), axis=0)
    windows = sums[width:] - sums[:-width]
    windows /= np.maximum(np.linalg.norm(windows, axis=1, keepdims=True), 1e-9)
    return windows, silent_sums[width:] == silent_sums[:-width]


def _fade_candidates(clean, step):
    positions = np.arange(0, clean.size, step) if step > 0 else np.arange(clean.size - 1, -1, step)
    kept = positions[clean[positions]]
    return kept if kept.size else positions


def _fade_profile(edges):
    head, tail, count = edges
    width = min(_FADE_WINDOW_ROWS, head.shape[0])
    (intro_windows, intro_clean), (outro_windows, outro_clean) = _fade_windows(head, width), _fade_windows(tail, width)
    intro = _fade_candidates(intro_clean, _FADE_STEP_ROWS)
    outro = _fade_candidates(outro_clean, -_FADE_STEP_ROWS)
    return {
        'rows': count, 'width': width, 'intro': intro, 'outro': outro + count - tail.shape[0],
        'intro_windows': intro_windows[intro], 'outro_windows': outro_windows[outro],
    }


def _fade_times(profile, start_row, end_row):
    from tasks.neural_fingerprint import HOP_SECONDS, SAMPLE_RATE, SEGMENT_SAMPLES

    segment = SEGMENT_SAMPLES / SAMPLE_RATE
    last_row = profile['rows'] - 1
    return {
        'fade_in_seconds': round(start_row * HOP_SECONDS, 1),
        'fade_out_seconds': round((last_row if end_row is None else end_row) * HOP_SECONDS + segment, 1),
        'duration_seconds': round(last_row * HOP_SECONDS + segment, 1),
    }


def _fades_in_order(edges):
    profiles = [None if edge is None else _fade_profile(edge) for edge in edges]
    fade_in, fade_out = {}, {}
    for current in range(len(profiles) - 1):
        ending, opening = profiles[current], profiles[current + 1]
        if ending is None or opening is None:
            continue
        sims = ending['outro_windows'] @ opening['intro_windows'].T
        best_outro, best_intro = np.unravel_index(int(np.argmax(sims)), sims.shape)
        fade_out[current] = int(ending['outro'][best_outro]) + ending['width'] - 1
        fade_in[current + 1] = int(opening['intro'][best_intro])
    return [
        (index, None if profile is None else _fade_times(profile, fade_in.get(index, 0), fade_out.get(index)))
        for index, profile in enumerate(profiles)
    ]


def _pair_matrix(profiles):
    intro_windows = np.vstack([profile['intro_windows'] for profile in profiles])
    sizes = np.array([profile['intro'].size for profile in profiles])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    lanes = np.arange(int(sizes.max()))
    slots = np.where(lanes[None, :] < sizes[:, None], starts[:, None] + lanes[None, :], intro_windows.shape[0])
    count = len(profiles)
    every = np.arange(count)
    best = np.empty((count, count), dtype=np.float32)
    outro_at = np.empty((count, count), dtype=np.int64)
    intro_at = np.empty((count, count), dtype=np.int64)
    for current, profile in enumerate(profiles):
        sims = profile['outro_windows'] @ intro_windows.T
        rows = sims.argmax(axis=0)
        top = np.append(sims[rows, np.arange(sims.shape[1])], -np.inf)
        cells = top[slots]
        pick = cells.argmax(axis=1)
        column = slots[every, pick]
        best[current] = cells[every, pick]
        intro_at[current] = pick
        outro_at[current] = rows[np.minimum(column, rows.size - 1)]
    return best, outro_at, intro_at


def _intensity(traits, tempos):
    moods = [centered_moods((trait or {}).get('moods') or '') for trait in traits]
    signals = {
        'arousal': zscores([arousal_of(mood) for mood in moods]),
        'energy': zscores([(trait or {}).get('energy') for trait in traits]),
        'tempo': zscores([folded_tempo(tempo) for tempo in tempos]),
    }
    return zscores(list(sum(weight * signals[name] for name, weight in INTENSITY_WEIGHTS.items())))


def _sound_matrix(traits):
    vectors = [(trait or {}).get('sound') for trait in traits]
    size = next((vector.size for vector in vectors if vector is not None and vector.size), 0)
    count = len(vectors)
    known = np.array([vector is not None and vector.size == size > 0 for vector in vectors])
    if known.sum() < 2:
        return np.zeros((count, count), dtype=np.float32)
    rows = np.stack([vector if ok else np.zeros(size, dtype=np.float32) for vector, ok in zip(vectors, known)]).astype(np.float32)
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
    pairs = known[:, None] & known[None, :] & ~np.eye(count, dtype=bool)
    sims = rows @ rows.T
    return np.where(pairs, sims - float(sims[pairs].mean()), 0.0).astype(np.float32)


def _path_gain(score):
    others = ~np.eye(score.shape[0], dtype=bool)
    others[:, 0] = False
    floor = float(np.median(score[others])) if others.any() else 0.0
    return score - _FADE_WEAK_WEIGHT * np.maximum(0.0, floor - score)


def _greedy_path(gain):
    path, left = [0], np.ones(gain.shape[0], dtype=bool)
    left[0] = False
    while left.any():
        following = int(np.argmax(np.where(left, gain[path[-1]], -np.inf)))
        path.append(following)
        left[following] = False
    return np.array(path)


def _better_path(path, gain):
    count = path.size
    for _move in range(_FADE_MOVES_PER_SONG * count):
        links = gain[path[:-1], path[1:]]
        lanes = np.arange(count - 1)[None, :]
        best, move = _FADE_MIN_GAIN, None
        for length in range(1, min(_FADE_SEGMENT, count - 2) + 1):
            starts = np.arange(1, count - length + 1)
            ends = starts + length - 1
            firsts, lasts = path[starts], path[ends]
            tail = ends + 1 < count
            lift = np.where(tail, gain[path[starts - 1], path[np.minimum(ends + 1, count - 1)]] - links[np.minimum(ends, count - 2)], 0.0) - links[starts - 1]
            slots = gain[path[:-1][None, :], firsts[:, None]] + gain[lasts[:, None], path[1:][None, :]] - links[None, :]
            slots[(lanes >= (starts - 1)[:, None]) & (lanes <= ends[:, None])] = -np.inf
            slots = np.concatenate([slots, np.where(tail, gain[path[-1], firsts], -np.inf)[:, None]], axis=1) + lift[:, None]
            row, column = np.unravel_index(int(np.argmax(slots)), slots.shape)
            if slots[row, column] > best:
                best, move = float(slots[row, column]), (int(starts[row]), length, int(column))
        if move is None:
            break
        start, length, column = move
        rest = np.concatenate([path[:start], path[start + length:]])
        place = rest.size if column == count - 1 else column + 1 - (length if column > start else 0)
        path = np.concatenate([rest[:place], path[start:start + length], rest[place:]])
    return path


def smart_fade_order(edges, tempos=None, keep_order=False, traits=None):
    if keep_order:
        return _fades_in_order(edges)
    usable = [index for index, edge in enumerate(edges) if edge is not None]
    missing = [(index, None) for index in sorted(set(range(len(edges))) - set(usable))]
    if not usable:
        return missing
    known_tempos = [tempos[index] if tempos else None for index in usable]
    known_traits = [traits[index] if traits else None for index in usable]
    profiles = [_fade_profile(edges[index]) for index in usable]
    best, outro_at, intro_at = _pair_matrix(profiles)
    intensity = _intensity(known_traits, known_tempos)
    score = (best - _tempo_penalty(known_tempos) - _FADE_INTENSITY_WEIGHT * np.abs(intensity[:, None] - intensity[None, :])
             + _FADE_SOUND_WEIGHT * _sound_matrix(known_traits))
    gain = _path_gain(score)
    path = _better_path(_greedy_path(gain), gain) if len(usable) > 2 else _greedy_path(gain)
    fade_in, fade_out = {int(path[0]): 0}, {}
    for current, following in zip(path[:-1].tolist(), path[1:].tolist()):
        fade_out[current] = int(profiles[current]['outro'][outro_at[current, following]]) + profiles[current]['width'] - 1
        fade_in[following] = int(profiles[following]['intro'][intro_at[current, following]])
    ordered = [(usable[n], _fade_times(profiles[n], fade_in[n], fade_out.get(n))) for n in path.tolist()]
    return ordered + missing


def _stored_traits(item_ids):
    from database import get_db

    cur = get_db().cursor()
    try:
        cur.execute(
            'SELECT s.item_id, s.energy, s.other_features, e.embedding FROM score s '
            'LEFT JOIN embedding e ON e.item_id = s.item_id WHERE s.item_id = ANY(%s)',
            (list(dict.fromkeys(item_ids)),),
        )
        return {
            item_id: {'energy': energy, 'moods': moods, 'sound': None if blob is None else np.frombuffer(bytes(blob), dtype=np.float32)}
            for item_id, energy, moods, blob in cur.fetchall()
        }
    finally:
        cur.close()


def smart_fade_playlist(item_ids, keep_order=False):
    edges = _stored_edges(item_ids)
    if keep_order:
        return smart_fade_order([edges.get(item_id) for item_id in item_ids], keep_order=True)
    tempos = _stored_tempos(item_ids)
    traits = _stored_traits(item_ids)
    return smart_fade_order(
        [edges.get(item_id) for item_id in item_ids], [tempos.get(item_id) for item_id in item_ids],
        traits=[traits.get(item_id) for item_id in item_ids],
    )


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
    with _TIMER.lock():
        _arm_idle_unload()
    best = {}
    for window in query_windows(vectors):
        for row in neural_fingerprint_index.identify_vectors(window, n_results + max(20, n_results // 4), exclude_ids=(item_id,)):
            current = best.get(row['item_id'])
            if current is None or row['score'] > current['score']:
                best[row['item_id']] = dict(row)
    kept = _without_repeats(item_id, sorted(best.values(), key=lambda row: -row['score']), n_results)
    rows = _rows_with_metadata(neural_fingerprint_index.flag_identified(kept))
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
    if status['loaded']:
        return 'ready'
    if status['building']:
        return 'loading'
    if status.get('error'):
        return status['error']
    return 'not loaded yet'


def _index_summary(loaded=False, song_count=0, cells=0, cache_mb=0.0, state=''):
    return {'loaded': bool(loaded), 'song_count': int(song_count), 'cells': int(cells), 'cache_mb': cache_mb, 'state': state}


def get_index_status():
    from tasks import neural_fingerprint, neural_fingerprint_index

    if not neural_fingerprint.is_enabled():
        return _index_summary(state='disabled (NEURAL_FINGERPRINT_ENABLED=false)')
    status = neural_fingerprint_index.get_status()
    return _index_summary(
        status['loaded'], status['tracks'], status['cells'], status['cache_mb'], _pack_state(status, 'needs the model file'),
    )
