# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Beat-matched transition plan between two songs at the smart fade points.

Turns the smart fade points (tasks.recording_search_manager) into the seconds
where the page cuts and mixes the two songs: numbers only, the page plays them.

Main Features:
* Decodes one window per song (mono; PyAV when libsndfile fails): the
  LOOK_SECONDS before the ending song's point, and at most
  smart_fade_beats.WINDOW_SECONDS from PHRASE_LEAD_SECONDS before the next
  song's point. smart_fade_beats gives each window's beats, downbeats, tempo,
  grid and bar position, cached per file and window.
* The next song takes the first song's tempo (half or double time too) within
  MAX_LOCK_CENTS; tempos_match is shared with the smart fade order.
* A lock needs a known bar position on both songs ('bar_phase' else): it
  starts on a downbeat of the ending song and the next song enters on one of
  its own. It is checked over the fewest BEAT_STEP steps reaching
  LOCK_SECONDS (down to MIN_BEAT_MIX_SECONDS), needs onsets within MATCH_MS
  at MATCH_SHARE and a steady beat on both ('beat_tracking'), and plays its
  last whole bars nearest the pair's mix length; entry_bar is the next song's
  bar in its own seconds, the step the voice check moves its cue by.
* lengths counts felt beats (MIX_BEATS, FADE_BEATS), fewer when both parts
  are dense (drum share SPARSE_SHARE down to DENSE_SHARE, drumless is
  sparse); _felt doubles a tempo under FOLD_BPM with FAST_ONSETS_PER_BEAT.
  NoRoom carries the crossfade length; without a beat PLAIN_FADE_SECONDS.
* The next song may enter on a section start: the largest Foote novelty within
  PHRASE_REACH_BEATS of its point, snapped to a downbeat for a lock.
* style 'blend', 'rise' (RISE_DB louder), 'percussion_mix' (drum-led parts,
  also a fade's alternative within PERCUSSION_LOCK_CENTS) or 'fade', from the
  beat whose mix dips least (FLOOR_GAIN_DB, EDGE_SECONDS).
* enabled reads the wizard's SWITCHES or this thread's switched_on set (the
  wizard's test): a lock tries percussion_mix, rise, blend; with none on, or
  lock=False (the API's replan when the voice check drops a lock), it is the
  fade ('switched_off') without onset checks.
* confidence 'high' locked, 'low' a fade into a DROP_LU quieter start, else
  'medium'; k_filters is the BS.1770 K-weighting shared with the levels.
* The windows and both energy curves (_levels) are read at once; _hpss is
  librosa's, its median filters split over HPSS_THREADS threads (same output).
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

import config
from tasks import smart_fade_beats, smart_fade_cache

RATE = 44100
LOCK_SECONDS = 15.0
MIN_BEAT_MIX_SECONDS = 10.0
MAX_BEAT_MIX_SECONDS = 21.0
MIN_MIX_SECONDS = 5.0
MAX_FADE_SECONDS = 10.0
PLAIN_FADE_SECONDS = 7.5
MIX_BEATS = (32, 24)
FADE_BEATS = (18, 13)
SPARSE_SHARE = 0.55
DENSE_SHARE = 0.30
DRUMLESS_SHARE = 0.12
FOLD_BPM = 100.0
FAST_ONSETS_PER_BEAT = 3.9
FEEL_HOP = 512
BEAT_STEP = 8
BAR_BEATS = 4
LOOK_SECONDS = 26.0
MAX_LOCK_CENTS = 60.0
PERCUSSION_LOCK_CENTS = 90.0
PERCUSSION_SECONDS = (8.0, 12.0)
PERCUSSION_RATIO = 1.5
PERCUSSION_BAND_HZ = 150.0
HPSS_KERNEL = 31
HPSS_THREADS = 4
OCTAVES = (1.0, 0.5, 2.0)
RISE_DB = 6.0
PHRASE_LEAD_SECONDS = 4.0
PHRASE_KERNEL_BEATS = 8
PHRASE_REACH_BEATS = 8
SPAN_MARGIN_SECONDS = 2.0
MATCH_MS = 30.0
MATCH_SHARE = 0.6
ONSET_HOP = 256
EDGE_SECONDS = 30.0
LEVEL_RATE = 11025
BLOCK_SECONDS = 0.4
STEP_SECONDS = 0.1
K_SHELF_HZ = 1500.0
K_SHELF_DB = 4.0
K_PASS_HZ = 38.0
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = 10.0
FULL_LU = 0.0
DROP_LU = 2.0
FLOOR_GAIN_DB = 3.0
SWITCHES = {
    'blend': 'SMART_FADE_BEAT_MIX_ENABLED',
    'rise': 'SMART_FADE_BEAT_RISE_ENABLED',
    'percussion_mix': 'SMART_FADE_PERCUSSION_MIX_ENABLED',
    'eq_blend': 'SMART_FADE_EQ_BLEND_ENABLED',
    'filter_rise': 'SMART_FADE_FILTER_RISE_ENABLED',
    'vocal_handoff': 'SMART_FADE_VOCAL_HANDOFF_ENABLED',
}
LOCKED_STYLES = ('percussion_mix', 'rise', 'blend')
_switched_on = ContextVar('smart_fade_switched_on', default=None)


class NoRoom(ValueError):
    def __init__(self, fade_seconds):
        super().__init__('No room for the mix around the fade points.')
        self.fade_seconds = fade_seconds


def enabled(style):
    if style not in SWITCHES:
        return True
    chosen = _switched_on.get()
    return style in chosen if chosen is not None else bool(getattr(config, SWITCHES[style]))


@contextmanager
def switched_on(styles):
    token = _switched_on.set(frozenset(styles))
    try:
        yield
    finally:
        _switched_on.reset(token)


def tempos_match(tempo_out, tempo_in, limit=MAX_LOCK_CENTS, ratios=OCTAVES):
    ratio = np.asarray(tempo_out, dtype=np.float64) / np.asarray(tempo_in, dtype=np.float64)
    return np.any([np.abs(1200.0 * np.log2(ratio * relation)) <= limit for relation in ratios], axis=0)


def _load(path, rate, start=0.0, seconds=None):
    import librosa
    import soundfile

    try:
        audio, _sr = librosa.load(path, sr=rate, mono=True, offset=max(0.0, start), duration=seconds)
    except soundfile.SoundFileRuntimeError as err:
        from tasks.analysis.song import robust_load_audio_with_fallback

        audio, _sr = robust_load_audio_with_fallback(path, target_sr=rate)
        if audio is None:
            raise RuntimeError(f'{os.path.basename(path)} could not be decoded.') from err
        begin = int(round(max(0.0, start) * rate))
        audio = audio[begin:] if seconds is None else audio[begin:begin + int(round(seconds * rate))]
    return np.asarray(audio, dtype=np.float32)


def _region(path, start, seconds):
    return _load(path, RATE, start, seconds)


def k_filters(rate):
    angle = 2.0 * math.pi * K_SHELF_HZ / rate
    gain = 10.0 ** (K_SHELF_DB / 40.0)
    root = math.sqrt(2.0 * gain) * math.sin(angle)
    cos = math.cos(angle)
    shelf_b = [gain * ((gain + 1) + (gain - 1) * cos + root), -2 * gain * ((gain - 1) + (gain + 1) * cos), gain * ((gain + 1) + (gain - 1) * cos - root)]
    shelf_a = [(gain + 1) - (gain - 1) * cos + root, 2 * ((gain - 1) - (gain + 1) * cos), (gain + 1) - (gain - 1) * cos - root]
    angle = 2.0 * math.pi * K_PASS_HZ / rate
    cos, sin = math.cos(angle), math.sin(angle)
    pass_b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]
    pass_a = [1 + sin, -2 * cos, 1 - sin]
    return (np.float32(shelf_b), np.float32(shelf_a)), (np.float32(pass_b), np.float32(pass_a))


def _k_weighted(audio, rate):
    from scipy.signal import lfilter

    (shelf_b, shelf_a), (pass_b, pass_a) = k_filters(rate)
    shelved = lfilter(shelf_b, shelf_a, audio.astype(np.float32))
    return lfilter(pass_b, pass_a, shelved)


def _lufs(energy):
    return -0.691 + 10.0 * np.log10(np.maximum(energy, 1e-12))


def _integrated(energy):
    loud = _lufs(energy)
    gated = energy[loud > ABSOLUTE_GATE_LUFS]
    if not gated.size:
        return ABSOLUTE_GATE_LUFS
    kept = energy[(loud > ABSOLUTE_GATE_LUFS) & (loud > float(_lufs(gated.mean())) - RELATIVE_GATE_LU)]
    return float(_lufs(kept.mean()))


def _levels(path):
    profile = ('energy', LEVEL_RATE, BLOCK_SECONDS, STEP_SECONDS, K_SHELF_HZ, K_SHELF_DB, K_PASS_HZ, ABSOLUTE_GATE_LUFS, RELATIVE_GATE_LU)
    return smart_fade_cache.remember(path, profile, lambda: _measure_levels(path))


def _measure_levels(path):
    audio = _load(path, LEVEL_RATE)
    weighted = _k_weighted(audio, LEVEL_RATE)
    power = np.concatenate([[0.0], np.cumsum(np.square(weighted, dtype=np.float64))])
    size = int(round(BLOCK_SECONDS * LEVEL_RATE))
    count = max(1, int((audio.size - size) / (STEP_SECONDS * LEVEL_RATE)) + 1)
    starts = np.minimum(np.round(np.arange(count) * STEP_SECONDS * LEVEL_RATE).astype(np.int64), audio.size)
    ends = np.minimum(starts + size, audio.size)
    energy = (power[ends] - power[starts]) / np.maximum(ends - starts, 1)
    energy.setflags(write=False)
    return {'energy': energy, 'loudness': _integrated(energy), 'seconds': audio.size / LEVEL_RATE}


def _relative_energy(levels, start, count):
    first = int(round((start - BLOCK_SECONDS / 2) / STEP_SECONDS))
    blocks = np.clip(first + np.arange(count), 0, levels['energy'].size - 1)
    return levels['energy'][blocks] / 10.0 ** ((levels['loudness'] + 0.691) / 10.0)


def _levels_pair(path_out, path_in):
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='smart-fade-energy') as pool:
        return tuple(pool.map(_levels, (path_out, path_in)))


def _mix_floor(ending, a_at, starting, b_at, seconds):
    count = max(1, int(round(seconds / STEP_SECONDS)))
    share = (np.arange(count) + 0.5) / count
    power = np.cos(np.pi / 2 * share) ** 2 * _relative_energy(ending, a_at, count) + np.sin(np.pi / 2 * share) ** 2 * _relative_energy(starting, b_at, count)
    width = min(count, int(round(1.0 / STEP_SECONDS)))
    return float(10.0 * np.log10(np.convolve(power, np.ones(width) / width, mode='valid').min() + 1e-12))


def _fuller_start(ending, a_start, candidates, default, starting, b_at, seconds):
    allowed = [float(c) for c in candidates if a_start + c + seconds >= ending['seconds'] - EDGE_SECONDS] + [default]
    floors = {c: _mix_floor(ending, a_start + c, starting, b_at, seconds) for c in allowed}
    best = max(floors, key=floors.get)
    return best if floors[best] - floors[default] >= FLOOR_GAIN_DB else default


def _part_lufs(levels, start, seconds):
    energy = levels['energy']
    first = min(max(0, int(math.ceil(start / STEP_SECONDS - 1e-9))), energy.size - 1)
    last = max(first, min(int(math.floor((start + seconds - BLOCK_SECONDS) / STEP_SECONDS + 1e-9)), energy.size - 1))
    return float(_lufs(energy[first:last + 1].mean()))


def _confidence(ending, mix_at, starting, in_at, seconds, in_seconds, locked):
    if locked:
        return 'high'
    out_level = _part_lufs(ending, mix_at, seconds) - ending['loudness']
    in_level = _part_lufs(starting, in_at, in_seconds) - starting['loudness']
    return 'low' if out_level >= FULL_LU and in_level - out_level <= -DROP_LU else 'medium'


def _events(audio):
    return smart_fade_beats.events(audio, RATE)


def _rhythm(path, audio, start, seconds, span=None):
    profile = ('rhythm', smart_fade_beats.VERSION, round(float(start), 3), round(float(seconds), 3))
    found = smart_fade_cache.remember(path, profile, lambda: _events(audio))
    return smart_fade_beats.rhythm(found['beats'], found['downbeats'], span)


def _stretch_rate(tempo_out, tempo_in, limit=MAX_LOCK_CENTS):
    for octave in OCTAVES:
        rate = tempo_out * octave / tempo_in
        if abs(1200.0 * math.log2(rate)) <= limit:
            return rate
    return 1.0


def _level(audio, start, seconds):
    part = audio[int(start * RATE):int((start + seconds) * RATE)].astype(np.float64)
    return 10.0 * np.log10(float(np.mean(part * part)) + 1e-12) if part.size else -120.0


def _medians(magnitude):
    from scipy.ndimage import median_filter

    harmonic, percussive = np.empty_like(magnitude), np.empty_like(magnitude)

    def filtered(job):
        target, size, part = job
        target[part] = median_filter(magnitude[part], size=size, mode='reflect')

    rows = [np.s_[chunk[0]:chunk[-1] + 1] for chunk in np.array_split(np.arange(magnitude.shape[0]), HPSS_THREADS) if chunk.size]
    columns = [np.s_[:, chunk[0]:chunk[-1] + 1] for chunk in np.array_split(np.arange(magnitude.shape[1]), HPSS_THREADS) if chunk.size]
    jobs = [(harmonic, (1, HPSS_KERNEL), part) for part in rows] + [(percussive, (HPSS_KERNEL, 1), part) for part in columns]
    with ThreadPoolExecutor(max_workers=HPSS_THREADS, thread_name_prefix='smart-fade-hpss') as pool:
        list(pool.map(filtered, jobs))
    return harmonic, percussive


def _hpss(audio):
    import librosa

    magnitude, phase = librosa.magphase(librosa.stft(audio))
    harmonic, percussive = _medians(magnitude)
    masks = (librosa.util.softmask(harmonic, percussive, power=2.0, split_zeros=True),
             librosa.util.softmask(percussive, harmonic, power=2.0, split_zeros=True))
    return [librosa.istft((magnitude * mask) * phase, dtype=audio.dtype, length=audio.shape[-1]) for mask in masks]


def drum_share(audio, start, seconds, sample_rate=RATE):
    from scipy.signal import butter, sosfilt

    part = audio[max(0, round(start * sample_rate)):round((start + seconds) * sample_rate)]
    if part.size < sample_rate // 2:
        return None
    above = butter(4, PERCUSSION_BAND_HZ, btype='highpass', fs=sample_rate, output='sos')
    harmonic, percussive = (sosfilt(above, layer) for layer in _hpss(part))
    return float(np.dot(percussive, percussive) / max(float(np.dot(harmonic, harmonic)), 1e-12))


def _percussive(audio, start, seconds, ratio=PERCUSSION_RATIO, sample_rate=RATE):
    share = drum_share(audio, start, seconds, sample_rate)
    return share is not None and share > 0 and share >= ratio


def _handoff(style, start, entry, seconds, rate=1.0, beat=None, handover=None):
    return {
        'style': style, 'mix_at': round(start, 4), 'in_at': round(entry, 4),
        'mix_seconds': round(seconds, 4), 'resume_at': round(entry + seconds * rate, 4),
        'stretch': round(rate, 4), 'beat_seconds': round(beat, 4) if beat else None,
        'swap_at': round(start + (seconds / 2 if handover is None else handover), 4),
    }


def _novelty(audio, grid):
    import librosa

    hop = 512
    groups = []
    for feature in (librosa.feature.chroma_stft(y=audio, sr=RATE, hop_length=hop),
                    librosa.feature.mfcc(y=audio, sr=RATE, n_mfcc=13, hop_length=hop),
                    librosa.feature.rms(y=audio, hop_length=hop)):
        frames = np.minimum(librosa.time_to_frames(grid, sr=RATE, hop_length=hop), feature.shape[1] - 1)
        rows = np.stack([feature[:, a:max(a + 1, b)].mean(axis=1) for a, b in zip(frames[:-1], frames[1:])])
        groups.append((rows - rows.mean(axis=0)) / (rows.std(axis=0) + 1e-9) / np.sqrt(rows.shape[1]))
    features = np.concatenate(groups, axis=1)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-9
    similarity = features @ features.T
    half = PHRASE_KERNEL_BEATS
    offsets = np.arange(-half, half) + 0.5
    side = np.sign(offsets) * np.exp(-0.5 * (offsets / (half / 2.0)) ** 2)
    kernel = np.outer(side, side)
    novelty = np.zeros(len(grid))
    for i in range(half, similarity.shape[0] - half + 1):
        novelty[i] = float(np.sum(kernel * similarity[i - half:i + half, i - half:i + half]))
    return np.maximum(novelty, 0.0)


def _phrase_entry(audio, grid, target):
    if len(grid) < 2 * PHRASE_KERNEL_BEATS + 2:
        return None
    first = min(int(np.searchsorted(grid, target - 1e-6)), len(grid) - 1)
    novelty = _novelty(audio, grid)
    low, high = max(0, first - PHRASE_REACH_BEATS), min(len(grid) - 1, first + PHRASE_REACH_BEATS)
    window = novelty[low:high + 1]
    return float(grid[low + int(np.argmax(window))]) if window.max() > 0.0 else None


def _nearest(times, target):
    return float(times[np.argmin(np.abs(times - target))]) if len(times) else None


def _onsets(audio, sample_rate=RATE):
    import librosa

    hop = round(ONSET_HOP * sample_rate / RATE)
    envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop)
    return envelope - envelope.mean()


def _in_step(song_out, mix_at, song_in, first_in, overlap, rate, beat, sample_rate=RATE, halves=False):
    hop = round(ONSET_HOP * sample_rate / RATE)
    out_part = _onsets(song_out[int(mix_at * sample_rate):int((mix_at + overlap) * sample_rate)], sample_rate)
    in_part = _onsets(song_in[int(first_in * sample_rate):int((first_in + overlap * rate) * sample_rate)], sample_rate)
    in_part = np.interp(np.arange(out_part.size) * rate, np.arange(in_part.size), in_part)
    reach = max(1, int(beat / 2 * sample_rate / hop))
    near = np.abs(np.arange(-reach, reach + 1)) * hop / sample_rate * 1000.0 <= MATCH_MS

    def aligned(a, b):
        size = a.size
        scores = np.array([float(np.dot(a[max(0, lag):size + min(0, lag)], b[max(0, -lag):size - max(0, lag)]))
                           for lag in range(-reach, reach + 1)])
        best = float(scores.max())
        return best > 0.0 and float(scores[near].max()) >= MATCH_SHARE * best

    if not aligned(out_part, in_part):
        return False
    if overlap < LOCK_SECONDS and not halves:
        return True
    return all(aligned(a - a.mean(), b - b.mean()) for a, b in zip(np.array_split(out_part, 2), np.array_split(in_part, 2)))


def _percussion_mix(song_out, song_in, beats_out, entries, end, beat, rate, a_start, b_start):
    shortest, longest = PERCUSSION_SECONDS
    bar = BAR_BEATS
    for count in range(bar * math.floor((longest + 0.05) / (bar * beat)), 0, -bar):
        seconds = count * beat
        if seconds < shortest:
            break
        positions = beats_out[beats_out + seconds <= min(end, song_out.size / RATE)]
        if not positions.size:
            continue
        at = float(positions[-1])
        if not _percussive(song_out, at, seconds):
            continue
        aligned = _percussion_entry(song_out, at, song_in, entries, seconds, rate, beat)
        if aligned is not None:
            return _handoff('percussion_mix', a_start + at, b_start + aligned, seconds, rate, beat)
    return None


def _percussion_entry(song_out, at, song_in, entries, seconds, rate, beat):
    for entry in entries:
        if entry + seconds * rate <= song_in.size / RATE and _percussive(song_in, entry, seconds * rate):
            aligned = _aligned_entry(song_out, at, song_in, entry, seconds, rate, beat, halves=True)
            if aligned is not None:
                return aligned
    return None


def _aligned_entry(song_out, at, song_in, entry, seconds, rate, beat, halves=False):
    if _in_step(song_out, at, song_in, entry, seconds, rate, beat, halves=halves):
        return entry
    a = _onsets(song_out[round(at * RATE):round((at + seconds) * RATE)])
    b = _onsets(song_in[round(entry * RATE):round((entry + seconds * rate) * RATE)])
    if not b.size:
        return None
    b = np.interp(np.arange(a.size) * rate, np.arange(b.size), b)
    reach = round(0.5 * beat * RATE / ONSET_HOP)
    offsets = []
    for left, right in zip(np.array_split(a, 2), np.array_split(b, 2)):
        if left.size <= 2 * reach + 1:
            return None
        left = left - left.mean()
        right = right - right.mean()
        core = left[reach:-reach]
        corr = np.correlate(right, core, mode='valid')
        best = int(np.argmax(corr))
        norm = np.linalg.norm(core) * np.linalg.norm(right[best:best + core.size])
        if corr[best] < 0.25 * max(float(norm), 1e-9):
            return None
        offsets.append((best - reach) * ONSET_HOP / RATE)
    if max(map(abs, offsets)) > min(0.09, 0.18 * beat) or abs(offsets[0] - offsets[1]) > 0.025:
        return None
    corrected = entry + float(np.mean(offsets)) * rate
    if 0 <= corrected <= song_in.size / RATE - seconds * rate and _in_step(song_out, at, song_in, corrected, seconds, rate, beat, halves=halves):
        return corrected
    return None


def _mix_beats(beat, beats_out, out_length, first_in, in_length, rate):
    beats = BEAT_STEP * math.ceil((LOCK_SECONDS - 0.001) / (BEAT_STEP * beat))
    while beats * beat >= MIN_BEAT_MIX_SECONDS:
        usable = beats_out[beats_out + beats * beat <= out_length]
        if usable.size and first_in + beats * beat * rate <= in_length:
            return beats, usable
        beats -= BEAT_STEP
    return None


def _played_beats(beats, beat, seconds):
    bars = math.floor(seconds / (BAR_BEATS * beat) + 0.5)
    return max(BAR_BEATS, min(beats, BAR_BEATS * bars))


def _felt(audio, tempo):
    import librosa

    onsets = librosa.onset.onset_detect(y=audio, sr=RATE, hop_length=FEEL_HOP, units='time')
    per_beat = onsets.size / max(audio.size / RATE, 1e-9) * 60.0 / tempo
    return 60.0 / tempo / (2.0 if tempo < FOLD_BPM and per_beat >= FAST_ONSETS_PER_BEAT else 1.0)


def _part(audio, start, seconds):
    return audio[max(0, int(round(start * RATE))):max(0, int(round((start + seconds) * RATE)))]


def _feel(audio, tempo, start, seconds, drums=None):
    part = _part(audio, start, seconds)
    if part.size < RATE:
        return 60.0 / tempo, None
    return _felt(part, tempo), drums.result() if drums is not None else drum_share(part, 0.0, part.size / RATE)


def density(shares):
    dense = [0.0 if share is None or share <= DRUMLESS_SHARE
             else float(np.clip((SPARSE_SHARE - share) / (SPARSE_SHARE - DENSE_SHARE), 0.0, 1.0)) for share in shares]
    return min(dense) if dense else 0.0


def lengths(beat, weight):
    def span(counts, low, high):
        return float(np.clip(beat * (counts[0] - weight * (counts[0] - counts[1])), low, high))

    return span(MIX_BEATS, MIN_BEAT_MIX_SECONDS, MAX_BEAT_MIX_SECONDS), span(FADE_BEATS, MIN_MIX_SECONDS, MAX_FADE_SECONDS)


def _locked_style(song_out, mix_at, song_in, first_in, overlap, rate):
    if (enabled('percussion_mix') and _percussive(song_out, mix_at, overlap)
            and _percussive(song_in, first_in, overlap * rate)):
        return 'percussion_mix'
    if enabled('rise') and _level(song_in, first_in, overlap * rate) - _level(song_out, mix_at, overlap) >= RISE_DB:
        return 'rise'
    return 'blend' if enabled('blend') else None


def _track_region(path, start, seconds, target, focus, span):
    audio = _region(path, start, seconds)
    part = _part(audio, *focus)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='smart-fade-drums') as side:
        drums = side.submit(drum_share, part, 0.0, part.size / RATE) if part.size >= RATE else None
        found = _rhythm(path, audio, start, seconds, span)
        section = _phrase_entry(audio, found['grid'], target) if target is not None else None
        return audio, found, section, _feel(audio, found['tempo'], *focus, drums=drums)


def _entries(times, at, section):
    after = times[times >= at - 1e-6]
    if not after.size:
        return []
    entries = [float(after[0])]
    snapped = _nearest(times, section) if section is not None else None
    if snapped is not None and snapped != entries[0]:
        entries.insert(0, snapped)
    return entries


def _fits(beat, starts, song_out, entries, song_in, rate):
    fits = [(entry, _mix_beats(beat, starts, song_out.size / RATE, entry, song_in.size / RATE, rate)) for entry in entries]
    return sorted(((entry, fit) for entry, fit in fits if fit is not None), key=lambda item: item[1][0], reverse=True)


def plan_transition(path_out, out_seconds, path_in, in_seconds, lock=True):
    a_start = max(0.0, out_seconds - LOOK_SECONDS)
    b_start = max(0.0, in_seconds - PHRASE_LEAD_SECONDS)
    lead = in_seconds - b_start
    cut = out_seconds - a_start
    b_seconds = min(lead + LOOK_SECONDS + 2.0, smart_fade_beats.WINDOW_SECONDS)
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='smart-fade-beats') as pool:
        energy = [pool.submit(_levels, path) for path in (path_out, path_in)]
        ending = pool.submit(_track_region, path_out, a_start, cut + 2.0, None, (cut - LOCK_SECONDS, LOCK_SECONDS),
                             (cut - MAX_BEAT_MIX_SECONDS - SPAN_MARGIN_SECONDS, cut + SPAN_MARGIN_SECONDS))
        starting = pool.submit(_track_region, path_in, b_start, b_seconds, lead, (lead, LOCK_SECONDS),
                               (lead - SPAN_MARGIN_SECONDS, lead + MAX_BEAT_MIX_SECONDS + SPAN_MARGIN_SECONDS))
        song_out, rhythm_out, _, (felt_out, drums_out) = ending.result()
        song_in, rhythm_in, section, (felt_in, drums_in) = starting.result()
    tempo_out, tempo_in = rhythm_out['tempo'], rhythm_in['tempo']
    beats_out, grid = rhythm_out['grid'], rhythm_in['grid']
    mix_length, fade_seconds = lengths((felt_out + felt_in) / 2.0, density((drums_out, drums_in)))
    rate = _stretch_rate(tempo_out, tempo_in)
    locked = bool(tempos_match(tempo_out, tempo_in))
    beat = 60.0 / tempo_out
    bars = rhythm_out['phase'] is not None and rhythm_in['phase'] is not None
    downbeats_out = smart_fade_beats.downbeat_times(rhythm_out)
    bar_entries = _entries(smart_fade_beats.downbeat_times(rhythm_in), lead, section) if bars else []
    entries = _entries(grid, lead, section)
    if not entries:
        raise ValueError('No steady beat to match.')
    fits = _fits(beat, beats_out, song_out, entries, song_in, rate)
    if not fits:
        raise NoRoom(round(fade_seconds, 4))
    first_in, (beats, usable) = fits[0]
    overlap = beats * beat
    mix_at = float(usable[np.argmin(np.abs(usable - (cut - overlap)))])
    alternatives = []
    reason = 'tempo_gap' if not locked else None
    if locked and not (lock and any(enabled(kind) for kind in LOCKED_STYLES)):
        locked, reason = False, 'switched_off'
    if locked and not bars:
        locked, reason = False, 'bar_phase'
    if locked and not (rhythm_out['steady'] and rhythm_in['steady']):
        locked, reason = False, 'beat_tracking'
    if locked:
        for entry, (count, positions) in _fits(beat, downbeats_out, song_out, bar_entries, song_in, rate):
            duration = count * beat
            at = float(positions[np.argmin(np.abs(positions - (cut - duration)))])
            aligned = _aligned_entry(song_out, at, song_in, entry, duration, rate, beat)
            if aligned is not None:
                first_in, beats, usable, mix_at, overlap = aligned, count, positions, at, duration
                break
        else:
            locked = False
            reason = 'beat_alignment'
    if locked:
        played = _played_beats(beats, beat, mix_length)
        mix_at += (beats - played) * beat
        beats, overlap = played, played * beat
        style = _locked_style(song_out, mix_at, song_in, first_in, overlap, rate)
        if style is None:
            locked, reason = False, 'switched_off'
    if not locked:
        style = 'fade'
        short_rate = _stretch_rate(tempo_out, tempo_in, PERCUSSION_LOCK_CENTS)
        suitable_tempo = tempos_match(tempo_out, tempo_in, PERCUSSION_LOCK_CENTS)
        if suitable_tempo and bar_entries and enabled('percussion_mix'):
            drums = _percussion_mix(song_out, song_in, downbeats_out, bar_entries, cut,
                                    beat, short_rate, a_start, b_start)
            if drums is not None:
                alternatives.append(drums)
        rate = 1.0
        beats = max(1, min(math.ceil(fade_seconds / beat - 1e-3), int((song_in.size / RATE - first_in) / beat),
                           int((song_out.size / RATE - beats_out[0]) / beat)))
        overlap = beats * beat
        usable = beats_out[beats_out + overlap <= song_out.size / RATE]
        mix_at = float(usable[np.argmin(np.abs(usable - (cut - overlap)))])
        ending, starting = (future.result() for future in energy)
        mix_at = _fuller_start(ending, a_start, usable, mix_at, starting, b_start + first_in, overlap)
    resume_in = first_in + overlap * rate
    plan = {
        'mix_at': round(a_start + mix_at, 4),
        'mix_seconds': round(overlap, 4),
        'in_at': round(b_start + first_in, 4),
        'resume_at': round(b_start + resume_in, 4),
        'stretch': round(rate, 4),
        'tempo_from': round(tempo_out, 1),
        'tempo_to': round(tempo_in, 1),
        'style': style,
        'swap_at': round(a_start + mix_at + beats // 2 * beat, 4),
        'beat_seconds': round(beat, 4),
        'fade_seconds': round(fade_seconds, 4),
        'entry_bar': round(BAR_BEATS * rhythm_in['period'], 4),
    }
    plan['confidence'] = 'high' if locked else _confidence(ending, a_start + mix_at, starting, b_start + first_in, overlap, overlap * rate, locked)
    if alternatives:
        plan['alternatives'] = alternatives
    if reason:
        plan['beat_rejection'] = reason
    return plan
