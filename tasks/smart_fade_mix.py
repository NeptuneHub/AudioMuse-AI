# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Beat-matched transition plan between two songs at the smart fade points.

The smart fade (tasks.recording_search_manager) picks where one song should
end and the next start; this module turns those points into the seconds where
the page cuts and mixes the two songs. It returns numbers only: the page
fetches the songs and does the whole fade itself.

Main Features:
* A few seconds around each point are decoded (mono, one rate); a file
  libsndfile cannot open (AAC .m4a) goes through the analysis PyAV loader.
* Beats come from librosa's tracker, tempo and grid from a line fitted through
  the beat times. The next song takes the first song's tempo in the mix
  (stretch, half or double time too, none past MAX_LOCK_CENTS); tempos_match is
  that rule, shared with the smart fade order, so neighbours are picked to mix.
* The mix lasts at least MIX_SECONDS in whole BEAT_STEP steps, ending on the
  beat nearest the fade point; without room it loses steps down to
  MIN_MIX_SECONDS. No steady beat or no room raises ValueError.
* The next song enters on a section start: the beat nearest the highest peak,
  within PHRASE_REACH_BEATS, of a Foote novelty curve (PHRASE_KERNEL_BEATS
  checkerboard on beat-synchronous chroma, MFCC and loudness).
* A beat lock needs tempos_match: the two songs within MAX_LOCK_CENTS of each
  other in pitch (the page matches tempo by speed, which moves the pitch: a 3%
  gap is 50 cents, more puts two songs in one key a semitone apart), onsets
  lining up within MATCH_MS at
  MATCH_SHARE, and on both parts a committee of beat sequences (full, bass
  and treble band onsets, plus the pulse curve) agreeing by
  MIN_AGREEMENT_BITS of mean information gain (ISMIR 2012); about half of a
  50-song pop similar list locks.
* style: 'blend' (locked, bass crossing at swap_at) or 'rise' (also RISE_DB
  louder); else 'fade', the default crossfade: MIN_MIX_SECONDS unstretched,
  from the beat where the equal-power mix of the two loudness curves dips
  least when that lifts it FLOOR_GAIN_DB, never past EDGE_SECONDS.
* confidence: 'high' locked, 'low' a fade from FULL_LU into a next song
  DROP_LU lower (each against its own BS.1770 loudness), else 'medium'.
* k_filters is the BS.1770 K-weighting, shared with tasks.smart_fade_levels,
  which turns a plan into the curves a player executes.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tasks import smart_fade_cache

RATE = 44100
MIX_SECONDS = 15.0
MIN_MIX_SECONDS = 10.0
BEAT_STEP = 8
LOOK_SECONDS = 26.0
MAX_LOCK_CENTS = 50.0
OCTAVES = (1.0, 0.5, 2.0)
RISE_DB = 6.0
PHRASE_LEAD_SECONDS = 8.0
PHRASE_KERNEL_BEATS = 8
PHRASE_REACH_BEATS = 8
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
MIN_AGREEMENT_BITS = 1.0
AGREEMENT_BINS = 40
COMMITTEE_RATE = 22050
COMMITTEE_HOP = 512
COMMITTEE_MELS = 64
BASS_BAND_HZ = 200.0
TREBLE_BAND_HZ = 4000.0
FLOOR_GAIN_DB = 3.0


def tempos_match(tempo_out, tempo_in):
    ratio = np.asarray(tempo_out, dtype=np.float64) / np.asarray(tempo_in, dtype=np.float64)
    return np.any([np.abs(1200.0 * np.log2(ratio * octave)) <= MAX_LOCK_CENTS for octave in OCTAVES], axis=0)


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


def _beats(audio):
    import librosa

    _tempo, beats = librosa.beat.beat_track(y=audio, sr=RATE, units='time')
    beats = np.asarray(beats, dtype=np.float64)
    if beats.size < 4:
        raise ValueError('No steady beat to match.')
    index = np.round((beats - beats[0]) / np.median(np.diff(beats)))
    period, first = np.polyfit(index, beats, 1)
    grid = first + period * np.arange(index[-1] + 1)
    return 60.0 / float(period), grid[grid >= 0.0]


def _committee(audio):
    import librosa

    signal = librosa.resample(audio, orig_sr=RATE, target_sr=COMMITTEE_RATE)
    spectrum = librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=COMMITTEE_RATE, hop_length=COMMITTEE_HOP, n_mels=COMMITTEE_MELS))
    bands = librosa.mel_frequencies(n_mels=COMMITTEE_MELS, fmax=COMMITTEE_RATE / 2)
    envelopes = [
        librosa.onset.onset_strength(S=spectrum[rows], sr=COMMITTEE_RATE, hop_length=COMMITTEE_HOP)
        for rows in (bands >= 0.0, bands < BASS_BAND_HZ, bands > TREBLE_BAND_HZ)
    ]
    sequences = [
        librosa.beat.beat_track(onset_envelope=envelope, sr=COMMITTEE_RATE, hop_length=COMMITTEE_HOP, units='time')[1]
        for envelope in envelopes
    ]
    pulse = librosa.beat.plp(onset_envelope=envelopes[0], sr=COMMITTEE_RATE, hop_length=COMMITTEE_HOP)
    sequences.append(librosa.frames_to_time(np.flatnonzero(librosa.util.localmax(pulse)), sr=COMMITTEE_RATE, hop_length=COMMITTEE_HOP))
    return [np.asarray(sequence, dtype=np.float64) for sequence in sequences]


def _error_bits(reference, estimate):
    if reference.size < 2 or estimate.size < 1:
        return 0.0
    nearest = np.abs(estimate[:, None] - reference[None, :]).argmin(axis=1)
    error = estimate - reference[nearest]
    gaps = np.diff(reference)
    after = gaps[np.minimum(nearest, gaps.size - 1)]
    before = gaps[np.maximum(nearest - 1, 0)]
    share = (error / np.where(error >= 0.0, after, before) + 0.5) % 1.0 - 0.5
    counts = np.histogram(share, bins=AGREEMENT_BINS, range=(-0.5, 0.5))[0]
    shares = counts[counts > 0] / counts.sum()
    return float(np.log2(AGREEMENT_BINS) + np.sum(shares * np.log2(shares)))


def _agreement(audio):
    sequences = _committee(audio)
    gains = [
        min(_error_bits(first, second), _error_bits(second, first))
        for n, first in enumerate(sequences) for second in sequences[n + 1:]
    ]
    return float(np.mean(gains))


def _stretch_rate(tempo_out, tempo_in):
    for octave in OCTAVES:
        rate = tempo_out * octave / tempo_in
        if abs(1200.0 * math.log2(rate)) <= MAX_LOCK_CENTS:
            return rate
    return 1.0


def _level(audio, start, seconds):
    part = audio[int(start * RATE):int((start + seconds) * RATE)].astype(np.float64)
    return 10.0 * np.log10(float(np.mean(part * part)) + 1e-12) if part.size else -120.0


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


def _phrase_entry(audio, target):
    try:
        _tempo, grid = _beats(audio)
    except ValueError:
        return None
    if len(grid) < 2 * PHRASE_KERNEL_BEATS + 2:
        return None
    first = min(int(np.searchsorted(grid, target - 1e-6)), len(grid) - 1)
    novelty = _novelty(audio, grid)
    low, high = max(0, first - PHRASE_REACH_BEATS), min(len(grid) - 1, first + PHRASE_REACH_BEATS)
    window = novelty[low:high + 1]
    return float(grid[low + int(np.argmax(window))]) if window.max() > 0.0 else None


def _onsets(audio):
    import librosa

    envelope = librosa.onset.onset_strength(y=audio, sr=RATE, hop_length=ONSET_HOP)
    return envelope - envelope.mean()


def _in_step(song_out, mix_at, song_in, first_in, overlap, rate, beat):
    out_part = _onsets(song_out[int(mix_at * RATE):int((mix_at + overlap) * RATE)])
    in_part = _onsets(song_in[int(first_in * RATE):int((first_in + overlap * rate) * RATE)])
    in_part = np.interp(np.arange(out_part.size) * rate, np.arange(in_part.size), in_part)
    reach = max(1, int(beat / 2 * RATE / ONSET_HOP))
    size = out_part.size
    scores = np.array([float(np.dot(out_part[max(0, lag):size + min(0, lag)], in_part[max(0, -lag):size - max(0, lag)])) for lag in range(-reach, reach + 1)])
    near = np.abs(np.arange(-reach, reach + 1)) * ONSET_HOP / RATE * 1000.0 <= MATCH_MS
    best = float(scores.max())
    return best > 0.0 and float(scores[near].max()) >= MATCH_SHARE * best


def _mix_beats(beat, beats_out, out_length, first_in, in_length, rate):
    beats = BEAT_STEP * math.ceil(MIX_SECONDS / (BEAT_STEP * beat))
    while beats * beat >= MIN_MIX_SECONDS:
        usable = beats_out[beats_out + beats * beat <= out_length]
        if usable.size and first_in + beats * beat * rate <= in_length:
            return beats, usable
        beats -= BEAT_STEP
    return None


def _track_region(path, start, seconds, lead=None):
    audio = _region(path, start, seconds)
    tempo, beats = _beats(audio if lead is None else audio[int(round(lead * RATE)):])
    section = _phrase_entry(audio, lead) if lead is not None else None
    return audio, tempo, beats, section


def plan_transition(path_out, out_seconds, path_in, in_seconds):
    a_start = max(0.0, out_seconds - LOOK_SECONDS)
    b_start = max(0.0, in_seconds - PHRASE_LEAD_SECONDS)
    lead = in_seconds - b_start
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='smart-fade-beats') as pool:
        ending = pool.submit(_track_region, path_out, a_start, out_seconds - a_start + 2.0)
        starting = pool.submit(_track_region, path_in, b_start, lead + LOOK_SECONDS + 2.0, lead)
        song_out, tempo_out, beats_out, _ = ending.result()
        song_in, tempo_in, beats_after, section = starting.result()
    rate = _stretch_rate(tempo_out, tempo_in)
    locked = bool(tempos_match(tempo_out, tempo_in))
    beat = 60.0 / tempo_out
    before = np.arange(beats_after[0] - 60.0 / tempo_in, -lead, -60.0 / tempo_in)[::-1]
    grid = np.concatenate([before, beats_after]) + lead
    entries = [float(beats_after[0] + lead)]
    if section is not None:
        entries.insert(0, float(grid[np.argmin(np.abs(grid - section))]))
    fits = [(entry, _mix_beats(beat, beats_out, song_out.size / RATE, entry, song_in.size / RATE, rate)) for entry in entries]
    fits = [(entry, fit) for entry, fit in fits if fit is not None]
    if not fits:
        raise ValueError('No room for the mix around the fade points.')
    first_in, (beats, usable) = max(fits, key=lambda item: item[1][0])
    overlap = beats * beat
    mix_at = float(usable[np.argmin(np.abs(usable - (out_seconds - a_start - overlap)))])
    locked = (locked and _in_step(song_out, mix_at, song_in, first_in, overlap, rate, beat)
              and _agreement(song_out) >= MIN_AGREEMENT_BITS and _agreement(song_in[int(round(lead * RATE)):]) >= MIN_AGREEMENT_BITS)
    if not locked:
        rate = 1.0
        beats = max(1, min(beats, math.ceil(MIN_MIX_SECONDS / beat), int((song_in.size / RATE - first_in) / beat)))
        overlap = beats * beat
        usable = beats_out[beats_out + overlap <= song_out.size / RATE]
        mix_at = float(usable[np.argmin(np.abs(usable - (out_seconds - a_start - overlap)))])
        ending, starting = _levels_pair(path_out, path_in)
        mix_at = _fuller_start(ending, a_start, usable, mix_at, starting, b_start + first_in, overlap)
    resume_in = first_in + overlap * rate
    if not locked:
        style = 'fade'
    elif _level(song_in, first_in, overlap * rate) - _level(song_out, mix_at, overlap) >= RISE_DB:
        style = 'rise'
    else:
        style = 'blend'
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
    }
    plan['confidence'] = 'high' if locked else _confidence(ending, a_start + mix_at, starting, b_start + first_in, overlap, overlap * rate, locked)
    return plan
