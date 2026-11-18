# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Beat-matched transition plan between two songs at the smart fade points.

The smart fade (tasks.recording_search_manager) picks from the neural
fingerprint where one song should end and the next start; this module turns
that pair of points into the exact seconds where the page cuts and mixes the
two songs so the hand-over keeps the beat. It returns numbers only: the page
fetches the songs and does the whole fade itself.

Main Features:
* A few seconds around each point are decoded (mono, one rate); a file
  libsndfile cannot open (AAC .m4a) goes through the analysis PyAV loader.
* Beats come from librosa's tracker, tempo and grid from a line fitted through
  the beat times. The next song takes the first song's tempo in the mix
  (stretch, pitch kept by the page, half or double time too, none past
  MAX_STRETCH); tempos_match is that rule, shared with the smart fade order.
* The mix lasts at least MIX_SECONDS in whole BEAT_STEP steps (real DJ
  transitions peak at 32 and 64 beats) and ends on the beat nearest the fade
  point; without room it loses steps down to MIN_MIX_SECONDS. No steady beat
  or no room raises ValueError.
* The next song enters on a section start: the beat nearest the highest peak,
  within PHRASE_REACH_BEATS, of a Foote novelty curve (PHRASE_KERNEL_BEATS
  checkerboard on beat-synchronous chroma, MFCC and loudness) read from
  PHRASE_LEAD_SECONDS before the point; on Harmonix Set annotations it hit a
  real section start about three times as often.
* style: 'blend' (tempos match, the bass crosses at swap_at), 'rise' (also
  RISE_DB louder, in from the treble), 'fade' (tempos or onsets do not line up
  within MATCH_MS at MATCH_SHARE: unstretched, MIN_MIX_SECONDS, first song
  muffled).
* confidence: 'high' beat-locked, 'low' a fade where the first song is at
  least FULL_LU over its own loudness and the next comes in DROP_LU lower
  (relative to its own), else 'medium'; loudness is BS.1770 of the whole song
  at LEVEL_RATE, decoded only for a fade. The page plays a 'low' plan as its
  plain crossfade.
"""

import math
import os

import numpy as np

RATE = 44100
MIX_SECONDS = 15.0
MIN_MIX_SECONDS = 10.0
BEAT_STEP = 8
LOOK_SECONDS = 26.0
MAX_STRETCH = 0.08
OCTAVES = (1.0, 0.5, 2.0)
RISE_DB = 6.0
PHRASE_LEAD_SECONDS = 8.0
PHRASE_KERNEL_BEATS = 8
PHRASE_REACH_BEATS = 8
MATCH_MS = 30.0
MATCH_SHARE = 0.8
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


def tempos_match(tempo_out, tempo_in):
    ratio = np.asarray(tempo_out, dtype=np.float64) / np.asarray(tempo_in, dtype=np.float64)
    return np.any([np.abs(ratio * octave - 1.0) <= MAX_STRETCH for octave in OCTAVES], axis=0)


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


def _k_weighted(audio, rate):
    from scipy.signal import lfilter

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
    shelved = lfilter(np.float32(shelf_b), np.float32(shelf_a), audio.astype(np.float32))
    return lfilter(np.float32(pass_b), np.float32(pass_a), shelved)


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
    audio = _load(path, LEVEL_RATE)
    weighted = _k_weighted(audio, LEVEL_RATE)
    power = np.concatenate([[0.0], np.cumsum(np.square(weighted, dtype=np.float64))])
    size = int(round(BLOCK_SECONDS * LEVEL_RATE))
    count = max(1, int((audio.size - size) / (STEP_SECONDS * LEVEL_RATE)) + 1)
    starts = np.minimum(np.round(np.arange(count) * STEP_SECONDS * LEVEL_RATE).astype(np.int64), audio.size)
    ends = np.minimum(starts + size, audio.size)
    energy = (power[ends] - power[starts]) / np.maximum(ends - starts, 1)
    return {'energy': energy, 'loudness': _integrated(energy)}


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


def _stretch_rate(tempo_out, tempo_in):
    for octave in OCTAVES:
        rate = tempo_out * octave / tempo_in
        if abs(rate - 1.0) <= MAX_STRETCH:
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


def plan_transition(path_out, out_seconds, path_in, in_seconds):
    a_start = max(0.0, out_seconds - LOOK_SECONDS)
    b_start = max(0.0, in_seconds - PHRASE_LEAD_SECONDS)
    lead = in_seconds - b_start
    song_out = _region(path_out, a_start, out_seconds - a_start + 2.0)
    song_in = _region(path_in, b_start, lead + LOOK_SECONDS + 2.0)
    tempo_out, beats_out = _beats(song_out)
    tempo_in, beats_after = _beats(song_in[int(round(lead * RATE)):])
    locked = bool(tempos_match(tempo_out, tempo_in))
    rate = _stretch_rate(tempo_out, tempo_in)
    beat = 60.0 / tempo_out
    before = np.arange(beats_after[0] - 60.0 / tempo_in, -lead, -60.0 / tempo_in)[::-1]
    grid = np.concatenate([before, beats_after]) + lead
    entries = [float(beats_after[0] + lead)]
    section = _phrase_entry(song_in, lead)
    if section is not None:
        entries.insert(0, float(grid[np.argmin(np.abs(grid - section))]))
    fits = [(entry, _mix_beats(beat, beats_out, song_out.size / RATE, entry, song_in.size / RATE, rate)) for entry in entries]
    fits = [(entry, fit) for entry, fit in fits if fit is not None]
    if not fits:
        raise ValueError('No room for the mix around the fade points.')
    first_in, (beats, usable) = max(fits, key=lambda item: item[1][0])
    overlap = beats * beat
    mix_at = float(usable[np.argmin(np.abs(usable - (out_seconds - a_start - overlap)))])
    locked = locked and _in_step(song_out, mix_at, song_in, first_in, overlap, rate, beat)
    if not locked:
        rate = 1.0
        beats = max(1, min(beats, math.ceil(MIN_MIX_SECONDS / beat), int((song_in.size / RATE - first_in) / beat)))
        overlap = beats * beat
        usable = beats_out[beats_out + overlap <= song_out.size / RATE]
        mix_at = float(usable[np.argmin(np.abs(usable - (out_seconds - a_start - overlap)))])
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
    plan['confidence'] = 'high' if locked else _confidence(_levels(path_out), a_start + mix_at, _levels(path_in), b_start + first_in, overlap, overlap * rate, locked)
    return plan
