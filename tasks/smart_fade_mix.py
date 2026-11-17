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
* Only a few seconds around each point are decoded (mono, one common rate).
* The beats come from librosa's beat tracker, the one the analysis uses; the
  tempo and the grid come from a line fitted through the beat times, finer
  than the tracker's frames.
* The next song matches the first song's tempo during the mix (stretch, pitch
  kept by the page), half or double time included; past MAX_STRETCH it is not
  stretched. tempos_match is that rule, shared with the smart fade order.
* The mix lasts at least MIX_SECONDS, in whole steps of BEAT_STEP beats of
  the first song (real DJ transitions peak at 32 and 64 beats), and ends on
  its beat nearest to its fade point; the next song plays on alone from
  resume_at. Without room (a late first beat) the mix loses whole steps down
  to MIN_MIX_SECONDS, the page's plain crossfade. No steady beat or no room
  raises ValueError.
* The next song enters on a section start near its fade point: the beat
  nearest the highest peak, within PHRASE_REACH_BEATS beats, of a Foote
  novelty curve (PHRASE_KERNEL_BEATS checkerboard on beat-synchronous chroma,
  MFCC and loudness) read from PHRASE_LEAD_SECONDS before the point, unless
  that leaves room for a shorter mix. On Harmonix Set section annotations of
  library songs this hit a real section start about three times as often;
  at the first song's fade point (deep in its outro) it did not help.
* style says how the page mixes: 'blend' when the tempos match (the bass of
  the two songs crosses around swap_at, the middle beat of the mix), 'rise'
  when the next song is also RISE_DB louder (it comes in from the treble up),
  'fade' when they do not match: the same mix unstretched, the first song
  muffled as it goes.
"""

import math

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


def tempos_match(tempo_out, tempo_in):
    ratio = np.asarray(tempo_out, dtype=np.float64) / np.asarray(tempo_in, dtype=np.float64)
    return np.any([np.abs(ratio * octave - 1.0) <= MAX_STRETCH for octave in OCTAVES], axis=0)


def _region(path, start, seconds):
    import librosa

    audio, _sr = librosa.load(path, sr=RATE, mono=True, offset=max(0.0, start), duration=seconds)
    return np.asarray(audio, dtype=np.float32)


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
    resume_in = first_in + overlap * rate
    if not locked:
        style = 'fade'
    elif _level(song_in, first_in, overlap * rate) - _level(song_out, mix_at, overlap) >= RISE_DB:
        style = 'rise'
    else:
        style = 'blend'
    return {
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
