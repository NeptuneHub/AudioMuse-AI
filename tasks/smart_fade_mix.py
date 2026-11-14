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
  tempo and the beat grid come from a line fitted through the beat times,
  finer than the tracker's frames (a median gap is off by up to 2%, about
  80 ms of drift over the mix).
* The next song plays at the first song's tempo during the mix (stretch, pitch
  kept by the page), half or double time included; past MAX_STRETCH it is not
  stretched. tempos_match is that rule, shared with the smart fade order.
* The mix starts on the first song's beat nearest to MIX_BEATS beats before
  its fade point and lasts MIX_BEATS of its beats; the next song's first beat
  lands on it and it plays on alone from resume_at. No steady beat raises
  ValueError.
"""

import numpy as np

RATE = 44100
MIX_BEATS = 8
LOOK_SECONDS = 12.0
MAX_STRETCH = 0.08
OCTAVES = (1.0, 0.5, 2.0)


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


def plan_transition(path_out, out_seconds, path_in, in_seconds):
    a_start = max(0.0, out_seconds - LOOK_SECONDS)
    song_out = _region(path_out, a_start, out_seconds - a_start + 2.0)
    song_in = _region(path_in, in_seconds, LOOK_SECONDS + 2.0)
    tempo_out, beats_out = _beats(song_out)
    tempo_in, beats_in = _beats(song_in)
    rate = _stretch_rate(tempo_out, tempo_in)
    overlap = MIX_BEATS * 60.0 / tempo_out
    usable = beats_out[beats_out + overlap <= song_out.size / RATE]
    if not usable.size:
        raise ValueError('No beat leaves room for the mix.')
    mix_at = float(usable[np.argmin(np.abs(usable - (out_seconds - a_start - overlap)))])
    first_in = float(beats_in[0])
    resume_in = first_in + overlap * rate
    if resume_in > song_in.size / RATE:
        raise ValueError('The next song is too short after its fade point.')
    return {
        'mix_at': round(a_start + mix_at, 4),
        'mix_seconds': round(overlap, 4),
        'in_at': round(in_seconds + first_in, 4),
        'resume_at': round(in_seconds + resume_in, 4),
        'stretch': round(rate, 4),
        'tempo_from': round(tempo_out, 1),
        'tempo_to': round(tempo_in, 1),
    }
