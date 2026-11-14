# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Beat-matched smart fade transition planned from two real audio files.

Main Features:
* Two click tracks at 120 and 125 BPM give a plan only (no audio): the measured
  tempos, a stretch of 120/125, a mix of MIX_BEATS beats of the first song
  starting on its beat grid before its fade point, the second song entering
  on one of its own beats and resuming exactly where the stretched part ends
* Tempos too far apart (120 and 150 BPM, no half or double time near) are not
  stretched; tempos_match is the same rule on numbers and arrays
* A tempo whose beat gap falls between the tracker's frames is measured to a
  fraction of a BPM, not to the nearest frame
* Songs without a beat raise ValueError
"""

import numpy as np
import pytest
import soundfile

from tasks import smart_fade_mix as mix


def _clicks(path, bpm, seconds, first=0.0):
    rate = mix.RATE
    audio = np.zeros(int(seconds * rate), dtype=np.float32)
    burst = (0.8 * np.sin(2 * np.pi * 1000 * np.arange(int(0.03 * rate)) / rate)).astype(np.float32)
    beat = first
    while beat + 0.03 < seconds:
        start = int(beat * rate)
        audio[start:start + burst.size] += burst
        beat += 60.0 / bpm
    soundfile.write(path, np.stack([audio, audio], axis=1), rate)
    return str(path)


def test_two_click_tracks_mix_on_the_beat_at_the_first_tempo(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 125, 40.0, first=0.2)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert sorted(plan) == ['in_at', 'mix_at', 'mix_seconds', 'resume_at', 'stretch', 'tempo_from', 'tempo_to']
    assert plan['tempo_from'] == pytest.approx(120, abs=2)
    assert plan['tempo_to'] == pytest.approx(125, abs=2)
    assert plan['stretch'] == pytest.approx(120 / 125, abs=0.01)
    assert plan['mix_seconds'] == pytest.approx(mix.MIX_BEATS * 0.5, abs=0.05)
    assert abs(plan['mix_at'] * 2 - round(plan['mix_at'] * 2)) <= 0.08
    assert 30.0 - 6.0 <= plan['mix_at'] <= 30.0
    beats_in = (plan['in_at'] - 0.2) * 125 / 60
    assert plan['in_at'] >= 5.0 and abs(beats_in - round(beats_in)) <= 0.08
    assert plan['resume_at'] - plan['in_at'] == pytest.approx(plan['mix_seconds'] * plan['stretch'], abs=0.002)


def test_tempos_too_far_apart_are_not_stretched(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 150, 40.0)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['stretch'] == 1.0
    assert plan['resume_at'] - plan['in_at'] == pytest.approx(plan['mix_seconds'], abs=0.002)


def test_tempos_match_allows_the_stretch_limit_and_half_or_double_time():
    assert mix.tempos_match(120.0, 125.0)
    assert not mix.tempos_match(120.0, 150.0)
    assert mix.tempos_match(120.0, 61.0) and mix.tempos_match(120.0, 246.0)
    assert mix.tempos_match(np.array([[120.0], [150.0]]), np.array([[125.0, 150.0]])).tolist() == [[True, False], [False, True]]


def test_a_tempo_between_the_tracker_frames_is_measured_finely(tmp_path):
    bpm = 60.0 / (42.33 * 512 / mix.RATE)
    song_out = _clicks(tmp_path / 'a.wav', bpm, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', bpm, 40.0, first=0.2)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['tempo_from'] == pytest.approx(bpm, abs=0.3)
    assert plan['tempo_to'] == pytest.approx(bpm, abs=0.3)


def test_songs_without_a_beat_raise_value_error(tmp_path):
    silent = tmp_path / 'silent.wav'
    soundfile.write(silent, np.zeros((mix.RATE * 40, 2), dtype=np.float32), mix.RATE)
    with pytest.raises(ValueError):
        mix.plan_transition(str(silent), 30.0, str(silent), 5.0)
