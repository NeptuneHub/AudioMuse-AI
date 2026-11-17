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
  tempos, a stretch of 120/125, a mix of the fewest BEAT_STEP beats of the
  first song that last MIX_SECONDS, on its beat grid and ending at its fade
  point, the second song entering on one of its own beats and resuming
  exactly where the stretched part ends; style 'blend', swap_at on the middle
  beat of the mix; a late first beat in the next song shortens the mix by
  whole steps, never under MIN_MIX_SECONDS (else ValueError)
* The next song enters on a section start a few beats after its fade point
  (a chord coming in over the clicks), and never more than PHRASE_REACH_BEATS
  beats away from it
* A next song RISE_DB louder at the same tempos rises in instead
* Tempos too far apart (120 and 150 BPM, no half or double time near) are not
  stretched and fade: the same mix ending at the fade point, swap_at in its
  middle; tempos_match is the same rule on numbers and arrays
* A tempo whose beat gap falls between the tracker's frames is measured to a
  fraction of a BPM, not to the nearest frame
* Songs without a beat raise ValueError
"""

import numpy as np
import pytest
import soundfile

from tasks import smart_fade_mix as mix


def _clicks(path, bpm, seconds, first=0.0, level=0.8):
    rate = mix.RATE
    audio = np.zeros(int(seconds * rate), dtype=np.float32)
    burst = (level * np.sin(2 * np.pi * 1000 * np.arange(int(0.03 * rate)) / rate)).astype(np.float32)
    beat = first
    while beat + 0.03 < seconds:
        start = int(beat * rate)
        audio[start:start + burst.size] += burst
        beat += 60.0 / bpm
    soundfile.write(path, np.stack([audio, audio], axis=1), rate)
    return str(path)


def _assert_mix_length(plan):
    beats = round(plan['mix_seconds'] / plan['beat_seconds'])
    assert beats % mix.BEAT_STEP == 0
    assert plan['mix_seconds'] >= mix.MIX_SECONDS - 0.01
    assert (beats - mix.BEAT_STEP) * plan['beat_seconds'] < mix.MIX_SECONDS
    assert plan['swap_at'] == pytest.approx(plan['mix_at'] + beats // 2 * plan['beat_seconds'], abs=0.002)


def test_two_click_tracks_mix_on_the_beat_at_the_first_tempo(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 125, 40.0, first=0.2)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert sorted(plan) == ['beat_seconds', 'in_at', 'mix_at', 'mix_seconds', 'resume_at', 'stretch', 'style', 'swap_at', 'tempo_from', 'tempo_to']
    assert plan['style'] == 'blend'
    assert plan['beat_seconds'] == pytest.approx(0.5, abs=0.01)
    _assert_mix_length(plan)
    assert plan['tempo_from'] == pytest.approx(120, abs=2)
    assert plan['tempo_to'] == pytest.approx(125, abs=2)
    assert plan['stretch'] == pytest.approx(120 / 125, abs=0.01)
    assert abs(plan['mix_at'] * 2 - round(plan['mix_at'] * 2)) <= 0.08
    assert plan['mix_at'] + plan['mix_seconds'] == pytest.approx(30.0, abs=0.3)
    beats_in = (plan['in_at'] - 0.2) * 125 / 60
    assert plan['in_at'] >= 5.0 and abs(beats_in - round(beats_in)) <= 0.08
    assert plan['resume_at'] - plan['in_at'] == pytest.approx(plan['mix_seconds'] * plan['stretch'], abs=0.002)


def test_the_mix_is_the_fewest_beat_steps_that_reach_the_mix_length(tmp_path, monkeypatch):
    monkeypatch.setattr(mix, 'MIX_SECONDS', 9.0)
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 120, 40.0, first=0.1)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['mix_seconds'] == pytest.approx(24 * plan['beat_seconds'], abs=0.002)
    _assert_mix_length(plan)
    assert plan['mix_at'] + plan['mix_seconds'] == pytest.approx(30.0, abs=0.2)


def test_a_late_first_beat_shortens_the_mix_by_whole_steps_down_to_the_minimum(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 120, 45.0, first=14.0)
    plan = mix.plan_transition(song_out, 30.0, song_in, 0.0)
    beats = round(plan['mix_seconds'] / plan['beat_seconds'])
    assert plan['in_at'] == pytest.approx(14.0, abs=0.05)
    assert beats % mix.BEAT_STEP == 0 and mix.MIN_MIX_SECONDS <= plan['mix_seconds'] < mix.MIX_SECONDS
    assert plan['resume_at'] <= mix.LOOK_SECONDS + 2.0
    late = _clicks(tmp_path / 'c.wav', 120, 45.0, first=19.0)
    with pytest.raises(ValueError):
        mix.plan_transition(song_out, 30.0, late, 0.0)


def test_the_next_song_enters_on_a_nearby_section_start(tmp_path):
    rate = mix.RATE
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    clicks = soundfile.read(_clicks(tmp_path / 'b.wav', 120, 45.0))[0][:, 0]
    t = np.arange(clicks.size) / rate
    chord = sum(0.12 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63, 440.0))
    audio = (clicks + np.where(t >= 7.5, chord, 0.0)).astype(np.float32)
    soundfile.write(tmp_path / 'c.wav', np.stack([audio, audio], axis=1), rate)
    plan = mix.plan_transition(song_out, 30.0, str(tmp_path / 'c.wav'), 5.0)
    assert plan['in_at'] == pytest.approx(7.5, abs=0.5 + 0.01)
    plain = mix.plan_transition(song_out, 30.0, _clicks(tmp_path / 'd.wav', 120, 45.0), 5.0)
    assert abs(plain['in_at'] - 5.0) <= mix.PHRASE_REACH_BEATS * 0.5 + 0.01
    _assert_mix_length(plan)


def test_a_clearly_louder_next_song_rises_in(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0, level=0.1)
    song_in = _clicks(tmp_path / 'b.wav', 125, 40.0, first=0.2)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['style'] == 'rise'
    assert plan['stretch'] == pytest.approx(120 / 125, abs=0.01)
    _assert_mix_length(plan)


def test_tempos_too_far_apart_fade_unstretched_with_the_bass_swap(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0, level=0.1)
    song_in = _clicks(tmp_path / 'b.wav', 150, 40.0)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['style'] == 'fade'
    assert plan['stretch'] == 1.0
    assert plan['beat_seconds'] == pytest.approx(0.5, abs=0.01)
    _assert_mix_length(plan)
    assert plan['mix_at'] + plan['mix_seconds'] == pytest.approx(30.0, abs=0.3)
    beats_in = plan['in_at'] * 150 / 60
    assert plan['in_at'] >= 5.0 and abs(beats_in - round(beats_in)) <= 0.08
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
