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
  stretched and fade over only MIN_MIX_SECONDS of whole beats ending at the
  fade point, swap_at in its middle; tempos_match is the same rule on numbers
  and arrays
* A beat grid 150 ms off the next song's clicks fades the same way although
  the tempos match; the onset check takes clicks on the beat (stretched too)
  and refuses them 100 ms off
* A tempo whose beat gap falls between the tracker's frames is measured to a
  fraction of a BPM, not to the nearest frame
* Songs without a beat raise ValueError
* Beat agreement (information gain) is full for the same beats, high for half
  of them, low for random ones; a click track's committee agrees, noise's
  does not. Only a confirmed lock mixes: a committee under
  MIN_AGREEMENT_BITS gets the fade plan although tempos and onsets match
* A default crossfade over a quiet ending into a quiet intro starts on an
  earlier beat whose equal-power mix dips FLOOR_GAIN_DB less, never past the
  last EDGE_SECONDS, and steady songs keep their start
* A full-scale 997 Hz tone measures -3.01 LUFS
* An unstretched fade from a song at full level into a next song that starts
  quieter than its own level is 'low' (a steady next song 'medium', a beat
  lock 'high'), on curves and on real files
* A real AAC .m4a, which libsndfile cannot open, plans like a WAV
"""

import numpy as np
import pytest
import soundfile

from tasks import smart_fade_mix as mix


def _click_audio(bpm, seconds, first=0.0, level=0.8):
    rate = mix.RATE
    audio = np.zeros(int(seconds * rate), dtype=np.float32)
    burst = (level * np.sin(2 * np.pi * 1000 * np.arange(int(0.03 * rate)) / rate)).astype(np.float32)
    beat = first
    while beat + 0.03 < seconds:
        start = int(beat * rate)
        audio[start:start + burst.size] += burst
        beat += 60.0 / bpm
    return audio


def _clicks(path, bpm, seconds, first=0.0, level=0.8):
    audio = _click_audio(bpm, seconds, first, level)
    soundfile.write(path, np.stack([audio, audio], axis=1), mix.RATE)
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
    assert sorted(plan) == [
        'beat_seconds', 'confidence', 'in_at', 'mix_at', 'mix_seconds', 'resume_at', 'stretch', 'style', 'swap_at', 'tempo_from', 'tempo_to',
    ]
    assert plan['confidence'] == 'high'
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


def _assert_short_fade(plan):
    beats = round(plan['mix_seconds'] / plan['beat_seconds'])
    assert plan['style'] == 'fade' and plan['stretch'] == 1.0
    assert beats == np.ceil(mix.MIN_MIX_SECONDS / plan['beat_seconds'] - 1e-6)
    assert plan['swap_at'] == pytest.approx(plan['mix_at'] + beats // 2 * plan['beat_seconds'], abs=0.002)
    assert plan['resume_at'] - plan['in_at'] == pytest.approx(plan['mix_seconds'], abs=0.002)
    assert plan['mix_at'] + plan['mix_seconds'] == pytest.approx(30.0, abs=0.3)


def test_tempos_too_far_apart_fade_unstretched_over_the_short_mix(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0, level=0.1)
    song_in = _clicks(tmp_path / 'b.wav', 150, 40.0)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['beat_seconds'] == pytest.approx(0.5, abs=0.01)
    _assert_short_fade(plan)
    beats_in = plan['in_at'] * 150 / 60
    assert plan['in_at'] >= 5.0 and abs(beats_in - round(beats_in)) <= 0.08


def test_a_beat_grid_that_misses_the_beats_fades_instead_of_matching(tmp_path, monkeypatch):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 120, 40.0, first=0.1)
    tracked = mix._beats
    calls = []

    def off_grid(audio):
        tempo, grid = tracked(audio)
        calls.append(tempo)
        return tempo, grid + (0.15 if len(calls) > 1 else 0.0)

    monkeypatch.setattr(mix, '_phrase_entry', lambda audio, target: None)
    assert mix.plan_transition(song_out, 30.0, song_in, 5.0)['style'] == 'blend'
    monkeypatch.setattr(mix, '_beats', off_grid)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['tempo_from'] == pytest.approx(plan['tempo_to'], abs=1.0)
    _assert_short_fade(plan)


def test_the_onset_check_accepts_beats_on_the_beat_and_refuses_them_100_ms_off():
    song_out = _click_audio(120, 30.0)
    song_in = _click_audio(120, 30.0, first=0.25)
    assert mix._in_step(song_out, 10.0, song_in, 10.25, 8.0, 1.0, 0.5)
    assert not mix._in_step(song_out, 10.0, song_in, 10.15, 8.0, 1.0, 0.5)
    stretched = _click_audio(125, 30.0)
    assert mix._in_step(song_out, 10.0, stretched, 9.6, 8.0, 0.96, 0.5)


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


def test_beat_agreement_is_full_for_the_same_beats_and_low_for_random_ones():
    steady = np.arange(0.0, 20.0, 0.5)
    assert mix._error_bits(steady, steady) == pytest.approx(np.log2(mix.AGREEMENT_BINS))
    assert mix._error_bits(steady, steady[::2]) > 4.0
    scattered = np.sort(np.random.default_rng(3).uniform(0.0, 20.0, steady.size))
    assert min(mix._error_bits(steady, scattered), mix._error_bits(scattered, steady)) < mix.MIN_AGREEMENT_BITS
    assert mix._agreement(_click_audio(120, 28.0)) >= mix.MIN_AGREEMENT_BITS
    noise = np.random.default_rng(4).standard_normal(int(28.0 * mix.RATE)).astype(np.float32) * 0.1
    assert mix._agreement(noise) < mix.MIN_AGREEMENT_BITS


def test_only_a_confirmed_beat_lock_mixes_everything_else_is_the_crossfade_plan(tmp_path, monkeypatch):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    near = _clicks(tmp_path / 'c.wav', 125, 40.0, first=0.2)
    assert mix.plan_transition(song_out, 30.0, near, 5.0)['style'] == 'blend'
    monkeypatch.setattr(mix, '_agreement', lambda audio: mix.MIN_AGREEMENT_BITS - 0.1)
    plan = mix.plan_transition(song_out, 30.0, near, 5.0)
    _assert_short_fade(plan)
    assert plan['confidence'] != 'high'


def test_a_full_scale_997_hz_tone_measures_minus_3_lufs(tmp_path):
    rate = 48000
    tone = np.sin(2 * np.pi * 997 * np.arange(20 * rate) / rate)
    assert float(mix._lufs(np.mean(mix._k_weighted(tone, rate)[rate:] ** 2))) == pytest.approx(-3.01, abs=0.05)
    path = tmp_path / 'tone.wav'
    soundfile.write(path, (0.5 * tone).astype(np.float32), rate, subtype='FLOAT')
    levels = mix._levels(str(path))
    assert levels['loudness'] == pytest.approx(-3.01 - 6.02, abs=0.3)


SILENT = -120.0


def _curve(*pieces):
    return np.concatenate([np.linspace(a, b, int(round(seconds / mix.STEP_SECONDS)), endpoint=False) for seconds, a, b in pieces])


def _db_levels(curve):
    energy = 10.0 ** ((curve + 0.691) / 10.0)
    return {'energy': energy, 'loudness': mix._integrated(energy), 'seconds': curve.size * mix.STEP_SECONDS + mix.BLOCK_SECONDS - mix.STEP_SECONDS}


def test_a_default_crossfade_starts_where_the_mix_dips_least_within_the_last_30_seconds():
    ending = _db_levels(_curve((180.0, -14.0, -14.0), (12.0, -14.0, -40.0), (8.0, -40.0, -40.0)))
    quiet_intro = _db_levels(_curve((8.0, -32.0, -32.0), (100.0, -14.0, -14.0)))
    beats = np.arange(165.0, 190.0, 0.5)
    chosen = mix._fuller_start(ending, 0.0, beats, 185.0, quiet_intro, 0.0, 10.0)
    assert chosen < 185.0 and chosen + 10.0 >= ending['seconds'] - mix.EDGE_SECONDS
    gain = mix._mix_floor(ending, chosen, quiet_intro, 0.0, 10.0) - mix._mix_floor(ending, 185.0, quiet_intro, 0.0, 10.0)
    assert gain >= mix.FLOOR_GAIN_DB
    early = mix._fuller_start(ending, 0.0, np.arange(100.0, 190.0, 0.5), 185.0, quiet_intro, 0.0, 10.0)
    assert early + 10.0 >= ending['seconds'] - mix.EDGE_SECONDS
    steady = _db_levels(_curve((200.0, -14.0, -14.0)))
    assert mix._fuller_start(steady, 0.0, beats, 185.0, steady, 0.0, 10.0) == 185.0


def test_an_unlocked_fade_from_full_level_into_a_quieter_start_has_low_confidence():
    ending = _db_levels(_curve((150.0, -16.0, -16.0), (50.0, -12.0, -12.0), (6.0, -12.0, -60.0), (2.0, SILENT, SILENT)))
    quiet = _db_levels(_curve((1.0, SILENT, SILENT), (30.0, -20.0, -20.0), (120.0, -12.0, -12.0)))
    strong = _db_levels(_curve((40.0, -10.0, -10.0), (110.0, -14.0, -14.0)))
    assert mix._confidence(ending, 180.0, quiet, 10.0, 10.0, 10.0, False) == 'low'
    assert mix._confidence(ending, 180.0, strong, 10.0, 10.0, 10.0, False) == 'medium'
    assert mix._confidence(ending, 100.0, quiet, 10.0, 10.0, 10.0, False) == 'medium'
    assert mix._confidence(ending, 180.0, quiet, 10.0, 10.0, 10.0, True) == 'high'


def _bedded(bpm, seconds, quiet_seconds, quiet_db):
    rate = mix.RATE
    audio = _click_audio(bpm, seconds) + 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * rate)) / rate).astype(np.float32)
    audio[:int(quiet_seconds * rate)] *= 10.0 ** (quiet_db / 20.0)
    return audio


def _write(path, audio):
    soundfile.write(path, np.stack([audio, audio], axis=1), mix.RATE)
    return str(path)


def test_a_planned_fade_into_a_quieter_next_song_has_low_confidence(tmp_path):
    song_out = _write(tmp_path / 'a.wav', _bedded(120, 40.0, 20.0, -3.0))
    quiet = _write(tmp_path / 'b.wav', _bedded(150, 40.0, 20.0, -4.0))
    steady = _write(tmp_path / 'c.wav', _bedded(150, 40.0, 0.0, 0.0))
    plan = mix.plan_transition(song_out, 30.0, quiet, 5.0)
    assert plan['style'] == 'fade' and plan['confidence'] == 'low'
    assert mix.plan_transition(song_out, 30.0, steady, 5.0)['confidence'] == 'medium'


def _m4a(path, audio):
    av = pytest.importorskip('av')
    with av.open(str(path), 'w', format='mp4') as out:
        stream = out.add_stream('aac', rate=mix.RATE)
        stream.layout = 'mono'
        for start in range(0, audio.size, 1024):
            frame = av.AudioFrame.from_ndarray(audio[None, start:start + 1024].astype(np.float32), format='fltp', layout='mono')
            frame.sample_rate = mix.RATE
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return str(path)


def test_an_aac_m4a_song_that_libsndfile_cannot_open_plans_through_the_analysis_loader(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _m4a(tmp_path / 'b.m4a', _click_audio(120, 40.0, first=0.2))
    with pytest.raises(soundfile.SoundFileRuntimeError):
        soundfile.info(song_in)
    assert mix._region(song_in, 5.0, 10.0).size == pytest.approx(10.0 * mix.RATE, abs=2)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['style'] == 'blend' and plan['confidence'] == 'high'
    assert plan['tempo_to'] == pytest.approx(120, abs=2)
