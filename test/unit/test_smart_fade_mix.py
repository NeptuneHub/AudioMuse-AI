# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Beat-matched smart fade transition planned from two real audio files.

Main Features:
* Click tracks at 120 and 122 BPM plan a stretch of 120/122, a lock checked
  over the fewest BEAT_STEP beats reaching LOCK_SECONDS, a mix of its last
  whole bars nearest the pair's mix length ending at the fade point, the
  next song entering on its beat and resuming where the stretch ends; style
  'blend', swap_at mid-mix; a late first beat shortens the checked span by
  whole steps, never under MIN_BEAT_MIX_SECONDS (else NoRoom)
* Lengths count felt beats (MIX_BEATS, FADE_BEATS), fewer for a dense pair;
  FAST_ONSETS_PER_BEAT doubles a half-speed tempo; a tone over the clicks
  crossfades shorter than clicks alone
* The next song enters on a section start a few beats after its fade point,
  never more than PHRASE_REACH_BEATS away; RISE_DB louder, it rises in
* Tempos too far apart (120 and 150 BPM) or a beat grid 150 ms off fade
  unstretched over the whole beats reaching fade_seconds, swap_at in the middle;
  tempos_match (60 cents, half or double time) is one rule on numbers and arrays; the onset check takes
  clicks on the beat (stretched too) and refuses them 100 ms off
* A tempo between the tracker's frames is measured to a fraction of a BPM;
  songs without a beat raise ValueError
* Tempos over MAX_LOCK_CENTS apart (120 and 125 BPM, 70 cents) crossfade even
  when their beats lock; a double-time song within it (244 BPM) still mixes
* Beat agreement (information gain) is full for the same beats, high for half
  of them, low for random ones; a click track's committee agrees, noise's
  does not; under MIN_AGREEMENT_BITS the plan is the fade
* A default crossfade over a quiet ending into a quiet intro starts on an
  earlier beat dipping FLOOR_GAIN_DB less, within the last EDGE_SECONDS
* A full-scale 997 Hz tone measures -3.01 LUFS; a fade from full level into
  a quieter next song is 'low' ('medium' steady, 'high' locked)
* A real AAC .m4a, which libsndfile cannot open, plans like a WAV
* Without a full lock, drum-led parts whose beats line up carry one
  'percussion_mix' alternative of up to PERCUSSION_SECONDS; a tone is not
  drum-led and gets none, clicks over a sub-bass are (the bass is left out); a locked pair of drum-led parts is a
  'percussion_mix' too (the other tests treat click tracks as full songs)
"""

import numpy as np
import pytest
import soundfile
import threading

from tasks import smart_fade_mix as mix

_REAL_PERCUSSIVE = mix._percussive
_LENGTHS = []


@pytest.fixture(autouse=True)
def _not_drum_led(monkeypatch):
    monkeypatch.setattr(mix, '_percussive', lambda *args, **kwargs: False)


@pytest.fixture(autouse=True)
def _pair_lengths(monkeypatch):
    real = mix.lengths
    _LENGTHS.clear()

    def spy(beat, weight):
        _LENGTHS.append(real(beat, weight))
        return _LENGTHS[-1]

    monkeypatch.setattr(mix, 'lengths', spy)


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


def _assert_mix_length(plan, checked=None):
    beats = round(plan['mix_seconds'] / plan['beat_seconds'])
    target = _LENGTHS[-1][0]
    assert beats % mix.BAR_BEATS == 0
    if checked is not None and checked < target:
        assert plan['mix_seconds'] == pytest.approx(checked, abs=0.002)
    else:
        assert abs(plan['mix_seconds'] - target) <= mix.BAR_BEATS * plan['beat_seconds'] / 2 + 0.01
    assert plan['fade_seconds'] == pytest.approx(_LENGTHS[-1][1], abs=1e-4)
    assert plan['swap_at'] == pytest.approx(plan['mix_at'] + beats // 2 * plan['beat_seconds'], abs=0.002)


def _checked_spans(monkeypatch):
    spans = []
    aligned = mix._aligned_entry

    def spy(song_out, at, song_in, entry, seconds, *args, **kwargs):
        spans.append((at, seconds))
        return aligned(song_out, at, song_in, entry, seconds, *args, **kwargs)

    monkeypatch.setattr(mix, '_aligned_entry', spy)
    return spans


def test_two_click_tracks_mix_on_the_beat_at_the_first_tempo(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 122, 40.0, first=0.2)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert sorted(plan) == [
        'beat_seconds', 'confidence', 'fade_seconds', 'in_at', 'mix_at', 'mix_seconds', 'resume_at', 'stretch', 'style', 'swap_at',
        'tempo_from', 'tempo_to',
    ]
    assert plan['confidence'] == 'high'
    assert plan['style'] == 'blend'
    assert plan['beat_seconds'] == pytest.approx(0.5, abs=0.01)
    _assert_mix_length(plan)
    assert plan['tempo_from'] == pytest.approx(120, abs=2)
    assert plan['tempo_to'] == pytest.approx(122, abs=2)
    assert plan['stretch'] == pytest.approx(120 / 122, abs=0.01)
    assert abs(plan['mix_at'] * 2 - round(plan['mix_at'] * 2)) <= 0.08
    assert plan['mix_at'] + plan['mix_seconds'] == pytest.approx(30.0, abs=0.3)
    beats_in = (plan['in_at'] - 0.2) * 122 / 60
    assert abs(plan['in_at'] - 5.0) <= mix.PHRASE_REACH_BEATS * 60 / 122 + 0.01 and abs(beats_in - round(beats_in)) <= 0.08
    assert plan['resume_at'] - plan['in_at'] == pytest.approx(plan['mix_seconds'] * plan['stretch'], abs=0.002)


def test_both_regions_are_tracked_in_parallel(tmp_path, monkeypatch):
    paths = [_clicks(tmp_path / name, bpm, 40.0) for name, bpm in (('a.wav', 120), ('b.wav', 122))]
    ready = threading.Barrier(2)
    track = mix._track_region

    def together(*args):
        ready.wait(timeout=3)
        return track(*args)

    monkeypatch.setattr(mix, '_track_region', together)
    assert mix.plan_transition(paths[0], 30.0, paths[1], 5.0)['style'] == 'blend'


def test_the_lock_is_checked_over_whole_beat_steps_and_the_mix_plays_its_last_bars(tmp_path, monkeypatch):
    spans = _checked_spans(monkeypatch)
    monkeypatch.setattr(mix, 'lengths', lambda beat, weight: (7.4, mix.MIN_MIX_SECONDS))
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 120, 40.0, first=0.1)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    at, seconds = spans[0]
    assert seconds == pytest.approx(32 * plan['beat_seconds'], abs=0.002) and seconds >= mix.LOCK_SECONDS
    assert plan['mix_seconds'] == pytest.approx(16 * plan['beat_seconds'], abs=0.002)
    assert plan['mix_at'] + plan['mix_seconds'] == pytest.approx(30.0 - mix.LOOK_SECONDS + at + seconds, abs=0.002)
    assert plan['mix_at'] + plan['mix_seconds'] == pytest.approx(30.0, abs=0.2)
    assert plan['resume_at'] - plan['in_at'] == pytest.approx(plan['mix_seconds'] * plan['stretch'], abs=0.002)


def test_a_late_first_beat_shortens_the_checked_span_by_whole_steps_down_to_the_minimum(tmp_path, monkeypatch):
    spans = _checked_spans(monkeypatch)
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 120, 45.0, first=14.0)
    plan = mix.plan_transition(song_out, 30.0, song_in, 0.0)
    checked = round(spans[0][1] / plan['beat_seconds'])
    assert plan['in_at'] == pytest.approx(14.0, abs=0.05)
    assert checked % mix.BEAT_STEP == 0 and mix.MIN_BEAT_MIX_SECONDS <= spans[0][1] < mix.LOCK_SECONDS
    _assert_mix_length(plan, spans[0][1])
    assert plan['resume_at'] <= mix.LOOK_SECONDS + 2.0
    late = _clicks(tmp_path / 'c.wav', 120, 45.0, first=19.0)
    with pytest.raises(mix.NoRoom) as caught:
        mix.plan_transition(song_out, 30.0, late, 0.0)
    assert caught.value.fade_seconds == pytest.approx(_LENGTHS[-1][1], abs=1e-4)


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
    _assert_mix_length(plan)
    plain = mix.plan_transition(song_out, 30.0, _clicks(tmp_path / 'd.wav', 120, 45.0), 5.0)
    assert abs(plain['in_at'] - 5.0) <= mix.PHRASE_REACH_BEATS * 0.5 + 0.01


def test_a_clearly_louder_next_song_rises_in(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0, level=0.1)
    song_in = _clicks(tmp_path / 'b.wav', 122, 40.0, first=0.2)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert plan['style'] == 'rise'
    assert plan['stretch'] == pytest.approx(120 / 122, abs=0.01)
    _assert_mix_length(plan)


def _assert_short_fade(plan):
    beats = round(plan['mix_seconds'] / plan['beat_seconds'])
    assert plan['style'] == 'fade' and plan['stretch'] == 1.0
    assert plan['fade_seconds'] == pytest.approx(_LENGTHS[-1][1], abs=1e-4)
    assert beats == np.ceil(plan['fade_seconds'] / plan['beat_seconds'] - 1e-3)
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
    def off_grid(audio):
        tempo, grid = tracked(audio)
        incoming = np.flatnonzero(np.abs(audio) > 0.01)[0] > 0.05 * mix.RATE
        return tempo, grid + (0.15 if incoming else 0.0)

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


def test_small_consistent_onset_offset_is_repaired_but_drift_is_rejected():
    a = _click_audio(120, 16)
    b = _click_audio(120, 16, first=0.07)
    cue = mix._aligned_entry(a, 2, b, 2, 8, 1, 0.5)
    assert cue == pytest.approx(2.07, abs=0.015)
    drifting = np.r_[b[:6 * mix.RATE], _click_audio(120, 10, first=0.145)]
    assert mix._aligned_entry(a, 2, drifting, 2, 8, 1, 0.5) is None
    steady = _click_audio(120, 16)
    late = np.r_[steady[:6 * mix.RATE], _click_audio(120, 10, first=0.12)]
    assert mix._in_step(a, 2, steady, 2, 8, 1, 0.5, halves=True)
    assert not mix._in_step(a, 2, late, 2, 8, 1, 0.5, halves=True)


def test_tempos_match_allows_60_cents_and_half_or_double_time():
    assert mix.tempos_match(120.0, 122.0) and mix.tempos_match(120.0, 118.0)
    assert mix.tempos_match(120.0, 124.0) and mix.tempos_match(124.0, 120.0)
    assert not mix.tempos_match(120.0, 125.0) and not mix.tempos_match(120.0, 150.0)
    assert mix.tempos_match(120.0, 60.5) and mix.tempos_match(120.0, 243.0)
    assert mix.tempos_match(np.array([[120.0], [150.0]]), np.array([[122.0, 150.0]])).tolist() == [[True, False], [False, True]]


def test_the_percussion_mix_accepts_tempos_up_to_its_own_wider_limit():
    assert mix.tempos_match(120.0, 125.0, mix.PERCUSSION_LOCK_CENTS)
    assert mix.tempos_match(120.0, 62.5, mix.PERCUSSION_LOCK_CENTS)
    assert not mix.tempos_match(120.0, 127.0, mix.PERCUSSION_LOCK_CENTS)


def test_lengths_count_felt_beats_fewer_for_a_dense_pair_within_their_limits():
    pop, rock, fast, slow = 60.0 / 104, 60.0 / 115, 60.0 / 170, 60.0 / 70
    assert mix.lengths(pop, 0.0) == pytest.approx((32 * pop, mix.MAX_FADE_SECONDS))
    assert mix.lengths(rock, 1.0) == pytest.approx((24 * rock, 13 * rock))
    assert mix.lengths(fast, 0.0) == pytest.approx((32 * fast, 18 * fast))
    assert mix.lengths(fast, 1.0) == pytest.approx((mix.MIN_BEAT_MIX_SECONDS, mix.MIN_MIX_SECONDS))
    assert mix.lengths(slow, 0.0) == pytest.approx((mix.MAX_BEAT_MIX_SECONDS, mix.MAX_FADE_SECONDS))
    middle = mix.lengths(rock, 0.5)
    assert mix.lengths(rock, 1.0)[0] < middle[0] < mix.lengths(rock, 0.0)[0]


def test_density_needs_both_parts_dense_and_counts_a_drumless_part_as_sparse():
    assert mix.density((0.2, 0.25)) == 1.0 and mix.density((mix.DENSE_SHARE, mix.DENSE_SHARE)) == 1.0
    assert mix.density((0.8, 0.25)) == 0.0 and mix.density((mix.SPARSE_SHARE, 0.2)) == 0.0
    assert mix.density((0.05, 0.25)) == 0.0 and mix.density((None, 0.25)) == 0.0
    assert mix.density((0.425, 0.2)) == pytest.approx(0.5)


def test_a_song_read_at_half_speed_is_felt_at_its_real_pace():
    filled = _click_audio(360, 12.0)
    assert mix._felt(filled, 80.0) == pytest.approx(60.0 / 160)
    assert mix._felt(_click_audio(160, 12.0), 80.0) == pytest.approx(60.0 / 80)
    assert mix._felt(filled, mix.FOLD_BPM) == pytest.approx(60.0 / mix.FOLD_BPM)
    assert mix._feel(filled, 80.0, 11.5, 5.0) == (60.0 / 80, None)


def test_a_dense_pair_crossfades_in_fewer_seconds_than_clicks_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(mix, '_felt', lambda audio, tempo: 0.5)
    t = np.arange(int(40.0 * mix.RATE)) / mix.RATE
    tone_out = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    tone_in = (0.3 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
    shares = [mix.drum_share(_click_audio(bpm, 20.0, level=0.48) + tone[:20 * mix.RATE], 0.0, 15.0)
              for bpm, tone in ((120, tone_out), (150, tone_in))]
    assert all(mix.DRUMLESS_SHARE < share <= mix.DENSE_SHARE for share in shares)
    dense = mix.plan_transition(_write(tmp_path / 'a.wav', _click_audio(120, 40.0, level=0.48) + tone_out), 30.0,
                                _write(tmp_path / 'b.wav', _click_audio(150, 40.0, level=0.48) + tone_in), 5.0)
    _assert_short_fade(dense)
    sparse = mix.plan_transition(_clicks(tmp_path / 'c.wav', 120, 40.0), 30.0, _clicks(tmp_path / 'd.wav', 150, 40.0), 5.0)
    _assert_short_fade(sparse)
    assert dense['fade_seconds'] == pytest.approx(mix.FADE_BEATS[1] * 0.5)
    assert sparse['fade_seconds'] == pytest.approx(mix.FADE_BEATS[0] * 0.5)
    assert dense['mix_seconds'] < sparse['mix_seconds']


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
    near = _clicks(tmp_path / 'c.wav', 122, 40.0, first=0.2)
    assert mix.plan_transition(song_out, 30.0, near, 5.0)['style'] == 'blend'
    monkeypatch.setattr(mix, '_agreement', lambda audio: mix.MIN_AGREEMENT_BITS - 0.1)
    plan = mix.plan_transition(song_out, 30.0, near, 5.0)
    _assert_short_fade(plan)
    assert plan['confidence'] != 'high'


def test_tempos_more_than_60_cents_apart_crossfade_even_when_their_beats_lock(tmp_path):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    apart = _clicks(tmp_path / 'b.wav', 125, 40.0, first=0.2)
    assert not mix.tempos_match(120.0, 125.0) and 1200 * np.log2(125 / 120) > mix.MAX_LOCK_CENTS
    plan = mix.plan_transition(song_out, 30.0, apart, 5.0)
    _assert_short_fade(plan)
    assert plan['confidence'] != 'high'
    double = mix.plan_transition(song_out, 30.0, _clicks(tmp_path / 'c.wav', 244, 40.0, first=0.2), 5.0)
    assert double['style'] == 'blend' and double['stretch'] == pytest.approx(240 / 244, abs=0.01)


def test_a_locked_pair_of_drum_led_parts_is_mixed_as_a_percussion_mix(tmp_path, monkeypatch):
    monkeypatch.setattr(mix, '_percussive', _REAL_PERCUSSIVE)
    drums_out = tmp_path / 'a.wav'
    drums_in = tmp_path / 'b.wav'
    soundfile.write(drums_out, np.stack([_click_audio(120, 40.0)] * 2, axis=1), mix.RATE)
    soundfile.write(drums_in, np.stack([_click_audio(122, 40.0, first=0.2)] * 2, axis=1), mix.RATE)
    plan = mix.plan_transition(str(drums_out), 30.0, str(drums_in), 5.0)
    assert plan['style'] == 'percussion_mix' and plan['confidence'] == 'high' and 'beat_rejection' not in plan
    assert plan['stretch'] == pytest.approx(120 / 122, abs=0.01)
    _assert_mix_length(plan)
    t = np.arange(int(40.0 * mix.RATE)) / mix.RATE
    chords = _click_audio(122, 40.0, first=0.2) + 0.3 * np.sin(2 * np.pi * 220.0 * t) * np.minimum(1.0, t)
    soundfile.write(tmp_path / 'c.wav', np.stack([chords] * 2, axis=1).astype(np.float32), mix.RATE)
    assert mix.plan_transition(str(drums_out), 30.0, str(tmp_path / 'c.wav'), 5.0)['style'] in ('blend', 'fade')


def test_drum_led_parts_whose_beats_line_up_get_a_percussion_mix_when_the_long_mix_fails(tmp_path, monkeypatch):
    outgoing = _clicks(tmp_path / 'a.wav', 120, 40.0)
    incoming = _clicks(tmp_path / 'b.wav', 126, 40.0, first=0.2)
    monkeypatch.setattr(mix, '_in_step', lambda a, start, b, entry, seconds, rate, beat, **kwargs: seconds < mix.PERCUSSION_SECONDS[1] + 0.1)
    monkeypatch.setattr(mix, '_percussive', lambda *args: True)

    plan = mix.plan_transition(outgoing, 30, incoming, 5)

    assert plan['style'] == 'fade' and plan['beat_rejection'] == 'tempo_gap'
    option, = plan['alternatives']
    assert option['style'] == 'percussion_mix' and option['beat_seconds'] == pytest.approx(0.5, abs=0.01)
    assert option['mix_seconds'] == pytest.approx(mix.PERCUSSION_SECONDS[1], abs=0.06)
    assert option['stretch'] == pytest.approx(120 / 126, abs=0.01)
    assert option['mix_at'] + option['mix_seconds'] <= 30.0 + 0.01


def test_the_percussion_mix_needs_its_drums_in_time_in_both_halves(tmp_path, monkeypatch):
    outgoing = _clicks(tmp_path / 'a.wav', 120, 40.0)
    incoming = _clicks(tmp_path / 'b.wav', 126, 40.0, first=0.2)
    monkeypatch.setattr(mix, '_percussive', lambda *args: True)
    asked = []
    real = mix._aligned_entry

    def recorded(*args, halves=False):
        asked.append(halves)
        return real(*args, halves=halves)

    monkeypatch.setattr(mix, '_aligned_entry', recorded)
    plan = mix.plan_transition(outgoing, 30, incoming, 5)
    assert asked and all(asked) and plan['alternatives'][0]['style'] == 'percussion_mix'


def test_parts_that_are_not_drum_led_get_no_percussion_mix(tmp_path, monkeypatch):
    rate = mix.RATE
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(10 * rate) / rate)).astype(np.float32)
    assert _REAL_PERCUSSIVE(_click_audio(120, 10.0), 0.0, 8.0) and not _REAL_PERCUSSIVE(tone, 0.0, 8.0)
    t = np.arange(10 * rate) / rate
    sub_bass = (0.5 * np.sin(2 * np.pi * 55.0 * t)).astype(np.float32)
    assert _REAL_PERCUSSIVE(_click_audio(120, 10.0) + sub_bass, 0.0, 8.0)
    assert not _REAL_PERCUSSIVE(_click_audio(120, 10.0) + tone, 0.0, 8.0)
    outgoing = _clicks(tmp_path / 'a.wav', 120, 40.0)
    incoming = _clicks(tmp_path / 'b.wav', 126, 40.0, first=0.2)
    monkeypatch.setattr(mix, '_in_step', lambda a, start, b, entry, seconds, rate, beat, **kwargs: seconds < mix.PERCUSSION_SECONDS[1] + 0.1)
    monkeypatch.setattr(mix, '_percussive', lambda *args: False)
    assert 'alternatives' not in mix.plan_transition(outgoing, 30, incoming, 5)


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
