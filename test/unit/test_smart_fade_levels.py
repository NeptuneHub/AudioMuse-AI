# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""What a player executes for a smart fade transition, from real audio files.

Main Features:
* fader_scale leaves a steady mix alone, fills most of the hole a song fading
  on its own or a quiet intro leaves (from exactly 1 at both ends, smoothly),
  never passes SHAPE_LIMIT_DB and leaves a silent neighbour alone
* time_maps: a fade plays both songs as they are; a blend glides from the
  first song's beat to the next song's, lasts the mean of the two parts,
  starts and ends at the plan's seconds and goes on unstretched past them
* mix_shapes: a fade stays equal power without filters; a beat mix hands the
  bands over like a DJ's 3-band EQ (next song's highs, its mids, a two-beat
  equal-power bass swap, then the first song's mids and highs, every move
  monotonic), the faders only fade the two ends; a rise sweeps a high-pass
  that is open by the end of the swap
* swap_point puts the swap on the middle phrase when the next song has bass
  all along, on the first bar where its bass plays when it starts later, on
  the last bar in range when it starts after, the middle on a short mix
* song_level: a quiet song is raised only up to PEAK_CEILING, a loud one
  lowered to TARGET_LUFS, a silent one left at 1; an AAC read as mono
  measures like the same audio as a stereo WAV; a file that breaks while it
  is read is measured again from the fallback decode
* transition on real files: the answer's keys in their order (type, start,
  duration, middle, step, filters, from, to, info), one value per
  point in every list, volumes from exactly 1 and 0 to 0 and 1, speeds
  that integrate from start to the plan's seconds, filters only for a beat mix (all 0 for a
  crossfade); a next song with a louder bass gets the first song's bass
  raised before the swap and its own lowered after; a boost over quiet clicks is lowered wherever the sum would
  peak above the louder song; crossfade_plan is the default crossfade at
  the fade points, its tempos null
* percussion_mix: equal-power faders, one kick at a time (amplitude
  crossover, correlated kicks) and a hats exchange, executable on real files
* unlocked pairs: a drumless part gets the EQ blend (both parts' drum share
  measured at once), a louder next song the
  filter rise, overlapping voices the vocal handoff, else the crossfade; all
  at native speed, equal-power faders and one bass swap for the first two;
  into a much quieter intro the louder ending song still fades
"""

import numpy as np
import pytest
import soundfile
from concurrent.futures import ThreadPoolExecutor
import threading

from tasks import smart_fade_levels as levels
from tasks import smart_fade_mix as mix
from test.unit.conftest import smart_fade_click_events


@pytest.fixture(autouse=True)
def fresh_level_cache(monkeypatch):
    levels.clear_level_cache()
    monkeypatch.setattr(mix, '_events', smart_fade_click_events)
    yield
    levels.clear_level_cache()


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


def _write(path, audio, channels=2):
    soundfile.write(path, np.stack([audio] * channels, axis=1), mix.RATE)
    return str(path)


def _equal_power(count):
    share = np.linspace(0.0, 1.0, count)
    return np.cos(np.pi / 2 * share), np.sin(np.pi / 2 * share)


def test_level_cache_uses_content_and_invalidates_when_audio_changes(tmp_path, monkeypatch):
    levels.clear_level_cache()
    t = np.arange(mix.RATE) / mix.RATE
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    path = _write(tmp_path / 'first.wav', audio)
    copy = tmp_path / 'copy.wav'
    copy.write_bytes((tmp_path / 'first.wav').read_bytes())
    original, calls = levels.song_level, []

    def measured(source):
        calls.append(source['path'])
        return original(source)

    monkeypatch.setattr(levels, 'song_level', measured)
    first, second = levels.prepare_song(path), levels.prepare_song(str(copy))
    assert len(calls) == 1 and first[1] == second[1]
    first[1]['gain'] = 999.0
    assert levels.prepare_song(str(copy))[1]['gain'] != 999.0
    _write(tmp_path / 'first.wav', audio / 2)
    assert levels.prepare_song(path)[1] != second[1]
    assert len(calls) == 2


def test_level_cache_is_bounded_and_failed_measurements_can_be_retried(tmp_path, monkeypatch):
    levels.clear_level_cache()
    monkeypatch.setattr(levels.config, 'SMART_FADE_LEVEL_CACHE_SIZE', 2, raising=False)
    paths = [_write(tmp_path / f'{n}.wav', np.full(1000, 0.01 * (n + 1))) for n in range(3)]
    calls = []

    def measured(source):
        calls.append(source['path'])
        return {'gain': 1.0}

    monkeypatch.setattr(levels, 'song_level', measured)
    for path in paths + [paths[0]]:
        levels.prepare_song(path)
    assert len(calls) == 4
    levels.clear_level_cache()

    def broken(source):
        raise RuntimeError('measurement failed')

    monkeypatch.setattr(levels, 'song_level', broken)
    with pytest.raises(RuntimeError, match='measurement failed'):
        levels.prepare_song(paths[0])
    monkeypatch.setattr(levels, 'song_level', measured)
    assert levels.prepare_song(paths[0])[1] == {'gain': 1.0}


def test_same_audio_has_one_inflight_loudness_scan(tmp_path, monkeypatch):
    path = _write(tmp_path / 'same.wav', np.zeros(1000))
    original_key = levels.smart_fade_cache._key
    hashed = threading.Barrier(2)
    calls = []

    def key(path, source):
        result = original_key(path, source)
        hashed.wait(timeout=3)
        return result

    def measured(source):
        calls.append(source['path'])
        return {'gain': 1.0}

    monkeypatch.setattr(levels.smart_fade_cache, '_key', key)
    monkeypatch.setattr(levels, 'song_level', measured)
    with ThreadPoolExecutor(max_workers=2) as pool:
        prepared = list(pool.map(levels.prepare_song, (path, path)))
    assert len(calls) == 1 and prepared[0][1] == prepared[1][1]


def test_prepared_song_levels_are_reused_with_an_identical_transition(tmp_path, monkeypatch):
    t = np.arange(10 * mix.RATE) / mix.RATE
    paths = [_write(tmp_path / name, 0.1 * np.sin(2 * np.pi * hz * t))
             for name, hz in (('out.wav', 220), ('in.wav', 440))]
    plan = {'mix_at': 4.0, 'mix_seconds': 2.0, 'in_at': 1.0, 'resume_at': 3.0,
            'style': 'fade', 'swap_at': 5.0, 'beat_seconds': None, 'stretch': 1.0,
            'tempo_from': None, 'tempo_to': None, 'confidence': 'medium'}
    expected = levels.transition(*paths, plan)
    prepared = [levels.prepare_song(path) for path in paths]

    def repeated(path):
        raise AssertionError('A prepared song must not be scanned again.')

    monkeypatch.setattr(levels, 'prepare_song', repeated)
    assert levels.transition(*paths, plan, prepared=prepared) == expected


def test_the_faders_stay_put_on_a_steady_mix_and_fill_the_hole_of_a_fading_or_quiet_song():
    count, step = 101, mix.STEP_SECONDS
    fall, rise = _equal_power(count)
    assert levels.fader_scale(np.ones(count), 1.0, 1.0, step) == pytest.approx(np.ones(count))
    steady = np.ones(count)
    fading = 10.0 ** (np.linspace(0.0, -18.0, count) / 10.0)
    intro = 10.0 ** (np.concatenate([np.full(60, -12.0), np.linspace(-12.0, 0.0, count - 60)]) / 10.0)
    for out_energy, in_energy in ((fading, steady), (steady, intro)):
        mixed = fall ** 2 * out_energy + rise ** 2 * in_energy
        scale = levels.fader_scale(mixed, 1.0, 1.0, step)
        plain, shaped = 10.0 * np.log10(mixed), 10.0 * np.log10(scale ** 2 * mixed)
        assert plain.min() < -2.5 and shaped.min() > plain.min() + 2.0, (plain.min(), shaped.min())
        assert scale[0] == 1.0 and scale[-1] == pytest.approx(1.0)
        boost = 20.0 * np.log10(scale)
        assert boost.max() <= levels.SHAPE_LIMIT_DB + 1e-9 and np.abs(np.diff(boost)).max() < 1.0


def test_the_faders_never_pass_the_limit_and_leave_a_silent_neighbour_alone():
    count, step = 101, mix.STEP_SECONDS
    assert 20.0 * np.log10(levels.fader_scale(np.full(count, 1e-6), 1.0, 1.0, step).max()) == pytest.approx(levels.SHAPE_LIMIT_DB, abs=0.01)
    assert 20.0 * np.log10(levels.fader_scale(np.full(count, 10.0), 1.0, 1.0, step).min()) == pytest.approx(-levels.SHAPE_LIMIT_DB, abs=0.01)
    assert levels.fader_scale(np.full(count, 1e-6), 1.0, 1e-9, step) == pytest.approx(np.ones(count))


def test_a_fade_plays_both_songs_as_they_are_and_a_blend_glides_the_tempo():
    fade = {'mix_at': 100.0, 'mix_seconds': 10.0, 'in_at': 20.0, 'resume_at': 30.0, 'style': 'fade', 'swap_at': 105.0, 'beat_seconds': 0.5}
    seconds, swap, lead, follow, beats = levels.time_maps(fade)
    assert (seconds, swap, beats) == (10.0, 5.0, None)
    assert lead(np.array([0.0, 4.0, 10.0])).tolist() == [100.0, 104.0, 110.0] and follow(3.0) == 23.0
    blend = {'mix_at': 100.0, 'mix_seconds': 16.0, 'in_at': 20.0, 'resume_at': 35.36, 'style': 'blend', 'swap_at': 108.0, 'beat_seconds': 0.5}
    seconds, swap, lead, follow, beats = levels.time_maps(blend)
    assert seconds == pytest.approx((16.0 + 15.36) / 2) and swap == pytest.approx(16 * 0.5 - 0.02 * sum(k + 0.5 for k in range(16)) / 32)
    assert len(beats) == 33 and beats[0] == 0.0 and beats[-1] == pytest.approx(seconds) and beats[16] == pytest.approx(swap)
    assert float(lead(0.0)) == 100.0 and float(lead(seconds)) == pytest.approx(116.0)
    assert float(follow(0.0)) == 20.0 and float(follow(seconds)) == pytest.approx(35.36)
    steps = np.diff(lead(np.linspace(0.0, seconds, 200)))
    assert (steps > 0).all() and steps[0] < steps[-1]
    assert float(lead(0.1) - lead(0.0)) / 0.1 == pytest.approx(1.0, abs=0.002)
    assert float(follow(seconds) - follow(seconds - 0.1)) / 0.1 == pytest.approx(1.0, abs=0.002)
    assert float(lead(-1.0)) == 99.0 and float(follow(seconds + 2.0)) == pytest.approx(37.36)


def _first(times, values, test):
    return float(times[np.argmax(test(values))])


def test_a_beat_mix_hands_highs_mids_and_bass_over_like_a_dj_and_a_crossfade_stays_equal_power():
    times = np.linspace(0.0, 16.0, 1601)
    swap, beat = 8.0, 0.5
    fade = levels.mix_shapes('fade', 16.0, swap, beat, times)
    assert fade['eq'] is None and fade['from_volume'] ** 2 + fade['to_volume'] ** 2 == pytest.approx(np.ones(times.size))
    blend = levels.mix_shapes('blend', 16.0, swap, beat, times)
    eq = blend['eq']
    for curve in (blend['to_volume'], eq['to_high_db'], eq['to_mid_db'], eq['to_bass_db'],
                  -blend['from_volume'], -eq['from_high_db'], -eq['from_mid_db'], -eq['from_bass_db']):
        assert (np.diff(curve) >= -1e-9).all()
    assert (blend['to_volume'][0], blend['from_volume'][0], blend['from_volume'][-1], blend['to_volume'][-1]) == (0.0, 1.0, 0.0, 1.0)
    assert (eq['to_high_db'][0], eq['from_high_db'][-1]) == (levels.HIGH_CUT_DB, levels.HIGH_CUT_DB)
    highs_in = _first(times, eq['to_high_db'], lambda v: v >= 0.0)
    mids_in = _first(times, eq['to_mid_db'], lambda v: v >= 0.0)
    bass_out = _first(times, eq['from_bass_db'], lambda v: v < 0.0)
    bass_gone = _first(times, eq['from_bass_db'], lambda v: v <= levels.BASS_CUT_DB)
    mids_out = _first(times, eq['from_mid_db'], lambda v: v < 0.0)
    highs_out = _first(times, eq['from_high_db'], lambda v: v < 0.0)
    fader_in = _first(times, blend['to_volume'], lambda v: v >= 1.0)
    fader_out = _first(times, blend['from_volume'], lambda v: v < 1.0)
    assert fader_in <= levels.EDGE_BEATS * beat + 0.01 and fader_in < highs_in <= swap / 3.0 + 0.01 < mids_in <= swap - beat + 0.01
    assert swap - beat <= bass_out and bass_gone <= swap + beat + 0.01 and bass_gone - bass_out <= 2 * beat + 0.01
    assert swap + beat <= mids_out < highs_out < fader_out == pytest.approx(16.0 - levels.EDGE_BEATS * beat, abs=0.011)
    power = 10.0 ** (eq['from_bass_db'] / 10.0) + 10.0 ** (eq['to_bass_db'] / 10.0)
    assert power[(times > swap - beat) & (times < swap + beat)] == pytest.approx(1.0, abs=0.01)
    assert eq['to_highpass_hz'] is None
    rise = levels.mix_shapes('rise', 16.0, swap, beat, times)['eq']
    assert rise['to_bass_db'] is None and rise['to_highpass_hz'][0] == levels.RISE_FROM_HZ
    assert _first(times, rise['to_highpass_hz'], lambda v: v <= levels.RISE_TO_HZ + 1e-6) <= swap + beat + 0.01


def test_the_rise_high_pass_is_butterworth_like_the_page_filter():
    from scipy.signal import freqz

    for hz in (150, 1200, 4800):
        response = freqz(*levels._high_pass(mix.RATE, hz), worN=np.array([hz / 4, hz, 4 * hz]), fs=mix.RATE)[1]
        assert np.abs(response[1]) ** 2 == pytest.approx(0.5)
        assert np.abs(response[2]) ** 2 == pytest.approx(1.0, abs=0.01) and np.abs(response[0]) < 0.07


def test_a_percussion_mix_hands_one_kick_at_a_time_and_keeps_equal_power_faders():
    points = np.linspace(0.0, 10.0, 1001)
    shapes = levels.mix_shapes('percussion_mix', 10.0, 5.0, 0.5, points)
    eq = shapes['eq']
    assert shapes['from_volume'] ** 2 + shapes['to_volume'] ** 2 == pytest.approx(np.ones(points.size))
    assert shapes['from_volume'][500] == pytest.approx(shapes['to_volume'][500])
    out_kick, in_kick = 10.0 ** (eq['from_bass_db'] / 20.0), 10.0 ** (eq['to_bass_db'] / 20.0)
    inside = (out_kick > 0.2) & (in_kick > 0.2)
    assert inside.sum() > 50 and (out_kick + in_kick)[inside] == pytest.approx(1.0, abs=1e-6)
    assert eq['from_bass_db'][0] == 0.0 and eq['to_bass_db'][0] == levels.DRUM_BASS_CUT_DB[1]
    assert eq['from_bass_db'][-1] == levels.DRUM_BASS_CUT_DB[0] and eq['to_bass_db'][-1] == 0.0
    assert eq['from_high_db'][-1] == levels.DRUM_HIGH_CUT_DB[0] and eq['to_high_db'][0] == levels.DRUM_HIGH_CUT_DB[1]
    assert not np.any(eq['from_mid_db']) and not np.any(eq['to_mid_db']) and eq['to_highpass_hz'] is None


def test_vocal_ducking_preserves_the_locked_mix_treble_entrance():
    points = np.linspace(0.0, 8.0, 81)
    shapes = levels.mix_shapes('blend', 8, 4, 0.5, points)
    plan = {'style': 'blend', 'voice': {'clash': True, 'overlap': 0.33}}
    result = levels.vocal_shapes(plan, 8, points, shapes)
    assert result['eq']['to_high_db'][0] == levels.HIGH_CUT_DB
    assert result['eq']['from_high_db'][-1] == levels.HIGH_CUT_DB


def test_each_shelf_cuts_its_own_band_and_leaves_the_other_end_alone():
    from scipy.signal import freqz

    def response_db(coefficients, hz):
        _w, h = freqz(coefficients[0], coefficients[1], worN=[hz], fs=mix.RATE)
        return float(20.0 * np.log10(abs(h[0])))

    assert response_db(levels._high_shelf(mix.RATE, levels.HIGH_CUT_DB), 16000.0) == pytest.approx(levels.HIGH_CUT_DB, abs=1.0)
    assert response_db(levels._high_shelf(mix.RATE, levels.HIGH_CUT_DB), 200.0) == pytest.approx(0.0, abs=0.5)
    assert response_db(levels._low_shelf(mix.RATE, levels.BASS_CUT_DB), 30.0) == pytest.approx(levels.BASS_CUT_DB, abs=1.0)
    assert response_db(levels._low_shelf(mix.RATE, levels.BASS_CUT_DB), 5000.0) == pytest.approx(0.0, abs=0.5)
    assert response_db(levels._high_shelf(mix.RATE, 0.0), 8000.0) == pytest.approx(0.0, abs=1e-6)


def test_the_bass_swap_lands_on_the_bar_nearest_the_middle_where_the_next_song_has_bass():
    beats = np.arange(33) * 0.5
    points = np.linspace(0.0, 16.0, 161)
    everywhere = np.ones(points.size)
    assert levels.swap_point(beats, points, everywhere) == 8.0
    late = np.where(points >= 9.6, 1.0, 0.01)
    assert levels.swap_point(beats, points, late) == 10.0
    after_mix = np.where(points >= 12.0, 1.0, 0.01)
    assert levels.swap_point(beats, points, after_mix) == 10.0
    gap = np.where((points >= 8.0) & (points < 10.0), 0.01, 1.0)
    assert levels.swap_point(beats, points, gap) == 10.0
    assert levels.swap_point(beats[:9], points[:41], np.ones(41)) == 2.0
    speeding = np.concatenate([[0.0], np.cumsum(0.5 - 0.005 * np.arange(24))])
    assert levels.swap_point(speeding, np.linspace(0.0, speeding[-1], 107), np.ones(107)) == speeding[12]


def test_a_file_that_breaks_while_it_is_measured_is_measured_from_the_fallback_decode(tmp_path, monkeypatch):
    noise = (0.1 * np.random.default_rng(9).standard_normal(int(8.0 * mix.RATE))).astype(np.float32)
    path = _write(tmp_path / 'broken.wav', noise)
    whole = levels.song_level
    calls = []

    def lost_sync(source):
        calls.append(source['mono'] is None)
        if source['mono'] is None:
            raise soundfile.LibsndfileError(2, prefix='Error : ')
        return whole(source)

    monkeypatch.setattr(levels, 'song_level', lost_sync)
    source, level = levels.prepare_song(path)
    assert calls == [True, False] and source['mono'] is not None and source['mono'].size == noise.size
    assert level['loudness'] == pytest.approx(whole({'path': path, 'mono': None})['loudness'], abs=0.05)


def test_each_song_gets_the_gain_that_brings_it_to_the_target_without_passing_the_peak_ceiling(tmp_path):
    t = np.arange(int(20.0 * mix.RATE)) / mix.RATE
    noise = np.clip(0.15 * np.random.default_rng(5).standard_normal(t.size), -0.95, 0.95).astype(np.float32)
    loud = levels.song_level({'path': _write(tmp_path / 'loud.wav', noise), 'mono': None})
    assert loud['gain'] < 1.0 and loud['loudness'] + 20.0 * np.log10(loud['gain']) == pytest.approx(levels.TARGET_LUFS, abs=0.01)
    ticks = np.zeros(t.size, dtype=np.float32)
    for start in range(0, t.size - 300, mix.RATE):
        ticks[start:start + 220] = 0.8 * np.sin(2 * np.pi * 1000 * np.arange(220) / mix.RATE)
    quiet = levels.song_level({'path': _write(tmp_path / 'quiet.wav', ticks), 'mono': None})
    assert quiet['gain'] > 1.0 and quiet['gain'] * quiet['peak'] == pytest.approx(levels.PEAK_CEILING, rel=1e-3)
    assert quiet['loudness'] + 20.0 * np.log10(quiet['gain']) < levels.TARGET_LUFS - 1.0
    silent = levels.song_level({'path': _write(tmp_path / 'silent.wav', np.zeros(t.size, dtype=np.float32)), 'mono': None})
    assert silent['gain'] == 1.0 and silent['seconds'] == pytest.approx(20.0)
    guessed = levels.song_level({'path': 'unused', 'mono': noise})
    assert guessed['loudness'] == pytest.approx(loud['loudness'], abs=0.05)
    assert guessed['peak'] == pytest.approx(loud['peak'] * levels.GUESSED_PEAK_MARGIN, rel=1e-3)


def _assert_executable(plan, answer):
    assert list(answer) == ['type', 'start', 'duration', 'middle', 'step', 'filters', 'from', 'to', 'info']
    assert list(answer['from']) == ['gain', 'speed', 'volume', 'high_db', 'mid_db', 'bass_db']
    assert list(answer['to']) == ['gain', 'start', 'speed', 'volume', 'high_db', 'mid_db', 'bass_db', 'highpass_hz', 'continue_from']
    assert list(answer['info']) == ['tempo_from', 'tempo_to', 'confidence', 'loudness_from', 'loudness_to',
                                   *[key for key in ('beat_rejection', 'selection_reason') if key in plan]]
    assert list(answer['filters'].items()) == [('high_hz', levels.HIGH_HZ), ('mid_hz', levels.MID_HZ), ('mid_q', levels.MID_Q), ('bass_hz', levels.BASS_HZ)]
    current, upcoming = answer['from'], answer['to']
    count = len(current['volume'])
    lists = ('volume', 'high_db', 'mid_db', 'bass_db')
    assert all(len(song[key]) == count for song in (current, upcoming) for key in ('speed',) + lists)
    assert not any(v == 0.0 and np.signbit(v) for song in (current, upcoming) for key in lists for v in song[key])
    assert (count - 1) * answer['step'] == pytest.approx(answer['duration'], abs=1e-3)
    assert (current['volume'][0], upcoming['volume'][0], upcoming['volume'][-1]) == (1.0, 0.0, 1.0) and abs(current['volume'][-1]) < 1e-3
    _, _, lead, follow, _ = levels.time_maps(plan)
    points = np.arange(count) * answer['step']
    for song, first, truth in ((current, answer['start'], lead), (upcoming, upcoming['start'], follow)):
        played = first + np.concatenate([[0.0], np.cumsum((np.array(song['speed'][1:]) + song['speed'][:-1]) / 2 * answer['step'])])
        assert np.abs(played - truth(points)).max() < 0.002
        assert 1200 * np.log2(max(song['speed']) / min(song['speed'])) <= mix.MAX_LOCK_CENTS + 1
    assert answer['start'] == pytest.approx(plan['mix_at'], abs=1e-4)
    assert upcoming['start'] == pytest.approx(plan['in_at'], abs=1e-4)
    assert upcoming['continue_from'] == pytest.approx(played[-1], abs=0.002) == pytest.approx(plan['resume_at'], abs=1e-3)
    assert answer['info']['tempo_from'] == plan['tempo_from'] and answer['info']['confidence'] == plan['confidence']


def test_a_percussion_mix_plays_executable_curves_on_the_original_audio(tmp_path):
    outgoing = _write(tmp_path / 'out.wav', _click_audio(120, 24.0))
    incoming = _write(tmp_path / 'in.wav', _click_audio(124, 24.0))
    plan = dict(mix._handoff('percussion_mix', 3.0, 1.0, 10.0, 120 / 124, 0.5),
                tempo_from=120, tempo_to=124, confidence='high', beat_rejection='beat_tracking',
                selection_reason='percussion_mix')

    answer = levels.transition(outgoing, incoming, plan)

    assert answer['type'] == 'percussion_mix' and answer['to']['highpass_hz'] is None
    assert 'lowpass_hz' not in answer['from'] and 'delay' not in answer['to']
    assert min(answer['from']['bass_db']) < -15 and min(answer['to']['bass_db']) < -15
    _assert_executable(plan, answer)


def test_beatless_crossfade_catches_a_late_energy_dip_without_changing_cues(tmp_path, monkeypatch):
    monkeypatch.setattr(levels, 'unlocked_style', lambda *args: 'fade')
    clock = np.arange(20 * mix.RATE) / mix.RATE
    out = 0.29 * np.sin(2 * np.pi * 220 * clock) * np.where((clock > 7) & (clock < 10), 1.0, 0.10)
    inc = 0.27 * np.sin(2 * np.pi * 330 * clock) * np.where((clock > 8) & (clock < 12), 1.0, 0.10)
    song_a = _write(tmp_path / 'a.wav', out)
    song_b = _write(tmp_path / 'b.wav', inc)
    plan = dict(mix._handoff('fade', 6.0, 4.0, 10.0), tempo_from=None, tempo_to=None, confidence='medium')
    old = levels.transition(song_a, song_b, plan)
    tuned = levels.transition(song_a, song_b, dict(plan, beat_rejection='no_beat_grid'))
    assert old['duration'] == tuned['duration'] == 10.0 and old['start'] == tuned['start'] == 6.0
    assert tuned['from']['volume'][50] > old['from']['volume'][50]
    assert tuned['to']['volume'][50] > old['to']['volume'][50]
    assert tuned['from']['bass_db'] == old['from']['bass_db'] == [0.0] * 101


def test_unlocked_pairs_get_the_transition_their_parts_allow():
    rate = mix.RATE
    t = np.arange(10 * rate) / rate
    drums = _click_audio(110, 10.0)
    chords = (0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 277.2 * t)).astype(np.float32)
    plan = {'voice': {'overlap': 0.0}, 'tempo_from': 97.3, 'tempo_to': 108.0}
    assert levels.unlocked_style(plan, drums, chords, 1.0, 1.0) == 'eq_blend'
    assert levels.unlocked_style(plan, chords, drums, 1.0, 1.0) == 'eq_blend'
    assert levels.unlocked_style(plan, drums, drums, 1.0, 1.0) == 'fade'
    assert levels.unlocked_style(plan, drums, drums, 1.0, 0.5) == 'fade'
    assert levels.unlocked_style(plan, drums, drums, 1.0, 1.1) == 'filter_rise'
    assert levels.unlocked_style(dict(plan, tempo_from=92.9, tempo_to=122.8), drums, drums, 1.0, 1.1) == 'filter_rise'
    assert levels.unlocked_style(dict(plan, tempo_from=166.2, tempo_to=96.0), drums, drums, 1.0, 1.1) == 'fade'
    assert levels.unlocked_style(dict(plan, tempo_from=None), drums, drums, 1.0, 1.1) == 'fade'
    unrelated = {'voice': {'overlap': 0.5}, 'tempo_from': 166.2, 'tempo_to': 96.0}
    assert levels.unlocked_style(unrelated, drums, drums, 1.0, 1.1) == 'vocal_handoff'
    assert levels.unlocked_style({'voice': {'overlap': 0.5}}, drums, drums, 1.0, 1.0) == 'vocal_handoff'
    assert levels.unlocked_style({'voice': {'overlap': 0.2}}, drums, drums, 1.0, 1.0) == 'fade'


def test_both_parts_drum_shares_are_measured_at_once(monkeypatch):
    ready = threading.Barrier(2)

    def share(part, start, seconds, sample_rate=mix.RATE):
        ready.wait(timeout=3)
        return float(part[0])

    monkeypatch.setattr(mix, 'drum_share', share)
    with mix.switched_on({'eq_blend'}):
        assert levels.unlocked_style({}, np.full(10, 0.9), np.full(10, 0.05), 1.0, 1.0) == 'eq_blend'


def test_the_eq_blend_and_the_filter_rise_cross_equal_power_and_swap_the_bass_once():
    points = np.linspace(0.0, 10.0, 1001)
    for style in ('eq_blend', 'filter_rise'):
        shapes = levels.mix_shapes(style, 10.0, 5.0, 0.5, points)
        eq = shapes['eq']
        assert shapes['from_volume'] ** 2 + shapes['to_volume'] ** 2 == pytest.approx(np.ones(points.size))
        bass_out = 10.0 ** (eq['from_bass_db'] / 10.0)
        assert eq['from_bass_db'][0] == 0.0 and eq['from_bass_db'][-1] == levels.BASS_CUT_DB
        if style == 'eq_blend':
            power = bass_out + 10.0 ** (eq['to_bass_db'] / 10.0)
            assert power[(points > 4.5) & (points < 5.5)] == pytest.approx(1.0, abs=0.01)
            assert eq['to_mid_db'][0] == levels.BLEND_CUT_DB and eq['from_mid_db'][-1] == levels.BLEND_CUT_DB
            assert eq['to_high_db'][250] == pytest.approx(0.0) and eq['from_high_db'][700] == pytest.approx(0.0)
            alone = levels.mix_shapes(style, 10.0, 5.0, 0.5, points, bass_swap=False)['eq']
            assert alone['from_bass_db'] is None and alone['to_bass_db'] is None
            assert np.array_equal(alone['to_mid_db'], eq['to_mid_db'])
        else:
            assert eq['to_bass_db'] is None and not np.any(eq['to_mid_db']) and not np.any(eq['from_mid_db'])
            assert eq['to_highpass_hz'][0] == levels.RISE_FROM_HZ and eq['to_highpass_hz'][600] == pytest.approx(levels.RISE_TO_HZ)
            held = points <= 4.5
            assert eq['to_highpass_hz'][held] == pytest.approx(np.maximum(eq['to_highpass_hz'][held], levels.BASS_HZ))
            assert eq['to_highpass_hz'][450] == pytest.approx(levels.BASS_HZ) and (np.diff(eq['to_highpass_hz']) <= 1e-9).all()
            opening = (points > 4.5) & (points < 5.5)
            assert (eq['to_highpass_hz'][opening] < levels.BASS_HZ).all() and (eq['from_bass_db'][opening] < 0.0).all()
            assert eq['from_bass_db'][points <= 4.5].max() == 0.0 and eq['to_highpass_hz'][550] == pytest.approx(levels.RISE_TO_HZ)


def test_unlocked_kinds_play_the_eq_blend_and_the_rise_shapes_at_native_speed(tmp_path, monkeypatch):
    t = np.arange(int(40.0 * mix.RATE)) / mix.RATE
    song_out = _write(tmp_path / 'a.wav', _click_audio(120, 40.0) + (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32))
    song_in = _write(tmp_path / 'b.wav', _click_audio(150, 40.0) + (0.2 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32))
    plan = dict(levels.crossfade_plan(song_out, 30.0, song_in, 5.0), beat_rejection='tempo_gap')
    for kind, highpass in (('eq_blend', False), ('filter_rise', True), ('vocal_handoff', False)):
        monkeypatch.setattr(levels, 'unlocked_style', lambda *args, kind=kind: kind)
        answer = levels.transition(song_out, song_in, plan)
        _assert_executable(dict(plan, beat_rejection='tempo_gap'), answer)
        assert answer['type'] == kind and set(answer['from']['speed']) == set(answer['to']['speed']) == {1.0}
        assert (answer['to']['highpass_hz'] is not None) == highpass
        fall, rise = np.array(answer['from']['volume']), np.array(answer['to']['volume'])
        assert fall[: fall.size // 4].mean() > fall[-fall.size // 4:].mean() and rise[: rise.size // 4].mean() < rise[-rise.size // 4:].mean()
        if kind == 'eq_blend':
            assert answer['to']['bass_db'][0] <= -30 and answer['from']['bass_db'][-1] <= -30
            assert answer['to']['mid_db'][0] < -6 and answer['from']['mid_db'][-1] < -6
        if kind == 'filter_rise':
            assert answer['to']['highpass_hz'][0] == levels.RISE_FROM_HZ and answer['to']['highpass_hz'][-1] <= levels.RISE_TO_HZ + 0.01
            assert answer['from']['bass_db'][-1] <= -30


def test_an_unlocked_crossfade_into_a_much_quieter_intro_hands_over_instead_of_holding_the_loud_song(tmp_path, monkeypatch):
    t = np.arange(int(40.0 * mix.RATE)) / mix.RATE
    song_out = _write(tmp_path / 'a.wav', _click_audio(120, 40.0) + (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32))
    quiet = np.where(t < 25.0, 10.0 ** (-18.0 / 20.0), 1.0).astype(np.float32)
    song_in = _write(tmp_path / 'b.wav', (_click_audio(150, 40.0) + (0.2 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)) * quiet)
    monkeypatch.setattr(levels, 'unlocked_style', lambda *args: 'fade')
    plan = dict(levels.crossfade_plan(song_out, 30.0, song_in, 5.0), beat_rejection='tempo_gap')
    answer = levels.transition(song_out, song_in, plan)
    _assert_executable(plan, answer)
    fall, rise = np.array(answer['from']['volume']), np.array(answer['to']['volume'])
    late = int(0.75 * (fall.size - 1))
    assert fall[late] < 10.0 ** (levels.CONTINUITY_FLOOR_DB / 20.0) and fall[late] <= _equal_power(fall.size)[0][late] + 0.01
    assert rise.max() <= 1.0 + 1e-6


def test_an_eq_blend_swaps_the_bass_only_when_both_songs_have_one(tmp_path, monkeypatch):
    t = np.arange(int(40.0 * mix.RATE)) / mix.RATE
    song_out = _write(tmp_path / 'a.wav', _click_audio(120, 40.0) + (0.3 * np.sin(2 * np.pi * 70.0 * t)).astype(np.float32))
    monkeypatch.setattr(levels, 'unlocked_style', lambda *args: 'eq_blend')
    for name, hz, swapped in (('b.wav', 55.0, True), ('c.wav', 1320.0, False)):
        song_in = _write(tmp_path / name, _click_audio(150, 40.0) + (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32))
        plan = dict(levels.crossfade_plan(song_out, 30.0, song_in, 5.0), beat_rejection='tempo_gap')
        answer = levels.transition(song_out, song_in, plan)
        _assert_executable(plan, answer)
        assert answer['type'] == 'eq_blend' and answer['to']['highpass_hz'] is None
        if swapped:
            assert answer['to']['bass_db'][0] <= -30 and answer['from']['bass_db'][-1] <= -30
        else:
            assert answer['from']['bass_db'] == answer['to']['bass_db'] == [0.0] * len(answer['from']['volume'])
            assert answer['to']['mid_db'][0] < -6 and answer['from']['mid_db'][-1] < -6


def test_a_beat_mix_and_a_crossfade_give_every_list_a_player_executes_in_order(tmp_path):
    t = np.arange(int(40.0 * mix.RATE)) / mix.RATE
    bed = (0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
    song_out = _write(tmp_path / 'a.wav', _click_audio(120, 40.0) + bed)
    song_in = _write(tmp_path / 'b.wav', _click_audio(122, 40.0, first=0.2) + bed)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    answer = levels.transition(song_out, song_in, plan)
    _assert_executable(plan, answer)
    assert answer['type'] == 'beat_mix' and answer['to']['highpass_hz'] is None
    assert answer['from']['bass_db'][-1] == levels.BASS_CUT_DB and answer['to']['bass_db'][0] == levels.BASS_CUT_DB
    assert answer['from']['high_db'][-1] == levels.HIGH_CUT_DB and answer['to']['high_db'][0] == levels.HIGH_CUT_DB
    far = _write(tmp_path / 'c.wav', _click_audio(150, 40.0))
    with mix.switched_on(set()):
        fade = mix.plan_transition(song_out, 30.0, far, 5.0)
        flat = levels.transition(song_out, far, fade)
    _assert_executable(fade, flat)
    assert flat['type'] == 'crossfade' and flat['to']['highpass_hz'] is None
    assert all(set(flat[song][key]) == {0.0} for song in ('from', 'to') for key in ('high_db', 'mid_db', 'bass_db'))
    assert set(flat['from']['speed']) == set(flat['to']['speed']) == {1.0}
    assert max(abs(v - 1.0) for v in answer['to']['speed']) > 0.01
    assert flat['to']['continue_from'] - flat['to']['start'] == pytest.approx(flat['duration'], abs=1e-3)
    assert mix.MIN_BEAT_MIX_SECONDS - 1.0 <= answer['duration'] <= mix.MAX_BEAT_MIX_SECONDS
    assert fade['fade_seconds'] - 1e-3 <= flat['duration'] < fade['fade_seconds'] + fade['beat_seconds'] + 1e-3
    assert mix.MIN_MIX_SECONDS <= fade['fade_seconds'] <= mix.MAX_FADE_SECONDS


def test_each_band_is_fitted_to_the_two_songs_so_a_louder_next_bass_does_not_jump_in(tmp_path):
    t = np.arange(int(40.0 * mix.RATE)) / mix.RATE
    hum = np.sin(2 * np.pi * 60.0 * t).astype(np.float32)
    bed = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    song_out = _write(tmp_path / 'a.wav', _click_audio(120, 40.0) + 0.05 * hum + bed)
    song_in = _write(tmp_path / 'b.wav', _click_audio(122, 40.0, first=0.2) + 0.2 * hum + bed)
    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    answer = levels.transition(song_out, song_in, plan)
    _assert_executable(plan, answer)
    points = np.arange(len(answer['from']['volume'])) * answer['step']
    before, after = points < answer['middle'] - 1.0, points > answer['middle'] + 1.0
    assert max(np.array(answer['from']['bass_db'])[before]) > 1.0
    assert min(np.array(answer['to']['bass_db'])[after]) < -0.5
    assert max(answer['from']['bass_db'] + answer['to']['bass_db']) <= levels.BAND_LIMIT_DB
    assert answer['from']['bass_db'][0] == 0.0 and answer['to']['bass_db'][-1] == 0.0


def test_a_boost_is_lowered_wherever_the_sum_would_peak_above_the_louder_song(tmp_path):
    rate = mix.RATE
    audio = np.zeros(int(40.0 * rate), dtype=np.float32)
    burst = (0.9 * np.sin(2 * np.pi * 1000 * np.arange(int(0.006 * rate)) / rate)).astype(np.float32)
    for tenth in range(399):
        if not 200 <= tenth < 300 or tenth % 10 == 0:
            audio[int(tenth * rate / 10):int(tenth * rate / 10) + burst.size] += burst
    song_out = _write(tmp_path / 'a.wav', audio)
    song_in = _write(tmp_path / 'b.wav', audio)
    plan = levels.crossfade_plan(song_out, 30.0, song_in, 30.0 - mix.PLAIN_FADE_SECONDS)
    answer = levels.transition(song_out, song_in, plan)
    _assert_executable(plan, answer)
    assert answer['start'] == answer['to']['start'] == 30.0 - mix.PLAIN_FADE_SECONDS
    fall, rise = np.array(answer['from']['volume']), np.array(answer['to']['volume'])
    plain_fall, plain_rise = _equal_power(fall.size)
    assert (fall[5:-5] / plain_fall[5:-5]).max() > 1.2
    t = np.arange(int(answer['duration'] * rate)) / rate
    points = np.arange(fall.size) * answer['step']
    start = int(answer['start'] * rate)
    region = audio[start:start + t.size] * answer['from']['gain']
    assert (np.abs(region) * (np.interp(t, points, plain_fall) + np.interp(t, points, plain_rise))).max() > levels.PEAK_CEILING
    assert (np.abs(region) * (np.interp(t, points, fall) + np.interp(t, points, rise))).max() <= levels.PEAK_CEILING + 0.01


def test_a_pair_without_a_steady_beat_gets_an_eq_blend_at_the_default_crossfade_points(tmp_path):
    t = np.arange(int(40.0 * mix.RATE)) / mix.RATE
    song_out = _write(tmp_path / 'a.wav', (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32))
    song_in = _write(tmp_path / 'b.wav', (0.1 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32))
    with pytest.raises(ValueError):
        mix.plan_transition(song_out, 30.0, song_in, 5.0)
    plan = levels.crossfade_plan(song_out, 30.0, song_in, 5.0)
    assert (plan['style'], plan['stretch'], plan['tempo_from'], plan['tempo_to'], plan['beat_seconds']) == ('fade', 1.0, None, None, None)
    fade = mix.PLAIN_FADE_SECONDS
    assert (plan['mix_at'], plan['mix_seconds'], plan['in_at'], plan['resume_at'], plan['swap_at']) == (30.0 - fade, fade, 5.0, 5.0 + fade, 30.0 - fade / 2)
    assert plan['fade_seconds'] == fade
    answer = levels.transition(song_out, song_in, plan)
    _assert_executable(plan, answer)
    assert (answer['type'], answer['start'], answer['duration'], answer['middle']) == ('eq_blend', 30.0 - fade, fade, fade / 2)
    assert set(answer['from']['speed']) == set(answer['to']['speed']) == {1.0} and answer['to']['highpass_hz'] is None
    assert answer['to']['bass_db'][0] == levels.BASS_CUT_DB and answer['from']['bass_db'][-1] == levels.BASS_CUT_DB
    assert answer['info']['tempo_from'] is None and answer['info']['tempo_to'] is None
    assert answer['info']['loudness_from'] - answer['info']['loudness_to'] == pytest.approx(9.54, abs=0.3)
    assert answer['to']['gain'] / answer['from']['gain'] == pytest.approx(10.0 ** (9.54 / 20.0), rel=0.05)
    measured = levels.crossfade_plan(song_out, 30.0, song_in, 5.0, 6.0)
    assert (measured['mix_at'], measured['mix_seconds'], measured['fade_seconds']) == (24.0, 6.0, 6.0)
    with pytest.raises(ValueError):
        levels.crossfade_plan(song_out, 30.0, song_in, 39.9)


@pytest.mark.parametrize('seconds', [0.2, 2.0])
def test_short_audio_can_still_use_executable_positive_crossfades(tmp_path, seconds):
    song_out = _write(tmp_path / 'a.wav', _click_audio(120, 5.0))
    song_in = _write(tmp_path / 'b.wav', _click_audio(122, 5.0))
    plan = {'mix_at': 5.0 - seconds, 'mix_seconds': seconds, 'in_at': 0.0, 'resume_at': seconds,
            'swap_at': 5.0 - seconds / 2, 'style': 'fade', 'beat_seconds': None, 'stretch': 1.0,
            'tempo_from': 120.0, 'tempo_to': 122.0, 'confidence': 'medium'}
    answer = levels.transition(song_out, song_in, plan)
    assert answer['duration'] == seconds and answer['to']['start'] == 0.0
    assert answer['start'] + answer['duration'] == 5.0
    assert answer['to']['continue_from'] == seconds
    assert answer['from']['volume'][0] == answer['to']['volume'][-1] == 1.0
    assert answer['from']['volume'][-1] == answer['to']['volume'][0] == 0.0
    for side in ('from', 'to'):
        assert all(np.isfinite(answer[side][key]).all() for key in ('speed', 'volume', 'high_db', 'mid_db', 'bass_db'))
    _assert_executable(plan, answer)


@pytest.mark.parametrize('style', ['fade', 'blend', 'rise'])
def test_competing_vocals_keep_a_sustained_crossfade_and_gradual_midrange_exchange(tmp_path, style):
    out_path = _write(tmp_path / 'out.wav', _click_audio(120, 20.0))
    in_path = _write(tmp_path / 'in.wav', _click_audio(122, 20.0))
    plan = {'mix_at': 0.0, 'mix_seconds': 10.0, 'in_at': 2.0, 'resume_at': 12.0,
            'swap_at': 5.0, 'style': style, 'beat_seconds': 0.5, 'stretch': 1.0,
            'tempo_from': 120.0, 'tempo_to': 120.0, 'confidence': 'medium',
            'voice': {'clash': True, 'overlap': 1.0, 'end_margin': 10.0}}
    answer = levels.transition(out_path, in_path, plan)
    assert answer['to']['start'] == 2.0 and 'delay' not in answer['to']
    assert answer['to']['continue_from'] == 12.0
    assert set(answer['to']['speed']) == {1.0}
    points = np.linspace(0.0, answer['duration'], len(answer['to']['volume']))
    old, new = np.array(answer['from']['volume']), np.array(answer['to']['volume'])
    assert np.interp(2.0, points, new / np.maximum(old, 1e-9)) < 0.4
    assert np.sum((old > 0.25) & (new > 0.25)) * answer['step'] >= 6.0
    phase = np.arctan2(new, old)
    original = np.pi / 2 * points / answer['duration']
    assert np.allclose(phase[points <= 0.4 * answer['duration']], original[points <= 0.4 * answer['duration']], atol=0.001)
    assert np.max(phase - original) < 0.13
    difference = np.array(answer['to']['mid_db']) - np.array(answer['from']['mid_db'])
    assert difference[0] == -6.0 and difference[-1] == 12.0
    assert np.max(np.abs(np.diff(difference))) < 0.65
    assert answer['info']['vocal_overlap'] == 1.0


def test_a_pair_without_competing_vocals_keeps_the_original_mixer():
    points = np.linspace(0.0, 16.0, 161)
    shapes = levels.mix_shapes('blend', 16.0, 8.0, 0.5, points)
    result = levels.vocal_shapes({'voice': {'clash': False}}, 16.0, points, shapes)
    assert result is shapes
    assert np.min(result['eq']['to_mid_db']) == levels.MID_CUT_DB


@pytest.mark.parametrize('seconds', [10.0, 14.0])
@pytest.mark.parametrize('base_db', [6.0, 18.0])
def test_the_old_vocal_does_not_linger_under_the_new_singer_late_in_the_fade(monkeypatch, seconds, base_db):
    monkeypatch.setattr(levels.config, 'SMART_FADE_VOCAL_DUCK_DB', base_db)
    points = np.linspace(0.0, seconds, 141)
    plan = {'voice': {'clash': True, 'overlap': 1.0}}
    base = levels.mix_shapes('fade', seconds, seconds / 2, 0.5, points)
    incoming = base['to_volume'].copy()
    shapes = levels.vocal_shapes(plan, seconds, points, base)
    old, new = shapes['from_volume'], shapes['to_volume']
    eq = shapes['eq']
    late = int(0.75 * (len(points) - 1))
    relative = 20 * np.log10(old[late] / new[late]) + eq['from_mid_db'][late] - eq['to_mid_db'][late]
    assert relative <= -19.0
    assert np.interp(0.2 * seconds, points, eq['from_mid_db']) > -1.0
    assert np.array_equal(new, incoming)
    assert np.min(old ** 2 + new ** 2) > 0.9
    assert np.sum((old > 0.25) & (new > 0.25)) * seconds / (len(points) - 1) > 0.58 * seconds
    assert np.max(np.abs(np.diff(eq['from_mid_db']) / np.diff(points))) < 6.0


def test_the_late_vocal_taper_can_be_disabled_without_changing_the_incoming_fade(monkeypatch):
    monkeypatch.setattr(levels.config, 'SMART_FADE_VOCAL_TAIL_DB', 0.0)
    points = np.linspace(0.0, 10.0, 101)
    shapes = levels.mix_shapes('fade', 10.0, 5.0, 0.5, points)
    old, new = shapes['from_volume'].copy(), shapes['to_volume'].copy()
    result = levels.vocal_shapes({'voice': {'clash': True, 'overlap': 1.0}}, 10.0, points, shapes)
    assert np.array_equal(result['from_volume'], old) and np.array_equal(result['to_volume'], new)
    assert result['eq']['from_mid_db'][-1] == -6.0
