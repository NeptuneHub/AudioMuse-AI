# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Beats, downbeats and bar positions of one smart fade window.

Main Features:
* Peaks are logits above zero that are the largest within PEAK_FRAMES, with
  adjacent frames merged
* rhythm numbers beats by whole periods, so a missed or an extra beat keeps the
  tempo exact, and a Huber fit ignores one late beat; a wandering beat is not
  steady
* The bar position comes from the downbeats inside the span only: four beats
  a bar at the voted phase, None for three beats a bar or scattered downbeats
* A missing model raises BeatModelError, a ValueError, every time
* The real beat_this_small.onnx finds the tempo and the bar of synthetic drum
  patterns (kick on 1 and 3, snare on 2 and 4, a chord change on 1) at every
  bar shift and at three tempos
"""

import os

import numpy as np
import pytest

import config
from tasks import smart_fade_beats as beats


def test_peaks_are_logits_above_zero_that_win_their_neighbourhood():
    logit = np.full(40, -5.0)
    logit[[5, 6]] = 2.0
    logit[20] = 1.0
    logit[22] = 0.5
    logit[30] = -0.5
    assert beats.peaks(logit).tolist() == [5.5, 20.0]
    assert beats.peaks(np.zeros(0)).size == 0


def test_a_missed_and_an_extra_beat_keep_the_tempo_exact():
    period = 60.0 / 117.3
    times = list(0.4 + period * np.arange(48))
    del times[20]
    times.insert(30, times[29] + 0.3 * period)
    found = beats.rhythm(np.array(times), np.zeros(0))
    assert found['tempo'] == pytest.approx(117.3, abs=0.01)
    assert found['steady'] and found['residual'] < 0.005
    assert found['grid'][0] == pytest.approx(0.4, abs=1e-6)


def test_one_late_beat_does_not_bend_the_fitted_grid():
    period = 0.5
    times = 1.0 + period * np.arange(40)
    times[25] += 0.12
    found = beats.rhythm(times, np.zeros(0))
    assert found['period'] == pytest.approx(period, abs=1e-4)
    assert found['grid'][25] == pytest.approx(1.0 + 25 * period, abs=0.002)


def test_a_wandering_beat_is_not_steady():
    times = 0.5 * np.arange(50) + np.random.default_rng(1).uniform(-0.09, 0.09, 50)
    assert not beats.rhythm(times, np.zeros(0))['steady']


def test_too_few_beats_raise_value_error():
    with pytest.raises(ValueError):
        beats.rhythm(np.array([0.5, 1.0, 1.5]), np.zeros(0))


@pytest.mark.parametrize('phase', [0, 1, 2, 3])
def test_the_bar_position_is_the_voted_downbeat_phase(phase):
    times = 0.3 + 0.5 * np.arange(56)
    found = beats.rhythm(times, times[phase::4])
    assert found['phase'] == phase
    assert beats.downbeat_times(found).tolist() == pytest.approx(times[phase::4].tolist(), abs=1e-6)


def test_three_beats_a_bar_or_scattered_downbeats_have_no_bar_position():
    times = 0.3 + 0.5 * np.arange(56)
    assert beats.rhythm(times, times[::3])['phase'] is None
    scattered = times[[0, 3, 5, 10, 13, 18, 19, 26]]
    assert beats.rhythm(times, scattered)['phase'] is None
    assert beats.rhythm(times, times[:8:4])['phase'] is None


def test_only_the_downbeats_inside_the_span_decide_the_bar():
    times = 0.3 + 0.5 * np.arange(56)
    early = times[1:24:3]
    late = times[24::4]
    assert beats.rhythm(times, np.r_[early, late])['phase'] is None
    assert beats.rhythm(times, np.r_[early, late], span=(times[24] - 0.1, times[-1]))['phase'] == 0


def test_a_missing_model_raises_value_error_every_time(monkeypatch):
    monkeypatch.setattr(config, 'BEAT_THIS_MODEL_PATH', os.path.join(os.path.dirname(__file__), 'no_such_model.onnx'))
    monkeypatch.setattr(beats, '_session', None)
    for _ in range(2):
        with pytest.raises(ValueError):
            beats.events(np.zeros(beats.RATE, dtype=np.float32), beats.RATE)
    assert issubclass(beats.BeatModelError, ValueError)


def test_the_front_end_has_fifty_frames_a_second_and_128_bands():
    mel = beats.log_mel(np.random.default_rng(2).standard_normal(beats.RATE * 2).astype(np.float32) * 0.1)
    assert mel.shape == (2 * beats.RATE // beats.HOP + 1, beats.MELS)
    assert np.isfinite(mel).all() and mel.min() >= 0.0


def _drums(bpm, seconds, first, phase, rate=44100):
    rng = np.random.default_rng(3)
    out = np.zeros(int(seconds * rate))
    beat = 60.0 / bpm
    tail = np.arange(int(0.25 * rate)) / rate
    kick = np.sin(2 * np.pi * (50 + 60 * np.exp(-tail * 30)) * tail) * np.exp(-tail * 12)
    snare = rng.normal(0, 1, tail.size) * np.exp(-tail * 25) * 0.5
    hat = rng.normal(0, 1, int(0.05 * rate)) * np.exp(-np.arange(int(0.05 * rate)) / rate * 80) * 0.15
    chords = [(110.0, 138.6, 164.8), (98.0, 123.5, 146.8), (87.3, 110.0, 130.8), (82.4, 103.8, 123.5)]

    def hit(at, wave):
        start = int(at * rate)
        if start < out.size:
            size = min(wave.size, out.size - start)
            out[start:start + size] += wave[:size]

    count, at = 0, first
    while at < seconds:
        place = (count + phase) % 4
        hit(at, kick if place in (0, 2) else snare)
        hit(at, hat)
        hit(at + beat / 2, hat)
        if place == 0:
            notes = chords[((count + phase) // 4) % 4]
            span = np.arange(int(4 * beat * rate)) / rate
            pad = sum(0.08 * np.sin(2 * np.pi * f * span) for f in notes) * np.minimum(1, span * 20) * np.exp(-span * 0.5)
            hit(at, pad + 0.25 * np.sin(np.pi * notes[0] * span) * np.exp(-span * 1.5))
        at += beat
        count += 1
    return (out / np.max(np.abs(out)) * 0.8).astype(np.float32)


@pytest.mark.skipif(not os.path.isfile(config.BEAT_THIS_MODEL_PATH), reason='beat_this_small.onnx is not downloaded (BEAT_THIS_MODEL_PATH)')
@pytest.mark.parametrize('bpm,phase', [(120, 0), (120, 1), (97, 2), (128, 3)])
def test_the_real_model_finds_the_tempo_and_the_bar_of_a_drum_pattern(bpm, phase):
    found = beats.events(_drums(bpm, 25.0, 0.3, phase), 44100)
    result = beats.rhythm(found['beats'], found['downbeats'])
    assert result['tempo'] == pytest.approx(bpm, abs=0.5)
    assert result['steady']
    assert result['phase'] == (-phase) % 4
    assert result['grid'][0] == pytest.approx(0.3, abs=0.05)
