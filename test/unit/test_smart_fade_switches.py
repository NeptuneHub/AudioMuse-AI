# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The setup wizard's Smart Fade switches, through the real planner and mixer.

Main Features:
* Every switch ships on and maps one transition; the crossfade has none
* All beat-matched kinds off: a click-track lock is planned at its own speed
  over the crossfade length ('switched_off'), with no beat committee, onset
  or drum check; lock=False plans the same pair the same way
* Beat mix off: a lock plays only where the rise (6 dB louder) or the
  percussion mix fits; rise off: a louder next song gets the beat mix;
  percussion off: drum-led parts get the beat mix and no alternative
* The mixer never turns a lock into a kind switched off; without a lock the
  EQ blend, filter rise and vocal handoff fall through to the next that fits
  (the drum check skipped with the EQ blend off), and with the vocal handoff
  off the crossfade has flat filters even where voices overlap
* A locked percussion mix meeting two voices becomes the crossfade ending on
  the same second when the beat mix is off
"""

import math

import numpy as np
import pytest
import soundfile

import config
from tasks import smart_fade_levels as levels
from tasks import smart_fade_mix as mix
from tasks import smart_fade_voice as voice

KINDS = {
    'blend': 'SMART_FADE_BEAT_MIX_ENABLED',
    'rise': 'SMART_FADE_BEAT_RISE_ENABLED',
    'percussion_mix': 'SMART_FADE_PERCUSSION_MIX_ENABLED',
    'eq_blend': 'SMART_FADE_EQ_BLEND_ENABLED',
    'filter_rise': 'SMART_FADE_FILTER_RISE_ENABLED',
    'vocal_handoff': 'SMART_FADE_VOCAL_HANDOFF_ENABLED',
}
PLAN = {
    'mix_at': 100.0, 'mix_seconds': 16.0, 'in_at': 8.0, 'resume_at': 24.0, 'stretch': 1.0,
    'tempo_from': 120.0, 'tempo_to': 120.0, 'style': 'blend', 'swap_at': 108.0, 'beat_seconds': 0.5,
    'confidence': 'high',
}


@pytest.fixture(autouse=True)
def _every_kind_on_and_no_drum_led_part(monkeypatch):
    for flag in KINDS.values():
        monkeypatch.setattr(config, flag, True)
    monkeypatch.setattr(mix, '_percussive', lambda *args, **kwargs: False)
    levels.clear_level_cache()
    yield
    levels.clear_level_cache()


def _switch(monkeypatch, **states):
    for style, on in states.items():
        monkeypatch.setattr(config, KINDS[style], on)


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


def _write(path, audio):
    soundfile.write(path, np.stack([audio, audio], axis=1), mix.RATE)
    return str(path)


def _clicks(path, bpm, seconds, first=0.0, level=0.8):
    return _write(path, _click_audio(bpm, seconds, first, level))


def _never(*args, **kwargs):
    raise AssertionError('a beat check ran with every beat-matched kind switched off')


def test_every_switch_ships_on_and_names_one_transition_the_crossfade_has_none():
    with open(config.__file__, encoding='utf-8') as handle:
        source = handle.read()
    for flag in KINDS.values():
        assert 'os.environ.get("%s", "true")' % flag in source, flag
    assert mix.SWITCHES == KINDS
    assert mix.enabled('fade') and 'fade' not in mix.SWITCHES
    assert set(mix.LOCKED_STYLES) == {'blend', 'rise', 'percussion_mix'}


def test_with_every_beat_matched_kind_off_a_lock_is_planned_at_its_own_speed_without_the_beat_checks(tmp_path, monkeypatch):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 122, 40.0, first=0.2)
    locked = mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert locked['style'] == 'blend' and locked['stretch'] != 1.0
    _switch(monkeypatch, blend=False, rise=False, percussion_mix=False)
    for name in ('_agreement', '_aligned_entry', '_percussive', '_percussion_mix'):
        monkeypatch.setattr(mix, name, _never)

    plan = mix.plan_transition(song_out, 30.0, song_in, 5.0)

    assert plan['style'] == 'fade' and plan['beat_rejection'] == 'switched_off' and 'alternatives' not in plan
    assert plan['stretch'] == 1.0 and plan['confidence'] != 'high'
    assert plan['fade_seconds'] == locked['fade_seconds'] and plan['mix_seconds'] < locked['mix_seconds']
    beats = round(plan['mix_seconds'] / plan['beat_seconds'])
    assert beats == math.ceil(plan['fade_seconds'] / plan['beat_seconds'] - 1e-3)
    assert plan['resume_at'] - plan['in_at'] == pytest.approx(plan['mix_seconds'], abs=0.002)
    assert plan['mix_at'] + plan['mix_seconds'] == pytest.approx(30.0, abs=0.3)


def test_a_plan_without_a_lock_is_the_one_every_beat_kind_switched_off_gives(tmp_path, monkeypatch):
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    song_in = _clicks(tmp_path / 'b.wav', 122, 40.0, first=0.2)
    for name in ('_agreement', '_aligned_entry'):
        monkeypatch.setattr(mix, name, _never)

    again = mix.plan_transition(song_out, 30.0, song_in, 5.0, lock=False)
    _switch(monkeypatch, blend=False, rise=False, percussion_mix=False)

    assert again == mix.plan_transition(song_out, 30.0, song_in, 5.0)
    assert again['style'] == 'fade' and again['stretch'] == 1.0 and again['beat_rejection'] == 'switched_off'


def test_with_the_beat_mix_off_a_lock_plays_only_where_the_rise_fits(tmp_path, monkeypatch):
    song_in = _clicks(tmp_path / 'b.wav', 122, 40.0, first=0.2)
    even = _clicks(tmp_path / 'a.wav', 120, 40.0)
    quiet = _clicks(tmp_path / 'q.wav', 120, 40.0, level=0.1)
    _switch(monkeypatch, blend=False)

    plain = mix.plan_transition(even, 30.0, song_in, 5.0)
    rising = mix.plan_transition(quiet, 30.0, song_in, 5.0)

    assert plain['style'] == 'fade' and plain['beat_rejection'] == 'switched_off' and plain['stretch'] == 1.0
    assert rising['style'] == 'rise' and rising['confidence'] == 'high' and 'beat_rejection' not in rising


def test_with_the_rise_off_a_louder_locked_song_gets_the_beat_mix(tmp_path, monkeypatch):
    quiet = _clicks(tmp_path / 'q.wav', 120, 40.0, level=0.1)
    song_in = _clicks(tmp_path / 'b.wav', 122, 40.0, first=0.2)
    assert mix.plan_transition(quiet, 30.0, song_in, 5.0)['style'] == 'rise'
    _switch(monkeypatch, rise=False)
    assert mix.plan_transition(quiet, 30.0, song_in, 5.0)['style'] == 'blend'


def test_with_the_percussion_mix_off_drum_led_parts_get_the_beat_mix_and_no_alternative(tmp_path, monkeypatch):
    monkeypatch.setattr(mix, '_percussive', lambda *args: True)
    song_out = _clicks(tmp_path / 'a.wav', 120, 40.0)
    near = _clicks(tmp_path / 'b.wav', 122, 40.0, first=0.2)
    apart = _clicks(tmp_path / 'c.wav', 126, 40.0, first=0.2)
    assert mix.plan_transition(song_out, 30.0, near, 5.0)['style'] == 'percussion_mix'
    _switch(monkeypatch, blend=False, rise=False)
    assert mix.plan_transition(song_out, 30.0, near, 5.0)['style'] == 'percussion_mix'
    _switch(monkeypatch, blend=True, percussion_mix=False)
    assert mix.plan_transition(song_out, 30.0, near, 5.0)['style'] == 'blend'
    monkeypatch.setattr(mix, '_in_step', lambda a, start, b, entry, seconds, rate, beat, **kwargs: seconds < mix.PERCUSSION_SECONDS[1] + 0.1)
    assert 'alternatives' not in mix.plan_transition(song_out, 30.0, apart, 5.0)
    _switch(monkeypatch, percussion_mix=True)
    assert mix.plan_transition(song_out, 30.0, apart, 5.0)['alternatives'][0]['style'] == 'percussion_mix'


def _locked(style):
    return {'mix_at': 2.0, 'mix_seconds': 8.0, 'in_at': 2.0, 'resume_at': 10.0, 'swap_at': 6.0, 'style': style,
            'beat_seconds': 0.5, 'stretch': 1.0, 'tempo_from': 120.0, 'tempo_to': 120.0, 'confidence': 'high'}


def test_the_mixer_never_plays_a_locked_kind_that_is_switched_off(tmp_path, monkeypatch):
    dip = _click_audio(120, 20.0)
    dip[int(2.0 * mix.RATE):int(12.0 * mix.RATE)] *= 0.05
    quiet_there = _write(tmp_path / 'a.wav', dip)
    steady = _clicks(tmp_path / 'b.wav', 120, 20.0)
    assert levels.transition(quiet_there, steady, _locked('blend'))['type'] == 'beat_rise'
    assert levels.transition(steady, steady, _locked('rise'))['type'] == 'beat_mix'
    _switch(monkeypatch, rise=False)
    assert levels.transition(quiet_there, steady, _locked('blend'))['type'] == 'beat_mix'
    _switch(monkeypatch, rise=True, blend=False)
    assert levels.transition(steady, steady, _locked('rise'))['type'] == 'beat_rise'


def test_unlocked_kinds_switched_off_fall_through_to_the_next_that_fits(monkeypatch):
    rate = mix.RATE
    t = np.arange(10 * rate) / rate
    drums = _click_audio(110, 10.0)
    chords = (0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 277.2 * t)).astype(np.float32)
    plan = {'voice': {'overlap': 0.5}, 'tempo_from': 97.3, 'tempo_to': 108.0}
    assert levels.unlocked_style(plan, drums, chords, 1.0, 1.1) == 'eq_blend'
    _switch(monkeypatch, eq_blend=False)
    monkeypatch.setattr(mix, 'drum_share', _never)
    assert levels.unlocked_style(plan, drums, chords, 1.0, 1.1) == 'filter_rise'
    assert levels.unlocked_style(plan, drums, chords, 1.0, 1.0) == 'vocal_handoff'
    _switch(monkeypatch, filter_rise=False)
    assert levels.unlocked_style(plan, drums, chords, 1.0, 1.1) == 'vocal_handoff'
    _switch(monkeypatch, vocal_handoff=False)
    assert levels.unlocked_style(plan, drums, chords, 1.0, 1.1) == 'fade'


def test_with_the_vocal_handoff_off_the_crossfade_never_filters_the_voices(tmp_path, monkeypatch):
    _switch(monkeypatch, eq_blend=False, filter_rise=False)
    out_path = _clicks(tmp_path / 'out.wav', 120, 20.0)
    in_path = _clicks(tmp_path / 'in.wav', 122, 20.0)
    plan = dict(_locked('fade'), beat_seconds=None, confidence='medium',
                voice={'clash': True, 'overlap': 1.0, 'end_margin': 10.0})
    handed = levels.transition(out_path, in_path, plan)
    assert handed['type'] == 'vocal_handoff' and min(handed['from']['mid_db']) < -3.0
    _switch(monkeypatch, vocal_handoff=False)

    plain = levels.transition(out_path, in_path, plan)

    assert plain['type'] == 'crossfade'
    flat = [0.0] * len(plain['from']['volume'])
    for side in ('from', 'to'):
        assert plain[side]['high_db'] == plain[side]['mid_db'] == plain[side]['bass_db'] == flat
    assert plain['to']['highpass_hz'] is None
    assert (plain['from']['volume'][0], plain['to']['volume'][0], plain['to']['volume'][-1]) == (1.0, 0.0, 1.0)


def test_a_locked_percussion_mix_meeting_two_voices_crossfades_on_the_same_second_with_the_beat_mix_off(monkeypatch):
    def heard(path, start, seconds):
        length = 140.0 - start if path == 'out' else seconds
        count = max(1, int(round(length / mix.STEP_SECONDS)))
        return {'at': start if path == 'out' else 0.0, 'seconds': length, 'spans': [(0.0, length)],
                'bounds': (0.0, length), 'loudness': np.zeros(count)}

    monkeypatch.setattr(voice, '_heard', heard)
    plan = dict(PLAN, style='percussion_mix', fade_seconds=7.0)
    blend = voice.clear_voice('out', 'in', plan)
    assert blend['style'] == 'blend' and blend['voice']['clash']
    _switch(monkeypatch, blend=False)

    moved = voice.clear_voice('out', 'in', plan)

    assert moved['style'] == 'fade' and moved['beat_rejection'] == 'switched_off'
    assert moved['beat_seconds'] is None and moved['stretch'] == 1.0 and moved['confidence'] == 'medium'
    assert moved['mix_seconds'] == 7.0 and moved['resume_at'] == pytest.approx(moved['in_at'] + 7.0)
    assert moved['mix_at'] + moved['mix_seconds'] == pytest.approx(blend['mix_at'] + blend['mix_seconds'], abs=1e-4)
