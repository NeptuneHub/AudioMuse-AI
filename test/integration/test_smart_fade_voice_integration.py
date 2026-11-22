# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Real audio crossfades retain overlap and avoid silent or decaying cue windows.

Main Features:
* Sine mixtures exercise decoding, energy ranking, EQ and peak control together.
* Immediate voices keep sustained faders rather than a delayed sequential entry.
* Leading silence and natural fade-outs are rejected as useful mixing material.
* Recovery preserves fade duration while retaining more ending/opening audio.
* A large ending margin yields a few seconds when the requested ending remains energetic.
"""

import numpy as np
import pytest
import soundfile

from tasks import smart_fade_levels as levels
from tasks import smart_fade_voice as voice


@pytest.mark.parametrize('silent_intro', [0.0, 3.0])
def test_real_audio_keeps_a_gradual_overlap_and_avoids_silent_and_fading_edges(tmp_path, monkeypatch, silent_intro):
    rate = voice.VAD_RATE
    paths = [str(tmp_path / name) for name in ('out.wav', 'in.wav')]
    t = np.arange(40 * rate) / rate
    out = 0.12 * np.sin(2 * np.pi * 440 * t) + 0.12 * np.sin(2 * np.pi * 120 * t)
    out *= 10 ** (-np.maximum(0.0, t - 30.0) * 3.0 / 20)
    incoming = 0.12 * np.sin(2 * np.pi * 770 * t) + 0.12 * np.sin(2 * np.pi * 180 * t)
    incoming[t < silent_intro] = 0.0
    for path, audio in zip(paths, (out, incoming)):
        soundfile.write(path, audio, rate)
    monkeypatch.setattr(voice, 'voiced_spans', lambda audio: [(0.0, len(audio) / rate)])
    plan = {'mix_at': 28.0, 'mix_seconds': 10.0, 'in_at': 6.0, 'resume_at': 16.0,
            'style': 'fade', 'swap_at': 33.0, 'beat_seconds': None, 'stretch': 1.0,
            'tempo_from': None, 'tempo_to': None, 'confidence': 'medium'}
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', 0.0)
    before = voice.clear_voice(*paths, plan)
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', 4.0)
    chosen = voice.clear_voice(*paths, plan)
    saved = chosen['mix_at'] - before['mix_at'] + before['in_at'] - chosen['in_at']
    assert 3.0 <= saved <= 4.0
    assert chosen['mix_at'] >= before['mix_at'] and chosen['in_at'] <= before['in_at']
    assert chosen['mix_seconds'] == before['mix_seconds']
    assert chosen['voice']['recovered_seconds'] == pytest.approx(saved)
    answer = levels.transition(*paths, chosen)
    assert answer['duration'] == 10.0
    assert answer['info']['recovered_seconds'] == pytest.approx(saved)
    assert answer['start'] < 28.0
    assert silent_intro <= answer['to']['start'] <= silent_intro + 12.0
    assert 'delay' not in answer['to']
    assert answer['to']['continue_from'] == pytest.approx(answer['to']['start'] + answer['duration'], abs=1e-4)
    old, new = np.array(answer['from']['volume']), np.array(answer['to']['volume'])
    assert np.sum((old > 0.25) & (new > 0.25)) * answer['step'] >= 6.0
    phase = np.arctan2(new, old)
    original = np.linspace(0, np.pi / 2, len(old))
    assert np.allclose(phase[:40], original[:40], atol=0.001)
    assert np.max(phase - original) < 0.13
    mids = np.array(answer['to']['mid_db']) - np.array(answer['from']['mid_db'])
    assert 0.5 < mids[len(mids) // 2] < 1.5
    assert mids[0] == -6 and mids[-1] == 12


def test_ending_recovery_uses_strong_audio_without_shortening_the_fade_or_releasing_the_vocal_taper(tmp_path, monkeypatch):
    rate = voice.VAD_RATE
    t = np.arange(120 * rate) / rate
    ending = (0.12 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 120 * t))
    ending *= 10 ** (-np.maximum(0.0, t - 116.0) * 4.0 / 20)
    incoming = 0.12 * np.sin(2 * np.pi * 770 * t) + 0.1 * np.sin(2 * np.pi * 180 * t)
    paths = [str(tmp_path / name) for name in ('out.wav', 'in.wav')]
    for path, audio in zip(paths, (ending, incoming)):
        soundfile.write(path, audio, rate)
    monkeypatch.setattr(voice, 'voiced_spans', lambda audio: [(0.0, len(audio) / rate)])
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_MARGIN_SECONDS', 10.0)
    plan = {'mix_at': 110.0, 'mix_seconds': 10.0, 'in_at': 8.0, 'resume_at': 18.0,
            'style': 'fade', 'swap_at': 115.0, 'beat_seconds': None, 'stretch': 1.0,
            'tempo_from': None, 'tempo_to': None, 'confidence': 'medium', 'requested_end': 120.0}
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', 0.0)
    before = voice.clear_voice(*paths, plan)
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', 4.0)
    after = voice.clear_voice(*paths, plan)
    assert after['mix_at'] - before['mix_at'] == 4.0
    assert after['in_at'] == before['in_at'] and after['resume_at'] == before['resume_at']
    assert after['mix_seconds'] == before['mix_seconds'] == 10.0
    assert after['voice']['end_margin'] == 6.0
    assert after['voice']['overlap'] == before['voice']['overlap']
    answers = [levels.transition(*paths, selected) for selected in (before, after)]
    old_phase, new_phase = [np.arctan2(answer['to']['volume'], answer['from']['volume']) for answer in answers]
    assert np.allclose(new_phase, old_phase, atol=0.001)
    assert answers[1]['info']['ending_recovered_seconds'] == 4.0
    outgoing = np.asarray(answers[1]['from']['volume'])
    incoming = np.asarray(answers[1]['to']['volume'])
    assert np.sum((outgoing > 0.25) & (incoming > 0.25)) * answers[1]['step'] >= 6.0
