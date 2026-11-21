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
    chosen = voice.clear_voice(*paths, plan)
    answer = levels.transition(*paths, chosen)
    assert answer['duration'] == 10.0
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
