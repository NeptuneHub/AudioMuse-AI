# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Balanced smart fades retain overlap while preferring audible, nearby cues.

Main Features:
* Energy and vocal competition guide a joint, bounded cue search.
* Continuous vocals keep a real crossfade with moderate incoming trimming.
* Musical grids survive cue changes; missing detectors impose no veto.
* Local recovery keeps up to four more source seconds without shortening fades.
* MusiCNN covers short regions and edge patches; Silero complements it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from tasks import smart_fade_voice as voice

PLAN = {
    'mix_at': 100.0, 'mix_seconds': 16.0, 'in_at': 8.0, 'resume_at': 24.0, 'stretch': 1.0,
    'tempo_from': 120.0, 'tempo_to': 120.0, 'style': 'blend', 'swap_at': 108.0, 'beat_seconds': 0.5,
    'confidence': 'high',
}


def _inspection(start, seconds, spans, loudness=None):
    count = max(1, int(round(seconds / voice.mix.STEP_SECONDS)))
    return {'at': start, 'seconds': seconds, 'spans': spans, 'bounds': (0.0, seconds),
            'loudness': np.zeros(count) if loudness is None else loudness}


def _inspect_pair(monkeypatch, out_spans=None, in_spans=None, fade_at=None, duration=140.0, failed=False):
    def heard(path, start, seconds):
        if path == 'out':
            length = duration - start
            spans = [(0.0, length)] if out_spans is None else [(a - start, b - start) for a, b in out_spans]
            grid = start + np.arange(max(1, int(round(length / voice.mix.STEP_SECONDS)))) * voice.mix.STEP_SECONDS
            loudness = -np.maximum(0.0, grid - fade_at) if fade_at is not None else np.zeros(grid.size)
            return _inspection(start, length, None if failed else spans, loudness)
        return _inspection(0.0, seconds, None if failed else ([(0.0, seconds)] if in_spans is None else in_spans))
    monkeypatch.setattr(voice, '_heard', heard)


def test_continuous_vocals_keep_a_multisecond_overlap_and_a_moderate_incoming_cut(monkeypatch):
    _inspect_pair(monkeypatch)
    moved = voice.clear_voice('out', 'in', PLAN)
    assert 10.0 <= moved['mix_seconds'] <= 14.0
    assert moved['mix_at'] + moved['mix_seconds'] == 118.0
    assert moved['in_at'] == 6.0 and 'in_delay' not in moved
    assert moved['voice']['recovered_seconds'] == 4.0
    assert moved['resume_at'] == moved['in_at'] + moved['mix_seconds']
    assert moved['style'] == 'blend' and moved['beat_seconds'] == 0.5
    assert moved['voice']['clash'] is True
    assert PLAN['mix_seconds'] == 16.0


def test_a_terminal_cue_leaves_a_small_audible_tail_without_discarding_ten_seconds(monkeypatch):
    _inspect_pair(monkeypatch)
    late = dict(PLAN, mix_at=130.0, mix_seconds=10.0, swap_at=135.0)
    moved = voice.clear_voice('out', 'in', late)
    assert moved['mix_at'] + moved['mix_seconds'] == 138.0
    assert moved['voice']['end_margin'] == 2.0
    assert moved['mix_seconds'] == 10.0


def test_a_natural_fade_out_is_penalized_before_the_audible_ending(monkeypatch):
    _inspect_pair(monkeypatch, fade_at=110.0)
    moved = voice.clear_voice('out', 'in', dict(PLAN, mix_at=108.0))
    assert moved['mix_at'] < 108.0
    assert moved['mix_at'] + moved['mix_seconds'] < 126.0


def test_a_nearby_vocal_gap_reduces_competition_without_waiting_for_the_last_word(monkeypatch):
    _inspect_pair(monkeypatch, out_spans=[(0.0, 92.0), (112.0, 140.0)])
    moved = voice.clear_voice('out', 'in', PLAN)
    assert moved['voice']['overlap'] < 0.5
    assert 92.0 <= moved['mix_at'] <= 104.0
    assert moved['mix_at'] + moved['mix_seconds'] <= 116.0


def test_an_instrumental_intro_recovers_audio_and_keeps_the_beat_lock(monkeypatch):
    _inspect_pair(monkeypatch, in_spans=[(28.0, 45.0)])
    moved = voice.clear_voice('out', 'in', PLAN)
    assert moved['style'] == 'blend' and moved['beat_seconds'] == 0.5
    assert moved['in_at'] == 6.0 and moved['mix_at'] == 102.0
    assert moved['mix_seconds'] == 16.0 and moved['resume_at'] == 22.0
    assert moved['voice']['clash'] is False


def test_excessive_intro_trimming_is_bounded_without_forcing_a_zero_cue(monkeypatch):
    _inspect_pair(monkeypatch)
    plan = dict(PLAN, in_at=22.0, resume_at=38.0)
    moved = voice.clear_voice('out', 'in', plan)
    assert 0.0 < moved['in_at'] <= voice.config.SMART_FADE_ENTRY_LIMIT_SECONDS
    assert moved['in_at'] == 10.0


def test_failed_detectors_preserve_the_normal_cues_and_overlap(monkeypatch):
    _inspect_pair(monkeypatch, failed=True)
    moved = voice.clear_voice('out', 'in', PLAN)
    assert moved['mix_seconds'] == 16.0 and moved['mix_at'] == 100.0
    assert moved['in_at'] == 8.0 and 'in_delay' not in moved
    assert moved['voice']['clash'] is False


def test_short_clips_still_have_a_positive_transition(monkeypatch):
    _inspect_pair(monkeypatch, duration=12.0)
    plan = dict(PLAN, mix_at=4.0, mix_seconds=8.0, in_at=0.0, resume_at=8.0, style='fade')
    moved = voice.clear_voice('out', 'in', plan)
    assert 0.0 < moved['mix_seconds'] <= 12.0
    assert moved['mix_at'] >= 0.0
    assert moved['mix_at'] + moved['mix_seconds'] <= 12.0


def test_silent_audio_is_rejected_instead_of_returning_a_fake_transition(monkeypatch):
    monkeypatch.setattr(voice, '_heard', lambda path, start, seconds: dict(_inspection(start, 30.0, []), bounds=None))
    with pytest.raises(ValueError, match='audible crossfade'):
        voice.clear_voice('out', 'in', PLAN)


def test_leading_silence_does_not_count_toward_the_incoming_trim_limit(monkeypatch):
    def heard(path, start, seconds):
        result = _inspection(start, 140.0 - start if path == 'out' else 42.0, [])
        if path == 'in':
            result['bounds'] = (20.0, 42.0)
        return result

    monkeypatch.setattr(voice, '_heard', heard)
    moved = voice.clear_voice('out', 'in', PLAN)
    assert moved['in_at'] == 20.0
    assert moved['resume_at'] <= 42.0


def test_a_short_incoming_track_cannot_overrun_when_a_tempo_lock_is_dropped(monkeypatch):
    def heard(path, start, seconds):
        return _inspection(start, 140.0 - start if path == 'out' else 10.0, [])

    monkeypatch.setattr(voice, '_heard', heard)
    moved = voice.clear_voice('out', 'in', dict(PLAN, resume_at=26.0))
    assert moved['style'] == 'fade' and moved['beat_seconds'] is None
    assert moved['in_at'] + moved['mix_seconds'] <= 10.0001
    assert moved['resume_at'] <= 10.0001


def test_recovery_keeps_the_fade_and_recovers_four_seconds_on_both_beat_grids(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', 4.0, raising=False)
    plan = dict(PLAN, voice={'clash': True, 'overlap': 1.0, 'end_margin': 24.0})
    ending = _inspection(0.0, 140.0, [(0.0, 140.0)])
    starting = _inspection(0.0, 40.0, [(0.0, 40.0)])
    moved = voice._recover(ending, starting, plan)
    assert moved['mix_at'] == 102.0 and moved['in_at'] == 6.0
    assert moved['resume_at'] == 22.0 and moved['swap_at'] == 110.0
    assert moved['voice']['recovered_seconds'] == 4.0
    for key in ('mix_seconds', 'beat_seconds', 'style', 'stretch', 'confidence'):
        assert moved[key] == plan[key]
    assert plan['mix_at'] == 100.0 and plan['in_at'] == 8.0


def test_recovery_preserves_each_source_grid_at_different_tempos(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', 4.0, raising=False)
    plan = dict(PLAN, resume_at=23.2, stretch=0.95,
                voice={'clash': False, 'overlap': 0.0, 'end_margin': 24.0})
    ending = _inspection(0.0, 140.0, [])
    starting = _inspection(0.0, 40.0, [])
    moved = voice._recover(ending, starting, plan)
    a, b = moved['mix_at'] - plan['mix_at'], plan['in_at'] - moved['in_at']
    assert a / 1.0 == pytest.approx(round(a / 1.0))
    assert b / 0.95 == pytest.approx(round(b / 0.95), abs=1e-4)
    assert 3.0 <= a + b <= 4.0001
    assert moved['resume_at'] - moved['in_at'] == pytest.approx(15.2)


def test_recovery_stops_at_leading_silence_and_the_outgoing_tail_margin(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', 4.0, raising=False)
    plan = dict(PLAN, voice={'clash': False, 'overlap': 0.0, 'end_margin': 2.0})
    ending = _inspection(0.0, 118.0, [])
    starting = dict(_inspection(0.0, 40.0, []), bounds=(6.0, 40.0))
    moved = voice._recover(ending, starting, plan)
    assert moved['mix_at'] == plan['mix_at']
    assert moved['in_at'] == 6.0 and moved['voice']['recovered_seconds'] == 2.0
    assert moved['voice']['end_margin'] == 2.0


def test_recovery_rejects_later_fading_audio_instead_of_forcing_four_seconds(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', 4.0, raising=False)
    plan = dict(PLAN, style='fade', beat_seconds=None,
                voice={'clash': False, 'overlap': 0.0, 'end_margin': 24.0})
    grid = np.arange(1400) * voice.mix.STEP_SECONDS
    ending = _inspection(0.0, 140.0, [], -6 * np.maximum(0.0, grid - 100.0))
    starting = dict(_inspection(0.0, 40.0, []), bounds=(8.0, 40.0))
    moved = voice._recover(ending, starting, plan)
    assert moved['mix_at'] == 100.0 and moved['in_at'] == 8.0


def test_recovery_limits_new_vocal_competition(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', 4.0, raising=False)
    plan = dict(PLAN, voice={'clash': False, 'overlap': 0.0, 'end_margin': 2.0})
    ending = _inspection(0.0, 118.0, [(0.0, 118.0)])
    starting = _inspection(0.0, 40.0, [(0.0, 8.0)])
    moved = voice._recover(ending, starting, plan)
    assert moved['voice']['overlap'] <= 0.05
    assert moved['voice']['recovered_seconds'] < 4.0
    assert moved['voice']['clash'] is False


@pytest.mark.parametrize('budget,spans', [(0.0, []), (4.0, None)])
def test_recovery_can_be_disabled_and_missing_detectors_do_not_move_cues(monkeypatch, budget, spans):
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', budget, raising=False)
    plan = dict(PLAN, voice={'clash': False, 'overlap': 0.0, 'end_margin': 24.0})
    ending = _inspection(0.0, 140.0, spans)
    starting = _inspection(0.0, 40.0, [])
    assert voice._recover(ending, starting, plan) is plan


def test_sung_patches_and_silero_form_a_union_and_one_failed_detector_is_tolerated(monkeypatch):
    starts = np.arange(10) * 0.5
    scores = np.array([0.05, 0.2, 0.05, 0.2, 0.3, 0.25, 0.05, 0.05, 0.2, 0.2])
    monkeypatch.setattr(voice, 'vocal_scores', lambda audio: (starts, scores))
    assert voice.sung_spans(np.zeros(1)) == [(1.5, 2.5 + voice.PATCH_SECONDS), (4.0, 4.5 + voice.PATCH_SECONDS)]
    monkeypatch.setattr(voice, 'spoken_spans', lambda audio: [(0.2, 0.9)])
    assert voice.voiced_spans(np.zeros(1))[0] == (0.2, 0.9)

    def broken(audio):
        raise RuntimeError('model missing')

    monkeypatch.setattr(voice, 'sung_spans', broken)
    assert voice.voiced_spans(np.zeros(1)) == [(0.2, 0.9)]
    monkeypatch.setattr(voice, 'spoken_spans', broken)
    with pytest.raises(RuntimeError):
        voice.voiced_spans(np.zeros(1))


@pytest.mark.parametrize('seconds', [1.0, 4.3])
def test_musicnn_covers_short_regions_and_the_final_patch(monkeypatch, seconds):
    embedding = SimpleNamespace(run=lambda names, feed: [np.zeros((len(next(iter(feed.values()))), 1), dtype=np.float32)])
    prediction = SimpleNamespace(run=lambda names, feed: [np.ones((len(next(iter(feed.values()))), 1), dtype=np.float32)])
    monkeypatch.setattr(voice, '_musicnn', lambda: (embedding, prediction, [0]))
    audio = np.zeros(int(seconds * voice.VAD_RATE), dtype=np.float32)
    starts, scores = voice.vocal_scores(audio)
    assert starts[0] == 0.0 and len(starts) == len(scores)
    assert starts[-1] + voice.PATCH_SECONDS >= seconds - voice.MEL_HOP / voice.VAD_RATE
    assert voice.sung_spans(audio)


def test_real_audio_inspection_measures_energy_and_clips_detector_padding(tmp_path, monkeypatch):
    path = str(tmp_path / 'tone.wav')
    t = np.arange(3 * voice.VAD_RATE) / voice.VAD_RATE
    soundfile.write(path, 0.2 * np.sin(2 * np.pi * 440.0 * t), voice.VAD_RATE)
    monkeypatch.setattr(voice, 'voiced_spans', lambda audio: [(-0.2, 5.0)])
    heard = voice._heard(path, 1.0, None)
    assert heard['spans'] == [(0.0, 2.0)] and heard['at'] == 1.0
    assert np.isfinite(heard['loudness']).all() and len(heard['loudness']) > 10
