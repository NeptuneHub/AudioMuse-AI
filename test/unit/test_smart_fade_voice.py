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
* Local recovery keeps up to four more source seconds without shortening fades,
  moving a beat mix only by whole bars (the next song by its own entry_bar);
  final cues off that grid are caught.
* Ending-only recovery restores early cues while preserving the approved vocal taper.
* MusiCNN covers short regions and edge patches; Silero complements it, both
  hearing one read at once; MusiCNN runs PREDICT_PATCHES patches a call, in order.
* A percussion mix alternative wins only where no voices overlap; a locked
  percussion mix with competing voices is a beat mix (it gets the vocal EQ).
* A beat mix whose moved cues lose the beat keeps its verified cues when no voices
  clash there, else it is the default 10-second crossfade.
"""

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading

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


def test_rounded_beat_period_keeps_the_already_verified_beat_lock(monkeypatch):
    _inspect_pair(monkeypatch, out_spans=[], in_spans=[])
    duration = round(32 * 0.474032, 4)
    plan = dict(PLAN, mix_seconds=duration, beat_seconds=0.4740, resume_at=8 + duration)

    moved = voice.clear_voice('out', 'in', plan)

    assert moved['style'] == 'blend'
    assert moved['mix_seconds'] == duration
    assert moved['confidence'] == 'high'


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


def test_a_beat_mix_whose_moved_cues_lose_the_beat_becomes_the_default_crossfade(monkeypatch):
    _inspect_pair(monkeypatch)
    heard = voice._heard
    monkeypatch.setattr(voice, '_heard', lambda path, start, seconds: dict(heard(path, start, seconds), samples=np.zeros(16000)))
    monkeypatch.setattr(voice.mix, '_in_step', lambda *args: False)
    moved = voice.clear_voice('out', 'in', PLAN)
    assert moved['style'] == 'fade' and moved['beat_rejection'] == 'audible_cue' and moved['beat_seconds'] is None
    assert moved['mix_seconds'] == voice.mix.MIN_MIX_SECONDS and moved['mix_at'] + moved['mix_seconds'] == 118.0
    assert moved['resume_at'] == moved['in_at'] + voice.mix.MIN_MIX_SECONDS


def test_a_verified_beat_mix_stays_when_moved_cues_lose_the_beat_and_no_voices_clash(monkeypatch):
    _inspect_pair(monkeypatch, out_spans=[], in_spans=[], fade_at=110.0)
    heard = voice._heard
    monkeypatch.setattr(voice, '_heard', lambda path, start, seconds: dict(heard(path, start, seconds), samples=np.zeros(16000)))
    monkeypatch.setattr(voice.mix, '_in_step', lambda *args: False)
    plan = dict(PLAN, mix_at=108.0, swap_at=116.0)
    kept = voice.clear_voice('out', 'in', plan)
    assert kept['style'] == 'blend' and kept['beat_seconds'] == 0.5 and 'beat_rejection' not in kept
    assert (kept['mix_at'], kept['in_at'], kept['mix_seconds'], kept['resume_at']) == (108.0, 8.0, 16.0, 24.0)
    assert kept['voice']['clash'] is False and kept['voice']['end_margin'] == 16.0


def test_a_locked_percussion_mix_with_competing_voices_becomes_a_beat_mix(monkeypatch):
    _inspect_pair(monkeypatch)
    assert voice.clear_voice('out', 'in', dict(PLAN, style='percussion_mix'))['style'] == 'blend'
    _inspect_pair(monkeypatch, out_spans=[], in_spans=[])
    assert voice.clear_voice('out', 'in', dict(PLAN, style='percussion_mix'))['style'] == 'percussion_mix'


def test_the_ending_song_is_read_only_a_minute_past_the_planned_end(monkeypatch):
    _inspect_pair(monkeypatch, duration=3600.0)
    heard = voice._heard
    asked = []

    def recorded(path, start, seconds):
        asked.append((path, start, seconds))
        return heard(path, start, seconds)

    monkeypatch.setattr(voice, '_heard', recorded)
    voice.clear_voice('out', 'in', PLAN)
    (_path, start, seconds), = [call for call in asked if call[0] == 'out']
    assert seconds is not None and start + seconds == PLAN['mix_at'] + PLAN['mix_seconds'] + voice.TAIL_READ_SECONDS


def test_a_short_incoming_track_cannot_overrun_when_a_tempo_lock_is_dropped(monkeypatch):
    def heard(path, start, seconds):
        return _inspection(start, 140.0 - start if path == 'out' else 10.0, [])

    monkeypatch.setattr(voice, '_heard', heard)
    moved = voice.clear_voice('out', 'in', dict(PLAN, resume_at=26.0))
    assert moved['style'] == 'fade' and moved['beat_seconds'] is None
    assert moved['in_at'] + moved['mix_seconds'] <= 10.0001
    assert moved['resume_at'] <= 10.0001


def test_a_percussion_mix_alternative_is_taken_only_where_no_voices_overlap(monkeypatch):
    drums = voice.mix._handoff('percussion_mix', 104.0, 6.0, 10.0, 0.98, 0.5)
    plan = dict(PLAN, style='fade', beat_seconds=None, mix_seconds=10.0, resume_at=18.0, swap_at=105.0,
                confidence='medium', beat_rejection='beat_tracking', alternatives=[drums])
    _inspect_pair(monkeypatch, out_spans=[(60.0, 100.0)], in_spans=[(20.0, 40.0)])
    chosen = voice.clear_voice('out', 'in', plan)
    assert chosen['style'] == 'percussion_mix' and chosen['selection_reason'] == 'percussion_mix'
    assert chosen['confidence'] == 'high' and chosen['mix_seconds'] == 10.0 and chosen['beat_seconds'] == 0.5
    _inspect_pair(monkeypatch, out_spans=[(60.0, 120.0)], in_spans=[(0.0, 40.0)])
    assert voice.clear_voice('out', 'in', plan)['style'] == 'fade'


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
    assert a / 2.0 == pytest.approx(round(a / 2.0))
    assert b / 1.9 == pytest.approx(round(b / 1.9), abs=1e-4)
    assert 3.0 <= a + b <= 4.0001
    assert moved['resume_at'] - moved['in_at'] == pytest.approx(15.2)


def test_recovery_and_later_ending_move_a_beat_mix_only_by_whole_bars(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_RECOVER_SECONDS', 3.0, raising=False)
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', 3.0, raising=False)
    plan = dict(PLAN, requested_end=125.0, voice={'clash': False, 'overlap': 0.0, 'end_margin': 24.0})
    ending = _inspection(0.0, 140.0, [])
    starting = _inspection(0.0, 40.0, [])
    moved = voice._recover(ending, starting, plan)
    for delta in (moved['mix_at'] - plan['mix_at'], plan['in_at'] - moved['in_at']):
        assert delta % 2.0 == pytest.approx(0.0, abs=1e-6)
    later = voice._later_ending(ending, starting, plan)
    assert (later['mix_at'] - plan['mix_at']) % 2.0 == pytest.approx(0.0, abs=1e-6)
    assert later['mix_at'] == 102.0


def test_a_half_time_next_song_moves_its_cue_by_its_own_bar():
    plan = dict(PLAN, entry_bar=4.0)
    assert voice._steps(plan) == (2.0, 4.0)
    assert voice._steps(dict(PLAN, entry_bar=2.0)) == (2.0, 2.0)
    assert voice._steps(dict(PLAN, entry_bar=1.0)) == (2.0, 2.0)


def test_final_cues_off_the_whole_bar_grid_are_caught():
    plan = dict(PLAN, entry_bar=2.0)
    assert voice._whole_bars(plan, dict(plan, mix_at=104.0, in_at=6.0))
    assert not voice._whole_bars(plan, dict(plan, mix_at=101.0))
    assert not voice._whole_bars(plan, dict(plan, in_at=7.0))
    assert voice._whole_bars(plan, dict(plan, mix_at=101.0, style='fade'))


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


def test_later_ending_recovers_reserved_tail_without_changing_the_approved_blend(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', 4.0, raising=False)
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_MARGIN_SECONDS', 10.0)
    plan = dict(PLAN, requested_end=125.0, voice={'clash': True, 'overlap': 1.0, 'end_margin': 10.0})
    ending = _inspection(0.0, 126.0, [(0.0, 126.0)])
    starting = _inspection(0.0, 40.0, [(0.0, 40.0)])
    moved = voice._later_ending(ending, starting, plan)
    assert moved['mix_at'] == 104.0 and moved['swap_at'] == 112.0
    assert moved['voice']['end_margin'] == 6.0
    assert moved['voice']['ending_recovered_seconds'] == 4.0
    for key in ('in_at', 'resume_at', 'mix_seconds', 'style', 'stretch', 'beat_seconds'):
        assert moved[key] == plan[key]
    assert moved['voice']['clash'] is True and moved['voice']['overlap'] == 1.0
    assert plan['mix_at'] == 100.0


def test_later_ending_stops_at_the_original_requested_endpoint_on_the_beat_grid(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', 4.0, raising=False)
    plan = dict(PLAN, requested_end=117.3, voice={'clash': False, 'overlap': 0.0, 'end_margin': 24.0})
    moved = voice._later_ending(_inspection(0.0, 140.0, []), _inspection(0.0, 40.0, []), plan)
    assert moved['mix_at'] == 100.0
    assert moved['mix_at'] + moved['mix_seconds'] <= plan['requested_end']
    moved = voice._later_ending(_inspection(0.0, 140.0, []), _inspection(0.0, 40.0, []), dict(plan, requested_end=118.3))
    assert moved['mix_at'] == 102.0


def test_later_ending_preserves_a_small_audible_tail_even_with_a_large_budget(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', 4.0, raising=False)
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_MARGIN_SECONDS', 2.0)
    plan = dict(PLAN, requested_end=118.0, voice={'clash': False, 'overlap': 0.0, 'end_margin': 2.0})
    moved = voice._later_ending(_inspection(0.0, 118.0, []), _inspection(0.0, 40.0, []), plan)
    assert moved['mix_at'] == 100.0 and moved['voice']['end_margin'] == 2.0


def test_later_ending_rejects_a_fading_window(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', 4.0, raising=False)
    plan = dict(PLAN, requested_end=125.0, voice={'clash': False, 'overlap': 0.0, 'end_margin': 24.0})
    grid = np.arange(1400) * voice.mix.STEP_SECONDS
    ending = _inspection(0.0, 140.0, [], -12.0 * np.maximum(0.0, grid - 100.0))
    moved = voice._later_ending(ending, _inspection(0.0, 40.0, []), plan)
    assert moved['mix_at'] == 100.0


def test_later_ending_cannot_trade_early_vocals_for_a_lingering_late_singer(monkeypatch):
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', 4.0, raising=False)
    plan = dict(PLAN, requested_end=125.0, voice={'clash': True, 'overlap': 0.5, 'end_margin': 24.0})
    ending = _inspection(0.0, 140.0, [(100.0, 108.0), (116.0, 140.0)])
    moved = voice._later_ending(ending, _inspection(0.0, 40.0, [(0.0, 40.0)]), plan)
    assert 0.0 <= moved['mix_at'] - plan['mix_at'] <= 2.0
    assert moved['voice']['overlap'] == plan['voice']['overlap']


@pytest.mark.parametrize('budget,spans,target', [(0.0, [], 125.0), (4.0, None, 125.0), (4.0, [], 115.0)])
def test_later_ending_does_not_force_movement_without_a_valid_reason(monkeypatch, budget, spans, target):
    monkeypatch.setattr(voice.config, 'SMART_FADE_END_RECOVER_SECONDS', budget, raising=False)
    plan = dict(PLAN, requested_end=target, voice={'clash': False, 'overlap': 0.0, 'end_margin': 24.0})
    assert voice._later_ending(_inspection(0.0, 140.0, spans), _inspection(0.0, 40.0, []), plan) is plan


def test_requested_ending_is_used_without_another_detector_pass(monkeypatch):
    _inspect_pair(monkeypatch)
    original = voice._heard
    calls = []

    def heard(path, start, seconds):
        calls.append(path)
        return original(path, start, seconds)

    monkeypatch.setattr(voice, '_heard', heard)
    moved = voice.clear_voice('out', 'in', dict(PLAN, requested_end=130.0))
    assert moved['mix_at'] + moved['mix_seconds'] == 122.0
    assert moved['in_at'] == 6.0 and moved['mix_seconds'] == 14.0
    assert moved['voice']['ending_recovered_seconds'] == 4.0
    assert sorted(calls) == ['in', 'out']


def test_sung_patches_and_silero_form_a_union_and_one_failed_detector_is_tolerated(monkeypatch):
    starts = np.arange(10) * 0.5
    scores = np.array([0.05, 0.2, 0.05, 0.2, 0.3, 0.25, 0.05, 0.05, 0.2, 0.2])
    expected = [(1.5, 2.5 + voice.PATCH_SECONDS), (4.0, 4.5 + voice.PATCH_SECONDS)]
    assert voice._patch_spans(starts, scores) == expected
    monkeypatch.setattr(voice, 'sung_spans', lambda audio: expected)
    monkeypatch.setattr(voice, 'spoken_spans', lambda audio: [(0.2, 0.9)])
    assert voice.voiced_spans(np.zeros(1))[0] == (0.2, 0.9)

    def broken(audio):
        raise RuntimeError('model missing')

    monkeypatch.setattr(voice, 'sung_spans', broken)
    assert voice.voiced_spans(np.zeros(1)) == [(0.2, 0.9)]
    monkeypatch.setattr(voice, 'spoken_spans', broken)
    with pytest.raises(RuntimeError):
        voice.voiced_spans(np.zeros(1))


def test_musicnn_and_silero_hear_one_read_at_once(monkeypatch):
    ready = threading.Barrier(2)

    def detector(spans):
        def heard(audio):
            ready.wait(timeout=3)
            return spans
        return heard

    monkeypatch.setattr(voice, 'sung_spans', detector([(1.0, 2.0)]))
    monkeypatch.setattr(voice, 'spoken_spans', detector([(0.5, 0.8)]))
    assert voice.voiced_spans(np.zeros(voice.VAD_RATE, dtype=np.float32)) == [(0.5, 0.8), (1.0, 2.0)]


def test_musicnn_runs_a_few_patches_a_call_and_keeps_their_order(monkeypatch):
    sizes = []

    def embed(names, feed):
        batch = next(iter(feed.values()))
        sizes.append(len(batch))
        return [batch[:, 0, :1]]

    prediction = SimpleNamespace(run=lambda names, feed: [next(iter(feed.values()))])
    monkeypatch.setattr(voice, '_musicnn', lambda: (SimpleNamespace(run=embed), prediction, [0]))
    patches = np.arange(7, dtype=np.float32)[:, None, None] * np.ones((1, voice.PATCH_FRAMES, 96), dtype=np.float32)
    scores = voice._predict(patches)
    assert max(sizes) <= voice.PREDICT_PATCHES and sum(sizes) == 7
    assert scores == pytest.approx(1.0 / (1.0 + np.exp(-np.arange(7.0))))


def test_adaptive_patch_skipping_preserves_the_full_resolution_mask():
    rng = np.random.default_rng(42)
    for count in (1, 2, 5, 7, 19, 60, 121):
        starts = np.arange(count) * voice.PATCH_HOP_FRAMES * voice.MEL_HOP / voice.VAD_RATE
        if count > 1:
            starts[-1] -= 0.1
        edges = np.r_[starts, starts + voice.PATCH_SECONDS]
        points = np.unique(np.r_[edges, np.nextafter(edges, -np.inf), np.nextafter(edges, np.inf)])
        for _ in range(40):
            positive = rng.random(count) < rng.random()
            accepted = positive & (np.r_[False, positive[:-1]] | np.r_[positive[1:], False])
            accepted[0], accepted[-1] = positive[0], positive[-1]
            expected = np.zeros(points.size, dtype=bool)
            for start in starts[accepted]:
                expected |= (points >= start) & (points < start + voice.PATCH_SECONDS)
            evaluated = []

            def predict(indices, evaluated=evaluated, positive=positive):
                evaluated.extend(indices)
                return np.where(positive[indices], 0.3, 0.05)

            spans = voice._sung_patches(starts, predict)
            actual = voice._activity({'at': 0.0, 'spans': spans}, points)
            assert np.array_equal(actual, expected)
            assert len(evaluated) == len(set(evaluated)) <= count


@pytest.mark.parametrize('score', [0.05, 0.3])
def test_adaptive_patches_skip_only_proven_covered_windows(score):
    starts = np.arange(120) * voice.PATCH_HOP_FRAMES * voice.MEL_HOP / voice.VAD_RATE
    evaluated = []

    def predict(indices):
        evaluated.extend(indices)
        return np.full(len(indices), score)

    spans = voice._sung_patches(starts, predict)
    if score < voice.VOCAL_SCORE:
        assert len(evaluated) == len(starts) and spans == []
    else:
        assert len(evaluated) < len(starts) / 2
        assert voice._activity({'at': 0.0, 'spans': spans}, np.linspace(0, starts[-1] + voice.PATCH_SECONDS - 1e-6, 1000)).all()


def test_warmup_initializes_detectors_and_the_beat_tracker_without_library_audio(monkeypatch):
    from tasks import paged_ivf

    calls = []
    monkeypatch.setattr(voice, 'voiced_spans', lambda audio: calls.append(('voice', len(audio))))
    monkeypatch.setattr(voice.mix, '_events', lambda audio: calls.append(('beats', len(audio))))
    monkeypatch.setattr(paged_ivf, '_pin_blas_single_thread', lambda: calls.append(('blas', 1)))
    monkeypatch.setattr(voice.mix, '_phrase_entry', lambda audio, grid, at: calls.append(('phrase', at)))
    voice.warmup()
    assert calls == [('voice', (voice.PATCH_FRAMES - 1) * voice.MEL_HOP + voice.FFT_SIZE), ('beats', 12 * voice.mix.RATE),
                     ('blas', 1), ('phrase', 4.0)]


def test_model_initialization_is_shared_and_keeps_the_existing_cpu_budget(monkeypatch):
    from tasks import onnx_utils

    ready = threading.Barrier(2)
    created = []
    monkeypatch.setattr(voice, '_sessions', None)
    monkeypatch.setattr(onnx_utils, 'usable_cpu_count', lambda: 2)
    monkeypatch.setattr(onnx_utils, 'resolve_providers', lambda **kw: [('CPUExecutionProvider', {})])

    def create(path, providers, label, sess_options):
        assert sess_options.intra_op_num_threads == 2
        assert sess_options.get_session_config_entry('session.intra_op.allow_spinning') == '0'
        assert sess_options.get_session_config_entry('session.inter_op.allow_spinning') == '0'
        created.append(path)
        return object()

    def initialize():
        ready.wait(timeout=3)
        return voice._musicnn()

    monkeypatch.setattr(onnx_utils, 'create_onnx_session', create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sessions = [future.result() for future in [pool.submit(initialize), pool.submit(initialize)]]
    assert len(created) == 2 and sessions[0] is sessions[1]


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
