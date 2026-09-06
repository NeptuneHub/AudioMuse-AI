# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Chromaprint identification over the stored fingerprints.

Main Features:
* split_planes keeps every bit of a fingerprint across the two uint16 planes
* the weighted lookup tables reproduce an explicit per-bit weighted popcount,
  and a strided score uses only every n-th query frame
* a clip taken from one track and corrupted by random bit flips is found at
  rank one, at its true offset, and flagged identified, in a synthetic library
* the strided first pass keeps the true track in the candidate pool, the exact
  pass runs only on the pool plus the null sample, and each variant re-scores
  only the top candidates plus the same null sample
* a clip of a track that is not in the library flags nothing
* a clip shorter than RECORDING_SEARCH_IDENTIFY_MIN_SECONDS is refused before
  any scan, a clip just above it is accepted although its fingerprint is 2.6 s
  shorter than the audio, and a missing fpcalc is a RuntimeError
* the clip is fingerprinted once per sub-hop phase and once per valid
  configured playback speed, ignoring malformed entries and 1.0, and the one
  variant whose best candidate is most extreme is used for every candidate
* the thread count honours the config override and stays bounded otherwise
"""

import numpy as np
import pytest

from tasks import chromaprint_identify as ci


def test_split_planes_keeps_every_bit():
    values = np.array([0, 1, 0xFFFF, 0x10000, 0xDEADBEEF, 0xFFFFFFFF], dtype=np.uint32)
    lo, hi = ci.split_planes(values)
    assert lo.dtype == np.uint16 and hi.dtype == np.uint16
    back = lo.astype(np.uint32) | (hi.astype(np.uint32) << np.uint32(16))
    assert np.array_equal(back, values)


def test_weight_luts_reproduce_an_explicit_weighted_popcount():
    luts = ci.weight_luts()
    rng = np.random.default_rng(3)
    for value in rng.integers(0, 65536, size=50):
        explicit_lo = sum(ci._BIT_WEIGHTS[b] for b in range(16) if (int(value) >> b) & 1)
        explicit_hi = sum(ci._BIT_WEIGHTS[16 + b] for b in range(16) if (int(value) >> b) & 1)
        assert luts['lo'][value] == pytest.approx(explicit_lo * ci._LUT_SCALE, abs=0.5)
        assert luts['hi'][value] == pytest.approx(explicit_hi * ci._LUT_SCALE, abs=0.5)
    assert luts['total'] == pytest.approx(float(ci._BIT_WEIGHTS.sum()) * ci._LUT_SCALE)


def test_strided_score_uses_every_nth_query_frame_and_stays_a_rate():
    rng = np.random.default_rng(5)
    track = rng.integers(0, 2 ** 32, size=400, dtype=np.uint64).astype(np.uint32)
    query = track[100:301].copy()
    lo, hi = ci.split_planes(track)
    q_lo, q_hi = ci.split_planes(query)
    exact, off = ci.score_rows(lo[None, :], hi[None, :], q_lo, q_hi)
    strided, off4 = ci.score_rows(lo[None, :], hi[None, :], q_lo, q_hi, stride=4)
    assert exact[0] == 0.0 and off[0] == 100
    assert strided[0] == 0.0 and off4[0] == 100
    other = rng.integers(0, 2 ** 32, size=201, dtype=np.uint64).astype(np.uint32)
    o_lo, o_hi = ci.split_planes(other)
    unrelated, _ = ci.score_rows(lo[None, :], hi[None, :], o_lo, o_hi, stride=4)
    assert 0.3 < unrelated[0] < 0.5


def _synthetic_library(monkeypatch, n_tracks=700, length=948, seed=7):
    rng = np.random.default_rng(seed)
    fingerprints = rng.integers(0, 2 ** 32, size=(n_tracks, length), dtype=np.uint64).astype(np.uint32)
    lo, hi = ci.split_planes(fingerprints.reshape(-1))
    lengths = np.full(n_tracks, length, dtype=np.int64)
    starts = np.arange(n_tracks, dtype=np.int64) * length
    ids = np.array([f'fp_{i:05d}' for i in range(n_tracks)])
    for name, value in (('lo', lo), ('hi', hi), ('starts', starts), ('lengths', lengths), ('ids', ids)):
        monkeypatch.setitem(ci._STATE, name, value)
    monkeypatch.setattr(ci, 'ensure_loaded', lambda: True)
    return fingerprints, rng


def _corrupt(segment, rng, flip_rate):
    mask = np.zeros(segment.size, dtype=np.uint32)
    for bit in range(32):
        flips = rng.random(segment.size) < flip_rate
        mask |= flips.astype(np.uint32) << np.uint32(bit)
    return segment ^ mask


def _fake_fpcalc(monkeypatch, query):
    import tasks.chromaprint as cp

    monkeypatch.setattr(cp, 'is_available', lambda: True)
    monkeypatch.setattr(cp, 'fingerprint_audio', lambda audio, sr: query)


def test_corrupted_clip_from_one_track_is_found_at_rank_one_with_its_offset(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    true_track, start = 42, 80
    query = _corrupt(fingerprints[true_track, start:start + 201], rng, 0.25)
    _fake_fpcalc(monkeypatch, query)
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 10)
    assert rows[0]['item_id'] == 'fp_00042'
    assert rows[0]['offset_seconds'] == pytest.approx(start * ci.HOP_SECONDS, abs=0.2)
    assert rows[0]['identified'] is True
    assert rows[0]['z'] > 6
    assert rows[0]['ber'] < rows[1]['ber']
    assert len(rows) == 10


def test_the_exact_pass_runs_only_on_the_pool_and_the_null_sample(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch, n_tracks=3000)
    query = _corrupt(fingerprints[10, 30:231], rng, 0.2)
    _fake_fpcalc(monkeypatch, query)
    calls = []
    original = ci._score_chunk

    def counting(idx, q_lo, q_hi, stride, positions):
        calls.append((stride, idx.size))
        return original(idx, q_lo, q_hi, stride, positions)

    monkeypatch.setattr(ci, '_score_chunk', counting)
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_SPEEDS', '')
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_RERANK', 100)
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert rows[0]['item_id'] == 'fp_00010'
    assert rows[0]['identified'] is True
    strided = sum(n for stride, n in calls if stride == ci._STAGE1_STRIDE)
    exact = sum(n for stride, n in calls if stride == 1)
    assert strided == 3000
    assert exact == ci._POOL_MIN + ci._NULL_SAMPLE + (ci.PHASES - 1) * (100 + ci._NULL_SAMPLE)


def test_frames_near_the_clip_majority_value_are_left_out_and_a_noise_clip_is_refused(monkeypatch):
    rng = np.random.default_rng(11)
    music = rng.integers(0, 2 ** 32, size=120, dtype=np.uint64).astype(np.uint32)
    noise = _corrupt(np.full(80, 0xF57DF57D, dtype=np.uint32), rng, 0.05)
    query = np.concatenate([noise[:40], music, noise[40:]])
    kept = ci.informative_positions(query)
    assert 100 <= kept.size <= 120
    assert kept.min() >= 30 and kept.max() < 170
    assert ci.informative_positions(noise).size < 8
    _synthetic_library(monkeypatch, n_tracks=20)
    _fake_fpcalc(monkeypatch, noise)
    monkeypatch.setattr(ci, '_scan', lambda *a, **k: pytest.fail('scan must not run'))
    with pytest.raises(ValueError, match='too noisy'):
        ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)


def test_noise_frames_around_a_true_segment_do_not_stop_the_track_from_being_found(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    segment = _corrupt(fingerprints[21, 50:171], rng, 0.2)
    noise = _corrupt(np.full(80, 0xF57DF57D, dtype=np.uint32), rng, 0.05)
    query = np.concatenate([noise[:40], segment, noise[40:]])
    _fake_fpcalc(monkeypatch, query)
    seen = []
    original = ci.score_rows

    def recording(lo, hi, q_lo, q_hi, stride=1, positions=None):
        seen.append(None if positions is None else int(np.asarray(positions).size))
        return original(lo, hi, q_lo, q_hi, stride, positions)

    monkeypatch.setattr(ci, 'score_rows', recording)
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_SPEEDS', '')
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert rows[0]['item_id'] == 'fp_00021'
    assert rows[0]['offset_seconds'] == pytest.approx(10 * ci.HOP_SECONDS, abs=0.2)
    assert seen and all(n is not None and 100 <= n <= 121 for n in seen)


def test_a_duplicate_of_the_best_track_shares_the_flag_and_the_lead_skips_it(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    fingerprints[300] = fingerprints[42]
    lo, hi = ci.split_planes(fingerprints.reshape(-1))
    monkeypatch.setitem(ci._STATE, 'lo', lo)
    monkeypatch.setitem(ci._STATE, 'hi', hi)
    query = _corrupt(fingerprints[42, 80:281], rng, 0.2)
    _fake_fpcalc(monkeypatch, query)
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_SPEEDS', '')
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert {rows[0]['item_id'], rows[1]['item_id']} == {'fp_00042', 'fp_00300'}
    assert rows[0]['identified'] is True and rows[1]['identified'] is True
    assert rows[0]['lead'] == rows[1]['lead'] and rows[0]['lead'] > 5
    assert rows[2]['identified'] is False and rows[2]['lead'] is None


def test_a_close_runner_up_that_is_a_different_recording_withholds_the_flag(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    query = _corrupt(fingerprints[42, 80:281], rng, 0.3)
    fingerprints[301, 80:281] = _corrupt(query, rng, 0.3)
    lo, hi = ci.split_planes(fingerprints.reshape(-1))
    monkeypatch.setitem(ci._STATE, 'lo', lo)
    monkeypatch.setitem(ci._STATE, 'hi', hi)
    _fake_fpcalc(monkeypatch, query)
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_SPEEDS', '')
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_LEAD', 3.0)
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert {rows[0]['item_id'], rows[1]['item_id']} == {'fp_00042', 'fp_00301'}
    assert rows[0]['z'] > ci.expected_extreme_z(700) + 1.0
    assert 0 < rows[0]['lead'] < 3.0
    assert not any(row['identified'] for row in rows)
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_LEAD', 0.0)
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert rows[0]['identified'] is True and rows[1]['identified'] is False


def test_a_clip_of_a_track_not_in_the_library_flags_nothing(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    query = rng.integers(0, 2 ** 32, size=201, dtype=np.uint64).astype(np.uint32)
    _fake_fpcalc(monkeypatch, query)
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert len(rows) == 5
    assert not any(row['identified'] for row in rows), rows
    assert all(row['z'] < ci.expected_extreme_z(700) + 1.0 for row in rows), rows


def test_a_clip_too_short_to_identify_is_refused_before_scanning(monkeypatch):
    _synthetic_library(monkeypatch, n_tracks=20)
    _fake_fpcalc(monkeypatch, np.zeros(10, dtype=np.uint32))
    monkeypatch.setattr(ci, '_scan', lambda *a, **k: pytest.fail('scan must not run'))
    with pytest.raises(ValueError, match='too short'):
        ci.identify(np.zeros(48000, dtype=np.float32), 48000, 5)


def test_a_nine_second_clip_is_accepted_despite_the_window_the_fingerprint_loses(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    query = _corrupt(fingerprints[5, 40:92], rng, 0.2)
    _fake_fpcalc(monkeypatch, query)
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_MIN_SECONDS', 8.0)
    rows = ci.identify(np.zeros(48000 * 9, dtype=np.float32), 48000, 5)
    assert rows[0]['item_id'] == 'fp_00005'


def test_every_phase_and_every_configured_speed_is_fingerprinted_once(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    query = _corrupt(fingerprints[7, 60:261], rng, 0.2)
    calls = []

    import tasks.chromaprint as cp

    monkeypatch.setattr(cp, 'is_available', lambda: True)

    def recording(audio, sr):
        calls.append(np.asarray(audio).shape[0])
        return query

    monkeypatch.setattr(cp, 'fingerprint_audio', recording)
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_SPEEDS', '0.98, bogus, 1.02, 1.0')
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert rows[0]['item_id'] == 'fp_00007'
    assert rows[0]['variant'] == 'as recorded'
    assert len(calls) == 1 + (ci.PHASES - 1) + 2
    assert min(calls) < 48000 * 25 < max(calls)


def test_the_variant_whose_best_candidate_is_most_extreme_is_chosen_for_everybody(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    clean = fingerprints[9, 100:301].copy()
    noisy = _corrupt(clean, rng, 0.45)
    answers = [noisy, noisy, clean, noisy, noisy, noisy]

    import tasks.chromaprint as cp

    monkeypatch.setattr(cp, 'is_available', lambda: True)
    monkeypatch.setattr(cp, 'fingerprint_audio', lambda audio, sr: answers.pop(0))
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_SPEEDS', '0.99,1.01')
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert rows[0]['item_id'] == 'fp_00009'
    assert rows[0]['variant'] == 'phase 2/4'
    assert rows[0]['identified'] is True
    assert rows[0]['z'] > rows[1]['z'] + 5


def test_a_winning_speed_is_refined_over_its_phases_and_named_with_both(monkeypatch):
    fingerprints, rng = _synthetic_library(monkeypatch)
    clean = fingerprints[11, 100:301].copy()
    noisy = _corrupt(clean, rng, 0.45)
    better = _corrupt(clean, rng, 0.25)
    answers = [noisy, noisy, noisy, noisy, noisy, better, noisy, clean, noisy]
    lengths = []

    import tasks.chromaprint as cp

    monkeypatch.setattr(cp, 'is_available', lambda: True)

    def recording(audio, sr):
        lengths.append(np.asarray(audio).shape[0])
        return answers.pop(0)

    monkeypatch.setattr(cp, 'fingerprint_audio', recording)
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_SPEEDS', '0.99,1.01')
    rows = ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)
    assert rows[0]['item_id'] == 'fp_00011'
    assert rows[0]['variant'] == 'speed 1.01, phase 2/4'
    assert rows[0]['identified'] is True
    assert len(lengths) == 1 + (ci.PHASES - 1) + 2 + (ci.PHASES - 1)
    slow = lengths[5]
    assert slow > 48000 * 25
    assert lengths[6:] == [slow - int(round(p * ci.HOP_SECONDS / ci.PHASES * 48000)) for p in range(1, ci.PHASES)]


def test_missing_fpcalc_is_a_runtime_error(monkeypatch):
    import tasks.chromaprint as cp

    monkeypatch.setattr(cp, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='fpcalc'):
        ci.identify(np.zeros(48000 * 25, dtype=np.float32), 48000, 5)


def test_thread_count_honours_the_override_and_stays_bounded(monkeypatch):
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_THREADS', 7)
    assert ci._thread_count() == 7
    monkeypatch.setattr(ci.config, 'RECORDING_SEARCH_IDENTIFY_THREADS', 0)
    monkeypatch.setattr(ci, 'usable_cpu_count', lambda: 64)
    assert ci._thread_count() == ci._MAX_THREADS
    monkeypatch.setattr(ci, 'usable_cpu_count', lambda: None)
    assert 1 <= ci._thread_count() <= ci._MAX_THREADS


def test_expected_extreme_z_grows_with_the_scanned_population():
    assert ci.expected_extreme_z(1000) < ci.expected_extreme_z(200000)
    assert ci.expected_extreme_z(200000) == pytest.approx(4.94, abs=0.05)
