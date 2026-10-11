# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The P2P additions to the similarity hash (tasks.simhash).

A peer's offer is only taken for the same recording when every identity signal
agrees, and the node keeps its whole library in a compact numpy index that must
answer exactly like the in-memory SignatureIndex the catalogue resolver uses.

Main Features:
* signature_to_bytes / signature_from_bytes round trip 25 big-endian bytes and
  refuse anything else
* p2p_same_recording: a lightly perturbed copy with the same length passes; a
  cosine-close but Hamming-far pair, a same-signature different song, a length
  mismatch, an unknown length and a chromaprint disagreement (gate on) fail
* PackedSignatureIndex answers like SignatureIndex.find_candidates (same ids,
  same nearest-first order) on random libraries with near-duplicates inside
  and just outside Hamming 10, shared bands, a homogeneous same-signature
  cluster, with and without a query duration and a smaller tolerance
* the vectorised band keys equal the per-signature band keys
* rows with an unknown or non-positive duration or no signature are skipped;
  packed-matrix input and incremental extends keep the parity
* the resident size stays at about 121 bytes per row plus the ids
"""

import numpy as np
import pytest

from tasks import simhash


def _embedding(seed, dim=simhash.SIGNATURE_BITS):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _random_signature(rng):
    return int.from_bytes(rng.integers(0, 256, simhash.SIGNATURE_BYTES, dtype=np.uint8).tobytes(), 'big')


def _flip(rng, signature, bits):
    for bit in rng.choice(simhash.SIGNATURE_BITS, size=bits, replace=False):
        signature ^= 1 << int(bit)
    return signature


def _flip_outside_band(rng, signature, band, bits):
    low, high = simhash._BAND_BITS[band]
    allowed = [position for position in range(simhash.SIGNATURE_BITS) if not low <= position < high]
    for position in rng.choice(allowed, size=bits, replace=False):
        signature ^= 1 << (simhash.SIGNATURE_BITS - 1 - int(position))
    return signature


def _library(seed, size=1500):
    rng = np.random.default_rng(seed)
    ids, signatures, durations = [], [], []
    for index in range(size):
        ids.append(f'row{index}')
        signatures.append(_random_signature(rng))
        durations.append(float(rng.uniform(60, 400)))
    for index in range(size // 4):
        base = int(rng.integers(0, size))
        ids.append(f'near{index}')
        signatures.append(_flip(rng, signatures[base], int(rng.integers(0, 14))))
        durations.append(durations[base] + float(rng.uniform(-2.0, 2.0)))
    for index in range(size // 10):
        base = int(rng.integers(0, size))
        band = int(rng.integers(0, simhash._BAND_COUNT))
        ids.append(f'band{index}')
        signatures.append(_flip_outside_band(rng, signatures[base], band, int(rng.integers(8, 40))))
        durations.append(durations[base] + float(rng.uniform(-0.5, 0.5)))
    shared = _random_signature(rng)
    for index in range(40):
        ids.append(f'same{index}')
        signatures.append(shared)
        durations.append(100.0 + index * 0.6)
    return ids, signatures, durations


def _both(ids, signatures, durations, max_hamming=simhash.SIGNATURE_MATCH_MAX_HAMMING):
    reference = simhash.SignatureIndex(max_hamming=max_hamming)
    for item_id, signature, duration in zip(ids, signatures, durations):
        reference.add(item_id, signature, duration=duration)
    packed = simhash.PackedSignatureIndex(max_hamming=max_hamming)
    packed.extend(ids, signatures, durations)
    return reference, packed


def _probes(seed, signatures, durations):
    rng = np.random.default_rng(seed + 1000)
    probes = []
    for position in rng.choice(len(signatures), size=300, replace=False):
        probes.append((_flip(rng, signatures[position], int(rng.integers(0, 13))),
                       durations[position] + float(rng.uniform(-1.5, 1.5))))
    for _ in range(100):
        probes.append((_random_signature(rng), float(rng.uniform(60, 400))))
    probes.append((signatures[-1], 110.0))
    return probes


def _assert_parity(reference, packed, probes):
    matched = 0
    for signature, duration in probes:
        for query_duration in (duration, None):
            expected = reference.find_candidates(signature, duration=query_duration)
            found = packed.find(signature, query_duration, limit=None)
            assert [item_id for item_id, _ in found] == expected
            for item_id, hamming in found:
                assert 0 <= hamming <= reference._max_hamming
            matched += bool(expected)
            assert packed.find(signature, query_duration, limit=2) == found[:2]
    return matched


class TestSignatureBytes:
    def test_round_trip(self):
        signature = simhash.embedding_signature(_embedding(1))
        raw = simhash.signature_to_bytes(signature)
        assert isinstance(raw, bytes) and len(raw) == simhash.SIGNATURE_BYTES == 25
        assert simhash.signature_from_bytes(raw) == signature
        assert simhash.signature_from_bytes(bytearray(raw)) == signature
        assert simhash.signature_from_bytes(memoryview(raw)) == signature

    def test_matches_the_canonical_id_hex(self):
        signature = simhash.embedding_signature(_embedding(2))
        item_id = simhash.canonical_id_str(signature)
        assert simhash.signature_to_bytes(signature).hex() == item_id[len(simhash.CURRENT_ID_HEAD):]

    def test_extreme_values(self):
        assert simhash.signature_to_bytes(0) == bytes(25)
        top = (1 << simhash.SIGNATURE_BITS) - 1
        assert simhash.signature_to_bytes(top) == b'\xff' * 25
        assert simhash.signature_from_bytes(b'\xff' * 25) == top

    @pytest.mark.parametrize('raw', [None, b'', b'\x00' * 24, b'\x00' * 26, 'a' * 25, 12345])
    def test_anything_but_25_bytes_is_refused(self, raw):
        assert simhash.signature_from_bytes(raw) is None


class TestP2PSameRecording:
    def test_a_lightly_perturbed_copy_with_the_same_length_is_the_same_recording(self):
        base = _embedding(3)
        copy = base + 1e-3 * _embedding(4)
        signatures = simhash.embedding_signature(base), simhash.embedding_signature(copy)
        assert simhash.signature_hamming(*signatures) <= simhash.SIGNATURE_MATCH_MAX_HAMMING
        assert simhash.p2p_same_recording(base, copy, 200.0, 200.4)
        assert simhash.p2p_same_recording(base.tobytes(), copy.tobytes(), 200.0, 200.0)

    def test_cosine_close_but_hamming_far_is_not_the_same_recording(self):
        base = np.full(simhash.SIGNATURE_BITS, 1e-4, dtype=np.float32)
        base[1::2] = -1e-4
        base[:10] = [5.0, -5.0] * 5
        flipped = base.copy()
        flipped[10:] = -flipped[10:]
        hamming = simhash.signature_hamming(
            simhash.embedding_signature(base), simhash.embedding_signature(flipped)
        )
        assert hamming > simhash.SIGNATURE_MATCH_MAX_HAMMING
        assert simhash.cosine_distance(base, flipped) <= simhash.DUPLICATE_DISTANCE_THRESHOLD_COSINE
        assert not simhash.p2p_same_recording(base, flipped, 200.0, 200.0)

    def test_same_signature_but_a_different_song_is_not_the_same_recording(self):
        half = simhash.SIGNATURE_BITS // 2
        first = np.concatenate([np.full(half, 1.0), np.full(half, -1.0)]).astype(np.float32)
        second = first.copy()
        second[0:half:2] = 2.0
        second[1:half:2] = 0.1
        second[half::2] = -2.0
        second[half + 1::2] = -0.1
        assert simhash.embedding_signature(first) == simhash.embedding_signature(second)
        assert not simhash.p2p_same_recording(first, second, 200.0, 200.0)

    def test_a_length_mismatch_or_an_unknown_length_is_not_the_same_recording(self):
        base = _embedding(5)
        assert not simhash.p2p_same_recording(base, base, 200.0, 200.0 + simhash.DURATION_TOLERANCE_SECONDS + 0.5)
        assert not simhash.p2p_same_recording(base, base, None, 200.0)
        assert not simhash.p2p_same_recording(base, base, 200.0, None)
        assert not simhash.p2p_same_recording(base, base, 0.0, 0.0)
        assert not simhash.p2p_same_recording(base, base, float('nan'), 200.0)

    def test_an_unusable_embedding_is_not_the_same_recording(self):
        base = _embedding(6)
        assert not simhash.p2p_same_recording(base, np.ones(simhash.SIGNATURE_BITS), 200.0, 200.0)
        assert not simhash.p2p_same_recording(None, base, 200.0, 200.0)
        assert not simhash.p2p_same_recording(base, base[:-1], 200.0, 200.0)

    def test_the_chromaprint_gate(self, monkeypatch):
        base = _embedding(7)
        verdicts = {}
        monkeypatch.setattr(simhash, 'chromaprints_agree', lambda a, b: verdicts['value'])
        monkeypatch.setattr(simhash, 'CHROMAPRINT_GATE_ENABLED', True)
        verdicts['value'] = False
        assert not simhash.p2p_same_recording(base, base, 200.0, 200.0, 'fp-a', 'fp-b')
        verdicts['value'] = None
        assert simhash.p2p_same_recording(base, base, 200.0, 200.0, 'fp-a', None)
        verdicts['value'] = True
        assert simhash.p2p_same_recording(base, base, 200.0, 200.0, 'fp-a', 'fp-b')
        monkeypatch.setattr(simhash, 'CHROMAPRINT_GATE_ENABLED', False)
        verdicts['value'] = False
        assert simhash.p2p_same_recording(base, base, 200.0, 200.0, 'fp-a', 'fp-b')


class TestPackedBandKeys:
    def test_vectorised_band_keys_equal_the_per_signature_keys(self):
        rng = np.random.default_rng(11)
        signatures = [_random_signature(rng) for _ in range(500)] + [0, (1 << simhash.SIGNATURE_BITS) - 1]
        packed = np.stack([simhash._pack_signature(signature) for signature in signatures])
        keys = simhash._packed_band_keys(packed)
        for row, signature in enumerate(signatures):
            assert [int(key) for key in keys[row]] == [
                simhash._band_key(signature, band) for band in range(simhash._BAND_COUNT)
            ]


class TestPackedSignatureIndexParity:
    @pytest.mark.parametrize('seed', [1, 2, 3])
    def test_same_answers_as_the_signature_index(self, seed):
        ids, signatures, durations = _library(seed)
        reference, packed = _both(ids, signatures, durations)
        assert len(packed) == len(ids)
        matched = _assert_parity(reference, packed, _probes(seed, signatures, durations))
        assert matched > 200

    def test_same_answers_with_a_smaller_tolerance(self):
        ids, signatures, durations = _library(4, size=800)
        reference, packed = _both(ids, signatures, durations, max_hamming=3)
        _assert_parity(reference, packed, _probes(4, signatures, durations))

    def test_the_tolerance_is_capped_like_the_signature_index(self):
        assert simhash.PackedSignatureIndex(max_hamming=50)._max_hamming == simhash.SignatureIndex(
            max_hamming=50)._max_hamming == simhash.SIGNATURE_MATCH_MAX_HAMMING

    def test_hamming_is_reported_exactly_and_nearest_first(self):
        rng = np.random.default_rng(12)
        base = _random_signature(rng)
        ids = ['exact', 'two', 'seven', 'ten', 'eleven']
        signatures = [base, _flip(rng, base, 2), _flip(rng, base, 7), _flip(rng, base, 10), _flip(rng, base, 11)]
        packed = simhash.PackedSignatureIndex()
        packed.extend(ids[::-1], signatures[::-1], [200.0] * 5)
        assert packed.find(base, 200.0, limit=None) == [('exact', 0), ('two', 2), ('seven', 7), ('ten', 10)]
        assert packed.find(base, 200.0) == [('exact', 0), ('two', 2)]
        assert packed.find(simhash.signature_to_bytes(base), 200.0, limit=1) == [('exact', 0)]
        assert packed.find(base, 200.0, limit=0) == []

    def test_the_query_duration_filters_like_the_signature_index(self):
        packed = simhash.PackedSignatureIndex()
        packed.extend(['a', 'b'], [5, 5], [200.0, 200.0 + simhash.DURATION_TOLERANCE_SECONDS + 0.01])
        assert packed.find(5, 200.0, limit=None) == [('a', 0)]
        assert [item_id for item_id, _ in packed.find(5, None, limit=None)] == ['a', 'b']
        assert packed.find(5, float('nan'), limit=None) == []


class TestPackedSignatureIndexRows:
    def test_rows_without_a_usable_duration_or_signature_are_skipped(self):
        packed = simhash.PackedSignatureIndex()
        added = packed.extend(
            ['none', 'nan', 'zero', 'negative', 'text', 'nosig', 'good'],
            [1, 1, 1, 1, 1, None, 1],
            [None, float('nan'), 0.0, -3.0, 'long', 120.0, 120.0],
        )
        assert added == 1
        assert len(packed) == 1
        assert packed.find(1, 120.0, limit=None) == [('good', 0)]

    def test_empty_index_and_empty_extend(self):
        packed = simhash.PackedSignatureIndex()
        assert packed.find(1, 100.0) == []
        assert packed.find(None, 100.0) == []
        assert packed.extend([], [], []) == 0
        assert len(packed) == 0

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            simhash.PackedSignatureIndex().extend(['a', 'b'], [1], [100.0, 100.0])
        with pytest.raises(ValueError):
            simhash.PackedSignatureIndex().extend(['a'], [1], [100.0, 100.0])
        with pytest.raises(ValueError):
            simhash.PackedSignatureIndex().extend(['a'], np.zeros((1, 24), dtype=np.uint8), [100.0])

    def test_packed_matrix_input_equals_integer_input(self):
        ids, signatures, durations = _library(5, size=400)
        from_ints = simhash.PackedSignatureIndex()
        from_ints.extend(ids, signatures, durations)
        from_matrix = simhash.PackedSignatureIndex()
        from_matrix.extend(ids, np.stack([simhash._pack_signature(s) for s in signatures]), np.asarray(durations))
        for signature, duration in _probes(5, signatures, durations):
            assert from_ints.find(signature, duration, limit=None) == from_matrix.find(signature, duration, limit=None)

    def test_incremental_extends_equal_one_extend(self):
        ids, signatures, durations = _library(6, size=600)
        reference, _ = _both(ids, signatures, durations)
        packed = simhash.PackedSignatureIndex()
        for start in range(0, len(ids), 97):
            packed.extend(ids[start:start + 97], signatures[start:start + 97], durations[start:start + 97])
        assert len(packed) == len(ids)
        _assert_parity(reference, packed, _probes(6, signatures, durations))

    def test_resident_size_per_row(self):
        ids, signatures, durations = _library(8, size=4000)
        packed = simhash.PackedSignatureIndex()
        packed.extend(ids, signatures, durations)
        arrays = [packed._packed, packed._durations] + packed._band_keys + packed._band_rows
        per_row = sum(array.nbytes for array in arrays) / len(packed)
        assert per_row <= 25 + 8 + simhash._BAND_COUNT * 8
