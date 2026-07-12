# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Chromaprint catalogue ids plus legacy embedding-LSH compatibility.

Verifies the MusiCNN-embedding -> 64 seeded hyperplanes -> 64-bit SimHash-LSH ->
signed BIGINT -> fp_<hex> id pipeline: identical embeddings give identical ids,
near-identical embeddings stay within a small Hamming distance, different
embeddings diverge, and the id string round-trips the signed fingerprint.

Main Features:
* Determinism across calls and input forms (array, list, raw float32 bytes).
* Near-duplicate robustness and different-content separation via Hamming distance.
* Signed BIGINT range safety and fp_<hex> id round-trip.
"""

import numpy as np
from types import SimpleNamespace

from tasks import audio_fingerprint as afp


class TestChromaprintIdentity:
    def test_keeps_raw_value_and_hashes_deterministically(self, monkeypatch):
        raw = 'AQAAE0mUaEkSRZEGAA'
        monkeypatch.setitem(afp._fpcalc_state, 'available', True)
        monkeypatch.setattr(
            afp.subprocess,
            'run',
            lambda *args, **kwargs: SimpleNamespace(
                stdout='{"duration": 123.0, "fingerprint": "' + raw + '"}'
            ),
        )

        assert afp.compute_chromaprint('/tmp/song.flac') == raw
        first = afp.chromaprint_canonical_id(raw)
        assert first == afp.chromaprint_canonical_id(raw)
        assert first.startswith('fp_') and len(first) == 35
        assert first != afp.chromaprint_canonical_id(raw + 'different')

    def test_missing_fpcalc_returns_none(self, monkeypatch):
        monkeypatch.setitem(afp._fpcalc_state, 'available', True)
        def missing(*args, **kwargs):
            raise FileNotFoundError('fpcalc')

        monkeypatch.setattr(afp.subprocess, 'run', missing)
        assert afp.compute_chromaprint('/tmp/song.flac') is None


def _hamming(a, b):
    return bin(afp.from_signed_bigint(a) ^ afp.from_signed_bigint(b)).count('1')


def _embedding(seed, dim=200):
    rng = np.random.RandomState(seed)
    return rng.standard_normal(dim).astype(np.float32)


class TestEmbeddingFingerprint:
    def test_deterministic(self):
        emb = _embedding(1)
        assert afp.embedding_fingerprint(emb) == afp.embedding_fingerprint(emb.copy())

    def test_bytes_and_array_forms_agree(self):
        emb = _embedding(2)
        assert afp.embedding_fingerprint(emb) == afp.embedding_fingerprint(emb.tobytes())
        assert afp.embedding_fingerprint(emb) == afp.embedding_fingerprint(list(emb))

    def test_near_identical_embeddings_stay_close(self):
        emb = _embedding(3)
        noisy = emb + np.float32(1e-5) * _embedding(99)
        assert _hamming(afp.embedding_fingerprint(emb), afp.embedding_fingerprint(noisy)) <= 3

    def test_different_embeddings_diverge(self):
        a = afp.embedding_fingerprint(_embedding(4))
        b = afp.embedding_fingerprint(_embedding(5))
        assert _hamming(a, b) > 10

    def test_invalid_inputs_return_none(self):
        assert afp.embedding_fingerprint(None) is None
        assert afp.embedding_fingerprint([]) is None
        assert afp.embedding_fingerprint(np.zeros(200, dtype=np.float32)) is None
        bad = np.full(200, np.nan, dtype=np.float32)
        assert afp.embedding_fingerprint(bad) is None

    def test_fits_signed_bigint(self):
        for seed in range(10):
            value = afp.embedding_fingerprint(_embedding(seed))
            assert -(1 << 63) <= value <= (1 << 63) - 1


class TestBigintRange:
    def test_round_trip(self):
        for u in [0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]:
            signed = afp.to_signed_bigint(u)
            assert -(1 << 63) <= signed <= (1 << 63) - 1
            assert afp.from_signed_bigint(signed) == u


class TestCanonicalIdStr:
    def test_id_encodes_and_recovers_signed_fingerprint(self):
        for signed in [0, 1, -5, (1 << 63) - 1, -(1 << 63)]:
            item_id = afp.canonical_id_str(signed)
            assert afp.fingerprint_from_canonical_id(item_id) == signed

    def test_round_trip_and_prefix(self):
        for signed in [0, 1, -5, (1 << 63) - 1, -(1 << 63)]:
            cid = afp.canonical_id_str(signed)
            assert cid.startswith('fp_')
            assert afp.is_fingerprint_id(cid)
        assert afp.canonical_id_str(None) is None

    def test_same_fingerprint_same_id(self):
        assert afp.canonical_id_str(123456789) == afp.canonical_id_str(123456789)

    def test_embedding_canonical_id(self):
        emb = _embedding(7)
        cid = afp.embedding_canonical_id(emb)
        assert afp.is_fingerprint_id(cid)
        assert afp.fingerprint_from_canonical_id(cid) == afp.embedding_fingerprint(emb)

    def test_non_fingerprint_ids_rejected(self):
        assert not afp.is_fingerprint_id('abc123')
        assert not afp.is_fingerprint_id(None)
        assert not afp.is_fingerprint_id(42)
