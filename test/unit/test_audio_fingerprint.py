# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Embedding-SimHash catalogue-id behaviour.

Verifies that the MusiCNN-embedding fingerprint is deterministic and
similarity-preserving (near-identical embeddings differ by few bits, distinct
ones by many), that the fp_<16hex> id round-trips back to its hash, and that
the Hamming-tolerant FingerprintIndex matches near hashes but never distant
ones.

Main Features:
* Deterministic SimHash from arrays and raw float32 bytes.
* Similarity preservation and id round-trip, retired-scheme ids rejected.
* FingerprintIndex tolerance window (0..3 bits in, 4+ bits out).
"""

import numpy as np

from tasks import audio_fingerprint as afp


def _embedding(seed, dim=200):
    rng = np.random.RandomState(seed)
    return rng.standard_normal(dim).astype(np.float32)


class TestEmbeddingFingerprint:
    def test_deterministic_across_input_forms(self):
        emb = _embedding(1)
        from_array = afp.embedding_fingerprint(emb)
        from_bytes = afp.embedding_fingerprint(emb.tobytes())
        from_list = afp.embedding_fingerprint(list(emb))
        assert from_array == from_bytes == from_list
        assert isinstance(from_array, int)

    def test_similar_embeddings_land_within_tolerance(self):
        emb = _embedding(2)
        nudged = emb + np.float32(1e-4) * _embedding(3)
        a = afp.embedding_fingerprint(emb)
        b = afp.embedding_fingerprint(nudged)
        assert afp.hamming_distance(a, b) <= afp.FINGERPRINT_MATCH_MAX_HAMMING

    def test_distinct_embeddings_land_far_apart(self):
        a = afp.embedding_fingerprint(_embedding(4))
        b = afp.embedding_fingerprint(_embedding(5))
        assert afp.hamming_distance(a, b) > afp.FINGERPRINT_MATCH_MAX_HAMMING

    def test_invalid_embeddings_have_no_fingerprint(self):
        assert afp.embedding_fingerprint(None) is None
        assert afp.embedding_fingerprint([]) is None
        assert afp.embedding_fingerprint(np.zeros(200, dtype=np.float32)) is None
        assert afp.embedding_canonical_id(None) is None


class TestCanonicalIdRoundTrip:
    def test_id_encodes_and_recovers_the_hash(self):
        fingerprint = afp.embedding_fingerprint(_embedding(6))
        cid = afp.canonical_id_str(fingerprint)
        assert cid.startswith('fp_') and len(cid) == 19
        assert afp.is_fingerprint_id(cid)
        assert afp.fingerprint_from_canonical_id(cid) == fingerprint

    def test_retired_scheme_and_provider_ids_do_not_decode(self):
        assert afp.fingerprint_from_canonical_id('fp_' + 'a' * 32) is None
        assert afp.fingerprint_from_canonical_id('jellyfin-track-1') is None
        assert afp.fingerprint_from_canonical_id(None) is None
        assert not afp.is_fingerprint_id('plain-id')
        assert not afp.is_fingerprint_id(None)

    def test_signed_bigint_round_trip(self):
        high_bit = afp.to_signed_bigint((1 << 63) | 5)
        assert high_bit < 0
        assert afp.from_signed_bigint(high_bit) == (1 << 63) | 5


class TestFingerprintIndex:
    def test_finds_exact_and_near_hashes(self):
        base = afp.embedding_fingerprint(_embedding(7))
        index = afp.FingerprintIndex()
        index.add('fp_base', base)
        assert index.find(base) == 'fp_base'
        flipped = afp.to_signed_bigint(afp.from_signed_bigint(base) ^ 0b101)
        assert afp.hamming_distance(base, flipped) == 2
        assert index.find(flipped) == 'fp_base'

    def test_rejects_hashes_beyond_tolerance(self):
        base = afp.embedding_fingerprint(_embedding(8))
        index = afp.FingerprintIndex()
        index.add('fp_base', base)
        far = afp.to_signed_bigint(afp.from_signed_bigint(base) ^ 0b11110000)
        assert afp.hamming_distance(base, far) == 4
        assert index.find(far) is None

    def test_from_item_ids_skips_undecodable_ids(self):
        fingerprint = afp.embedding_fingerprint(_embedding(9))
        cid = afp.canonical_id_str(fingerprint)
        index = afp.FingerprintIndex.from_item_ids(
            [cid, 'provider-1', 'fp_' + 'a' * 32]
        )
        assert index.find(fingerprint) == cid

    def test_prefers_the_closest_match(self):
        base = afp.from_signed_bigint(afp.embedding_fingerprint(_embedding(10)))
        near = afp.to_signed_bigint(base ^ 0b1)
        exact = afp.to_signed_bigint(base)
        index = afp.FingerprintIndex()
        index.add('one-bit', near)
        index.add('exact', exact)
        assert index.find(afp.to_signed_bigint(base)) == 'exact'

    def test_find_similar_fingerprint_helper(self):
        base = afp.embedding_fingerprint(_embedding(11))
        near = afp.to_signed_bigint(afp.from_signed_bigint(base) ^ 0b11)
        assert afp.find_similar_fingerprint(base, [near]) == near
        far = afp.to_signed_bigint(afp.from_signed_bigint(base) ^ 0b1111100000)
        assert afp.find_similar_fingerprint(base, [far]) is None
