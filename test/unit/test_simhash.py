# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Home-made similarity-hash identity behaviour.

Verifies the 200-bit per-dimension sign signature: deterministic across input
forms, similarity-preserving (a re-encode flips few bits, distinct songs flip
many), the fp_2<50hex> id round-trip, the banded candidate index, and the
resolver contract: identity is decided by the raw-embedding cosine (the Similar
Songs duplicate rule) with the signature only proposing candidates - two songs
sharing a signature but failing the cosine stay separate rows.

Main Features:
* Signature determinism, similarity window, invalid-input handling.
* fp_2 id round-trip and legacy-scheme rejection.
* Resolver: cosine confirms, collisions mint the next free id.
"""

import numpy as np
import pytest

from tasks import simhash


def _embedding(seed, dim=simhash.SIGNATURE_BITS):
    rng = np.random.RandomState(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _same_signature_different_song():
    half = simhash.SIGNATURE_BITS // 2
    first = np.concatenate([np.full(half, 1.0), np.full(half, -1.0)]).astype(np.float32)
    second = first.copy()
    second[0:half:2] = 2.0
    second[1:half:2] = 0.1
    second[half::2] = -2.0
    second[half + 1::2] = -0.1
    return first, second


def _hamming(a, b):
    return bin(a ^ b).count('1')


class TestSignature:
    def test_deterministic_across_input_forms(self):
        emb = _embedding(1)
        from_array = simhash.embedding_signature(emb)
        from_bytes = simhash.embedding_signature(emb.tobytes())
        from_list = simhash.embedding_signature(list(emb))
        assert from_array == from_bytes == from_list
        assert isinstance(from_array, int)

    def test_shared_offset_does_not_collapse_signatures(self):
        offset = np.float32(25.0)
        a = simhash.embedding_signature(_embedding(20) + offset)
        b = simhash.embedding_signature(_embedding(21) + offset)
        assert _hamming(a, b) > simhash.SIGNATURE_MATCH_MAX_HAMMING

    def test_reencode_stays_within_tolerance(self):
        emb = _embedding(2)
        nudged = emb + np.float32(1e-4) * _embedding(3)
        a = simhash.embedding_signature(emb)
        b = simhash.embedding_signature(nudged)
        assert _hamming(a, b) <= simhash.SIGNATURE_MATCH_MAX_HAMMING

    def test_distinct_songs_land_far_apart(self):
        a = simhash.embedding_signature(_embedding(4))
        b = simhash.embedding_signature(_embedding(5))
        assert _hamming(a, b) > simhash.SIGNATURE_MATCH_MAX_HAMMING

    def test_invalid_embeddings_have_no_signature(self):
        assert simhash.embedding_signature(None) is None
        assert simhash.embedding_signature([]) is None
        assert simhash.embedding_signature(np.zeros(simhash.SIGNATURE_BITS)) is None
        assert simhash.embedding_signature(np.full(simhash.SIGNATURE_BITS, 3.0)) is None
        assert simhash.embedding_signature(_embedding(6, dim=64)) is None

    def test_batch_matches_single(self):
        embeddings = [_embedding(7), None, _embedding(8)]
        batch = simhash.signature_batch(embeddings)
        assert batch[0] == simhash.embedding_signature(embeddings[0])
        assert batch[1] is None
        assert batch[2] == simhash.embedding_signature(embeddings[2])


class TestCanonicalId:
    def test_id_round_trip(self):
        signature = simhash.embedding_signature(_embedding(9))
        cid = simhash.canonical_id_str(signature)
        assert cid.startswith('fp_2') and len(cid) == simhash.CANONICAL_ID_LEN
        assert simhash.is_fingerprint_id(cid)
        assert simhash.signature_from_canonical_id(cid) == signature

    def test_legacy_scheme_ids_do_not_decode(self):
        assert simhash.signature_from_canonical_id('fp_' + 'a' * 16) is None
        assert simhash.signature_from_canonical_id('fp_1' + 'a' * 16) is None
        assert simhash.signature_from_canonical_id('fp_' + 'a' * 32) is None
        assert simhash.signature_from_canonical_id('provider-1') is None
        assert simhash.signature_from_canonical_id(None) is None
        assert simhash.is_fingerprint_id('fp_' + 'a' * 16)
        assert not simhash.is_fingerprint_id('plain')


class TestSignatureIndex:
    def test_finds_exact_and_near(self):
        base = simhash.embedding_signature(_embedding(10))
        index = simhash.SignatureIndex()
        index.add('one', base)
        assert index.find_candidates(base)[0] == 'one'
        flipped = base ^ 0b1011
        assert _hamming(base, flipped) == 3
        assert index.find_candidates(flipped)[0] == 'one'

    def test_rejects_beyond_tolerance(self):
        base = simhash.embedding_signature(_embedding(11))
        index = simhash.SignatureIndex()
        index.add('one', base)
        far = base
        for bit in range(simhash.SIGNATURE_MATCH_MAX_HAMMING + 1):
            far ^= (1 << (bit * 7))
        assert (
            _hamming(base, far)
            == simhash.SIGNATURE_MATCH_MAX_HAMMING + 1
        )
        assert index.find_candidates(far) == []

    def test_candidates_sorted_nearest_first(self):
        base = simhash.embedding_signature(_embedding(12))
        index = simhash.SignatureIndex()
        index.add('two-bits', base ^ 0b11)
        index.add('exact', base)
        assert index.find_candidates(base) == ['exact', 'two-bits']


class TestCatalogResolver:
    def test_same_audio_resolves_to_existing(self):
        emb = _embedding(13)
        resolver = simhash.CatalogResolver()
        kind, first = resolver.resolve(emb)
        assert kind == 'new' and first.startswith('fp_2')
        reencoded = emb + np.float32(1e-4) * _embedding(14)
        kind2, second = resolver.resolve(reencoded)
        assert (kind2, second) == ('existing', first)

    def test_same_signature_different_audio_gets_own_id(self):
        first, second = _same_signature_different_song()
        assert (
            simhash.embedding_signature(first)
            == simhash.embedding_signature(second)
        )
        assert simhash.cosine_distance(first, second) > 0.01
        resolver = simhash.CatalogResolver()
        _kind, first_id = resolver.resolve(first)
        kind, second_id = resolver.resolve(second)
        assert kind == 'new'
        assert second_id != first_id
        kind3, again = resolver.resolve(second)
        assert (kind3, again) == ('existing', second_id)

    def test_lazy_fetcher_supplies_preexisting_embeddings(self):
        emb = _embedding(16)
        signature = simhash.embedding_signature(emb)
        cid = simhash.canonical_id_str(signature)
        fetched = []

        def fetcher(item_id):
            fetched.append(item_id)
            return emb.tobytes()

        resolver = simhash.CatalogResolver(embedding_fetcher=fetcher)
        resolver.register(cid)
        kind, resolved = resolver.resolve(emb)
        assert (kind, resolved) == ('existing', cid)
        assert fetched == [cid]

    def test_unusable_embedding_resolves_to_nothing(self):
        resolver = simhash.CatalogResolver()
        assert resolver.resolve(None) == ('new', None)
        assert resolver.resolve(np.zeros(simhash.SIGNATURE_BITS)) == ('new', None)


class TestBatchResolveMatchesStreaming:
    """The whole-catalogue resolver must decide identity exactly like the
    streaming one - it rewrites every id in the catalogue, so "faster" is only
    acceptable if it is also "the same answer"."""

    @staticmethod
    def _catalogue(n, seed, clusters=40, dup_frac=0.08):
        rng = np.random.default_rng(seed)
        centers = rng.standard_normal((clusters, simhash.SIGNATURE_BITS)).astype(np.float32)
        idx = rng.integers(0, clusters, size=n)
        rows = centers[idx] + rng.standard_normal(
            (n, simhash.SIGNATURE_BITS)
        ).astype(np.float32) * 0.35
        # Genuine re-encodes of earlier tracks: these must merge.
        ndup = int(n * dup_frac)
        dst = rng.choice(np.arange(n // 2, n), size=ndup, replace=False)
        src = rng.choice(np.arange(0, n // 2), size=ndup, replace=False)
        rows[dst] = rows[src] + rng.standard_normal(
            (ndup, simhash.SIGNATURE_BITS)
        ) * 0.002
        return rows.astype(np.float32)

    @staticmethod
    def _streaming_parents(rows):
        blobs = [row.tobytes() for row in rows]
        signatures = simhash.signature_batch(blobs)
        resolver = simhash.CatalogResolver()
        minted = {}
        parents = []
        for index, (blob, signature) in enumerate(zip(blobs, signatures)):
            kind, item_id = resolver.resolve(blob, signature=signature)
            if kind == 'existing':
                parents.append(minted[item_id])
            else:
                minted[item_id] = index
                parents.append(index)
        return np.array(parents)

    def _batch_parents(self, packed, valid, rows):
        """The whole-catalogue resolution, composed exactly as the startup
        migration composes it (near_duplicate_pairs -> confirm_pairs ->
        merge_pairs). There is no in-memory shortcut for this in production."""
        left, right = simhash.near_duplicate_pairs(packed, valid)
        if left.size == 0:
            return np.arange(packed.shape[0], dtype=np.int64)
        confirmed = simhash.confirm_pairs(rows[left], rows[right])
        return simhash.merge_pairs(
            packed.shape[0], packed, left[confirmed], right[confirmed]
        )

    @pytest.mark.parametrize('seed', [0, 1, 2, 3])
    def test_same_merges_as_the_streaming_resolver(self, seed):
        rows = self._catalogue(600, seed)
        packed, valid = simhash.signature_matrix(rows)

        batch = self._batch_parents(packed, valid, rows)
        streaming = self._streaming_parents(rows)

        assert np.array_equal(batch, streaming)
        assert int((batch != np.arange(len(rows))).sum()) > 0

    def test_a_merged_row_is_never_a_merge_target(self):
        rows = self._catalogue(400, seed=9)
        packed, valid = simhash.signature_matrix(rows)
        parent = self._batch_parents(packed, valid, rows)
        for child, target in enumerate(parent):
            if target != child:
                assert parent[target] == target, "merge chains must not form"
                assert target < child, "a track merges into an EARLIER row"

    def test_distinct_audio_is_never_merged(self):
        rng = np.random.default_rng(3)
        rows = rng.standard_normal((300, simhash.SIGNATURE_BITS)).astype(np.float32)
        packed, valid = simhash.signature_matrix(rows)
        parent = self._batch_parents(packed, valid, rows)
        assert np.array_equal(parent, np.arange(len(rows)))

    def test_unusable_embeddings_stay_their_own_track(self):
        rows = self._catalogue(50, seed=5)
        rows[7] = 0.0
        packed, valid = simhash.signature_matrix(rows)
        assert valid[7] is np.False_
        parent = self._batch_parents(packed, valid, rows)
        assert parent[7] == 7
