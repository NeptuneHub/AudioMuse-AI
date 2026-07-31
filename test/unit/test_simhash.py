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
Songs duplicate rule) PLUS track-duration agreement within
DURATION_TOLERANCE_SECONDS, with the signature only proposing candidates - two
songs sharing a signature but failing the cosine or the duration stay separate
rows, and an unknown duration always splits.

Main Features:
* Signature determinism, similarity window, invalid-input handling.
* fp_2 id round-trip and legacy-scheme rejection.
* Resolver: cosine plus duration confirm, collisions mint the next free id.
* Duration ladder: same-sounding tracks partition into one id per length.
"""

import numpy as np

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
        assert cid.startswith(simhash.CURRENT_ID_HEAD) and len(cid) == simhash.CANONICAL_ID_LEN
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

    def test_unsignable_ids_are_scheme_zero_and_never_decode_as_signatures(self):
        fp0 = simhash.unsignable_canonical_id('srv', 'track-1')
        assert simhash.is_unsignable_id(fp0)
        assert simhash.is_fingerprint_id(fp0)
        assert not simhash.is_signature_id(fp0)
        assert simhash.signature_from_canonical_id(fp0) is None
        signature_id = simhash.canonical_id_str(
            simhash.embedding_signature(_embedding(9))
        )
        assert not simhash.is_unsignable_id(signature_id)
        assert not simhash.is_unsignable_id(fp0[:-1])
        assert not simhash.is_unsignable_id(None)
        assert simhash.is_unsignable_id('fp_0' + 'a' * 50)


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


class TestDurationsCompatible:
    def test_within_tolerance_is_compatible(self):
        tol = simhash.DURATION_TOLERANCE_SECONDS
        assert simhash.durations_compatible(200.0, 200.0)
        assert simhash.durations_compatible(200.0, 200.0 + tol)   # exactly at the tolerance
        assert simhash.durations_compatible(200.0 + tol, 200.0)

    def test_beyond_tolerance_is_not_compatible(self):
        tol = simhash.DURATION_TOLERANCE_SECONDS
        assert not simhash.durations_compatible(200.0, 200.0 + tol + 0.1)
        assert not simhash.durations_compatible(200.0, 200.0 + tol + 3.0)

    def test_unknown_or_invalid_duration_never_compatible(self):
        assert not simhash.durations_compatible(None, 200.0)
        assert not simhash.durations_compatible(200.0, None)
        assert not simhash.durations_compatible(None, None)
        assert not simhash.durations_compatible(0.0, 0.0)
        assert not simhash.durations_compatible(-5.0, -5.0)
        assert not simhash.durations_compatible(float('nan'), 200.0)
        assert not simhash.durations_compatible('junk', 200.0)


class TestCatalogResolver:
    def test_same_audio_same_duration_resolves_to_existing(self):
        emb = _embedding(13)
        resolver = simhash.CatalogResolver()
        kind, first = resolver.resolve(emb, duration=200.0)
        assert kind == 'new' and first.startswith(simhash.CURRENT_ID_HEAD)
        reencoded = emb + np.float32(1e-4) * _embedding(14)
        # A re-encode within the length tolerance is the same recording.
        kind2, second = resolver.resolve(
            reencoded, duration=200.0 + simhash.DURATION_TOLERANCE_SECONDS)
        assert (kind2, second) == ('existing', first)

    def test_same_audio_unknown_duration_mints_new_id(self):
        emb = _embedding(13)
        resolver = simhash.CatalogResolver()
        _kind, first = resolver.resolve(emb, duration=200.0)
        kind, second = resolver.resolve(emb)
        assert kind == 'new'
        assert second != first

    def test_same_audio_different_duration_mints_new_id(self):
        emb = _embedding(13)
        resolver = simhash.CatalogResolver()
        _kind, first = resolver.resolve(emb, duration=200.0)
        kind, second = resolver.resolve(emb, duration=210.0)
        assert kind == 'new'
        assert second != first

    def test_duration_ladder_partitions_same_sounding_tracks_by_length(self):
        emb = _embedding(13)
        durations = [200.0, 200.0, 210.0, 210.0, 220.0, 220.0]
        resolver = simhash.CatalogResolver()
        results = [resolver.resolve(emb, duration=d) for d in durations]
        ids = [item_id for _kind, item_id in results]
        kinds = [kind for kind, _item_id in results]
        assert kinds == ['new', 'existing', 'new', 'existing', 'new', 'existing']
        assert ids[0] == ids[1]
        assert ids[2] == ids[3]
        assert ids[4] == ids[5]
        assert len({ids[0], ids[2], ids[4]}) == 3

    def test_same_signature_different_audio_gets_own_id(self):
        first, second = _same_signature_different_song()
        assert (
            simhash.embedding_signature(first)
            == simhash.embedding_signature(second)
        )
        assert simhash.cosine_distance(first, second) > 0.01
        resolver = simhash.CatalogResolver()
        _kind, first_id = resolver.resolve(first, duration=200.0)
        kind, second_id = resolver.resolve(second, duration=200.0)
        assert kind == 'new'
        assert second_id != first_id
        kind3, again = resolver.resolve(second, duration=200.0)
        assert (kind3, again) == ('existing', second_id)

    def test_lazy_fetchers_supply_preexisting_embedding_and_duration(self):
        emb = _embedding(16)
        signature = simhash.embedding_signature(emb)
        cid = simhash.canonical_id_str(signature)
        fetched = []
        duration_fetched = []

        def fetcher(item_id):
            fetched.append(item_id)
            return emb.tobytes()

        def duration_fetcher(item_id):
            duration_fetched.append(item_id)
            return 200.0

        resolver = simhash.CatalogResolver(
            embedding_fetcher=fetcher, duration_fetcher=duration_fetcher
        )
        resolver.register(cid)
        kind, resolved = resolver.resolve(emb, duration=200.0)
        assert (kind, resolved) == ('existing', cid)
        assert fetched == [cid]
        assert duration_fetched == [cid]

    def test_duration_mismatch_skips_the_expensive_cosine(self):
        # In a homogeneous library resolve() walks every same-signature candidate;
        # a length mismatch must reject BEFORE the embedding is fetched, or every
        # candidate costs an embedding fetch + 200-dim cosine and analysis pins a
        # core scanning the whole cluster per track (the O(n^2) regression).
        emb = _embedding(13)
        signature = simhash.embedding_signature(emb)
        cid = simhash.canonical_id_str(signature)
        fetched = []

        resolver = simhash.CatalogResolver(
            embedding_fetcher=lambda item_id: fetched.append(item_id) or emb.tobytes(),
            duration_fetcher=lambda item_id: 500.0,
        )
        resolver.register(cid)
        kind, _resolved = resolver.resolve(emb, duration=200.0)
        assert kind == 'new'
        assert fetched == [], (
            "a length mismatch must skip the embedding fetch and cosine entirely"
        )

    def test_catalogue_row_without_stored_duration_never_absorbs(self):
        emb = _embedding(16)
        signature = simhash.embedding_signature(emb)
        cid = simhash.canonical_id_str(signature)

        resolver = simhash.CatalogResolver(
            embedding_fetcher=lambda _item_id: emb.tobytes(),
            duration_fetcher=lambda _item_id: None,
        )
        resolver.register(cid)
        kind, resolved = resolver.resolve(emb, duration=200.0)
        assert kind == 'new'
        assert resolved != cid

    def test_unusable_embedding_resolves_to_nothing(self):
        resolver = simhash.CatalogResolver()
        assert resolver.resolve(None) == ('new', None)
        assert resolver.resolve(np.zeros(simhash.SIGNATURE_BITS)) == ('new', None)


class TestFolderRule:
    def test_folder_key_strips_mount_and_filename(self):
        assert simhash.folder_key(
            '/media/music/Artist/Album/01 - Song.flac'
        ) == 'artist/album'
        assert simhash.folder_key(
            '/mnt/music/Artist/Album/02 - Other.flac'
        ) == 'artist/album'

    def test_group_conflict_detects_two_files_in_one_folder(self):
        assert simhash.folder_conflict_in_group([
            '/media/music/A/Alb/01 - x.flac',
            '/media/music/A/Alb/02 - y.flac',
        ]) is True

    def test_group_across_folders_has_no_conflict(self):
        assert simhash.folder_conflict_in_group([
            '/media/music/A/Alb1/01 - x.flac',
            '/media/music/A/Alb2/01 - x.flac',
        ]) is False

    def test_same_audio_same_duration_same_folder_mints_new_id(self):
        emb = _embedding(21)
        resolver = simhash.CatalogResolver()
        _kind, first = resolver.resolve(
            emb, duration=200.0, path='/media/music/A/Alb/01 - One.flac'
        )
        kind, second = resolver.resolve(
            emb, duration=200.0, path='/media/music/A/Alb/02 - Two.flac'
        )
        assert kind == 'new'
        assert second != first

    def test_same_audio_same_duration_different_folder_merges(self):
        emb = _embedding(21)
        resolver = simhash.CatalogResolver()
        _kind, first = resolver.resolve(
            emb, duration=200.0, path='/media/music/A/Alb1/01 - One.flac'
        )
        kind, second = resolver.resolve(
            emb, duration=200.0, path='/media/music/A/Alb2/01 - One.flac'
        )
        assert (kind, second) == ('existing', first)

    def test_folder_veto_uses_the_path_fetcher_for_persisted_rows(self):
        emb = _embedding(22)
        cid = simhash.canonical_id_str(simhash.embedding_signature(emb))
        resolver = simhash.CatalogResolver(
            embedding_fetcher=lambda _id: emb.tobytes(),
            duration_fetcher=lambda _id: 200.0,
            path_fetcher=lambda _id: ['/media/music/A/Alb/01 - One.flac'],
        )
        resolver.register(cid)
        kind, resolved = resolver.resolve(
            emb, duration=200.0, path='/media/music/A/Alb/02 - Two.flac'
        )
        assert kind == 'new'
        assert resolved != cid

    def test_merge_pairs_folder_rule_is_group_level_not_pairwise(self):
        # Three near-identical rows; 0 and 2 share a folder. Both match row 1 (a
        # different folder), so a pairwise reject of the 0-2 pair would still let
        # them co-merge via row 1. The group-level rule must keep 2 out of 0's
        # group, so a same-folder merge is never formed in the first place.
        packed = np.stack([simhash._pack_signature(0)] * 3)
        left = np.array([0, 0, 1], dtype=np.int64)
        right = np.array([1, 2, 2], dtype=np.int64)
        folders = ['a/alb', 'b/alb', 'a/alb']

        parent = simhash.merge_pairs(3, packed, left, right, folders=folders)
        assert parent[1] == 0, "different-folder file still merges"
        assert parent[2] == 2, "same-folder file gets its own id, not 0's group"

        plain = simhash.merge_pairs(3, packed, left, right)
        assert plain[2] == 0, "without folders they would all collapse into one id"
