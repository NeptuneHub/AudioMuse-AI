# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Chromaprint fingerprint compute, compare and the optional dedup gate.

Verifies the three-state comparison (agree / disagree / ABSTAIN): identical
fingerprints agree, bit-flipped ones disagree, and a missing, empty or
undecodable side abstains (returns None) so the caller keeps its cosine+duration
verdict. Also checks fpcalc parse/encode/decode round-trips and that the
CatalogResolver gate only ever ADDS splits - a disagreeing fingerprint splits a
pair the cosine+duration would have merged, an agreeing one confirms it, and a
missing one leaves today's behaviour byte-for-byte unchanged.

Main Features:
* chromaprints_agree three-state (agree / disagree / abstain), symmetric, and
  abstaining on missing, empty, undecodable or bad-length blobs without raising.
* fpcalc compute fail-soft and the raw-int parse / encode / decode round-trip.
* CatalogResolver + confirm_pairs gate: disagree splits, agree merges, a missing
  fingerprint abstains (byte-identical to the pre-Chromaprint decision).
"""

import zlib

import numpy as np
import pytest

from tasks import chromaprint, simhash


def _fp_blob(values):
    return zlib.compress(np.asarray(values, dtype=np.uint32).tobytes())


FP_A = _fp_blob(list(range(200)))
FP_A_SHIFTED = _fp_blob(list(range(3, 200)))
FP_FLIPPED = _fp_blob([v ^ 0xFFFFFFFF for v in range(200)])


class TestChromaprintsAgree:
    def test_identical_fingerprints_agree(self):
        assert chromaprints_result(FP_A, FP_A) is True

    def test_bit_flipped_fingerprints_disagree(self):
        assert chromaprints_result(FP_A, FP_FLIPPED) is False

    def test_small_offset_still_aligns_and_agrees(self):
        assert chromaprints_result(FP_A, FP_A_SHIFTED) is True

    def test_missing_side_abstains(self):
        assert chromaprint.chromaprints_agree(None, FP_A) is None
        assert chromaprint.chromaprints_agree(FP_A, None) is None
        assert chromaprint.chromaprints_agree(b'', FP_A) is None
        assert chromaprint.chromaprints_agree(FP_A, b'') is None

    def test_undecodable_side_abstains(self):
        assert chromaprint.chromaprints_agree(b'not-zlib', FP_A) is None

    def test_bad_length_blob_abstains_not_raises(self):
        odd = zlib.compress(b'abc')
        assert chromaprint.chromaprints_agree(odd, FP_A) is None

    def test_too_short_overlap_abstains(self):
        short = _fp_blob([1, 2, 3, 4, 5])
        assert chromaprint.chromaprints_agree(short, short) is None

    def test_is_symmetric(self):
        assert (
            chromaprint.chromaprints_agree(FP_A, FP_FLIPPED)
            == chromaprint.chromaprints_agree(FP_FLIPPED, FP_A)
        )
        assert (
            chromaprint.chromaprints_agree(FP_A, FP_A_SHIFTED)
            == chromaprint.chromaprints_agree(FP_A_SHIFTED, FP_A)
        )


def chromaprints_result(a, b):
    return chromaprint.chromaprints_agree(a, b)


class TestParseEncodeDecode:
    def test_parse_raw_reads_the_fingerprint_line(self):
        assert chromaprint._parse_raw("DURATION=180\nFINGERPRINT=1,2,3\n") == [1, 2, 3]

    def test_encode_decode_round_trips(self):
        blob = chromaprint._encode("FINGERPRINT=10,20,30\n")
        assert list(chromaprint._decode(blob)) == [10, 20, 30]

    def test_negative_ints_wrap_to_uint32(self):
        blob = chromaprint._encode("FINGERPRINT=-1,0,1\n")
        assert list(chromaprint._decode(blob)) == [0xFFFFFFFF, 0, 1]

    def test_empty_or_garbage_fingerprint_encodes_to_none(self):
        assert chromaprint._encode("FINGERPRINT=\n") is None
        assert chromaprint._encode("no fingerprint here") is None


class TestComputeFailSoft:
    def test_compute_returns_none_without_fpcalc(self, monkeypatch):
        monkeypatch.setattr(chromaprint, "is_available", lambda: False)
        assert chromaprint.compute("nonexistent-track.flac") is None

    def test_compute_returns_none_on_empty_path(self):
        assert chromaprint.compute(None) is None
        assert chromaprint.compute("") is None

    def test_compute_pipes_fpcalc_output_into_a_blob(self, monkeypatch):
        monkeypatch.setattr(chromaprint, "is_available", lambda: True)

        class _Result:
            stdout = "DURATION=180\nFINGERPRINT=4,5,6\n"

        monkeypatch.setattr(chromaprint.subprocess, "run", lambda *a, **k: _Result())
        blob = chromaprint.compute("some-track.flac")
        assert list(chromaprint._decode(blob)) == [4, 5, 6]


def _embedding(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(simhash.SIGNATURE_BITS).astype(np.float32)


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setattr(simhash, "CHROMAPRINT_GATE_ENABLED", True)


class TestResolverChromaprintGate:
    def _resolver_with_candidate(self, emb):
        signature = simhash.embedding_signature(emb)
        cid = simhash.canonical_id_str(signature)
        resolver = simhash.CatalogResolver(
            embedding_fetcher=lambda _id: emb.tobytes(),
            duration_fetcher=lambda _id: 200.0,
            fingerprint_fetcher=lambda _id: FP_A,
        )
        resolver.register(cid)
        return resolver, cid

    def test_disagreeing_fingerprint_splits_a_cosine_duration_match(self, gate_on):
        emb = _embedding(21)
        resolver, cid = self._resolver_with_candidate(emb)
        kind, resolved = resolver.resolve(emb, duration=200.0, fingerprint=FP_FLIPPED)
        assert kind == 'new'
        assert resolved != cid

    def test_agreeing_fingerprint_confirms_the_merge(self, gate_on):
        emb = _embedding(21)
        resolver, cid = self._resolver_with_candidate(emb)
        kind, resolved = resolver.resolve(emb, duration=200.0, fingerprint=FP_A)
        assert (kind, resolved) == ('existing', cid)

    def test_missing_new_fingerprint_abstains_and_merges(self, gate_on):
        emb = _embedding(21)
        resolver, cid = self._resolver_with_candidate(emb)
        kind, resolved = resolver.resolve(emb, duration=200.0, fingerprint=None)
        assert (kind, resolved) == ('existing', cid)


class TestConfirmPairsChromaprintGate:
    def test_veto_only_when_both_present_and_disagree(self, gate_on):
        emb = _embedding(30)
        left = np.stack([emb])
        right = np.stack([emb])
        dur = [200.0]
        assert simhash.confirm_pairs(left, right, dur, dur, [FP_A], [FP_A])[0]
        assert not simhash.confirm_pairs(left, right, dur, dur, [FP_A], [FP_FLIPPED])[0]
        assert simhash.confirm_pairs(left, right, dur, dur, [FP_A], [None])[0]

    def test_no_fingerprints_is_identical_to_base(self, gate_on):
        emb = _embedding(31)
        left = np.stack([emb])
        right = np.stack([emb])
        dur = [200.0]
        assert simhash.confirm_pairs(left, right, dur, dur)[0]
