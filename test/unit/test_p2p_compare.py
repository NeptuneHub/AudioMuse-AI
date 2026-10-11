# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The decode checks for peer results (p2p.compare).

A peer result is taken as it is when it decodes into the stored shape; the
version rule (v3.0.0 or newer) lives in p2p.wire and is pinned in
test_p2p_wire. These tests show that each check refuses only what cannot be
stored, and nothing more.

Main Features:
* musicnn_ok, clap_ok, axis_ok and lyrics_ok refuse a wrong size, a missing
  value or a non-finite value and accept any other finite vector (no norm or
  axis-sum rule); a constant MusiCNN embedding has no signature and is refused
* the lyrics instrumental sentinel is never exchanged
* neural_ok accepts any well-formed NFP2 blob of the local codebook, whatever
  its length, and refuses another codebook or a malformed blob
* axis_ok uses wire.AXIS_DIM (27), which matches the lyrics axis columns when
  the lyrics package imports; compare keeps no copy of the version rule
"""

import numpy as np
import pytest

import config
from p2p import compare, wire
from tasks import nfp_format

CODEBOOK = 0xC0DEB00C


def _unit(dim, seed=0):
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _neural_blob(count, book_id=CODEBOOK):
    header = nfp_format.HEADER.pack(nfp_format.BLOB_MAGIC, nfp_format.DIM, count, nfp_format.PQ_SUBSPACES, book_id)
    return header + bytes(count * nfp_format.CODE_BYTES)


class TestAsVector:
    def test_bytes_and_arrays_of_the_exact_size_decode(self):
        vector = _unit(8)
        assert np.array_equal(compare.as_vector(vector.astype('<f4').tobytes(), 8), vector)
        assert np.array_equal(compare.as_vector(vector.reshape(2, 4), 8), vector)

    def test_any_other_size_or_none_does_not(self):
        vector = _unit(8)
        assert compare.as_vector(vector.astype('<f4').tobytes()[:-1], 8) is None
        assert compare.as_vector(vector[:-1], 8) is None
        assert compare.as_vector(None, 8) is None


class TestMusicnn:
    def test_a_real_shaped_embedding_passes(self):
        vector = np.random.default_rng(1).standard_normal(config.EMBEDDING_DIMENSION)
        assert compare.musicnn_ok(vector)
        assert compare.musicnn_ok(vector.astype('<f4').tobytes())

    def test_wrong_size_and_non_finite_fail(self):
        assert not compare.musicnn_ok(np.ones(config.EMBEDDING_DIMENSION + 1))
        vector = np.random.default_rng(1).standard_normal(config.EMBEDDING_DIMENSION)
        vector[3] = np.nan
        assert not compare.musicnn_ok(vector)
        assert not compare.musicnn_ok(None)


class TestClap:
    def test_any_finite_vector_of_the_right_size_passes(self):
        vector = _unit(config.CLAP_EMBEDDING_DIMENSION)
        assert compare.clap_ok(vector)
        assert compare.clap_ok(vector.astype('<f4').tobytes())
        assert compare.clap_ok(vector * 3.0)
        assert compare.clap_ok(np.zeros(config.CLAP_EMBEDDING_DIMENSION))

    def test_wrong_size_non_finite_and_missing_fail(self):
        vector = _unit(config.CLAP_EMBEDDING_DIMENSION)
        assert not compare.clap_ok(vector[:-1])
        assert not compare.clap_ok(vector.astype('<f4').tobytes()[:-4])
        broken = vector.copy()
        broken[0] = np.inf
        assert not compare.clap_ok(broken)
        assert not compare.clap_ok(None)


class TestAxis:
    def test_any_finite_vector_of_the_right_size_passes(self):
        assert wire.AXIS_DIM == 27
        axis = np.random.default_rng(3).random(wire.AXIS_DIM).astype(np.float32)
        assert compare.axis_ok(axis)
        assert compare.axis_ok(axis.astype('<f4').tobytes())
        assert compare.axis_ok(np.full(wire.AXIS_DIM, config.LYRICS_INSTRUMENTAL_AXIS_FILL, dtype=np.float32))

    def test_non_finite_wrong_size_and_missing_fail(self):
        axis = np.full(wire.AXIS_DIM, 0.2, dtype=np.float32)
        broken = axis.copy()
        broken[5] = np.nan
        assert not compare.axis_ok(broken)
        assert not compare.axis_ok(axis[:-1])
        assert not compare.axis_ok(None)


class TestLyrics:
    def test_a_finite_embedding_with_a_finite_axis_passes(self):
        axis = np.full(wire.AXIS_DIM, 0.2, dtype=np.float32)
        assert compare.lyrics_ok(_unit(config.LYRICS_EMBEDDING_DIMENSION), axis)
        assert compare.lyrics_ok(_unit(config.LYRICS_EMBEDDING_DIMENSION) * 2.0, axis)

    def test_the_instrumental_sentinel_never_passes(self):
        sentinel = np.array(config.LYRICS_INSTRUMENTAL_EMBEDDING, dtype=np.float32)
        assert compare.is_lyrics_sentinel(sentinel)
        assert compare.is_lyrics_sentinel(sentinel.tobytes())
        assert not compare.lyrics_ok(sentinel, np.full(wire.AXIS_DIM, 0.2, dtype=np.float32))

    def test_a_real_embedding_is_not_the_sentinel(self):
        assert not compare.is_lyrics_sentinel(_unit(config.LYRICS_EMBEDDING_DIMENSION))
        assert not compare.is_lyrics_sentinel(None)

    def test_a_bad_embedding_or_a_bad_axis_fails(self):
        good = _unit(config.LYRICS_EMBEDDING_DIMENSION)
        axis = np.full(wire.AXIS_DIM, 0.2, dtype=np.float32)
        broken = good.copy()
        broken[0] = np.nan
        assert not compare.lyrics_ok(broken, axis)
        assert not compare.lyrics_ok(good[:-1], axis)
        assert not compare.lyrics_ok(good, None)
        assert not compare.lyrics_ok(good, axis[:-2])
        assert not compare.lyrics_ok(None, axis)


class TestNeural:
    @pytest.mark.parametrize('count', [0, 1, 400, 100000])
    def test_a_blob_of_the_local_codebook_passes_whatever_its_length(self, count):
        assert compare.neural_ok(_neural_blob(count), CODEBOOK)

    def test_another_codebook_or_no_local_codebook_fails(self):
        assert not compare.neural_ok(_neural_blob(10), CODEBOOK + 1)
        assert not compare.neural_ok(_neural_blob(10), None)

    def test_a_malformed_blob_fails(self):
        assert not compare.neural_ok(_neural_blob(10)[:-1], CODEBOOK)
        assert not compare.neural_ok(b'NFP1' + _neural_blob(10)[4:], CODEBOOK)
        assert not compare.neural_ok(b'', CODEBOOK)
        assert not compare.neural_ok(None, CODEBOOK)


def test_axis_dim_matches_the_lyrics_axis_columns():
    try:
        from lyrics import lyrics_transcriber
    except Exception as exc:
        pytest.skip(f'lyrics.lyrics_transcriber does not import here: {exc}')
    assert len(lyrics_transcriber.axis_columns()) == wire.AXIS_DIM


def test_the_version_rule_has_one_home_in_wire():
    for name in ('MIN_OUTPUT_VERSION', 'parse_version', 'version_allowed', 'AXIS_DIM'):
        assert not hasattr(compare, name), name
        assert hasattr(wire, name), name
