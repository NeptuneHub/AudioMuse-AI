# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The stored neural fingerprint format module (tasks.nfp_format).

The P2P node reads a peer's neural fingerprint header without librosa or the
ONNX runtime, so the format lives in its own numpy-only module and the
fingerprint module re-exports it. These tests pin the header layout and prove
the re-export is the same object, not a second copy that could drift.

Main Features:
* parse_header returns (count, codebook id) for a well-formed blob, including
  an empty one, and None for a short, wrong-magic, wrong-dimension,
  wrong-subspace or wrong-length blob and for None
* codebook_id is the crc32 of the float32 bytes whatever the input dtype
* load_codebook accepts the expected shape and refuses any other
* tasks.neural_fingerprint exposes the very same constants and helpers
"""

import zlib

import numpy as np
import pytest

from tasks import nfp_format


def _blob(count=3, magic=None, dim=None, subspaces=None, book_id=0x1234ABCD, extra=0):
    header = nfp_format.HEADER.pack(
        magic or nfp_format.BLOB_MAGIC,
        nfp_format.DIM if dim is None else dim,
        count,
        nfp_format.PQ_SUBSPACES if subspaces is None else subspaces,
        book_id,
    )
    body = bytes(range(256)) * ((count * nfp_format.CODE_BYTES) // 256 + 1)
    return header + body[:count * nfp_format.CODE_BYTES + extra]


class TestLayout:
    def test_header_is_fifteen_little_endian_bytes(self):
        assert nfp_format.HEADER.format == '<4sHIBI'
        assert nfp_format.HEADER_BYTES == 15

    def test_code_bytes_are_one_per_subspace(self):
        assert nfp_format.CODE_BYTES == nfp_format.PQ_SUBSPACES == 32
        assert nfp_format.PQ_SUBDIM * nfp_format.PQ_SUBSPACES == nfp_format.DIM == 128


class TestParseHeader:
    def test_well_formed_blob_returns_count_and_codebook_id(self):
        assert nfp_format.parse_header(_blob(count=7, book_id=0xDEADBEEF)) == (7, 0xDEADBEEF)

    def test_memoryview_and_bytearray_are_accepted(self):
        raw = _blob(count=2)
        assert nfp_format.parse_header(memoryview(raw)) == (2, 0x1234ABCD)
        assert nfp_format.parse_header(bytearray(raw)) == (2, 0x1234ABCD)

    def test_an_empty_fingerprint_parses_with_count_zero(self):
        assert nfp_format.parse_header(_blob(count=0)) == (0, 0x1234ABCD)

    @pytest.mark.parametrize('blob', [
        None,
        b'',
        b'NFP2',
        _blob()[:nfp_format.HEADER_BYTES - 1],
        _blob(magic=b'NFP1'),
        _blob(dim=64),
        _blob(subspaces=16),
        _blob(extra=1),
        _blob()[:-1],
    ])
    def test_malformed_blobs_are_refused(self, blob):
        assert nfp_format.parse_header(blob) is None


class TestCodebook:
    def test_codebook_id_is_the_crc32_of_the_float32_bytes(self):
        book = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        expected = zlib.crc32(book.astype(np.float32).tobytes()) & 0xFFFFFFFF
        assert nfp_format.codebook_id(book) == expected
        assert nfp_format.codebook_id(book.astype(np.float32)) == expected

    def test_load_codebook_accepts_the_expected_shape(self, tmp_path):
        shape = (nfp_format.PQ_SUBSPACES, nfp_format.PQ_CENTROIDS, nfp_format.PQ_SUBDIM)
        book = np.random.default_rng(0).standard_normal(shape)
        path = tmp_path / 'book.npz'
        np.savez(path, codebook=book)
        loaded = nfp_format.load_codebook(str(path))
        assert loaded.dtype == np.float32
        assert loaded.shape == shape
        assert np.allclose(loaded, book.astype(np.float32))

    def test_load_codebook_refuses_another_shape(self, tmp_path):
        path = tmp_path / 'book.npz'
        np.savez(path, codebook=np.zeros((4, 4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match='expected'):
            nfp_format.load_codebook(str(path))


def test_the_fingerprint_module_re_exports_the_same_format():
    nf = pytest.importorskip('tasks.neural_fingerprint')
    assert nf.BLOB_MAGIC is nfp_format.BLOB_MAGIC
    assert nf.DIM == nfp_format.DIM
    assert nf.PQ_SUBSPACES == nfp_format.PQ_SUBSPACES
    assert nf.PQ_CENTROIDS == nfp_format.PQ_CENTROIDS
    assert nf.PQ_SUBDIM == nfp_format.PQ_SUBDIM
    assert nf.CODE_BYTES == nfp_format.CODE_BYTES
    assert nf.HEADER_BYTES == nfp_format.HEADER_BYTES
    assert nf.codebook_id is nfp_format.codebook_id
    assert nf.load_codebook is nfp_format.load_codebook
