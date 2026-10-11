# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The stored neural fingerprint format (NFP2), with nothing but numpy behind it.

tasks/neural_fingerprint.py computes fingerprints and needs librosa and the ONNX
runtime; the P2P node only has to read a blob's header, check its codebook and
size before accepting it from a peer, and must not load either. Both import the
constants and helpers from here, so the format has one definition.

Main Features:
* DIM, PQ_SUBSPACES, PQ_CENTROIDS, PQ_SUBDIM, CODE_BYTES, BLOB_MAGIC and HEADER
  (magic, dimension, count, slices, codebook id) define the blob layout
* codebook_id and load_codebook identify and load the product-quantisation
  codebook shipped next to the model
* parse_header returns (count, codebook id) for a well-formed NFP2 blob whose
  body length matches its count, None for anything else
"""

import struct
import zlib

import numpy as np

DIM = 128
PQ_SUBSPACES = 32
PQ_CENTROIDS = 256
PQ_SUBDIM = DIM // PQ_SUBSPACES
CODE_BYTES = PQ_SUBSPACES
BLOB_MAGIC = b'NFP2'
HEADER = struct.Struct('<4sHIBI')
HEADER_BYTES = HEADER.size


def codebook_id(book):
    return zlib.crc32(np.ascontiguousarray(book, dtype=np.float32).tobytes()) & 0xFFFFFFFF


def load_codebook(path):
    with np.load(path, allow_pickle=False) as data:
        book = np.ascontiguousarray(data['codebook'], dtype=np.float32)
    expected = (PQ_SUBSPACES, PQ_CENTROIDS, PQ_SUBDIM)
    if book.shape != expected:
        raise ValueError(f'The codebook {path} has shape {book.shape}, expected {expected}')
    return book


def parse_header(blob):
    raw = bytes(blob) if blob is not None else b''
    if len(raw) < HEADER_BYTES:
        return None
    magic, dim, count, subspaces, book_id = HEADER.unpack_from(raw)
    if magic != BLOB_MAGIC or dim != DIM or subspaces != PQ_SUBSPACES:
        return None
    if len(raw) - HEADER_BYTES != count * CODE_BYTES:
        return None
    return count, book_id
