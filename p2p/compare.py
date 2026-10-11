# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The decode checks for results exchanged with peers.

A peer result is taken as it is: the only checks are the ones needed to store
its bytes in the local shape. Both the P2P node and the worker use them, so a
result that cannot be stored is dropped whichever side sees it first. The
version rule (v3.0.0 or newer) lives in p2p/wire.py.

Main Features:
* musicnn_ok, clap_ok, axis_ok and lyrics_ok: exact size and finite values
* is_lyrics_sentinel: the lyrics instrumental sentinel records a missing text
  source on the holder, not the song, so it is never exchanged
* neural_ok: a well-formed NFP2 blob encoded with the local codebook, the only
  one this server can decode
"""

import numpy as np

import config
from p2p import wire
from tasks import nfp_format


def as_vector(value, dim):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) != int(dim) * 4:
            return None
        return np.frombuffer(raw, dtype='<f4', count=int(dim)).astype(np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != int(dim):
        return None
    return array


def _finite(vector):
    return vector is not None and bool(np.all(np.isfinite(vector)))


def musicnn_ok(vector):
    vec = as_vector(vector, config.EMBEDDING_DIMENSION)
    return _finite(vec)


def clap_ok(vector):
    return _finite(as_vector(vector, config.CLAP_EMBEDDING_DIMENSION))


def is_lyrics_sentinel(vector):
    vec = as_vector(vector, config.LYRICS_EMBEDDING_DIMENSION)
    if vec is None:
        return False
    return bool(np.allclose(vec, config.LYRICS_INSTRUMENTAL_EMBEDDING, atol=1e-6))


def axis_ok(axis):
    return _finite(as_vector(axis, wire.AXIS_DIM))


def lyrics_ok(vector, axis):
    vec = as_vector(vector, config.LYRICS_EMBEDDING_DIMENSION)
    return _finite(vec) and not is_lyrics_sentinel(vec) and axis_ok(axis)


def neural_ok(blob, codebook_crc):
    header = nfp_format.parse_header(blob)
    return header is not None and codebook_crc is not None and header[1] == codebook_crc
