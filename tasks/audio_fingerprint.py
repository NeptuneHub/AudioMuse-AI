# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Content-based catalogue id derived from the MusiCNN embedding.

Every analyzed track already has a MusiCNN embedding, so the canonical catalogue
id is computed from it with no extra downloads or audio decoding: the embedding
is projected onto 64 fixed random hyperplanes (seeded, so identical on every
install) and the sign of each projection becomes one bit of a 64-bit SimHash-LSH
value. Near-identical audio yields near-identical embeddings and therefore the
same (or almost the same) id. The signed BIGINT form is encoded directly into
the ``fp_<hex>`` item_id, looked up by exact equality.

Main Features:
* ``embedding_fingerprint`` maps an embedding vector (array or raw float32
  bytes) to a deterministic signed 64-bit fingerprint.
* ``canonical_id_str`` / ``fingerprint_from_canonical_id`` / ``is_fingerprint_id``
  encode and recover the fingerprint from the catalogue id string.
"""

import logging

logger = logging.getLogger(__name__)

_SIMHASH_BITS = 64
_UINT64_MASK = (1 << 64) - 1
_LSH_SEED = 662607
_hyperplanes_by_dim = {}


def _hyperplanes(dim):
    planes = _hyperplanes_by_dim.get(dim)
    if planes is None:
        import numpy as np

        rng = np.random.RandomState(_LSH_SEED)
        planes = rng.standard_normal((_SIMHASH_BITS, dim)).astype(np.float64)
        _hyperplanes_by_dim[dim] = planes
    return planes


def _as_vector(embedding):
    import numpy as np

    if embedding is None:
        return None
    if isinstance(embedding, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(embedding), dtype=np.float32)
    else:
        arr = np.asarray(embedding)
    arr = arr.astype(np.float64, copy=False).ravel()
    if arr.size == 0 or not np.isfinite(arr).all() or not arr.any():
        return None
    return arr


def embedding_fingerprint(embedding):
    """Signed 64-bit SimHash-LSH fingerprint of a MusiCNN embedding, or None.

    Accepts a numpy array, a list, or the raw float32 bytes stored in the
    ``embedding`` table. Deterministic across platforms: the hyperplanes come
    from the frozen numpy RandomState generator with a fixed seed.
    """
    vector = _as_vector(embedding)
    if vector is None:
        return None
    projections = _hyperplanes(vector.size) @ vector
    value = 0
    for bit in range(_SIMHASH_BITS):
        if projections[bit] > 0:
            value |= (1 << bit)
    return to_signed_bigint(value)


def to_signed_bigint(value):
    """Map an unsigned 64-bit value into the signed range Postgres BIGINT accepts."""
    if value is None:
        return None
    value &= _UINT64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value


def from_signed_bigint(value):
    if value is None:
        return None
    return value + (1 << 64) if value < 0 else value


_ID_PREFIX = "fp_"


def canonical_id_str(fingerprint):
    """The catalogue item_id string for a signed-BIGINT fingerprint, or None."""
    if fingerprint is None:
        return None
    return _ID_PREFIX + ("%016x" % from_signed_bigint(fingerprint))


def fingerprint_from_canonical_id(item_id):
    """Recover the signed 64-bit fingerprint encoded by an ``fp_<hex>`` id."""
    if not is_fingerprint_id(item_id):
        return None
    try:
        return to_signed_bigint(int(item_id[len(_ID_PREFIX):], 16))
    except (TypeError, ValueError):
        return None


def is_fingerprint_id(item_id):
    return isinstance(item_id, str) and item_id.startswith(_ID_PREFIX)


def embedding_canonical_id(embedding):
    """The ``fp_<hex>`` catalogue id for an embedding, or None."""
    return canonical_id_str(embedding_fingerprint(embedding))
