# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Content-based catalogue id derived from the MusiCNN embedding.

Every analyzed track has a MusiCNN embedding, so the canonical catalogue id is
computed from it with no extra downloads, binaries, or audio decoding: the
embedding is projected onto 64 fixed seeded hyperplanes and the sign of each
projection becomes one bit of a 64-bit SimHash value encoded as the ``fp_<hex>``
item_id. The hash is similarity-preserving: near-identical audio yields
embeddings whose hashes differ by at most a few bits, so the catalogue lookup is
Hamming-tolerant rather than exact and the same song analyzed from two servers
lands on one shared row.

Main Features:
* ``embedding_fingerprint`` / ``embedding_canonical_id`` map an embedding
  (array, list, or raw float32 bytes) to the deterministic catalogue id.
* ``fingerprint_from_canonical_id`` recovers the 64-bit hash from an id.
* ``find_similar_fingerprint`` Hamming-tolerant match against known hashes.
* ``is_fingerprint_id`` recognizes canonical ``fp_``-prefixed catalogue ids.
"""

import logging

logger = logging.getLogger(__name__)

_SIMHASH_BITS = 64
_UINT64_MASK = (1 << 64) - 1
_LSH_SEED = 662607
_hyperplanes_by_dim = {}

FINGERPRINT_MATCH_MAX_HAMMING = 3


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
    """Signed 64-bit SimHash of a MusiCNN embedding, or None.

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
_ID_HEX_LEN = 16


def canonical_id_str(fingerprint):
    """The catalogue item_id string for a signed 64-bit fingerprint, or None."""
    if fingerprint is None:
        return None
    return _ID_PREFIX + ("%016x" % from_signed_bigint(fingerprint))


def embedding_canonical_id(embedding):
    """The ``fp_<hex>`` catalogue id for an embedding, or None."""
    return canonical_id_str(embedding_fingerprint(embedding))


def fingerprint_from_canonical_id(item_id):
    """Recover the signed 64-bit fingerprint from an embedding-hash ``fp_<hex>`` id.

    Returns None for provider ids and for ids minted by retired schemes (for
    example the 32-hex Chromaprint digests), so those rows simply relabel on the
    next startup migration instead of aliasing a wrong hash.
    """
    if not is_fingerprint_id(item_id) or len(item_id) != len(_ID_PREFIX) + _ID_HEX_LEN:
        return None
    try:
        return to_signed_bigint(int(item_id[len(_ID_PREFIX):], 16))
    except (TypeError, ValueError):
        return None


def is_fingerprint_id(item_id):
    return isinstance(item_id, str) and item_id.startswith(_ID_PREFIX)


def hamming_distance(fingerprint_a, fingerprint_b):
    return bin(from_signed_bigint(fingerprint_a) ^ from_signed_bigint(fingerprint_b)).count("1")


def find_similar_fingerprint(fingerprint, known_fingerprints,
                             max_hamming=FINGERPRINT_MATCH_MAX_HAMMING):
    """The closest known fingerprint within ``max_hamming`` bits, or None.

    The SimHash of the same song analyzed twice (different file, encoder, or
    server) can flip a few near-zero-projection bits, so catalogue membership is
    decided by near-equality. Distinct songs differ by tens of bits, far above
    any sane threshold. ``known_fingerprints`` is any iterable of signed 64-bit
    values; ties resolve to the smallest distance, then first seen.
    """
    if fingerprint is None:
        return None
    target = from_signed_bigint(fingerprint)
    best = None
    best_distance = None
    for known in known_fingerprints:
        if known is None:
            continue
        distance = bin(target ^ from_signed_bigint(known)).count("1")
        if distance <= max_hamming and (best_distance is None or distance < best_distance):
            best = known
            best_distance = distance
            if distance == 0:
                break
    return best


_BAND_COUNT = 4
_BAND_BITS = 16
_BAND_MASK = (1 << _BAND_BITS) - 1


class FingerprintIndex:
    """Hamming-tolerant lookup over many catalogue fingerprints.

    Buckets each 64-bit hash by its four disjoint 16-bit bands: with a
    tolerance of at most 3 flipped bits, at least one band is always intact
    (pigeonhole), so a lookup only Hamming-checks the handful of hashes sharing
    a band with the probe instead of the whole catalogue. Values are the
    canonical ids owning each fingerprint.
    """

    def __init__(self, max_hamming=FINGERPRINT_MATCH_MAX_HAMMING):
        self._max_hamming = min(int(max_hamming), _BAND_COUNT - 1)
        self._bands = [{} for _ in range(_BAND_COUNT)]

    @classmethod
    def from_item_ids(cls, item_ids, max_hamming=FINGERPRINT_MATCH_MAX_HAMMING):
        index = cls(max_hamming=max_hamming)
        for item_id in item_ids:
            fingerprint = fingerprint_from_canonical_id(item_id)
            if fingerprint is not None:
                index.add(item_id, fingerprint)
        return index

    def add(self, canonical_id, fingerprint):
        if fingerprint is None:
            return
        value = from_signed_bigint(fingerprint)
        for band in range(_BAND_COUNT):
            key = (value >> (band * _BAND_BITS)) & _BAND_MASK
            self._bands[band].setdefault(key, []).append((value, canonical_id))

    def find(self, fingerprint):
        """The canonical id whose hash is nearest within tolerance, or None."""
        if fingerprint is None:
            return None
        value = from_signed_bigint(fingerprint)
        best_id = None
        best_distance = None
        seen = set()
        for band in range(_BAND_COUNT):
            key = (value >> (band * _BAND_BITS)) & _BAND_MASK
            for known_value, canonical_id in self._bands[band].get(key, ()):
                if known_value in seen:
                    continue
                seen.add(known_value)
                distance = bin(value ^ known_value).count("1")
                if distance <= self._max_hamming and (
                    best_distance is None or distance < best_distance
                ):
                    best_id = canonical_id
                    best_distance = distance
                    if distance == 0:
                        return best_id
        return best_id
