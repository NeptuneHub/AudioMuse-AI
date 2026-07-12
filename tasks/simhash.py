# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Home-made similarity hash: content identity purely from the MusiCNN embedding.

The catalogue id is a 200-bit signature, one bit per embedding dimension: bit d
is "dimension d is above this song's own average". No random projections, no
external binaries, no metadata - the id IS the shape of the song's MusiCNN
profile, encoded as the scheme-versioned ``fp_2<50hex>`` item_id. The signature
is similarity-preserving (a re-encode of the same recording flips only a few
borderline bits, distinct songs differ by tens), so near signatures propose
identity; the decision is then confirmed by the EXACT cosine distance between
the raw embeddings using the same ``DUPLICATE_DISTANCE_THRESHOLD_COSINE`` the
Similar Songs duplicate filter already trusts, with an optional duration check
when both durations are known. Everything deciding identity is derived from the
audio itself.

Main Features:
* ``embedding_signature`` / ``signature_batch`` (vectorized) compute the
  200-bit code; ``canonical_id_str`` / ``signature_from_canonical_id`` encode
  and recover it from the ``fp_2`` id.
* ``SignatureIndex`` banded Hamming-tolerant candidate lookup (pigeonhole
  guarantee within ``SIGNATURE_MATCH_MAX_HAMMING`` bits).
* ``CatalogResolver.resolve``: signature proposes, raw-embedding cosine (and
  optional duration) confirms, collisions mint the next free id.
* ``is_fingerprint_id`` recognizes any ``fp_``-prefixed catalogue id.
"""

import logging

import numpy as np

from config import DUPLICATE_DISTANCE_THRESHOLD_COSINE

logger = logging.getLogger(__name__)

SIGNATURE_BITS = 200
SIGNATURE_MATCH_MAX_HAMMING = 10
DURATION_TOLERANCE_SECONDS = 2.0

_ID_PREFIX = "fp_"
_ID_SCHEME = "2"
_ID_HEAD = _ID_PREFIX + _ID_SCHEME
_HEX_LEN = ((SIGNATURE_BITS + 7) // 8) * 2
CANONICAL_ID_LEN = len(_ID_HEAD) + _HEX_LEN
_SIGNATURE_MASK = (1 << SIGNATURE_BITS) - 1

_BAND_COUNT = SIGNATURE_MATCH_MAX_HAMMING + 1
_BAND_BOUNDS = [
    (SIGNATURE_BITS * band) // _BAND_COUNT
    for band in range(_BAND_COUNT + 1)
]


def _as_matrix(embeddings):
    rows = []
    for embedding in embeddings:
        if isinstance(embedding, (bytes, bytearray, memoryview)):
            rows.append(np.frombuffer(bytes(embedding), dtype=np.float32))
        else:
            rows.append(np.asarray(embedding, dtype=np.float32).ravel())
    return rows


def signature_batch(embeddings):
    """Signatures for many embeddings at once (vectorized), None where invalid.

    Invalid means missing, wrong dimensionality, non-finite, or constant - those
    tracks keep their provider id instead of receiving a degenerate signature.
    """
    rows = _as_matrix(embeddings)
    out = [None] * len(rows)
    valid_positions = [
        i for i, row in enumerate(rows)
        if row.size == SIGNATURE_BITS and np.isfinite(row).all() and np.ptp(row) > 0
    ]
    if not valid_positions:
        return out
    matrix = np.stack([rows[i] for i in valid_positions]).astype(np.float64)
    matrix -= matrix.mean(axis=1, keepdims=True)
    bits = (matrix > 0).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    for position, row_bytes in zip(valid_positions, packed):
        out[position] = int.from_bytes(row_bytes.tobytes(), "big")
    return out


def embedding_signature(embedding):
    """The 200-bit signature of one embedding, or None when it is unusable."""
    if embedding is None:
        return None
    return signature_batch([embedding])[0]


def canonical_id_str(signature):
    """The catalogue item_id string for a signature, or None.

    Scheme-versioned (``fp_2<50hex>``): ids minted by earlier schemes have a
    different shape and relabel on the next startup migration.
    """
    if signature is None:
        return None
    return _ID_HEAD + format(signature & _SIGNATURE_MASK, "0%dx" % _HEX_LEN)


def embedding_canonical_id(embedding):
    """The ``fp_2<hex>`` catalogue id for an embedding, or None."""
    return canonical_id_str(embedding_signature(embedding))


def signature_from_canonical_id(item_id):
    """Recover the signature from a current-scheme id, or None for anything else."""
    if (
        not is_fingerprint_id(item_id)
        or len(item_id) != CANONICAL_ID_LEN
        or not item_id.startswith(_ID_HEAD)
    ):
        return None
    try:
        return int(item_id[len(_ID_HEAD):], 16) & _SIGNATURE_MASK
    except (TypeError, ValueError):
        return None


def is_fingerprint_id(item_id):
    return isinstance(item_id, str) and item_id.startswith(_ID_PREFIX)


def hamming_distance(signature_a, signature_b):
    return bin((signature_a & _SIGNATURE_MASK) ^ (signature_b & _SIGNATURE_MASK)).count("1")


def cosine_distance(embedding_a, embedding_b):
    """Cosine distance between two raw embeddings (the Similar Songs metric)."""
    a = _as_matrix([embedding_a])[0].astype(np.float64)
    b = _as_matrix([embedding_b])[0].astype(np.float64)
    if a.size != b.size or a.size == 0:
        return 1.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 1.0
    return 1.0 - float(np.dot(a, b)) / denominator


def _band_key(signature, band):
    low, high = _BAND_BOUNDS[band], _BAND_BOUNDS[band + 1]
    return (signature >> low) & ((1 << (high - low)) - 1)


class SignatureIndex:
    """Hamming-tolerant lookup over many signatures.

    The 200 bits are split into ``tolerance + 1`` disjoint bands: at most
    ``tolerance`` flipped bits always leave one band intact (pigeonhole), so a
    lookup only Hamming-checks the signatures sharing a band with the probe.
    """

    def __init__(self, max_hamming=SIGNATURE_MATCH_MAX_HAMMING):
        self._max_hamming = min(int(max_hamming), _BAND_COUNT - 1)
        self._bands = [{} for _ in range(_BAND_COUNT)]

    def add(self, canonical_id, signature):
        if signature is None:
            return
        signature &= _SIGNATURE_MASK
        for band in range(_BAND_COUNT):
            self._bands[band].setdefault(
                _band_key(signature, band), []
            ).append((signature, canonical_id))

    def find_candidates(self, signature):
        """All canonical ids within tolerance, sorted nearest-first."""
        if signature is None:
            return []
        signature &= _SIGNATURE_MASK
        matches = []
        seen = set()
        for band in range(_BAND_COUNT):
            for known, canonical_id in self._bands[band].get(
                _band_key(signature, band), ()
            ):
                if (known, canonical_id) in seen:
                    continue
                seen.add((known, canonical_id))
                distance = bin(signature ^ known).count("1")
                if distance <= self._max_hamming:
                    matches.append((distance, canonical_id))
        matches.sort(key=lambda pair: pair[0])
        return [canonical_id for _distance, canonical_id in matches]

    def find(self, signature):
        candidates = self.find_candidates(signature)
        return candidates[0] if candidates else None


class CatalogResolver:
    """Identity resolver: the signature proposes, the raw embedding confirms.

    A track resolves to an existing catalogue row only when its signature lands
    within Hamming tolerance of that row AND the exact cosine distance between
    the raw embeddings is within ``DUPLICATE_DISTANCE_THRESHOLD_COSINE`` (the
    Similar Songs duplicate rule). When both durations are known they must also
    agree within ``DURATION_TOLERANCE_SECONDS``. Anything else mints its own id;
    an exact id-string collision of genuinely different content takes the next
    free signature (identity across installs never relies on id equality, only
    on track_server_map).

    ``embedding_fetcher(item_id)`` supplies the raw embedding of a catalogue row
    that was not registered with one (for example rows predating this run).
    """

    def __init__(self, embedding_fetcher=None):
        self._index = SignatureIndex()
        self._taken = set()
        self._embeddings = {}
        self._durations = {}
        self._fetcher = embedding_fetcher

    def register(self, item_id, embedding=None, duration=None, signature=None):
        item_id = str(item_id)
        self._taken.add(item_id)
        if embedding is not None:
            row = _as_matrix([embedding])[0]
            self._embeddings[item_id] = row
        if duration is not None:
            self._durations[item_id] = float(duration)
        if signature is None:
            signature = signature_from_canonical_id(item_id)
        if signature is not None:
            self._index.add(item_id, signature)

    def _embedding_for(self, item_id):
        cached = self._embeddings.get(item_id)
        if cached is not None:
            return cached
        if self._fetcher is None:
            return None
        try:
            fetched = self._fetcher(item_id)
        except Exception:
            logger.exception("Embedding fetch failed for %s", item_id)
            return None
        if fetched is None:
            return None
        row = _as_matrix([fetched])[0]
        self._embeddings[item_id] = row
        return row

    def _confirms(self, embedding, duration, candidate_id):
        candidate_embedding = self._embedding_for(candidate_id)
        if candidate_embedding is None:
            return False
        if cosine_distance(embedding, candidate_embedding) > DUPLICATE_DISTANCE_THRESHOLD_COSINE:
            return False
        candidate_duration = self._durations.get(candidate_id)
        if duration is not None and candidate_duration is not None:
            if abs(float(duration) - candidate_duration) > DURATION_TOLERANCE_SECONDS:
                return False
        return True

    def resolve(self, embedding, duration=None, signature=None):
        """('existing', id) when the audio is already catalogued, else ('new', id).

        A 'new' resolution registers the returned id (with this embedding), so
        the next copy of the same audio in the same run resolves to it.
        """
        if signature is None:
            signature = embedding_signature(embedding)
        if signature is None:
            return ('new', None)
        for candidate_id in self._index.find_candidates(signature):
            if self._confirms(embedding, duration, candidate_id):
                return ('existing', candidate_id)
        value = signature & _SIGNATURE_MASK
        new_id = canonical_id_str(value)
        while new_id in self._taken:
            value = (value + 1) & _SIGNATURE_MASK
            new_id = canonical_id_str(value)
        self.register(new_id, embedding=embedding, duration=duration, signature=value)
        return ('new', new_id)
