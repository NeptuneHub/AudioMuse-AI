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
_SIGNATURE_BYTES = (SIGNATURE_BITS + 7) // 8


def _band_byte_ranges():
    """Split the packed signature into ``_BAND_COUNT`` disjoint BYTE ranges.

    Any disjoint partition keeps the pigeonhole guarantee (at most ``tolerance``
    flipped bits cannot touch all tolerance+1 bands). Aligning the bands to byte
    boundaries is what lets a whole catalogue's band keys be computed in one
    vectorized pass instead of one big-integer shift per track per band.
    """
    base, extra = divmod(_SIGNATURE_BYTES, _BAND_COUNT)
    ranges = []
    start = 0
    for band in range(_BAND_COUNT):
        width = base + (1 if band < extra else 0)
        ranges.append((start, start + width))
        start += width
    return ranges


_BAND_BYTES = _band_byte_ranges()

# Byte-wise popcount table: "how many bits differ" becomes one vectorized lookup
# and sum over packed signatures, instead of a Python XOR + bin().count('1').
_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)

# Candidate pairs are filtered in slices so peak memory stays flat no matter how
# crowded a band gets.
_PAIR_CHUNK = 1_000_000


def _pack_signature(signature):
    return np.frombuffer(
        int(signature & _SIGNATURE_MASK).to_bytes(_SIGNATURE_BYTES, "big"),
        dtype=np.uint8,
    )


def _unpack_signature(packed_row):
    return int.from_bytes(bytes(packed_row), "big") & _SIGNATURE_MASK


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


def signature_matrix(rows):
    """Packed signatures for a whole embedding MATRIX: (n, 25) uint8 + valid mask.

    Same bits as ``signature_batch``, kept packed instead of converted to Python
    integers: the batch resolver compares them as bytes, and for a 200k-track
    migration the round-trip through big integers costs more than the hashing.
    """
    rows = np.asarray(rows)
    count = rows.shape[0]
    packed = np.zeros((count, _SIGNATURE_BYTES), dtype=np.uint8)
    valid = np.zeros(count, dtype=bool)
    if count == 0 or rows.ndim != 2 or rows.shape[1] != SIGNATURE_BITS:
        return packed, valid
    matrix = rows.astype(np.float64, copy=False)
    valid = np.isfinite(matrix).all(axis=1) & (
        matrix.max(axis=1) > matrix.min(axis=1)
    )
    if not valid.any():
        return packed, valid
    usable = matrix[valid]
    usable = usable - usable.mean(axis=1, keepdims=True)
    packed[valid] = np.packbits(usable > 0, axis=1)
    return packed, valid


def resolve_catalog(packed, valid, embeddings, durations=None):
    """Identity-resolve a WHOLE catalogue at once: ``parent[i]`` is i's row.

    One vectorized pass instead of one probe per track. The rule is unchanged -
    the signature proposes (within SIGNATURE_MATCH_MAX_HAMMING, banded), the
    EXACT raw-embedding cosine confirms (DUPLICATE_DISTANCE_THRESHOLD_COSINE),
    and known durations must agree - and so is the outcome: a track merges into
    the NEAREST EARLIER row that confirms, and a row that merged is never itself
    a merge target, so chains cannot form. ``parent[i] == i`` means "its own
    track"; anything else means "the same audio as row parent[i]".

    Rows must be ordered oldest-first (already-canonical rows before the legacy
    ones being migrated), which is what makes "earlier wins" mean "keep the id
    the catalogue already has".
    """
    count = packed.shape[0]
    parent = np.arange(count, dtype=np.int64)
    left, right = near_duplicate_pairs(packed, valid)
    if left.size == 0:
        return parent

    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    safe = np.where(norms > 0, norms, 1.0).astype(np.float32)
    unit = embeddings / safe[:, None]
    similarity = np.einsum("ij,ij->i", unit[left], unit[right])
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    confirmed = (
        (distance <= DUPLICATE_DISTANCE_THRESHOLD_COSINE)
        & (norms[left] > 0)
        & (norms[right] > 0)
    )
    if durations is not None:
        durations = np.asarray(durations, dtype=np.float64)
        both_known = np.isfinite(durations[left]) & np.isfinite(durations[right])
        agree = np.abs(durations[left] - durations[right]) <= DURATION_TOLERANCE_SECONDS
        confirmed &= ~both_known | agree
    if not confirmed.any():
        return parent

    left = left[confirmed]
    right = right[confirmed]
    hamming = _POPCOUNT[packed[left] ^ packed[right]].sum(axis=1, dtype=np.int16)
    # Settle each row against its NEAREST earlier match, oldest row first - the
    # order the streaming resolver would have seen them in.
    for index in np.lexsort((left, hamming, right)):
        child = int(right[index])
        target = int(left[index])
        if parent[child] != child or parent[target] != target:
            continue
        parent[child] = target
    return parent


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


def mint_canonical_id(signature, taken):
    """The catalogue id for ``signature``, stepping past ids already ``taken``.

    An exact id-string collision means two genuinely DIFFERENT recordings hashed
    to the same 200 bits, so the newcomer takes the next free id rather than
    stealing the other's identity.
    """
    value = signature & _SIGNATURE_MASK
    item_id = canonical_id_str(value)
    while item_id in taken:
        value = (value + 1) & _SIGNATURE_MASK
        item_id = canonical_id_str(value)
    return item_id


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
    """Cosine distance between two raw embeddings (the Similar Songs metric).

    Clipped to [0, 2] like the index's own distance, so floating-point drift on
    a near-identical pair can never produce a tiny negative value that reads as
    "closer than identical" against the duplicate threshold.
    """
    a = _as_matrix([embedding_a])[0].astype(np.float64)
    b = _as_matrix([embedding_b])[0].astype(np.float64)
    if a.size != b.size or a.size == 0:
        return 1.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 1.0
    similarity = float(np.dot(a, b)) / denominator
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def _band_key(signature, band):
    low, high = _BAND_BYTES[band]
    shift = (_SIGNATURE_BYTES - high) * 8
    width = (high - low) * 8
    return (signature >> shift) & ((1 << width) - 1)


def _band_keys(packed, band):
    """The band key of EVERY packed signature at once (one uint64 per row)."""
    low, high = _BAND_BYTES[band]
    keys = np.zeros(packed.shape[0], dtype=np.uint64)
    for column in range(low, high):
        keys = (keys << np.uint64(8)) | packed[:, column].astype(np.uint64)
    return keys


def _pairs_within_groups(order, sizes, starts):
    """Every (a, b), a < b, inside each sorted group - fully vectorized.

    Inverts the triangular pair index instead of looping over groups in Python:
    a catalogue of 200k tracks has ~1M groups per band, and a per-group Python
    loop costs more than the comparison itself.
    """
    counts = sizes * (sizes - 1) // 2
    total = int(counts.sum())
    if not total:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    group = np.repeat(np.arange(sizes.size, dtype=np.int64), counts)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    within = np.arange(total, dtype=np.int64) - offsets[group]
    length = sizes[group].astype(np.int64)
    first = (
        length - 2
        - np.floor(np.sqrt(-8.0 * within + 4.0 * length * (length - 1) - 7.0) / 2.0 - 0.5)
    ).astype(np.int64)
    second = (
        within + first + 1
        - length * (length - 1) // 2
        + (length - first) * ((length - first) - 1) // 2
    ).astype(np.int64)
    base = starts[group].astype(np.int64)
    return order[base + first], order[base + second]


def near_duplicate_pairs(packed, valid, max_hamming=SIGNATURE_MATCH_MAX_HAMMING):
    """Every pair of rows whose signatures are within ``max_hamming`` bits.

    Blocks on the byte-aligned bands (pigeonhole: a pair within tolerance MUST
    share a whole band), then measures the surviving candidates with one
    vectorized XOR + popcount. This is the whole-catalogue form of
    ``SignatureIndex.find_candidates``: the same comparisons, done as a handful
    of big array operations instead of one call per track.
    """
    rows = np.flatnonzero(valid)
    if rows.size < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    left_out = []
    right_out = []
    for band in range(_BAND_COUNT):
        keys = _band_keys(packed[rows], band)
        order = np.argsort(keys, kind="stable")
        _unique, starts, sizes = np.unique(
            keys[order], return_index=True, return_counts=True
        )
        crowded = sizes > 1
        if not crowded.any():
            continue
        left, right = _pairs_within_groups(order, sizes[crowded], starts[crowded])
        for begin in range(0, left.size, _PAIR_CHUNK):
            slice_left = rows[left[begin:begin + _PAIR_CHUNK]]
            slice_right = rows[right[begin:begin + _PAIR_CHUNK]]
            distances = _POPCOUNT[packed[slice_left] ^ packed[slice_right]].sum(
                axis=1, dtype=np.int16
            )
            keep = distances <= max_hamming
            if keep.any():
                left_out.append(slice_left[keep])
                right_out.append(slice_right[keep])
    if not left_out:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    left = np.concatenate(left_out)
    right = np.concatenate(right_out)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    # The same pair can surface in several bands.
    unique = np.unique(low.astype(np.int64) * packed.shape[0] + high.astype(np.int64))
    return unique // packed.shape[0], unique % packed.shape[0]


class SignatureIndex:
    """Hamming-tolerant lookup over many signatures.

    The 200 bits are split into ``tolerance + 1`` disjoint bands: at most
    ``tolerance`` flipped bits always leave one band intact (pigeonhole), so a
    lookup only Hamming-checks the signatures sharing a band with the probe.

    Signatures live in one packed uint8 matrix and the bands hold row indices,
    so a probe compares against its candidates with a single vectorized
    XOR + popcount. The obvious Python loop over each bucket entry is what made
    the startup migration quadratic AND single-core: bucket occupancy grows with
    the catalogue (real music clusters, so the bands fill unevenly), and the bit
    twiddling holds the GIL, so no thread pool can rescue it.
    """

    def __init__(self, max_hamming=SIGNATURE_MATCH_MAX_HAMMING):
        self._max_hamming = min(int(max_hamming), _BAND_COUNT - 1)
        self._bands = [{} for _ in range(_BAND_COUNT)]
        self._ids = []
        self._packed = np.empty((0, _SIGNATURE_BYTES), dtype=np.uint8)
        self._count = 0

    def _reserve(self, rows):
        if rows <= self._packed.shape[0]:
            return
        capacity = max(1024, rows, self._packed.shape[0] * 2)
        grown = np.zeros((capacity, _SIGNATURE_BYTES), dtype=np.uint8)
        grown[: self._count] = self._packed[: self._count]
        self._packed = grown

    def add(self, canonical_id, signature):
        if signature is None:
            return
        signature &= _SIGNATURE_MASK
        row = self._count
        self._reserve(row + 1)
        self._packed[row] = _pack_signature(signature)
        self._ids.append(canonical_id)
        self._count += 1
        for band in range(_BAND_COUNT):
            self._bands[band].setdefault(_band_key(signature, band), []).append(row)

    def find_candidates(self, signature):
        """All canonical ids within tolerance, sorted nearest-first."""
        if signature is None or not self._count:
            return []
        signature &= _SIGNATURE_MASK
        rows = []
        for band in range(_BAND_COUNT):
            bucket = self._bands[band].get(_band_key(signature, band))
            if bucket:
                rows.extend(bucket)
        if not rows:
            return []
        candidates = np.unique(np.asarray(rows, dtype=np.int64))
        distances = _POPCOUNT[self._packed[candidates] ^ _pack_signature(signature)].sum(
            axis=1, dtype=np.int16
        )
        keep = distances <= self._max_hamming
        if not keep.any():
            return []
        candidates = candidates[keep]
        order = np.argsort(distances[keep], kind="stable")
        return [self._ids[row] for row in candidates[order]]

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

    def resolve(self, embedding, duration=None, signature=None, store_embedding=True):
        """('existing', id) when the audio is already catalogued, else ('new', id).

        A 'new' resolution registers the returned id (with this embedding), so
        the next copy of the same audio in the same run resolves to it. Pass
        ``store_embedding=False`` to register the id only and rely on the
        ``embedding_fetcher`` for later confirmations - bulk migrations use
        this to keep memory bounded.
        """
        if signature is None:
            signature = embedding_signature(embedding)
        if signature is None:
            return ('new', None)
        for candidate_id in self._index.find_candidates(signature):
            if self._confirms(embedding, duration, candidate_id):
                return ('existing', candidate_id)
        new_id = mint_canonical_id(signature, self._taken)
        self.register(
            new_id,
            embedding=embedding if store_embedding else None,
            duration=duration,
            signature=signature_from_canonical_id(new_id),
        )
        return ('new', new_id)
