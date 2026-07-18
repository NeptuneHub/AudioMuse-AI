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
Similar Songs duplicate filter already trusts, AND by the track duration:
two tracks are the same recording only when their lengths agree within
``DURATION_TOLERANCE_SECONDS`` (the AcoustID rule). A missing duration on
either side means "cannot prove same recording" and identity splits rather
than merges - a false split is a harmless duplicate row, a false merge
deletes a song. Everything deciding identity is derived from the audio
itself.

Main Features:
* ``embedding_signature`` / ``signature_batch`` (vectorized) compute the
  200-bit code; ``canonical_id_str`` / ``signature_from_canonical_id`` encode
  and recover it from the ``fp_2`` id.
* ``SignatureIndex`` banded Hamming-tolerant candidate lookup (pigeonhole
  guarantee within ``SIGNATURE_MATCH_MAX_HAMMING`` bits).
* ``CatalogResolver.resolve``: signature proposes, raw-embedding cosine plus
  duration agreement confirm, collisions mint the next free id.
* ``near_duplicate_pairs`` / ``confirm_pairs`` / ``merge_pairs``: the streaming
  whole-catalogue form the startup migration drives itself.
* ``is_fingerprint_id`` recognizes any ``fp_``-prefixed catalogue id.
"""

import hashlib
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import (
    CATALOGUE_ID_SCHEME_VERSION,
    DUPLICATE_DISTANCE_THRESHOLD_COSINE,
    DURATION_TOLERANCE_SECONDS,
)

logger = logging.getLogger(__name__)

SIGNATURE_BITS = 200
SIGNATURE_MATCH_MAX_HAMMING = 10

_ID_PREFIX = "fp_"
# The current scheme digit comes from config so a future bump is a one-line change.
# Every fp_<n> with n in 1..9 encodes the same 200-bit signature the same way, so an
# older-version id still decodes; only the digit (and thus the id STRING) differs, and
# the startup migration relabels older versions up to the current one.
_ID_SCHEME = str(CATALOGUE_ID_SCHEME_VERSION)
_ID_HEAD = _ID_PREFIX + _ID_SCHEME
CURRENT_ID_HEAD = _ID_HEAD  # public: the current-scheme head, e.g. "fp_3"
_HEX_LEN = ((SIGNATURE_BITS + 7) // 8) * 2
CANONICAL_ID_LEN = len(_ID_HEAD) + _HEX_LEN
_SIGNATURE_MASK = (1 << SIGNATURE_BITS) - 1

_ID_SCHEME_UNSIGNABLE = "0"
_UNSIGNABLE_HEAD = _ID_PREFIX + _ID_SCHEME_UNSIGNABLE

_BAND_COUNT = SIGNATURE_MATCH_MAX_HAMMING + 1
SIGNATURE_BYTES = (SIGNATURE_BITS + 7) // 8


def _band_bit_ranges():
    """Split the signature into ``_BAND_COUNT`` disjoint BIT ranges.

    Any disjoint partition keeps the pigeonhole guarantee - at most ``tolerance``
    flipped bits cannot touch all tolerance+1 bands - so this changes only how
    many FALSE candidates the blocking lets through, never which pairs it can
    find. Splitting on bits rather than bytes is what makes the bands even: 200
    bits over 11 bands is 18-19 bits each, where byte alignment produced eight
    16-bit bands, and a 16-bit band has just 65k buckets for a whole library to
    fall into. On a real (heavily clustered) catalogue those narrow bands emitted
    candidate pairs by the hundred million.
    """
    base, extra = divmod(SIGNATURE_BITS, _BAND_COUNT)
    ranges = []
    start = 0
    for band in range(_BAND_COUNT):
        width = base + (1 if band < extra else 0)
        ranges.append((start, start + width))
        start += width
    return ranges


_BAND_BITS = _band_bit_ranges()

# Popcount tables: "how many bits differ" becomes one vectorized lookup and sum
# over packed signatures, instead of a Python XOR + bin().count('1'). The 16-bit
# table halves the lookups (and the adds) of the 8-bit one.
_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)
_POPCOUNT16 = (
    np.unpackbits(
        np.arange(65536, dtype=np.uint16).view(np.uint8).reshape(-1, 2), axis=1
    )
    .sum(axis=1)
    .astype(np.uint8)
)

# A candidate pair is measured on this PREFIX first. Two unrelated signatures
# differ in about half their bits, so a random pair blows a 10-bit budget inside
# its first 8 bytes and never needs the other 17 read at all - and unrelated pairs
# are almost all of them, because a shared band is mostly a coincidence.
_HEAD_BYTES = 8

# Candidate pairs are GENERATED and filtered in slices of this size, so peak
# memory stays flat no matter how crowded a band gets. A real library clusters
# hard enough that one band can hold hundreds of millions of candidate pairs;
# materializing them all at once is worth gigabytes and gets the container
# OOM-killed, while a slice is worth tens of megabytes - per scanning thread.
_PAIR_CHUNK = 150_000


def _scan_threads():
    """Threads for the band scan; each holds one slice of candidates at a time."""
    return max(1, min(4, (os.cpu_count() or 2) // 2, _BAND_COUNT))


def _pack_signature(signature):
    return np.frombuffer(
        int(signature & _SIGNATURE_MASK).to_bytes(SIGNATURE_BYTES, "big"),
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
    packed = np.zeros((count, SIGNATURE_BYTES), dtype=np.uint8)
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


def durations_compatible(duration_a, duration_b):
    try:
        a = float(duration_a) if duration_a is not None else None
        b = float(duration_b) if duration_b is not None else None
    except (TypeError, ValueError):
        return False
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return False
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) <= DURATION_TOLERANCE_SECONDS


def _duration_mask(left_durations, right_durations, size):
    left = np.full(size, np.nan) if left_durations is None else np.asarray(
        [np.nan if d is None else d for d in left_durations], dtype=np.float64
    )
    right = np.full(size, np.nan) if right_durations is None else np.asarray(
        [np.nan if d is None else d for d in right_durations], dtype=np.float64
    )
    with np.errstate(invalid='ignore'):
        return (
            np.isfinite(left)
            & np.isfinite(right)
            & (left > 0)
            & (right > 0)
            & (np.abs(left - right) <= DURATION_TOLERANCE_SECONDS)
        )


def confirm_pairs(left_vectors, right_vectors, left_durations=None, right_durations=None):
    """Which candidate pairs the EXACT raw-embedding cosine AND duration confirm.

    Row-wise, so a caller can feed it one batch of embeddings at a time instead
    of holding the catalogue's. A zero/absent vector never confirms, and neither
    does a pair whose track lengths are unknown or disagree beyond
    ``DURATION_TOLERANCE_SECONDS`` - the embedding says "sounds the same", only
    the duration can say "is the same recording".
    """
    left_vectors = np.asarray(left_vectors, dtype=np.float32)
    right_vectors = np.asarray(right_vectors, dtype=np.float32)
    left_norms = np.linalg.norm(left_vectors, axis=1)
    right_norms = np.linalg.norm(right_vectors, axis=1)
    safe_left = np.where(left_norms > 0, left_norms, 1.0).astype(np.float32)
    safe_right = np.where(right_norms > 0, right_norms, 1.0).astype(np.float32)
    similarity = np.einsum(
        "ij,ij->i",
        left_vectors / safe_left[:, None],
        right_vectors / safe_right[:, None],
    )
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    return (
        (distance <= DUPLICATE_DISTANCE_THRESHOLD_COSINE)
        & (left_norms > 0)
        & (right_norms > 0)
        & _duration_mask(left_durations, right_durations, left_vectors.shape[0])
    )


def merge_pairs(count, packed, left, right):
    """``parent[i]``: the row whose identity row i takes, from CONFIRMED pairs.

    Each row settles against its NEAREST earlier match, oldest row first - the
    order the streaming resolver saw them in - and a row that merged is never
    itself a merge target, so chains cannot form. ``parent[i] == i`` means "its
    own track"; anything else means "the same audio as row parent[i]".
    """
    parent = np.arange(count, dtype=np.int64)
    if left.size == 0:
        return parent
    hamming = _POPCOUNT[packed[left] ^ packed[right]].sum(axis=1, dtype=np.int16)
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


def unsignable_canonical_id(server_id, provider_track_id):
    """A stable catalogue id for audio whose embedding yields NO signature.

    A constant or non-finite embedding has no signature, so such a track cannot be
    given a content id. It used to keep its raw PROVIDER id as its catalogue id,
    and that is a leak: a non-``fp_`` id is what the availability rule calls a
    pre-migration row and silently grants to the DEFAULT server
    (``left(item_id,3) <> 'fp_'``), so a SECONDARY server's provider id ended up
    counted as present on the default - in clustering, search, sync and the
    dashboard alike. Worse, two servers can share a provider-id namespace, so the
    ids could collide outright.

    So it is ``fp_``-prefixed (never mistaken for a legacy row), scheme-tagged 0
    (never mistaken for a signature id: a signature id is scheme 2), and scoped by
    server (two servers' provider ids can never collide). It is deterministic, so
    the same file resolves to the same id on every run and is skipped.
    """
    if not server_id:
        return str(provider_track_id)
    digest = hashlib.sha256(
        f"{server_id}\x00{provider_track_id}".encode('utf-8')
    ).hexdigest()
    return _UNSIGNABLE_HEAD + digest[:_HEX_LEN]


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


def is_signature_id(item_id):
    """A content id of ANY scheme version (fp_1..fp_9), current or older.

    Scheme 0 is the no-signature ``fp_0`` id, so it is excluded. Length pins it to
    the canonical shape so a retired-shape id (different hex length) is not counted.
    """
    return (
        isinstance(item_id, str)
        and len(item_id) == CANONICAL_ID_LEN
        and item_id[:len(_ID_PREFIX)] == _ID_PREFIX
        and '1' <= item_id[len(_ID_PREFIX):len(_ID_HEAD)] <= '9'
    )


def is_current_scheme_id(item_id):
    return isinstance(item_id, str) and item_id.startswith(_ID_HEAD) \
        and len(item_id) == CANONICAL_ID_LEN


def to_current_scheme_id(item_id):
    """Rewrite any signature id to the current scheme, keeping its 200-bit body.

    Just the version digit changes, so the mapping is total and collision-free: two
    different versions of the same signature converge to the same current id.
    """
    if not is_signature_id(item_id):
        return item_id
    return _ID_HEAD + item_id[len(_ID_HEAD):]


def signature_from_canonical_id(item_id):
    """Recover the signature from any-version content id, or None for anything else."""
    if not is_signature_id(item_id):
        return None
    try:
        return int(item_id[len(_ID_HEAD):], 16) & _SIGNATURE_MASK
    except (TypeError, ValueError):
        return None


def is_fingerprint_id(item_id):
    return isinstance(item_id, str) and item_id.startswith(_ID_PREFIX)


def signature_id_sql(alias=''):
    col = f"{alias}.item_id" if alias else "item_id"
    sql = (
        f"{col} LIKE 'fp\\_%%' AND length({col}) = %s "
        f"AND substring({col} from 4 for 1) BETWEEN '1' AND '9' "
        f"AND left({col}, %s) <> %s"
    )
    return sql, [CANONICAL_ID_LEN, len(CURRENT_ID_HEAD), CURRENT_ID_HEAD]


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
    low, high = _BAND_BITS[band]
    shift = SIGNATURE_BITS - high
    return (signature >> shift) & ((1 << (high - low)) - 1)


def _band_keys(bits, band):
    """The band key of EVERY signature at once, from an unpacked bit matrix."""
    low, high = _BAND_BITS[band]
    keys = np.zeros(bits.shape[0], dtype=np.uint64)
    for column in range(low, high):
        keys = (keys << np.uint64(1)) | bits[:, column].astype(np.uint64)
    return keys


def _iter_group_pairs(order, starts, sizes, limit=_PAIR_CHUNK):
    """Every (a, b), a < b, inside each sorted group - yielded in bounded slices.

    Inverts the triangular pair index over a RANGE of the pair space, so a slice
    costs the same whether the band holds a thousand candidate pairs or a
    billion. Building them all at once (a Python loop over groups is far slower,
    so the whole band went through the inversion in one go) is what an embedded
    container gets OOM-killed for: real libraries cluster hard, and a crowded
    band on 200k tracks can carry hundreds of millions of pairs.
    """
    sizes = sizes.astype(np.int64)
    counts = sizes * (sizes - 1) // 2
    offsets = np.concatenate(([0], np.cumsum(counts)))
    total = int(offsets[-1])
    starts = starts.astype(np.int64)
    for begin in range(0, total, limit):
        pair_index = np.arange(begin, min(begin + limit, total), dtype=np.int64)
        group = np.searchsorted(offsets, pair_index, side="right") - 1
        within = pair_index - offsets[group]
        length = sizes[group]
        first = (
            length - 2
            - np.floor(
                np.sqrt(-8.0 * within + 4.0 * length * (length - 1) - 7.0) / 2.0 - 0.5
            )
        ).astype(np.int64)
        second = (
            within + first + 1
            - length * (length - 1) // 2
            + (length - first) * ((length - first) - 1) // 2
        )
        base = starts[group]
        yield order[base + first], order[base + second]


def near_duplicate_pairs(
    packed, valid, max_hamming=SIGNATURE_MATCH_MAX_HAMMING, progress=None
):
    """Every pair of rows whose signatures are within ``max_hamming`` bits.

    Blocks on the bit-aligned bands (pigeonhole: a pair within tolerance MUST
    share a whole band), then measures the candidates with one vectorized XOR +
    popcount per slice. This is the whole-catalogue form of
    ``SignatureIndex.find_candidates`` - the same comparisons, done as big array
    operations instead of one call per track - and it streams: only ``_PAIR_CHUNK``
    candidate pairs are ever resident, however crowded the band.

    ``progress(band, bands, candidates, survivors)`` is called after each band, so
    a caller running this over a whole library can say what it is doing.
    """
    rows = np.flatnonzero(valid).astype(np.int64)
    if rows.size < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    subject = packed[rows]
    # Unpacked once, not per band: 200 bits x n rows is a byte a bit, which for a
    # 200k-track library is 38 MB - a rounding error next to the candidate pairs
    # this is here to avoid generating.
    bits = np.unpackbits(subject, axis=1)
    head = np.ascontiguousarray(subject[:, :_HEAD_BYTES]).view(np.uint16)
    left_out = []
    right_out = []
    counters = {"candidates": 0, "survivors": 0, "bands": 0}
    lock = threading.Lock()

    def scan_band(band):
        local_left = []
        local_right = []
        candidates = 0
        survivors = 0
        keys = _band_keys(bits, band)
        order = np.argsort(keys, kind="stable")
        _unique, starts, sizes = np.unique(
            keys[order], return_index=True, return_counts=True
        )
        crowded = sizes > 1
        del keys
        if crowded.any():
            for first, second in _iter_group_pairs(order, starts[crowded], sizes[crowded]):
                candidates += first.size
                near = _POPCOUNT16[head[first] ^ head[second]].sum(
                    axis=1, dtype=np.int16
                ) <= max_hamming
                if not near.any():
                    continue
                first = first[near]
                second = second[near]
                distances = _POPCOUNT[subject[first] ^ subject[second]].sum(
                    axis=1, dtype=np.int16
                )
                keep = distances <= max_hamming
                if keep.any():
                    # Survivors are a thin residue of the candidates - int32 rows
                    # keep even a pathological catalogue's residue small.
                    local_left.append(rows[first[keep]].astype(np.int32))
                    local_right.append(rows[second[keep]].astype(np.int32))
                    survivors += int(keep.sum())
        with lock:
            left_out.extend(local_left)
            right_out.extend(local_right)
            counters["candidates"] += candidates
            counters["survivors"] += survivors
            counters["bands"] += 1
            if progress is not None:
                progress(
                    counters["bands"], _BAND_COUNT,
                    counters["candidates"], counters["survivors"],
                )

    # The bands are independent, and every operation in a band - the sort, the
    # XOR, the popcount lookup - is a numpy call that drops the GIL, so this is
    # one of the rare places where THREADS genuinely parallelize Python. The pair
    # order does not matter: the pairs are deduplicated and sorted at the end.
    workers = _scan_threads()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(scan_band, range(_BAND_COUNT)))
    else:
        for band in range(_BAND_COUNT):
            scan_band(band)
    bits = None
    head = None
    if not left_out:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    left = np.concatenate(left_out).astype(np.int64)
    right = np.concatenate(right_out).astype(np.int64)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    # The same pair can surface in several bands.
    unique = np.unique(low * packed.shape[0] + high)
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
        self._packed = np.empty((0, SIGNATURE_BYTES), dtype=np.uint8)
        self._durations = np.empty(0, dtype=np.float64)
        self._count = 0

    def _reserve(self, rows):
        if rows <= self._packed.shape[0]:
            return
        capacity = max(1024, rows, self._packed.shape[0] * 2)
        grown = np.zeros((capacity, SIGNATURE_BYTES), dtype=np.uint8)
        grown[: self._count] = self._packed[: self._count]
        self._packed = grown
        grown_durations = np.full(capacity, np.nan, dtype=np.float64)
        grown_durations[: self._count] = self._durations[: self._count]
        self._durations = grown_durations

    def add(self, canonical_id, signature, duration=None):
        if signature is None:
            return
        signature &= _SIGNATURE_MASK
        row = self._count
        self._reserve(row + 1)
        self._packed[row] = _pack_signature(signature)
        self._durations[row] = (
            np.nan if duration is None else float(duration)
        )
        self._ids.append(canonical_id)
        self._count += 1
        for band in range(_BAND_COUNT):
            self._bands[band].setdefault(_band_key(signature, band), []).append(row)

    def find_candidates(self, signature, duration=None):
        """Canonical ids within Hamming tolerance, sorted nearest-first.

        When ``duration`` is given, candidates whose stored length disagrees by
        more than ``DURATION_TOLERANCE_SECONDS`` are dropped BEFORE the popcount -
        a single vectorized number compare. This is what keeps a sonically
        homogeneous library (thousands of tracks sharing one signature) from
        making the caller walk the whole cluster per track: only the handful with
        a plausible length survive to be confirmed. A candidate whose length is
        unknown (NaN) is kept, so the confirm step can still fetch and judge it.
        """
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
        if duration is not None:
            candidate_durations = self._durations[candidates]
            with np.errstate(invalid='ignore'):
                length_ok = np.isnan(candidate_durations) | (
                    np.abs(candidate_durations - float(duration))
                    <= DURATION_TOLERANCE_SECONDS
                )
            candidates = candidates[length_ok]
            if not candidates.size:
                return []
        distances = _POPCOUNT[self._packed[candidates] ^ _pack_signature(signature)].sum(
            axis=1, dtype=np.int16
        )
        keep = distances <= self._max_hamming
        if not keep.any():
            return []
        candidates = candidates[keep]
        order = np.argsort(distances[keep], kind="stable")
        return [self._ids[row] for row in candidates[order]]

class CatalogResolver:
    """Identity resolver: the signature proposes, embedding + duration confirm.

    A track resolves to an existing catalogue row only when its signature lands
    within Hamming tolerance of that row AND the exact cosine distance between
    the raw embeddings is within ``DUPLICATE_DISTANCE_THRESHOLD_COSINE`` (the
    Similar Songs duplicate rule) AND the two track lengths agree within
    ``DURATION_TOLERANCE_SECONDS``. An unknown duration on either side splits:
    a sonically homogeneous library puts genuinely DIFFERENT recordings inside
    the cosine threshold, and only the length can tell them apart. Anything
    else mints its own id; an exact id-string collision of genuinely different
    content takes the next free signature (identity across installs never
    relies on id equality, only on track_server_map).

    ``embedding_fetcher(item_id)`` / ``duration_fetcher(item_id)`` supply the
    raw embedding and stored duration of a catalogue row that was not
    registered with them (for example rows predating this run).
    """

    def __init__(self, embedding_fetcher=None, duration_fetcher=None):
        self._index = SignatureIndex()
        self._taken = set()
        self._embeddings = {}
        self._durations = {}
        self._fetcher = embedding_fetcher
        self._duration_fetcher = duration_fetcher

    def drop_cached_embeddings(self):
        self._embeddings.clear()

    def register(self, item_id, embedding=None, signature=None, duration=None):
        item_id = str(item_id)
        self._taken.add(item_id)
        if embedding is not None:
            row = _as_matrix([embedding])[0]
            self._embeddings[item_id] = row
        if duration is not None:
            self._durations[item_id] = duration
        if signature is None:
            signature = signature_from_canonical_id(item_id)
        if signature is not None:
            self._index.add(item_id, signature, duration=duration)

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

    def _duration_for(self, item_id):
        if item_id in self._durations:
            return self._durations[item_id]
        if self._duration_fetcher is None:
            return None
        try:
            fetched = self._duration_fetcher(item_id)
        except Exception:
            logger.exception("Duration fetch failed for %s", item_id)
            return None
        self._durations[item_id] = fetched
        return fetched

    def confirms(self, embedding, candidate_id, duration=None):
        """Is ``candidate_id`` the same recording as ``embedding``?

        The signature only ever PROPOSES; the exact cosine plus the duration
        agreement take the decision, and an unknown duration refuses. Public
        because the analysis mint path needs it to tell a concurrently minted
        duplicate (adopt it) from a genuine signature collision between two
        different recordings (step to the next free id).

        Duration is checked FIRST because it is a single number compare while the
        cosine needs the candidate's embedding fetched and a 200-dim dot product.
        A homogeneous library (every track shares a signature) makes ``resolve``
        walk every candidate, and the length rejects almost all of them; doing
        that reject before the cosine keeps the walk O(candidates) cheap instead
        of O(candidates) EMBEDDING FETCHES - the difference between a fast analysis
        and one that pins a core scanning the whole cluster per track.
        """
        if not durations_compatible(duration, self._duration_for(candidate_id)):
            return False
        candidate_embedding = self._embedding_for(candidate_id)
        if candidate_embedding is None:
            return False
        return (
            cosine_distance(embedding, candidate_embedding)
            <= DUPLICATE_DISTANCE_THRESHOLD_COSINE
        )

    _confirms = confirms

    def resolve(self, embedding, signature=None, duration=None):
        """('existing', id) when the audio is already catalogued, else ('new', id).

        A 'new' resolution registers the returned id (with this embedding and
        duration), so the next copy of the same audio in the same run resolves
        to it. Candidates come back nearest-first; the duration veto then walks
        past same-sounding tracks of a different length, so each distinct
        recording anchors its own id even when many share a signature.
        """
        if signature is None:
            signature = embedding_signature(embedding)
        if signature is None:
            return ('new', None)
        for candidate_id in self._index.find_candidates(signature, duration=duration):
            if self._confirms(embedding, candidate_id, duration=duration):
                return ('existing', candidate_id)
        new_id = mint_canonical_id(signature, self._taken)
        self.register(
            new_id,
            embedding=embedding,
            signature=signature_from_canonical_id(new_id),
            duration=duration,
        )
        return ('new', new_id)
