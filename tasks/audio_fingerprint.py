# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Content identity helpers for stable audio Chromaprints and legacy embeddings.

The canonical catalogue id is the SHA-256 digest of the intact Chromaprint text
returned by ``fpcalc``. The older embedding SimHash helpers remain temporarily
for decoding/migrating catalogue ids produced by earlier builds; new identity
creation never depends on a model embedding.

Main Features:
* ``fpcalc_available`` reports (and caches) whether the fpcalc binary exists.
* ``compute_chromaprint`` retains the exact fpcalc value.
* ``chromaprint_canonical_id`` hashes it into a stable ``fp_<hex>`` id.
* Legacy SimHash encode/decode helpers remain backward compatible.
"""

import hashlib
import json
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

_fpcalc_state = {'available': None}


def fpcalc_available():
    """Whether the fpcalc binary is on PATH; a runtime without it (for example a
    standalone bundle) must skip Chromaprint work instead of re-downloading every
    already-analyzed track on every run just to retry an impossible fingerprint."""
    cached = _fpcalc_state['available']
    if cached is not None:
        return cached
    available = shutil.which('fpcalc') is not None
    if not available:
        logger.warning(
            "fpcalc (Chromaprint) not found on PATH; content-id fingerprinting is "
            "disabled and catalogue ids stay provider-based on this install"
        )
    _fpcalc_state['available'] = available
    return available


def compute_chromaprint(file_path, max_seconds=120):
    """Return fpcalc's intact Chromaprint text for an audio file, or None."""
    if not fpcalc_available():
        return None
    try:
        proc = subprocess.run(
            ['fpcalc', '-json', '-length', str(int(max_seconds)), str(file_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=max(60, int(max_seconds) + 30),
        )
        value = json.loads(proc.stdout).get('fingerprint')
        return str(value) if value else None
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        logger.exception("Chromaprint calculation failed for %s", file_path)
        return None


def chromaprint_canonical_id(chromaprint):
    """Hash the intact Chromaprint into a compact, stable catalogue id."""
    if not chromaprint:
        return None
    digest = hashlib.sha256(str(chromaprint).encode('utf-8')).hexdigest()
    return _ID_PREFIX + digest[:32]


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
