# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Real Chromaprint acoustic fingerprints via the fpcalc binary.

Turns a downloaded audio file into a compact fingerprint and compares two of them, so the
duplicate/identity decision can confirm "same recording" acoustically on top of the MusiCNN
embedding and the track duration. The bytes stored in the chromaprint table are the fpcalc raw
integer fingerprint, zlib-compressed. Only collection shells out to fpcalc; the comparison is
pure numpy, so the dedup path carries no native dependency.

Main Features:
* is_available / compute: probe fpcalc once and turn a file into a compressed fingerprint blob,
  returning None on any failure so a missing or broken fpcalc never breaks analysis.
* chromaprints_agree: three-state comparison (True agree / False disagree / None abstain); None
  whenever either side is missing or undecodable, so a caller can fall through to its existing
  verdict. Symmetric, so the streaming and batch dedup paths reach the same answer.
"""

import logging
import subprocess
import zlib

import numpy as np

from config import (
    CHROMAPRINT_MATCH_THRESHOLD,
    CHROMAPRINT_MAX_ALIGN_OFFSET,
    CHROMAPRINT_MIN_OVERLAP,
    FPCALC_BINARY,
)

logger = logging.getLogger(__name__)

_FPCALC_TIMEOUT = 120
_available = None

_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)


def is_available():
    global _available
    if _available is None:
        try:
            subprocess.run(  # nosec B603 B607
                [FPCALC_BINARY, "-version"],
                capture_output=True, timeout=15, check=True,
            )
            _available = True
        except Exception:
            logger.warning(
                "fpcalc not runnable at '%s'; Chromaprint collection disabled", FPCALC_BINARY
            )
            _available = False
    return _available


def compute(audio_path):
    if not audio_path or not is_available():
        return None
    try:
        result = subprocess.run(  # nosec B603 B607
            [FPCALC_BINARY, "-raw", audio_path],
            capture_output=True, text=True, timeout=_FPCALC_TIMEOUT, check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "fpcalc exited %s for %s: %s",
            exc.returncode, audio_path, (exc.stderr or "").strip()[:300],
        )
        return None
    except Exception:
        logger.exception("fpcalc could not be run (binary '%s') for %s",
                         FPCALC_BINARY, audio_path)
        return None
    blob = _encode(result.stdout)
    if blob is None:
        logger.warning(
            "fpcalc ran but produced no parseable fingerprint for %s; "
            "stdout head=%r stderr head=%r",
            audio_path, (result.stdout or "")[:200], (result.stderr or "")[:200],
        )
    return blob


def _encode(fpcalc_stdout):
    ints = _parse_raw(fpcalc_stdout)
    if not ints:
        return None
    arr = np.asarray(ints, dtype=np.int64).astype(np.uint32)
    return zlib.compress(arr.tobytes())


def _parse_raw(fpcalc_stdout):
    for line in fpcalc_stdout.splitlines():
        if line.startswith("FINGERPRINT="):
            body = line[len("FINGERPRINT="):].strip()
            if not body:
                return None
            try:
                return [int(value) for value in body.split(",") if value]
            except ValueError:
                return None
    return None


def _decode(blob):
    if not blob:
        return None
    try:
        raw = zlib.decompress(bytes(blob))
        arr = np.frombuffer(raw, dtype=np.uint32)
    except Exception:
        return None
    return arr if arr.size else None


def _best_match_fraction(a, b):
    best = None
    for offset in range(-CHROMAPRINT_MAX_ALIGN_OFFSET, CHROMAPRINT_MAX_ALIGN_OFFSET + 1):
        if offset >= 0:
            x, y = a[offset:], b[: a.size - offset]
        else:
            x, y = a[: b.size + offset], b[-offset:]
        n = min(x.size, y.size)
        if n < CHROMAPRINT_MIN_OVERLAP:
            continue
        diff_bits = int(_POPCOUNT[(x[:n] ^ y[:n]).view(np.uint8)].sum(dtype=np.int64))
        fraction = 1.0 - diff_bits / (32.0 * n)
        if best is None or fraction > best:
            best = fraction
    return best


def chromaprints_agree(fp_a, fp_b):
    a = _decode(fp_a)
    b = _decode(fp_b)
    if a is None or b is None:
        return None
    best = _best_match_fraction(a, b)
    if best is None:
        return None
    return best >= CHROMAPRINT_MATCH_THRESHOLD
