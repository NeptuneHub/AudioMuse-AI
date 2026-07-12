# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Content identity helpers for stable audio Chromaprint catalogue ids.

The canonical catalogue id is the SHA-256 digest of the intact Chromaprint text
returned by ``fpcalc``; identity never depends on a model embedding.

Main Features:
* ``fpcalc_available`` reports (and caches) whether the fpcalc binary exists.
* ``compute_chromaprint`` retains the exact fpcalc value.
* ``chromaprint_canonical_id`` hashes it into a stable ``fp_<hex>`` id.
* ``is_fingerprint_id`` recognizes canonical ``fp_``-prefixed catalogue ids.
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


_ID_PREFIX = "fp_"


def chromaprint_canonical_id(chromaprint):
    """Hash the intact Chromaprint into a compact, stable catalogue id."""
    if not chromaprint:
        return None
    digest = hashlib.sha256(str(chromaprint).encode('utf-8')).hexdigest()
    return _ID_PREFIX + digest[:32]


def is_fingerprint_id(item_id):
    return isinstance(item_id, str) and item_id.startswith(_ID_PREFIX)
