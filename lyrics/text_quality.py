# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared text-quality metrics for the lyrics pipeline.

Holds the zlib compression-ratio metric both the whisper transcript gate and the
embedding text-quality gate threshold against, so the two gates that reject
repetitive hallucinated text cannot drift apart.

Main Features:
* compression_ratio: utf-8 bytes over zlib-compressed bytes of the same text.
"""

from __future__ import annotations

import zlib


def compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    encoded = text.encode('utf-8')
    if not encoded:
        return 0.0
    return len(encoded) / max(1, len(zlib.compress(encoded)))
