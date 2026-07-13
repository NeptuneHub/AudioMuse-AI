# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""CodSpeed performance benchmarks for the playlist sonic-ordering distances.

Measures the pure distance helpers that drive the greedy nearest-neighbour walk
used when sequencing playlists. These run once per candidate pair, so their cost
scales quadratically with playlist size and is worth tracking.

Main Features:
* Benchmarks the circle-of-fifths key distance across many key/scale pairs.
* Benchmarks the composite tempo/energy/key distance over a realistic song set.
"""

import pytest

from test.unit.conftest import _import_module


playlist_ordering = _import_module(
    'tasks.playlist_ordering', 'tasks/playlist_ordering.py'
)

KEYS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F', None, 'XYZ']
SCALES = ['major', 'minor', None]

KEY_PAIRS = [
    (k1, s1, k2, s2)
    for k1 in KEYS
    for s1 in SCALES
    for k2 in KEYS
    for s2 in SCALES
]

SONGS = [
    {
        'tempo': 60 + (i * 7) % 140,
        'energy': (i % 20) / 20.0,
        'key': KEYS[i % len(KEYS)],
        'scale': SCALES[i % len(SCALES)],
    }
    for i in range(200)
]


def _run_key_distances():
    total = 0.0
    for k1, s1, k2, s2 in KEY_PAIRS:
        total += playlist_ordering._key_distance(k1, s1, k2, s2)
    return total


def _run_composite_distances():
    total = 0.0
    for a in SONGS:
        for b in SONGS:
            total += playlist_ordering._composite_distance(a, b)
    return total


@pytest.mark.benchmark
def test_bench_key_distance(benchmark):
    result = benchmark(_run_key_distances)
    assert result >= 0.0


@pytest.mark.benchmark
def test_bench_composite_distance(benchmark):
    result = benchmark(_run_composite_distances)
    assert result >= 0.0
