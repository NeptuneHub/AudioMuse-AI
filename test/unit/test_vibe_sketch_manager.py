# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the Vibe Sketch manager.

Covers the polyline resampling and the waypoint-to-song snapping without a
database: the KD-tree entry is monkeypatched with a tiny in-memory tree.

Main Features:
* Resampling preserves the start/end and the requested count.
* Snapping returns the requested count, ordered, and deduplicated.
* The availability filter drops songs absent from the server mapping.
"""

import numpy as np
import pytest

from tasks import vibe_sketch_manager as vsm


def test_resample_polyline_straight_line():
    points = [[0.0, 0.0], [10.0, 0.0]]
    waypoints = vsm._resample_polyline(points, 5)
    assert waypoints.shape == (5, 2)
    assert np.allclose(waypoints[0], [0.0, 0.0])
    assert np.allclose(waypoints[-1], [10.0, 0.0])
    assert np.allclose(waypoints[:, 1], 0.0)
    assert np.allclose(waypoints[:, 0], np.linspace(0.0, 10.0, 5))


def test_resample_polyline_single_point_repeats():
    waypoints = vsm._resample_polyline([[2.0, 3.0]], 4)
    assert waypoints.shape == (4, 2)
    assert np.allclose(waypoints, [[2.0, 3.0]] * 4)


def test_resample_polyline_zero_length_path():
    waypoints = vsm._resample_polyline([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], 3)
    assert waypoints.shape == (3, 2)
    assert np.allclose(waypoints, [[1.0, 1.0]] * 3)


def _fake_entry():
    from scipy.spatial import cKDTree

    id_map = ['a', 'b', 'c', 'd']
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return {'id_map': id_map, 'coords': coords, 'tree': cKDTree(coords)}


@pytest.fixture
def tree_entry(monkeypatch):
    entry = _fake_entry()
    monkeypatch.setattr(vsm, '_tree_entry', lambda: entry)
    return entry


def test_sketch_follows_line_and_deduplicates(tree_entry):
    outcome = vsm.sketch_playlist(
        [[0.0, 0.0], [1.0, 1.0]], 3, available=lambda ids: {i: i for i in ids}, variety=0.0
    )
    results = outcome['results']
    assert len(results) == 3
    assert results[0]['item_id'] == 'a'
    assert results[-1]['item_id'] == 'd'
    assert len({r['item_id'] for r in results}) == 3


def test_sketch_respects_availability_filter(tree_entry):
    outcome = vsm.sketch_playlist(
        [[0.0, 0.0], [1.0, 1.0]], 4,
        available=lambda ids: {i: i for i in ids if i != 'd'},
        variety=0.0,
    )
    results = outcome['results']
    assert 'd' not in {r['item_id'] for r in results}


def test_sketch_caps_length(tree_entry):
    outcome = vsm.sketch_playlist(
        [[0.0, 0.0], [1.0, 0.0]], 9999,
        available=lambda ids: {i: i for i in ids}, variety=0.0,
    )
    assert outcome['sampled'] == vsm._MAX_LENGTH


def test_sketch_missing_projection_raises(monkeypatch):
    monkeypatch.setattr(vsm, '_tree_entry', lambda: None)
    with pytest.raises(RuntimeError):
        vsm.sketch_playlist([[0.0, 0.0], [1.0, 1.0]], 3)
