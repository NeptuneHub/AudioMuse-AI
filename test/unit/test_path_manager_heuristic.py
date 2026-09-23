# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Which neighbour searches find_path_between_songs issues per path mode.

Drives the path orchestration with injected vector and neighbour functions and a
fake score table, so no index or database is needed.

Main Features:
* The default single-pass mode never runs the start/end neighbour-overlap
  heuristic, which only sizes the centroid jobs of the fixed-size mode
* The fixed-size mode still samples both endpoints once to size its jobs
* Both modes return the same start, intermediate and end songs as before
"""

import numpy as np
import pytest

import database
from tasks import path_manager

VECTORS = {
    's': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    'e': np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    'c1': np.array([0.7, 0.7, 0.1, 0.0], dtype=np.float32),
    'c2': np.array([0.2, 0.9, 0.0, 0.3], dtype=np.float32),
}

DETAILS = {
    's': {'item_id': 's', 'title': 'Song 1', 'author': 'Artist A'},
    'e': {'item_id': 'e', 'title': 'Song 2', 'author': 'Artist B'},
    'c1': {'item_id': 'c1', 'title': 'Song 3', 'author': 'Artist C'},
    'c2': {'item_id': 'c2', 'title': 'Song 4', 'author': 'Artist D'},
}


def _neighbors(vec, n=10):
    ranked = sorted(
        VECTORS, key=lambda iid: path_manager.get_angular_distance(np.asarray(vec), VECTORS[iid])
    )
    return [{'item_id': iid} for iid in ranked[:n]]


@pytest.fixture
def by_id_calls(monkeypatch):
    monkeypatch.setattr(
        database, 'get_score_data_by_ids', lambda ids: [dict(DETAILS[i]) for i in ids if i in DETAILS]
    )
    monkeypatch.setattr(
        database, 'get_tracks_by_ids', lambda ids: [dict(DETAILS[i]) for i in ids if i in DETAILS]
    )
    calls = []

    def by_id(item_id, n=10):
        calls.append(item_id)
        return _neighbors(VECTORS[item_id], n)

    return calls, by_id


def _run(by_id, path_fix_size):
    return path_manager.find_path_between_songs(
        's',
        'e',
        3,
        path_fix_size=path_fix_size,
        get_vector_fn=VECTORS.get,
        neighbors_fn=_neighbors,
        neighbors_by_id_fn=by_id,
        metric='angular',
    )


def test_single_pass_mode_never_runs_the_endpoint_overlap_heuristic(by_id_calls):
    calls, by_id = by_id_calls
    path, _ = _run(by_id, path_fix_size=False)
    assert calls == []
    assert [song['item_id'] for song in path] == ['s', 'c1', 'e']


def test_fixed_size_mode_still_samples_both_endpoints_once(by_id_calls):
    calls, by_id = by_id_calls
    path, _ = _run(by_id, path_fix_size=True)
    assert calls == ['s', 'e']
    assert [song['item_id'] for song in path] == ['s', 'c1', 'e']
