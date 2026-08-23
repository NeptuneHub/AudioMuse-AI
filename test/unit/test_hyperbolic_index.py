# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the disk-paged exact Poincare index.

Covers the radial band edges, the exact radial lower bound used to order band
probes, the branch-and-bound nearest search against a brute-force baseline,
the not-built fallback signal, and a build/store/load round trip with mocked
storage helpers.
"""

import gzip
import json

import numpy as np

import tasks.hyperbolic_index as hji
from tasks.hyperbolic_geometry import hyperbolic_distance


def _ball(*values):
    vec = np.array(values, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm >= 1.0:
        vec = vec / (norm * 1.05)
    return vec.astype(np.float32), float(np.linalg.norm(vec))


def _index_from_rows(rows, n_bands=3):
    ids = list(rows.keys())
    vectors = np.stack([rows[i][0] for i in ids]).astype(np.float64)
    radii = np.array([rows[i][1] for i in ids], dtype=np.float64)
    edges = hji._band_edges(radii, n_bands)
    n = edges.size - 1
    assigned = np.clip(np.searchsorted(edges, radii, side="right") - 1, 0, n - 1)
    bands = []
    for b in range(n):
        members = [i for i, a in zip(ids, assigned) if int(a) == b]
        bands.append({"blob": f"band_{b}", "count": len(members), "item_ids": members})
    return {"server_key": "s", "dim": vectors.shape[1], "band_edges": edges, "bands": bands}


def test_band_edges_cover_the_ball_and_stay_monotonic():
    radii = np.array([0.1, 0.2, 0.4, 0.7, 0.9, 0.95], dtype=np.float64)
    edges = hji._band_edges(radii, 4)
    assert edges[0] == 0.0
    assert edges[-1] == 1.0
    assert np.all(np.diff(edges) > 0)


def test_radial_lower_bound_is_zero_inside_the_band():
    assert hji._radial_lower_bound(0.5, 0.4, 0.6) == 0.0


def test_radial_lower_bound_grows_with_the_radius_gap():
    near = hji._radial_lower_bound(0.55, 0.0, 0.1)
    far = hji._radial_lower_bound(0.95, 0.0, 0.1)
    assert near > 0.0
    assert far > near


def test_nearest_matches_a_bruteforce_scan(monkeypatch):
    rows = {
        "a": _ball(0.90, 0.00),
        "b": _ball(0.80, 0.10),
        "c": _ball(0.40, 0.40),
        "d": _ball(0.05, 0.02),
        "e": _ball(-0.30, 0.20),
        "f": _ball(-0.70, 0.05),
    }
    index = _index_from_rows(rows, n_bands=3)

    def fake_load(band, idx):
        members = idx["bands"][band]["item_ids"]
        if not members:
            return np.empty((0, idx["dim"]), dtype=np.float32), members
        vecs = np.stack([rows[i][0] for i in members]).astype(np.float64)
        return vecs, members

    monkeypatch.setattr(hji, "_load_band", fake_load)
    target = np.array([0.6, 0.0], dtype=np.float64)
    expected = sorted(
        rows,
        key=lambda i: hyperbolic_distance(target, rows[i][0].astype(np.float64)),
    )[:3]

    got = hji._nearest(target, 3, index, frozenset())
    assert [item_id for item_id, _distance in got] == expected
    for item_id, distance in got:
        exact = hyperbolic_distance(target, rows[item_id][0].astype(np.float64))
        assert abs(distance - exact) < 1e-9


def test_nearest_respects_the_exclude_set(monkeypatch):
    rows = {
        "a": _ball(0.90, 0.00),
        "b": _ball(0.80, 0.10),
        "c": _ball(0.40, 0.40),
    }
    index = _index_from_rows(rows, n_bands=2)

    def fake_load(band, idx):
        members = idx["bands"][band]["item_ids"]
        if not members:
            return np.empty((0, idx["dim"]), dtype=np.float32), members
        vecs = np.stack([rows[i][0] for i in members]).astype(np.float64)
        return vecs, members

    monkeypatch.setattr(hji, "_load_band", fake_load)
    target = np.array([0.6, 0.0], dtype=np.float64)
    got = hji._nearest(target, 3, index, frozenset({"a"}))
    assert "a" not in [item_id for item_id, _distance in got]


def test_hyperbolic_nearest_returns_none_when_not_built():
    hji.reset_hyperbolic_index()
    assert hji.hyperbolic_nearest(np.array([0.5, 0.0]), 3) is None


class _FakeConn:
    def commit(self):
        pass

    def rollback(self):
        pass

    def cursor(self):
        return _FakeCursor()


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, *args, **kwargs):
        pass


def test_build_and_load_roundtrip(monkeypatch):
    rows = {
        "a": _ball(0.90, 0.00),
        "b": _ball(0.80, 0.10),
        "c": _ball(0.40, 0.40),
        "d": _ball(0.05, 0.02),
        "e": _ball(-0.30, 0.20),
    }
    stored = {}

    def fake_fetch_all(server_id=None, include_legacy_default=True):
        return dict(rows)

    def fake_store(conn, table, name, blob, max_part_size_mb=None):
        stored[name] = blob

    def fake_load(conn, table, name):
        return stored.get(name)

    monkeypatch.setattr("tasks.hyperbolic_manager.fetch_all_poincare_rows", fake_fetch_all)
    monkeypatch.setattr("tasks.index_build_helpers.store_segmented_blob", fake_store)
    monkeypatch.setattr("tasks.index_build_helpers.load_segmented_blob", fake_load)
    monkeypatch.setattr(hji, "_scan_index_names", lambda: [hji._DEFAULT_SERVER_KEY])
    monkeypatch.setattr(hji, "_resolve_default_server_id", lambda: None)

    conn = _FakeConn()
    hji.build_and_store_hyperbolic_index(conn)

    dir_name = hji._dir_name(hji._DEFAULT_SERVER_KEY)
    assert dir_name in stored
    directory = json.loads(gzip.decompress(stored[dir_name]).decode("utf-8"))
    assert directory["version"] == 1
    assert sum(band["count"] for band in directory["bands"]) == len(rows)

    monkeypatch.setattr("database.get_db", lambda: conn)
    hji.reset_hyperbolic_index()
    assert hji.load_hyperbolic_index() == 1
    index = hji._index_for(None)
    assert index is not None
    assert len(index["bands"]) == len(directory["bands"])
