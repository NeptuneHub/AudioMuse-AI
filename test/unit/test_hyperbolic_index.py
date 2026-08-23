# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the disk-paged Poincare IVF index.

Covers the cell-count rule, the cell scan against a brute-force baseline, the
not-built signal, and a build/store/load round trip with mocked storage
helpers.

Main Features:
* Cell count follows 8*sqrt(n) capped at IVF_NLIST_MAX and never exceeds the
  track count, the same rule the other IVF indexes use
* Probing every cell reproduces a brute-force exact Poincare ranking, and the
  exclude set is honoured
* An unbuilt index reports None so the caller can raise instead of scanning
* build/store/load round trips through mocked segmented-blob helpers, and the
  directory carries the centroid blob the query path needs
* Cell matrices are written in the configured IVF_STORAGE_DTYPE through the
  shared ivf_quant codec, and the blob byte length matches that element size
  whatever dtype the projected rows arrive in
* IVF_STORAGE_DTYPE=i8 is taken literally here and is NOT downgraded to f16
  the way ivf_quant.effective_code would for a non-angular metric
* An i8 scan is overfetched and re-ranked against the exact float32 rows, so
  hyperbolic_nearest still returns the exact ordering and exact distances
* Every probed cell is read exactly once per query, from the one bulk read,
  even when the cell cache is too small to retain anything between the
  prefetch and the scan
"""

import gzip
import json

import numpy as np
import pytest

import config
import tasks.hyperbolic_index as hji
from tasks import ivf_quant as quant
from tasks.hyperbolic_geometry import hyperbolic_distance, hyperbolic_distances_to


def _ball(*values):
    vec = np.array(values, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm >= 1.0:
        vec = vec / (norm * 1.05)
    return vec.astype(np.float32), float(np.linalg.norm(vec))


def _index_from_rows(rows, n_cells=3):
    from tasks.hyperbolic_geometry import nearest_centroid, poincare_kmeans

    ids = list(rows.keys())
    vectors = np.stack([rows[i][0] for i in ids]).astype(np.float64)
    centroids, _labels = poincare_kmeans(vectors, n_cells, iterations=5)
    assigned = nearest_centroid(vectors, centroids)
    cells = []
    for c in range(centroids.shape[0]):
        members = [i for i, a in zip(ids, assigned) if int(a) == c]
        cells.append({"blob": f"cell_{c}", "count": len(members), "item_ids": members})
    return {
        "server_key": "s",
        "dim": vectors.shape[1],
        "code": quant.DTYPE_F32,
        "centroids": centroids,
        "centroid_norms2": np.sum(centroids * centroids, axis=1),
        "cells": cells,
    }


def test_cell_count_follows_eight_root_n_like_the_other_indexes():
    assert hji._cell_count(0) == 1
    assert hji._cell_count(1) == 1
    assert hji._cell_count(10_000) == 800
    assert hji._cell_count(200_000) == round(8.0 * 200_000 ** 0.5)
    assert hji._cell_count(10_000_000) == config.IVF_NLIST_MAX


def test_cell_count_never_exceeds_the_track_count():
    for n in (2, 5, 50, 500):
        assert hji._cell_count(n) <= n


def test_nearest_matches_a_bruteforce_scan(monkeypatch):
    rows = {
        "a": _ball(0.90, 0.00),
        "b": _ball(0.80, 0.10),
        "c": _ball(0.40, 0.40),
        "d": _ball(0.05, 0.02),
        "e": _ball(-0.30, 0.20),
        "f": _ball(-0.70, 0.05),
    }
    index = _index_from_rows(rows, n_cells=3)

    def fake_load(cell, idx):
        members = idx["cells"][cell]["item_ids"]
        if not members:
            return np.empty((0, idx["dim"]), dtype=np.float32), members
        vecs = np.stack([rows[i][0] for i in members]).astype(np.float64)
        return vecs, members

    monkeypatch.setattr(hji, "_load_cell", fake_load)
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
    index = _index_from_rows(rows, n_cells=2)

    def fake_load(cell, idx):
        members = idx["cells"][cell]["item_ids"]
        if not members:
            return np.empty((0, idx["dim"]), dtype=np.float32), members
        vecs = np.stack([rows[i][0] for i in members]).astype(np.float64)
        return vecs, members

    monkeypatch.setattr(hji, "_load_cell", fake_load)
    target = np.array([0.6, 0.0], dtype=np.float64)
    got = hji._nearest(target, 3, index, frozenset({"a"}))
    assert "a" not in [item_id for item_id, _distance in got]


def test_hyperbolic_nearest_returns_none_when_not_built():
    hji.reset_hyperbolic_index()
    assert hji.hyperbolic_nearest(np.array([0.5, 0.0]), 3) is None


class _FakeConn:
    def __init__(self, blobs=None, reads=None):
        self._blobs = blobs if blobs is not None else {}
        self._reads = reads if reads is not None else []

    def commit(self):
        pass

    def rollback(self):
        pass

    def cursor(self):
        return _FakeCursor(self._blobs, self._reads)


class _FakeCursor:
    def __init__(self, blobs=None, reads=None):
        self._blobs = blobs if blobs is not None else {}
        self._reads = reads if reads is not None else []
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self._rows = []
        if "blob_data" in sql and "ANY" in sql and params:
            for name in params[0]:
                self._reads.append(name)
                if name in self._blobs:
                    self._rows.append((name, self._blobs[name]))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


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
    assert directory["version"] == hji._VERSION
    assert sum(cell["count"] for cell in directory["cells"]) == len(rows)
    assert directory["centroids_blob"] in stored

    monkeypatch.setattr("database.get_db", lambda: conn)
    hji.reset_hyperbolic_index()
    assert hji.load_hyperbolic_index() == 1
    index = hji._index_for(None)
    assert index is not None
    assert len(index["cells"]) == len(directory["cells"])
    assert index["centroids"].shape[0] == len(directory["cells"])


def _build_into(monkeypatch, rows, dtype_name):
    stored = {}
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", dtype_name)
    monkeypatch.setattr(
        "tasks.hyperbolic_manager.fetch_all_poincare_rows",
        lambda server_id=None, include_legacy_default=True: dict(rows),
    )
    monkeypatch.setattr(
        "tasks.index_build_helpers.store_segmented_blob",
        lambda conn, table, name, blob, max_part_size_mb=None: stored.__setitem__(name, blob),
    )
    monkeypatch.setattr(hji, "_scan_index_names", lambda: [hji._DEFAULT_SERVER_KEY])
    monkeypatch.setattr(hji, "_resolve_default_server_id", lambda: None)
    hji.build_and_store_hyperbolic_index(_FakeConn())
    directory = json.loads(
        gzip.decompress(stored[hji._dir_name(hji._DEFAULT_SERVER_KEY)]).decode("utf-8")
    )
    return stored, directory


_FLOAT64_ROWS = {
    "a": (np.array([0.90, 0.00, 0.00], dtype=np.float64), 0.90),
    "b": (np.array([0.40, 0.40, 0.00], dtype=np.float64), 0.5657),
    "c": (np.array([0.05, 0.02, 0.01], dtype=np.float64), 0.0548),
}


@pytest.mark.parametrize(
    "configured, expected_name, expected_elem",
    [("f32", "f32", 4), ("f16", "f16", 2), ("i8", "i8", 1)],
)
def test_bands_are_written_in_the_configured_storage_dtype(
    monkeypatch, configured, expected_name, expected_elem
):
    dim = 3
    stored, directory = _build_into(monkeypatch, _FLOAT64_ROWS, configured)

    assert directory["dtype"] == expected_name
    written = 0
    for cell in directory["cells"]:
        blob = stored.get(cell["blob"])
        if cell["count"] == 0:
            assert blob is None
            continue
        assert len(blob) == cell["count"] * dim * expected_elem
        written += cell["count"]
    assert written == len(_FLOAT64_ROWS)


def test_int8_is_taken_literally_and_not_downgraded_to_f16(monkeypatch):
    monkeypatch.setattr(config, "IVF_STORAGE_DTYPE", "i8")
    assert hji._storage_code() == quant.DTYPE_I8


def test_a_quantized_scan_is_reranked_to_the_exact_ordering(monkeypatch):
    rng = np.random.default_rng(0)
    dim = 24
    directions = rng.standard_normal((300, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    vectors = (directions * (0.95 * rng.random((300, 1)) ** (1.0 / dim))).astype(np.float32)
    rows = {
        f"t{i}": (vectors[i], float(np.linalg.norm(vectors[i].astype(np.float64))))
        for i in range(len(vectors))
    }

    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", dim)
    stored, _directory = _build_into(monkeypatch, rows, "i8")

    monkeypatch.setattr(
        "tasks.index_build_helpers.load_segmented_blob",
        lambda conn, table, name: stored.get(name),
    )
    monkeypatch.setattr(
        "tasks.hyperbolic_manager.fetch_poincare_rows",
        lambda ids: {i: rows[i] for i in ids if i in rows},
    )
    monkeypatch.setattr("database.get_db", lambda: _FakeConn(stored))
    hji.reset_hyperbolic_index()
    assert hji.load_hyperbolic_index() == 1

    query = "t7"
    target = rows[query][0].astype(np.float64)
    ids = [i for i in rows if i != query]
    exact = hyperbolic_distances_to(
        target, np.stack([rows[i][0] for i in ids]).astype(np.float64)
    )
    truth = [ids[i] for i in np.argsort(exact)[:10]]

    got = hji.hyperbolic_nearest(target, 10, exclude={query})
    assert [item_id for item_id, _d in got] == truth
    for (item_id, distance), expected in zip(got, np.sort(exact)[:10]):
        assert distance == pytest.approx(float(expected), rel=1e-12)


def test_a_probed_cell_is_read_once_per_query_even_when_the_cache_evicts(monkeypatch):
    rng = np.random.default_rng(3)
    dim = 16
    directions = rng.standard_normal((240, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    vectors = (directions * (0.9 * rng.random((240, 1)) ** (1.0 / dim))).astype(np.float32)
    rows = {
        f"t{i}": (vectors[i], float(np.linalg.norm(vectors[i].astype(np.float64))))
        for i in range(len(vectors))
    }

    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", dim)
    stored, directory = _build_into(monkeypatch, rows, "i8")

    single_reads = []
    monkeypatch.setattr(
        "tasks.index_build_helpers.load_segmented_blob",
        lambda conn, table, name: (single_reads.append(name), stored.get(name))[1],
    )
    monkeypatch.setattr(
        "tasks.hyperbolic_manager.fetch_poincare_rows",
        lambda ids: {i: rows[i] for i in ids if i in rows},
    )
    bulk_reads = []
    monkeypatch.setattr("database.get_db", lambda: _FakeConn(stored, bulk_reads))
    hji.reset_hyperbolic_index()
    assert hji.load_hyperbolic_index() == 1

    monkeypatch.setattr(config, "HYPERBOLIC_INDEX_CACHE_MB", 0)
    hji.reset_hyperbolic_index()
    assert hji.load_hyperbolic_index() == 1

    single_reads.clear()
    bulk_reads.clear()
    got = hji.hyperbolic_nearest(rows["t3"][0].astype(np.float64), 10, exclude={"t3"})

    assert got
    cell_blobs = [cell["blob"] for cell in directory["cells"] if cell["count"]]
    probed = [name for name in bulk_reads if name in cell_blobs]
    assert len(probed) == len(set(probed))
    assert not [name for name in single_reads if name in cell_blobs]
