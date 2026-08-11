# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the Hyperbolic Explorer manager engine.

Covers scale resolution (config, persisted, or catalog-calibrated), the
similarity endpoint's radial mode filtering and hyperbolic re-ranking, the
backfill job, and the directory-tree node building, all with mocked database
access following the repo's unit-test conventions.

Main Features:
* resolve_hyperbolic_scale prefers config, then persisted, then calibrates
* hyperbolic_similar re-ranks candidates by hyperbolic distance and applies
  roots / niche radial filtering relative to the target radius
* Invalid modes and missing target projections raise ValueError
* backfill_hyperbolic_columns writes projections for every streamed batch
* build_hyperbolic_tree renders root / band / cluster / track nodes with
  quantile bands and deterministic k-means folders
* build_hyperbolic_tree_cache persists the built tree; load_hyperbolic_tree_cache
  restores it from that persisted blob without touching the embedding table
  or refitting any k-means, and init_hyperbolic_cache always loads, never builds
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import config
import tasks.hyperbolic_manager as hm


def _vec(*values):
    return np.array(values, dtype=np.float32)


def _fake_rows(mapping):
    def _fetch(item_ids):
        if not item_ids:
            return {}
        return {iid: mapping[iid] for iid in item_ids if iid in mapping}

    return _fetch


def test_resolve_scale_uses_configured_value(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", 2.5)
    hm.reset_hyperbolic_scale_cache()
    try:
        assert hm.resolve_hyperbolic_scale() == 2.5
    finally:
        hm.reset_hyperbolic_scale_cache()


def test_resolve_scale_uses_persisted_value(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", None)
    with patch("database.get_app_config_value", return_value="1.75"):
        hm.reset_hyperbolic_scale_cache()
        try:
            assert hm.resolve_hyperbolic_scale() == 1.75
        finally:
            hm.reset_hyperbolic_scale_cache()


def test_resolve_scale_calibrates_and_persists(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", None)
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_PERCENTILE", 80.0)
    norms = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [10.0, 0.0]])
    with patch("database.get_app_config_value", return_value=None), \
         patch("database.set_app_config_value") as persist, \
         patch("tasks.index_build_helpers.iter_embedding_batches",
               return_value=[(norms, ["a", "b", "c", "d", "e"])]):
        hm.reset_hyperbolic_scale_cache()
        try:
            scale = hm.resolve_hyperbolic_scale()
        finally:
            hm.reset_hyperbolic_scale_cache()
    assert scale == pytest.approx(float(np.percentile([1, 2, 3, 4, 10], 80.0)), rel=1e-6)
    persist.assert_called_once()


def test_resolve_scale_without_auto_calibrate_does_not_persist(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", None)
    with patch("database.get_app_config_value", return_value=None), \
         patch("database.set_app_config_value") as persist, \
         patch("tasks.index_build_helpers.iter_embedding_batches") as iter_batches:
        hm.reset_hyperbolic_scale_cache()
        try:
            scale = hm.resolve_hyperbolic_scale(auto_calibrate=False)
        finally:
            hm.reset_hyperbolic_scale_cache()
    assert scale is None
    iter_batches.assert_not_called()
    persist.assert_not_called()


def test_compute_projection_without_auto_calibrate_skips_when_unset(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", None)
    with patch("database.get_app_config_value", return_value=None), \
         patch("database.set_app_config_value") as persist, \
         patch("tasks.index_build_helpers.iter_embedding_batches") as iter_batches:
        hm.reset_hyperbolic_scale_cache()
        try:
            proj, radius = hm.compute_hyperbolic_projection(_vec(3.0, -4.0, 0.0), auto_calibrate=False)
        finally:
            hm.reset_hyperbolic_scale_cache()
    assert proj is None
    assert radius is None
    iter_batches.assert_not_called()
    persist.assert_not_called()


def test_compute_projection_without_auto_calibrate_uses_persisted_scale(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", None)
    with patch("database.get_app_config_value", return_value="2.0"):
        hm.reset_hyperbolic_scale_cache()
        try:
            proj, radius = hm.compute_hyperbolic_projection(_vec(3.0, -4.0, 0.0), auto_calibrate=False)
        finally:
            hm.reset_hyperbolic_scale_cache()
    assert proj is not None
    assert radius == pytest.approx(float(np.tanh(5.0 / 2.0)), rel=1e-5)


def test_compute_projection_returns_vector_and_radius(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", 2.0)
    hm.reset_hyperbolic_scale_cache()
    try:
        proj, radius = hm.compute_hyperbolic_projection(_vec(3.0, -4.0, 0.0), scale=2.0)
    finally:
        hm.reset_hyperbolic_scale_cache()
    assert proj.dtype == np.float32
    assert radius == pytest.approx(float(np.tanh(5.0 / 2.0)), rel=1e-5)
    assert float(np.linalg.norm(proj)) == pytest.approx(radius, abs=1e-5)


def test_similar_mode_returns_sorted_by_distance(monkeypatch):
    mapping = {
        "fp_t": (_vec(0.3, 0.1), 0.4),
        "fp_a": (_vec(0.2, 0.05), 0.3),
        "fp_b": (_vec(-0.1, 0.2), 0.5),
        "fp_c": (_vec(0.4, -0.2), 0.35),
    }
    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows(mapping))
    with patch("tasks.ivf_manager.find_nearest_neighbors_by_id",
               return_value=[{"item_id": "fp_c"}, {"item_id": "fp_a"},
                             {"item_id": "fp_b"}, {"item_id": "fp_x"}]):
        results = hm.hyperbolic_similar("fp_t", mode="similar", limit=2)
    assert len(results) == 2
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    assert all("distance" in r and "hyperbolic_radius" in r for r in results)


def test_roots_mode_filters_radius_below_target(monkeypatch):
    mapping = {
        "fp_t": (_vec(0.5, 0.0), 0.6),
        "fp_inner": (_vec(0.1, 0.0), 0.2),
        "fp_outer": (_vec(0.7, 0.0), 0.8),
    }
    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows(mapping))
    with patch("tasks.ivf_manager.find_nearest_neighbors_by_id",
               return_value=[{"item_id": "fp_inner"}, {"item_id": "fp_outer"}]):
        results = hm.hyperbolic_similar("fp_t", mode="roots", limit=20)
    assert [r["item_id"] for r in results] == ["fp_inner"]


def test_niche_mode_filters_radius_above_target(monkeypatch):
    mapping = {
        "fp_t": (_vec(0.5, 0.0), 0.6),
        "fp_inner": (_vec(0.1, 0.0), 0.2),
        "fp_outer": (_vec(0.7, 0.0), 0.8),
    }
    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows(mapping))
    with patch("tasks.ivf_manager.find_nearest_neighbors_by_id",
               return_value=[{"item_id": "fp_inner"}, {"item_id": "fp_outer"}]):
        results = hm.hyperbolic_similar("fp_t", mode="niche", limit=20)
    assert [r["item_id"] for r in results] == ["fp_outer"]


def test_similar_missing_target_projection_raises(monkeypatch):
    monkeypatch.setattr(hm, "_fetch_poincare_rows", lambda ids: {})
    with pytest.raises(ValueError):
        hm.hyperbolic_similar("fp_missing", mode="similar", limit=5)


def test_similar_invalid_mode_raises(monkeypatch):
    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows({"fp_t": (_vec(0.1, 0.0), 0.2)}))
    with pytest.raises(ValueError):
        hm.hyperbolic_similar("fp_t", mode="bogus", limit=5)


def test_backfill_writes_each_batch(monkeypatch):
    batch1 = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    batch2 = np.array([[3.0, 0.0]], dtype=np.float32)
    written = []

    def _upsert(ids, proj, radii):
        written.append((list(ids), proj.copy(), radii.copy()))

    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", 2.0)
    with patch("tasks.index_build_helpers.iter_embedding_batches",
               return_value=[(batch1, ["a", "b"]), (batch2, ["c"])]), \
         patch.object(hm, "_bulk_upsert_hyperbolic", side_effect=_upsert):
        hm.reset_hyperbolic_scale_cache()
        try:
            total = hm.backfill_hyperbolic_columns()
        finally:
            hm.reset_hyperbolic_scale_cache()
    assert total == 3
    assert len(written) == 2
    assert written[0][0] == ["a", "b"]
    assert written[1][0] == ["c"]
    all_batches = np.concatenate([w[1] for w in written])
    all_radii = np.concatenate([w[2] for w in written])
    assert np.all(np.linalg.norm(all_batches, axis=1) < 1.0)
    expected_radii = np.tanh(np.linalg.norm(np.concatenate([batch1, batch2]), axis=1) / 2.0)
    np.testing.assert_allclose(all_radii, expected_radii, rtol=1e-5)


def test_backfill_skips_non_finite_rows(monkeypatch):
    batch = np.array([[1.0, 0.0], [np.nan, 0.0], [2.0, 0.0]], dtype=np.float32)
    written = []

    def _upsert(ids, proj, radii):
        written.append((list(ids), proj.copy(), radii.copy()))

    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", 2.0)
    with patch("tasks.index_build_helpers.iter_embedding_batches",
               return_value=[(batch, ["a", "bad", "c"])]), \
         patch.object(hm, "_bulk_upsert_hyperbolic", side_effect=_upsert):
        hm.reset_hyperbolic_scale_cache()
        try:
            total = hm.backfill_hyperbolic_columns()
        finally:
            hm.reset_hyperbolic_scale_cache()
    assert total == 2
    assert len(written) == 1
    assert written[0][0] == ["a", "c"]


def test_calibrate_scale_ignores_non_finite_norms(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_SCALE", None)
    monkeypatch.setattr(config, "HYPERBOLIC_RADIUS_PERCENTILE", 100.0)
    norms = np.array([[1.0, 0.0], [2.0, 0.0], [np.nan, 0.0], [np.inf, 0.0]])
    with patch("database.get_app_config_value", return_value=None), \
         patch("database.set_app_config_value"), \
         patch("tasks.index_build_helpers.iter_embedding_batches",
               return_value=[(norms, ["a", "b", "bad1", "bad2"])]):
        hm.reset_hyperbolic_scale_cache()
        try:
            scale = hm.resolve_hyperbolic_scale()
        finally:
            hm.reset_hyperbolic_scale_cache()
    assert scale == pytest.approx(2.0, rel=1e-6)


def _make_catalogue(n_per_band=8, bands=3):
    mapping = {}
    for b in range(bands):
        base = 0.2 + 0.25 * b
        for i in range(n_per_band):
            iid = f"fp_{b}_{i}"
            vec = np.array([base + 0.01 * i, 0.02 * i], dtype=np.float32)
            mapping[iid] = (vec, float(np.tanh(np.linalg.norm(vec) / 2.0)))
    return mapping


def _score_row(item_id):
    return {
        "item_id": item_id,
        "title": f"Title {item_id}",
        "author": f"Author {item_id}",
    }


def _rebuild_tree_cache(mapping, n_bands=None, score_rows=None):
    hm.reset_hyperbolic_tree_cache()
    with patch.object(hm, "_fetch_all_poincare_rows", return_value=mapping), \
         patch("app_helper.get_score_data_by_ids", return_value=score_rows or []), \
         patch.object(hm, "_load_projected_mood_centroids", return_value=[]), \
         patch.object(hm, "_persist_tree_cache_blob"):
        return hm.build_hyperbolic_tree_cache(n_bands=n_bands)


def test_tree_root_returns_band_folders(monkeypatch):
    mapping = _make_catalogue()
    try:
        _rebuild_tree_cache(mapping, n_bands=3)
        node, flat = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["type"] == "folder"
    assert node["id"] == "root"
    assert node["children_count"] == len(node["items"])
    assert all(item["type"] == "folder" for item in node["items"])
    assert all(item["id"].startswith("b") for item in node["items"])
    # Non-leaf nodes carry no per-track ids - only the leaf being displayed
    # needs its own members for id translation; ancestors never aggregate them.
    assert flat == []


def test_tree_build_cache_returns_track_count(monkeypatch):
    mapping = _make_catalogue()
    try:
        track_count = _rebuild_tree_cache(mapping, n_bands=3)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert track_count == len(mapping)


def test_tree_band_node_lists_tracks_when_small(monkeypatch):
    mapping = _make_catalogue(n_per_band=5, bands=2)
    try:
        _rebuild_tree_cache(mapping, n_bands=2)
        node, flat = hm.build_hyperbolic_tree("b0")
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["type"] == "folder"
    assert node["id"] == "b0"
    assert node.get("leaf") is True
    assert all(item["type"] == "track" for item in node["items"])
    assert len(flat) == len(node["items"])


def test_tree_band_node_clusters_when_large(monkeypatch):
    mapping = _make_catalogue(n_per_band=200, bands=1)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 60)
    try:
        _rebuild_tree_cache(mapping, n_bands=1)
        node, flat = hm.build_hyperbolic_tree("b0")
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["type"] == "folder"
    assert node.get("leaf") is False
    assert node["items"]
    assert all(item["type"] == "folder" and item["id"].startswith("b0.c")
               for item in node["items"])
    assert all(item["items"] == [] for item in node["items"])
    assert all(item["children_count"] > 0 for item in node["items"])
    assert flat == []


def test_tree_cluster_node_returns_tracks(monkeypatch):
    mapping = _make_catalogue(n_per_band=200, bands=1)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 30)
    try:
        _rebuild_tree_cache(mapping, n_bands=1)
        band, _ = hm.build_hyperbolic_tree("b0")
        cluster_id = band["items"][0]["id"]
        node, flat = hm.build_hyperbolic_tree(cluster_id)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["id"] == cluster_id
    assert node.get("leaf") is True
    assert all(item["type"] == "track" for item in node["items"])
    assert len(flat) == len(node["items"])


def test_tree_repeated_reads_return_the_same_cached_node(monkeypatch):
    mapping = _make_catalogue(n_per_band=200, bands=1)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 30)
    try:
        _rebuild_tree_cache(mapping, n_bands=1)
        band, _ = hm.build_hyperbolic_tree("b0")
        cluster_id = band["items"][0]["id"]
        first = hm.build_hyperbolic_tree(cluster_id)[0]
        second = hm.build_hyperbolic_tree(cluster_id)[0]
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert first is second


def test_tree_empty_catalogue_returns_empty_node(monkeypatch):
    try:
        _rebuild_tree_cache({}, n_bands=3)
        node, flat = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["type"] == "folder"
    assert node["children_count"] == 0
    assert node["items"] == []
    assert flat == []


def test_tree_read_before_cache_built_returns_empty_node(monkeypatch):
    hm.reset_hyperbolic_tree_cache()
    node, flat = hm.build_hyperbolic_tree(None)
    assert node["type"] == "folder"
    assert node["items"] == []
    assert flat == []


def test_tree_unknown_node_raises(monkeypatch):
    mapping = _make_catalogue()
    try:
        _rebuild_tree_cache(mapping, n_bands=3)
        with pytest.raises(ValueError):
            hm.build_hyperbolic_tree("zz9")
    finally:
        hm.reset_hyperbolic_tree_cache()


def test_plan_band_count_scales_up_for_a_large_catalogue(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 150)
    monkeypatch.setattr(config, "HYPERBOLIC_MIN_BANDS", 4)
    monkeypatch.setattr(config, "HYPERBOLIC_MAX_BANDS", 10)
    small = hm._plan_band_count(500)
    large = hm._plan_band_count(200_000)
    assert small >= 4
    assert large > 3
    assert large >= small
    assert 4 <= large <= 10


def test_plan_band_count_clamps_to_configured_bounds(monkeypatch):
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 150)
    monkeypatch.setattr(config, "HYPERBOLIC_MIN_BANDS", 5)
    monkeypatch.setattr(config, "HYPERBOLIC_MAX_BANDS", 7)
    assert hm._plan_band_count(1) == 5
    assert hm._plan_band_count(10_000_000) == 7


def test_tree_root_has_more_than_three_bands_for_a_large_catalogue(monkeypatch):
    mapping = _make_catalogue(n_per_band=60, bands=8)
    monkeypatch.setattr(config, "HYPERBOLIC_MIN_BANDS", 4)
    monkeypatch.setattr(config, "HYPERBOLIC_MAX_BANDS", 10)
    try:
        _rebuild_tree_cache(mapping)  # n_bands=None -> auto-planned
        node, _ = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert len(node["items"]) > 3


def test_tree_recurses_more_than_one_level_for_a_big_band(monkeypatch):
    mapping = _make_catalogue(n_per_band=400, bands=1)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_BRANCHING", 4)
    try:
        _rebuild_tree_cache(mapping, n_bands=1)
        band, _ = hm.build_hyperbolic_tree("b0")
        nodes = hm._TREE_CACHE["nodes"]
        leaf_depths = [
            node_id.count(".c") for node_id, n in nodes.items()
            if n.get("type") == "folder" and n.get("leaf")
        ]
    finally:
        hm.reset_hyperbolic_tree_cache()
    # A band of 400 tracks split by branching=4 gives ~100 tracks per first
    # level cluster - still above the leaf target of 20, so at least one
    # branch must recurse a second time instead of dumping ~100 tracks
    # straight into one folder.
    assert band.get("leaf") is False
    assert leaf_depths
    assert max(leaf_depths) >= 2


def test_tree_leaf_folders_stay_near_target_size(monkeypatch):
    mapping = _make_catalogue(n_per_band=400, bands=1)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_BRANCHING", 4)
    try:
        _rebuild_tree_cache(mapping, n_bands=1)
        nodes = hm._TREE_CACHE["nodes"]
        leaf_sizes = [
            n["summary"]["track_count"] for n in nodes.values()
            if n.get("type") == "folder" and n.get("leaf")
        ]
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert leaf_sizes
    # Generous slack over the target: k-means cluster sizes are never exactly
    # even, and the degenerate-split guard can let one leaf run a bit larger.
    assert all(size <= 20 * 3 for size in leaf_sizes)


def test_materialize_children_bails_out_on_degenerate_clustering(monkeypatch):
    members = [f"t{i}" for i in range(50)]
    vec_map = {m: np.array([1.0, 0.0], dtype=np.float64) for m in members}
    radii_map = {m: 0.5 for m in members}
    monkeypatch.setattr(hm, "_fit_clusters", lambda vecs, k: np.zeros(len(vecs), dtype=int))
    result = hm._materialize_children("b0", members, vec_map, radii_map, {}, [], {}, {}, level=1)
    assert result is None


def _mood_centroid(vec, tags):
    return {"vec": np.array(vec, dtype=np.float64), "tags": tags, "mood": "test"}


def test_cluster_name_blends_top_tag_from_each_of_the_two_nearest_centroids():
    vec_map = {
        "a": np.array([0.50, 0.00], dtype=np.float64),
        "b": np.array([0.50, 0.01], dtype=np.float64),
    }
    mood_centroids = [
        _mood_centroid([0.50, 0.00], ["pop", "electronic", "dance"]),
        _mood_centroid([0.49, 0.00], ["indie", "rock"]),
        _mood_centroid([-0.50, 0.00], ["jazz", "blues"]),
    ]
    name = hm._cluster_name(["a", "b"], vec_map, mood_centroids)
    assert name == "Pop / Indie (2 tracks)"


def test_cluster_name_skips_a_repeated_tag_from_the_second_centroid():
    vec_map = {"a": np.array([0.50, 0.00], dtype=np.float64)}
    mood_centroids = [
        _mood_centroid([0.50, 0.00], ["pop", "electronic"]),
        _mood_centroid([0.49, 0.00], ["pop", "indie"]),
    ]
    name = hm._cluster_name(["a"], vec_map, mood_centroids)
    # The second centroid's top tag ("pop") duplicates the first pick, so its
    # next distinct tag ("indie") is used instead of showing "Pop / Pop".
    assert name == "Pop / Indie (1 tracks)"


def test_cluster_name_falls_back_to_mixed_without_centroids():
    name = hm._cluster_name(["a", "b"], {"a": np.zeros(2), "b": np.zeros(2)}, [])
    assert name == "Mixed (2 tracks)"


def test_track_node_uses_title_and_author():
    score_by_id = {"a": _score_row("a")}
    node = hm._track_node("a", score_by_id)
    assert node["name"] == "Title a - Author a"
    assert node["type"] == "track"


def test_track_node_falls_back_to_id_without_score_data():
    node = hm._track_node("missing", {})
    assert node["name"] == "missing"


def _fake_segmented_store():
    store = {}

    def _store(db_conn, target_table, name, blob, max_part_size_mb=None):
        store[name] = blob

    def _load(db_conn, target_table, name):
        return store.get(name)

    return store, _store, _load


def test_build_tree_cache_raises_when_persist_fails(monkeypatch):
    # A worker-side persist failure must propagate to the caller
    # (_run_all_index_builds' per-step handler records it and the run
    # continues) rather than being swallowed into a log line while the
    # function still reports success - that silent-failure mode is exactly
    # what let a real deployment's tree cache never actually reach the DB.
    mapping = _make_catalogue()
    hm.reset_hyperbolic_tree_cache()
    try:
        with patch.object(hm, "_fetch_all_poincare_rows", return_value=mapping), \
             patch("app_helper.get_score_data_by_ids", return_value=[]), \
             patch.object(hm, "_load_projected_mood_centroids", return_value=[]), \
             patch.object(hm, "_persist_tree_cache_blob", side_effect=RuntimeError("db exploded")):
            with pytest.raises(RuntimeError, match="db exploded"):
                hm.build_hyperbolic_tree_cache(n_bands=3)
    finally:
        hm.reset_hyperbolic_tree_cache()


def test_build_tree_cache_persists_a_loadable_blob(monkeypatch):
    mapping = _make_catalogue()
    store, fake_store, fake_load = _fake_segmented_store()
    hm.reset_hyperbolic_tree_cache()
    try:
        with patch.object(hm, "_fetch_all_poincare_rows", return_value=mapping), \
             patch("app_helper.get_score_data_by_ids", return_value=[]), \
             patch("app_helper.get_db", return_value=MagicMock()), \
             patch("tasks.index_build_helpers.store_segmented_blob", side_effect=fake_store):
            hm.build_hyperbolic_tree_cache(n_bands=3)
        built_node_ids = set(hm._TREE_CACHE["nodes"].keys())
        built_root = hm._TREE_CACHE["nodes"]["root"]
        assert store.get(hm._TREE_CACHE_BLOB_NAME)

        hm.reset_hyperbolic_tree_cache()
        with patch("app_helper.get_db", return_value=MagicMock()), \
             patch("tasks.index_build_helpers.load_segmented_blob", side_effect=fake_load):
            track_count = hm.load_hyperbolic_tree_cache()
        assert track_count == len(mapping)
        assert set(hm._TREE_CACHE["nodes"].keys()) == built_node_ids
        assert hm._TREE_CACHE["nodes"]["root"] == built_root
    finally:
        hm.reset_hyperbolic_tree_cache()


def test_load_tree_cache_never_scans_the_embedding_table_or_reclusters(monkeypatch):
    mapping = _make_catalogue()
    store, fake_store, fake_load = _fake_segmented_store()
    hm.reset_hyperbolic_tree_cache()
    with patch.object(hm, "_fetch_all_poincare_rows", return_value=mapping), \
         patch("app_helper.get_score_data_by_ids", return_value=[]), \
         patch("app_helper.get_db", return_value=MagicMock()), \
         patch("tasks.index_build_helpers.store_segmented_blob", side_effect=fake_store):
        hm.build_hyperbolic_tree_cache(n_bands=3)

    hm.reset_hyperbolic_tree_cache()
    try:
        with patch.object(hm, "_fetch_all_poincare_rows") as fetch_rows, \
             patch.object(hm, "_fit_clusters") as fit_clusters, \
             patch("app_helper.get_db", return_value=MagicMock()), \
             patch("tasks.index_build_helpers.load_segmented_blob", side_effect=fake_load):
            track_count = hm.load_hyperbolic_tree_cache()
    finally:
        hm.reset_hyperbolic_tree_cache()
    fetch_rows.assert_not_called()
    fit_clusters.assert_not_called()
    assert track_count == len(mapping)


def test_load_tree_cache_empty_when_nothing_persisted(monkeypatch):
    hm.reset_hyperbolic_tree_cache()
    try:
        with patch("app_helper.get_db", return_value=MagicMock()), \
             patch("tasks.index_build_helpers.load_segmented_blob", return_value=None):
            track_count = hm.load_hyperbolic_tree_cache()
        node, flat = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert track_count == 0
    assert node["items"] == []
    assert flat == []


def test_init_hyperbolic_cache_loads_not_builds():
    with patch.object(hm, "load_hyperbolic_tree_cache") as load_fn, \
         patch.object(hm, "build_hyperbolic_tree_cache") as build_fn:
        hm.init_hyperbolic_cache()
    load_fn.assert_called_once()
    build_fn.assert_not_called()
