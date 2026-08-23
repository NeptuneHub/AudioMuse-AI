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
* build_hyperbolic_tree renders root / mood / main-genre / second-genre /
  third-genre / cluster / track nodes, with a k-means fallback for groups
  with no further genre metadata
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
    monkeypatch.setattr(hm, "_fetch_all_poincare_rows", lambda: dict(mapping))
    monkeypatch.setattr(hm, "_deduplicate_and_cap_results", lambda results: results)
    results = hm.hyperbolic_similar("fp_t", mode="similar", limit=2)
    assert len(results) == 2
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    assert all("distance" in r and "hyperbolic_radius" in r for r in results)
    assert all(r["item_id"] != "fp_t" for r in results)


def test_roots_mode_filters_radius_below_target(monkeypatch):
    captured = {}

    def _fake_window(bound, below=True, limit=100):
        captured["bound"] = bound
        captured["below"] = below
        captured["limit"] = limit
        return {"fp_inner": (_vec(0.1, 0.0), 0.2), "fp_deep": (_vec(0.02, 0.0), 0.05)}

    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows({"fp_t": (_vec(0.5, 0.0), 0.6)}))
    monkeypatch.setattr(hm, "_fetch_poincare_rows_in_radius", _fake_window)
    monkeypatch.setattr(hm, "_deduplicate_and_cap_results", lambda results: results)
    monkeypatch.setattr(config, "HYPERBOLIC_RADIAL_SPREAD", 0.15)
    results = hm.hyperbolic_similar("fp_t", mode="roots", limit=2)
    assert captured["below"] is True
    assert captured["bound"] == pytest.approx(0.6 * (1.0 - 0.15))
    # both candidates sit inside the window; the nearest (fp_inner, R 0.2) ranks first
    assert [r["item_id"] for r in results] == ["fp_inner", "fp_deep"]


def test_niche_mode_filters_radius_above_target(monkeypatch):
    captured = {}

    def _fake_window(bound, below=True, limit=100):
        captured["bound"] = bound
        captured["below"] = below
        captured["limit"] = limit
        return {"fp_outer": (_vec(0.7, 0.0), 0.8), "fp_edge": (_vec(0.9, 0.0), 0.95)}

    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows({"fp_t": (_vec(0.5, 0.0), 0.6)}))
    monkeypatch.setattr(hm, "_fetch_poincare_rows_in_radius", _fake_window)
    monkeypatch.setattr(hm, "_deduplicate_and_cap_results", lambda results: results)
    monkeypatch.setattr(config, "HYPERBOLIC_RADIAL_SPREAD", 0.15)
    results = hm.hyperbolic_similar("fp_t", mode="niche", limit=2)
    assert captured["below"] is False
    assert captured["bound"] == pytest.approx(0.6 + (1.0 - 0.6) * 0.15)
    # both candidates sit outside the window; the nearest (fp_outer, R 0.8) ranks first
    assert [r["item_id"] for r in results] == ["fp_outer", "fp_edge"]


def test_roots_empty_window_returns_empty(monkeypatch):
    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows({"fp_t": (_vec(0.5, 0.0), 0.6)}))
    monkeypatch.setattr(hm, "_fetch_poincare_rows_in_radius", lambda bound, below=True, limit=100: {})
    results = hm.hyperbolic_similar("fp_t", mode="roots", limit=5)
    assert results == []


def test_roots_spread_clamped_and_zero_keeps_inner_pool(monkeypatch):
    # spread=0 degenerates to the whole inner half (radius < seed), so the
    # closest inward candidate is the one just below the seed's radius.
    captured = {}

    def _fake_window(bound, below=True, limit=100):
        captured["bound"] = bound
        return {}

    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows({"fp_t": (_vec(0.5, 0.0), 0.6)}))
    monkeypatch.setattr(hm, "_fetch_poincare_rows_in_radius", _fake_window)
    monkeypatch.setattr(config, "HYPERBOLIC_RADIAL_SPREAD", 0.0)
    hm.hyperbolic_similar("fp_t", mode="roots", limit=2)
    assert captured["bound"] == pytest.approx(0.6)


def test_roots_mode_caller_spread_overrides_config(monkeypatch):
    captured = {}

    def _fake_window(bound, below=True, limit=100):
        captured["bound"] = bound
        return {}

    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows({"fp_t": (_vec(0.5, 0.0), 0.6)}))
    monkeypatch.setattr(hm, "_fetch_poincare_rows_in_radius", _fake_window)
    monkeypatch.setattr(config, "HYPERBOLIC_RADIAL_SPREAD", 0.15)
    hm.hyperbolic_similar("fp_t", mode="roots", limit=2, radial_spread=0.5)
    # A caller-supplied spread wins over the HYPERBOLIC_RADIAL_SPREAD default.
    assert captured["bound"] == pytest.approx(0.6 * (1.0 - 0.5))


def test_niche_mode_caller_spread_overrides_config(monkeypatch):
    captured = {}

    def _fake_window(bound, below=True, limit=100):
        captured["bound"] = bound
        return {}

    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows({"fp_t": (_vec(0.5, 0.0), 0.6)}))
    monkeypatch.setattr(hm, "_fetch_poincare_rows_in_radius", _fake_window)
    monkeypatch.setattr(config, "HYPERBOLIC_RADIAL_SPREAD", 0.15)
    hm.hyperbolic_similar("fp_t", mode="niche", limit=2, radial_spread=0.5)
    assert captured["bound"] == pytest.approx(0.6 + (1.0 - 0.6) * 0.5)


def test_roots_mode_clamps_out_of_range_caller_spread(monkeypatch):
    captured = {}

    def _fake_window(bound, below=True, limit=100):
        captured["bound"] = bound
        return {}

    monkeypatch.setattr(hm, "_fetch_poincare_rows", _fake_rows({"fp_t": (_vec(0.5, 0.0), 0.6)}))
    monkeypatch.setattr(hm, "_fetch_poincare_rows_in_radius", _fake_window)
    hm.hyperbolic_similar("fp_t", mode="roots", limit=2, radial_spread=5.0)
    assert captured["bound"] == pytest.approx(0.6 * (1.0 - 0.99))


def test_deduplicate_and_cap_results_matches_similar_song_rules(monkeypatch):
    fake_details = [
        {"item_id": "d1", "title": "Same Song", "author": "Artist A"},
        {"item_id": "d2", "title": "same song", "author": "artist a"},  # content dup of d1 (case-insensitive)
        {"item_id": "d3", "title": "Other", "author": "Artist A"},
        {"item_id": "d4", "title": "Another", "author": "Artist A"},
        {"item_id": "d5", "title": "Yet Another", "author": "Artist A"},
        {"item_id": "d6", "title": "Lone", "author": "Artist B"},
        {"item_id": "d7", "title": "No Author", "author": ""},
    ]
    monkeypatch.setattr(
        "database.get_score_data_by_ids",
        lambda ids: [d for d in fake_details if d["item_id"] in ids],
    )
    results = [
        {"item_id": "d1", "distance": 0.1, "hyperbolic_radius": 0.5},
        {"item_id": "d2", "distance": 0.2, "hyperbolic_radius": 0.5},  # duplicate -> dropped
        {"item_id": "d3", "distance": 0.3, "hyperbolic_radius": 0.6},
        {"item_id": "d4", "distance": 0.4, "hyperbolic_radius": 0.6},
        {"item_id": "d5", "distance": 0.5, "hyperbolic_radius": 0.7},  # 4th Artist A -> capped
        {"item_id": "d6", "distance": 0.6, "hyperbolic_radius": 0.7},
        {"item_id": "d7", "distance": 0.7, "hyperbolic_radius": 0.8},  # no author -> skipped
    ]
    out = hm._deduplicate_and_cap_results(results)
    assert [r["item_id"] for r in out] == ["d1", "d3", "d4", "d6"]


def test_deduplicate_and_cap_results_empty_is_noop():
    assert hm._deduplicate_and_cap_results([]) == []


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


def _score_row(item_id, mood_vector=None, other_features=None):
    row = {
        "item_id": item_id,
        "title": f"Title {item_id}",
        "author": f"Author {item_id}",
    }
    if mood_vector is not None:
        row["mood_vector"] = mood_vector
    if other_features is not None:
        row["other_features"] = other_features
    return row


_GENRE_PATTERNS = [
    ("rock", "pop", "indie"),
    ("rock", "pop", "alternative"),
    ("rock", "metal", "punk"),
    ("rock", "metal", "hard rock"),
    ("jazz", "blues", "folk"),
    ("jazz", "blues", "country"),
    ("jazz", "swing", "bebop"),
    ("jazz", "swing", "dixieland"),
]


def _make_mood_catalogue(n_per_mood=8, moods=("happy", "sad", "danceable")):
    mapping = {}
    score_rows = []
    for mi, mood in enumerate(moods):
        base = 0.15 + 0.25 * mi
        for pattern, (g1, g2, g3) in enumerate(_GENRE_PATTERNS):
            for i in range(n_per_mood):
                iid = f"fp_{mood}_{pattern}_{i}"
                vec = np.array([base + 0.004 * pattern, 0.008 * i], dtype=np.float32)
                mapping[iid] = (vec, float(np.tanh(np.linalg.norm(vec) / 2.0)))
                score_rows.append(_score_row(
                    iid,
                    mood_vector=f"{g1}:0.9,{g2}:0.6,{g3}:0.3",
                    other_features=f"{mood}:0.9,other:0.1",
                ))
    return mapping, score_rows


def _mood_centroids_for(moods):
    return [
        {"vec": np.array([0.15 + 0.25 * mi, 0.0], dtype=np.float64),
         "tags": ["pop", "rock"], "mood": mood}
        for mi, mood in enumerate(moods)
    ]


def _rebuild_tree_cache(mapping, score_rows=None, mood_centroids=None,
                        genre_subgenres=None):
    hm.reset_hyperbolic_tree_cache()
    with patch.object(hm, "_tree_build_targets",
                      return_value=[(hm._DEFAULT_SERVER_KEY, None, True)]), \
         patch.object(hm, "_fetch_all_poincare_rows", return_value=mapping), \
         patch("database.get_score_data_by_ids", return_value=score_rows or []), \
         patch.object(hm, "_load_projected_mood_centroids",
                      return_value=mood_centroids if mood_centroids is not None else []), \
         patch.object(hm, "_load_projected_genre_subgenres",
                      return_value=genre_subgenres if genre_subgenres is not None else {}), \
         patch.object(hm, "_persist_tree_cache_blob"):
        return hm.build_hyperbolic_tree_cache()


def test_tree_root_partitions_by_mood(monkeypatch):
    mapping, score_rows = _make_mood_catalogue(moods=("happy", "sad", "danceable"))
    mood_centroids = _mood_centroids_for(("happy", "sad", "danceable"))
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids)
        node, flat = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["type"] == "folder"
    assert node["id"] == "root"
    assert node["children_count"] == len(node["items"])
    assert all(item["type"] == "folder" for item in node["items"])
    assert {item["id"] for item in node["items"]} == {"mhappy", "msad", "mdanceable"}
    assert all(item.get("kind") == "mood" for item in node["items"])
    # Non-leaf nodes carry no per-track ids - only the leaf being displayed
    # needs its own members for id translation; ancestors never aggregate them.
    assert flat == []


def test_tree_build_cache_returns_track_count(monkeypatch):
    mapping = _make_catalogue()
    try:
        track_count = _rebuild_tree_cache(mapping)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert track_count == len(mapping)


def test_tree_cache_builds_separate_trees_per_server(monkeypatch):
    # The default server and each secondary server get their OWN tree, built
    # from only that server's tracks (the caller feeds per-server rows via
    # _fetch_all_poincare_rows(server_id=...)), persisted under distinct blob
    # names, and the request path resolves the right tree per server.
    default_mapping = _make_catalogue(n_per_band=200, bands=1)
    sec_mapping = _make_catalogue(n_per_band=20, bands=1)
    # Make the secondary set distinguishable: fewer bands -> different cluster
    # structure than the default tree.
    sec_mapping = {f"sec_{iid}": row for iid, row in sec_mapping.items()}
    persisted = {}

    def _fake_fetch(server_id=None, include_legacy_default=True):
        if server_id == "sec":
            return sec_mapping
        return default_mapping

    def _fake_persist(payload, name=None):
        persisted[name] = payload

    hm.reset_hyperbolic_tree_cache()
    try:
        with patch.object(
                hm, "_tree_build_targets",
                return_value=[(hm._DEFAULT_SERVER_KEY, None, True), ("sec", "sec", False)]), \
             patch.object(hm, "_fetch_all_poincare_rows", side_effect=_fake_fetch), \
             patch("database.get_score_data_by_ids", return_value=[]), \
             patch.object(hm, "_load_projected_mood_centroids", return_value=[]), \
             patch.object(hm, "_load_projected_genre_subgenres", return_value={}), \
             patch.object(hm, "_persist_tree_cache_blob", side_effect=_fake_persist):
            default_count = hm.build_hyperbolic_tree_cache()

        # The default tree is mirrored into the legacy top-level fields; the
        # secondary tree lives only under its own server key.
        assert default_count == len(default_mapping)
        assert hm._TREE_CACHE["servers"][hm._DEFAULT_SERVER_KEY]["track_count"] == len(default_mapping)
        assert hm._TREE_CACHE["servers"]["sec"]["track_count"] == len(sec_mapping)
        assert hm._TREE_CACHE["nodes"]["root"]["summary"]["track_count"] == len(default_mapping)
        # Distinct blob names per server.
        assert hm._blob_name_for(hm._DEFAULT_SERVER_KEY) == hm._TREE_CACHE_BLOB_NAME
        assert hm._blob_name_for("sec") == f"{hm._TREE_CACHE_BLOB_NAME}__sec"
        assert set(persisted) == {
            hm._TREE_CACHE_BLOB_NAME, f"{hm._TREE_CACHE_BLOB_NAME}__sec",
            hm._TREE_SKELETON_BLOB_NAME, f"{hm._TREE_SKELETON_BLOB_NAME}__sec",
        }
        assert set(persisted[hm._TREE_SKELETON_BLOB_NAME]["nodes"]) == {
            nid for nid, n in persisted[hm._TREE_CACHE_BLOB_NAME]["nodes"].items()
            if not n.get("leaf")
        }
        # The request path resolves the right tree per server.
        assert hm.tree_for_server(None)["track_count"] == len(default_mapping)
        assert hm.tree_for_server("sec")["track_count"] == len(sec_mapping)
        assert hm.tree_for_server("sec")["nodes"]["root"]["summary"]["track_count"] == len(sec_mapping)
        # A server with no tree of its own (e.g. added after the last analysis
        # run) must never fall back to the default server's tree - that would
        # leak the default server's genres/subgenres under another server's
        # selection. tree_for_server reports no tree, and build_hyperbolic_tree
        # raises a clear "not available yet" error instead of silently
        # rendering an empty folder the user could mistake for a real result.
        assert hm.tree_for_server("third") == {}
        with pytest.raises(ValueError, match="not available"):
            hm.build_hyperbolic_tree(None, server_id="third")
    finally:
        hm.reset_hyperbolic_tree_cache()


def test_tree_for_server_resolves_default_by_its_real_server_id(monkeypatch):
    # The default tree is stored under the sentinel _DEFAULT_SERVER_KEY, but a
    # request can legitimately pass the default server's real id (a client
    # explicitly selecting "the server that happens to be default"). That must
    # still resolve to the default tree, not to the untreed-server empty case.
    default_mapping = _make_catalogue(n_per_band=20, bands=1)
    hm.reset_hyperbolic_tree_cache()
    try:
        with patch.object(
                hm, "_tree_build_targets",
                return_value=[(hm._DEFAULT_SERVER_KEY, "def-real", True)]), \
             patch.object(hm, "_fetch_all_poincare_rows", return_value=default_mapping), \
             patch("database.get_score_data_by_ids", return_value=[]), \
             patch.object(hm, "_load_projected_mood_centroids", return_value=[]), \
             patch.object(hm, "_load_projected_genre_subgenres", return_value={}), \
             patch.object(hm, "_persist_tree_cache_blob"):
            hm.build_hyperbolic_tree_cache()

        assert hm.tree_for_server("def-real")["track_count"] == len(default_mapping)
        assert hm.tree_for_server("def-real")["nodes"] == hm.tree_for_server(None)["nodes"]
    finally:
        hm.reset_hyperbolic_tree_cache()


def test_tree_mood_node_lists_tracks_when_small(monkeypatch):
    mapping, score_rows = _make_mood_catalogue(n_per_mood=2, moods=("happy",))
    mood_centroids = _mood_centroids_for(("happy",))
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids)
        node, flat = hm.build_hyperbolic_tree("mhappy")
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["type"] == "folder"
    assert node["id"] == "mhappy"
    assert node.get("leaf") is True
    assert all(item["type"] == "track" for item in node["items"])
    assert len(flat) == len(node["items"])


def test_tree_mood_node_splits_by_main_genre_when_large(monkeypatch):
    mapping, score_rows = _make_mood_catalogue(n_per_mood=100, moods=("happy",))
    mood_centroids = _mood_centroids_for(("happy",))
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 60)
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids)
        node, flat = hm.build_hyperbolic_tree("mhappy")
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["type"] == "folder"
    assert node.get("leaf") is False
    assert node["items"]
    assert all(item["type"] == "folder" and item["id"].startswith("mhappy.g")
               for item in node["items"])
    assert all(item.get("kind") == "main_genre" for item in node["items"])
    assert all(item["items"] == [] for item in node["items"])
    assert all(item["children_count"] > 0 for item in node["items"])
    assert flat == []


def test_tree_cluster_node_returns_tracks(monkeypatch):
    mapping = _make_catalogue(n_per_band=200, bands=1)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 30)
    try:
        _rebuild_tree_cache(mapping)
        mood, _ = hm.build_hyperbolic_tree("mgeneral")
        cluster_id = mood["items"][0]["id"]
        node, flat = hm.build_hyperbolic_tree(cluster_id)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert node["id"] == cluster_id
    assert node.get("leaf") is True
    assert all(item["type"] == "track" for item in node["items"])
    assert len(flat) == len(node["items"])


def test_tree_falls_back_to_general_mood_without_centroids(monkeypatch):
    mapping = _make_catalogue()
    try:
        _rebuild_tree_cache(mapping)
        node, _ = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert len(node["items"]) == 1
    assert node["items"][0]["id"] == "mgeneral"
    assert node["items"][0]["name"] == "General"


def test_tree_mood_falls_back_to_other_features_without_centroids(monkeypatch):
    mapping = {}
    score_rows = []
    for i in range(10):
        iid = f"fp_{i}"
        mapping[iid] = (np.array([0.3, 0.0], dtype=np.float32), 0.3)
        mood = "happy" if i < 6 else "sad"
        score_rows.append(_score_row(iid, other_features=f"{mood}:0.9,other:0.1"))
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows)
        node, _ = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    ids = [item["id"] for item in node["items"]]
    assert "mhappy" in ids
    assert "msad" in ids


def test_tree_repeated_reads_return_the_same_cached_node(monkeypatch):
    mapping = _make_catalogue(n_per_band=200, bands=1)
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 30)
    try:
        _rebuild_tree_cache(mapping)
        mood, _ = hm.build_hyperbolic_tree("mgeneral")
        cluster_id = mood["items"][0]["id"]
        first = hm.build_hyperbolic_tree(cluster_id)[0]
        second = hm.build_hyperbolic_tree(cluster_id)[0]
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert first is second


def test_tree_empty_catalogue_returns_empty_node(monkeypatch):
    try:
        _rebuild_tree_cache({})
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
        _rebuild_tree_cache(mapping)
        with pytest.raises(ValueError):
            hm.build_hyperbolic_tree("zz9")
    finally:
        hm.reset_hyperbolic_tree_cache()


def test_tree_root_has_all_six_moods_for_a_large_catalogue(monkeypatch):
    moods = ("happy", "sad", "danceable", "party", "relaxed", "aggressive")
    mapping, score_rows = _make_mood_catalogue(n_per_mood=8, moods=moods)
    mood_centroids = _mood_centroids_for(moods)
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids)
        node, _ = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert {item["id"] for item in node["items"]} == {f"m{m}" for m in moods}


def test_tree_genre_nesting_stops_at_named_clusters(monkeypatch):
    mapping, score_rows = _make_mood_catalogue(n_per_mood=50, moods=("happy",))
    mood_centroids = _mood_centroids_for(("happy",))
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids)
        mood, _ = hm.build_hyperbolic_tree("mhappy")
        nodes = hm._TREE_CACHE["nodes"]
        kinds = {n.get("kind") for n in nodes.values() if n.get("type") == "folder"}
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert mood.get("leaf") is False
    # The genre-less fallback is mood -> main genre -> named clusters: the
    # old second/third genre levels are gone.
    assert "main_genre" in kinds
    assert "second_genre" not in kinds
    assert "third_genre" not in kinds


def test_tree_root_is_genre_and_splits_into_subgenres(monkeypatch):
    mapping, score_rows = _make_mood_catalogue(n_per_mood=50, moods=("happy",))
    mood_centroids = _mood_centroids_for(("happy",))
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    genre_subgenres = {
        "rock": {
            "vec": np.array([0.156, 0.0]),
            "subgenres": [
                {"name": "pop", "vec": np.array([0.152, 0.0])},
                {"name": "metal", "vec": np.array([0.160, 0.0])},
            ],
        },
        "jazz": {
            "vec": np.array([0.172, 0.0]),
            "subgenres": [
                {"name": "blues", "vec": np.array([0.168, 0.0])},
                {"name": "swing", "vec": np.array([0.176, 0.0])},
            ],
        },
    }
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids,
                            genre_subgenres=genre_subgenres)
        root, _ = hm.build_hyperbolic_tree(None)
        root_ids = {item["id"] for item in root["items"]}
        nodes = hm._TREE_CACHE["nodes"]
        kinds = {n.get("kind") for n in nodes.values() if n.get("type") == "folder"}
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert root["type"] == "folder"
    assert root_ids == {"root.grock", "root.gjazz"}
    assert all(item.get("kind") == "main_genre" for item in root["items"])
    assert "subgenre" in kinds
    assert "mood" not in kinds


def test_tree_genre_root_lists_tracks_directly_when_subgenres_cannot_cluster(monkeypatch):
    # On a small library no subgenre can form a cluster of HYPERBOLIC_MIN_CLUSTER_SIZE
    # or more, so the strict path used to hide every genre and silently fall back to
    # the mood root. A genre whose subgenres all vanished must instead list its
    # tracks directly under the genre, keeping the genre root (not mood).
    mapping, score_rows = _make_mood_catalogue(n_per_mood=2, moods=("happy",))
    mood_centroids = _mood_centroids_for(("happy",))
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    genre_subgenres = {
        "rock": {
            "vec": np.array([0.156, 0.0]),
            "subgenres": [
                {"name": "pop", "vec": np.array([0.152, 0.0])},
                {"name": "metal", "vec": np.array([0.160, 0.0])},
            ],
        },
        "jazz": {
            "vec": np.array([0.172, 0.0]),
            "subgenres": [
                {"name": "blues", "vec": np.array([0.168, 0.0])},
                {"name": "swing", "vec": np.array([0.176, 0.0])},
            ],
        },
    }
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids,
                            genre_subgenres=genre_subgenres)
        root, _ = hm.build_hyperbolic_tree(None)
        root_ids = {item["id"] for item in root["items"]}
        nodes = hm._TREE_CACHE["nodes"]
        kinds = {n.get("kind") for n in nodes.values() if n.get("type") == "folder"}
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert root_ids == {"root.grock", "root.gjazz"}
    assert all(item.get("kind") == "main_genre" for item in root["items"])
    assert all(item.get("leaf") is True for item in root["items"])
    assert "subgenre" not in kinds
    assert "mood" not in kinds


def test_tree_genre_centroids_fall_back_when_dims_differ(monkeypatch):
    mapping, score_rows = _make_mood_catalogue(n_per_mood=50, moods=("happy",))
    mood_centroids = _mood_centroids_for(("happy",))
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    genre_subgenres = {
        "rock": {
            "vec": np.zeros(200),
            "subgenres": [
                {"name": "pop", "vec": np.zeros(200)},
                {"name": "metal", "vec": np.zeros(200)},
            ],
        },
    }
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids,
                            genre_subgenres=genre_subgenres)
        mood, _ = hm.build_hyperbolic_tree("mhappy")
        nodes = hm._TREE_CACHE["nodes"]
        kinds = {n.get("kind") for n in nodes.values() if n.get("type") == "folder"}
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert "main_genre" in kinds
    assert "second_genre" not in kinds
    assert "third_genre" not in kinds


def test_tree_leaf_folders_stay_near_target_size(monkeypatch):
    mapping, score_rows = _make_mood_catalogue(n_per_mood=50, moods=("happy",))
    mood_centroids = _mood_centroids_for(("happy",))
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    try:
        _rebuild_tree_cache(mapping, score_rows=score_rows, mood_centroids=mood_centroids)
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
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    monkeypatch.setattr(hm, "_fit_clusters", lambda vecs, k: np.zeros(len(vecs), dtype=int))
    result = hm._materialize_children("b0", members, vec_map, radii_map, {}, [], {}, {})
    assert result is None


def test_genre_path_prefix_builds_ancestor_chain():
    assert hm._genre_path_prefix("root.grock.gprogressive-rock.c0") == "ROCK_PROGRESSIVE_ROCK"
    assert hm._genre_path_prefix("root.gpop") == "POP"
    assert hm._genre_path_prefix("root.c0") == ""


def test_cluster_descriptor_prefers_dominant_clap_mood():
    score_by_id = {
        "a": {"other_features": "happy:0.9,party:0.1", "mood_vector": "pop:0.5,female vocalists:0.9"},
        "b": {"other_features": "happy:0.8,party:0.2", "mood_vector": "pop:0.5"},
        "c": {"other_features": "happy:0.7,sad:0.1", "mood_vector": "rock:0.5"},
    }
    assert hm._cluster_descriptor(["a", "b", "c"], score_by_id) == "HAPPY"


def test_cluster_descriptor_voice_not_used_when_mood_is_confident():
    score_by_id = {
        "a": {"other_features": "happy:0.5", "mood_vector": "female vocalists:0.9,pop:0.5"},
        "b": {"other_features": "happy:0.5", "mood_vector": "female vocalists:0.8,pop:0.5"},
        "c": {"other_features": "happy:0.5", "mood_vector": "female vocalists:0.8,pop:0.5"},
    }
    assert hm._cluster_descriptor(["a", "b", "c"], score_by_id) == "HAPPY"


def test_cluster_descriptor_voice_only_when_no_confident_mood():
    score_by_id = {
        "a": {"other_features": "happy:0.4,sad:0.4", "mood_vector": "female vocalists:0.9,pop:0.5"},
        "b": {"other_features": "happy:0.4,sad:0.4", "mood_vector": "female vocalists:0.8,pop:0.5"},
        "c": {"other_features": "happy:0.4,sad:0.4", "mood_vector": "female vocalists:0.8,pop:0.5"},
    }
    assert hm._cluster_descriptor(["a", "b", "c"], score_by_id) == "FEMALE_VOCALISTS"


def test_cluster_descriptor_none_when_not_confident():
    score_by_id = {
        "a": {"other_features": "happy:0.5,sad:0.5", "mood_vector": "rock:0.5"},
        "b": {"other_features": "sad:0.5,happy:0.5", "mood_vector": "rock:0.5"},
        "c": {"other_features": "happy:0.5,sad:0.5", "mood_vector": "rock:0.5"},
    }
    assert hm._cluster_descriptor(["a", "b", "c"], score_by_id) is None


def test_genre_cluster_names_use_ancestor_prefix_and_dedupe(monkeypatch):
    members = [f"t{i}" for i in range(60)]
    vec_map = {m: np.array([0.01 * i, 0.0], dtype=np.float64) for i, m in enumerate(members)}
    radii_map = {m: 0.5 for m in members}
    score_by_id = {
        m: {"other_features": "happy:0.9", "mood_vector": "rock:0.5"}
        for m in members
    }
    monkeypatch.setattr(config, "HYPERBOLIC_TARGET_LEAF_SIZE", 20)
    monkeypatch.setattr(
        hm, "_fit_clusters",
        lambda vecs, k: np.array([i % k for i in range(len(vecs))], dtype=int),
    )
    nodes = {}
    flat_ids = {}
    children = hm._materialize_children(
        "root.grock.gprogressive-rock", members, vec_map, radii_map,
        score_by_id, [], nodes, flat_ids,
        name_prefix="ROCK_PROGRESSIVE_ROCK",
    )
    assert children
    names = [c["name"] for c in children]
    assert all(n.startswith("ROCK_PROGRESSIVE_ROCK") for n in names)
    assert all(" / " not in n for n in names)
    assert len(names) == len(set(names))


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
             patch("database.get_score_data_by_ids", return_value=[]), \
             patch.object(hm, "_load_projected_mood_centroids", return_value=[]), \
             patch.object(hm, "_persist_tree_cache_blob", side_effect=RuntimeError("db exploded")):
            with pytest.raises(RuntimeError, match="db exploded"):
                hm.build_hyperbolic_tree_cache()
    finally:
        hm.reset_hyperbolic_tree_cache()


def test_build_tree_cache_persists_a_loadable_blob(monkeypatch):
    mapping = _make_catalogue()
    store, fake_store, fake_load = _fake_segmented_store()
    hm.reset_hyperbolic_tree_cache()
    try:
        with patch.object(hm, "_fetch_all_poincare_rows", return_value=mapping), \
             patch("database.get_score_data_by_ids", return_value=[]), \
             patch.object(hm, "_load_projected_mood_centroids", return_value=[]), \
             patch.object(hm, "_load_projected_genre_subgenres", return_value={}), \
             patch("database.get_db", return_value=MagicMock()), \
             patch("tasks.index_build_helpers.store_segmented_blob", side_effect=fake_store):
            hm.build_hyperbolic_tree_cache()
        built_node_ids = set(hm._TREE_CACHE["nodes"].keys())
        built_root = hm._TREE_CACHE["nodes"]["root"]
        assert store.get(hm._TREE_CACHE_BLOB_NAME)

        hm.reset_hyperbolic_tree_cache()
        with patch("database.get_db", return_value=MagicMock()), \
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
         patch("database.get_score_data_by_ids", return_value=[]), \
         patch.object(hm, "_load_projected_mood_centroids", return_value=[]), \
         patch.object(hm, "_load_projected_genre_subgenres", return_value={}), \
         patch("database.get_db", return_value=MagicMock()), \
         patch("tasks.index_build_helpers.store_segmented_blob", side_effect=fake_store):
        hm.build_hyperbolic_tree_cache()

    hm.reset_hyperbolic_tree_cache()
    try:
        with patch.object(hm, "_fetch_all_poincare_rows") as fetch_rows, \
             patch.object(hm, "_fit_clusters") as fit_clusters, \
             patch("database.get_db", return_value=MagicMock()), \
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
        with patch("database.get_db", return_value=MagicMock()), \
             patch("tasks.index_build_helpers.load_segmented_blob", return_value=None):
            track_count = hm.load_hyperbolic_tree_cache()
        node, flat = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert track_count == 0
    assert node["items"] == []
    assert flat == []


def test_load_tree_cache_discards_an_old_schema_blob(monkeypatch):
    # A blob written by a previous tree schema (no "version" or an older one)
    # must be discarded, not served: after an upgrade Flask would otherwise
    # keep showing the stale pre-upgrade tree until the next analysis run.
    import gzip
    import json as _json

    stale = gzip.compress(_json.dumps({
        "n_bands": 3, "nodes": {"root": {"id": "root"}},
        "flat_ids": {}, "track_count": 42,
    }).encode("utf-8"))
    deleted = {"called": False}

    def _fake_delete():
        deleted["called"] = True

    hm.reset_hyperbolic_tree_cache()
    try:
        with patch("database.get_db", return_value=MagicMock()), \
             patch("tasks.index_build_helpers.load_segmented_blob", return_value=stale), \
             patch.object(hm, "_delete_tree_cache_blob", side_effect=_fake_delete):
            track_count = hm.load_hyperbolic_tree_cache()
        node, flat = hm.build_hyperbolic_tree(None)
    finally:
        hm.reset_hyperbolic_tree_cache()
    assert track_count == 0
    assert node["items"] == []
    assert deleted["called"] is True


def test_init_hyperbolic_cache_loads_not_builds():
    with patch.object(hm, "load_hyperbolic_tree_cache") as load_fn, \
         patch.object(hm, "build_hyperbolic_tree_cache") as build_fn:
        hm.init_hyperbolic_cache()
    load_fn.assert_called_once()
    build_fn.assert_not_called()
