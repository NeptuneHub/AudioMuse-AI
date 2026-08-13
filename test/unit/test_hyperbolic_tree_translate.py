# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the per-server tree id translation.

Covers ``app_hyperbolic._translate_tree_ids``, the pass that rewrites the
cached tree's canonical ids to the selected server's provider ids. The tree is
a shared in-memory cache built from every server's tracks, so translation must
rebuild every node as a copy. With no per-server info supplied a lazy (non-leaf)
folder - a band holding cluster summaries, or a summary with an empty ``items``
list by design - is kept so the root never collapses. When per-server info is
supplied, a lazy folder whose subtree has no track on the selected server is
pruned, so switching servers never shows genres or subgenres that only exist on
another server.

Main Features:
* Track nodes are translated to provider ids
* Leaf folders translate their tracks and prune only when every track is dropped
* Lazy folders (empty items) and non-leaf bands survive without per-server info
* With per-server info, lazy folders with no track on the server are pruned
* The root keeps its bands instead of collapsing to an empty node
* Translation never mutates the shared cached node (returns copies)
"""

from app_hyperbolic import _translate_tree_ids


def _track(item_id):
    return {"id": item_id, "name": item_id, "type": "track", "children_count": 0, "items": []}


def _leaf_band(band_id, track_ids):
    return {
        "id": band_id,
        "name": "Band",
        "type": "folder",
        "leaf": True,
        "children_count": len(track_ids),
        "summary": {"track_count": len(track_ids)},
        "items": [_track(i) for i in track_ids],
    }


def _cluster_summary(cid, track_count):
    return {
        "id": cid,
        "name": "Cluster",
        "type": "folder",
        "children_count": track_count,
        "summary": {"track_count": track_count},
        "items": [],
    }


def _non_leaf_band(band_id, summaries):
    return {
        "id": band_id,
        "name": "Band",
        "type": "folder",
        "leaf": False,
        "children_count": len(summaries),
        "summary": {"track_count": sum(s["children_count"] for s in summaries)},
        "items": summaries,
    }


def _identity_mapping(ids):
    return {i: i for i in ids}


def test_track_is_translated():
    node = _track("fp_1")
    out = _translate_tree_ids(node, {"fp_1": "prov-1"})
    assert out["id"] == "prov-1"
    assert out["type"] == "track"


def test_track_not_on_server_is_dropped():
    node = _track("fp_1")
    assert _translate_tree_ids(node, {}) is None


def test_leaf_band_translates_and_recounts():
    node = _leaf_band("b0", ["fp_a", "fp_b", "fp_c"])
    out = _translate_tree_ids(node, _identity_mapping(["fp_a", "fp_b", "fp_c"]))
    assert out is not None
    assert [c["id"] for c in out["items"]] == ["fp_a", "fp_b", "fp_c"]
    assert out["children_count"] == 3


def test_leaf_band_prunes_tracks_missing_on_server():
    node = _leaf_band("b0", ["fp_a", "fp_b", "fp_c"])
    out = _translate_tree_ids(node, {"fp_a": "prov-a", "fp_c": "prov-c"})
    assert out is not None
    assert [c["id"] for c in out["items"]] == ["prov-a", "prov-c"]
    assert out["children_count"] == 2


def test_leaf_band_pruned_when_all_tracks_dropped():
    node = _leaf_band("b0", ["fp_a", "fp_b"])
    assert _translate_tree_ids(node, {}) is None


def test_lazy_cluster_summary_is_preserved():
    node = _cluster_summary("b0.c0", 12)
    out = _translate_tree_ids(node, {})
    assert out is not None
    assert out["id"] == "b0.c0"
    assert out["children_count"] == 12
    assert out["items"] == []


def test_non_leaf_band_survives_with_its_summaries():
    node = _non_leaf_band("b0", [_cluster_summary("b0.c0", 10), _cluster_summary("b0.c1", 8)])
    out = _translate_tree_ids(node, _identity_mapping([]))
    assert out is not None
    assert [c["id"] for c in out["items"]] == ["b0.c0", "b0.c1"]
    assert out["children_count"] == 2


def test_root_with_non_leaf_bands_does_not_collapse():
    root = {
        "id": "root",
        "name": "Hyperbolic Explorer",
        "type": "folder",
        "children_count": 2,
        "summary": {"track_count": 200},
        "items": [
            _non_leaf_band("b0", [_cluster_summary("b0.c0", 60), _cluster_summary("b0.c1", 60)]),
            _leaf_band("b1", ["fp_x", "fp_y"]),
        ],
    }
    mapping = {"fp_x": "prov-x", "fp_y": "prov-y"}
    out = _translate_tree_ids(root, mapping)
    assert out is not None
    assert out["id"] == "root"
    assert out["children_count"] == 2
    band_ids = [c["id"] for c in out["items"]]
    assert "b0" in band_ids
    assert "b1" in band_ids
    b1 = next(c for c in out["items"] if c["id"] == "b1")
    assert [t["id"] for t in b1["items"]] == ["prov-x", "prov-y"]


def test_translation_does_not_mutate_the_cached_node():
    node = _leaf_band("b0", ["fp_a", "fp_b"])
    original_items = list(node["items"])
    _translate_tree_ids(node, {"fp_a": "prov-a"})
    assert node["items"] == original_items
    assert node["items"][0]["id"] == "fp_a"
    assert node["children_count"] == 2


def _cached_folder(cid, children):
    return {
        "id": cid,
        "name": cid,
        "type": "folder",
        "leaf": False,
        "children_count": len(children),
        "items": children,
    }


def _lazy_summary(cid):
    return {"id": cid, "name": cid, "type": "folder", "children_count": 1, "items": []}


def test_lazy_folder_pruned_when_no_track_on_server():
    tree_nodes = {
        "g0": _cached_folder("g0", [{"id": "g0.c0", "type": "folder", "items": []}]),
        "g0.c0": _cached_folder("g0.c0", []),
        "g1": _cached_folder("g1", [{"id": "g1.c0", "type": "folder", "items": []}]),
        "g1.c0": _cached_folder("g1.c0", []),
    }
    tree_flat_ids = {"g0.c0": ["fp_a", "fp_b"], "g1.c0": ["fp_c"]}
    root = {
        "id": "root",
        "name": "Hyperbolic Explorer",
        "type": "folder",
        "children_count": 2,
        "items": [_lazy_summary("g0"), _lazy_summary("g1")],
    }
    out = _translate_tree_ids(
        root, {"fp_a": "prov-a"},
        tree_nodes=tree_nodes, tree_flat_ids=tree_flat_ids,
        present_ids={"fp_a"},
    )
    assert out is not None
    assert [c["id"] for c in out["items"]] == ["g0"]
    assert out["children_count"] == 1


def test_lazy_folder_kept_when_subtree_present_on_server():
    tree_nodes = {
        "g0": _cached_folder("g0", [{"id": "g0.c0", "type": "folder", "items": []}]),
        "g0.c0": _cached_folder("g0.c0", []),
    }
    tree_flat_ids = {"g0.c0": ["fp_a", "fp_b"]}
    root = {
        "id": "root",
        "name": "Hyperbolic Explorer",
        "type": "folder",
        "children_count": 1,
        "items": [_lazy_summary("g0")],
    }
    out = _translate_tree_ids(
        root, {"fp_a": "prov-a", "fp_b": "prov-b"},
        tree_nodes=tree_nodes, tree_flat_ids=tree_flat_ids,
        present_ids={"fp_a", "fp_b"},
    )
    assert out is not None
    assert [c["id"] for c in out["items"]] == ["g0"]
    assert out["children_count"] == 1


def test_nested_lazy_folder_pruning_propagates_up():
    tree_nodes = {
        "genre": _cached_folder("genre", [{"id": "genre.sub", "type": "folder", "items": []}]),
        "genre.sub": _cached_folder("genre.sub", []),
    }
    tree_flat_ids = {"genre.sub": ["fp_x"]}
    root = {
        "id": "root",
        "name": "Hyperbolic Explorer",
        "type": "folder",
        "children_count": 1,
        "items": [_lazy_summary("genre")],
    }
    # The only genre's only subgenre has no track on the selected server, so the
    # whole root collapses to None; the API turns that into an empty node.
    out = _translate_tree_ids(
        root, {},
        tree_nodes=tree_nodes, tree_flat_ids=tree_flat_ids,
        present_ids=set(),
    )
    assert out is None
