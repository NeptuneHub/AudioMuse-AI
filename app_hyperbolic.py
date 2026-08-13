# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Hyperbolic Explorer Flask blueprint (hyperbolic_bp).

Serves the ``/hyperbolic`` UI and its APIs, delegating the projection engine
to ``tasks.hyperbolic_manager``. PER SERVER scope: caller-supplied ids are
resolved to canonical ids on input and every response is translated to the
selected server's provider ids on output, so no internal canonical id leaks.

Main Features:
* POST /api/hyperbolic/similar: seed-song hyperbolic similarity re-ranked by
  exact Poincare distance, with similar / roots / niche radial modes.
* GET /api/hyperbolic/tree: served from the in-memory tree cache, lazily
  loaded on demand (see warmup below) and rebuilt on every index-reload
  NOTIFY while warm (like the music map), so a request is a dict lookup,
  never a catalogue scan or a k-means refit. There is no manual rebuild
  endpoint by design - the cache is always kept in sync automatically.
* POST /api/hyperbolic/warmup + GET /api/hyperbolic/warmup/status: the tree
  cache is a fully materialized Python object tree (not disk-paged like the
  other indexes), so it is not loaded at Flask startup. The page calls
  warmup on load so the tree is ready for the first browse, and it
  auto-unloads after HYPERBOLIC_TREE_WARMUP_DURATION idle seconds.
* GET /api/hyperbolic/cache_status: read-only diagnostic for that cache.
* GET /hyperbolic: the two-tab explorer page.
"""

import logging

import config
from flask import Blueprint, jsonify, render_template, request

logger = logging.getLogger(__name__)

hyperbolic_bp = Blueprint("hyperbolic_bp", __name__, template_folder="../templates")


@hyperbolic_bp.route("/hyperbolic", methods=["GET"])
def hyperbolic_page():
    """
    Serves the Hyperbolic Explorer frontend page.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the Hyperbolic Explorer page.
        content:
          text/html:
            schema:
              type: string
    """
    from config import APP_VERSION

    try:
        return render_template(
            "hyperbolic.html",
            title="AudioMuse-AI - Hyperbolic Explorer",
            active="hyperbolic",
            app_version=APP_VERSION,
            hyperbolic_radial_spread_default=min(max(config.HYPERBOLIC_RADIAL_SPREAD, 0.0), 0.99),
        )
    except Exception:
        logger.exception("Error rendering hyperbolic.html")
        return "Hyperbolic Explorer page not implemented yet. Use the API at /api/hyperbolic/similar"


@hyperbolic_bp.route("/api/hyperbolic/similar", methods=["POST"])
def hyperbolic_similar_api():
    """
    Hyperbolic similarity by seed song.
    ---
    tags:
      - Hyperbolic Explorer
    summary: Re-rank candidates by exact Poincare distance. similar re-ranks
      raw-space IVF neighbors; roots / niche draw their pool by radius (at
      least radial_spread, default HYPERBOLIC_RADIAL_SPREAD and
      caller-overridable, of the radial range away from the seed) so they
      visibly move inward / outward instead of hugging the seed's radius
      band.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [item_id]
            properties:
              item_id:
                type: string
                description: Media server item id of the seed song.
              mode:
                type: string
                enum: [similar, roots, niche]
                default: similar
              limit:
                type: integer
                minimum: 1
                maximum: 100
                default: 20
              radial_spread:
                type: number
                format: float
                minimum: 0
                maximum: 0.99
                description: >-
                  Only used by "roots" / "niche" modes. Fraction of the
                  seed's remaining radial range (toward the origin for
                  roots, toward the ball edge for niche) the candidate pool
                  must clear before it is considered - i.e. how far inward
                  or outward the search goes. Defaults to
                  HYPERBOLIC_RADIAL_SPREAD.
    responses:
      200:
        description: Tracks sorted by hyperbolic distance.
        content:
          application/json:
            schema:
              type: object
              properties:
                results:
                  type: array
                  items:
                    type: object
                    properties:
                      item_id:
                        type: string
                      title:
                        type: string
                      author:
                        type: string
                      album:
                        type: string
                      distance:
                        type: number
                        format: float
                      hyperbolic_radius:
                        type: number
                        format: float
                      top_genre:
                        type: string
                count:
                  type: integer
                mode:
                  type: string
                seed_radius:
                  type: number
                  format: float
                  nullable: true
                  description: Radius of the seed in the Poincare ball (mode boundary).
                seed_item_id:
                  type: string
                  description: Provider id of the seed song on the selected server.
                radial_spread:
                  type: number
                  format: float
                  description: The radial_spread value actually used for this search.
      400:
        description: Missing item_id, invalid mode, or seed without a projection.
      500:
        description: Internal error.
    """
    import app_server_context
    from app_helper import attach_song_features
    from tasks.hyperbolic_manager import get_poincare_radius, hyperbolic_similar

    try:
        data = request.get_json() or {}
        item_id = (data.get("item_id") or "").strip()
        if not item_id:
            return jsonify({"error": 'Missing "item_id".'}), 400
        mode = (data.get("mode") or "similar").strip().lower()
        if mode not in ("similar", "roots", "niche"):
            return jsonify({"error": 'Invalid "mode"; use "similar", "roots" or "niche".'}), 400
        try:
            limit = int(data.get("limit", config.HYPERBOLIC_DEFAULT_LIMIT))
        except (TypeError, ValueError):
            return jsonify({"error": 'Invalid "limit" value.'}), 400
        limit = min(max(1, limit), config.HYPERBOLIC_MAX_LIMIT)

        radial_spread = data.get("radial_spread")
        if radial_spread is None:
            radial_spread = config.HYPERBOLIC_RADIAL_SPREAD
        else:
            try:
                radial_spread = float(radial_spread)
            except (TypeError, ValueError):
                return jsonify({"error": 'Invalid "radial_spread" value.'}), 400
            if not (0.0 <= radial_spread <= 0.99):
                return jsonify({"error": '"radial_spread" must be between 0 and 0.99.'}), 400

        canonical_id = app_server_context.resolve_input_item_id(item_id, data)
        results = hyperbolic_similar(canonical_id, mode=mode, limit=limit, radial_spread=radial_spread)
        _attach_title_author(results)
        attach_song_features(results)
        results = app_server_context.scope_results(results, id_key="item_id")
        # Expose the seed's radius so the frontend can draw the mode boundary
        # (roots = inside the seed radius, niche = outside it) on its ball view.
        seed_id_map = app_server_context.translate_ids_for_request([canonical_id])
        return jsonify({
            "results": results,
            "count": len(results),
            "mode": mode,
            "seed_radius": get_poincare_radius(canonical_id),
            "seed_item_id": seed_id_map.get(canonical_id) or canonical_id,
            "radial_spread": radial_spread,
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Hyperbolic similar search failed")
        return jsonify({"error": "An internal error occurred."}), 500


def _attach_title_author(results):
    if not results:
        return
    from app_helper import get_score_data_by_ids

    ids = [r.get("item_id") for r in results if r.get("item_id")]
    details = {d["item_id"]: d for d in get_score_data_by_ids(ids)}
    for r in results:
        info = details.get(r.get("item_id"))
        r["title"] = info.get("title") if info else None
        r["author"] = info.get("author") if info else None


@hyperbolic_bp.route("/api/hyperbolic/tree", methods=["GET"])
def hyperbolic_tree_api():
    """
    Browse the Hyperbolic Explorer directory tree.
    ---
    tags:
      - Hyperbolic Explorer
    summary: Return a node of the mood/genre taxonomy tree; root, mood, genre,
      or k-means fallback nodes render folders, leaf folders render tracks.
    parameters:
      - name: node_id
        in: query
        required: false
        description: >-
          Node id (root, m{mood}, m{mood}.g{genre}, nested genre ids, and
          .c{cluster} ids for the k-means fallback); defaults to root. The
          mood partition and genre depth are derived automatically from the
          mood centroids and each track's mood_vector when the cache is built
          - there is no caller-facing depth parameter.
        schema:
          type: string
    responses:
      200:
        description: A directory node with folder and track children.
        content:
          application/json:
            schema:
              type: object
              properties:
                node:
                  type: object
                  properties:
                    id:
                      type: string
                    name:
                      type: string
                    type:
                      type: string
                      enum: [folder, track]
                    children_count:
                      type: integer
                    items:
                      type: array
      400:
        description: Invalid or unknown node_id.
      500:
        description: Internal error.
    """
    import app_server_context
    from tasks.hyperbolic_manager import (
        build_hyperbolic_tree,
        tree_for_server,
        warmup_hyperbolic_tree_cache,
    )

    try:
        node_id = (request.args.get("node_id") or "").strip() or None

        # Lazy-load (or reset the idle-unload timer on) the tree cache so a
        # browse after an idle unload self-heals instead of returning empty.
        warmup_hyperbolic_tree_cache()

        # The tree is built PER SERVER at analysis time, so a request scoped to
        # a server already sees only that server's genres/subgenres/clusters.
        server_id = app_server_context.resolve_request_server_id()
        node, _flat = build_hyperbolic_tree(node_id, server_id=server_id)
        # Rewrite canonical ids to the selected server's provider ids for the
        # returned node (the tree structure itself is already per-server).
        tree = tree_for_server(server_id)
        tree_nodes = tree.get("nodes") or {}
        tree_flat_ids = tree.get("flat_ids") or {}
        subtree_ids = _tree_subtree_ids(node["id"], tree_nodes, tree_flat_ids)
        mapping = app_server_context.translate_ids_for_request(subtree_ids)
        node = _translate_tree_ids(
            node, mapping,
            tree_nodes=tree_nodes, tree_flat_ids=tree_flat_ids,
            present_ids=set(mapping),
        )
        if node is None:
            node = {
                "id": node_id or "root",
                "name": "Hyperbolic Explorer",
                "type": "folder",
                "children_count": 0,
                "items": [],
            }
        return jsonify({"node": node})

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Hyperbolic tree browse failed")
        return jsonify({"error": "An internal error occurred."}), 500


@hyperbolic_bp.route("/api/hyperbolic/warmup", methods=["POST"])
def hyperbolic_warmup_api():
    """
    Warm up the Hyperbolic Explorer tree cache.
    ---
    tags:
      - Hyperbolic Explorer
    summary: Preload the tree cache and (re)set its idle-eviction timer.
    description: >-
      Call this when the /hyperbolic page loads so the Hierarchy Directory
      tab is ready without a cold-load delay on first browse. The tree is a
      fully materialized Python object tree (not disk-paged like the other
      indexes), so it is not loaded at Flask startup and auto-unloads after
      HYPERBOLIC_TREE_WARMUP_DURATION idle seconds.
    responses:
      200:
        description: Tree cache loaded (or already warm); idle timer reset.
      500:
        description: Warmup failed.
    """
    from tasks.hyperbolic_manager import warmup_hyperbolic_tree_cache

    try:
        return jsonify(warmup_hyperbolic_tree_cache())
    except Exception:
        logger.exception("Hyperbolic tree cache warmup failed")
        return jsonify({"error": "An internal error occurred.", "loaded": False}), 500


@hyperbolic_bp.route("/api/hyperbolic/warmup/status", methods=["GET"])
def hyperbolic_warmup_status_api():
    """
    Hyperbolic Explorer tree cache warm status.
    ---
    tags:
      - Hyperbolic Explorer
    summary: Return whether the tree cache is warm and seconds until idle-unload.
    responses:
      200:
        description: Warm cache state.
    """
    from tasks.hyperbolic_manager import get_hyperbolic_tree_warm_status

    try:
        return jsonify(get_hyperbolic_tree_warm_status())
    except Exception:
        logger.exception("Failed to get Hyperbolic tree cache warmup status")
        return jsonify({"active": False, "seconds_remaining": 0})


@hyperbolic_bp.route("/api/hyperbolic/cache_status", methods=["GET"])
def hyperbolic_cache_status():
    """
    Diagnostic info for the Hyperbolic Explorer tree cache.
    ---
    tags:
      - Hyperbolic Explorer
    summary: Return the cached tree's root folder count, node count and track count.
    responses:
      200:
        description: Cache summary.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                n_bands:
                  type: integer
                  nullable: true
                node_count:
                  type: integer
                track_count:
                  type: integer
                reason:
                  type: string
      500:
        description: Internal error.
    """
    from tasks.hyperbolic_manager import _TREE_CACHE

    try:
        nodes = _TREE_CACHE.get("nodes")
        if not nodes:
            return jsonify({"ok": False, "reason": "empty_cache", "n_bands": None,
                             "node_count": 0, "track_count": 0})
        # n_bands is the number of root folders in the cached tree.
        return jsonify({
            "ok": True,
            "n_bands": _TREE_CACHE.get("n_bands"),
            "node_count": len(nodes),
            "track_count": _TREE_CACHE.get("track_count") or 0,
        }), 200
    except Exception:
        logger.exception("hyperbolic_cache_status failed")
        return jsonify({"ok": False, "reason": "exception", "error": "Internal server error"}), 500


def _tree_subtree_ids(node_id, nodes, flat_ids):
    """All canonical track ids beneath a cached tree node."""
    ids = set()
    seen = set()
    stack = [node_id]
    while stack:
        nid = stack.pop()
        if not nid or nid in seen:
            continue
        seen.add(nid)
        ids.update(flat_ids.get(nid) or [])
        node = nodes.get(nid) or {}
        for child in node.get("items") or []:
            cid = child.get("id")
            if cid:
                stack.append(cid)
    return ids


def _subtree_has_present_track(n, present_ids, tree_nodes, tree_flat_ids):
    stack = [n.get("id")]
    seen = set()
    while stack:
        nid = stack.pop()
        if not nid or nid in seen:
            continue
        seen.add(nid)
        if any(i in present_ids for i in (tree_flat_ids.get(nid) or [])):
            return True
        node = tree_nodes.get(nid) or {}
        for child in node.get("items") or []:
            cid = child.get("id")
            if cid:
                stack.append(cid)
    return False


def _translate_tree_ids(node, mapping, tree_nodes=None, tree_flat_ids=None, present_ids=None):
    """Rewrite canonical ids to the selected server's provider ids.

    Walks the tree and rebuilds each node as a copy (never mutating the shared
    cache). Tracks not present on the request's selected server are dropped.
    Lazy folders - a non-leaf mood or genre folder whose children are cluster
    summaries, or a summary with an empty ``items`` list by design - are kept
    unless per-server info is supplied, in which case a folder with no track on
    the selected server is pruned so switching servers never shows genres or
    subgenres that only exist on another server.
    """

    def walk(n):
        if n.get("type") == "track":
            translated = mapping.get(n["id"])
            if translated is None:
                return None
            return {**n, "id": translated}
        items = n.get("items") or []
        if not items:
            if present_ids is not None and not _subtree_has_present_track(
                n, present_ids, tree_nodes, tree_flat_ids
            ):
                return None
            return {**n}
        kept = []
        for child in items:
            rebuilt = walk(child)
            if rebuilt is not None:
                kept.append(rebuilt)
        if not kept:
            return None
        return {**n, "items": kept, "children_count": len(kept)}

    return walk(node)
