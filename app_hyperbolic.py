# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Hyperbolic Explorer Flask blueprint (hyperbolic_bp).

Serves the ``/hyperbolic`` UI and its two APIs, delegating the projection
engine to ``tasks.hyperbolic_manager``. PER SERVER scope: caller-supplied ids
are resolved to canonical ids on input and every response is translated to the
selected server's provider ids on output, so no internal canonical id leaks.

Main Features:
* POST /api/hyperbolic/similar: seed-song hyperbolic similarity re-ranked by
  exact Poincare distance, with similar / roots / niche radial modes.
* GET /api/hyperbolic/tree: served from the in-memory tree cache built at
  Flask startup and rebuilt on every index-reload NOTIFY (like the music
  map), so a request is a dict lookup, never a catalogue scan or a k-means
  refit. There is no manual rebuild endpoint by design - the cache is always
  kept in sync automatically, at analysis end and at process startup.
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
    summary: Re-rank raw-space candidates by exact Poincare distance, optionally
      filtered inward (roots) or outward (niche) relative to the seed's radius.
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
      400:
        description: Missing item_id, invalid mode, or seed without a projection.
      500:
        description: Internal error.
    """
    import app_server_context
    from app_helper import attach_song_features
    from tasks.hyperbolic_manager import hyperbolic_similar

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

        canonical_id = app_server_context.resolve_input_item_id(item_id, data)
        results = hyperbolic_similar(canonical_id, mode=mode, limit=limit)
        _attach_title_author(results)
        attach_song_features(results)
        results = app_server_context.scope_results(results, id_key="item_id")
        return jsonify({"results": results, "count": len(results), "mode": mode})

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
    summary: Return a node of the radial-band/k-means tree; root, band, or
      cluster nodes render folders, leaf bands render tracks.
    parameters:
      - name: node_id
        in: query
        required: false
        description: >-
          Node id (root, b{band}, b{band}.c{cluster}.c{cluster}...); defaults
          to root. The tree's radial band count and k-means recursion depth
          are both derived automatically from the catalogue size when the
          cache is built - there is no caller-facing depth parameter.
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
    from tasks.hyperbolic_manager import build_hyperbolic_tree

    try:
        node_id = (request.args.get("node_id") or "").strip() or None

        node, flat_ids = build_hyperbolic_tree(node_id)
        mapping = app_server_context.translate_ids_for_request(flat_ids)
        node = _translate_tree_ids(node, mapping)
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


@hyperbolic_bp.route("/api/hyperbolic/cache_status", methods=["GET"])
def hyperbolic_cache_status():
    """
    Diagnostic info for the Hyperbolic Explorer tree cache.
    ---
    tags:
      - Hyperbolic Explorer
    summary: Return the cached tree's root band count, node count and track count.
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
        return jsonify({
            "ok": True,
            "n_bands": _TREE_CACHE.get("n_bands"),
            "node_count": len(nodes),
            "track_count": _TREE_CACHE.get("track_count") or 0,
        }), 200
    except Exception:
        logger.exception("hyperbolic_cache_status failed")
        return jsonify({"ok": False, "reason": "exception", "error": "Internal server error"}), 500


def _translate_tree_ids(node, mapping):
    """Rewrite canonical ids to the selected server's provider ids.

    Walks the tree and rebuilds each node as a copy (never mutating the shared
    cache). Tracks not present on the request's selected server are dropped.
    Only a LEAF folder (whose children are tracks) is pruned when every track
    was dropped; lazy folders - a non-leaf band whose children are cluster
    summaries, or a cluster summary with an empty ``items`` list by design -
    are kept so the root and band levels survive the per-server pass and the
    client can expand them on demand.
    """

    def walk(n):
        if n.get("type") == "track":
            translated = mapping.get(n["id"])
            if translated is None:
                return None
            return {**n, "id": translated}
        items = n.get("items") or []
        if not items:
            return {**n}
        if items[0].get("type") == "track":
            kept = []
            for child in items:
                rebuilt = walk(child)
                if rebuilt is not None:
                    kept.append(rebuilt)
            if not kept:
                return None
            return {**n, "items": kept, "children_count": len(kept)}
        kept = []
        for child in items:
            rebuilt = walk(child)
            if rebuilt is not None:
                kept.append(rebuilt)
        return {**n, "items": kept, "children_count": len(kept)}

    return walk(node)
