# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Semantic and Groove (SemGrove) search Flask blueprint (sem_grove_bp).

Backs the "By Song" tab on the Lyrics Search page, delegating every query to
``tasks.sem_grove_manager`` and its merged lyrics+audio IVF index.

Main Features:
* Similarity search by seed song id, returning sorted tracks with the seed
  itself flagged (``is_seed=true``).
* Endpoints to refresh the merged-index cache and report its stats for the UI.
"""

import logging

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

sem_grove_bp = Blueprint("sem_grove_bp", __name__, template_folder="../templates")


@sem_grove_bp.route("/api/sem_grove/search", methods=["POST"])
def sem_grove_search_api():
    """
    SemGrove similarity search by seed song.
    ---
    tags:
      - SemGrove
    summary: Find songs similar to a seed song using the merged lyrics+audio IVF index.
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
              limit:
                type: integer
                minimum: 1
                maximum: 500
                default: 50
    responses:
      200:
        description: Sorted similar tracks (seed song included with `is_seed=true`).
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
                      similarity:
                        type: number
                        format: float
                      is_seed:
                        type: boolean
                count:
                  type: integer
      400:
        description: Missing or invalid request fields.
      404:
        description: Seed song not in the SemGrove index (requires both lyrics and audio analysis).
      500:
        description: Internal error.
    """
    from tasks.sem_grove_manager import search_by_song
    from app_helper import attach_song_features

    try:
        data = request.get_json() or {}
        item_id = (data.get("item_id") or "").strip()
        if not item_id:
            return jsonify({"error": 'Missing "item_id".'}), 400

        try:
            limit = int(data.get("limit", 50))
        except (TypeError, ValueError):
            return jsonify({"error": 'Invalid "limit" value.'}), 400
        limit = min(max(1, limit), 500)

        import app_server_context

        item_id = app_server_context.resolve_input_item_id(item_id, data)
        results = search_by_song(item_id, limit=app_server_context.overfetch_limit(limit))
        # results[0] is always the seed itself; if that's the only entry, no similar songs were found
        similar_count = sum(1 for r in results if not r.get("is_seed"))
        if not results or similar_count == 0:
            return jsonify(
                {
                    "error": "No similar songs found. "
                    "The song may not be in the SemGrove index yet "
                    "(requires both lyrics and audio analysis).",
                    "results": [],
                }
            ), 404

        results = app_server_context.scope_results(results, limit, id_key='item_id')
        attach_song_features(results)
        return jsonify({"results": results, "count": len(results)})

    except Exception:
        logger.exception("SemGrove search failed")
        return jsonify({"error": "An internal error occurred."}), 500


@sem_grove_bp.route("/api/sem_grove/cache/refresh", methods=["POST"])
def sem_grove_refresh_api():
    """
    Refresh the SemGrove merged index.
    ---
    tags:
      - SemGrove
    summary: Reload the merged lyrics+audio IVF index from the database.
    responses:
      200:
        description: Refresh attempted; updated stats returned.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                stats:
                  type: object
      500:
        description: Internal error.
    """
    from tasks.sem_grove_manager import get_sem_grove_stats, refresh_sem_grove_cache

    try:
        success = refresh_sem_grove_cache()
        return jsonify({"success": success, "stats": get_sem_grove_stats()})
    except Exception:
        logger.exception("SemGrove cache refresh failed")
        return jsonify({"success": False, "error": "Internal error."}), 500


@sem_grove_bp.route("/api/sem_grove/stats", methods=["GET"])
def sem_grove_stats_api():
    """
    SemGrove index stats.
    ---
    tags:
      - SemGrove
    summary: Return size and freshness metadata for the merged SemGrove index.
    responses:
      200:
        description: Index statistics.
        content:
          application/json:
            schema:
              type: object
    """
    from tasks.sem_grove_manager import get_sem_grove_stats

    return jsonify(get_sem_grove_stats())
