"""
Semantic & Groove (SemGrove) Search Blueprint

Provides the API for the "By Song" tab in the Lyrics Search page.
Uses the merged lyrics+audio Voyager index built by tasks/sem_grove_manager.py.
"""

import logging

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

sem_grove_bp = Blueprint("sem_grove_bp", __name__, template_folder="../templates")


@sem_grove_bp.route("/api/sem_grove/search", methods=["POST"])
def sem_grove_search_api():
    """Find songs similar to a seed song using the merged lyrics+audio index.

    POST JSON:
    {
        "item_id": "<media_server_item_id>",
        "limit": 50
    }
    Returns a list of {item_id, title, author, similarity} sorted by
    descending merged-cosine similarity.
    """
    from tasks.sem_grove_manager import search_by_song

    try:
        data    = request.get_json() or {}
        item_id = (data.get("item_id") or "").strip()
        if not item_id:
            return jsonify({"error": 'Missing "item_id".'}), 400

        try:
            limit = int(data.get("limit", 50))
        except (TypeError, ValueError):
            return jsonify({"error": 'Invalid "limit" value.'}), 400
        limit = min(max(1, limit), 500)

        results = search_by_song(item_id, limit=limit)
        # results[0] is always the seed itself; if that's the only entry, no similar songs were found
        similar_count = sum(1 for r in results if not r.get("is_seed"))
        if not results or similar_count == 0:
            return jsonify({
                "error": "No similar songs found. "
                         "The song may not be in the SemGrove index yet "
                         "(requires both lyrics and audio analysis).",
                "results": [],
            }), 404

        return jsonify({"results": results, "count": len(results)})

    except Exception:
        logger.exception("SemGrove search failed")
        return jsonify({"error": "An internal error occurred."}), 500


@sem_grove_bp.route("/api/sem_grove/cache/preload", methods=["POST"])
def sem_grove_preload_api():
    """Schedule a background preload of the SemGrove merged index."""
    from tasks.sem_grove_manager import (
        load_sem_grove_cache_from_db, is_sem_grove_cache_loaded,
        get_sem_grove_stats, _SEM_GROVE_IDLE,
    )
    from tasks._preload_queue import PRELOAD_QUEUE
    import config as _cfg

    def _touch():
        try:
            _SEM_GROVE_IDLE.set_idle_seconds(int(getattr(
                _cfg, 'SEM_GROVE_INDEX_IDLE_SECONDS', 300)))
        except Exception:
            pass
        _SEM_GROVE_IDLE.touch()

    if is_sem_grove_cache_loaded():
        _touch()
        return jsonify({'queued': False, 'reason': 'already_loaded',
                        'stats': get_sem_grove_stats()})

    def _do_load():
        if not is_sem_grove_cache_loaded():
            load_sem_grove_cache_from_db()
        if is_sem_grove_cache_loaded():
            _touch()

    queued = PRELOAD_QUEUE.enqueue('sem_grove', _do_load)
    return jsonify({'queued': queued})


@sem_grove_bp.route("/api/sem_grove/cache/refresh", methods=["POST"])
def sem_grove_refresh_api():
    """Reload the SemGrove index from the database (hot-reload)."""
    from tasks.sem_grove_manager import get_sem_grove_stats, refresh_sem_grove_cache

    try:
        success = refresh_sem_grove_cache()
        return jsonify({"success": success, "stats": get_sem_grove_stats()})
    except Exception:
        logger.exception("SemGrove cache refresh failed")
        return jsonify({"success": False, "error": "Internal error."}), 500


@sem_grove_bp.route("/api/sem_grove/stats", methods=["GET"])
def sem_grove_stats_api():
    """Return SemGrove index stats."""
    from tasks.sem_grove_manager import get_sem_grove_stats

    return jsonify(get_sem_grove_stats())
