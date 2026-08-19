# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for the external track-lookup API (mounted at `/external`).

Read-only endpoints that expose stored analysis for third-party integrations,
returning per-track scores/embeddings and running unified similarity search via
`tasks.ivf_manager.search_tracks_unified`.

Main Features:
* Routes: `/get_score`, `/get_embedding` (by item id) and `/search` (similarity).
* Imports `get_db` lazily inside each handler to avoid a circular import.
"""

from flask import Blueprint, jsonify, request
from psycopg2.extras import DictCursor
import numpy as np
import logging

from app_logging import sanitize_log_value
# Import ivf_manager functions for track lookups
from tasks.ivf_manager import search_tracks_unified
from error import error_manager
from error.error_dictionary import ERR_DB_QUERY
# NOTE: The import of 'get_db' has been moved inside each function to prevent circular imports.

logger = logging.getLogger(__name__)

# Create a Blueprint for external API routes
external_bp = Blueprint('external_bp', __name__)


def _resolve_external_id(raw_id):
    """Resolve a caller-supplied track id to the canonical catalogue id.

    External callers (media-server plugins, scripts) send THEIR server's track
    id; the optional ``server`` parameter (unique display name or internal id)
    says which server that is - default when absent. The shared input resolver
    guarantees canonical or unknown ids pass through unchanged on EVERY server,
    so external endpoints resolve exactly like internal ones. Raises ValueError
    on an unknown server.
    """
    from app_server_context import resolve_input_item_id

    return resolve_input_item_id(raw_id)


def _decode_embedding_bytes(payload):
    if payload.get('embedding'):
        payload['embedding'] = np.frombuffer(payload['embedding'], dtype=np.float32).tolist()


def _external_row_response(sql, label, decode=None):
    # Local import to prevent circular dependency
    from database import get_db

    raw_id = request.args.get('id')
    if not raw_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400
    safe_raw_id = sanitize_log_value(raw_id)
    try:
        try:
            item_id = _resolve_external_id(raw_id)
        except ValueError:
            # The resolver's own text can carry internal detail, so it goes to
            # the log and the caller gets the parameter that was wrong.
            logger.warning("Could not resolve external id %s", safe_raw_id, exc_info=True)
            return jsonify({"error": "Unknown or invalid 'server' parameter"}), 400
        row = None
        if item_id:
            db = get_db()
            with db.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql, (item_id,))
                row = cur.fetchone()
        if row is None:
            return jsonify({"error": f"{label} not found for id: {raw_id}"}), 404
        # Convert DictRow to a standard dictionary for consistent JSON output;
        # echo the id the caller asked with so their key keeps matching.
        from app_server_context import provider_echo_id
        payload = dict(row)
        payload['item_id'] = provider_echo_id(raw_id)
        if decode is not None:
            decode(payload)
        return jsonify(payload)
    except Exception as e:
        logger.exception("Error fetching %s for id %s", label.lower(), safe_raw_id)
        err, status = error_manager.error_response(error_manager.classify(e, ERR_DB_QUERY))
        return jsonify(err), status


@external_bp.route('/get_score', methods=['GET'])
def get_score_endpoint():
    """
    Get all content from the score database for a given id.
    ---
    tags:
      - External
    parameters:
      - name: id
        in: query
        required: true
        description: The Item ID of the track.
        schema:
          type: string
    responses:
      200:
        description: Score data for the track.
        content:
          application/json:
            schema:
              type: object
      400:
        description: Missing id parameter.
      404:
        description: Score not found for the given id.
      500:
        description: Internal server error.
    """
    return _external_row_response("SELECT * FROM score WHERE item_id = %s", 'Score')


@external_bp.route('/get_embedding', methods=['GET'])
def get_embedding_endpoint():
    """
    Get the embedding vector from the database for a given id.
    ---
    tags:
      - External
    parameters:
      - name: id
        in: query
        required: true
        description: The Item ID of the track.
        schema:
          type: string
    responses:
      200:
        description: Embedding data for the track, with the vector as a list of floats.
      400:
        description: Missing id parameter.
      404:
        description: Embedding not found for the given id.
      500:
        description: Internal server error.
    """
    return _external_row_response(
        "SELECT * FROM embedding WHERE item_id = %s", 'Embedding', _decode_embedding_bytes
    )


@external_bp.route('/search', methods=['GET'])
def search_tracks_endpoint():
    """
    Provides autocomplete suggestions for tracks based on a unified search query
    or legacy title/artist parameters.
    A query must be at least 3 characters long.
    ---
    tags:
      - External
    parameters:
      - name: search_query
        in: query
        description: Partial or full elements of songs' titles, artist or album names.
        schema:
          type: string
      - name: title
        in: query
        description: (Legacy) Partial or full title of the track. Used as fallback when search_query is absent.
        schema:
          type: string
      - name: artist
        in: query
        description: (Legacy) Partial or full name of the artist. Used as fallback when search_query is absent.
        schema:
          type: string
    responses:
      200:
        description: A list of matching tracks.
      400:
        description: Query string too short.
      500:
        description: Internal server error.
    """
    search_query = request.args.get('search_query', '', type=str)

    # Backward compatibility: support legacy 'title' and 'artist' params
    # so external apps using the old API continue to work.
    if not search_query:
        legacy_title = request.args.get('title', '', type=str).strip()
        legacy_artist = request.args.get('artist', '', type=str).strip()
        search_query = f"{legacy_artist} {legacy_title}".strip()

    # Return empty list if query is empty
    if not search_query:
        return jsonify([])

    # Enforce minimum length constraint
    if len(search_query) < 1:
        return jsonify({"error": "Query must be at least 1 character long"}), 400

    try:
        from app_server_context import resolve_request_server_id, selected_server_scope
        from tasks.mediaserver import registry

        try:
            server_id = resolve_request_server_id()
            selected_server_id, include_legacy = selected_server_scope()
        except ValueError:
            logger.warning("Invalid server selection.", exc_info=True)
            return jsonify({"error": "Invalid server selection."}), 400
        results = search_tracks_unified(
            search_query,
            server_id=selected_server_id,
            include_legacy_default=include_legacy,
        )
        try:
            mapping = registry.translate_ids([r['item_id'] for r in results], server_id)
        except Exception:
            # Fail closed: never emit untranslated canonical ids to the client.
            logger.exception("External search id translation failed")
            return jsonify({"error": "An error occurred during search."}), 500
        translated = []
        for r in results:
            if r['item_id'] not in mapping:
                continue
            row = dict(r)
            row['item_id'] = mapping[r['item_id']]
            translated.append(row)
        return jsonify(translated)
    except Exception:
        logger.exception("Error during external track search")
        return jsonify({"error": "An error occurred during search."}), 500
