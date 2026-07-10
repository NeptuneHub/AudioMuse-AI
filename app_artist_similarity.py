# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for Artist Similarity: find and play similar artists.

Serves the `/artist_similarity` UI and its API, delegating lookups to
`tasks.artist_gmm_manager` which backs similarity with a GMM-based index.

Main Features:
* Routes: `/artist_similarity` page, `/api/search_artists` (autocomplete),
  `/api/similar_artists`, and `/api/artist_tracks` (all tracks for an artist).
* Pure route layer with no local state; playlist creation is driven from the
  returned track lists.
"""

from flask import Blueprint, jsonify, request, render_template
import logging

from error import error_manager
from error.error_dictionary import ERR_INDEX_EMPTY, UNKNOWN_ERROR_CODE
from tasks.artist_gmm_manager import find_similar_artists, search_artists_by_name, get_artist_tracks

logger = logging.getLogger(__name__)


# Structured error body with a stable, user-facing 'error' string plus the numeric
# error_code so API consumers can distinguish an unbuilt index from a real crash.
def _index_error_body(code, message):
    payload = error_manager.build(code)
    payload["error"] = message
    return payload

# Create Blueprint
artist_similarity_bp = Blueprint('artist_similarity_bp', __name__, template_folder='templates')


@artist_similarity_bp.route('/artist_similarity', methods=['GET'])
def artist_similarity_page():
    """
    Serves the frontend page for finding similar artists.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the artist similarity page.
    """
    return render_template(
        'artist_similarity.html',
        title='AudioMuse-AI - Artist Similarity',
        active='artist_similarity',
    )


@artist_similarity_bp.route('/api/search_artists', methods=['GET'])
def search_artists_endpoint():
    """
    Provides autocomplete suggestions for artists based on name.
    ---
    tags:
      - Artist Similarity
    parameters:
      - name: query
        in: query
        description: Partial or full name of the artist.
        schema:
          type: string
    responses:
      200:
        description: A list of matching artists.
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
                properties:
                  artist:
                    type: string
                  track_count:
                    type: integer
    """
    query = request.args.get('query', '', type=str)

    if not query or len(query) < 2:
        return jsonify([])

    # Pagination: start / end (0-based). Defaults to first 20 results.
    start = request.args.get('start', 0, type=int)
    end = request.args.get('end', None, type=int)
    if start < 0:
        start = 0
    if end is not None and end <= start:
        return jsonify([])
    limit = (end - start) if end is not None else 20
    limit = min(limit, 100)
    offset = start

    try:
        results = search_artists_by_name(query, limit=limit, offset=offset)
        return jsonify(results)
    except Exception:
        logger.exception("Error during artist search")
        return jsonify(_index_error_body(UNKNOWN_ERROR_CODE, "An error occurred during search.")), 500


@artist_similarity_bp.route('/api/similar_artists', methods=['GET'])
def get_similar_artists_endpoint():
    """
    Find similar artists for a given artist using GMM-based similarity.
    Accepts either artist name or artist_id.
    ---
    tags:
      - Artist Similarity
    parameters:
      - name: artist
        in: query
        description: The name of the artist.
        schema:
          type: string
      - name: artist_id
        in: query
        description: The ID of the artist from the media server.
        schema:
          type: string
      - name: n
        in: query
        description: The number of similar artists to return.
        schema:
          type: integer
          default: 10
      - name: ef_search
        in: query
        description: HNSW search parameter (higher = more accurate but slower).
        schema:
          type: integer
      - name: include_component_matches
        in: query
        description: Include component-level similarity explanation.
        schema:
          type: boolean
          default: false
    responses:
      200:
        description: A list of similar artists with divergence scores, artist names and IDs.
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
                properties:
                  artist:
                    type: string
                  artist_id:
                    type: string
                    nullable: true
                  divergence:
                    type: number
                  component_matches:
                    type: array
                    description: Component-level matches (only if include_component_matches=true)
      400:
        description: Bad request, missing artist parameter.
      404:
        description: Artist not found.
      503:
        description: Artist similarity service unavailable.
    """
    artist = request.args.get('artist')
    artist_id = request.args.get('artist_id')
    n = request.args.get('n', 10, type=int)
    ef_search = request.args.get('ef_search', type=int)
    include_component_matches = (
        request.args.get('include_component_matches', 'false').lower() == 'true'
    )

    # Accept either artist name or artist_id
    query_artist = artist or artist_id

    if not query_artist:
        return jsonify({"error": "Missing 'artist' or 'artist_id' parameter"}), 400

    try:
        similar_artists = find_similar_artists(
            query_artist,
            n=n,
            ef_search=ef_search,
            include_component_matches=include_component_matches,
        )

        if not similar_artists:
            return jsonify(
                {
                    "error": f"Artist '{query_artist}' not found in index or no similar artists found."
                }
            ), 404

        return jsonify(similar_artists)

    except RuntimeError:
        logger.exception(
            f"Runtime error finding similar artists for '{query_artist}'"
        )
        return jsonify(
            _index_error_body(
                ERR_INDEX_EMPTY, "The artist similarity search service is currently unavailable."
            )
        ), 503
    except Exception:
        logger.exception(
            f"Unexpected error finding similar artists for '{query_artist}'"
        )
        return jsonify(_index_error_body(UNKNOWN_ERROR_CODE, "An unexpected error occurred.")), 500


@artist_similarity_bp.route('/api/artist_tracks', methods=['GET'])
def get_artist_tracks_endpoint():
    """
    Get all tracks for a given artist (by name or ID).
    ---
    tags:
      - Artist Similarity
    parameters:
      - name: artist
        in: query
        description: The name of the artist.
        schema:
          type: string
      - name: artist_id
        in: query
        description: The ID of the artist from the media server.
        schema:
          type: string
    responses:
      200:
        description: A list of tracks by the artist.
        content:
          application/json:
            schema:
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
      400:
        description: Bad request, missing artist parameter.
    """
    artist = request.args.get('artist')
    artist_id = request.args.get('artist_id')

    # Accept either artist name or artist_id
    query_artist = artist or artist_id

    if not query_artist:
        return jsonify({"error": "Missing 'artist' or 'artist_id' parameter"}), 400

    try:
        tracks = get_artist_tracks(query_artist)
        return jsonify(tracks)
    except Exception:
        logger.exception(f"Error getting tracks for artist '{query_artist}'")
        return jsonify(
            _index_error_body(UNKNOWN_ERROR_CODE, "An error occurred while fetching tracks.")
        ), 500
