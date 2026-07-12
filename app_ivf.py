# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Similarity-search Flask blueprint (ivf_bp) over the disk-paged IVF index.

Serves the ``/similarity`` UI and the similarity REST API, delegating every
vector query to ``tasks.ivf_manager`` (the disk-paged IVF index in Postgres).

Main Features:
* Routes for track search, similar tracks by id or by mood centroid, per-track
  max distance, track lookup, and playlist creation from a result set.
* Lazily loads and caches mood centroids once (thread-locked) so a
  mood-seeded search can start from a centroid vector instead of a song id.
"""

from flask import Blueprint, jsonify, request, render_template
import logging
import json
import threading
import numpy as np

import app_server_context

# Import the new config option
from config import (
    SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT,
    SIMILARITY_RADIUS_DEFAULT,
    MOOD_CENTROIDS_FILE,
)
from app_helper import serialize_neighbor_results
from error import error_manager
from error.error_dictionary import ERR_INDEX_EMPTY, UNKNOWN_ERROR_CODE
from tasks.ivf_manager import (
    find_nearest_neighbors_by_id,
    find_nearest_neighbors_by_vector,
    get_max_distance_for_id,
    create_playlist_from_ids,
    search_tracks_unified,
    get_item_id_by_title_and_artist,
)

logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_MSG = "An unexpected error occurred."


# Build a structured error body (error_code/error_class/error_message) while keeping
# a stable, user-facing 'error' string so API consumers can key on the numeric code
# without changing the human-readable text the UI already renders.
def _index_error_body(code, message):
    payload = error_manager.build(code)
    payload["error"] = message
    return payload


# Map a neighbor-search exception to the shared JSON error response (one contract,
# reused by every similarity mode). A not-loaded/empty index surfaces ERR_INDEX_EMPTY.
def _neighbor_search_error_response(ctx, exc, is_runtime):
    if is_runtime:
        logger.exception(f"Runtime error finding neighbors for {ctx}: {exc}")
        body = _index_error_body(
            ERR_INDEX_EMPTY, "The similarity search service is currently unavailable."
        )
        return jsonify(body), 503
    logger.exception(f"Unexpected error finding neighbors for {ctx}: {exc}")
    body = _index_error_body(UNKNOWN_ERROR_CODE, _UNEXPECTED_ERROR_MSG)
    return jsonify(body), 500


# Wrap the shared vector-neighbor search + error mapping
def _vector_neighbors_or_error(vector, num_neighbors, eliminate_duplicates, ctx, empty_msg):
    try:
        results = find_nearest_neighbors_by_vector(
            vector, n=num_neighbors, eliminate_duplicates=eliminate_duplicates
        )
    except RuntimeError as e:
        return None, _neighbor_search_error_response(ctx, e, is_runtime=True)
    except Exception as e:
        return None, _neighbor_search_error_response(ctx, e, is_runtime=False)
    if not results:
        return None, (jsonify({"error": empty_msg}), 404)
    return results, None


# Build similar-tracks JSON list from neighbor results (shared serializer)
def _serialize_neighbor_results(neighbor_results, requested_n=None):
    rows = serialize_neighbor_results(neighbor_results)
    return app_server_context.scope_results(rows, requested_n, id_key='item_id')


_MOOD_CENTROIDS_DATA = {}  # mood_name -> list of centroid dicts (with vectors)
_MOOD_CENTROIDS_META = {}  # mood_name -> list of {cluster_id, top_tags (top 3)} for API
_mood_centroids_loaded = False
_mood_centroids_lock = threading.Lock()


def _load_mood_centroids_for_similarity():
    try:
        with open(MOOD_CENTROIDS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        for mood, info in data.items():
            centroids = info.get('centroids', [])
            _MOOD_CENTROIDS_DATA[mood] = centroids
            meta_list = []
            for i, c in enumerate(centroids):
                tags = c.get('top_tags', {})
                top5 = sorted(tags.items(), key=lambda x: -x[1])[:5]
                meta_list.append(
                    {
                        'index': i,
                        'top_tags': [t[0] for t in top5],
                        'n_songs': c.get('n_songs', 0),
                        'mood_score': c.get('mood_score', 0),
                        'cluster_id': c.get('cluster_id', i),
                    }
                )
            _MOOD_CENTROIDS_META[mood] = meta_list
        logger.info(
            f"Loaded mood centroids for similarity: {', '.join(f'{m}({len(cs)})' for m, cs in _MOOD_CENTROIDS_DATA.items())}"
        )
    except Exception as e:
        logger.warning(f"Could not load mood centroids from {MOOD_CENTROIDS_FILE}: {e}")


def _ensure_mood_centroids_loaded():
    """Parse the mood-centroids JSON on first use instead of at import.

    The file is ~1MB; loading it lazily keeps module import (and therefore
    web/worker startup) free of the parse cost. A single attempt is made,
    matching the old import-time behavior where a failed load left the
    dicts empty without retrying.
    """
    global _mood_centroids_loaded
    if _mood_centroids_loaded:
        return
    with _mood_centroids_lock:
        if _mood_centroids_loaded:
            return
        _load_mood_centroids_for_similarity()
        _mood_centroids_loaded = True


# Create a Blueprint for IVF (similarity) related routes
ivf_bp = Blueprint('ivf_bp', __name__, template_folder='../templates')


@ivf_bp.route('/similarity', methods=['GET'])
def similarity_page():
    """
    Serves the frontend page for finding similar tracks.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the similarity page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template(
        'similarity.html', title='AudioMuse-AI - Playlist from Similar Song', active='similarity'
    )


@ivf_bp.route('/api/search_tracks', methods=['GET'])
def search_tracks_endpoint():
    """
    Provides autocomplete suggestions for tracks based on title and artist.
    ---
    tags:
      - Similarity
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
                  album:
                    type: string
                    description: Album name or 'unknown' if missing
    """
    search_query = request.args.get('search_query', '', type=str)

    # Backward compatibility: support legacy 'title' and 'artist' params
    # so external apps using the old API continue to work.
    if not search_query:
        legacy_title = request.args.get('title', '', type=str).strip()
        legacy_artist = request.args.get('artist', '', type=str).strip()
        search_query = f"{legacy_artist} {legacy_title}".strip()

    if not search_query:
        return jsonify([])

    if len(search_query) < 1:
        return jsonify([])

    # Optional index filter: 'musicnn' (default) or 'sem_grove'
    index_param = request.args.get('index', 'musicnn', type=str).strip().lower()
    item_id_filter = None
    if index_param == 'sem_grove':
        try:
            from tasks.sem_grove_manager import get_sem_grove_item_ids

            item_id_filter = get_sem_grove_item_ids()
            if not item_id_filter:
                # Index not loaded yet - don't fall back to showing all songs
                return jsonify([])
        except Exception as e:
            logger.warning(f"Could not load SemGrove item IDs for autocomplete filter: {e}")
            return jsonify([])

    # Pagination: start / end (0-based). Defaults to first 20 results.
    start = request.args.get('start', 0, type=int)
    end = request.args.get('end', None, type=int)
    if start < 0:
        start = 0
    if end is not None and end <= start:
        return jsonify([])
    limit = (end - start) if end is not None else 20
    limit = min(limit, 500)
    offset = start

    try:
        raw_results = search_tracks_unified(
            search_query,
            limit=app_server_context.overfetch_limit(limit),
            offset=offset,
            item_id_filter=item_id_filter,
        )
        results = []
        for r in raw_results:
            # Be defensive in case the source returns non-dict entries
            if isinstance(r, dict):
                album = (r.get('album') or '').strip() or 'unknown'
                results.append(
                    {
                        'item_id': r.get('item_id'),
                        'title': r.get('title'),
                        'author': r.get('author'),
                        'album': album,
                        'album_artist': (r.get('album_artist') or '').strip() or 'unknown',
                    }
                )
            else:
                results.append({'item_id': None, 'title': None, 'author': None, 'album': 'unknown'})
        results = app_server_context.scope_results(results, limit, id_key='item_id')
        return jsonify(results)
    except Exception:
        logger.exception("Error during track search")
        return jsonify(_index_error_body(UNKNOWN_ERROR_CODE, "An error occurred during search.")), 500


@ivf_bp.route('/api/mood_centroids', methods=['GET'])
def get_mood_centroids_endpoint():
    """
    Returns available mood categories and their centroids (top 3 tags only, no vectors).
    Optionally filter by mood name.
    ---
    tags:
      - Similarity
    parameters:
      - name: mood
        in: query
        description: Optional mood name to filter centroids for a specific mood.
        schema:
          type: string
    responses:
      200:
        description: Dictionary of mood names to lists of centroid metadata.
    """
    _ensure_mood_centroids_loaded()
    mood_filter = request.args.get('mood', '', type=str).strip().lower()
    if mood_filter:
        if mood_filter not in _MOOD_CENTROIDS_META:
            return jsonify(
                {
                    "error": f"Unknown mood '{mood_filter}'. Available: {list(_MOOD_CENTROIDS_META.keys())}"
                }
            ), 400
        return jsonify({mood_filter: _MOOD_CENTROIDS_META[mood_filter]})
    return jsonify(_MOOD_CENTROIDS_META)


@ivf_bp.route('/api/similar_tracks', methods=['GET'])
def get_similar_tracks_endpoint():
    """
    Find similar tracks for a given track, identified either by item_id or title/artist.
    ---
    tags:
      - Similarity
    parameters:
      - name: item_id
        in: query
        description: The media server Item ID of the track. Use this OR title/artist.
        schema:
          type: string
      - name: title
        in: query
        description: The title of the track. Must be used with 'artist'.
        schema:
          type: string
      - name: artist
        in: query
        description: The artist of the track. Must be used with 'title'.
        schema:
          type: string
      - name: n
        in: query
        description: The number of similar tracks to return.
        schema:
          type: integer
          default: 10
      - name: eliminate_duplicates
        in: query
        description: If 'true', limits the number of songs per artist in the results. If 'false', this is disabled. If the parameter is omitted, the server's default behavior is used.
        schema:
          type: string
          enum: ['true', 'false']
      - name: mood_similarity
        in: query
        description: If 'true', filters results by mood similarity using stored mood features (danceability, aggressive, happy, party, relaxed, sad). If 'false', only acoustic similarity is used. Defaults to 'true' if omitted.
        schema:
          type: string
          enum: ['true', 'false']
    responses:
      200:
        description: A list of similar tracks with their details.
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
                  album:
                    type: string
                    description: Album name or 'unknown' if missing
                  distance:
                    type: number
      400:
        description: Bad request, missing required parameters.
      404:
        description: Target track not found.
      500:
        description: Server error.
    """
    item_id = request.args.get('item_id')
    title = request.args.get('title')
    artist = request.args.get('artist')
    num_neighbors = request.args.get('n', 10, type=int)
    num_neighbors = max(1, num_neighbors)

    # Optional mood centroid parameters
    mood_param = request.args.get('mood', '', type=str).strip().lower()
    centroid_index_param = request.args.get('centroid_index', None, type=int)

    # Optional anchor parameter
    anchor_id_param = request.args.get('anchor_id', None, type=int)

    eliminate_duplicates_str = request.args.get('eliminate_duplicates')
    if eliminate_duplicates_str is None:
        eliminate_duplicates = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT
    else:
        eliminate_duplicates = eliminate_duplicates_str.lower() == 'true'

    radius_similarity_str = request.args.get('radius_similarity')
    if radius_similarity_str is None:
        # Use configured default when parameter is omitted
        radius_similarity = SIMILARITY_RADIUS_DEFAULT
    else:
        radius_similarity = radius_similarity_str.lower() == 'true'

    mood_similarity_str = request.args.get('mood_similarity')
    if mood_similarity_str is None:
        mood_similarity = None  # Respect config default when parameter is omitted
    else:
        mood_similarity = mood_similarity_str.lower() == 'true'

    # --- Mood centroid mode: use centroid vector instead of a song ---
    if mood_param and centroid_index_param is not None:
        _ensure_mood_centroids_loaded()
        if mood_param not in _MOOD_CENTROIDS_DATA:
            return jsonify(
                {
                    "error": f"Unknown mood '{mood_param}'. Available: {list(_MOOD_CENTROIDS_DATA.keys())}"
                }
            ), 400
        centroids = _MOOD_CENTROIDS_DATA[mood_param]
        if centroid_index_param < 0 or centroid_index_param >= len(centroids):
            return jsonify(
                {
                    "error": f"Invalid centroid_index {centroid_index_param} for mood '{mood_param}' (0-{len(centroids) - 1})."
                }
            ), 400

        centroid_vector = np.array(centroids[centroid_index_param]['centroid'], dtype=np.float32)
        neighbor_results, err = _vector_neighbors_or_error(
            centroid_vector,
            app_server_context.overfetch_limit(num_neighbors),
            eliminate_duplicates,
            "mood centroid",
            "No similar tracks found for this mood centroid.",
        )
        if err:
            return err
        return jsonify(_serialize_neighbor_results(neighbor_results, num_neighbors))

    # --- Anchor mode: use anchor's centroid vector ---
    if anchor_id_param is not None:
        from database import get_alchemy_anchor_by_id

        anchor = get_alchemy_anchor_by_id(anchor_id_param)
        if not anchor or not anchor.get('centroid'):
            return jsonify(
                {"error": f"Anchor with id {anchor_id_param} not found or has no centroid."}
            ), 404

        anchor_vector = np.array(anchor['centroid'], dtype=np.float32)
        neighbor_results, err = _vector_neighbors_or_error(
            anchor_vector,
            app_server_context.overfetch_limit(num_neighbors),
            eliminate_duplicates,
            f"anchor {anchor_id_param}",
            "No similar tracks found for this anchor.",
        )
        if err:
            return err
        return jsonify(_serialize_neighbor_results(neighbor_results, num_neighbors))

    # --- Standard song-based mode ---
    target_item_id = None

    if item_id:
        target_item_id = app_server_context.resolve_input_item_id(item_id)
    elif title and artist:
        resolved_id = get_item_id_by_title_and_artist(title, artist)
        if not resolved_id:
            return jsonify(
                {"error": f"Track '{title}' by '{artist}' not found in the database."}
            ), 404
        target_item_id = resolved_id
    else:
        return jsonify(
            {
                "error": "Request must include either 'item_id' or both 'title' and 'artist', or 'mood' and 'centroid_index'."
            }
        ), 400

    try:
        neighbor_results = find_nearest_neighbors_by_id(
            target_item_id,
            n=app_server_context.overfetch_limit(num_neighbors),
            eliminate_duplicates=eliminate_duplicates,
            mood_similarity=mood_similarity,
            radius_similarity=radius_similarity,
        )
        if not neighbor_results:
            return jsonify(
                {"error": "Target track not found in index or no similar tracks found."}
            ), 404

        return jsonify(_serialize_neighbor_results(neighbor_results, num_neighbors))
    except RuntimeError as e:
        return _neighbor_search_error_response(target_item_id, e, is_runtime=True)
    except Exception as e:
        return _neighbor_search_error_response(target_item_id, e, is_runtime=False)


@ivf_bp.route('/api/max_distance', methods=['GET'])
def get_max_distance_endpoint():
    """
    Maximum distance from a track to any other.
    ---
    tags:
      - Similarity
    summary: Return the largest cosine/euclidean distance between the given item and any other item in the IVF index.
    parameters:
      - name: item_id
        in: query
        required: true
        schema: { type: string }
    responses:
      200:
        description: Distance and farthest item.
        content:
          application/json:
            schema:
              type: object
              properties:
                max_distance:
                  type: number
                  format: float
                farthest_item_id:
                  type: string
                  nullable: true
      400:
        description: Missing item_id.
      404:
        description: Item not found in the index.
      503:
        description: IVF index unavailable.
    """
    item_id = request.args.get('item_id')
    if not item_id:
        return jsonify({"error": "Missing 'item_id' parameter."}), 400
    item_id = app_server_context.resolve_input_item_id(item_id)

    try:
        result = get_max_distance_for_id(item_id)
        if result is None:
            return jsonify(
                {"error": f"Item '{item_id}' not found in index or index unavailable."}
            ), 404
        return jsonify(result)
    except RuntimeError:
        logger.exception(f"Runtime error computing max distance for {item_id}")
        return jsonify(
            _index_error_body(
                ERR_INDEX_EMPTY, "The similarity search service is currently unavailable."
            )
        ), 503
    except Exception:
        logger.exception(f"Unexpected error computing max distance for {item_id}")
        return jsonify(_index_error_body(UNKNOWN_ERROR_CODE, _UNEXPECTED_ERROR_MSG)), 500


@ivf_bp.route('/api/track', methods=['GET'])
def get_track_endpoint():
    """
    Basic track metadata.
    ---
    tags:
      - Similarity
    summary: Return title, author, album, and album_artist for a given item_id.
    parameters:
      - name: item_id
        in: query
        required: true
        schema: { type: string }
    responses:
      200:
        description: Track metadata.
        content:
          application/json:
            schema:
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
                album_artist:
                  type: string
      400:
        description: Missing item_id.
      404:
        description: Item not found.
    """
    item_id = request.args.get('item_id')
    if not item_id:
        return jsonify({"error": "Missing 'item_id' parameter."}), 400

    try:
        from app_helper import get_score_data_by_ids

        details = get_score_data_by_ids([item_id])
        if not details:
            return jsonify({"error": f"Item '{item_id}' not found."}), 404
        # Return only the basic fields
        d = details[0]
        return jsonify(
            {
                "item_id": d.get('item_id'),
                "title": d.get('title'),
                "author": d.get('author'),
                "album": (d.get('album') or 'unknown'),
                "album_artist": (d.get('album_artist') or 'unknown'),
            }
        ), 200
    except Exception:
        logger.exception(f"Unexpected error fetching track {item_id}")
        return jsonify(_index_error_body(UNKNOWN_ERROR_CODE, _UNEXPECTED_ERROR_MSG)), 500


@ivf_bp.route('/api/create_playlist', methods=['POST'])
def create_media_server_playlist():
    """
    Creates a new playlist in the configured media server with the provided tracks.
    ---
    tags:
      - Similarity
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              playlist_name:
                type: string
                description: The name for the new playlist.
              track_ids:
                type: array
                items:
                  type: string
                description: A list of track Item IDs to add to the playlist.
    responses:
      201:
        description: Playlist created successfully.
      400:
        description: Bad request, invalid payload.
      500:
        description: Server error during playlist creation.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Debug log incoming payload to help trace client/server mismatch
    try:
        logger.info(f"/api/create_playlist called with payload: {data}")
    except Exception:
        logger.info('/api/create_playlist called (unable to serialize payload)')

    playlist_name = data.get('playlist_name')
    track_ids_raw = data.get('track_ids', [])

    if not isinstance(playlist_name, str) or not playlist_name:
        return jsonify({"error": "Invalid or missing 'playlist_name'"}), 400

    final_track_ids = []
    if isinstance(track_ids_raw, list):
        for item in track_ids_raw:
            item_id = None
            if isinstance(item, str):
                item_id = item
            elif isinstance(item, dict) and 'item_id' in item:
                item_id = item['item_id']

            if item_id and item_id not in final_track_ids:
                final_track_ids.append(item_id)

    if not final_track_ids:
        return jsonify({"error": "No valid track IDs were provided to create the playlist"}), 400

    # Optional user credentials may be provided by the client (e.g., from the Sonic Fingerprint UI)
    user_creds = data.get('user_creds') if isinstance(data, dict) else None

    from app_server_context import (
        resolve_request_server_id,
        create_instant_playlist_for_server,
        needs_translation,
    )
    try:
        server_id = resolve_request_server_id(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        if needs_translation(server_id):
            try:
                info = create_instant_playlist_for_server(
                    playlist_name, final_track_ids, server_id, user_creds=user_creds
                )
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            result = info['result']
            new_playlist_id = result.get('Id') if isinstance(result, dict) else result
            if not new_playlist_id:
                logger.error(
                    "Playlist '%s' was not created on server %s: the media server "
                    "returned no playlist (see worker/container logs)",
                    playlist_name, server_id,
                )
                return jsonify(
                    {"error": "The media server did not create the playlist; check container logs."}
                ), 502
            logger.info(
                f"Created playlist '{playlist_name}' on server {server_id} "
                f"({info['mapped']} mapped, {info['skipped']} unavailable)."
            )
            return jsonify(
                {
                    "message": f"Playlist '{playlist_name}' created on the selected server ({info['mapped']} tracks, {info['skipped']} unavailable).",
                    "playlist_id": new_playlist_id,
                    "mapped": info['mapped'],
                    "skipped": info['skipped'],
                }
            ), 201

        new_playlist_id = create_playlist_from_ids(
            playlist_name, final_track_ids, user_creds=user_creds
        )

        logger.info(f"Successfully created playlist '{playlist_name}' with ID {new_playlist_id}.")

        return jsonify(
            {
                "message": f"Playlist '{playlist_name}' created successfully!",
                "playlist_id": new_playlist_id,
            }
        ), 201

    except Exception:
        logger.exception(
            f"Failed to create media server playlist '{playlist_name}'"
        )
        return jsonify(
            {"error": "An error occurred while creating the playlist on the media server."}
        ), 500
