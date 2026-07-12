# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Sonic-path Flask blueprint (path_bp) building a track-to-track journey.

Serves the ``/path`` UI and the ``/api/find_path`` endpoint, delegating the
graph walk to ``tasks.path_manager`` and vector lookups to
``tasks.ivf_manager``.

Main Features:
* Builds an ordered playlist that smoothly interpolates between a start and end
  song (or a mood centroid resolved to its nearest real song).
* Loads mood centroids at import for mood-seeded endpoints and can pick a start
  or end that excludes the other endpoint to avoid degenerate same-song paths.
"""

from flask import Blueprint, jsonify, request, render_template
import logging
import json
from error import error_manager
from error.error_dictionary import UNKNOWN_ERROR_CODE
from tasks.path_manager import find_path_between_songs, get_distance
from tasks.ivf_manager import get_vector_by_id, find_nearest_neighbors_by_vector
from config import (
    PATH_DEFAULT_LENGTH,
    PATH_FIX_SIZE,
    MOOD_CENTROIDS_FILE,
    DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS,
)
from app_helper import top_stratified_genre
import app_server_context
import numpy as np
import math  # Import the math module

logger = logging.getLogger(__name__)

# --- Load mood centroids at module level ---
_MOOD_CENTROIDS = {}  # mood_name -> list of np.array centroids


def _load_mood_centroids():
    try:
        with open(MOOD_CENTROIDS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        for mood, info in data.items():
            _MOOD_CENTROIDS[mood] = [
                np.array(c['centroid'], dtype=np.float32) for c in info['centroids']
            ]
        logger.info(
            f"Loaded mood centroids: {', '.join(f'{m}({len(cs)})' for m, cs in _MOOD_CENTROIDS.items())}"
        )
    except Exception as e:
        logger.warning(f"Could not load mood centroids from {MOOD_CENTROIDS_FILE}: {e}")


_load_mood_centroids()

VALID_MOODS = {'happy', 'sad', 'aggressive', 'relaxed', 'danceable'}


def _find_nearest_song_excluding_vector(vec, exclude_id=None):
    if vec is None:
        return None
    # If wants neighbor excluding current song to avoid same-start/end on low pct
    n = 2 if exclude_id else 1
    neighbors = find_nearest_neighbors_by_vector(vec, n=n)
    if not neighbors:
        return None
    if exclude_id and len(neighbors) > 1:
        for ninfo in neighbors:
            if str(ninfo['item_id']) != str(exclude_id):
                return ninfo['item_id']
    # fallback to best neighbor
    return neighbors[0]['item_id']


def _resolve_mood_to_song_id(mood, other_song_id, pct=100):
    """
    Given a mood name and the other endpoint's song ID, find the nearest
    centroid of that mood to the song, then return the real song closest
    to a target point.
    pct=100 means use the centroid directly; lower values interpolate
    between the other song and the centroid (e.g. 50 = halfway).
    """
    if mood not in _MOOD_CENTROIDS or not _MOOD_CENTROIDS[mood]:
        return None

    if other_song_id is None:
        return None

    other_vector = get_vector_by_id(other_song_id)
    if other_vector is None:
        return None

    centroids = _MOOD_CENTROIDS[mood]
    best_centroid = min(centroids, key=lambda c: get_distance(other_vector, c))

    t = max(0, min(100, pct)) / 100.0
    target = other_vector + t * (best_centroid - other_vector)

    return _find_nearest_song_excluding_vector(target, exclude_id=other_song_id)


def _resolve_anchor_to_song_id(anchor_id, other_song_id=None, pct=100):
    from database import get_alchemy_anchor_by_id

    try:
        anchor = get_alchemy_anchor_by_id(int(anchor_id))
    except Exception:
        anchor = None
    if not anchor or not anchor.get('centroid'):
        return None
    try:
        centroid = anchor['centroid']
        if not isinstance(centroid, list):
            return None
        centroid_vec = np.array(centroid, dtype=np.float32)
    except Exception:
        return None

    if other_song_id is not None and pct is not None and pct != 100:
        other_vector = get_vector_by_id(other_song_id)
        if other_vector is None:
            return None
        t = max(0, min(100, pct)) / 100.0
        target = other_vector + t * (centroid_vec - other_vector)
        return _find_nearest_song_excluding_vector(target, exclude_id=other_song_id)

    return _find_nearest_song_excluding_vector(centroid_vec, exclude_id=other_song_id)


# Create a Blueprint for the path finding routes
path_bp = Blueprint('path_bp', __name__, template_folder='../templates')


@path_bp.route('/path', methods=['GET'])
def path_page():
    """
    Song-path UI page.
    ---
    tags:
      - Path
    summary: HTML page for building a sequence of similar songs between two endpoints.
    responses:
      200:
        description: HTML page rendered.
    """
    # Pass the server default for path_fix_size so the UI checkbox reflects config/env
    return render_template(
        'path.html', path_fix_size=PATH_FIX_SIZE, title='AudioMuse-AI - Song Path', active='path'
    )


@path_bp.route('/api/find_path', methods=['GET'])
def find_path_endpoint():
    """
    Find a path of similar songs between two endpoints.
    ---
    tags:
      - Path
    summary: Build a smooth sequence between a start and end specified by song id, mood, or anchor.
    description: |
      Each endpoint can be a song id, a mood label, or an alchemy anchor.
      Only one endpoint may be a mood/anchor at a time (the other must be a
      song id). Mood / anchor endpoints are resolved to the nearest real song
      by walking from the other endpoint toward the centroid by `mood_pct`.
    parameters:
      - name: start_song_id
        in: query
        schema: { type: string }
        description: Song id for the start endpoint.
      - name: end_song_id
        in: query
        schema: { type: string }
        description: Song id for the end endpoint.
      - name: start_mood
        in: query
        schema:
          type: string
          enum: [happy, sad, aggressive, relaxed, danceable]
      - name: end_mood
        in: query
        schema:
          type: string
          enum: [happy, sad, aggressive, relaxed, danceable]
      - name: start_anchor
        in: query
        schema: { type: integer }
        description: Alchemy anchor id for the start endpoint.
      - name: end_anchor
        in: query
        schema: { type: integer }
      - name: mood_pct
        in: query
        schema: { type: integer, default: 100 }
        description: 0-100. How far to interpolate from the other endpoint toward the mood/anchor centroid (100 = use the centroid directly).
      - name: max_steps
        in: query
        schema: { type: integer }
        description: Maximum number of intermediate songs. Defaults to PATH_DEFAULT_LENGTH from config.
      - name: path_fix_size
        in: query
        schema: { type: boolean }
        description: When true, force the path to exactly `max_steps` songs. Defaults to PATH_FIX_SIZE.
      - name: path_space
        in: query
        schema:
          type: string
          enum: [audio, lyrics]
          default: audio
        description: |
          Which vector space to build the path in. `audio` (default) uses the
          MusiCNN similarity index. `lyrics` uses the merged lyrics+audio
          SemGrove index, so the path follows lyrical meaning while staying
          acoustically smooth. Both endpoints must be songs present in the
          SemGrove index (require both lyrics and audio analysis).
    responses:
      200:
        description: Path found.
        content:
          application/json:
            schema:
              type: object
              properties:
                path:
                  type: array
                  items:
                    type: object
                total_distance:
                  type: number
                  format: float
      400:
        description: Invalid combination of endpoints, unknown mood, or both endpoints identical.
      404:
        description: Mood/anchor could not be resolved or no path found within `max_steps`.
      500:
        description: Unexpected server error.
    """
    start_song_id = request.args.get('start_song_id')
    end_song_id = request.args.get('end_song_id')
    try:
        app_server_context.resolve_request_server_id()
        resolved_endpoints = app_server_context.resolve_input_item_ids(
            [i for i in (start_song_id, end_song_id) if i]
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    if start_song_id:
        start_song_id = resolved_endpoints.get(start_song_id, start_song_id)
    if end_song_id:
        end_song_id = resolved_endpoints.get(end_song_id, end_song_id)
    start_mood = request.args.get('start_mood')
    end_mood = request.args.get('end_mood')
    start_anchor = request.args.get('start_anchor')
    end_anchor = request.args.get('end_anchor')
    mood_pct = request.args.get('mood_pct', 100, type=int)
    # Use the default from config if max_steps is not provided in the request
    max_steps = request.args.get('max_steps', PATH_DEFAULT_LENGTH, type=int)

    # Cannot have more than one special endpoint among start/end (mood or anchor)
    if (start_mood or start_anchor) and (end_mood or end_anchor):
        return jsonify(
            {"error": "Only one endpoint can be a mood/anchor and the other a song."}
        ), 400

    # Validate mood values
    if start_mood and start_mood not in VALID_MOODS:
        return jsonify(
            {"error": f"Invalid mood '{start_mood}'. Valid: {', '.join(sorted(VALID_MOODS))}"}
        ), 400
    if end_mood and end_mood not in VALID_MOODS:
        return jsonify(
            {"error": f"Invalid mood '{end_mood}'. Valid: {', '.join(sorted(VALID_MOODS))}"}
        ), 400

    # Each endpoint must have either a song ID, a mood, or an anchor
    if not start_song_id and not start_mood and not start_anchor:
        return jsonify({"error": "Start endpoint must be a song, mood, or anchor."}), 400
    if not end_song_id and not end_mood and not end_anchor:
        return jsonify({"error": "End endpoint must be a song, mood, or anchor."}), 400

    # Resolve mood/anchor to song IDs
    if start_anchor:
        resolved_id = _resolve_anchor_to_song_id(
            start_anchor, other_song_id=end_song_id, pct=mood_pct
        )
        if not resolved_id:
            return jsonify({"error": f"Could not resolve anchor '{start_anchor}' to a song."}), 404
        start_song_id = resolved_id
        logger.info(f"Resolved start anchor '{start_anchor}' to song {start_song_id}")
    elif start_mood:
        resolved_id = _resolve_mood_to_song_id(
            start_mood, end_song_id or start_song_id, pct=mood_pct
        )
        if not resolved_id:
            return jsonify({"error": f"Could not resolve mood '{start_mood}' to a song."}), 404
        start_song_id = resolved_id
        logger.info(f"Resolved start mood '{start_mood}' ({mood_pct}%) to song {start_song_id}")

    if end_anchor:
        resolved_id = _resolve_anchor_to_song_id(
            end_anchor, other_song_id=start_song_id, pct=mood_pct
        )
        if not resolved_id:
            return jsonify({"error": f"Could not resolve anchor '{end_anchor}' to a song."}), 404
        end_song_id = resolved_id
        logger.info(f"Resolved end anchor '{end_anchor}' to song {end_song_id}")
    elif end_mood:
        resolved_id = _resolve_mood_to_song_id(end_mood, start_song_id or end_song_id, pct=mood_pct)
        if not resolved_id:
            return jsonify({"error": f"Could not resolve mood '{end_mood}' to a song."}), 404
        end_song_id = resolved_id
        logger.info(f"Resolved end mood '{end_mood}' ({mood_pct}%) to song {end_song_id}")

    if start_song_id == end_song_id:
        return jsonify({"error": "Start and end songs cannot be the same."}), 400

    try:
        # parse optional path_fix_size override from request (query param)
        pfs = request.args.get('path_fix_size')
        if pfs is None:
            path_fix_size = PATH_FIX_SIZE
        else:
            path_fix_size = str(pfs).lower() in ('1', 'true', 'yes', 'y')

        # Note: `find_path_between_songs` does not accept mood direction options.
        # Path mood/anchor resolution is already done above (start/end resolved to song id).
        path_space = (request.args.get('path_space') or 'audio').strip().lower()

        if path_space in ('lyrics', 'sem_grove', 'semgrove'):
            from tasks.sem_grove_manager import (
                is_sem_grove_cache_loaded,
                get_sem_grove_item_ids,
                get_sem_grove_vector_by_id,
                find_sem_grove_neighbors_by_vector,
                find_sem_grove_neighbors_by_id,
            )

            if not is_sem_grove_cache_loaded():
                return jsonify(
                    {
                        "error": "The Lyrics (SemGrove) index is not loaded yet. Analyze lyrics and build the SemGrove index first."
                    }
                ), 404
            sem_ids = get_sem_grove_item_ids()
            if start_song_id not in sem_ids or end_song_id not in sem_ids:
                return jsonify(
                    {
                        "error": "One or both selected songs are not in the Lyrics index (they need both lyrics and audio analysis)."
                    }
                ), 404
            path, total_distance = find_path_between_songs(
                start_song_id,
                end_song_id,
                max_steps,
                path_fix_size=path_fix_size,
                get_vector_fn=get_sem_grove_vector_by_id,
                neighbors_fn=find_sem_grove_neighbors_by_vector,
                neighbors_by_id_fn=find_sem_grove_neighbors_by_id,
                metric="angular",
                dup_threshold_cosine=DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS,
            )
        else:
            path, total_distance = find_path_between_songs(
                start_song_id, end_song_id, max_steps, path_fix_size=path_fix_size
            )

        if not path:
            return jsonify(
                {"error": f"No path found between the selected songs within {max_steps} steps."}
            ), 404

        # --- CHANGED: Process embedding vectors for JSON response ---
        for song in path:
            # The raw 'embedding' is a memoryview/bytes object and is not JSON serializable.
            if 'embedding' in song:
                del song['embedding']

            # Convert numpy array 'embedding_vector' to a plain list if it exists
            if 'embedding_vector' in song and isinstance(song['embedding_vector'], np.ndarray):
                song['embedding_vector'] = song['embedding_vector'].tolist()
            else:
                song['embedding_vector'] = []

            # Ensure album field is present (for frontend)
            if 'album' not in song:
                song['album'] = song.get('album', '')

            song.setdefault('top_genre', top_stratified_genre(song.get('mood_vector')))

        # --- FIX: Convert total_distance from numpy.float32 to a standard Python float ---
        final_distance = (
            float(total_distance)
            if total_distance is not None and math.isfinite(total_distance)
            else 0.0
        )

        path = app_server_context.scope_results(path, None, id_key='item_id')

        return jsonify({"path": path, "total_distance": final_distance})

    except Exception:
        logger.exception(
            f"Error finding path between {start_song_id} and {end_song_id}"
        )
        body = {
            **error_manager.build(UNKNOWN_ERROR_CODE),
            "error": "An unexpected error occurred while finding the path.",
        }
        return jsonify(body), 500
