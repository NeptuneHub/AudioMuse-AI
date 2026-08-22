# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for Vibe Sketch: draw a curve and play along it.

Serves the ``/vibe_sketch`` UI and its API, delegating the geometry and
nearest-song snapping to ``tasks.vibe_sketch_manager.sketch_playlist``.

Main Features:
* Routes: ``/vibe_sketch`` page, ``/api/vibe_sketch/generate`` (turn a drawn
  polyline into an ordered playlist) and ``/api/vibe_sketch/export`` (push the
  result to the selected media server).
* Pure route layer; per-server scoping and id translation reuse
  ``app_server_context``.
"""

from flask import Blueprint, jsonify, request, render_template
import logging
import math

import app_server_context
from app_helper import index_error_body, serialize_neighbor_results
from error.error_dictionary import ERR_INDEX_EMPTY, UNKNOWN_ERROR_CODE
from tasks.vibe_sketch_manager import sketch_playlist

logger = logging.getLogger(__name__)

vibe_sketch_bp = Blueprint('vibe_sketch_bp', __name__, template_folder='templates')

MAX_SKETCH_POINTS = 500


def _parse_points(raw):
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    parsed = []
    for entry in raw[:MAX_SKETCH_POINTS]:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return None
        try:
            x = float(entry[0])
            y = float(entry[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        parsed.append([x, y])
    if len(parsed) < 2:
        return None
    return parsed


@vibe_sketch_bp.route('/vibe_sketch', methods=['GET'])
def vibe_sketch_page():
    """
    Vibe Sketch UI page.
    ---
    tags:
      - UI
    summary: HTML page for drawing a curve across the music map and playing along it.
    responses:
      200:
        description: HTML content of the Vibe Sketch page.
    """
    from flask import make_response

    response = make_response(
        render_template(
            'vibe_sketch.html',
            title='AudioMuse-AI - Vibe Sketch',
            active='vibe_sketch',
        )
    )
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@vibe_sketch_bp.route('/api/vibe_sketch/generate', methods=['POST'])
def generate_sketch():
    """
    Turn a drawn polyline into an ordered playlist that follows the curve.
    ---
    tags:
      - Vibe Sketch
    summary: Sample a hand-drawn map path into waypoints and snap each to a song.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [points]
            properties:
              points:
                type: array
                description: The drawn polyline as [[x, y], ...] in map coordinates.
                items:
                  type: array
                  items:
                    type: number
                    format: float
              length:
                type: integer
                minimum: 1
                maximum: 500
                default: 20
                description: Number of songs in the resulting playlist.
              variety:
                type: number
                minimum: 0
                maximum: 1
                default: 0
                description: 0 snaps to the nearest song; higher picks from nearby candidates for surprise.
              server:
                type: string
                description: Optional server name or id (sidebar selection).
    responses:
      200:
        description: Ordered playlist whose songs follow the drawn curve.
      400:
        description: Invalid points, length or variety.
      503:
        description: No music map projection is available yet.
    """
    data = request.get_json(silent=True) or {}
    points = _parse_points(data.get('points'))
    if points is None:
        return jsonify(
            {'error': 'points must be a list of at least two [x, y] pairs.'}
        ), 400
    try:
        length = int(data.get('length', 20))
    except (TypeError, ValueError):
        return jsonify({'error': 'length must be an integer.'}), 400
    try:
        variety = float(data.get('variety', 0.0))
    except (TypeError, ValueError):
        return jsonify({'error': 'variety must be a number.'}), 400

    try:
        app_server_context.resolve_request_server_id(data)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        outcome = sketch_playlist(
            points,
            length,
            available=app_server_context.translate_ids_for_request,
            variety=variety,
        )
    except RuntimeError as exc:
        logger.warning('Vibe Sketch unavailable: %s', exc)
        return jsonify(index_error_body(ERR_INDEX_EMPTY, str(exc))), 503
    except Exception:
        logger.exception('Vibe Sketch generation failed')
        return jsonify(
            index_error_body(UNKNOWN_ERROR_CODE, 'An unexpected error occurred.')
        ), 500

    mapping = outcome.get('mapping') or {}
    coords = {str(r['item_id']): r for r in (outcome.get('results') or [])}
    serialized = []
    for row in serialize_neighbor_results(outcome.get('results') or []):
        canonical = str(row.get('item_id'))
        provider_id = mapping.get(canonical)
        if provider_id is None:
            continue
        row['item_id'] = provider_id
        meta = coords.get(canonical) or {}
        row['x'] = meta.get('x')
        row['y'] = meta.get('y')
        row['waypoint'] = meta.get('waypoint')
        serialized.append(row)
    return jsonify(
        {
            'results': serialized,
            'count': len(serialized),
            'sampled': outcome.get('sampled'),
        }
    )


@vibe_sketch_bp.route('/api/vibe_sketch/export', methods=['POST'])
def export_sketch():
    """
    Create a media-server playlist from a Vibe Sketch result.
    ---
    tags:
      - Vibe Sketch
    summary: Export the drawn playlist to the selected media server.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [playlist_name, item_ids]
            properties:
              playlist_name:
                type: string
                description: Name of the playlist to create.
              item_ids:
                type: array
                items:
                  type: string
                description: Provider song ids, in playlist order.
              server:
                type: string
                description: Optional server name or id (sidebar selection).
    responses:
      200:
        description: Playlist created on the media server.
      400:
        description: Missing name, missing songs, or unknown server.
      500:
        description: The media server rejected the playlist creation.
    """
    data = request.get_json(silent=True) or {}
    playlist_name = str(data.get('playlist_name') or '').strip()
    item_ids = data.get('item_ids')
    if not playlist_name:
        return jsonify({'message': 'Playlist name cannot be empty.'}), 400
    if not isinstance(item_ids, list) or not item_ids:
        return jsonify({'message': 'No songs provided to create the playlist.'}), 400

    try:
        server_id = app_server_context.resolve_request_server_id(data)
    except ValueError as exc:
        return jsonify({'message': f'Error: {exc}'}), 400

    resolved = app_server_context.resolve_input_item_ids(item_ids, data)
    item_ids = [resolved.get(str(i), i) for i in item_ids]

    try:
        info = app_server_context.create_instant_playlist_for_server(
            playlist_name, item_ids, server_id
        )
    except ValueError as exc:
        return jsonify({'message': f'Error: {exc}'}), 400
    except Exception:
        logger.exception('Vibe Sketch export failed')
        return jsonify({'message': 'An internal error occurred while creating the playlist.'}), 500

    created = info.get('result')
    if not created:
        return jsonify({'message': 'Media server did not return playlist information after creation.'}), 500
    return jsonify(
        {
            'message': f"Successfully created playlist '{playlist_name}' on the media server.",
            'playlist_id': created.get('Id'),
        }
    ), 200
