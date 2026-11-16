# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for song demos (`/api/demo`).

Answers where the demo of each given song starts and ends: the part a preview
plays, usually the chorus or the drop. tasks.demo_manager finds it from the
neural fingerprint the analysis stored; static/song_demo.js asks this API and
static/song_player.js plays the answer.

Main Features:
* One route, POST `/api/demo`: a list of one or more song ids of the selected
  server and an optional length in seconds (config.SONG_DEMO_SECONDS, set in
  the setup wizard, when none is sent); the answer keeps the order sent.
* Ids are resolved for the selected server and echoed as sent, never as an
  internal canonical id; a song without a neural fingerprint gets null times.
* A missing or empty list, or a length (sent or configured) that is not a
  number of at least MIN_SECONDS, answers 400; NEURAL_FINGERPRINT_ENABLED false answers 503 naming
  the flag; anything else is a generic 500 with the detail in the log only.
"""

import logging
import math

from flask import Blueprint, jsonify, request

import app_server_context
import config
from error.error_dictionary import ERR_INDEX_EMPTY, ERR_INVALID_REQUEST, ERR_SEARCH_FAILED
from error.responses import json_error, json_exception

logger = logging.getLogger(__name__)

demo_bp = Blueprint('demo_bp', __name__)


def _requested_seconds(raw, default, minimum):
    if raw is None:
        raw = default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    seconds = float(raw)
    return seconds if math.isfinite(seconds) and seconds >= minimum else None


@demo_bp.route('/api/demo', methods=['POST'])
def demo_api():
    """
    Find the demo of one or more songs.
    ---
    tags:
      - Demo
    summary: Say where the demo of each song starts and ends (its chorus or drop), from the stored neural fingerprint.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [item_ids]
            properties:
              item_ids:
                type: array
                items:
                  type: string
                description: The songs, as ids of the selected server. One song is a list of one.
              seconds:
                type: number
                minimum: 5
                default: 30
                description: How long each demo is, in seconds. When omitted, SONG_DEMO_SECONDS from the setup wizard (30 unless changed). A song that is not longer plays whole.
    responses:
      200:
        description: One row per song, in the order sent; null times for a song without a neural fingerprint.
        content:
          application/json:
            schema:
              type: object
              properties:
                seconds:
                  type: number
                count:
                  type: integer
                results:
                  type: array
                  items:
                    type: object
                    properties:
                      item_id:
                        type: string
                      start_seconds:
                        type: number
                        nullable: true
                      end_seconds:
                        type: number
                        nullable: true
                      duration_seconds:
                        type: number
                        nullable: true
      400:
        description: Missing or empty item_ids list, a bad length, or an unknown server.
      503:
        description: The neural fingerprint is disabled.
      500:
        description: Internal error.
    """
    from tasks.demo_manager import MIN_SECONDS, demo_sections
    from tasks.neural_fingerprint import DISABLED_MESSAGE, is_enabled

    if not is_enabled():
        return json_error(ERR_INDEX_EMPTY, DISABLED_MESSAGE, results=[])
    data = request.get_json(silent=True) or {}
    raw_ids = data.get('item_ids')
    sent = [str(raw).strip() for raw in raw_ids if raw is not None and str(raw).strip()] if isinstance(raw_ids, list) else []
    if not sent:
        return json_error(ERR_INVALID_REQUEST, 'Missing "item_ids" list.', results=[])
    seconds = _requested_seconds(data.get('seconds'), config.SONG_DEMO_SECONDS, MIN_SECONDS)
    if seconds is None:
        return json_error(
            ERR_INVALID_REQUEST,
            f'"seconds" (SONG_DEMO_SECONDS when the request sends none) must be a number of at least {MIN_SECONDS:g}.',
            results=[],
        )
    try:
        canonical = app_server_context.resolve_input_item_ids(sent, data)
    except ValueError as exc:
        return json_exception(exc, ERR_INVALID_REQUEST, results=[])
    try:
        sections = demo_sections([canonical.get(item_id, item_id) for item_id in sent], seconds)
    except Exception as exc:
        logger.exception('Demo sections failed')
        return json_exception(exc, ERR_SEARCH_FAILED, 'The demos could not be found. Check the container logs.', results=[])
    empty = {'start_seconds': None, 'end_seconds': None, 'duration_seconds': None}
    results = [
        dict({'item_id': app_server_context.provider_echo_id(item_id)}, **(section or empty))
        for item_id, section in zip(sent, sections)
    ]
    return jsonify({'seconds': seconds, 'results': results, 'count': len(results)})
