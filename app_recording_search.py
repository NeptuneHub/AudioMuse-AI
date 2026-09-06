# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for Search by Recording (mounted at `/recording_search`).

Serves the page where a user records a few seconds in the browser or uploads a
clip, and the API that turns that clip into a ranked list of library songs via
`tasks.recording_search_manager`.

Main Features:
* Routes: `/recording_search` page, `/api/recording_search/search` (multipart:
  the clip, a mode, a result count) and `/api/recording_search/warmup`.
* The upload is handed to the manager as a stream, never read into memory
  here; a Content-Length past RECORDING_SEARCH_MAX_UPLOAD_MB answers 413 before
  any byte is copied.
* One mode per page tab: `identify` (the exact recording, from the stored
  chromaprints of the first two minutes), `neural` (the neural fingerprint
  sequence, any part of a song) or `lyrics`.
* The manager's ValueError (the clip is at fault) answers 400 and its
  RuntimeError (an index or model is unavailable) answers 503; everything else
  is a generic 500 with the detail only in the container log.
* Results are scoped and id-translated to the selected server like every other
  per-server search page.
"""

from flask import Blueprint, render_template, request, jsonify
import logging

import app_server_context

logger = logging.getLogger(__name__)

recording_search_bp = Blueprint('recording_search_bp', __name__, template_folder='../templates')


@recording_search_bp.route('/recording_search', methods=['GET'])
def recording_search_page():
    """
    Search by Recording UI page.
    ---
    tags:
      - Recording Search
    summary: HTML page to record or upload a clip and find the library songs that sound like it.
    responses:
      200:
        description: HTML page rendered.
    """
    from config import (
        APP_VERSION,
        CLAP_ENABLED,
        LYRICS_ENABLED,
        RECORDING_SEARCH_DEFAULT_N_RESULTS,
        RECORDING_SEARCH_MAX_UPLOAD_MB,
        RECORDING_SEARCH_RECORD_SECONDS,
    )
    from tasks.recording_search_manager import get_index_status
    from tls_listener import https_status

    try:
        index_status = get_index_status()
    except Exception:
        logger.exception('Could not read the index status for the recording search page')
        index_status = {'identify': 'not built yet', 'neural': 'not built yet', 'lyrics': False}
    https = https_status()

    return render_template(
        'recording_search.html',
        title='Search by Recording - AudioMuse-AI',
        active='recording_search',
        app_version=APP_VERSION,
        clap_enabled=CLAP_ENABLED,
        lyrics_enabled=LYRICS_ENABLED,
        index_status=index_status,
        n_results_default=RECORDING_SEARCH_DEFAULT_N_RESULTS,
        record_seconds=RECORDING_SEARCH_RECORD_SECONDS,
        max_upload_mb=RECORDING_SEARCH_MAX_UPLOAD_MB,
        https_port=https['port'] if https['enabled'] else 0,
        https_error=https['error'] or '',
    )


@recording_search_bp.route('/api/recording_search/search', methods=['POST'])
def recording_search_api():
    """
    Find the library songs that sound like an audio clip.
    ---
    tags:
      - Recording Search
    summary: Embed a recorded or uploaded clip and return the closest tracks from one index or all three fused by rank.
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            required: [clip]
            properties:
              clip:
                type: string
                format: binary
                description: The audio clip (webm/opus from the browser recorder, or any audio file).
              mode:
                type: string
                enum: [identify, neural, lyrics]
                default: identify
              n_results:
                type: integer
                minimum: 1
                default: 100
    responses:
      200:
        description: Ranked songs, most similar first. Combined rows carry the rank in each index and how many indexes agree.
        content:
          application/json:
            schema:
              type: object
              properties:
                mode:
                  type: string
                clip_seconds:
                  type: number
                transcript:
                  type: string
                  nullable: true
                warnings:
                  type: array
                  items:
                    type: string
                sources:
                  type: array
                  items:
                    type: string
                count:
                  type: integer
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
                      ranks:
                        type: object
                      agreement:
                        type: integer
                      rrf_score:
                        type: number
                        format: float
                      ber:
                        type: number
                        format: float
                        description: Identify rows only, weighted chromaprint bit error rate at the best alignment.
                      z:
                        type: number
                        format: float
                      offset_seconds:
                        type: number
                        format: float
                      identified:
                        type: boolean
      400:
        description: Missing clip, unknown mode, bad count, or a clip that cannot be used (undecodable, silent, too short, no words).
      413:
        description: The clip exceeds the upload limit.
      503:
        description: The requested index or model is not available yet.
      500:
        description: Internal error during the search.
    """
    from config import RECORDING_SEARCH_DEFAULT_N_RESULTS, RECORDING_SEARCH_MAX_UPLOAD_MB
    from tasks.recording_search_manager import MODES, run_recording_search
    from app_helper import attach_song_features

    try:
        app_server_context.resolve_request_server_id()
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return jsonify({'error': 'Invalid server selection.'}), 400

    upload = request.files.get('clip')
    if upload is None:
        return jsonify({'error': 'Missing "clip" audio file.', 'results': []}), 400

    mode = (request.form.get('mode') or 'identify').strip().lower()
    if mode not in MODES:
        return jsonify({'error': f'Unknown mode. Use one of: {", ".join(MODES)}.', 'results': []}), 400

    try:
        n_results = max(1, int(request.form.get('n_results', RECORDING_SEARCH_DEFAULT_N_RESULTS)))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid "n_results" value.', 'results': []}), 400

    limit_bytes = RECORDING_SEARCH_MAX_UPLOAD_MB * 1024 * 1024
    if request.content_length and request.content_length > limit_bytes:
        return jsonify(
            {'error': f'The clip is larger than {RECORDING_SEARCH_MAX_UPLOAD_MB} MB.', 'results': []}
        ), 413

    try:
        payload = run_recording_search(upload.stream, upload.filename, mode, n_results)
    except ValueError as exc:
        logger.warning('Recording search rejected the clip: %s', exc)
        return jsonify({'error': str(exc), 'results': []}), 400
    except RuntimeError as exc:
        logger.warning('Recording search unavailable: %s', exc)
        return jsonify({'error': str(exc), 'results': []}), 503
    except Exception:
        logger.exception('Recording search failed')
        return jsonify(
            {'error': 'An internal error occurred during the search. Check the container logs.', 'results': []}
        ), 500

    attach_song_features(payload['results'])
    payload['results'] = app_server_context.scope_results(
        payload['results'], n_results, id_key='item_id'
    )
    payload['count'] = len(payload['results'])
    return jsonify(payload)


@recording_search_bp.route('/api/recording_search/warmup', methods=['POST'])
def recording_search_warmup_api():
    """
    Warm up the recording search models.
    ---
    tags:
      - Recording Search
    summary: Preload the MusiCNN and DCLAP audio models (and Whisper when asked) and reset their idle-unload timer.
    requestBody:
      required: false
      content:
        application/json:
          schema:
            type: object
            properties:
              lyrics:
                type: boolean
                default: false
                description: Also preload Whisper for the lyrics mode.
    responses:
      200:
        description: Which models are loaded and when they unload if idle.
        content:
          application/json:
            schema:
              type: object
              properties:
                loaded:
                  type: boolean
                models:
                  type: object
                expiry_seconds:
                  type: integer
      500:
        description: Warmup failed.
    """
    from tasks.recording_search_manager import warmup_recording_models

    data = request.get_json(silent=True) or {}
    try:
        return jsonify(warmup_recording_models(include_lyrics=bool(data.get('lyrics'))))
    except Exception:
        logger.exception('Recording search warmup failed')
        return jsonify({'error': 'Warmup failed.', 'loaded': False}), 500
