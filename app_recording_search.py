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
`tasks.recording_search_manager`, which aligns the clip's neural fingerprint
on the sequences the analysis stored.

Main Features:
* Routes: `/recording_search` page, `/api/recording_search/search` (multipart:
  the clip and a result count), `/api/recording_search/by_track` (JSON: a
  library song's id and a result count, for the other recordings of that
  song) and `/api/recording_search/warmup`.
* The upload is handed to the manager as a stream, never read into memory
  here; a Content-Length past RECORDING_SEARCH_MAX_UPLOAD_MB answers 413 before
  any byte is copied.
* The manager's ValueError (the clip is at fault) answers 400 and its
  RuntimeError (the index or the model is unavailable) answers 503; everything
  else is a generic 500 with the detail only in the container log.
* Results are scoped and id-translated to the selected server like every other
  per-server search page.
* The page carries the built-in HTTPS state so the record button can open the
  secure page on a plain-HTTP address, or say why it cannot.
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
    summary: HTML page to record or upload a clip and find which library song it is.
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
        index_status = {'neural': 'not built yet'}
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
    Find which library song an audio clip is.
    ---
    tags:
      - Recording Search
    summary: Align the clip's neural fingerprint on every analysed track and return the best matches.
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
                description: The clip (webm/opus from the browser recorder, or any audio or video file).
              n_results:
                type: integer
                minimum: 1
                default: 100
    responses:
      200:
        description: Ranked songs, best match first, each with the position in the song the clip aligned on.
        content:
          application/json:
            schema:
              type: object
              properties:
                clip_seconds:
                  type: number
                clip_level_db:
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
                      title:
                        type: string
                      author:
                        type: string
                      album:
                        type: string
                      score:
                        type: number
                        format: float
                        description: Mean cosine between the clip and the track at the best alignment.
                      votes:
                        type: number
                        format: float
                      offset_seconds:
                        type: number
                        format: float
                      identified:
                        type: boolean
                      lead:
                        type: number
                        format: float
                        nullable: true
      400:
        description: Missing clip, bad count, or a clip that cannot be used (undecodable, silent, too short).
      413:
        description: The clip exceeds the upload limit.
      503:
        description: The neural fingerprint index or model is not available yet.
      500:
        description: Internal error during the search.
    """
    from config import RECORDING_SEARCH_DEFAULT_N_RESULTS, RECORDING_SEARCH_MAX_UPLOAD_MB
    from tasks.recording_search_manager import run_recording_search
    from app_helper import attach_song_features

    try:
        app_server_context.resolve_request_server_id()
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return jsonify({'error': 'Invalid server selection.'}), 400

    upload = request.files.get('clip')
    if upload is None:
        return jsonify({'error': 'Missing "clip" audio file.', 'results': []}), 400

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
        payload = run_recording_search(upload.stream, upload.filename, n_results)
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


@recording_search_bp.route('/api/recording_search/by_track', methods=['POST'])
def recording_search_by_track_api():
    """
    Find the other recordings of a library song.
    ---
    tags:
      - Recording Search
    summary: Align windows of a stored song's neural fingerprint on every other track and return the best matches.
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
                description: The song, as an id of the selected server. A playlist made from the results should start with it, since the results leave it out.
              n_results:
                type: integer
                minimum: 1
                default: 100
    responses:
      200:
        description: Ranked songs, best match first, the source song left out.
        content:
          application/json:
            schema:
              type: object
              properties:
                item_id:
                  type: string
                  description: The song, as an id of the selected server (never an internal id).
                count:
                  type: integer
                results:
                  type: array
                  items:
                    type: object
      400:
        description: Missing or unknown song, bad count, or a song without a fingerprint.
      503:
        description: The neural fingerprint index is not available yet.
      500:
        description: Internal error during the search.
    """
    from config import RECORDING_SEARCH_DEFAULT_N_RESULTS
    from tasks.recording_search_manager import search_by_track
    from app_helper import attach_song_features

    data = request.get_json(silent=True) or {}
    item_id = str(data.get('item_id') or '').strip()
    if not item_id:
        return jsonify({'error': 'Missing "item_id".', 'results': []}), 400
    try:
        n_results = max(1, int(data.get('n_results', RECORDING_SEARCH_DEFAULT_N_RESULTS)))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid "n_results" value.', 'results': []}), 400
    try:
        canonical_id = app_server_context.resolve_input_item_id(item_id)
    except ValueError as exc:
        return jsonify({'error': str(exc), 'results': []}), 400

    try:
        payload = search_by_track(canonical_id, n_results)
    except ValueError as exc:
        logger.warning('Recording search by track rejected %s: %s', item_id, exc)
        return jsonify({'error': str(exc), 'results': []}), 400
    except RuntimeError as exc:
        logger.warning('Recording search by track unavailable: %s', exc)
        return jsonify({'error': str(exc), 'results': []}), 503
    except Exception:
        logger.exception('Recording search by track failed')
        return jsonify(
            {'error': 'An internal error occurred during the search. Check the container logs.', 'results': []}
        ), 500

    payload['item_id'] = app_server_context.provider_echo_id(item_id)
    attach_song_features(payload['results'])
    payload['results'] = app_server_context.scope_results(
        payload['results'], n_results, id_key='item_id'
    )
    payload['count'] = len(payload['results'])
    return jsonify(payload)


@recording_search_bp.route('/api/recording_search/warmup', methods=['POST'])
def recording_search_warmup_api():
    """
    Warm up the recording search index.
    ---
    tags:
      - Recording Search
    summary: Start loading the neural fingerprint pack in the web process and reset its idle-unload timer.
    responses:
      200:
        description: Whether the pack is loaded and when it unloads if idle.
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

    try:
        return jsonify(warmup_recording_models())
    except Exception:
        logger.exception('Recording search warmup failed')
        return jsonify({'error': 'Warmup failed.', 'loaded': False}), 500
