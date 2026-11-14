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
  library song's id and a result count, for the songs closest to it,
  duplicates left out), `/api/recording_search/smart_fade` (JSON: any list of song ids,
  returned in smart fade order with where to start and end each),
  `/api/recording_search/play/<id>` (the song's audio from the media server,
  for the page's player), `/api/recording_search/transition` (JSON: where the
  page cuts and mixes two songs on the beat at their smart fade points) and
  `/api/recording_search/warmup`.
* The upload is handed to the manager as a stream, never read into memory
  here; a Content-Length past RECORDING_SEARCH_MAX_UPLOAD_MB answers 413 before
  any byte is copied.
* The manager's ValueError (the clip or the song is at fault) answers 400
  ERR_RECORDING_REJECTED and the index's IndexUnavailable (the index or the model
  is not ready) answers 503 ERR_RECORDING_INDEX_UNAVAILABLE, each registry message
  naming what to fix; everything else, any other RuntimeError included, is a
  generic 500. The exception text is only in the container log.
* Results are scoped and id-translated to the selected server like every other
  per-server search page.
* The page carries the built-in HTTPS state so that, on a plain-HTTP address,
  a short notice can give the same page's HTTPS address (self-signed
  certificate, one browser warning) or say why the built-in HTTPS is off.
"""

from flask import Blueprint, Response, render_template, request, jsonify
import logging
import math
import mimetypes
import os
import shutil
import tempfile

import app_server_context
from error.error_dictionary import (
    ERR_CACHE_REFRESH_FAILED,
    ERR_INDEX_EMPTY,
    ERR_INVALID_REQUEST,
    ERR_MEDIASERVER_UNREACHABLE,
    ERR_NOT_FOUND,
    ERR_PAYLOAD_TOO_LARGE,
    ERR_RECORDING_INDEX_UNAVAILABLE,
    ERR_RECORDING_REJECTED,
    ERR_SEARCH_FAILED,
)
from error.responses import json_error, json_exception

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
        NEURAL_FINGERPRINT_ENABLED,
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
        index_status = {'loaded': False, 'song_count': 0, 'cells': 0, 'cache_mb': 0.0, 'state': 'not built yet'}
    https = https_status()

    return render_template(
        'recording_search.html',
        title='Search by Recording - AudioMuse-AI',
        active='recording_search',
        app_version=APP_VERSION,
        clap_enabled=CLAP_ENABLED,
        lyrics_enabled=LYRICS_ENABLED,
        neural_enabled=NEURAL_FINGERPRINT_ENABLED,
        index_status=index_status,
        n_results_default=RECORDING_SEARCH_DEFAULT_N_RESULTS,
        record_seconds=RECORDING_SEARCH_RECORD_SECONDS,
        max_upload_mb=RECORDING_SEARCH_MAX_UPLOAD_MB,
        https_port=https['port'] if https['enabled'] else 0,
        https_error=https['error'] or '',
    )


def _disabled_response():
    from tasks.neural_fingerprint import DISABLED_MESSAGE, is_enabled

    if is_enabled():
        return None
    return json_error(ERR_INDEX_EMPTY, DISABLED_MESSAGE, results=[])


def _requested_count(raw):
    from config import RECORDING_SEARCH_DEFAULT_N_RESULTS

    try:
        return max(1, int(RECORDING_SEARCH_DEFAULT_N_RESULTS if raw is None else raw))
    except (TypeError, ValueError):
        return None


def _bad_count():
    return json_error(ERR_INVALID_REQUEST, 'Invalid "n_results" value.', results=[])


def _run_search(search, label):
    from tasks.neural_fingerprint_index import IndexUnavailable

    try:
        return search(), None
    except ValueError as exc:
        return None, json_exception(exc, ERR_RECORDING_REJECTED, results=[])
    except IndexUnavailable as exc:
        return None, json_exception(exc, ERR_RECORDING_INDEX_UNAVAILABLE, results=[])
    except Exception as exc:
        logger.exception('%s failed', label)
        return None, json_exception(
            exc, ERR_SEARCH_FAILED,
            'An internal error occurred during the search. Check the container logs.',
            results=[],
        )


def _scoped_response(payload, n_results):
    from app_helper import attach_song_features

    attach_song_features(payload['results'])
    payload['results'] = app_server_context.scope_results(payload['results'], n_results, id_key='item_id')
    payload['count'] = len(payload['results'])
    return jsonify(payload)


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
    from config import RECORDING_SEARCH_MAX_UPLOAD_MB
    from tasks.recording_search_manager import run_recording_search

    if disabled := _disabled_response():
        return disabled
    try:
        app_server_context.resolve_request_server_id()
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return json_error(ERR_INVALID_REQUEST, 'Invalid server selection.')

    upload = request.files.get('clip')
    if upload is None:
        return json_error(ERR_INVALID_REQUEST, 'Missing "clip" audio file.', results=[])

    n_results = _requested_count(request.form.get('n_results'))
    if n_results is None:
        return _bad_count()

    limit_bytes = RECORDING_SEARCH_MAX_UPLOAD_MB * 1024 * 1024
    if request.content_length and request.content_length > limit_bytes:
        return json_error(
            ERR_PAYLOAD_TOO_LARGE,
            f'The clip is larger than {RECORDING_SEARCH_MAX_UPLOAD_MB} MB.',
            results=[],
        )

    payload, failure = _run_search(
        lambda: run_recording_search(upload.stream, upload.filename, n_results), 'Recording search'
    )
    if failure is not None:
        return failure
    return _scoped_response(payload, n_results)


@recording_search_bp.route('/api/recording_search/by_track', methods=['POST'])
def recording_search_by_track_api():
    """
    Find the songs closest to a library song by neural fingerprint, without duplicates.
    ---
    tags:
      - Recording Search
    summary: Align windows of a stored song's neural fingerprint on every other track and return the best matches, duplicates left out.
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
        description: |
          Ranked songs, best match first, the source song left out. Like the similar-song
          search, repeats are left out, keeping the better-ranked one: a song with the same
          title and author as the source or as a result already kept, or the same song by
          neural fingerprint under another title (DUPLICATE_DISTANCE_THRESHOLD_NEURAL_FINGERPRINT),
          and no artist gets more than MAX_SONGS_PER_ARTIST results.
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
    from tasks.recording_search_manager import search_by_track

    if disabled := _disabled_response():
        return disabled
    data = request.get_json(silent=True) or {}
    item_id = str(data.get('item_id') or '').strip()
    if not item_id:
        return json_error(ERR_INVALID_REQUEST, 'Missing "item_id".', results=[])
    n_results = _requested_count(data.get('n_results'))
    if n_results is None:
        return _bad_count()
    try:
        canonical_id = app_server_context.resolve_input_item_id(item_id)
    except ValueError as exc:
        return json_exception(exc, ERR_INVALID_REQUEST, results=[])

    payload, failure = _run_search(lambda: search_by_track(canonical_id, n_results), 'Recording search by track')
    if failure is not None:
        return failure
    payload['item_id'] = app_server_context.provider_echo_id(item_id)
    return _scoped_response(payload, n_results)


@recording_search_bp.route('/api/recording_search/smart_fade', methods=['POST'])
def recording_search_smart_fade_api():
    """
    Order songs for a smart fade from their neural fingerprints.
    ---
    tags:
      - Recording Search
    summary: Reorder a list of songs so each flows into the next, and say where to start and end each one.
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
                description: The songs, as ids of the selected server. The first one stays first.
    responses:
      200:
        description: The same songs in smart fade order; songs without a neural fingerprint come last with null times.
        content:
          application/json:
            schema:
              type: object
              properties:
                count:
                  type: integer
                results:
                  type: array
                  items:
                    type: object
                    properties:
                      item_id:
                        type: string
                      fade_in_seconds:
                        type: number
                        nullable: true
                      fade_out_seconds:
                        type: number
                        nullable: true
                      duration_seconds:
                        type: number
                        nullable: true
      400:
        description: Missing or empty item_ids list, or an unknown server.
      503:
        description: Search by Recording is disabled.
      500:
        description: Internal error.
    """
    from tasks.recording_search_manager import smart_fade_playlist

    if disabled := _disabled_response():
        return disabled
    data = request.get_json(silent=True) or {}
    raw_ids = data.get('item_ids')
    sent = [str(raw).strip() for raw in raw_ids if raw is not None and str(raw).strip()] if isinstance(raw_ids, list) else []
    if not sent:
        return json_error(ERR_INVALID_REQUEST, 'Missing "item_ids" list.', results=[])
    try:
        canonical = app_server_context.resolve_input_item_ids(sent, data)
    except ValueError as exc:
        return json_exception(exc, ERR_INVALID_REQUEST, results=[])
    ordered, failure = _run_search(lambda: smart_fade_playlist([canonical.get(i, i) for i in sent]), 'Smart fade')
    if failure is not None:
        return failure
    empty = {'fade_in_seconds': None, 'fade_out_seconds': None, 'duration_seconds': None}
    results = [dict({'item_id': app_server_context.provider_echo_id(sent[index])}, **(fade or empty)) for index, fade in ordered]
    return jsonify({'results': results, 'count': len(results)})


@recording_search_bp.route('/api/recording_search/play/<item_id>', methods=['GET'])
def recording_search_play_api(item_id):
    """
    Stream a library song to the page's player.
    ---
    tags:
      - Recording Search
    summary: Fetch one library song from the selected media server and send its audio file.
    parameters:
      - in: path
        name: item_id
        required: true
        schema:
          type: string
        description: The song, as an id of the selected server.
    responses:
      200:
        description: The audio file as the media server delivers it.
      404:
        description: The id is not a song of the library.
      502:
        description: The media server did not deliver the song.
    """
    if disabled := _disabled_response():
        return disabled
    folder = tempfile.mkdtemp(prefix='audiomuse_play_')
    path, failure = _fetch_song(item_id, folder)
    if failure is not None:
        shutil.rmtree(folder, ignore_errors=True)
        return failure
    return Response(
        _stream_then_remove(path, folder),
        mimetype=mimetypes.guess_type(path)[0] or 'application/octet-stream',
        headers={'Content-Length': str(os.path.getsize(path))},
    )


def _fetch_song(item_id, folder):
    from tasks import mediaserver
    from tasks.commons import fetch_track_metadata_map

    try:
        canonical_id = app_server_context.resolve_input_item_id(item_id)
    except ValueError as exc:
        return None, json_exception(exc, ERR_INVALID_REQUEST)
    if not fetch_track_metadata_map([canonical_id]):
        return None, json_error(ERR_NOT_FOUND, 'This song is not in the library.')
    try:
        with app_server_context.use_request_server():
            path = mediaserver.download_track(folder, {'Id': item_id, 'id': item_id})
    except Exception:
        logger.exception('Playback download failed for %s', item_id)
        path = None
    if not path:
        return None, json_error(ERR_MEDIASERVER_UNREACHABLE, 'The media server did not deliver this song.', http_status=502)
    return path, None


@recording_search_bp.route('/api/recording_search/transition', methods=['GET'])
def recording_search_transition_api():
    """
    Plan a beat-matched transition between two songs.
    ---
    tags:
      - Recording Search
    summary: Say where the page cuts and mixes the end of one song into the next on the beat, at the smart fade points.
    parameters:
      - in: query
        name: from_id
        required: true
        schema:
          type: string
        description: The song that ends, as an id of the selected server.
      - in: query
        name: to_id
        required: true
        schema:
          type: string
        description: The song that starts.
      - in: query
        name: out
        required: true
        schema:
          type: number
        description: The smart fade end of the first song, in seconds.
      - in: query
        name: in
        required: true
        schema:
          type: number
        description: The smart fade start of the next song, in seconds.
    responses:
      200:
        description: |
          The plan, in seconds of each song; no audio. The first song plays
          until mix_at and fades out over mix_seconds; the next song's
          stretch of in_at to resume_at plays over those mix_seconds (sped up
          by stretch, pitch kept, so its beats land on the first song's) and
          fades in; then the next song plays on alone from resume_at.
          tempo_from and tempo_to are the measured tempos.
        content:
          application/json:
            schema:
              type: object
              properties:
                mix_at:
                  type: number
                mix_seconds:
                  type: number
                in_at:
                  type: number
                resume_at:
                  type: number
                stretch:
                  type: number
                tempo_from:
                  type: number
                tempo_to:
                  type: number
      400:
        description: Bad parameters, or the songs have no steady beat to match.
      404:
        description: An id is not a song of the library.
      502:
        description: The media server did not deliver a song.
    """
    from tasks.smart_fade_mix import plan_transition

    if disabled := _disabled_response():
        return disabled
    from_id = (request.args.get('from_id') or '').strip()
    to_id = (request.args.get('to_id') or '').strip()
    try:
        out_seconds = float(request.args.get('out', ''))
        in_seconds = float(request.args.get('in', ''))
    except ValueError:
        out_seconds = in_seconds = float('nan')
    if not (from_id and to_id and math.isfinite(out_seconds) and math.isfinite(in_seconds)) or min(out_seconds, in_seconds) < 0:
        return json_error(ERR_INVALID_REQUEST, 'Give from_id, to_id, out and in (seconds).')
    folder = tempfile.mkdtemp(prefix='audiomuse_mix_')
    try:
        paths = []
        for item_id in (from_id, to_id):
            path, failure = _fetch_song(item_id, folder)
            if failure is not None:
                return failure
            paths.append(path)
        try:
            plan = plan_transition(paths[0], out_seconds, paths[1], in_seconds)
        except ValueError:
            return json_error(ERR_INVALID_REQUEST, 'These two songs have no steady beat to match at the fade points.')
        except Exception as exc:
            logger.exception('Smart fade transition failed')
            return json_exception(exc, ERR_SEARCH_FAILED, 'The transition could not be made. Check the container logs.')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify(plan)


def _stream_then_remove(path, folder):
    # send_file's direct passthrough skips call_on_close, so the copy is
    # removed here, when the server closes the stream (sent or abandoned).
    try:
        with open(path, 'rb') as handle:
            while chunk := handle.read(1024 * 1024):
                yield chunk
    finally:
        shutil.rmtree(folder, ignore_errors=True)


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
    if disabled := _disabled_response():
        return disabled
    from tasks.recording_search_manager import warmup_recording_models

    try:
        return jsonify(warmup_recording_models())
    except Exception as exc:
        logger.exception('Recording search warmup failed')
        return json_exception(exc, ERR_CACHE_REFRESH_FAILED, 'Warmup failed.', loaded=False)
