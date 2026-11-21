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
  returned in smart fade order with where to start and end each, or in the
  order sent with keep_order), `/api/recording_search/play/<id>` (the song's
  audio from the media server, for the shared song player of every song list:
  FLAC, MP3 and AAC as they are, any other format converted to lossless FLAC
  by tasks.flac_stream while it is sent; with start and seconds only that
  piece, cut on the sample; each song downloaded once into app_play_cache;
  it answers with the neural
  fingerprint off too), `/api/recording_search/transition` (JSON:
  everything a player executes to mix the current song into the next, moved
  by tasks.smart_fade_voice toward audible cues with bounded trimming and softer vocal overlap) and
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
import contextlib
import json
import logging
import math
import os
import shutil
import tempfile

import app_play_cache
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
    ERR_TRACK_NOT_ANALYZABLE,
)
from error.responses import json_error, json_exception
from tasks import flac_stream

logger = logging.getLogger(__name__)

PIECE_MAX_SECONDS = 600

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
              keep_order:
                type: boolean
                default: false
                description: Keep the songs in the order sent (a path, an album) and only say where each one starts and ends.
    responses:
      200:
        description: |
          The same songs in smart fade order; songs without a neural fingerprint come last with null times.
          With keep_order the order sent is kept and a song without a fingerprint keeps its place.
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
    keep_order = data.get('keep_order') is True
    ordered, failure = _run_search(
        lambda: smart_fade_playlist([canonical.get(i, i) for i in sent], keep_order=keep_order), 'Smart fade'
    )
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
      - in: query
        name: start
        required: false
        schema:
          type: number
        description: >-
          Optional, with seconds: send only the piece of the song from this
          second (0 or more).
      - in: query
        name: seconds
        required: false
        schema:
          type: number
        description: Optional, with start - the length of the piece, in seconds (more than 0, at most 600).
    responses:
      200:
        description: >-
          Without start and seconds, the whole song, sent while the media server
          still delivers it (chunked, no Content-Length): FLAC, MP3 and AAC as
          they are, any other format (WAV, AIFF, ALAC, WavPack, WMA, Ogg, ...)
          as lossless FLAC (audio/flac, 16-bit or 24-bit at the song's own
          rate), converted while it is sent; a file the server cannot convert
          as it is. With start and seconds, that piece cut on the sample at
          the song's own rate and channels - audio/flac (16 or 24-bit) for
          integer audio, audio/wav (32-bit float) for float audio such as MP3 -
          with X-Song-Seconds, the whole song's length, when it is known. Each
          song is downloaded once and kept a while for the next requests.
      400:
        description: start or seconds is not a number in range, or only one of them is given.
      404:
        description: The id is not a song of the library.
      422:
        description: The server could not read the song to cut the piece.
      502:
        description: The media server did not deliver the song.
    """
    # The player of every song list uses this route; it needs no fingerprint,
    # so it answers whether or not NEURAL_FINGERPRINT_ENABLED is on.
    failure = _library_failure(item_id)
    if failure is not None:
        return failure
    asked = (request.args.get('start'), request.args.get('seconds'))
    piece = None
    if asked != (None, None):
        try:
            piece = (float(asked[0]), float(asked[1]))
        except (TypeError, ValueError):
            piece = (float('nan'), float('nan'))
        if not (math.isfinite(piece[0]) and math.isfinite(piece[1]) and piece[0] >= 0 and 0 < piece[1] <= PIECE_MAX_SECONDS):
            return json_error(ERR_INVALID_REQUEST, f'Give start (0 or more) and seconds (more than 0, at most {PIECE_MAX_SECONDS}) together, or neither.')
    entry = _acquire_song(item_id)
    download = entry.download
    head = download.head(flac_stream.SNIFF_BYTES)
    if len(head) >= flac_stream.SNIFF_BYTES:
        head = download.head(flac_stream.needed_bytes(head))
    if not head:
        app_play_cache.release(entry)
        return json_error(ERR_MEDIASERVER_UNREACHABLE, 'The media server did not deliver this song.', http_status=502)
    if piece is not None:
        return _piece_response(entry, item_id, head, *piece)
    kind = flac_stream.kind(head)
    flac = None
    try:
        if kind == flac_stream.KIND_CONVERT:
            flac = flac_stream.flac_chunks(download.chunks(), _log_path(entry), item_id)
        elif kind == flac_stream.KIND_MP4:
            download.done.wait()
            if download.file():
                flac = flac_stream.flac_chunks(download.file(), _log_path(entry), item_id)
    except Exception:
        logger.exception('Could not start the FLAC conversion for %s; sending the file as it is', item_id)
        flac = None
    if flac is not None:
        return Response(_send_then_release(flac, entry), mimetype='audio/flac')
    return Response(_send_then_release(download.chunks(), entry), mimetype=flac_stream.mimetype(head))


def _acquire_song(item_id):
    try:
        server_id = app_server_context.resolve_request_server_id()
    except ValueError:
        server_id = None
    with app_server_context.use_request_server():
        return app_play_cache.acquire((server_id, item_id), item_id)


def _log_path(entry):
    handle, path = tempfile.mkstemp(dir=entry.folder, suffix='.log')
    os.close(handle)
    return path


def _piece_response(entry, item_id, head, start, seconds):
    download = entry.download
    log_path = _log_path(entry)
    try:
        if download.done.is_set() or flac_stream.kind(head) == flac_stream.KIND_MP4:
            download.done.wait()
            source = download.file()
        else:
            source = download.chunks()
        cut = flac_stream.cut_piece(source, start, seconds, log_path, item_id) if source else None
    except Exception:
        logger.exception('Could not cut a piece of %s', item_id)
        cut = None
    finally:
        app_play_cache.release(entry)
        with contextlib.suppress(OSError):
            os.remove(log_path)
    if cut is None:
        return json_error(ERR_TRACK_NOT_ANALYZABLE, 'The server could not read this song.', http_status=422)
    header, body = cut
    headers = {'X-Song-Seconds': f"{header['seconds']:.6f}"} if header.get('seconds') else {}
    return Response(body, mimetype='audio/wav' if header['format'] == 'wav' else 'audio/flac', headers=headers)


def _library_failure(item_id):
    from tasks.commons import fetch_track_metadata_map

    try:
        canonical_id = app_server_context.resolve_input_item_id(item_id)
    except ValueError as exc:
        return json_exception(exc, ERR_INVALID_REQUEST)
    if not fetch_track_metadata_map([canonical_id]):
        return json_error(ERR_NOT_FOUND, 'This song is not in the library.')
    return None


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
    Everything a player executes to mix the current song into the next.
    ---
    tags:
      - Recording Search
    summary: The smart fade transition from the current song into the next, as numbers a player schedules.
    parameters:
      - in: query
        name: from_id
        required: true
        schema:
          type: string
        description: The song playing now (the answer's from), as an id of the selected server.
      - in: query
        name: to_id
        required: true
        schema:
          type: string
        description: The song that comes next (the answer's to).
      - in: query
        name: out
        required: true
        schema:
          type: number
        description: Where the current song should end, in seconds (its fade_out_seconds from the smart fade order).
      - in: query
        name: in
        required: true
        schema:
          type: number
        description: Where the next song should start, in seconds (its fade_in_seconds from the smart fade order).
    responses:
      200:
        description: |
          Everything to play, in the order a player uses it. Every list holds
          the curve measured for these two songs, one value per point, step
          seconds apart from start; points are close enough that joining them
          with straight lines (Web Audio setValueCurveAtTime) plays the curve.
          type: beat_mix (the beats lock: the tempo glides and the bass lines
          swap), beat_rise (the same, and the next song is louder: it comes in
          from the treble down) or crossfade (equal-power faders, with gentle
          EQ when voices compete).
          start: the second of the current song where the transition starts.
          duration: how many seconds both songs play together.
          middle: seconds after start where the next song takes over.
          step: seconds between two points of every list below.
          filters: high_hz (corner of the high shelf that high_db drives),
          mid_hz and mid_q (centre and width of the peaking filter that mid_db
          drives), bass_hz (corner of the low shelf that bass_db drives).
          from (the from_id song) and to (the to_id song):
          gain: the number to multiply the whole song by, from its first second
          to its last (not only in the transition), so every song plays equally
          loud (-14 LUFS): 1 = as decoded, 0.5 = half (-6 dB). A quiet song is
          raised only until its loudest peak reaches -1 dBFS; a song gets the
          same gain as from and as to.
          speed: the playback rate at each point (1 = normal, 1.02 = 2%
          faster; Web Audio playbackRate). Play the song's own audio at it:
          the pitch moves with the tempo like a DJ deck without key lock, no
          time-stretch. 1 all along in a crossfade. from plays from start, to
          from its own start.
          volume: the fader, the number to multiply the song by at each point,
          on top of gain: 0 = silent, 1 = full; from goes from 1 to 0, to from 0
          to 1. A beat mix normally fades the ends while three filters hand
          the song over; competing vocals use sustained crossfade faders
          with a gentle additional outgoing taper later in the mix.
          high_db, mid_db and bass_db: how much to turn down the highs (high
          shelf), the mids (peaking filter) and the bass (low shelf) at each
          point, in dB: 0 = unchanged, -40 = almost gone. A crossfade may also
          use gentle EQ when voices compete.
          to also has start (the second of the to_id song where it comes in),
          highpass_hz (beat_rise only: the corner of a high-pass
          used instead of its low shelf, removing what is below it; else null)
          and continue_from (the second where it plays on alone once the
          transition ends).
          info, not needed to play: tempo_from and tempo_to (BPM, null
          without a steady beat), confidence (high, medium or low),
          loudness_from and loudness_to (LUFS, whole songs), vocal_overlap
          (the fraction of competing vocals near the middle), and
          end_margin_seconds (audible outgoing tail), recovered_seconds
          (source audio retained by the final local cue adjustment, up to
          SMART_FADE_RECOVER_SECONDS in total; default 4). This adjustment
          preserves transition duration and only moves the ending later or
          the opening earlier when energy and vocal checks allow it.
        content:
          application/json:
            schema:
              type: object
              properties:
                type:
                  type: string
                  enum: [beat_mix, beat_rise, crossfade]
                start:
                  type: number
                duration:
                  type: number
                middle:
                  type: number
                step:
                  type: number
                filters:
                  type: object
                  properties:
                    high_hz:
                      type: number
                    mid_hz:
                      type: number
                    mid_q:
                      type: number
                    bass_hz:
                      type: number
                from:
                  type: object
                  properties:
                    gain:
                      type: number
                    speed:
                      type: array
                      items:
                        type: number
                    volume:
                      type: array
                      items:
                        type: number
                    high_db:
                      type: array
                      items:
                        type: number
                    mid_db:
                      type: array
                      items:
                        type: number
                    bass_db:
                      type: array
                      items:
                        type: number
                to:
                  type: object
                  properties:
                    gain:
                      type: number
                    start:
                      type: number
                    speed:
                      type: array
                      items:
                        type: number
                    volume:
                      type: array
                      items:
                        type: number
                    high_db:
                      type: array
                      items:
                        type: number
                    mid_db:
                      type: array
                      items:
                        type: number
                    bass_db:
                      type: array
                      items:
                        type: number
                    highpass_hz:
                      type: array
                      nullable: true
                      items:
                        type: number
                    continue_from:
                      type: number
                info:
                  type: object
                  properties:
                    tempo_from:
                      type: number
                      nullable: true
                    tempo_to:
                      type: number
                      nullable: true
                    confidence:
                      type: string
                      enum: [high, medium, low]
                    loudness_from:
                      type: number
                    loudness_to:
                      type: number
      400:
        description: Bad parameters, or no room even for a crossfade at these points.
      404:
        description: An id is not a song of the library.
      502:
        description: The media server did not deliver a song.
    """
    from tasks.smart_fade_levels import crossfade_plan, transition
    from tasks.smart_fade_mix import plan_transition
    from tasks.smart_fade_voice import clear_voice

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
            try:
                plan = plan_transition(paths[0], out_seconds, paths[1], in_seconds)
            except ValueError:
                plan = crossfade_plan(paths[0], out_seconds, paths[1], in_seconds)
            answer = transition(paths[0], paths[1], clear_voice(paths[0], paths[1], plan))
        except ValueError:
            return json_error(ERR_INVALID_REQUEST, 'These two songs have no audible room for a crossfade at these points.')
        except Exception as exc:
            logger.exception('Smart fade transition failed')
            return json_exception(exc, ERR_SEARCH_FAILED, 'The transition could not be made. Check the container logs.')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return Response(json.dumps(answer), mimetype='application/json')


def _send_then_release(chunks, entry):
    # send_file's direct passthrough skips call_on_close, so the song is
    # released here, when the server closes the stream (sent or abandoned).
    try:
        yield from chunks
    finally:
        chunks.close()
        app_play_cache.release(entry)


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
