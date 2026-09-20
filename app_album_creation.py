# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for Album Creation: one seed in, a sequenced album out.

Serves the `/album_creation` UI and its API. Every decision about which songs
make the album and in which order lives in `tasks.album_creation_manager`; this
layer parses the request, scopes it to the selected server and never leaks an
internal id.

Main Features:
* Routes: the `/album_creation` page and `/api/album_creation/generate`. A song
  seed reuses `/api/search_tracks` for its autocomplete; a description needs no
  lookup at all.
* Two seeds: `song` (an item_id) and `text` (a few words, which DCLAP turns into
  a point in the audio space and whose named genre and instrument are checked
  against the candidates). A text seed also takes the same `steering` concepts as
  the DCLAP search page.
* PER SERVER: the generated tracks are limited to the server picked in the
  sidebar (the default one when none is given), and every returned id is that
  server's own provider id, ready for `/api/create_playlist`.
* Off unless BOTH analyses are on: the lyric themes (LYRICS_ENABLED) that the
  sequencing needs and DCLAP (CLAP_ENABLED) that the mixed space and the text
  seed need. The menu entry, the page and the API all follow; the schedule does
  not, like every other schedule.
"""

from flask import Blueprint, jsonify, render_template, request
import logging

import config
import app_helper
import app_server_context
from error.error_dictionary import (
    ERR_ALBUM_CREATION_FAILED,
    ERR_INDEX_EMPTY,
    ERR_INVALID_REQUEST,
    ERR_NOT_FOUND,
)
from error.responses import json_error, json_exception
from tasks import album_creation_manager as album_manager

logger = logging.getLogger(__name__)

album_creation_bp = Blueprint('album_creation_bp', __name__, template_folder='templates')

_DISABLED = 'Album Creation is disabled: it needs both the lyrics analysis and DCLAP.'


class _UnknownConcept(ValueError):
    pass


def _text(value):
    return value.strip() if isinstance(value, str) else ''


@album_creation_bp.route('/album_creation', methods=['GET'])
def album_creation_page():
    """
    Serves the frontend page that turns one seed into a sequenced album.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the album creation page.
    """
    return render_template(
        'album_creation.html',
        title='AudioMuse-AI - Album Creation',
        active='album_creation',
        album_creation_enabled=album_manager.is_enabled(),
        album_tracks=config.ALBUM_CREATION_TRACKS,
    )


@album_creation_bp.route('/api/album_creation/generate', methods=['POST'])
def generate_album_endpoint():
    """
    Build a CD-format album proposal from one seed.
    ---
    tags:
      - Album Creation
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              seed_type:
                type: string
                enum: [song, text]
              item_id:
                type: string
                description: Song seed, as returned by /api/search_tracks.
              query:
                type: string
                description: A few words describing the album, for the text seed.
              steering:
                type: array
                description: >
                  Optional DCLAP concept refinement for the text seed, exactly as
                  /api/clap/search takes it. Only terms from /api/clap/concepts
                  are accepted.
                items:
                  type: object
                  properties:
                    term:
                      type: string
                    direction:
                      type: string
                      enum: [more, less]
                    weight:
                      type: number
              server:
                type: string
                description: Server name or id; the default server when omitted.
    responses:
      200:
        description: The ordered tracks with their slot and role, the album stats and a suggested name.
      400:
        description: Missing or invalid seed, unknown server, or the feature is disabled.
      404:
        description: The seed has no analysed songs, or too few songs surround it.
      503:
        description: The similarity index is not available.
    """
    if not album_manager.is_enabled():
        return json_error(ERR_INVALID_REQUEST, _DISABLED)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error(ERR_INVALID_REQUEST, "Invalid JSON payload")
    seed_type = _text(data.get('seed_type')).lower()
    if seed_type not in album_manager.SEED_TYPES:
        return json_error(ERR_INVALID_REQUEST, "Parameter 'seed_type' must be song or text.")

    try:
        app_server_context.resolve_request_server_id(data)
        seed = _seed_arguments(seed_type, data)
    except _UnknownConcept:
        logger.warning("A refinement concept is not available for this library.", exc_info=True)
        return json_error(
            ERR_INVALID_REQUEST,
            'One of the chosen concepts is not available; pick them from the list.',
        )
    except ValueError:
        logger.warning("Invalid server selection or seed id.", exc_info=True)
        return json_error(ERR_INVALID_REQUEST, 'Invalid server selection.')
    if seed is None:
        return json_error(ERR_INVALID_REQUEST, f"Missing the {seed_type} to build the album from.")

    try:
        album = album_manager.create_album(seed_type, **seed)
        tracks = app_helper.attach_song_features(album['tracks'])
        tracks = app_server_context.scope_results(tracks, None, id_key='item_id')
        if len(tracks) != len(album['tracks']):
            _restate(album, tracks)
        for slot, track in enumerate(tracks, start=1):
            track['slot'] = slot
        album['tracks'] = tracks
        album['stats']['tracks'] = len(tracks)
        return jsonify(album)
    except album_manager.AlbumSeedNotFound as exc:
        return json_exception(
            exc, ERR_NOT_FOUND,
            "This seed has no analysed songs, or too few songs surround it to build an album.",
        )
    except album_manager.AlbumSeedError as exc:
        return json_exception(exc, ERR_INVALID_REQUEST, "The album seed is missing or invalid.")
    except RuntimeError as exc:
        logger.exception("Album creation could not query the similarity index")
        return json_exception(
            exc, ERR_INDEX_EMPTY, "The similarity index is not available; run the analysis first.",
        )
    except Exception as exc:
        logger.exception("Unexpected error creating an album")
        return json_exception(
            exc, ERR_ALBUM_CREATION_FAILED, "An unexpected error occurred while creating the album.",
        )


def _restate(album, tracks):
    logger.info(
        "%d of %d album tracks are not on the selected server; restating the album.",
        len(album['tracks']) - len(tracks), len(album['tracks']),
    )
    seconds = sum(track.get('duration') or 0.0 for track in tracks)
    album['stats']['minutes'] = int(round(seconds / 60.0))
    album['stats']['artists'] = len({(track.get('author') or '').strip().lower() for track in tracks})
    album['stats'].pop('cohesion', None)
    for track in tracks:
        track['role'] = album_manager.ROLE_TRACK
    if tracks:
        tracks[0]['role'] = album_manager.ROLE_OPENER
        tracks[-1]['role'] = album_manager.ROLE_CLOSER


def _seed_arguments(seed_type, data):
    if seed_type == album_manager.SEED_SONG:
        raw_id = _text(data.get('item_id'))
        if not raw_id:
            return None
        return {'item_id': app_server_context.resolve_input_item_id(raw_id, data)}
    query = _text(data.get('query'))
    if not query:
        return None
    seed = {'query': query}
    raw = data.get('steering')
    if raw:
        from tasks.clap_steering import normalize_terms

        terms, problems = normalize_terms(raw)
        if problems and not terms:
            raise _UnknownConcept(problems[0])
        seed['steering'] = terms
    return seed
