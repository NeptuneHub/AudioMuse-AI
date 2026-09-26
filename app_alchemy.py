# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for Song Alchemy: blend songs/artists into a playlist.

Serves the `/alchemy` UI and its API, delegating the actual centroid-based
blending to `tasks.song_alchemy.song_alchemy`. Also manages the persisted
"anchors" and "radios" that let a saved blend be re-run on demand.

Main Features:
* Routes: `/alchemy` page, paged playlist autocomplete (artist autocomplete is
  the one `/api/search_artists` route in app_artist_similarity), `/api/alchemy`, plus
  CRUD for `/api/anchors` and `/api/radios` (with `/api/radios/run`). Every one
  of them acts on the server selected in the sidebar - the radio run included,
  since only the cron row is the all-servers batch path.
* Serves and builds the 2D artist projection; wraps the playlist list in a
  short-TTL in-process cache guarded by a lock.
"""

from flask import Blueprint, jsonify, request, render_template
import logging
import math
import threading
import time

from tasks.song_alchemy import anchor_embedding_tag, embedding_tags_match, song_alchemy
from app_helper import attach_song_features, search_page_response, search_page_window, search_query_arg
import app_server_context
import config
from error.error_dictionary import (
    ERR_CACHE_REFRESH_FAILED,
    ERR_DB_QUERY,
    ERR_INVALID_REQUEST,
    ERR_NOT_FOUND,
    ERR_SEARCH_FAILED,
    UNKNOWN_ERROR_CODE,
)
from error.responses import json_error, json_exception

logger = logging.getLogger(__name__)

alchemy_bp = Blueprint('alchemy_bp', __name__, template_folder='../templates')

_PLAYLIST_CACHE = {}
_PLAYLIST_CACHE_TTL = 30.0
_PLAYLIST_CACHE_LOCK = threading.Lock()


@alchemy_bp.route('/alchemy', methods=['GET'])
def alchemy_page():
    """
    Song Alchemy UI page.
    ---
    tags:
      - Alchemy
    summary: HTML page for blending songs/artists into a centroid-based recommendation set.
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template(
        'alchemy.html',
        title='AudioMuse-AI - Song Alchemy',
        active='alchemy',
        alchemy_n_results_default=config.ALCHEMY_DEFAULT_N_RESULTS,
        alchemy_max_n_results=config.ALCHEMY_MAX_N_RESULTS,
        alchemy_temperature_default=config.ALCHEMY_TEMPERATURE,
    )


def _cached_all_playlists(server_id):
    cache_key = server_id or '__default__'
    now = time.monotonic()
    cached = _PLAYLIST_CACHE.get(cache_key)
    if cached is not None and (now - cached['ts']) < _PLAYLIST_CACHE_TTL:
        return cached['data']
    with _PLAYLIST_CACHE_LOCK:
        now = time.monotonic()
        cached = _PLAYLIST_CACHE.get(cache_key)
        if cached is not None and (now - cached['ts']) < _PLAYLIST_CACHE_TTL:
            return cached['data']
        from tasks.mediaserver import get_all_playlists

        data = get_all_playlists() or []
        _PLAYLIST_CACHE[cache_key] = {'data': data, 'ts': now}
        return data


@alchemy_bp.route('/api/search_playlists', methods=['GET'])
def search_playlists():
    """
    Playlist autocomplete.
    ---
    tags:
      - Alchemy
    summary: Search media-server playlists by partial name for autocomplete suggestions.
    parameters:
      - name: query
        in: query
        schema: { type: string }
        description: Partial playlist name (one character is enough; a blank query returns nothing).
      - name: start
        in: query
        schema: { type: integer, default: 0 }
        description: 0-based pagination start.
      - name: end
        in: query
        schema: { type: integer }
        description: Exclusive pagination end (at most 100 rows per page). Default returns 50 items.
    responses:
      200:
        description: List of matching playlists (id, name, count), sorted by name.
    """
    query = search_query_arg(request.args, 'query').lower()
    window = search_page_window(request.args, 100, default=50)
    if not query or window is None:
        return jsonify([])
    offset, limit = window
    try:
        with app_server_context.use_request_server() as server_id:
            playlists = _cached_all_playlists(server_id)
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return json_error(ERR_INVALID_REQUEST, 'Invalid server selection.')
    except Exception:
        logger.exception("Playlist search failed")
        return jsonify([]), 200

    out = []
    for p in playlists:
        name = p.get('Name') or p.get('name') or ''
        pid = p.get('Id') or p.get('id')
        if not pid:
            continue
        if query not in name.lower():
            continue
        count = p.get('songCount') if p.get('songCount') is not None else p.get('ChildCount')
        out.append({'id': str(pid), 'name': name, 'count': count})
    out.sort(key=lambda row: (str(row['name']).casefold(), row['id']))
    return search_page_response(out[offset:offset + limit], len(out) > offset + limit)


@alchemy_bp.route('/api/alchemy', methods=['POST'])
def alchemy_api():
    """
    Run a Song Alchemy blend.
    ---
    tags:
      - Alchemy
    summary: Combine ADD/SUBTRACT items into a centroid and return the nearest songs.
    description: |
      At least one ADD item (song or artist) is required. SUBTRACT items are
      optional and pull the centroid away from those songs/artists.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              items:
                type: array
                items:
                  type: object
                  required: [id, op]
                  properties:
                    id:
                      type: string
                    op:
                      type: string
                      enum: [ADD, SUBTRACT]
                    type:
                      type: string
                      enum: [song, artist, anchor, mood, playlist]
                      default: song
              n:
                type: integer
                description: Number of results to return. Defaults to ALCHEMY_DEFAULT_N_RESULTS.
              temperature:
                type: number
                format: float
                description: Softmax temperature for probabilistic sampling. Defaults to ALCHEMY_TEMPERATURE.
              subtract_distance:
                type: number
                format: float
                description: Optional override for the SUBTRACT exclusion radius.
    responses:
      200:
        description: Recommendation results (each row contains the song and its centroid for save-as-anchor).
      400:
        description: Validation error (no ADD items, malformed payload).
      500:
        description: Internal error.
    """
    payload = request.get_json() or {}
    try:
        app_server_context.resolve_request_server_id(payload)
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return json_error(ERR_INVALID_REQUEST, 'Invalid server selection.')
    items = payload.get('items', [])
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return json_error(ERR_INVALID_REQUEST, '"items" must be a list of objects')
    try:
        n = int(payload.get('n', config.ALCHEMY_DEFAULT_N_RESULTS))
    except (TypeError, ValueError):
        n = config.ALCHEMY_DEFAULT_N_RESULTS
    n = max(1, n)
    # Temperature parameter for probabilistic sampling (softmax temperature)
    temperature = payload.get('temperature', config.ALCHEMY_TEMPERATURE)

    # Separate items by operation
    add_items = [
        {'type': i.get('type', 'song'), 'id': i['id']}
        for i in items
        if i.get('op', '').upper() == 'ADD' and i.get('id')
    ]
    subtract_items = [
        {'type': i.get('type', 'song'), 'id': i['id']}
        for i in items
        if i.get('op', '').upper() == 'SUBTRACT' and i.get('id')
    ]

    # Song seeds may arrive as the selected server's provider ids; canonicalize
    # them before they reach the shared index (canonical ids pass through).
    seed_ids = [
        entry['id'] for entry in add_items + subtract_items
        if entry.get('type', 'song') == 'song'
    ]
    resolved_seed_ids = app_server_context.resolve_input_item_ids(seed_ids, payload)
    for entry in add_items + subtract_items:
        if entry.get('type', 'song') == 'song':
            entry['id'] = resolved_seed_ids.get(str(entry['id']), entry['id'])

    # Artist IDs are provider-specific too. The shared artist index is keyed by
    # normalized artist name, so resolve selected-server IDs before querying it.
    for entry in add_items + subtract_items:
        if entry.get('type') == 'artist':
            entry['id'] = app_server_context.resolve_artist_identifier(entry['id'], payload)

    # Allow optional override for subtract distance (from frontend slider)
    subtract_distance = payload.get('subtract_distance')
    try:
        with app_server_context.use_request_server(payload):
            results = song_alchemy(
                add_items=add_items,
                subtract_items=subtract_items,
                n_results=n,
                subtract_distance=subtract_distance,
                temperature=temperature,
            )
        attach_song_features(results.get('results'))
        # Translate every song id in the response with ONE registry round-trip: the
        # main results, the filtered_out set, and the song-type add/sub points all
        # resolve to the selected server's provider ids from a single mapping (the
        # rest of add/sub points are synthetic anchor/mood/artist/playlist markers).
        result_rows = results.get('results') or []
        filtered_rows = results.get('filtered_out') or []
        song_points = [
            point
            for key in ('add_points', 'sub_points')
            for point in (results.get(key) or [])
            if point.get('type') == 'song'
        ]
        all_ids = [
            row['item_id']
            for row in (result_rows + filtered_rows + song_points)
            if row.get('item_id')
        ]
        mapping = app_server_context.translate_ids_for_request(all_ids)

        def _translate_song_rows(rows):
            kept = []
            for row in rows:
                provider_id = mapping.get(str(row.get('item_id')))
                if provider_id is None:
                    continue
                row['item_id'] = provider_id
                kept.append(row)
            return kept

        results['results'] = _translate_song_rows(result_rows)[:n]
        results['filtered_out'] = _translate_song_rows(filtered_rows)
        for key in ('add_points', 'sub_points'):
            kept_points = []
            for point in results.get(key) or []:
                if point.get('type') == 'song':
                    provider_id = mapping.get(str(point.get('item_id')))
                    if provider_id is None:
                        continue
                    point['item_id'] = provider_id
                kept_points.append(point)
            results[key] = kept_points
        # Keep full centroid in response for client-side save action, but not in anchor list endpoint.
        return jsonify(results)
    except ValueError:
        # Log the validation error server-side but do not expose internal error text to clients
        logger.exception("Alchemy validation failure")
        return json_error(ERR_INVALID_REQUEST, "Invalid request")
    except Exception as exc:
        logger.exception("Alchemy failure")
        return json_exception(exc, ERR_SEARCH_FAILED, "Internal error")


@alchemy_bp.route('/api/anchors', methods=['GET'])
def list_anchors():
    """
    List saved alchemy anchors.
    ---
    tags:
      - Alchemy
    summary: Return id+name of every saved alchemy anchor (centroids omitted for size).
    responses:
      200:
        description: Anchor list.
        content:
          application/json:
            schema:
              type: object
              properties:
                anchors:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      name:
                        type: string
      500:
        description: Database error.
    """
    from database import get_alchemy_anchors

    try:
        anchors = get_alchemy_anchors()
        # no centroid returned here (name-only list)
        return jsonify({'anchors': [{'id': a['id'], 'name': a['name']} for a in anchors]})
    except Exception as exc:
        logger.exception('Failed to list anchors')
        return json_exception(
            exc, ERR_DB_QUERY, 'Unable to retrieve anchors at this time.', anchors=[]
        )


def _anchor_point_allowance(payload):
    cap = int(config.ALCHEMY_ANCHOR_MAX_STORED_POINTS)
    total = sum(
        len(payload.get(key)) for key in ('inclusions', 'exclusions')
        if isinstance(payload.get(key), list)
    )
    return None if total <= cap else cap // 2


def _dropped_anchor_points(payload, limit):
    dropped = {'inclusions': 0, 'exclusions': 0}
    if limit is None:
        return dropped
    for key in dropped:
        value = payload.get(key)
        if isinstance(value, list):
            dropped[key] = max(0, len(value) - limit)
    return dropped


def _heaviest_first(entries, limit):
    def weight_of(entry):
        try:
            return float(entry.get('weight', 1.0)) if isinstance(entry, dict) else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

    return sorted(entries, key=weight_of, reverse=True)[:limit]


def _parse_anchor_exclusions(payload, limit=None):
    exclusions = payload.get('exclusions')
    if not exclusions:
        return None, None
    if not isinstance(exclusions, list):
        return None, 'Anchor exclusions must be a list'
    if limit is not None:
        exclusions = exclusions[:limit]
    parsed = []
    for entry in exclusions:
        if not isinstance(entry, dict):
            return None, 'Each anchor exclusion must be an object'
        vector = entry.get('vector')
        if not isinstance(vector, list) or not vector:
            return None, 'Each anchor exclusion needs a non-empty vector list'
        distance = entry.get('distance')
        if distance is not None:
            try:
                distance = float(distance)
            except (TypeError, ValueError, OverflowError):
                return None, 'Anchor exclusion distance must be a number'
            if not math.isfinite(distance):
                return None, 'Anchor exclusion distance must be a finite number'
        parsed.append({'vector': vector, 'distance': distance})
    return parsed, None


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_embedding_vector(value):
    return (
        isinstance(value, list)
        and len(value) == int(config.EMBEDDING_DIMENSION)
        and all(_is_finite_number(number) for number in value)
    )


def _parse_anchor_inclusions(payload, limit=None):
    inclusions = payload.get('inclusions')
    if not inclusions:
        return None, None
    if not isinstance(inclusions, list):
        return None, 'Anchor inclusions must be a list'
    if limit is not None:
        inclusions = _heaviest_first(inclusions, limit)
    current = anchor_embedding_tag()
    run_embedding = payload.get('inclusions_embedding')
    if run_embedding is not None and not embedding_tags_match(run_embedding, current):
        return None, (
            'These inclusions were computed with a different embedding model. '
            'Run the alchemy again and save the anchor from the new run.'
        )
    dimension = int(config.EMBEDDING_DIMENSION)
    parsed = []
    for entry in inclusions:
        if not isinstance(entry, dict):
            return None, 'Each anchor inclusion must be an object'
        if not _is_embedding_vector(entry.get('vector')):
            return None, f'Each anchor inclusion needs a vector of {dimension} finite numbers'
        try:
            weight = float(entry.get('weight', 1.0))
        except (TypeError, ValueError, OverflowError):
            return None, 'Anchor inclusion weight must be a number'
        if not math.isfinite(weight) or weight < 0:
            return None, 'Anchor inclusion weight must be a finite number, 0 or greater'
        seed = entry.get('seed', False)
        if not isinstance(seed, bool):
            return None, 'Anchor inclusion seed must be true or false'
        group = entry.get('group')
        if group is not None and (isinstance(group, bool) or not isinstance(group, int) or group < 0):
            return None, 'Anchor inclusion group must be a whole number, 0 or greater'
        signature = entry.get('signature')
        if signature is not None and not (
            isinstance(signature, list)
            and len(signature) == 2
            and all(isinstance(part, str) for part in signature)
        ):
            return None, 'Anchor inclusion signature must be a [title, artist] pair'
        parsed.append(
            {'vector': entry['vector'], 'weight': weight, 'seed': seed, 'group': group, 'signature': signature}
        )
    return {**current, 'points': parsed}, None


@alchemy_bp.route('/api/anchors', methods=['POST'])
def create_anchor():
    """
    Save a new alchemy anchor.
    ---
    tags:
      - Alchemy
    summary: Persist an anchor (named centroid plus every include point and exclusion) for later reuse in path-finding or alchemy.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [name, centroid]
            properties:
              name:
                type: string
              centroid:
                type: array
                items:
                  type: number
                  format: float
                description: Embedding vector representing the anchor (EMBEDDING_DIMENSION finite numbers).
              exclusions:
                type: array
                description: |
                  Subtracted regions of the run being saved (from the alchemy
                  response `exclusions` field). Each entry is re-applied when
                  the anchor is used, making the anchor reproducible.
                items:
                  type: object
                  required: [vector]
                  properties:
                    vector:
                      type: array
                      items:
                        type: number
                        format: float
                    distance:
                      type: number
                      format: float
                      description: Exclusion radius around the vector.
              inclusions:
                type: array
                description: |
                  Every ADD point of the run being saved, not averaged (from the
                  alchemy response `inclusions` field). When present the anchor
                  is searched around each point instead of the single centroid,
                  so re-runs and radios keep the geometry of the original run.
                  The server stamps them with the current embedding model and
                  dimension; after a model change the anchor is ignored until
                  it is saved again.
                items:
                  type: object
                  required: [vector]
                  properties:
                    vector:
                      type: array
                      items:
                        type: number
                        format: float
                    weight:
                      type: number
                      format: float
                      description: Relative weight of the point (default 1.0).
                    seed:
                      type: boolean
                      description: True when the point is an input song, which is then kept out of the results.
                    group:
                      type: integer
                      description: Index of the run input the point came from, so a re-run picks the same query points.
                    signature:
                      type: array
                      items:
                        type: string
                      description: Lower-cased [title, artist] of a seed song, so its other copies stay out of the results too.
              inclusions_embedding:
                type: object
                description: |
                  The `inclusions_embedding` of the alchemy response the inclusions came
                  from. When it names a different embedding model than the current one
                  the save is refused, so a page left open across a model change cannot
                  store old vectors under the new model.
    responses:
      200:
        description: Anchor saved.
      400:
        description: Missing or invalid name/centroid/inclusions/exclusions.
      500:
        description: Database failure.
    """
    from database import save_alchemy_anchor

    payload = request.get_json() or {}
    raw_name = payload.get('name')
    name = raw_name.strip() if isinstance(raw_name, str) else ''
    centroid = payload.get('centroid')
    if not name:
        return json_error(ERR_INVALID_REQUEST, 'Anchor name is required')
    if not centroid or not isinstance(centroid, list):
        return json_error(ERR_INVALID_REQUEST, 'Anchor centroid is required and must be a list')
    if not _is_embedding_vector(centroid):
        return json_error(
            ERR_INVALID_REQUEST,
            f'Anchor centroid must be a list of {int(config.EMBEDDING_DIMENSION)} finite numbers',
        )
    limit = _anchor_point_allowance(payload)
    exclusions, exclusions_error = _parse_anchor_exclusions(payload, limit)
    if exclusions_error:
        return json_error(ERR_INVALID_REQUEST, exclusions_error)
    inclusions, inclusions_error = _parse_anchor_inclusions(payload, limit)
    if inclusions_error:
        return json_error(ERR_INVALID_REQUEST, inclusions_error)
    anchor = save_alchemy_anchor(name, centroid, exclusions, inclusions=inclusions)
    if not anchor:
        return json_error(ERR_DB_QUERY, 'Failed to save anchor', http_status=500)
    dropped = _dropped_anchor_points(payload, limit)
    if dropped['inclusions'] or dropped['exclusions']:
        logger.warning(
            "Anchor %s kept the %s heaviest include points and the first %s exclusions; "
            "%s and %s were dropped over the %s-point ceiling.",
            anchor['id'], limit, limit, dropped['inclusions'], dropped['exclusions'],
            config.ALCHEMY_ANCHOR_MAX_STORED_POINTS,
        )
    return jsonify({'anchor': {'id': anchor['id'], 'name': anchor['name']}, 'dropped_points': dropped})


@alchemy_bp.route('/api/anchors/<int:anchor_id>', methods=['DELETE'])
def remove_anchor(anchor_id):
    """
    Delete an alchemy anchor.
    ---
    tags:
      - Alchemy
    summary: Remove a saved anchor by id.
    parameters:
      - name: anchor_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Anchor deleted.
      404:
        description: Anchor not found.
    """
    from database import delete_alchemy_anchor

    ok = delete_alchemy_anchor(anchor_id)
    if not ok:
        return json_error(ERR_NOT_FOUND, 'Anchor not found')
    return jsonify({'deleted': True})


@alchemy_bp.route('/api/anchors/<int:anchor_id>', methods=['PUT'])
def rename_anchor(anchor_id):
    """
    Rename an alchemy anchor.
    ---
    tags:
      - Alchemy
    summary: Update the display name of a saved anchor.
    parameters:
      - name: anchor_id
        in: path
        required: true
        schema: { type: integer }
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [name]
            properties:
              name:
                type: string
    responses:
      200:
        description: Anchor renamed.
      400:
        description: Empty name.
      404:
        description: Anchor not found.
    """
    from database import update_alchemy_anchor_name

    payload = request.get_json() or {}
    raw_name = payload.get('name')
    name = raw_name.strip() if isinstance(raw_name, str) else ''
    if not name:
        return json_error(ERR_INVALID_REQUEST, 'Anchor name is required')
    anchor = update_alchemy_anchor_name(anchor_id, name)
    if not anchor:
        return json_error(ERR_NOT_FOUND, 'Anchor not found or rename failed')
    return jsonify({'anchor': {'id': anchor['id'], 'name': anchor['name']}})


def _parse_radio_settings(payload, current=None):
    current = current or {}
    temperature = payload.get('temperature')
    n_results = payload.get('n_results')
    if temperature is None:
        temperature = current.get('temperature', config.ALCHEMY_TEMPERATURE)
    if n_results is None:
        n_results = current.get('n_results', config.ALCHEMY_DEFAULT_N_RESULTS)
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        return None, None, 'Radio temperature must be a number'
    if not math.isfinite(temperature):
        return None, None, 'Radio temperature must be a finite number'
    try:
        n_results = int(n_results)
    except (TypeError, ValueError):
        return None, None, 'Radio number of results must be an integer'
    if temperature < 0:
        return None, None, 'Radio temperature must be 0 or greater'
    if n_results < 1:
        return None, None, 'Radio number of results must be 1 or greater'
    return temperature, n_results, None


@alchemy_bp.route('/api/radios', methods=['GET'])
def list_radios():
    """
    List saved alchemy radios.
    ---
    tags:
      - Alchemy
    summary: Return every saved radio (anchor + temperature + number of results) with its enabled state.
    responses:
      200:
        description: Radio list.
        content:
          application/json:
            schema:
              type: object
              properties:
                radios:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      anchor_id:
                        type: integer
                      name:
                        type: string
                        description: Name of the underlying anchor (the radio shares it).
                      temperature:
                        type: number
                        format: float
                      n_results:
                        type: integer
                      enabled:
                        type: boolean
      500:
        description: Database error.
    """
    from database import get_alchemy_radios

    try:
        radios = get_alchemy_radios()
        return jsonify(
            {
                'radios': [
                    {
                        'id': r['id'],
                        'anchor_id': r['anchor_id'],
                        'name': r['name'],
                        'temperature': r['temperature'],
                        'n_results': r['n_results'],
                        'enabled': bool(r['enabled']),
                    }
                    for r in radios
                ]
            }
        )
    except Exception as exc:
        logger.exception('Failed to list radios')
        return json_exception(
            exc, ERR_DB_QUERY, 'Unable to retrieve radios at this time.', radios=[]
        )


@alchemy_bp.route('/api/radios', methods=['POST'])
def create_radio():
    """
    Save a new alchemy radio.
    ---
    tags:
      - Alchemy
    summary: Persist a radio (anchor + temperature + number of results) for batch playlist generation.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [anchor_id]
            properties:
              anchor_id:
                type: integer
                description: Saved anchor the radio is built on (one radio per anchor).
              temperature:
                type: number
                format: float
              n_results:
                type: integer
              enabled:
                type: boolean
                default: true
    responses:
      200:
        description: Radio saved.
      400:
        description: Missing or invalid anchor/temperature/number of results.
      500:
        description: Database failure.
    """
    from database import create_alchemy_radio

    payload = request.get_json() or {}
    anchor_id = payload.get('anchor_id')
    try:
        anchor_id = int(anchor_id)
    except (TypeError, ValueError):
        return json_error(ERR_INVALID_REQUEST, 'Radio anchor is required')
    temperature, n_results, error = _parse_radio_settings(payload)
    if error:
        return json_error(ERR_INVALID_REQUEST, error)
    enabled = bool(payload.get('enabled', True))
    radio = create_alchemy_radio(anchor_id, temperature, n_results, enabled)
    if not radio:
        return json_error(
            ERR_INVALID_REQUEST,
            'Failed to save radio. Check that the anchor exists and has no radio yet.',
        )
    return jsonify({'radio': radio})


@alchemy_bp.route('/api/radios/<int:radio_id>', methods=['PUT'])
def update_radio(radio_id):
    """
    Update an alchemy radio.
    ---
    tags:
      - Alchemy
    summary: Update temperature, number of results and enabled state of a saved radio.
    parameters:
      - name: radio_id
        in: path
        required: true
        schema: { type: integer }
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: []
            properties:
              temperature:
                type: number
                format: float
              n_results:
                type: integer
              enabled:
                type: boolean
    responses:
      200:
        description: Radio updated.
      400:
        description: Invalid temperature/number of results.
      404:
        description: Radio not found.
    """
    from database import get_alchemy_radios, update_alchemy_radio

    payload = request.get_json() or {}
    current = next((r for r in get_alchemy_radios() if r['id'] == radio_id), None)
    temperature, n_results, error = _parse_radio_settings(payload, current)
    if error:
        return json_error(ERR_INVALID_REQUEST, error)
    enabled = bool(payload.get('enabled', current['enabled'] if current else True))
    radio = update_alchemy_radio(radio_id, temperature, n_results, enabled)
    if not radio:
        return json_error(ERR_NOT_FOUND, 'Radio not found or update failed')
    return jsonify({'radio': radio})


@alchemy_bp.route('/api/radios/<int:radio_id>', methods=['DELETE'])
def remove_radio(radio_id):
    """
    Delete an alchemy radio.
    ---
    tags:
      - Alchemy
    summary: Remove a saved radio by id (the underlying anchor is kept).
    parameters:
      - name: radio_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Radio deleted.
      404:
        description: Radio not found.
    """
    from database import delete_alchemy_radio

    ok = delete_alchemy_radio(radio_id)
    if not ok:
        return json_error(ERR_NOT_FOUND, 'Radio not found')
    return jsonify({'deleted': True})


@alchemy_bp.route('/api/radios/run', methods=['POST'])
def run_radio_playlists_endpoint():
    """
    Create playlists for all enabled radios on the selected server.
    ---
    tags:
      - Alchemy
    summary: Upsert one playlist per enabled radio (reuses existing playlist by name, preserving its server-side ID).
    description: |
      Runs only against the server selected in the sidebar (the default server
      when none is selected), because Alchemy is a per-server page. The
      alchemy_radio cron row is the batch path and always covers every server.
    parameters:
      - name: server
        in: query
        required: false
        schema:
          type: string
        description: Target music server name or id. Defaults to the default server.
    responses:
      200:
        description: Run summary.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                radios_enabled:
                  type: integer
                playlists_created:
                  type: integer
                failed:
                  type: array
                  items:
                    type: string
      400:
        description: Invalid server selection.
      500:
        description: Run failed.
    """
    from tasks.radio_manager import run_radio_playlists

    try:
        server_id = app_server_context.resolve_request_server_id()
    except ValueError:
        logger.warning('Invalid server selection.', exc_info=True)
        return json_error(ERR_INVALID_REQUEST, 'Invalid server selection.')

    try:
        summary = run_radio_playlists(server_scope=server_id or 'default')
        return jsonify(summary)
    except Exception as exc:
        logger.exception('Radio playlist creation failed')
        return json_exception(
            exc,
            UNKNOWN_ERROR_CODE,
            'Failed to create radio playlists. Check container logs.',
            http_status=500,
        )


@alchemy_bp.route('/api/artist_projections', methods=['GET'])
def artist_projections_api():
    """
    Precomputed artist component projections.
    ---
    tags:
      - Alchemy
    summary: Return cached 2D projections of artist GMM components for the artist map.
    responses:
      200:
        description: Component list (empty if cache is cold).
        content:
          application/json:
            schema:
              type: object
              properties:
                components:
                  type: array
                  items:
                    type: object
                    properties:
                      artist_id:
                        type: string
                      artist_name:
                        type: string
                      component_idx:
                        type: integer
                      weight:
                        type: number
                        format: float
                      projection:
                        type: array
                        items:
                          type: number
                          format: float
                        description: 2D x/y projection.
                count:
                  type: integer
      500:
        description: Failure to read cache.
    """
    from database import ARTIST_PROJECTION_CACHE
    from tasks.mediaserver import registry

    try:
        server_id = app_server_context.resolve_request_server_id()
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return json_error(ERR_INVALID_REQUEST, 'Invalid server selection.')

    try:
        if not ARTIST_PROJECTION_CACHE:
            return jsonify({'components': [], 'count': 0})

        component_map = ARTIST_PROJECTION_CACHE.get('component_map', [])
        projection = ARTIST_PROJECTION_CACHE.get('projection')

        if projection is None or len(component_map) == 0:
            return jsonify({'components': [], 'count': 0})

        # The cache stores the legacy/default artist_id; expose the selected
        # server's provider artist id instead, falling back to the artist NAME
        # (a safe, non-internal identifier the similar-artists endpoint accepts)
        # so a node without an artist_server_map row still has a live click-through.
        artist_names = [
            comp_info.get('artist_name')
            for comp_info in component_map
            if comp_info.get('artist_name')
        ]
        provider_artist_ids = registry.artist_ids_for_names(artist_names, server_id)

        # Build response with components and their 2D projections
        components = []
        for idx, comp_info in enumerate(component_map):
            if idx < len(projection):
                artist_name = comp_info.get('artist_name')
                components.append(
                    {
                        'artist_id': (provider_artist_ids.get(artist_name) or artist_name) if artist_name else None,
                        'artist_name': comp_info.get('artist_name', comp_info['artist_id']),
                        'component_idx': comp_info['component_idx'],
                        'weight': comp_info['weight'],
                        'projection': [float(projection[idx][0]), float(projection[idx][1])],
                    }
                )

        return jsonify({'components': components, 'count': len(components)})
    except Exception as exc:
        logger.exception("Failed to retrieve artist projections")
        return json_exception(
            exc, ERR_SEARCH_FAILED, 'Unable to retrieve artist projections at this time.',
            components=[], count=0,
        )


@alchemy_bp.route('/api/build_artist_projection', methods=['POST'])
def build_artist_projection_endpoint():
    """
    Rebuild artist component projections.
    ---
    tags:
      - Alchemy
    summary: Manually compute and store artist projections (requires GMM params already in DB).
    description: |
      Useful for rebuilding the artist map without running a full analysis.
      Returns 400 if no artist GMM parameters are present.
    responses:
      200:
        description: Projection rebuilt and cached.
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  enum: [success]
                message:
                  type: string
      400:
        description: No GMM parameters available.
      500:
        description: Build failed.
    """
    from app_helper import build_and_store_artist_projection

    try:
        success = build_and_store_artist_projection('artist_map')
        if success:
            return jsonify(
                {
                    'status': 'success',
                    'message': 'Artist component projection built and stored successfully',
                }
            )
        else:
            no_data = 'Artist projection build returned no data (no GMM parameters found?)'
            return json_error(ERR_INVALID_REQUEST, no_data, status='error', message=no_data)
    except Exception as exc:
        logger.exception("Failed to build artist projection")
        failed = 'Failed to build artist projection. Please try again later.'
        return json_exception(
            exc, ERR_CACHE_REFRESH_FAILED, failed, status='error', message=failed
        )
