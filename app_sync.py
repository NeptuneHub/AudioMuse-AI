# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Mobile-sync Flask blueprint (sync_bp) for companion client apps.

Exposes ``GET /api/sync``, a read-only export of the analysis library
(metadata, mood/energy, MusiCNN + CLAP embeddings, UMAP 2D coordinates) with
no schema, triggers, or write path.

Main Features:
* Three modes: ``?fields=index`` returns a lightweight ``{id, fp}`` manifest
  (<=1000/page), ``?ids=a,b,c`` returns full payloads for a specific id set
  (<=500), and the default returns the full paginated export (<=500/page).
* ``fp`` is a read-time md5 fingerprint over the analysis columns (UMAP and
  rating excluded), so a client diffs the manifest against its local
  fingerprints to derive adds/updates/deletes without a global re-projection
  flipping every row.
"""

import base64
import logging

import psycopg2.extras
from flask import Blueprint, request, jsonify
from flasgger import swag_from

import config
from database import get_db
from app_helper import load_map_projection
from error import error_manager
from error.error_dictionary import ERR_DB_QUERY


logger = logging.getLogger(__name__)

sync_bp = Blueprint('sync_bp', __name__)


_MAX_PAYLOAD_LIMIT = 500
_MAX_MANIFEST_LIMIT = 1000
_DEFAULT_LIMIT = 500
_DEFAULT_PROJECTION_NAME = 'main_map'


class _IdTranslationError(Exception):
    """Raised when the canonical->server id translation for a sync page fails.

    Surfaced as a 503 so sync clients retry the page instead of interpreting an
    empty track list as a mass deletion.
    """


# Read-time fingerprint over the audio-analysis columns. Opaque to the client
# (compared for equality only). Changes whenever a track is re-analyzed, so a
# manifest diff catches in-place updates. UMAP/rating are deliberately excluded
# so a global re-projection doesn't flip every fp.
_FP_SQL = (
    "substr(md5("
    "coalesce(s.mood_vector,'')||'|'||"
    "coalesce(s.energy::text,'')||'|'||"
    "coalesce(s.other_features,'')||'|'||"
    "coalesce(s.tempo::text,'')||'|'||"
    "coalesce(s.key,'')||'|'||"
    "coalesce(s.scale,'')"
    "), 1, 16)"
)


@sync_bp.route('/api/sync', methods=['GET'])
@swag_from(
    {
        'tags': ['Mobile Sync'],
        'summary': 'Read-only export of the analysis library for client apps.',
        'description': (
            'Three modes: `?fields=index` returns a lightweight {id, fp} manifest '
            '(<=1000/page) for client-side change detection; `?ids=a,b,c` returns full '
            'payloads for a specific id set (<=500); default returns the full library '
            'page by page (<=500).'
        ),
        'parameters': [
            {
                'name': 'fields',
                'in': 'query',
                'required': False,
                'description': "Set to `index` for the {id, fp} manifest.",
                'schema': {'type': 'string', 'enum': ['index']},
            },
            {
                'name': 'ids',
                'in': 'query',
                'required': False,
                'description': 'Comma-separated mediaserver GUIDs (<=500) - full payloads for just these.',
                'schema': {'type': 'string'},
            },
            {
                'name': 'include_embeddings',
                'in': 'query',
                'required': False,
                'description': 'Set to `false` to omit the MusiCNN/CLAP embedding payload.',
                'schema': {'type': 'string', 'enum': ['true', 'false'], 'default': 'true'},
            },
            {
                'name': 'page',
                'in': 'query',
                'required': False,
                'description': '1-based page number.',
                'schema': {'type': 'integer', 'minimum': 1, 'default': 1},
            },
            {
                'name': 'limit',
                'in': 'query',
                'required': False,
                'description': 'Tracks per page (payload <=500, manifest <=1000).',
                'schema': {'type': 'integer', 'minimum': 1, 'default': 500},
            },
        ],
        'responses': {
            '200': {'description': 'A page of the manifest or the full payload.'},
            '500': {'description': 'Internal server error.'},
        },
    }
)
def sync_endpoint():
    manifest_mode = request.args.get('fields') == 'index'
    page = max(1, request.args.get('page', 1, type=int))
    max_limit = _MAX_MANIFEST_LIMIT if manifest_mode else _MAX_PAYLOAD_LIMIT
    limit = min(max(1, request.args.get('limit', _DEFAULT_LIMIT, type=int)), max_limit)
    include_embeddings = request.args.get('include_embeddings', 'true').lower() != 'false'

    from app_server_context import resolve_request_server_id
    from tasks.mediaserver import registry

    try:
        server_id = resolve_request_server_id()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        server = registry.get_server(server_id) if server_id else registry.get_default_server()
    except Exception:
        # A canonicalized catalogue must NEVER be served with raw fp_ ids: a
        # client keying on provider ids would read that as a mass delete+add.
        # Fail closed (retryable 503) unless the catalogue holds no canonical
        # ids at all, where the historical config-default identity feed is
        # still exact (single-server/unit-test compatibility).
        logger.exception("Music-server registry unavailable for sync")
        if _catalogue_has_canonical_ids():
            return jsonify(
                {'error': 'Music-server registry unavailable; retry shortly'}
            ), 503
        logger.warning(
            "Registry unavailable but the catalogue has no canonical ids; "
            "using the config default"
        )
        server = None
    if server is not None:
        server_id = server['server_id']
    provider_type = server['server_type'] if server else config.MEDIASERVER_TYPE

    ids_raw = request.args.get('ids')
    id_filter = None
    if ids_raw is not None:
        provider_ids = [i for i in ids_raw.split(',') if i][:_MAX_PAYLOAD_LIMIT]
        id_filter = (
            list(registry.reverse_translate_ids(provider_ids, server_id).values())
            if server_id
            else provider_ids
        )

    try:
        conn = get_db()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if manifest_mode:
                return _manifest_page(cur, page, limit, server_id, provider_type)
            return _payload_page(
                cur, page, limit, include_embeddings, id_filter, server_id, provider_type
            )
    except _IdTranslationError:
        return jsonify({'error': 'Track id translation failed; check container logs'}), 503
    except Exception as e:
        logger.exception(
            "GET /api/sync failed (manifest=%s ids=%s page=%s limit=%s)",
            manifest_mode,
            ids_raw,
            page,
            limit,
        )
        err, status = error_manager.error_response(error_manager.classify(e, ERR_DB_QUERY))
        return jsonify(err), status


def _server_ids_for_rows(rows, server_id):
    """Canonical -> requested-server id mapping for a page of score rows.

    Identity fallback covers every row on the default server (the historical
    contract: clients receive their media server's real ids); rows the selected
    secondary server does not have are absent and get dropped by the caller.
    A registry failure raises ``_IdTranslationError`` so the endpoint answers
    503 and the client retries instead of diffing an empty page.
    """
    from tasks.mediaserver import registry

    ids = [r['item_id'] for r in rows]
    if not server_id:
        return {str(i): str(i) for i in ids}
    try:
        return registry.translate_ids(ids, server_id)
    except Exception as exc:
        logger.exception("Sync id translation failed")
        raise _IdTranslationError() from exc


def _catalogue_has_canonical_ids():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM score WHERE item_id LIKE 'fp\\_%%')"
            )
            return bool(cur.fetchone()[0])
    except Exception:
        logger.exception("Canonical-id probe failed; failing closed")
        return True


def _availability_sql(alias='s'):
    from tasks.mediaserver.registry import availability_sql

    return availability_sql(alias)


def _manifest_page(cur, page, limit, server_id, provider_type):
    if server_id:
        from tasks.mediaserver import registry
        is_default = server_id == registry.get_default_server_id()
        availability = _availability_sql('s')
        count_sql = "SELECT COUNT(*) AS n FROM score s WHERE " + availability
        count_params = (server_id, is_default)
        page_where = " WHERE " + availability
        page_params = (server_id, is_default)
    else:
        count_sql, count_params = "SELECT COUNT(*) AS n FROM score", ()
        page_where, page_params = "", ()
    cur.execute(count_sql, count_params)
    total_tracks = cur.fetchone()['n']
    offset = (page - 1) * limit
    cur.execute(
        "SELECT s.item_id, " + _FP_SQL + " AS fp "
        "FROM score s" + page_where + " ORDER BY s.item_id ASC LIMIT %s OFFSET %s",
        page_params + (limit, offset),
    )
    rows = cur.fetchall()
    server_ids = _server_ids_for_rows(rows, server_id)
    tracks = [
        {"id": server_ids[r['item_id']], "fp": r['fp']}
        for r in rows
        if r['item_id'] in server_ids
    ]
    has_more = (offset + len(rows)) < total_tracks
    return jsonify(
        {
            "tracks": tracks,
            "total_tracks": total_tracks,
            "provider_type": provider_type,
            "has_more": has_more,
            "next_page": page + 1 if has_more else None,
        }
    )


def _payload_page(cur, page, limit, include_embeddings, id_filter, server_id, provider_type):
    if server_id:
        from tasks.mediaserver import registry
        is_default = server_id == registry.get_default_server_id()
        availability = _availability_sql('s')
        availability_where = availability
        availability_params = (server_id, is_default)
    else:
        availability_where = "TRUE"
        availability_params = ()
    clap_on = include_embeddings and config.CLAP_ENABLED
    select_extra = ""
    join_extra = ""
    if include_embeddings:
        select_extra += ", e.embedding AS musicnn_blob"
        join_extra += " LEFT JOIN embedding e ON e.item_id = s.item_id"
    if clap_on:
        select_extra += ", c.embedding AS clap_blob"
        join_extra += " LEFT JOIN clap_embedding c ON c.item_id = s.item_id"

    base_select = (
        "SELECT s.item_id, s.title, s.author, s.album, s.album_artist, "
        "s.year, s.tempo, s.key, s.scale, s.mood_vector, s.other_features, "
        "s.energy, s.rating, " + _FP_SQL + " AS fp" + select_extra + " FROM score s" + join_extra
    )

    if id_filter is not None:
        if not id_filter:
            rows = []
            total_tracks = 0
        else:
            placeholders = ",".join(["%s"] * len(id_filter))
            cur.execute(
                base_select + " WHERE s.item_id IN (" + placeholders + ") AND "
                + availability_where
                + " ORDER BY s.item_id ASC",
                tuple(id_filter) + availability_params,
            )
            rows = cur.fetchall()
            total_tracks = len(rows)
        has_more = False
        next_page = None
    else:
        cur.execute(
            "SELECT COUNT(*) AS n FROM score s WHERE " + availability_where,
            availability_params,
        )
        total_tracks = cur.fetchone()['n']
        offset = (page - 1) * limit
        cur.execute(
            base_select + " WHERE " + availability_where + " ORDER BY s.item_id ASC LIMIT %s OFFSET %s",
            availability_params + (limit, offset),
        )
        rows = cur.fetchall()
        has_more = (offset + len(rows)) < total_tracks
        next_page = page + 1 if has_more else None

    id_map, proj = load_map_projection(_DEFAULT_PROJECTION_NAME)
    if id_map and proj is not None:
        umap_lookup = {iid: (float(proj[i][0]), float(proj[i][1])) for i, iid in enumerate(id_map)}
    else:
        umap_lookup = {}

    server_ids = _server_ids_for_rows(rows, server_id)
    tracks = []
    for r in rows:
        if r['item_id'] not in server_ids:
            continue
        ux, uy = umap_lookup.get(r['item_id'], (None, None))
        t = {
            "id": server_ids[r['item_id']],
            "title": r['title'],
            "artist": r['author'],
            "album_artist": r['album_artist'],
            "album": r['album'],
            "year": r['year'],
            "tempo": r['tempo'],
            "key": r['key'],
            "scale": r['scale'],
            "mood_vector": r['mood_vector'],
            "energy": r['energy'],
            "other_features": r['other_features'],
            "rating": r['rating'],
            "umap_x": ux,
            "umap_y": uy,
            "fp": r['fp'],
        }
        if include_embeddings:
            mb = r['musicnn_blob']
            t['embedding'] = base64.b64encode(bytes(mb)).decode('ascii') if mb else None
            if clap_on:
                cb = r['clap_blob']
                t['clap_embedding'] = base64.b64encode(bytes(cb)).decode('ascii') if cb else None
        tracks.append(t)

    return jsonify(
        {
            "tracks": tracks,
            "total_tracks": total_tracks,
            "provider_type": provider_type,
            "has_more": has_more,
            "next_page": next_page,
        }
    )
