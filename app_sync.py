"""Mobile sync endpoint for companion client apps.

Exposes ``GET /api/sync`` — a read-only export of the analysis library
(metadata, mood/energy, MusiCNN + CLAP embeddings, UMAP 2D coordinates).

Three modes, all read-only (no schema, no triggers, no write path):
  * ``?fields=index`` → lightweight ``{id, fp}`` manifest (<=1000/page) for
    client-side change detection.
  * ``?ids=a,b,c``    → full payloads for a specific id set (<=500).
  * (default)         → full paginated export (<=500/page).

``fp`` is a read-time fingerprint over the analysis columns; a client diffs
the manifest against its local fingerprints to derive adds/updates/deletes.
"""
import base64
import logging

import psycopg2.extras
from flask import Blueprint, request, jsonify
from flasgger import swag_from

import config
from app_helper import get_db, load_map_projection


logger = logging.getLogger(__name__)

sync_bp = Blueprint('sync_bp', __name__)


_MAX_PAYLOAD_LIMIT = 500
_MAX_MANIFEST_LIMIT = 1000
_DEFAULT_LIMIT = 500
_DEFAULT_PROJECTION_NAME = 'main_map'

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
@swag_from({
    'tags': ['Mobile Sync'],
    'summary': 'Read-only export of the analysis library for client apps.',
    'description': (
        'Three modes: `?fields=index` returns a lightweight {id, fp} manifest '
        '(<=1000/page) for client-side change detection; `?ids=a,b,c` returns full '
        'payloads for a specific id set (<=500); default returns the full library '
        'page by page (<=500). Not supported for the `mpd` media server (501).'
    ),
    'parameters': [
        {'name': 'fields', 'in': 'query', 'required': False,
         'description': "Set to `index` for the {id, fp} manifest.",
         'schema': {'type': 'string', 'enum': ['index']}},
        {'name': 'ids', 'in': 'query', 'required': False,
         'description': 'Comma-separated mediaserver GUIDs (<=500) — full payloads for just these.',
         'schema': {'type': 'string'}},
        {'name': 'include_embeddings', 'in': 'query', 'required': False,
         'description': 'Set to `false` to omit the MusiCNN/CLAP embedding payload.',
         'schema': {'type': 'string', 'enum': ['true', 'false'], 'default': 'true'}},
        {'name': 'page', 'in': 'query', 'required': False,
         'description': '1-based page number.',
         'schema': {'type': 'integer', 'minimum': 1, 'default': 1}},
        {'name': 'limit', 'in': 'query', 'required': False,
         'description': 'Tracks per page (payload <=500, manifest <=1000).',
         'schema': {'type': 'integer', 'minimum': 1, 'default': 500}},
    ],
    'responses': {
        '200': {'description': 'A page of the manifest or the full payload.'},
        '500': {'description': 'Internal server error.'},
        '501': {'description': 'Media server type not supported (e.g. `mpd`).'},
    },
})
def sync_endpoint():
    if config.MEDIASERVER_TYPE == 'mpd':
        return jsonify({"error": "mpd is not yet supported by the mobile sync endpoint"}), 501

    manifest_mode = request.args.get('fields') == 'index'
    page = max(1, request.args.get('page', 1, type=int))
    max_limit = _MAX_MANIFEST_LIMIT if manifest_mode else _MAX_PAYLOAD_LIMIT
    limit = min(max(1, request.args.get('limit', _DEFAULT_LIMIT, type=int)), max_limit)
    include_embeddings = request.args.get('include_embeddings', 'true').lower() != 'false'

    ids_raw = request.args.get('ids')
    id_filter = None
    if ids_raw is not None:
        id_filter = [i for i in ids_raw.split(',') if i][:_MAX_PAYLOAD_LIMIT]

    try:
        conn = get_db()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if manifest_mode:
                return _manifest_page(cur, page, limit)
            return _payload_page(cur, page, limit, include_embeddings, id_filter)
    except Exception:
        logger.exception(
            "GET /api/sync failed (manifest=%s ids=%s page=%s limit=%s)",
            manifest_mode, ids_raw, page, limit,
        )
        return jsonify({"error": "Internal server error"}), 500


def _manifest_page(cur, page, limit):
    cur.execute("SELECT COUNT(*) AS n FROM score", ())
    total_tracks = cur.fetchone()['n']
    offset = (page - 1) * limit
    cur.execute(
        "SELECT s.item_id, " + _FP_SQL + " AS fp "
        "FROM score s ORDER BY s.item_id ASC LIMIT %s OFFSET %s",
        (limit, offset),
    )
    rows = cur.fetchall()
    tracks = [{"id": r['item_id'], "fp": r['fp']} for r in rows]
    has_more = (offset + len(rows)) < total_tracks
    return jsonify({
        "tracks": tracks,
        "total_tracks": total_tracks,
        "provider_type": config.MEDIASERVER_TYPE,
        "has_more": has_more,
        "next_page": page + 1 if has_more else None,
    })


def _payload_page(cur, page, limit, include_embeddings, id_filter):
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
        "s.energy, s.rating, " + _FP_SQL + " AS fp" + select_extra +
        " FROM score s" + join_extra
    )

    if id_filter is not None:
        if not id_filter:
            rows = []
            total_tracks = 0
        else:
            placeholders = ",".join(["%s"] * len(id_filter))
            cur.execute(
                base_select + " WHERE s.item_id IN (" + placeholders + ") ORDER BY s.item_id ASC",
                tuple(id_filter),
            )
            rows = cur.fetchall()
            total_tracks = len(rows)
        has_more = False
        next_page = None
    else:
        cur.execute("SELECT COUNT(*) AS n FROM score", ())
        total_tracks = cur.fetchone()['n']
        offset = (page - 1) * limit
        cur.execute(
            base_select + " ORDER BY s.item_id ASC LIMIT %s OFFSET %s",
            (limit, offset),
        )
        rows = cur.fetchall()
        has_more = (offset + len(rows)) < total_tracks
        next_page = page + 1 if has_more else None

    id_map, proj = load_map_projection(_DEFAULT_PROJECTION_NAME)
    if id_map and proj is not None:
        umap_lookup = {
            iid: (float(proj[i][0]), float(proj[i][1]))
            for i, iid in enumerate(id_map)
        }
    else:
        umap_lookup = {}

    tracks = []
    for r in rows:
        ux, uy = umap_lookup.get(r['item_id'], (None, None))
        t = {
            "id": r['item_id'],
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

    return jsonify({
        "tracks": tracks,
        "total_tracks": total_tracks,
        "provider_type": config.MEDIASERVER_TYPE,
        "has_more": has_more,
        "next_page": next_page,
    })
