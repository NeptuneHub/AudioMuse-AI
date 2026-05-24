"""Mobile sync endpoint for companion client apps.

Exposes ``GET /api/sync`` — a paginated, read-only bulk export of the
analysis library (metadata, mood/energy, MusiCNN + CLAP embeddings, UMAP
2D coordinates, and deletion tombstones) that lets a client build and
incrementally maintain a local mirror of the library via the ``since``
query parameter.
"""
import base64
import logging
from datetime import datetime, timezone

import psycopg2.extras
from flask import Blueprint, request, jsonify

import config
from app_helper import get_db, load_map_projection


logger = logging.getLogger(__name__)

sync_bp = Blueprint('sync_bp', __name__)


_MAX_LIMIT = 1000
_DEFAULT_LIMIT = 500
_DEFAULT_PROJECTION_NAME = 'main_map'


@sync_bp.route('/api/sync', methods=['GET'])
def sync_endpoint():
    if config.MEDIASERVER_TYPE == 'mpd':
        return jsonify({
            "error": "mpd is not yet supported by the mobile sync endpoint"
        }), 501

    page = max(1, request.args.get('page', 1, type=int))
    limit = max(1, min(_MAX_LIMIT, request.args.get('limit', _DEFAULT_LIMIT, type=int)))
    include_embeddings = request.args.get('include_embeddings', 'true').lower() != 'false'

    since_raw = request.args.get('since')
    since_dt = None
    if since_raw:
        try:
            since_dt = datetime.fromisoformat(since_raw.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return jsonify({
                "error": "Invalid 'since' parameter. Use ISO 8601 format."
            }), 400
        if since_dt.tzinfo is not None:
            since_dt = since_dt.astimezone(timezone.utc).replace(tzinfo=None)

    offset = (page - 1) * limit
    clap_on = include_embeddings and config.CLAP_ENABLED

    try:
        conn = get_db()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # 1. total_tracks for the (optionally-filtered) library
            if since_dt is not None:
                cur.execute(
                    "SELECT COUNT(*) AS n FROM score WHERE updated_at > %s",
                    (since_dt,),
                )
            else:
                cur.execute("SELECT COUNT(*) AS n FROM score", ())
            total_tracks = cur.fetchone()['n']

            # 2. The page itself — build SELECT/JOIN dynamically to skip
            # the embedding tables entirely when not requested.
            select_extra = ""
            join_extra = ""
            if include_embeddings:
                select_extra += ", e.embedding AS musicnn_blob"
                join_extra += " LEFT JOIN embedding e ON e.item_id = s.item_id"
            if clap_on:
                select_extra += ", c.embedding AS clap_blob"
                join_extra += " LEFT JOIN clap_embedding c ON c.item_id = s.item_id"

            where_clause = "WHERE s.updated_at > %s" if since_dt is not None else ""
            page_sql = (
                "SELECT s.item_id, s.title, s.author, s.album, s.album_artist, "
                "s.year, s.tempo, s.key, s.scale, s.mood_vector, s.other_features, "
                "s.energy, s.rating, s.updated_at" + select_extra +
                " FROM score s" + join_extra +
                (" " + where_clause if where_clause else "") +
                " ORDER BY s.item_id ASC LIMIT %s OFFSET %s"
            )
            page_params = ([since_dt] if since_dt is not None else []) + [limit, offset]
            cur.execute(page_sql, tuple(page_params))
            rows = cur.fetchall()

            # 3. deleted_ids only on incremental syncs
            deleted_ids = []
            if since_dt is not None:
                cur.execute(
                    "SELECT item_id FROM deleted_tracks WHERE deleted_at > %s "
                    "ORDER BY deleted_at ASC",
                    (since_dt,),
                )
                deleted_ids = [r['item_id'] for r in cur.fetchall()]

        # 4. UMAP per-page lookup (load_map_projection is cached at startup).
        # Kept inside try/except so any decode/encode failure still returns
        # the JSON 500 contract instead of Flask's default HTML error page.
        id_map, proj = load_map_projection(_DEFAULT_PROJECTION_NAME)
        if id_map and proj is not None:
            umap_lookup = {
                iid: (float(proj[i][0]), float(proj[i][1]))
                for i, iid in enumerate(id_map)
            }
        else:
            umap_lookup = {}

        # 5. Per-track assembly + base64 encoding
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
                "updated_at": r['updated_at'].isoformat() if r['updated_at'] else None,
            }
            if include_embeddings:
                mb = r['musicnn_blob']
                t['embedding'] = base64.b64encode(bytes(mb)).decode('ascii') if mb else None
                if clap_on:
                    cb = r['clap_blob']
                    t['clap_embedding'] = base64.b64encode(bytes(cb)).decode('ascii') if cb else None
            tracks.append(t)

        has_more = (offset + len(rows)) < total_tracks
        return jsonify({
            "tracks": tracks,
            "deleted_ids": deleted_ids,
            "total_tracks": total_tracks,
            "provider_type": config.MEDIASERVER_TYPE,
            "has_more": has_more,
            "next_page": page + 1 if has_more else None,
        })
    except Exception:
        logger.exception(
            "GET /api/sync failed (page=%s limit=%s since=%s)",
            page, limit, since_raw,
        )
        return jsonify({"error": "Internal server error"}), 500
