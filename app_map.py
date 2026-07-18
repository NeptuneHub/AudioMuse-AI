# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Library map Flask blueprint (map_bp) serving the 2D song projection.

Renders the ``/map`` page and streams the projected library as JSON at
``/api/map``, reusing the UMAP / discriminant projection helpers from
``tasks.alchemy_projections`` and the stored projection from ``app_helper``.

Main Features:
* Serves the map at four density levels (100/75/50/25 percent), each cached
  in memory as pre-serialized JSON plus a gzip-compressed copy for fast reads.
* Endpoints to report cache status and to rebuild the cache on demand; songs
  are labelled by their top mood parsed from the stored mood_vector string.
* Multi-server: responses are filtered per request to the selected server's
  catalogue; the shared cache always holds the full union.
"""

import gc
import json
import math
import time
import logging
from flask import Blueprint, jsonify, render_template, request, Response
import numpy as np
import gzip

from database import get_db
from app_helper import load_map_projection
import app_server_context

# Try to reuse the shared projection helpers
try:
    from tasks.alchemy_projections import (
        _project_with_umap,
        _project_to_2d,
        _project_with_discriminant,
    )
except Exception:
    # Fallbacks will be used if import fails
    _project_with_umap = None
    _project_to_2d = None
    _project_with_discriminant = None

logger = logging.getLogger(__name__)

map_bp = Blueprint('map_bp', __name__)

# In-memory cached JSON (and compressed) for fast map responses.
# Keys: '100','75','50','25' each maps to dict with 'json_bytes' and 'json_gzip_bytes' and 'projection'
MAP_JSON_CACHE = {}

# Per-server precomputed map buckets, keyed by (server_key, percent). The union
# MAP_JSON_CACHE above stays in canonical (fp_) ids; this holds each server's OWN
# provider ids, already serialized and gzipped, so a canonicalized/multi-server
# request streams precomputed bytes instead of translating the whole catalogue on
# every call. Built for the default server at cache-build time, lazily for others.
MAP_SERVER_JSON_CACHE = {}

# Memoized canonical-id probe. Canonicalization is one-way, so a True is sticky
# forever; a False is re-probed at most once per TTL. This keeps the map fast path
# from seq-scanning score on every request of a not-yet-canonicalized library.
_HAS_CANONICAL_IDS = None
_HAS_CANONICAL_CHECKED_AT = 0.0
_HAS_CANONICAL_TTL = 60.0


def _catalogue_has_canonical_ids():
    """True when score holds canonical fp_ ids (memoized; fails closed on error)."""
    global _HAS_CANONICAL_IDS, _HAS_CANONICAL_CHECKED_AT
    if _HAS_CANONICAL_IDS:
        return True
    now = time.monotonic()
    if _HAS_CANONICAL_IDS is False and (now - _HAS_CANONICAL_CHECKED_AT) < _HAS_CANONICAL_TTL:
        return False
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM score WHERE item_id LIKE 'fp\\_%%')"
            )
            result = bool(cur.fetchone()[0])
    except Exception:
        logger.exception("Canonical-id probe failed; failing closed")
        return True
    _HAS_CANONICAL_IDS = result
    _HAS_CANONICAL_CHECKED_AT = now
    return result


def _pick_top_mood(mood_vector_str):
    """Return top mood label from 'label:score,label2:score' string.
    If parsing fails, return 'unknown'."""
    if not mood_vector_str:
        return 'unknown'
    try:
        parts = mood_vector_str.split(',')
        best_label = None
        best_val = -float('inf')
        for p in parts:
            if ':' not in p:
                continue
            lab, val = p.split(':', 1)
            try:
                v = float(val)
            except Exception:
                v = 0.0
            if v > best_val:
                best_val = v
                best_label = lab
        return best_label or 'unknown'
    except Exception:
        return 'unknown'


def _round_coord(coord):
    try:
        return [round(float(coord[0]), 3), round(float(coord[1]), 3)]
    except Exception:
        return [0.0, 0.0]


def _sample_items(items, fraction):
    """Deterministic downsample: choose M = max(1, int(len* fraction)) items using linspace indices."""
    n = len(items)
    if n == 0:
        return []
    m = max(1, int(math.floor(n * fraction)))
    if m >= n:
        return items.copy()
    idxs = np.linspace(0, n - 1, m, dtype=int)
    seen = set()
    out = []
    for i in idxs:
        if i in seen:
            continue
        seen.add(int(i))
        out.append(items[int(i)])
    return out


def build_map_cache():
    """Load all tracks & embeddings from DB, compute 2D projection (prefer precomputed),
    and build cached JSON blobs for 100/75/50/25 percent samples. This should be called
    once at startup inside app.app_context()."""
    global MAP_JSON_CACHE
    global _HAS_CANONICAL_IDS, _HAS_CANONICAL_CHECKED_AT
    logger = logging.getLogger(__name__)
    logger.info('Building map JSON cache (this reads the DB once).')

    # A rebuild is the moment the catalogue's id space can have flipped legacy->fp_
    # (e.g. a canonicalization just ran). Reset the memoized probe so the fast path
    # can never serve a stale "no canonical ids" verdict and leak raw fp_ bytes.
    _HAS_CANONICAL_IDS = None
    _HAS_CANONICAL_CHECKED_AT = 0.0

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT s.item_id, s.title, s.author, s.mood_vector, e.embedding
            FROM score s
            JOIN embedding e ON s.item_id = e.item_id
        """)
        rows = cur.fetchall()
    finally:
        cur.close()

    items = []
    for r in rows:
        # r: item_id, title, author, mood_vector, embedding_blob
        item_id = r[0]
        title = r[1]
        author = r[2]
        mood_vector = r[3]
        emb_blob = r[4]
        if emb_blob is None:
            continue
        try:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
        except Exception:
            # fallback if already stored as list
            try:
                emb = np.array(r[4], dtype=np.float32)
            except Exception:
                continue
        items.append(
            {
                'item_id': str(item_id),
                'title': title,
                'artist': author,
                'mood_vector': mood_vector,
                'embedding': emb,
            }
        )

    if not items:
        # empty cache
        MAP_JSON_CACHE = {}
        logger.warning('No items found to build map cache.')
        return

    # Try to use precomputed projection if available
    id_map, proj = None, None
    try:
        id_map, proj = load_map_projection('main_map', force_reload=True)
    except Exception as e:
        logger.debug('load_map_projection failed: %s', e)

    coords_by_id = {}
    used_projection = 'none'
    if id_map is not None and proj is not None and len(id_map) > 0:
        # id_map likely lists item ids in same order as proj rows
        try:
            for iid, coord in zip(id_map, proj.tolist()):
                coords_by_id[str(iid)] = (float(coord[0]), float(coord[1]))
            used_projection = 'precomputed'
        except Exception:
            coords_by_id = {}

    # For items still missing coordinates, compute projection on-the-fly using available helpers
    missing_indices = [i for i, it in enumerate(items) if str(it['item_id']) not in coords_by_id]
    if missing_indices:
        try:
            # Build matrix of missing emb
            mat = np.vstack([items[i]['embedding'] for i in missing_indices])
            projections = None
            used = 'none'
            # prefer UMAP helper if present
            if (
                '_project_with_umap' in globals()
                and globals().get('_project_with_umap') is not None
            ):
                try:
                    projections = globals()['_project_with_umap'](mat)
                    used = 'umap'
                except Exception as e:
                    logger.debug('UMAP helper failed during cache build: %s', e)
            if (
                projections is None
                and '_project_to_2d' in globals()
                and globals().get('_project_to_2d') is not None
            ):
                try:
                    projections = globals()['_project_to_2d'](mat)
                    used = 'pca'
                except Exception as e:
                    logger.debug('PCA helper failed during cache build: %s', e)
            if projections is None:
                projections = [(0.0, 0.0) for _ in missing_indices]
                used = 'none'

            del mat

            for idx, coord in zip(missing_indices, projections):
                coords_by_id[str(items[idx]['item_id'])] = (float(coord[0]), float(coord[1]))
            if used_projection == 'none':
                used_projection = used
        except Exception:
            logger.exception('Failed to compute missing projections')

    for it in items:
        it.pop('embedding', None)

    full_light = []
    for it in items:
        iid = str(it['item_id'])
        coord = coords_by_id.get(iid, (0.0, 0.0))
        light = {
            'artist': it.get('artist') or '',
            'embedding_2d': _round_coord(coord),
            'item_id': iid,
            'mood_vector': _pick_top_mood(it.get('mood_vector')),
            'title': it.get('title') or '',
        }
        full_light.append(light)
    del items
    gc.collect()

    n = len(full_light)
    frac_map = {'100': 1.0, '75': 0.75, '50': 0.5, '25': 0.25}
    new_cache = {}
    for k, frac in frac_map.items():
        sampled = _sample_items(full_light, frac)
        payload = {'items': sampled, 'projection': used_projection, 'count': len(sampled)}
        js = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        try:
            gz = gzip.compress(js)
            entry = {'json_gzip_bytes': gz, 'projection': used_projection, 'count': len(sampled)}
        except Exception:
            entry = {'json_bytes': js, 'projection': used_projection, 'count': len(sampled)}
        else:
            del js
        new_cache[k] = entry

    MAP_JSON_CACHE = new_cache
    MAP_SERVER_JSON_CACHE.clear()
    _warm_server_buckets()
    logger.info(
        'Map JSON cache built: %d total items; cache sizes: %s',
        n,
        {k: v['count'] for k, v in MAP_JSON_CACHE.items()},
    )


def _translated_bucket(entry, server_id):
    """A cached union bucket rewritten to ``server_id``'s provider ids, dropping
    fp_/unmapped rows, and re-serialized + gzipped. None when the bucket is empty.

    server_id None means the default server. Fails CLOSED on a registry error:
    legacy provider ids stay as identity, fp_ ids are dropped, never leaked.
    """
    raw = entry.get('json_gzip_bytes')
    raw = gzip.decompress(raw) if raw else entry.get('json_bytes')
    if not raw:
        return None
    from tasks.mediaserver import registry
    from tasks.simhash import is_fingerprint_id

    payload = json.loads(raw)
    items = payload.get('items') or []
    ids = [it.get('item_id') for it in items if it.get('item_id')]
    try:
        mapping = registry.translate_ids(ids, server_id)
    except Exception:
        logger.exception('Map id translation failed; dropping fp_ rows to avoid a leak')
        mapping = {i: i for i in ids if not is_fingerprint_id(i)}
    kept = []
    for it in items:
        provider_id = mapping.get(it.get('item_id'))
        if provider_id is None:
            continue
        it['item_id'] = provider_id
        kept.append(it)
    payload['items'] = kept
    payload['count'] = len(kept)
    js = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    try:
        gz = gzip.compress(js)
    except Exception:
        return {'json_bytes': js, 'projection': payload.get('projection'), 'count': len(kept)}
    return {'json_gzip_bytes': gz, 'projection': payload.get('projection'), 'count': len(kept)}


def _warm_server_buckets():
    """Precompute EVERY configured server's translated buckets (plus the default)
    so the first /api/map for ANY server streams bytes instead of translating the
    whole catalogue live. Runs at build time in the cache-build background thread.

    Skipped for a legacy single-server install with no canonical ids, where the
    zero-cost verbatim fast path already applies.
    """
    from tasks.mediaserver import registry

    try:
        needs = _catalogue_has_canonical_ids() or registry.has_secondary_servers()
    except Exception:
        logger.exception('Map pre-warm scope probe failed; warming defensively')
        needs = True
    if not needs:
        return

    try:
        servers = registry.list_servers()
    except Exception:
        logger.exception('Map pre-warm could not list servers; warming default only')
        servers = []
    try:
        default_id = registry.get_default_server_id()
    except Exception:
        default_id = None

    def _warm(server_key, server_id):
        for k, entry in MAP_JSON_CACHE.items():
            try:
                translated = _translated_bucket(entry, server_id)
            except Exception:
                logger.exception('Map pre-warm failed for server %s bucket %s', server_key, k)
                continue
            if translated is not None:
                MAP_SERVER_JSON_CACHE[(server_key, k)] = translated

    # None resolves to the default server, keyed '__default__' for requests with no
    # ?server=. Each configured server is also warmed under its own id so an explicit
    # ?server=<id> hits the cache; the default's own id reuses the '__default__' work
    # (same translation target) rather than translating the whole catalogue twice.
    _warm('__default__', None)
    for server in servers:
        sid = server['server_id']
        if sid == default_id:
            for k in MAP_JSON_CACHE:
                mirror = MAP_SERVER_JSON_CACHE.get(('__default__', k))
                if mirror is not None:
                    MAP_SERVER_JSON_CACHE[(sid, k)] = mirror
        else:
            _warm(sid, sid)


def init_map_cache():
    """Public initializer that can be called from app startup to build the cache."""
    try:
        build_map_cache()
    except Exception:
        logging.getLogger(__name__).exception('init_map_cache failed')


@map_bp.route('/map')
def map_ui():
    """
    Music map UI page.
    ---
    tags:
      - Map
    summary: HTML page for the 2D music map (UMAP/projection of song embeddings).
    responses:
      200:
        description: HTML page rendered with no-cache headers.
    """
    resp = render_template('map.html', title='AudioMuse-AI - Music Map', active='map')
    # Ensure the rendered page is not cached by browsers or intermediary caches.
    # We return a Response object below so Flask will set the appropriate headers.
    from flask import make_response

    response = make_response(resp)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


def _bucket_response(entry):
    """Stream a precomputed bucket's bytes, honoring Accept-Encoding: gzip."""
    accept_enc = request.headers.get('Accept-Encoding', '')
    gz = entry.get('json_gzip_bytes')
    if gz and 'gzip' in accept_enc.lower():
        resp = Response(gz, mimetype='application/json; charset=utf-8')
        resp.headers['Content-Encoding'] = 'gzip'
        resp.headers['Content-Length'] = str(len(gz))
    elif gz:
        raw = gzip.decompress(gz)
        resp = Response(raw, mimetype='application/json; charset=utf-8')
        resp.headers['Content-Length'] = str(len(raw))
    elif entry.get('json_bytes'):
        raw = entry['json_bytes']
        resp = Response(raw, mimetype='application/json; charset=utf-8')
        resp.headers['Content-Length'] = str(len(raw))
    else:
        return jsonify({'items': [], 'projection': 'none'})
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@map_bp.route('/api/map', methods=['GET'])
def map_api():
    """
    Music map data.
    ---
    tags:
      - Map
    summary: Return embeddings projected to 2D, sampled across configured genres, for the music-map UI.
    description: |
      Served exclusively from the in-memory `MAP_JSON_CACHE` built at startup
      (or rebuilt via `/api/rebuild_map_cache`). Supports four sampling
      buckets - 25/50/75/100 percent of the cached set - plus a legacy `n`
      parameter that maps the closest bucket. Honors `Accept-Encoding: gzip`
      when the cache contains a precompressed payload.
    parameters:
      - name: percent
        in: query
        schema:
          type: string
          enum: ["25", "50", "75", "100"]
        description: Percentage bucket of cached items to return. Default 25.
      - name: p
        in: query
        schema: { type: string }
        description: Alias for `percent`.
      - name: n
        in: query
        schema: { type: integer }
        description: Legacy parameter - mapped to the nearest available bucket.
    responses:
      200:
        description: JSON payload with `items` (each having `embedding_2d`, `title`, `author`, `mood_vector`, `other_features`) and `projection` name.
    """
    # Serve exclusively from the in-memory MAP_JSON_CACHE built at startup.
    # Accept either explicit percent param (?percent=25|50|75|100) or legacy ?n=<count>.
    if not MAP_JSON_CACHE:
        # Cache not built or empty
        return jsonify({'items': [], 'projection': 'none'})

    pct = None
    pct_param = request.args.get('percent') or request.args.get('p')
    if pct_param:
        pct = str(pct_param)
    else:
        # legacy n param mapping to nearest bucket
        n_param = request.args.get('n')
        if n_param:
            try:
                n = int(n_param)
                full = MAP_JSON_CACHE.get('100', {}).get('count') or 0
                if full <= 0:
                    pct = '100'
                else:
                    r = float(n) / float(full)
                    if r <= 0.25:
                        pct = '25'
                    elif r <= 0.5:
                        pct = '50'
                    elif r <= 0.75:
                        pct = '75'
                    else:
                        pct = '100'
            except Exception:
                pct = '25'
        else:
            pct = '25'

    if pct not in MAP_JSON_CACHE:
        # fallback to closest available
        for k in ['25', '50', '75', '100']:
            if k in MAP_JSON_CACHE:
                pct = k
                break

    entry = MAP_JSON_CACHE.get(pct)
    if not entry:
        return jsonify({'items': [], 'projection': 'none'})

    # Multi-server: filter the cached union per request. Single-server installs
    # with no explicit selection keep the zero-cost pre-serialized path below.
    try:
        server_id = app_server_context.resolve_request_server_id()
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return jsonify({'error': 'Invalid server selection.'}), 400
    from tasks.mediaserver import registry

    # The legacy fast path streams score.item_id verbatim; that is safe only while
    # ids are still legacy provider ids. Once the catalogue is canonicalized (fp_
    # ids), or a secondary server exists, or a ?server= is selected, serve the
    # per-server bucket instead: it is precomputed ONCE to that server's provider
    # ids (fp_ dropped) and pre-gzipped, so every request streams bytes rather than
    # re-translating the whole catalogue. ALL configured servers are warmed at build
    # time; a lookup miss (a server added since the last build) builds+caches lazily.
    if server_id is not None or registry.has_secondary_servers() or _catalogue_has_canonical_ids():
        server_key = server_id or '__default__'
        cached = MAP_SERVER_JSON_CACHE.get((server_key, pct))
        if cached is None:
            cached = _translated_bucket(entry, server_id)
            if cached is None:
                return jsonify({'items': [], 'projection': 'none'})
            MAP_SERVER_JSON_CACHE[(server_key, pct)] = cached
        return _bucket_response(cached)

    return _bucket_response(entry)


@map_bp.route('/api/map_cache_status', methods=['GET'])
def map_cache_status():
    """
    Diagnostic info for the map cache.
    ---
    tags:
      - Map
    summary: Return per-bucket stats (count, payload size, projection algorithm) for the in-memory map cache.
    responses:
      200:
        description: Cache summary.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                buckets:
                  type: object
                  additionalProperties:
                    type: object
                    properties:
                      count:
                        type: integer
                      json_bytes:
                        type: integer
                      projection:
                        type: string
                reason:
                  type: string
                  description: Set to `empty_cache` when no buckets exist.
      500:
        description: Internal error.
    """
    try:
        if not MAP_JSON_CACHE:
            return jsonify({'ok': False, 'reason': 'empty_cache', 'buckets': {}})
        info = {}
        for k, v in MAP_JSON_CACHE.items():
            payload = v.get('json_gzip_bytes') or v.get('json_bytes') or b''
            info[k] = {
                'count': v.get('count', 0),
                'json_bytes': len(payload),
                'projection': v.get('projection'),
            }
        return jsonify({'ok': True, 'buckets': info}), 200
    except Exception:
        # Log the full exception (including stack) for diagnostics, but do not expose
        # internal exception details to API clients.
        logger.exception('map_cache_status failed')
        return jsonify({'ok': False, 'reason': 'exception', 'error': 'Internal server error'}), 500


@map_bp.route('/api/rebuild_map_cache', methods=['POST'])
def rebuild_map_cache():
    """
    Synchronously rebuild the map cache.
    ---
    tags:
      - Map
    summary: Re-read embeddings from the DB and rebuild every percent bucket.
    description: Synchronous; can take a while on large libraries. Useful for debugging.
    responses:
      200:
        description: Cache rebuilt successfully.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                message:
                  type: string
      500:
        description: Internal error during rebuild.
    """
    try:
        build_map_cache()
        return jsonify({'ok': True, 'message': 'map cache rebuilt'}), 200
    except Exception:
        # Log the full exception for debugging, but return a generic error to the caller.
        logger.exception('rebuild_map_cache failed')
        return jsonify({'ok': False, 'error': 'Internal server error'}), 500
