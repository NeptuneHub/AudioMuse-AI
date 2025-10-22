import json
import math
import logging
from flask import Blueprint, jsonify, render_template, request, Response, current_app
import numpy as np
import gzip

from app_helper import get_db, load_map_projection
import config

# Try to reuse projection helpers from song_alchemy
try:
    from tasks.song_alchemy import _project_with_umap, _project_to_2d, _project_aligned_add_sub, _project_with_discriminant
except Exception:
    # Fallbacks will be used if import fails
    _project_with_umap = None
    _project_to_2d = None
    _project_aligned_add_sub = None
    _project_with_discriminant = None

logger = logging.getLogger(__name__)

map_bp = Blueprint('map_bp', __name__)

# In-memory cached JSON (and compressed) for fast map responses.
# Keys: '100','75','50','25' each maps to dict with 'json_bytes' and 'json_gzip_bytes' and 'projection'
MAP_JSON_CACHE = {}


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
        if i in seen: continue
        seen.add(int(i))
        out.append(items[int(i)])
    return out


def build_map_cache():
    """Load all tracks & embeddings from DB, compute 2D projection (prefer precomputed),
    and build cached JSON blobs for 100/75/50/25 percent samples. This should be called
    once at startup inside app.app_context()."""
    global MAP_JSON_CACHE
    logger = logging.getLogger(__name__)
    logger.info('Building map JSON cache (this reads the DB once).')

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
    ids = []
    embs = []
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
        ids.append(str(item_id))
        embs.append(emb)
        items.append({'item_id': str(item_id), 'title': title, 'artist': author, 'mood_vector': mood_vector, 'embedding': emb})

    if not items:
        # empty cache
        MAP_JSON_CACHE = {}
        logger.warning('No items found to build map cache.')
        return

    # Try to use precomputed projection if available
    id_map, proj = None, None
    try:
        id_map, proj = load_map_projection('main_map')
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
            if '_project_with_umap' in globals() and globals().get('_project_with_umap') is not None:
                try:
                    projections = globals()['_project_with_umap']([v for v in mat])
                    used = 'umap'
                except Exception as e:
                    logger.debug('UMAP helper failed during cache build: %s', e)
            if projections is None and '_project_to_2d' in globals() and globals().get('_project_to_2d') is not None:
                try:
                    projections = globals()['_project_to_2d']([v for v in mat])
                    used = 'pca'
                except Exception as e:
                    logger.debug('PCA helper failed during cache build: %s', e)
            if projections is None:
                projections = [(0.0, 0.0) for _ in missing_indices]
                used = 'none'

            for idx, coord in zip(missing_indices, projections):
                coords_by_id[str(items[idx]['item_id'])] = (float(coord[0]), float(coord[1]))
            if used_projection == 'none':
                used_projection = used
        except Exception as e:
            logger.exception('Failed to compute missing projections: %s', e)

    # Build full list of lightweight items and drop heavy embedding vectors
    full_light = []
    for it in items:
        iid = str(it['item_id'])
        coord = coords_by_id.get(iid, (0.0, 0.0))
        light = {
            'artist': it.get('artist') or '',
            'embedding_2d': _round_coord(coord),
            'item_id': iid,
            'mood_vector': _pick_top_mood(it.get('mood_vector')),
            'title': it.get('title') or ''
        }
        full_light.append(light)

    # create sampled versions
    n = len(full_light)
    frac_map = {'100': 1.0, '75': 0.75, '50': 0.5, '25': 0.25}
    new_cache = {}
    for k, frac in frac_map.items():
        sampled = _sample_items(full_light, frac)
        payload = {'items': sampled, 'projection': used_projection, 'count': len(sampled)}
        js = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        try:
            gz = gzip.compress(js)
        except Exception:
            gz = None
        new_cache[k] = {'json_bytes': js, 'json_gzip_bytes': gz, 'projection': used_projection, 'count': len(sampled)}

    MAP_JSON_CACHE = new_cache
    logger.info('Map JSON cache built: %d total items; cache sizes: %s', n, {k: v['count'] for k, v in MAP_JSON_CACHE.items()})


def init_map_cache():
    """Public initializer that can be called from app startup to build the cache."""
    try:
        build_map_cache()
    except Exception:
        logging.getLogger(__name__).exception('init_map_cache failed')


@map_bp.route('/map')
def map_ui():
    """Serve the map UI page."""
    return render_template('map.html')


def _fetch_genre_samples(conn, genre, limit):
    cur = conn.cursor()
    # mood_vector is stored as 'label:score,label2:score' so use ILIKE for simple match
    try:
        cur.execute("""
            SELECT s.item_id, s.title, s.author, s.mood_vector, s.other_features, e.embedding
            FROM score s
            JOIN embedding e ON s.item_id = e.item_id
            WHERE s.mood_vector ILIKE %s
            LIMIT %s
        """, (f"%{genre}%", limit))
        rows = cur.fetchall()
    finally:
        cur.close()
    return rows


def _rows_to_items(rows):
    items = []
    for r in rows:
        # r is a tuple-like from psycopg2; map by index to be robust
        item_id = r[0]
        title = r[1]
        author = r[2]
        mood_vector = r[3]
        other_features = r[4]
        embedding_blob = r[5]
        if embedding_blob is None:
            continue
        emb = np.frombuffer(embedding_blob, dtype=np.float32)
        items.append({
            'item_id': item_id,
            'title': title,
            'author': author,
            'mood_vector': mood_vector,
            'other_features': other_features,
            'embedding': emb
        })
    return items


@map_bp.route('/api/map', methods=['GET'])
def map_api():
    """Return up to 2000 embeddings sampled across configured genres, projected to 2D.

    Response: JSON list of items with title, artist, embedding_2d, mood_vector, other_feature
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

    # Prefer serving gzip if the client accepts it and gzip bytes were built
    accept_enc = request.headers.get('Accept-Encoding', '')
    gz = entry.get('json_gzip_bytes')
    if gz and 'gzip' in accept_enc.lower():
        resp = Response(gz, mimetype='application/json; charset=utf-8')
        resp.headers['Content-Encoding'] = 'gzip'
        resp.headers['Content-Length'] = str(len(gz))
        return resp

    # Fallback to plain JSON
    resp = Response(entry['json_bytes'], mimetype='application/json; charset=utf-8')
    resp.headers['Content-Length'] = str(len(entry['json_bytes']))
    return resp


@map_bp.route('/api/map_cache_status', methods=['GET'])
def map_cache_status():
    """Return diagnostic information about the in-memory map JSON cache."""
    try:
        if not MAP_JSON_CACHE:
            return jsonify({'ok': False, 'reason': 'empty_cache', 'buckets': {}})
        info = {}
        for k, v in MAP_JSON_CACHE.items():
            info[k] = {'count': v.get('count', 0), 'json_bytes': len(v.get('json_bytes') or b''), 'projection': v.get('projection')}
        return jsonify({'ok': True, 'buckets': info}), 200
    except Exception as e:
        logger.exception('map_cache_status failed: %s', e)
        return jsonify({'ok': False, 'reason': 'exception', 'error': str(e)}), 500


@map_bp.route('/api/rebuild_map_cache', methods=['POST'])
def rebuild_map_cache():
    """Trigger a synchronous rebuild of the in-memory cache. Useful for debugging.
    Note: this reads the DB and may take time."""
    try:
        build_map_cache()
        return jsonify({'ok': True, 'message': 'map cache rebuilt'}), 200
    except Exception as e:
        logger.exception('rebuild_map_cache failed: %s', e)
        return jsonify({'ok': False, 'error': str(e)}), 500
