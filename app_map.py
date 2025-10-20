import json
import math
import logging
from flask import Blueprint, jsonify, render_template, request
import numpy as np

from app_helper import get_db
import config
from app_helper import load_map_projection

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
    # Allow caller to override count via ?n=NN (safe-guarded)
    try:
        # Default remains 2000 when not specified, but we no longer enforce an upper cap.
        requested = int(request.args.get('n', 2000))
    except Exception:
        requested = 2000
    TARGET = max(1, requested)
    conn = get_db()

    # Prepare per-genre target
    genres = config.STRATIFIED_GENRES or []
    per_genre = max(1, TARGET // max(1, len(genres))) if genres else TARGET

    collected = []
    cur = conn.cursor()
    try:
        if genres:
            for g in genres:
                rows = _fetch_genre_samples(conn, g, per_genre)
                collected.extend(rows)

        # If not enough, fetch additional tracks with embeddings (no mood filter)
        if len(collected) < TARGET:
            need = TARGET - len(collected)
            cur.execute("""
                SELECT s.item_id, s.title, s.author, s.mood_vector, s.other_features, e.embedding
                FROM score s
                JOIN embedding e ON s.item_id = e.item_id
                LIMIT %s
            """, (need,))
            extra = cur.fetchall()
            collected.extend(extra)

    finally:
        cur.close()

    # Convert to item dicts with numpy embeddings
    items = _rows_to_items(collected)

    # Limit to TARGET
    items = items[:TARGET]

    if not items:
        return jsonify({'items': []})

    # Try to load precomputed projection (id map + projection) from DB first
    id_map, proj = load_map_projection('main_map')
    if id_map is not None and proj is not None and len(id_map) > 0:
        # Build response by mapping projection rows to items by id (support subsets)
        # zip(id_map, proj.tolist()) pairs ids with coordinates in stored order.
        id_to_coord = {str(i): tuple(coord) for i, coord in zip(id_map, proj.tolist())}
        resp_items = []
        used_any = False
        for it in items:
            coord = id_to_coord.get(str(it['item_id']))
            if coord is None:
                coord = (0.0, 0.0)
            else:
                used_any = True
            resp_items.append({
                'item_id': it.get('item_id'),
                'title': it.get('title'),
                'artist': it.get('author'),
                'embedding_2d': [float(coord[0]), float(coord[1])],
                'mood_vector': it.get('mood_vector'),
                'other_feature': it.get('other_features')
            })
        # If at least one item was matched from precomputed projection, return precomputed
        if used_any:
            return jsonify({'items': resp_items, 'projection': 'precomputed'})

    # Build numpy matrix
    mat = np.vstack([it['embedding'] for it in items])

    # Choose projection method. Prefer UMAP if available, else PCA via _project_to_2d
    projections = None
    used = 'none'
    try:
        if _project_with_umap is not None:
            projections = _project_with_umap([v for v in mat])
            used = 'umap'
    except Exception as e:
        logger.warning('UMAP projection failed: %s', e)
        projections = None

    if projections is None:
        try:
            if _project_to_2d is not None:
                projections = _project_to_2d([v for v in mat])
                used = 'pca'
        except Exception as e:
            logger.warning('PCA projection failed: %s', e)
            projections = None

    if projections is None:
        # As a last resort, return zeros
        projections = [(0.0, 0.0) for _ in items]
        used = 'none'

    # The projection helpers in tasks.song_alchemy already center and scale
    # the output to a comparable range. Use the projections as returned
    # (ensure shape Nx2). This preserves precision and matches
    # song_alchemy behaviour.
    proj_arr = np.array(projections, dtype=float)
    if proj_arr.ndim != 2 or proj_arr.shape[1] != 2:
        proj_arr = np.zeros((len(items), 2), dtype=float)
    scaled = proj_arr.tolist()

    # Build response
    resp_items = []
    for it, coord in zip(items, scaled):
        resp_items.append({
            'item_id': it.get('item_id'),
            'title': it.get('title'),
            'artist': it.get('author'),
            'embedding_2d': [float(coord[0]), float(coord[1])],
            'mood_vector': it.get('mood_vector'),
            'other_feature': it.get('other_features')
        })

    return jsonify({'items': resp_items, 'projection': used})
