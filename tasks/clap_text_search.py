# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Natural-language ("search by text") lookup over stored CLAP audio embeddings.

Serves the text-to-song search feature: an in-process cache of CLAP audio
embeddings plus their IVF index is loaded from the DB, and a query string is
embedded with the CLAP text encoder (tasks.clap_analyzer) and matched against it.
Manages its own warm-up and idle unload of the text model to bound worker RSS.

Main Features:
* build_and_store / load / refresh of the CLAP embedding cache and its IVF index.
* search_by_text: embed the query and return ranked nearest songs.
* search_by_embedding: the same ranked lookup for a caller that already holds a
  512-value CLAP vector (an audio clip embedded by the DCLAP tower, a steered
  query); search_by_text is a thin wrapper over it.
* search_by_song: songs that sound like a seed song, seeded with the seed's
  exact float32 clap_embedding row (no text model needed; it also serves songs
  analysed after the last index build), with the MusiCNN similar-song rules: the
  seed and any copy of it (same name or a near-identical vector) are never
  returned, the seed never uses an artist slot, and the per-artist cap follows
  eliminate_duplicates. With radius_similarity (default SIMILARITY_RADIUS_DEFAULT,
  as MusiCNN) a larger deduplicated pool goes through the shared radius walk, which
  orders the songs in distance buckets and applies MAX_SONGS_PER_ARTIST itself; a
  walk failure is logged and answers with the plain order. The index is read from
  one snapshot, and the pool is fetched deeper once only when the first one was
  not exhausted.
* picker_where: the song-picker filter (songs with a DCLAP embedding), offered
  only while the index is loaded.
* warmup_text_search_model with an idle-unload timer, plus a persisted top-queries
  table (ensure_text_search_queries_table) used to pre-warm popular searches.
"""

import logging
import sys
import time

import numpy as np
from psycopg2.extras import DictCursor
from typing import Dict, List, Optional
import config

from .idle_unload import IdleUnloadTimer
from .radius_walk_helper import execute_radius_walk
from .search_shaping import build_capped_results as _build_capped_results
from .search_shaping import cosine_distance, to_unit_vector
from .search_shaping import overfetch_size

logger = logging.getLogger(__name__)

_CLAP_CACHE = {'loaded': False}

_CLAP_INDEX_CACHE = {'index': None, 'id_map': None, 'reverse_id_map': None, 'loaded': False}

_TOP_QUERIES_CACHE = {'queries': [], 'ready': False, 'computing': False}

_WARMUP_DURATION = None
_TIMER = IdleUnloadTimer()


def get_clap_cache_size() -> int:
    if _CLAP_INDEX_CACHE['loaded'] and _CLAP_INDEX_CACHE['id_map'] is not None:
        return len(_CLAP_INDEX_CACHE['id_map'])
    return 0


def _fetch_clap_metadata(item_ids: list) -> Dict[str, Dict[str, str]]:
    from .commons import fetch_track_metadata_map

    return fetch_track_metadata_map(item_ids)


def _load_clap_index_from_db() -> bool:
    from config import CLAP_EMBEDDING_DIMENSION, IVF_METRIC
    from .index_build_helpers import load_index_into_cache

    if not load_index_into_cache(
        'clap_index', CLAP_EMBEDDING_DIMENSION, IVF_METRIC, 'CLAP', _CLAP_INDEX_CACHE
    ):
        return False
    _CLAP_CACHE['loaded'] = True
    return True


def build_and_store_clap_index(db_conn=None):
    from database import get_db
    from config import CLAP_EMBEDDING_DIMENSION, IVF_METRIC
    from .index_build_helpers import build_and_store_index_streaming

    if db_conn is None:
        db_conn = get_db()

    return build_and_store_index_streaming(
        db_conn,
        source_table="clap_embedding",
        source_column="embedding",
        dim=CLAP_EMBEDDING_DIMENSION,
        target_table="clap_index_data",
        index_name="clap_index",
        metric=IVF_METRIC,
        label="CLAP",
    )


def _unload_expired():
    from .clap_analyzer import unload_clap_model, is_clap_text_loaded

    if is_clap_text_loaded():
        logger.info("Warm cache timer expired - unloading CLAP text model")
        unload_clap_model()


def warmup_text_search_model():
    from .clap_analyzer import initialize_clap_text_model, is_clap_text_loaded

    global _WARMUP_DURATION
    with _TIMER.lock():
        if _WARMUP_DURATION is None:
            _WARMUP_DURATION = config.CLAP_TEXT_SEARCH_WARMUP_DURATION

        if not is_clap_text_loaded():
            logger.info("Warming up CLAP text model for text search (not loading audio model)...")
            success = initialize_clap_text_model()
            if not success:
                return {'loaded': False, 'expiry_seconds': 0}

        if _TIMER.arm(_WARMUP_DURATION, _unload_expired):
            logger.info(f"Started warm cache timer ({_WARMUP_DURATION}s)")
        else:
            logger.debug(f"Reset warm cache timer ({_WARMUP_DURATION}s)")

    return {'loaded': True, 'expiry_seconds': _WARMUP_DURATION}


def get_warm_cache_status() -> Dict:
    from .clap_analyzer import is_clap_model_loaded

    expiry = _TIMER.expiry()

    if expiry is None or not is_clap_model_loaded():
        return {'active': False, 'seconds_remaining': 0}

    remaining = max(0, int(expiry - time.time()))
    return {'active': True, 'seconds_remaining': remaining}


def load_clap_cache_from_db():
    from config import CLAP_ENABLED

    if not CLAP_ENABLED:
        logger.info("CLAP is disabled, skipping cache load.")
        return False

    if _load_clap_index_from_db():
        logger.info("CLAP text cache loaded from persisted index.")
        return True

    logger.error("Failed to load persisted CLAP index. CLAP text search will be unavailable.")
    _CLAP_CACHE['loaded'] = False
    _CLAP_INDEX_CACHE['index'] = None
    _CLAP_INDEX_CACHE['id_map'] = None
    _CLAP_INDEX_CACHE['reverse_id_map'] = None
    _CLAP_INDEX_CACHE['loaded'] = False
    return False


def refresh_clap_cache():
    old_count = get_clap_cache_size()
    logger.info(f"Refreshing CLAP cache... (current: {old_count} songs)")
    result = load_clap_cache_from_db()
    new_count = get_clap_cache_size()
    if result:
        logger.info(
            f"OK CLAP cache refreshed: {old_count} -> {new_count} songs ({new_count - old_count:+d})"
        )
    else:
        logger.error(f"X CLAP cache refresh failed! Still at {new_count} songs")
    return result


def is_clap_cache_loaded() -> bool:
    return _CLAP_CACHE['loaded']


def _query_clap_index(text_embedding, fetch_size, limit, artist_cap):
    results, _ = _capped_neighbours(
        _CLAP_INDEX_CACHE['index'], _CLAP_INDEX_CACHE['id_map'] or {},
        text_embedding, fetch_size, limit, artist_cap,
    )
    return results


def _capped_neighbours(
    ivf_index, id_map, embedding, fetch_size, limit, artist_cap, seed_id=None, metadata_map=None
):
    from .paged_ivf import begin_query

    begin_query(ivf_index)
    num_to_query = min(fetch_size, len(ivf_index))
    if num_to_query <= 0:
        logger.warning("CLAP index is loaded but contains no items.")
        return [], True

    neighbor_ids, distances = ivf_index.query(embedding, k=num_to_query)
    candidate_item_ids = [id_map.get(int(vec_id)) for vec_id in neighbor_ids]
    candidate_item_ids = [item_id for item_id in candidate_item_ids if item_id is not None]
    if seed_id is not None:
        candidate_item_ids.append(seed_id)
    if metadata_map is None:
        metadata_map = {}
    missing = [item_id for item_id in dict.fromkeys(candidate_item_ids) if item_id not in metadata_map]
    if missing:
        metadata_map.update(_fetch_clap_metadata(missing))

    seed = None
    if seed_id is not None:
        seed_meta = metadata_map.get(seed_id) or {}
        seed = {
            'item_id': seed_id,
            'title': seed_meta.get('title', ''),
            'author': seed_meta.get('author', ''),
            'vector': embedding,
        }
    results = _build_capped_results(
        ivf_index,
        id_map,
        metadata_map,
        neighbor_ids,
        distances,
        limit,
        artist_cap,
        dedup_names=True,
        dup_threshold=config.DUPLICATE_DISTANCE_THRESHOLD_COSINE_CLAP,
        lookback=config.DUPLICATE_DISTANCE_CHECK_LOOKBACK,
        seed=seed,
    )
    return results, len(neighbor_ids) < num_to_query or num_to_query >= len(ivf_index)


def search_by_text(
    query_text: str, limit: Optional[int] = None, steering: Optional[List[Dict]] = None
) -> List[Dict]:
    from .clap_analyzer import get_text_embedding
    from config import CLAP_ENABLED, CLAP_SEARCH_DEFAULT_LIMIT

    if limit is None:
        limit = CLAP_SEARCH_DEFAULT_LIMIT
    if not CLAP_ENABLED:
        return []

    if not _CLAP_INDEX_CACHE['loaded'] or _CLAP_INDEX_CACHE['index'] is None:
        logger.error(
            "Cannot search: persisted CLAP index not loaded. Ensure Flask startup loaded the CLAP index."
        )
        return []

    try:
        with _TIMER.lock():
            warmup_text_search_model()

            text_embedding = get_text_embedding(query_text)
        if text_embedding is None:
            logger.error("Failed to generate the text embedding for a %d-character query", len(query_text))
            return []

        if steering:
            from .clap_steering import apply_steering

            text_embedding, applied = apply_steering(text_embedding, steering)
            if applied:
                logger.info(
                    "Query steered by %s",
                    ", ".join(f"{t['direction']} {t['term']} x{t['weight']}" for t in applied),
                )

        results = search_by_embedding(text_embedding, limit=limit)
        logger.info("Text search (%d characters): found %d results via CLAP index", len(query_text), len(results))
        return results

    except Exception:
        logger.exception("Text search failed for a %d-character query", len(query_text))
        return []


def search_by_embedding(embedding, limit: Optional[int] = None) -> List[Dict]:
    from config import CLAP_SEARCH_DEFAULT_LIMIT

    if limit is None:
        limit = CLAP_SEARCH_DEFAULT_LIMIT
    if embedding is None:
        return []
    if not _CLAP_INDEX_CACHE['loaded'] or _CLAP_INDEX_CACHE['index'] is None:
        logger.error(
            "CLAP index went unloaded between the entry guard and the query; returning no results."
        )
        return []

    return _query_clap_index(embedding, overfetch_size(limit), limit, _artist_cap(limit))


def _artist_cap(limit):
    from config import MAX_SONGS_PER_ARTIST

    if limit >= 1000 or not MAX_SONGS_PER_ARTIST or MAX_SONGS_PER_ARTIST <= 0:
        return 0
    return MAX_SONGS_PER_ARTIST


def _seed_clap_vector(item_id):
    from database import get_db

    with get_db().cursor() as cur:
        cur.execute("SELECT embedding FROM clap_embedding WHERE item_id = %s", (item_id,))
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    blob = bytes(row[0])
    if len(blob) != config.CLAP_EMBEDDING_DIMENSION * 4:
        logger.error(
            "Song search: the stored CLAP embedding of '%s' has %d bytes, expected %d.",
            item_id, len(blob), config.CLAP_EMBEDDING_DIMENSION * 4,
        )
        return None
    vector = np.frombuffer(blob, dtype=np.float32)
    if not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) <= 1e-9:
        logger.error("Song search: the stored CLAP embedding of '%s' is empty or not finite.", item_id)
        return None
    return vector


class ClapIndexUnavailable(RuntimeError):
    pass


class SongSeedNotFound(LookupError):
    pass


PICKER_WHERE = ('EXISTS (SELECT 1 FROM clap_embedding c WHERE c.item_id = score.item_id)', ())


def picker_where():
    return PICKER_WHERE if is_clap_cache_loaded() else None


def _radius_pool_size(limit):
    return limit + max(20, limit * 3) + 1


def _radius_walk(ivf_index, reverse_id_map, candidates, limit, eliminate_duplicates):
    from config import MAX_SONGS_PER_ARTIST

    vec_ids = {c['item_id']: reverse_id_map.get(c['item_id']) for c in candidates}
    vectors = ivf_index.get_vectors([vid for vid in vec_ids.values() if vid is not None])
    candidate_data = []
    for c in candidates:
        vec = vectors.get(vec_ids[c['item_id']]) if vec_ids[c['item_id']] is not None else None
        if vec is None:
            continue
        candidate_data.append({
            'item_id': c['item_id'],
            'vector': to_unit_vector(vec),
            'dist_anchor': max(0.0, 1.0 - float(c['similarity'])),
            'title': c.get('title'),
            'author': c.get('author'),
        })
    walked = execute_radius_walk(
        candidate_data=candidate_data,
        n=limit,
        eliminate_duplicates=eliminate_duplicates,
        max_songs_per_artist=MAX_SONGS_PER_ARTIST,
        get_distance_fn=cosine_distance,
    )
    by_id = {c['item_id']: c for c in candidates}
    return [by_id[w['item_id']] for w in walked if w['item_id'] in by_id]


def _song_search_radius(ivf_index, id_map, reverse_id_map, vector, seed_item_id, limit, eliminate_duplicates):
    metadata_map = {}
    results = []
    for pool in (_radius_pool_size(limit), _radius_pool_size(limit * 4)):
        candidates, exhausted = _capped_neighbours(
            ivf_index, id_map, vector, pool, pool, 0, seed_id=seed_item_id, metadata_map=metadata_map,
        )
        results = _radius_walk(ivf_index, reverse_id_map, candidates, limit, eliminate_duplicates)
        if len(results) >= limit or exhausted:
            break
    return results


def _song_search_plain(ivf_index, id_map, vector, seed_item_id, limit, artist_cap):
    metadata_map = {}
    results = []
    for fetch_size in (overfetch_size(limit), overfetch_size(limit * 4)):
        results, exhausted = _capped_neighbours(
            ivf_index, id_map, vector, fetch_size, limit, artist_cap,
            seed_id=seed_item_id, metadata_map=metadata_map,
        )
        if len(results) >= limit or exhausted:
            break
    return results


def search_by_song(
    seed_item_id: str,
    limit: Optional[int] = None,
    eliminate_duplicates: Optional[bool] = None,
    radius_similarity: Optional[bool] = None,
) -> List[Dict]:
    from config import (
        CLAP_ENABLED,
        CLAP_SEARCH_DEFAULT_LIMIT,
        MAX_SONGS_PER_ARTIST,
        SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT,
        SIMILARITY_RADIUS_DEFAULT,
    )

    if limit is None:
        limit = CLAP_SEARCH_DEFAULT_LIMIT
    if eliminate_duplicates is None:
        eliminate_duplicates = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT
    if radius_similarity is None:
        radius_similarity = SIMILARITY_RADIUS_DEFAULT
    if not CLAP_ENABLED or limit <= 0:
        return []
    ivf_index = _CLAP_INDEX_CACHE['index']
    id_map = _CLAP_INDEX_CACHE['id_map'] or {}
    reverse_id_map = _CLAP_INDEX_CACHE['reverse_id_map'] or {}
    if not _CLAP_INDEX_CACHE['loaded'] or ivf_index is None:
        raise ClapIndexUnavailable('The DCLAP index is not loaded.')

    vector = _seed_clap_vector(seed_item_id)
    if vector is None:
        raise SongSeedNotFound(f"'{seed_item_id}' has no usable DCLAP embedding.")

    if radius_similarity:
        try:
            results = _song_search_radius(
                ivf_index, id_map, reverse_id_map, vector, seed_item_id, limit, eliminate_duplicates
            )
            logger.info("Song search for '%s': %d results via CLAP index and radius walk", seed_item_id, len(results))
            return results
        except Exception:
            logger.exception(
                "Song search for '%s': the radius walk failed; answering with the plain nearest-neighbour order",
                seed_item_id,
            )

    artist_cap = 0
    if eliminate_duplicates and MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_ARTIST > 0:
        artist_cap = MAX_SONGS_PER_ARTIST
    results = _song_search_plain(ivf_index, id_map, vector, seed_item_id, limit, artist_cap)
    logger.info("Song search for '%s': %d results via CLAP index", seed_item_id, len(results))
    return results


def get_cache_stats() -> Dict:
    if not _CLAP_INDEX_CACHE['loaded'] or _CLAP_INDEX_CACHE['index'] is None:
        return {'loaded': False, 'song_count': 0, 'embedding_dimension': 0, 'memory_mb': 0}

    index_obj = _CLAP_INDEX_CACHE['index']
    index_size = sys.getsizeof(index_obj)
    if isinstance(index_obj, np.ndarray):
        index_size = index_obj.nbytes
    elif hasattr(index_obj, 'embeddings') and isinstance(index_obj.embeddings, np.ndarray):
        index_size = index_obj.embeddings.nbytes

    id_map_size = (
        sys.getsizeof(_CLAP_INDEX_CACHE['id_map']) if _CLAP_INDEX_CACHE['id_map'] is not None else 0
    )
    reverse_map_size = (
        sys.getsizeof(_CLAP_INDEX_CACHE['reverse_id_map'])
        if _CLAP_INDEX_CACHE['reverse_id_map'] is not None
        else 0
    )
    total_size_mb = (index_size + id_map_size + reverse_map_size) / (1024 * 1024)
    song_count = len(_CLAP_INDEX_CACHE['id_map']) if _CLAP_INDEX_CACHE['id_map'] is not None else 0

    return {
        'loaded': True,
        'song_count': song_count,
        'embedding_dimension': config.CLAP_EMBEDDING_DIMENSION,
        'memory_mb': round(total_size_mb, 2),
    }


def ensure_text_search_queries_table():
    from database import ensure_text_search_queries_table as _ensure

    return _ensure()


def load_top_queries_from_db():
    from database import (
        DEFAULT_TEXT_SEARCH_QUERIES,
        get_db,
        insert_default_text_search_queries,
    )

    ensure_text_search_queries_table()

    try:
        conn = get_db()
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT query_text, score, rank
                FROM text_search_queries
                ORDER BY rank ASC
            """)
            rows = cur.fetchall()

            if rows:
                _TOP_QUERIES_CACHE['queries'] = [row['query_text'] for row in rows]
                _TOP_QUERIES_CACHE['ready'] = True
                logger.info(f"Loaded {len(rows)} top queries from database")
                return True

            logger.info("No top queries found - inserting default queries")
            inserted = insert_default_text_search_queries(cur)
            conn.commit()

            _TOP_QUERIES_CACHE['queries'] = list(DEFAULT_TEXT_SEARCH_QUERIES)
            _TOP_QUERIES_CACHE['ready'] = True
            logger.info(f"Inserted and loaded {inserted} default queries")
            return True
    except Exception as e:
        logger.warning(f"Could not load top queries from database: {e}")
        return False


def get_cached_top_queries() -> List[str]:
    if _TOP_QUERIES_CACHE['ready']:
        return _TOP_QUERIES_CACHE['queries']
    return []
