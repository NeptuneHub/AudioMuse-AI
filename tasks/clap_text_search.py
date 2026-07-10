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
* warmup_text_search_model with an idle-unload timer, plus a persisted top-queries
  table (ensure_text_search_queries_table) used to pre-warm popular searches.
"""

import logging
import sys
import threading
import time

import numpy as np
from psycopg2.extras import DictCursor
from typing import List, Dict
import config

logger = logging.getLogger(__name__)

_CLAP_CACHE = {'loaded': False}

_CLAP_INDEX_CACHE = {'index': None, 'id_map': None, 'reverse_id_map': None, 'loaded': False}

_TOP_QUERIES_CACHE = {'queries': [], 'ready': False, 'computing': False}

_WARM_CACHE_TIMER = {
    'expiry_time': None,
    'timer_thread': None,
    'lock': threading.RLock(),
    'duration_seconds': None,
}


def get_clap_cache_size() -> int:
    if _CLAP_INDEX_CACHE['loaded'] and _CLAP_INDEX_CACHE['id_map'] is not None:
        return len(_CLAP_INDEX_CACHE['id_map'])
    return 0


def _fetch_clap_metadata(item_ids: list) -> Dict[str, Dict[str, str]]:
    from .commons import fetch_track_metadata_map

    return fetch_track_metadata_map(item_ids)


def _load_clap_index_from_db() -> bool:
    from app_helper import get_db
    from config import CLAP_EMBEDDING_DIMENSION, IVF_METRIC
    from .paged_ivf import load_index_auto

    try:
        loaded = load_index_auto(
            get_db(),
            'clap_index',
            CLAP_EMBEDDING_DIMENSION,
            IVF_METRIC,
            label='CLAP',
        )
        if loaded is None:
            return False
        loaded_index, id_map, reverse_id_map = loaded

        _CLAP_CACHE['loaded'] = True

        _CLAP_INDEX_CACHE['index'] = loaded_index
        _CLAP_INDEX_CACHE['id_map'] = id_map
        _CLAP_INDEX_CACHE['reverse_id_map'] = reverse_id_map
        _CLAP_INDEX_CACHE['loaded'] = True

        logger.info(f"CLAP index loaded from database with {len(id_map)} items.")
        return True
    except Exception:
        logger.exception("Failed to load CLAP index from DB")
        return False


def build_and_store_clap_index(db_conn=None):
    from app_helper import get_db
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


def _unload_timer_worker():
    while True:
        with _WARM_CACHE_TIMER['lock']:
            expiry = _WARM_CACHE_TIMER['expiry_time']
            if expiry is None:
                break
            if expiry - time.time() <= 0:
                from .clap_analyzer import unload_clap_model, is_clap_text_loaded

                if is_clap_text_loaded():
                    logger.info("Warm cache timer expired - unloading CLAP text model")
                    unload_clap_model()
                _WARM_CACHE_TIMER['expiry_time'] = None
                _WARM_CACHE_TIMER['timer_thread'] = None
                break
            time_remaining = expiry - time.time()

        time.sleep(min(1.0, max(0.05, time_remaining)))


def warmup_text_search_model():
    from .clap_analyzer import initialize_clap_text_model, is_clap_text_loaded

    if _WARM_CACHE_TIMER['duration_seconds'] is None:
        _WARM_CACHE_TIMER['duration_seconds'] = config.CLAP_TEXT_SEARCH_WARMUP_DURATION

    with _WARM_CACHE_TIMER['lock']:
        if not is_clap_text_loaded():
            logger.info("Warming up CLAP text model for text search (not loading audio model)...")
            success = initialize_clap_text_model()
            if not success:
                return {'loaded': False, 'expiry_seconds': 0}

        _WARM_CACHE_TIMER['expiry_time'] = time.time() + _WARM_CACHE_TIMER['duration_seconds']

        if (
            _WARM_CACHE_TIMER['timer_thread'] is None
            or not _WARM_CACHE_TIMER['timer_thread'].is_alive()
        ):
            thread = threading.Thread(target=_unload_timer_worker, daemon=True)
            thread.start()
            _WARM_CACHE_TIMER['timer_thread'] = thread
            logger.info(f"Started warm cache timer ({_WARM_CACHE_TIMER['duration_seconds']}s)")
        else:
            logger.debug(f"Reset warm cache timer ({_WARM_CACHE_TIMER['duration_seconds']}s)")

    return {'loaded': True, 'expiry_seconds': _WARM_CACHE_TIMER['duration_seconds']}


def get_warm_cache_status() -> Dict:
    from .clap_analyzer import is_clap_model_loaded

    with _WARM_CACHE_TIMER['lock']:
        expiry = _WARM_CACHE_TIMER['expiry_time']

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


def search_by_text(query_text: str, limit: int = 100) -> List[Dict]:
    from .clap_analyzer import get_text_embedding
    from config import CLAP_ENABLED

    if not CLAP_ENABLED:
        return []

    if not _CLAP_INDEX_CACHE['loaded'] or _CLAP_INDEX_CACHE['index'] is None:
        logger.error(
            "Cannot search: persisted CLAP index not loaded. Ensure Flask startup loaded the CLAP index."
        )
        return []

    try:
        with _WARM_CACHE_TIMER['lock']:
            warmup_text_search_model()

            text_embedding = get_text_embedding(query_text)
        if text_embedding is None:
            logger.error(f"Failed to generate text embedding for: {query_text}")
            return []

        from config import MAX_SONGS_PER_ARTIST

        artist_cap = (
            MAX_SONGS_PER_ARTIST if MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_ARTIST > 0 else 0
        )
        if limit >= 1000:
            artist_cap = 0
        fetch_size = (limit + max(20, limit * 4) + 1) if artist_cap else limit

        if _CLAP_INDEX_CACHE['loaded'] and _CLAP_INDEX_CACHE['index'] is not None:
            ivf_index = _CLAP_INDEX_CACHE['index']
            id_map = _CLAP_INDEX_CACHE['id_map'] or {}
            from .paged_ivf import begin_query

            begin_query(ivf_index)
            num_to_query = min(fetch_size, len(ivf_index))

            if num_to_query <= 0:
                logger.warning("CLAP index is loaded but contains no items.")
                return []

            neighbor_ids, distances = ivf_index.query(text_embedding, k=num_to_query)
            candidate_item_ids = [id_map.get(int(vec_id)) for vec_id in neighbor_ids]
            candidate_item_ids = [item_id for item_id in candidate_item_ids if item_id is not None]

            metadata_map = _fetch_clap_metadata(candidate_item_ids)

            results = []
            artist_counts: dict = {}
            for vec_id, distance in zip(neighbor_ids, distances):
                if len(results) >= limit:
                    break
                item_id = id_map.get(int(vec_id))
                if item_id is None:
                    continue

                metadata = metadata_map.get(item_id, {'title': '', 'author': '', 'album': ''})
                author = metadata.get('author', '')

                if artist_cap and author:
                    author_norm = author.strip().lower()
                    if artist_counts.get(author_norm, 0) >= artist_cap:
                        continue
                    artist_counts[author_norm] = artist_counts.get(author_norm, 0) + 1

                similarity = ivf_index.distance_to_similarity(distance)
                results.append(
                    {
                        'item_id': item_id,
                        'title': metadata.get('title', ''),
                        'author': metadata.get('author', ''),
                        'album': metadata.get('album', ''),
                        'similarity': similarity,
                    }
                )

            logger.info(
                f"Text search '{query_text}': found {len(results)} results via CLAP index (artist cap: {artist_cap or 'disabled'})"
            )
            return results

    except Exception:
        logger.exception(f"Text search failed for '{query_text}'")
        return []


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
    from app_helper import get_db

    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(726354821)")
            try:
                cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                    ('text_search_queries',),
                )
                if not cur.fetchone()[0]:
                    cur.execute("""
                        CREATE TABLE text_search_queries (
                            id SERIAL PRIMARY KEY,
                            query_text TEXT NOT NULL,
                            score REAL NOT NULL,
                            rank INTEGER NOT NULL,
                            created_at TIMESTAMP DEFAULT NOW(),
                            UNIQUE(rank)
                        )
                    """)
            finally:
                cur.execute("SELECT pg_advisory_unlock(726354821)")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_search_queries_rank
                ON text_search_queries(rank)
            """)
            conn.commit()
            logger.info("Ensured text_search_queries table exists")
            return True
    except Exception:
        logger.exception("Failed to create text_search_queries table")
        if conn:
            conn.rollback()
        return False


def load_top_queries_from_db():
    from app_helper import get_db

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
            else:
                logger.info("No top queries found - inserting default queries")
                default_queries = [
                    "female vocal romantic trap",
                    "synth indie pop raspy",
                    "sad hard rock male vocal",
                    "funk falsetto energetic",
                    "groovy sax blues",
                    "classical relaxed piano",
                    "belting jazz happy",
                    "tabla afrobeat fast-paced",
                    "harmonized vocals slow-paced electronica",
                    "autotuned gospel excited",
                    "breathy aggressive house",
                    "smooth folk mid-tempo",
                    "deep voice r&b dark",
                    "punk guitar angry",
                    "metal choir dreamy",
                    "chant reggae trumpet",
                    "high-pitched brass hip-hop",
                    "disco whispered drum machine",
                    "happy whispered indie pop",
                    "synth energetic raspy",
                    "rock slow-paced cello",
                    "falsetto jazz excited",
                    "r&b male vocal romantic",
                    "harmonized vocals dark trap",
                    "smooth blues sax",
                    "high-pitched fast-paced soul",
                    "female vocal sad hip-hop",
                    "congas aggressive soul",
                    "mid-tempo afrobeat autotuned",
                    "belting funk groovy",
                    "angry alternative breathy",
                    "gospel choir steelpan",
                    "viola relaxed folk",
                    "dreamy rhodes metal",
                    "acoustic guitar country chant",
                    "deep voice orchestra reggae",
                    "fast-paced synth progressive rock",
                    "hard rock raspy romantic",
                    "fast-paced electric guitar progressive rock",
                    "hard rock aggressive breathy",
                    "rock high-pitched energetic",
                    "autotuned energetic hip-hop",
                    "raspy fast-paced blues",
                    "belting electronica energetic",
                    "whispered indie pop aggressive",
                    "harmonized vocals aggressive synth",
                    "orchestra whispered romantic",
                    "belting mid-tempo progressive rock",
                    "autotuned pop mid-tempo",
                    "pop energetic synthesizer",
                ]

                for rank, query in enumerate(default_queries, start=1):
                    cur.execute(
                        """
                        INSERT INTO text_search_queries (query_text, score, rank, created_at)
                        VALUES (%s, %s, %s, NOW())
                    """,
                        (query, 1.0, rank),
                    )

                conn.commit()

                _TOP_QUERIES_CACHE['queries'] = default_queries
                _TOP_QUERIES_CACHE['ready'] = True
                logger.info(f"Inserted and loaded {len(default_queries)} default queries")
                return True
    except Exception as e:
        logger.warning(f"Could not load top queries from database: {e}")
        return False


def get_cached_top_queries() -> List[str]:
    if _TOP_QUERIES_CACHE['ready']:
        return _TOP_QUERIES_CACHE['queries']
    return []
