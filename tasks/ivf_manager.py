# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Query manager for the audio similarity IVF index.

Builds, loads and queries the primary audio embedding index (module renamed
from the former Voyager manager; Voyager is fully removed). Delegates on-disk
storage and cell scanning to tasks.paged_ivf and shares build helpers with
tasks.index_build_helpers, exposing the high-level nearest-neighbor and search
API the playlist, similarity and path features call.

Main Features:
* build_and_store_ivf_index / load_ivf_index_for_querying: build the disk-paged
  IVF index from pgvector embeddings and hold one process-wide instance.
* find_nearest_neighbors_by_id / _by_vector, search_tracks_unified, radius walk:
  neighbor queries with f32 re-rank overfetch, artist capping, content
  de-duplication and optional mood-similarity filtering.
* TTL-bounded result caches and a shared thread pool for parallel cell fetches.
"""

import os
import time
import logging
import numpy as np
from collections import OrderedDict
from psycopg2.extras import DictCursor

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import (
    EMBEDDING_DIMENSION,
    INDEX_NAME,
    IVF_METRIC,
    MAX_SONGS_PER_ARTIST,
    DUPLICATE_DISTANCE_THRESHOLD_COSINE,
    DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN,
    DUPLICATE_DISTANCE_CHECK_LOOKBACK,
    MOOD_SIMILARITY_THRESHOLD,
    SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT,
    SIMILARITY_RADIUS_DEFAULT,
    MOOD_SIMILARITY_ENABLE,
    IVF_RESULT_CACHE_SECONDS,
    IVF_RESULT_CACHE_MAX,
    RADIUS_INSTRUMENTATION,
    IVF_RERANK_OVERFETCH,
)

logger = logging.getLogger(__name__)

INSTRUMENT_BUCKET_SKIPS = RADIUS_INSTRUMENTATION

ivf_index = None
id_map = None
reverse_id_map = None


class _ResultCache:
    def __init__(self, ttl_seconds, max_entries):
        self._ttl = float(ttl_seconds)
        self._max = max(1, int(max_entries))
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        if self._ttl <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expiry, value = item
            if expiry <= now:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key, value):
        if self._ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic() + self._ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()

    def sweep_expired(self):
        if self._ttl <= 0:
            return
        now = time.monotonic()
        with self._lock:
            stale = [k for k, (expiry, _v) in self._data.items() if expiry <= now]
            for k in stale:
                del self._data[k]


_neighbor_result_cache = _ResultCache(IVF_RESULT_CACHE_SECONDS, IVF_RESULT_CACHE_MAX)
_max_distance_cache = _ResultCache(IVF_RESULT_CACHE_SECONDS, IVF_RESULT_CACHE_MAX)


def _sweep_result_caches():
    _neighbor_result_cache.sweep_expired()
    _max_distance_cache.sweep_expired()


try:
    from .paged_ivf import register_idle_callback

    register_idle_callback(_sweep_result_caches)
except Exception:
    logger.debug("Could not register IVF result-cache idle sweep", exc_info=True)


_thread_pool = None
_thread_pool_lock = threading.Lock()

MAX_WORKER_THREADS = max(1, (os.cpu_count() or 1) - 1)
BATCH_SIZE_VECTOR_OPS = 50
BATCH_SIZE_DB_OPS = 100
SCORE_DETAIL_COLUMNS = 'title, author'


def _get_thread_pool():
    global _thread_pool
    with _thread_pool_lock:
        if _thread_pool is None:
            _thread_pool = ThreadPoolExecutor(
                max_workers=MAX_WORKER_THREADS, thread_name_prefix="ivf"
            )
        return _thread_pool


def _shutdown_thread_pool():
    global _thread_pool
    with _thread_pool_lock:
        if _thread_pool is not None:
            _thread_pool.shutdown(wait=True)
            _thread_pool = None


def _fetch_in_batches(item_ids, fetch_batch_fn):
    id_batches = [
        item_ids[i : i + BATCH_SIZE_DB_OPS] for i in range(0, len(item_ids), BATCH_SIZE_DB_OPS)
    ]
    merged = {}
    for batch in id_batches:
        merged.update(fetch_batch_fn(batch))
    return merged


def _fetch_details_map(db_conn, item_ids, columns):
    column_keys = [c.strip() for c in columns.split(',')]

    def fetch_batch(id_batch):
        out = {}
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                f"SELECT item_id, {columns} FROM score WHERE item_id = ANY(%s)", (id_batch,)
            )
            for row in cur.fetchall():
                out[row['item_id']] = {c: row.get(c) for c in column_keys}
        return out

    return _fetch_in_batches(item_ids, fetch_batch)


_tls = threading.local()


def _fetch_f32_embeddings(db_conn, item_ids) -> dict:
    if not item_ids:
        return {}
    out: dict = {}
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT item_id, embedding FROM embedding WHERE item_id = ANY(%s) AND embedding IS NOT NULL",
                (list(item_ids),),
            )
            for item_id, emb in cur.fetchall():
                v = np.frombuffer(bytes(emb), dtype=np.float32)
                if v.shape[0] == EMBEDDING_DIMENSION:
                    out[item_id] = v
    except Exception as e:
        logger.warning(
            "Exact-f32 rerank: could not read source embeddings (%s); using int8 vectors.", e
        )
    return out


def _prime_request_f32(vec_map: dict) -> None:
    _tls.f32 = vec_map


def _clear_request_f32() -> None:
    _tls.f32 = {}


def _get_cached_vector(item_id: str) -> np.ndarray | None:
    cached = getattr(_tls, "f32", None)
    if cached:
        v = cached.get(item_id)
        if v is not None:
            return v
    if ivf_index is None or reverse_id_map is None:
        return None
    vec_id = reverse_id_map.get(item_id)
    if vec_id is None:
        return None
    try:
        return ivf_index.get_vector(vec_id)
    except Exception:
        return None


def _get_direct_euclidean_distance(v1, v2):
    if v1 is None or v2 is None:
        return float('inf')
    try:
        dist = np.linalg.norm(v1.astype(np.float32) - v2.astype(np.float32))
        return float(dist)
    except Exception:
        return float('inf')


def _get_direct_cosine_distance(v1, v2):
    if v1 is None or v2 is None:
        return float('inf')
    try:
        v1_f32 = v1.astype(np.float32)
        v2_f32 = v2.astype(np.float32)

        norm_v1 = np.linalg.norm(v1_f32)
        norm_v2 = np.linalg.norm(v2_f32)

        denom = norm_v1 * norm_v2
        if denom == 0:
            return float('inf')

        dot_product = np.dot(v1_f32, v2_f32)
        cos_sim = dot_product / denom

        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        return 1.0 - float(cos_sim)
    except Exception:
        return float('inf')


def _get_direct_dot_distance(v1, v2):
    if v1 is None or v2 is None:
        return float('inf')
    try:
        return float(-np.dot(v1.astype(np.float32), v2.astype(np.float32)))
    except Exception:
        return float('inf')


def get_direct_distance(v1, v2):
    if IVF_METRIC == 'angular':
        return _get_direct_cosine_distance(v1, v2)
    if IVF_METRIC == 'dot':
        return _get_direct_dot_distance(v1, v2)
    return _get_direct_euclidean_distance(v1, v2)


def load_ivf_index_for_querying(force_reload=False):
    global ivf_index, id_map, reverse_id_map

    if ivf_index is not None and not force_reload:
        logger.info("Audio index is already loaded in memory. Skipping reload.")
        return

    _neighbor_result_cache.clear()
    _max_distance_cache.clear()

    from app_helper import get_db
    from .paged_ivf import load_paged_ivf_index

    conn = get_db()
    logger.info("Loading audio IVF index from database into memory...")
    try:
        loaded = load_paged_ivf_index(
            conn, INDEX_NAME, EMBEDDING_DIMENSION, IVF_METRIC, label="audio"
        )
    except Exception:
        logger.exception("Failed to load audio IVF index")
        ivf_index, id_map, reverse_id_map = None, None, None
        return
    if loaded is None:
        logger.warning(
            "Audio IVF index not found in the database. Cache will be empty (run analysis to build it)."
        )
        ivf_index, id_map, reverse_id_map = None, None, None
        return
    ivf_index, id_map, reverse_id_map = loaded
    logger.info("Audio IVF index with %d items loaded successfully into memory.", len(id_map))


def build_and_store_ivf_index(db_conn=None):
    if db_conn is None:
        try:
            from app_helper import get_db

            db_conn = get_db()
        except Exception:
            logger.exception("build_and_store_ivf_index: no db_conn provided and get_db() failed.")
            return

    from .index_build_helpers import stream_embeddings_to_buffer
    from .paged_ivf import build_and_store_paged_ivf

    logger.info("Starting to build and store audio IVF index (disk-paged)...")
    try:
        buf, item_ids = stream_embeddings_to_buffer(
            table="embedding",
            column="embedding",
            dim=EMBEDDING_DIMENSION,
            where_clause="embedding IS NOT NULL",
        )
        if buf.shape[0] == 0:
            logger.warning("No valid audio embeddings found for IVF index build. Aborting.")
            return
        if build_and_store_paged_ivf(
            db_conn, INDEX_NAME, buf, item_ids, EMBEDDING_DIMENSION, IVF_METRIC,
            consume_vectors=True,
        ):
            db_conn.commit()
            logger.info("Audio IVF index build and database storage complete.")
    except Exception:
        logger.exception("An error occurred during audio IVF index build")
        try:
            db_conn.rollback()
        except Exception:
            pass


def get_vector_by_id(item_id: str) -> np.ndarray | None:
    return _get_cached_vector(item_id)


def get_cell_groups_for_items(item_ids):
    if ivf_index is None or reverse_id_map is None:
        return []
    vec_ids = [vid for vid in (reverse_id_map.get(iid) for iid in item_ids) if vid is not None]
    if not vec_ids:
        return []
    return ivf_index.cell_groups(vec_ids)


def multi_query_ids(query_vectors, per_vector_n):
    if ivf_index is None or id_map is None:
        return []
    try:
        ivf_index.begin_request()
    except Exception:
        pass
    _clear_request_f32()
    k = max(1, int(per_vector_n))
    seen = {}
    for vec in query_vectors:
        try:
            vec_ids, _distances = ivf_index.query(vec, k=k)
        except Exception:
            logger.exception("IVF multi-query failed for a vector")
            continue
        for vid in vec_ids:
            item_id = id_map.get(vid)
            if item_id is not None:
                seen.setdefault(item_id, None)
    return list(seen.keys())


def _normalize_string(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower()


def _is_same_song(title1, artist1, title2, artist2):
    norm_title1 = _normalize_string(title1)
    norm_title2 = _normalize_string(title2)
    norm_artist1 = _normalize_string(artist1)
    norm_artist2 = _normalize_string(artist2)

    return norm_title1 == norm_title2 and norm_artist1 == norm_artist2


def _is_too_close(current_song, current_vector, window_songs, threshold, metric_name, details_map):
    for recent_song in window_songs:
        recent_vector = _get_cached_vector(recent_song['item_id'])
        if recent_vector is None:
            continue
        direct_dist = get_direct_distance(current_vector, recent_vector)
        if direct_dist < threshold:
            current_details = details_map.get(
                current_song['item_id'], {'title': 'N/A', 'author': 'N/A'}
            )
            recent_details = details_map.get(
                recent_song['item_id'], {'title': 'N/A', 'author': 'N/A'}
            )
            logger.info(
                f"Filtering song (DISTANCE FILTER) with {metric_name} distance: '{current_details['title']}' by '{current_details['author']}' "
                f"due to direct distance of {direct_dist:.4f} from "
                f"'{recent_details['title']}' by '{recent_details['author']}' (Threshold: {threshold})."
            )
            return True
    return False


def _compute_distance_batch(song_batch, lookback_songs, threshold, metric_name, details_map):
    batch_results = []
    for current_song in song_batch:
        current_vector = _get_cached_vector(current_song['item_id'])
        if current_vector is None:
            continue
        combined_recent = list(lookback_songs) + list(batch_results)
        if not _is_too_close(
            current_song, current_vector, combined_recent, threshold, metric_name, details_map
        ):
            batch_results.append(current_song)
    return batch_results


def _filter_by_distance(song_results: list, db_conn):
    if DUPLICATE_DISTANCE_CHECK_LOOKBACK <= 0:
        return song_results

    if not song_results:
        return []

    item_ids = [s['item_id'] for s in song_results]
    details_map = _fetch_details_map(db_conn, item_ids, SCORE_DETAIL_COLUMNS)

    threshold = (
        DUPLICATE_DISTANCE_THRESHOLD_COSINE
        if IVF_METRIC == 'angular'
        else DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    )
    metric_name = 'Angular' if IVF_METRIC == 'angular' else 'Euclidean'

    filtered_songs = []

    if len(song_results) <= BATCH_SIZE_VECTOR_OPS:
        for current_song in song_results:
            current_vector = _get_cached_vector(current_song['item_id'])
            if current_vector is None:
                continue
            lookback_window = filtered_songs[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]
            if not _is_too_close(
                current_song, current_vector, lookback_window, threshold, metric_name, details_map
            ):
                filtered_songs.append(current_song)
    else:
        remaining_songs = song_results.copy()

        while remaining_songs:
            current_batch = remaining_songs[:BATCH_SIZE_VECTOR_OPS]
            remaining_songs = remaining_songs[BATCH_SIZE_VECTOR_OPS:]

            lookback_window = (
                filtered_songs[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:] if filtered_songs else []
            )

            batch_results = _compute_distance_batch(
                current_batch, lookback_window, threshold, metric_name, details_map
            )
            filtered_songs.extend(batch_results)

    return filtered_songs


def _deduplicate_and_filter_neighbors(song_results: list, db_conn, original_song_details: dict):
    if not song_results:
        return []

    item_ids = [r['item_id'] for r in song_results]
    item_details = _fetch_details_map(db_conn, item_ids, SCORE_DETAIL_COLUMNS)

    unique_songs = []

    added_songs_signatures = set()

    original_title = _normalize_string(original_song_details.get('title'))
    original_author = _normalize_string(original_song_details.get('author'))
    added_songs_signatures.add((original_title, original_author))

    for song in song_results:
        current_details = item_details.get(song['item_id'])
        if not current_details:
            logger.warning(
                f"Could not find details for item_id {song['item_id']} during deduplication. Skipping."
            )
            continue

        current_title = _normalize_string(current_details.get('title'))
        current_author = _normalize_string(current_details.get('author'))
        current_signature = (current_title, current_author)

        if current_signature not in added_songs_signatures:
            unique_songs.append(song)
            added_songs_signatures.add(current_signature)
        else:
            logger.info(
                f"Found duplicate (NAME FILTER): '{current_details.get('title')}' by '{current_details.get('author')}' (Distance from source: {song.get('distance', 0.0):.4f})."
            )

    return unique_songs


def _mood_distance(target_mood_features, candidate_features, mood_features):
    total = sum(
        abs(target_mood_features.get(feature, 0.0) - candidate_features.get(feature, 0.0))
        for feature in mood_features
    )
    return total / len(mood_features)


def _compute_mood_distances_batch(
    song_batch, target_mood_features, candidate_mood_features, mood_features, mood_threshold
):
    batch_results = []
    for song in song_batch:
        candidate_features = candidate_mood_features.get(song['item_id'])
        if not candidate_features:
            continue
        normalized_mood_distance = _mood_distance(
            target_mood_features, candidate_features, mood_features
        )
        if normalized_mood_distance <= mood_threshold:
            song_with_mood = song.copy()
            song_with_mood['mood_distance'] = normalized_mood_distance
            batch_results.append(song_with_mood)
    return batch_results


def _resolve_target_other_features(target_item_id, db_conn, target_other_features):
    if target_other_features is not None:
        return target_other_features
    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT other_features FROM score WHERE item_id = %s", (target_item_id,))
        target_row = cur.fetchone()
        return target_row['other_features'] if target_row else None


def _fetch_candidate_mood_features(candidate_ids, db_conn):
    def fetch_mood_features_batch(id_batch):
        batch_features = {}
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                "SELECT item_id, other_features FROM score WHERE item_id = ANY(%s)", (id_batch,)
            )
            rows = cur.fetchall()
            for row in rows:
                if not row['other_features']:
                    continue
                parsed_features = _parse_mood_features(row['other_features'])
                if parsed_features:
                    batch_features[row['item_id']] = parsed_features
        return batch_features

    return _fetch_in_batches(candidate_ids, fetch_mood_features_batch)


def _filter_mood_single_threaded(
    song_results, candidate_mood_features, target_mood_features, mood_features, mood_threshold
):
    filtered_songs = []
    for song in song_results:
        candidate_features = candidate_mood_features.get(song['item_id'])
        if not candidate_features:
            logger.debug(f"Skipping song {song['item_id']}: no mood features found")
            continue

        normalized_mood_distance = _mood_distance(
            target_mood_features, candidate_features, mood_features
        )

        logger.debug(
            f"Song {song['item_id']} mood distance: {normalized_mood_distance:.4f}, features: {candidate_features}"
        )

        if normalized_mood_distance <= mood_threshold:
            song_with_mood = song.copy()
            song_with_mood['mood_distance'] = normalized_mood_distance
            filtered_songs.append(song_with_mood)
            logger.debug(f"  -> KEPT (distance: {normalized_mood_distance:.4f})")
        else:
            logger.debug(
                f"  -> FILTERED OUT (distance: {normalized_mood_distance:.4f} > threshold: {mood_threshold})"
            )
    return filtered_songs


def _filter_mood_threaded(
    song_results, candidate_mood_features, target_mood_features, mood_features, mood_threshold
):
    song_batches = [
        song_results[i : i + BATCH_SIZE_VECTOR_OPS]
        for i in range(0, len(song_results), BATCH_SIZE_VECTOR_OPS)
    ]

    executor = _get_thread_pool()
    future_to_batch = {
        executor.submit(
            _compute_mood_distances_batch,
            batch,
            target_mood_features,
            candidate_mood_features,
            mood_features,
            mood_threshold,
        ): batch
        for batch in song_batches
    }

    filtered_songs = []
    for future in as_completed(future_to_batch):
        filtered_songs.extend(future.result())
    return filtered_songs


def _filter_by_mood_similarity(
    song_results: list,
    target_item_id: str,
    db_conn,
    mood_threshold: float | None = None,
    target_other_features=None,
):
    if not song_results:
        return []

    if mood_threshold is None:
        mood_threshold = MOOD_SIMILARITY_THRESHOLD

    target_other_features = _resolve_target_other_features(
        target_item_id, db_conn, target_other_features
    )
    if not target_other_features:
        logger.warning("No mood features found for target song. Skipping mood filtering.")
        return song_results

    target_mood_features = _parse_mood_features(target_other_features)
    if not target_mood_features:
        logger.warning("Could not parse mood features for target song. Skipping mood filtering.")
        return song_results

    logger.info("Target mood features parsed (%d features).", len(target_mood_features))

    candidate_ids = [s['item_id'] for s in song_results]
    candidate_mood_features = _fetch_candidate_mood_features(candidate_ids, db_conn)

    mood_features = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']

    logger.info(
        f"Starting mood filtering with {len(song_results)} candidates, threshold: {mood_threshold}"
    )

    if len(song_results) <= BATCH_SIZE_VECTOR_OPS:
        filtered_songs = _filter_mood_single_threaded(
            song_results, candidate_mood_features, target_mood_features, mood_features, mood_threshold
        )
    else:
        filtered_songs = _filter_mood_threaded(
            song_results, candidate_mood_features, target_mood_features, mood_features, mood_threshold
        )

    logger.info(
        f"Mood filtering results: kept {len(filtered_songs)} of {len(song_results)} songs (threshold: {mood_threshold})"
    )
    return filtered_songs


def _parse_mood_features(other_features_str: str) -> dict:
    try:
        features = {}
        for pair in other_features_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                features[key.strip()] = float(value.strip())
        return features
    except Exception as e:
        logger.warning(f"Error parsing mood features '{other_features_str}': {e}")
        return {}


def _radius_walk_get_candidates(
    target_item_id: str,
    anchor_vector: np.ndarray,
    initial_results: list,
    db_conn,
    original_song_details: dict,
    eliminate_duplicates: bool,
    mood_similarity: bool | None = None,
) -> list:
    from app_helper import get_score_data_by_ids

    if not initial_results:
        return []

    try:
        original_for_filter = {"item_id": target_item_id, "distance": 0.0}
        results_with_original = [original_for_filter] + initial_results
        temp_filtered = _filter_by_distance(results_with_original, db_conn)
        distance_filtered_results = [s for s in temp_filtered if s['item_id'] != target_item_id]
        logger.info(
            f"Radius walk: distance-based filtering reduced candidates {len(initial_results)} -> {len(distance_filtered_results)}"
        )
    except Exception:
        logger.exception(
            "Radius walk: distance-based pre-filter failed, continuing with original candidate set."
        )
        distance_filtered_results = initial_results

    try:
        unique_results_by_song = _deduplicate_and_filter_neighbors(
            distance_filtered_results, db_conn, original_song_details
        )
        logger.info(
            f"Radius walk: name-based dedupe reduced candidates to {len(unique_results_by_song)}"
        )
    except Exception:
        logger.exception("Radius walk: name-based dedupe failed, continuing without it.")
        unique_results_by_song = distance_filtered_results

    try:
        effective_mood = MOOD_SIMILARITY_ENABLE if mood_similarity is None else mood_similarity
        if effective_mood:
            before_mood = len(unique_results_by_song)
            unique_results_by_song = _filter_by_mood_similarity(
                unique_results_by_song,
                target_item_id,
                db_conn,
                target_other_features=original_song_details.get('other_features'),
            )
            after_mood = len(unique_results_by_song)
            logger.info(
                f"Radius walk: mood-based filtering reduced candidates {before_mood} -> {after_mood}"
            )
        else:
            logger.debug("Radius walk: mood-based pre-filter disabled by caller/config. Skipping.")
    except Exception:
        logger.exception("Radius walk: mood-based pre-filter failed, continuing without it.")

    candidate_data = []
    if unique_results_by_song:
        item_ids_to_fetch = [r['item_id'] for r in unique_results_by_song]
        try:
            track_details_list = get_score_data_by_ids(item_ids_to_fetch)
            details_map = {
                d['item_id']: {
                    'title': d.get('title'),
                    'author': d.get('author'),
                    'album': d.get('album'),
                    'album_artist': d.get('album_artist'),
                }
                for d in track_details_list
            }
        except Exception:
            details_map = {}

        for song in unique_results_by_song:
            item_id = song['item_id']
            vector = _get_cached_vector(item_id)
            if vector is not None:
                try:
                    vector = vector.astype(np.float32)
                except Exception:
                    vector = np.array(vector, dtype=np.float32)
                dist_to_anchor = get_direct_distance(vector, anchor_vector)
                info = details_map.get(item_id, {'title': None, 'author': None})
                candidate_data.append(
                    {
                        "item_id": item_id,
                        "vector": vector,
                        "dist_anchor": dist_to_anchor,
                        "title": info.get('title'),
                        "author": info.get('author'),
                    }
                )

    logger.info(
        f"Radius walk: pre-calculated vectors and distances for {len(candidate_data)} candidates."
    )
    return candidate_data


def _execute_radius_walk(n: int, candidate_data: list, eliminate_duplicates: bool = False) -> list:
    from .radius_walk_helper import execute_radius_walk as _shared_walk

    return _shared_walk(
        candidate_data=candidate_data,
        n=n,
        eliminate_duplicates=eliminate_duplicates,
        max_songs_per_artist=MAX_SONGS_PER_ARTIST,
        get_distance_fn=get_direct_distance,
    )


def _dedup_by_content(songs, item_details):
    unique_songs = []
    added_songs_details = []
    for song in songs:
        current_details = item_details.get(song['item_id'])
        if not current_details:
            continue

        is_duplicate = any(
            _is_same_song(
                current_details['title'], current_details['author'], added['title'], added['author']
            )
            for added in added_songs_details
        )

        if not is_duplicate:
            unique_songs.append(song)
            added_songs_details.append(current_details)
    return unique_songs


def _apply_artist_cap(songs, author_resolver, warn_missing=False):
    if MAX_SONGS_PER_ARTIST is None or MAX_SONGS_PER_ARTIST <= 0:
        return songs

    artist_counts = {}
    capped = []
    for song in songs:
        author = author_resolver(song)
        if not author:
            if warn_missing:
                logger.warning(
                    f"Could not find author for item_id {song['item_id']} during artist deduplication. Skipping."
                )
            continue

        current_count = artist_counts.get(author, 0)
        if current_count < MAX_SONGS_PER_ARTIST:
            capped.append(song)
            artist_counts[author] = current_count + 1
    return capped


def _load_target_for_neighbor_search(target_item_id, get_score_data_by_ids):
    target_song_details_list = get_score_data_by_ids([target_item_id])
    if not target_song_details_list:
        logger.error(
            f"Could not retrieve details for the target song {target_item_id}. Aborting neighbor search."
        )
        return None

    target_vec_id = reverse_id_map.get(target_item_id)
    if target_vec_id is None:
        logger.warning(f"Target item_id '{target_item_id}' not found in the loaded IVF index map.")
        return None

    return target_song_details_list[0], target_vec_id


def _resolve_neighbor_query_vector(target_item_id, target_vec_id, db_conn):
    _clear_request_f32()
    anchor_f32 = _fetch_f32_embeddings(db_conn, [target_item_id]).get(target_item_id)
    if anchor_f32 is not None:
        return anchor_f32, anchor_f32

    try:
        return ivf_index.get_vector(target_vec_id), None
    except Exception:
        logger.exception(
            f"Could not retrieve vector for IVF ID {target_vec_id} (item_id: {target_item_id})"
        )
        return None


def _compute_num_to_query(n, radius_similarity, eliminate_duplicates, mood_similarity):
    if radius_similarity or eliminate_duplicates:
        k_increase = max(20, int(n * 3))
        num_to_query = n + k_increase + 1
        logger.info(
            f"Radius similarity enabled. Fetching a large candidate pool of {num_to_query} songs."
        )
    else:
        k_increase = max(3, int(n * 0.20))
        num_to_query = n + k_increase + 1
    if mood_similarity:
        base_multiplier = 8 if eliminate_duplicates else 4
        k_increase = max(20, int(n * base_multiplier))
        num_to_query = n + k_increase + 1
    return num_to_query


def _build_initial_neighbor_results(neighbor_vec_ids, distances, target_item_id):
    initial_results = []
    for vec_id, dist in zip(neighbor_vec_ids, distances):
        item_id = id_map.get(vec_id)
        if item_id and item_id != target_item_id:
            initial_results.append({"item_id": item_id, "distance": float(dist)})
    return initial_results


def _rerank_neighbors_by_f32(initial_results, anchor_f32, target_item_id, db_conn):
    f32_map = _fetch_f32_embeddings(db_conn, [r["item_id"] for r in initial_results])
    if not f32_map:
        return
    f32_map[target_item_id] = anchor_f32
    _prime_request_f32(f32_map)
    for r in initial_results:
        v = f32_map.get(r["item_id"])
        if v is not None:
            r["distance"] = get_direct_distance(anchor_f32, v)
    initial_results.sort(key=lambda r: r["distance"])


def _query_and_rerank_neighbors(query_vector, anchor_f32, num_to_query, n, target_item_id, db_conn):
    original_num_to_query = num_to_query
    if num_to_query > len(ivf_index):
        logger.warning(
            f"IVF query request for {n} final items was expanded to {original_num_to_query} neighbors for processing. "
            f"This exceeds the total items in the index ({len(ivf_index)}). "
            f"Capping the actual query to {len(ivf_index)} items."
        )
        num_to_query = len(ivf_index)

    rerank_k = min(len(ivf_index), max(num_to_query * IVF_RERANK_OVERFETCH, num_to_query + 200))
    try:
        if num_to_query <= 1:
            logger.warning(
                f"Number of neighbors to query ({num_to_query}) is too small. Skipping query."
            )
            neighbor_vec_ids, distances = [], []
        else:
            neighbor_vec_ids, distances = ivf_index.query(query_vector, k=rerank_k)
    except Exception:
        logger.exception(
            f"An unexpected error occurred during IVF query for item '{target_item_id}'"
        )
        return None

    initial_results = _build_initial_neighbor_results(neighbor_vec_ids, distances, target_item_id)

    if initial_results and anchor_f32 is not None:
        _rerank_neighbors_by_f32(initial_results, anchor_f32, target_item_id, db_conn)

    return initial_results[:num_to_query]


def _apply_artist_cap_by_ids(songs, get_score_data_by_ids):
    if MAX_SONGS_PER_ARTIST is None or MAX_SONGS_PER_ARTIST <= 0:
        return songs

    item_ids_to_check = [r['item_id'] for r in songs]
    track_details_list = get_score_data_by_ids(item_ids_to_check)
    details_map = {d['item_id']: {'author': d['author']} for d in track_details_list}

    return _apply_artist_cap(
        songs,
        lambda song: details_map.get(song['item_id'], {}).get('author'),
        warn_missing=True,
    )


def _finalize_nonradius_neighbors(
    initial_results,
    target_item_id,
    db_conn,
    target_song_details,
    mood_similarity,
    eliminate_duplicates,
    get_score_data_by_ids,
):
    original_song_for_filtering = {"item_id": target_item_id, "distance": 0.0}
    results_with_original = [original_song_for_filtering] + initial_results

    temp_filtered_results = _filter_by_distance(results_with_original, db_conn)

    distance_filtered_results = [
        song for song in temp_filtered_results if song['item_id'] != target_item_id
    ]
    unique_results_by_song = _deduplicate_and_filter_neighbors(
        distance_filtered_results, db_conn, target_song_details
    )

    effective_mood_nonradius = (
        MOOD_SIMILARITY_ENABLE if mood_similarity is None else mood_similarity
    )
    if effective_mood_nonradius:
        logger.info(
            f"Mood similarity filtering requested/enabled for target_item_id: {target_item_id}"
        )
        unique_results_by_song = _filter_by_mood_similarity(
            unique_results_by_song,
            target_item_id,
            db_conn,
            target_other_features=target_song_details.get('other_features'),
        )
    else:
        logger.info(
            f"Mood filtering skipped (mood_similarity={mood_similarity}, MOOD_SIMILARITY_ENABLE={MOOD_SIMILARITY_ENABLE})"
        )

    if eliminate_duplicates:
        return _apply_artist_cap_by_ids(unique_results_by_song, get_score_data_by_ids)
    return unique_results_by_song


def find_nearest_neighbors_by_id(
    target_item_id: str,
    n: int = 10,
    eliminate_duplicates: bool | None = None,
    mood_similarity: bool | None = None,
    radius_similarity: bool | None = None,
):
    try:
        return _find_nearest_neighbors_by_id_impl(
            target_item_id, n, eliminate_duplicates, mood_similarity, radius_similarity
        )
    finally:
        _clear_request_f32()


def _find_nearest_neighbors_by_id_impl(
    target_item_id: str,
    n: int = 10,
    eliminate_duplicates: bool | None = None,
    mood_similarity: bool | None = None,
    radius_similarity: bool | None = None,
):
    if ivf_index is None or id_map is None or reverse_id_map is None:
        raise RuntimeError(
            "IVF index is not loaded in memory. It may be missing, empty, or the server failed to load it on startup."
        )

    eff_radius = SIMILARITY_RADIUS_DEFAULT if radius_similarity is None else radius_similarity
    eff_eliminate = (
        SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT
        if eliminate_duplicates is None
        else eliminate_duplicates
    )
    eff_mood = MOOD_SIMILARITY_ENABLE if mood_similarity is None else mood_similarity
    from .paged_ivf import active_availability_scope
    _result_key = (
        active_availability_scope(), target_item_id, int(n), bool(eff_eliminate),
        bool(eff_mood), bool(eff_radius),
    )
    _cached = _neighbor_result_cache.get(_result_key)
    if _cached is not None:
        return [dict(r) for r in _cached]

    if ivf_index is not None:
        ivf_index.begin_request()

    from app_helper import get_db, get_score_data_by_ids

    db_conn = get_db()

    loaded = _load_target_for_neighbor_search(target_item_id, get_score_data_by_ids)
    if loaded is None:
        return []
    target_song_details, target_vec_id = loaded

    resolved = _resolve_neighbor_query_vector(target_item_id, target_vec_id, db_conn)
    if resolved is None:
        return []
    query_vector, anchor_f32 = resolved

    if radius_similarity is None:
        radius_similarity = SIMILARITY_RADIUS_DEFAULT

    if eliminate_duplicates is None:
        eliminate_duplicates = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT

    num_to_query = _compute_num_to_query(n, radius_similarity, eliminate_duplicates, mood_similarity)

    initial_results = _query_and_rerank_neighbors(
        query_vector, anchor_f32, num_to_query, n, target_item_id, db_conn
    )
    if initial_results is None:
        return []

    if radius_similarity:
        logger.info(f"Starting Radius Similarity walk for {n} songs...")

        candidate_data = _radius_walk_get_candidates(
            target_item_id=target_item_id,
            anchor_vector=query_vector,
            initial_results=initial_results,
            db_conn=db_conn,
            original_song_details=target_song_details,
            eliminate_duplicates=eliminate_duplicates,
            mood_similarity=mood_similarity,
        )

        final_results = _execute_radius_walk(
            n=n, candidate_data=candidate_data, eliminate_duplicates=eliminate_duplicates
        )
    else:
        final_results = _finalize_nonradius_neighbors(
            initial_results,
            target_item_id,
            db_conn,
            target_song_details,
            mood_similarity,
            eliminate_duplicates,
            get_score_data_by_ids,
        )
        final_results = final_results[:n]

    _neighbor_result_cache.put(_result_key, final_results)
    return [dict(r) for r in final_results]


def find_nearest_neighbors_by_vector(
    query_vector: np.ndarray, n: int = 100, eliminate_duplicates: bool | None = None
):
    try:
        return _find_nearest_neighbors_by_vector_impl(query_vector, n, eliminate_duplicates)
    finally:
        _clear_request_f32()


def _find_nearest_neighbors_by_vector_impl(
    query_vector: np.ndarray, n: int = 100, eliminate_duplicates: bool | None = None
):
    if ivf_index is None or id_map is None:
        raise RuntimeError("IVF index is not loaded in memory.")

    if ivf_index is not None:
        ivf_index.begin_request()
    _clear_request_f32()

    from app_helper import get_db

    db_conn = get_db()

    if eliminate_duplicates is None:
        eliminate_duplicates = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT

    if eliminate_duplicates:
        num_to_query = n + int(n * 4)
    else:
        num_to_query = n + int(n * 0.2)

    original_num_to_query = num_to_query
    if num_to_query > len(ivf_index):
        logger.warning(
            f"IVF query request for {n} final items was expanded to {original_num_to_query} neighbors for processing. "
            f"This exceeds the total items in the index ({len(ivf_index)}). "
            f"Capping the actual query to {len(ivf_index)} items."
        )
        num_to_query = len(ivf_index)

    try:
        if num_to_query <= 0:
            logger.warning("Number of neighbors to query is zero or less. Skipping query.")
            neighbor_vec_ids, distances = [], []
        else:
            neighbor_vec_ids, distances = ivf_index.query(query_vector, k=num_to_query)
    except Exception:
        logger.exception(
            "An unexpected error occurred during IVF query for synthetic vector",
        )
        return []

    initial_results = [
        {"item_id": id_map.get(vec_id), "distance": float(dist)}
        for vec_id, dist in zip(neighbor_vec_ids, distances)
        if id_map.get(vec_id) is not None
    ]

    if initial_results:
        _prime_request_f32(_fetch_f32_embeddings(db_conn, [r["item_id"] for r in initial_results]))

    distance_filtered_results = _filter_by_distance(initial_results, db_conn)

    item_ids = [r['item_id'] for r in distance_filtered_results]
    item_details = _fetch_details_map(db_conn, item_ids, SCORE_DETAIL_COLUMNS)

    unique_songs_by_content = _dedup_by_content(distance_filtered_results, item_details)

    if eliminate_duplicates:
        final_results = _apply_artist_cap(
            unique_songs_by_content,
            lambda song: (item_details.get(song['item_id']) or {}).get('author'),
        )
    else:
        final_results = unique_songs_by_content

    return final_results[:n]


def get_max_distance_for_id(target_item_id: str):
    if ivf_index is None or id_map is None or reverse_id_map is None:
        raise RuntimeError(
            "IVF index is not loaded in memory. It may be missing, empty, or the server failed to load it on startup."
        )

    target_vec_id = reverse_id_map.get(target_item_id)
    if target_vec_id is None:
        return None

    from .paged_ivf import active_availability_scope
    cache_key = (active_availability_scope(), target_item_id)
    _cached = _max_distance_cache.get(cache_key)
    if _cached is not None:
        return dict(_cached)

    ivf_index.begin_request()
    _clear_request_f32()
    try:
        max_d, far_vec_id = ivf_index.get_max_distance(target_vec_id)
    except Exception:
        logger.exception(f"Error computing IVF max distance for {target_item_id}")
        return None
    if max_d is None:
        return None
    far_item_id = id_map.get(far_vec_id) if far_vec_id is not None else None
    result = {'max_distance': float(max_d), 'farthest_item_id': far_item_id}
    _max_distance_cache.put(cache_key, result)
    return dict(result)


def get_item_id_by_title_and_artist(title: str, artist: str):
    from app_helper import get_db

    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        query = "SELECT item_id FROM score WHERE LOWER(title) = LOWER(%s) AND LOWER(author) = LOWER(%s) LIMIT 1"
        cur.execute(query, (title, artist))
        result = cur.fetchone()
        if result:
            return result['item_id']

        query = """
            SELECT item_id, title, author,
                   similarity(LOWER(title), LOWER(%s)) + similarity(LOWER(author), LOWER(%s)) AS score
            FROM score
            WHERE LOWER(title) ILIKE LOWER(%s) AND LOWER(author) ILIKE LOWER(%s)
            ORDER BY score DESC
            LIMIT 1
        """
        cur.execute(query, (title, artist, f"%{title}%", f"%{artist}%"))
        result = cur.fetchone()
        if result:
            logger.info(
                f"Fuzzy matched '{title}' by '{artist}' to '{result['title']}' by '{result['author']}'"
            )
            return result['item_id']

        return None
    except Exception:
        logger.exception(f"Error fetching item_id for '{title}' by '{artist}'")
        return None
    finally:
        cur.close()


def search_tracks_unified(
    search_query: str,
    limit: int = 20,
    offset: int = 0,
    item_id_filter: set | None = None,
    server_id: str | None = None,
    include_legacy_default: bool = False,
):
    from app_helper import get_db
    from psycopg2.extras import DictCursor

    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    results = []

    try:
        if not search_query:
            return []

        tokens = [t.lower() for t in search_query.strip().split() if t]
        if not tokens:
            return []

        where_clauses = []
        score_clauses = []
        params = []

        for token in tokens:
            like_pattern = f"%{token}%"
            where_clauses.append("search_u LIKE unaccent(%s)")
            params.append(like_pattern)

        for token in tokens:
            like_pattern = f"%{token}%"
            score_clauses.append("""
                (CASE WHEN lower(unaccent(title))  LIKE unaccent(%s) THEN 3 ELSE 0 END) +
                (CASE WHEN lower(unaccent(author)) LIKE unaccent(%s) THEN 2 ELSE 0 END) +
                (CASE WHEN lower(unaccent(album))  LIKE unaccent(%s) THEN 1 ELSE 0 END)
            """)
            params.extend([like_pattern, like_pattern, like_pattern])

        where_sql = " AND ".join(where_clauses)
        score_sql = " + ".join(score_clauses)

        if item_id_filter is not None and not item_id_filter:
            return []

        id_filter_sql = ""
        id_filter_params: list = []
        if item_id_filter:
            id_placeholders = ",".join(["%s"] * len(item_id_filter))
            id_filter_sql = f" AND item_id IN ({id_placeholders})"
            id_filter_params = list(item_id_filter)

        availability_sql = ""
        availability_params: list = []
        if server_id:
            availability_sql = (
                " AND (EXISTS (SELECT 1 FROM track_server_map availability "
                "WHERE availability.item_id = score.item_id AND availability.server_id = %s) "
                "OR (%s AND left(score.item_id, 3) <> 'fp_'))"
            )
            availability_params = [server_id, bool(include_legacy_default)]

        all_params = (
            params[: len(tokens)]
            + id_filter_params
            + availability_params
            + params[len(tokens) :]
        )

        query = f"""
            SELECT item_id, title, author, album, album_artist
            FROM score
            WHERE {where_sql}{id_filter_sql}{availability_sql}
            ORDER BY ({score_sql}) DESC,
                     title,
                     author,
                     album
            LIMIT %s OFFSET %s
        """

        all_params.append(limit)
        all_params.append(offset)

        cur.execute(query, tuple(all_params))
        results = [dict(row) for row in cur.fetchall()]

    except Exception:
        logger.exception(f"Error searching tracks with query '{search_query}'")
    finally:
        cur.close()

    return results


def create_playlist_from_ids(playlist_name: str, track_ids: list, user_creds: dict | None = None):
    try:
        from .mediaserver import create_instant_playlist

        created_playlist = create_instant_playlist(playlist_name, track_ids, user_creds=user_creds)

        if not created_playlist:
            raise RuntimeError(
                "Playlist creation failed. The media server did not return a playlist object."
            )

        playlist_id = created_playlist.get('Id')

        if not playlist_id:
            raise RuntimeError("Media server API response did not include a playlist ID.")

        return playlist_id

    except Exception as e:
        raise e


def cleanup_resources():
    logger.info("Cleaning up similarity manager resources...")
    _shutdown_thread_pool()
    try:
        from tasks.paged_ivf import shutdown_query_pool

        shutdown_query_pool()
    except Exception:
        logger.debug("IVF query pool shutdown failed during cleanup.", exc_info=True)
    logger.info("Similarity manager cleanup complete.")
