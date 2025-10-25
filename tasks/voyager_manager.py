import os
import json
import logging
import tempfile
import numpy as np
import voyager # type: ignore
import psycopg2 # type: ignore
from psycopg2.extras import DictCursor
import io 
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
import math # Import math for ceiling function

from config import (
    EMBEDDING_DIMENSION, INDEX_NAME, VOYAGER_METRIC, VOYAGER_EF_CONSTRUCTION,
    VOYAGER_M, VOYAGER_QUERY_EF, MAX_SONGS_PER_ARTIST,
    DUPLICATE_DISTANCE_THRESHOLD_COSINE, DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN,
    DUPLICATE_DISTANCE_CHECK_LOOKBACK, MOOD_SIMILARITY_THRESHOLD
    , SIMILARITY_RADIUS_DEFAULT
)
# Import from other project modules
from .mediaserver import create_instant_playlist

logger = logging.getLogger(__name__)

# --- Global cache for the loaded Voyager index ---
voyager_index = None
id_map = None # {voyager_int_id: item_id_str}
reverse_id_map = None # {item_id_str: voyager_int_id}

# --- Thread pool for parallel operations ---
_thread_pool = None
_thread_pool_lock = threading.Lock()

# --- Configuration for parallel processing ---
MAX_WORKER_THREADS = min(4, (os.cpu_count() or 1))  # Use up to 4 threads or available cores
BATCH_SIZE_VECTOR_OPS = 50  # Process vectors in batches
BATCH_SIZE_DB_OPS = 100     # Process database operations in batches

def _get_thread_pool():
    """Get or create the global thread pool for parallel operations."""
    global _thread_pool
    with _thread_pool_lock:
        if _thread_pool is None:
            _thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS, thread_name_prefix="voyager")
        return _thread_pool

def _shutdown_thread_pool():
    """Shutdown the global thread pool."""
    global _thread_pool
    with _thread_pool_lock:
        if _thread_pool is not None:
            _thread_pool.shutdown(wait=True)
            _thread_pool = None

# --- LRU cache for frequently accessed vectors ---
@lru_cache(maxsize=1000)
def _get_cached_vector(item_id: str) -> np.ndarray | None:
    """Cached version of get_vector_by_id for better performance."""
    if voyager_index is None or reverse_id_map is None:
        return None
    
    voyager_id = reverse_id_map.get(item_id)
    if voyager_id is None:
        return None
    
    try:
        return voyager_index.get_vector(voyager_id)
    except Exception:
        return None

# --- NEW HELPER FUNCTIONS FOR DIRECT DISTANCE CALCULATION ---
def _get_direct_euclidean_distance(v1, v2):
    """Compute direct Euclidean distance between two vectors. Returns +inf if unavailable."""
    if v1 is None or v2 is None:
        return float('inf')
    try:
        # Use fp32 for intermediate calculations to prevent overflow/underflow
        dist = np.linalg.norm(v1.astype(np.float32) - v2.astype(np.float32))
        return float(dist)
    except Exception:
        return float('inf')


def _get_direct_cosine_distance(v1, v2):
    """Compute cosine distance (1 - cosine_similarity). Returns +inf if unavailable."""
    if v1 is None or v2 is None:
        return float('inf')
    try:
        # Use fp32 for intermediate calculations
        v1_f32 = v1.astype(np.float32)
        v2_f32 = v2.astype(np.float32)

        norm_v1 = np.linalg.norm(v1_f32)
        norm_v2 = np.linalg.norm(v2_f32)

        denom = norm_v1 * norm_v2
        if denom == 0:
            return float('inf')

        dot_product = np.dot(v1_f32, v2_f32)
        cos_sim = dot_product / denom

        # Clamp value to [-1.0, 1.0] to correct for floating point inaccuracies
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        return 1.0 - float(cos_sim)
    except Exception:
        return float('inf')


def get_direct_distance(v1, v2):
    """Public helper that picks the metric according to VOYAGER_METRIC."""
    if VOYAGER_METRIC == 'angular':
        return _get_direct_cosine_distance(v1, v2)
    return _get_direct_euclidean_distance(v1, v2)


def load_voyager_index_for_querying(force_reload=False):
    """
    Loads the Voyager index from the database into the global in-memory cache.
    This function is imported at module-level by the app startup path; keep it stable.
    """
    global voyager_index, id_map, reverse_id_map

    if voyager_index is not None and not force_reload:
        logger.info("Voyager index is already loaded in memory. Skipping reload.")
        return

    # Clear the vector cache when reloading
    if force_reload:
        _get_cached_vector.cache_clear()

    from app_helper import get_db
    logger.info("Attempting to load Voyager index from database into memory...")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT index_data, id_map_json, embedding_dimension FROM voyager_index_data WHERE index_name = %s", (INDEX_NAME,))
        record = cur.fetchone()

        if not record:
            logger.warning(f"Voyager index '{INDEX_NAME}' not found in the database. Cache will be empty.")
            voyager_index, id_map, reverse_id_map = None, None, None
            return
        index_binary_data, id_map_json, db_embedding_dim = record

        if not index_binary_data:
            logger.error(f"Voyager index '{INDEX_NAME}' data in database is empty.")
            voyager_index, id_map, reverse_id_map = None, None, None
            return

        if db_embedding_dim != EMBEDDING_DIMENSION:
            logger.error(f"FATAL: Voyager index dimension mismatch! DB has {db_embedding_dim}, config expects {EMBEDDING_DIMENSION}.")
            voyager_index, id_map, reverse_id_map = None, None, None
            return

        index_stream = io.BytesIO(index_binary_data)
        loaded_index = voyager.Index.load(index_stream)
        loaded_index.ef = VOYAGER_QUERY_EF
        voyager_index = loaded_index
        id_map = {int(k): v for k, v in json.loads(id_map_json).items()}
        reverse_id_map = {v: k for k, v in id_map.items()}

        logger.info(f"Voyager index with {len(id_map)} items loaded successfully into memory.")

    except Exception as e:
        logger.error("Failed to load Voyager index from database: %s", e, exc_info=True)
        voyager_index, id_map, reverse_id_map = None, None, None
    finally:
        cur.close()
def build_and_store_voyager_index(db_conn=None, force_rebuild: bool = False):
        """
        Placeholder/compatibility shim for the original index build function.

        Many modules import this function at module-import time. The full index
        builder is an expensive operation and may be invoked from CLI/worker
        code paths. To avoid ImportError during startup, this stub provides a
        no-op implementation that logs the call. If you need the full builder
        behavior, replace this stub with the original implementation or call
        the dedicated rebuild script.

        Parameters:
            db_conn: optional database connection (may be required by real builder)
            force_rebuild: when True, a full rebuild should be forced (ignored by stub)

        Returns: None
        """
        logger.info("build_and_store_voyager_index called (stub). force_rebuild=%s. No action taken.", force_rebuild)
        return None

def get_vector_by_id(item_id: str) -> np.ndarray | None:
    """
    Retrieves the embedding vector for a given item_id from the loaded Voyager index.
    Uses caching for better performance.
    """
    return _get_cached_vector(item_id)

def _normalize_string(text: str) -> str:
    """Lowercase and strip whitespace."""
    if not text:
        return ""
    return text.strip().lower()

def _is_same_song(title1, artist1, title2, artist2):
    """
    Determines if two songs are identical based on title and artist.
    Comparison is case-insensitive.
    """
    norm_title1 = _normalize_string(title1)
    norm_title2 = _normalize_string(title2)
    norm_artist1 = _normalize_string(artist1)
    norm_artist2 = _normalize_string(artist2)
    
    return norm_title1 == norm_title2 and norm_artist1 == norm_artist2

def _compute_distance_batch(song_batch, lookback_songs, threshold, metric_name, details_map):
    """Compute distances for a batch of songs in parallel."""
    batch_results = []
    
    for current_song in song_batch:
        is_too_close = False
        current_vector = _get_cached_vector(current_song['item_id'])
        if current_vector is None:
            continue
        
        # Check against lookback window
        for recent_song in lookback_songs:
            recent_vector = _get_cached_vector(recent_song['item_id'])
            if recent_vector is None:
                continue

            direct_dist = get_direct_distance(current_vector, recent_vector)
            
            if direct_dist < threshold:
                current_details = details_map.get(current_song['item_id'], {'title': 'N/A', 'author': 'N/A'})
                recent_details = details_map.get(recent_song['item_id'], {'title': 'N/A', 'author': 'N/A'})
                logger.info(
                    f"Filtering song (DISTANCE FILTER) with {metric_name} distance: '{current_details['title']}' by '{current_details['author']}' "
                    f"due to direct distance of {direct_dist:.4f} from "
                    f"'{recent_details['title']}' by '{recent_details['author']}' (Threshold: {threshold})."
                )
                is_too_close = True
                break
        
        if not is_too_close:
            batch_results.append(current_song)
    
    return batch_results

def _filter_by_distance(song_results: list, db_conn):
    """
    Filters a list of songs to remove items that are too close in direct vector distance
    to a lookback window of previously kept songs. Uses parallel processing for better performance.
    """
    if DUPLICATE_DISTANCE_CHECK_LOOKBACK <= 0:
        return song_results

    if not song_results:
        return []

    # Fetch all song details in parallel batches
    item_ids = [s['item_id'] for s in song_results]
    details_map = {}
    
    # Process DB queries in batches
    def fetch_details_batch(id_batch):
        batch_details = {}
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)", (id_batch,))
            rows = cur.fetchall()
            for row in rows:
                batch_details[row['item_id']] = {'title': row['title'], 'author': row['author']}
        return batch_details
    
    # Split item_ids into batches for parallel DB queries
    id_batches = [item_ids[i:i + BATCH_SIZE_DB_OPS] for i in range(0, len(item_ids), BATCH_SIZE_DB_OPS)]
    
    if len(id_batches) > 1:
        # Use parallel DB queries for large datasets
        executor = _get_thread_pool()
        future_to_batch = {executor.submit(fetch_details_batch, batch): batch for batch in id_batches}
        
        for future in as_completed(future_to_batch):
            batch_details = future.result()
            details_map.update(batch_details)
    else:
        # Use single query for small datasets
        details_map = fetch_details_batch(item_ids)

    threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE if VOYAGER_METRIC == 'angular' else DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    metric_name = 'Angular' if VOYAGER_METRIC == 'angular' else 'Euclidean'
    
    filtered_songs = []
    
    # For small datasets, use sequential processing
    if len(song_results) <= BATCH_SIZE_VECTOR_OPS:
        for current_song in song_results:
            is_too_close = False
            current_vector = _get_cached_vector(current_song['item_id'])
            if current_vector is None:
                continue

            # Check against the last N songs in the filtered list
            lookback_window = filtered_songs[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]
            for recent_song in lookback_window:
                recent_vector = _get_cached_vector(recent_song['item_id'])
                if recent_vector is None:
                    continue

                direct_dist = get_direct_distance(current_vector, recent_vector)
                
                if direct_dist < threshold:
                    current_details = details_map.get(current_song['item_id'], {'title': 'N/A', 'author': 'N/A'})
                    recent_details = details_map.get(recent_song['item_id'], {'title': 'N/A', 'author': 'N/A'})
                    logger.info(
                        f"Filtering song (DISTANCE FILTER) with {metric_name} distance: '{current_details['title']}' by '{current_details['author']}' "
                        f"due to direct distance of {direct_dist:.4f} from "
                        f"'{recent_details['title']}' by '{recent_details['author']}' (Threshold: {threshold})."
                    )
                    is_too_close = True
                    break
            
            if not is_too_close:
                filtered_songs.append(current_song)
    else:
        # For larger datasets, use parallel processing with rolling lookback
        remaining_songs = song_results.copy()
        
        while remaining_songs:
            # Process next batch
            current_batch = remaining_songs[:BATCH_SIZE_VECTOR_OPS]
            remaining_songs = remaining_songs[BATCH_SIZE_VECTOR_OPS:]
            
            # Get current lookback window
            lookback_window = filtered_songs[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:] if filtered_songs else []
            
            # Process batch
            batch_results = _compute_distance_batch(current_batch, lookback_window, threshold, metric_name, details_map)
            filtered_songs.extend(batch_results)

    return filtered_songs


def _deduplicate_and_filter_neighbors(song_results: list, db_conn, original_song_details: dict):
    """
    Filters a list of songs to remove duplicates based on exact title/artist match.
    Uses parallel processing for better performance on large datasets.
    """
    if not song_results:
        return []

    # Fetch song details in parallel batches
    item_ids = [r['item_id'] for r in song_results]
    item_details = {}
    
    def fetch_details_batch(id_batch):
        batch_details = {}
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)", (id_batch,))
            rows = cur.fetchall()
            for row in rows:
                batch_details[row['item_id']] = {'title': row['title'], 'author': row['author']}
        return batch_details
    
    # Split item_ids into batches for parallel DB queries
    id_batches = [item_ids[i:i + BATCH_SIZE_DB_OPS] for i in range(0, len(item_ids), BATCH_SIZE_DB_OPS)]
    
    if len(id_batches) > 1:
        # Use parallel DB queries for large datasets
        executor = _get_thread_pool()
        future_to_batch = {executor.submit(fetch_details_batch, batch): batch for batch in id_batches}
        
        for future in as_completed(future_to_batch):
            batch_details = future.result()
            item_details.update(batch_details)
    else:
        # Use single query for small datasets
        item_details = fetch_details_batch(item_ids)

    unique_songs = []

    # --- PERFORMANCE OPTIMIZATION: Use a set for O(1) lookups ---
    # Store normalized (title, artist) tuples
    added_songs_signatures = set()
    
    # Add the original song to the set to filter it out
    original_title = _normalize_string(original_song_details.get('title'))
    original_author = _normalize_string(original_song_details.get('author'))
    added_songs_signatures.add((original_title, original_author))

    for song in song_results:
        current_details = item_details.get(song['item_id'])
        if not current_details:
            logger.warning(f"Could not find details for item_id {song['item_id']} during deduplication. Skipping.")
            continue

        # --- OPTIMIZATION: Normalize once ---
        current_title = _normalize_string(current_details.get('title'))
        current_author = _normalize_string(current_details.get('author'))
        current_signature = (current_title, current_author)

        # --- OPTIMIZATION: O(1) set lookup instead of O(N) list loop ---
        if current_signature not in added_songs_signatures:
            unique_songs.append(song)
            added_songs_signatures.add(current_signature)
        else:
            # This log was present before, keep it for consistency
            logger.info(f"Found duplicate (NAME FILTER): '{current_details.get('title')}' by '{current_details.get('author')}' (Distance from source: {song.get('distance', 0.0):.4f}).")

    return unique_songs

def _compute_mood_distances_batch(song_batch, target_mood_features, candidate_mood_features, mood_features, mood_threshold):
    """Compute mood distances for a batch of songs in parallel."""
    batch_results = []
    
    for song in song_batch:
        candidate_features = candidate_mood_features.get(song['item_id'])
        if not candidate_features:
            continue  # Skip songs without mood features

        # Calculate mood distance (sum of absolute differences)
        mood_distance = sum(
            abs(target_mood_features.get(feature, 0.0) - candidate_features.get(feature, 0.0))
            for feature in mood_features
        )
        
        # Normalize by number of features
        normalized_mood_distance = mood_distance / len(mood_features)
        
        if normalized_mood_distance <= mood_threshold:
            # Add mood distance info to the song result
            song_with_mood = song.copy()
            song_with_mood['mood_distance'] = normalized_mood_distance
            batch_results.append(song_with_mood)
    
    return batch_results

def _filter_by_mood_similarity(song_results: list, target_item_id: str, db_conn, mood_threshold: float = None):
    """
    Filters songs by mood similarity using the other_features stored in the database.
    Keeps songs with similar mood features (danceability, aggressive, happy, party, relaxed, sad).
    Uses parallel processing for better performance.
    """
    if not song_results:
        return []

    # Use config value if no threshold provided
    if mood_threshold is None:
        mood_threshold = MOOD_SIMILARITY_THRESHOLD

    # Get target song mood features
    with db_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT other_features FROM score WHERE item_id = %s", (target_item_id,))
        target_row = cur.fetchone()
        if not target_row or not target_row['other_features']:
            logger.warning(f"No mood features found for target song {target_item_id}. Skipping mood filtering.")
            return song_results

        target_mood_features = _parse_mood_features(target_row['other_features'])
        if not target_mood_features:
            logger.warning(f"Could not parse mood features for target song {target_item_id}. Skipping mood filtering.")
            return song_results

        logger.info(f"Target song {target_item_id} mood features: {target_mood_features}")

        # Get mood features for all candidate songs in batches
        candidate_ids = [s['item_id'] for s in song_results]
        candidate_mood_features = {}
        
        # Process DB queries in batches for better performance
        def fetch_mood_features_batch(id_batch):
            batch_features = {}
            with db_conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SELECT item_id, other_features FROM score WHERE item_id = ANY(%s)", (id_batch,))
                rows = cur.fetchall()
                for row in rows:
                    if row['other_features']:
                        parsed_features = _parse_mood_features(row['other_features'])
                        if parsed_features:
                            batch_features[row['item_id']] = parsed_features
            return batch_features
        
        # Split candidate_ids into batches
        id_batches = [candidate_ids[i:i + BATCH_SIZE_DB_OPS] for i in range(0, len(candidate_ids), BATCH_SIZE_DB_OPS)]
        
        if len(id_batches) > 1:
            # Use parallel DB queries for large datasets
            executor = _get_thread_pool()
            future_to_batch = {executor.submit(fetch_mood_features_batch, batch): batch for batch in id_batches}
            
            for future in as_completed(future_to_batch):
                batch_features = future.result()
                candidate_mood_features.update(batch_features)
        else:
            # Use single query for small datasets
            candidate_mood_features = fetch_mood_features_batch(candidate_ids)

    # Filter by mood similarity
    mood_features = ['danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad']
    
    logger.info(f"Starting mood filtering with {len(song_results)} candidates, threshold: {mood_threshold}")
    
    # For small datasets, use sequential processing
    if len(song_results) <= BATCH_SIZE_VECTOR_OPS:
        filtered_songs = []
        for song in song_results:
            candidate_features = candidate_mood_features.get(song['item_id'])
            if not candidate_features:
                logger.debug(f"Skipping song {song['item_id']}: no mood features found")
                continue

            # Calculate mood distance (sum of absolute differences)
            mood_distance = sum(
                abs(target_mood_features.get(feature, 0.0) - candidate_features.get(feature, 0.0))
                for feature in mood_features
            )
            
            # Normalize by number of features
            normalized_mood_distance = mood_distance / len(mood_features)
            
            logger.debug(f"Song {song['item_id']} mood distance: {normalized_mood_distance:.4f}, features: {candidate_features}")
            
            if normalized_mood_distance <= mood_threshold:
                song_with_mood = song.copy()
                song_with_mood['mood_distance'] = normalized_mood_distance
                filtered_songs.append(song_with_mood)
                logger.debug(f"  -> KEPT (distance: {normalized_mood_distance:.4f})")
            else:
                logger.debug(f"  -> FILTERED OUT (distance: {normalized_mood_distance:.4f} > threshold: {mood_threshold})")
    else:
        # For larger datasets, use parallel processing
        song_batches = [song_results[i:i + BATCH_SIZE_VECTOR_OPS] for i in range(0, len(song_results), BATCH_SIZE_VECTOR_OPS)]
        
        executor = _get_thread_pool()
        future_to_batch = {
            executor.submit(_compute_mood_distances_batch, batch, target_mood_features, candidate_mood_features, mood_features, mood_threshold): batch 
            for batch in song_batches
        }
        
        filtered_songs = []
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            filtered_songs.extend(batch_results)

    logger.info(f"Mood filtering results: kept {len(filtered_songs)} of {len(song_results)} songs (threshold: {mood_threshold})")
    return filtered_songs

def _parse_mood_features(other_features_str: str) -> dict:
    """
    Parses the other_features string to extract mood values.
    Expected format: "danceable:0.123,aggressive:0.456,..."
    """
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

# --- START: RADIUS SIMILARITY RE-IMPLEMENTATION ---

def _radius_walk_get_candidates(
    target_item_id: str, 
    anchor_vector: np.ndarray, 
    initial_results: list, 
    db_conn, 
    original_song_details: dict, 
    eliminate_duplicates: bool
) -> list:
    """
    Prepares the candidate pool for the radius walk.
    This involves:
    1. Prepending the original song.
    2. Filtering by distance (to remove echoes of the anchor song).
    3. Filtering by name/artist (to remove exact duplicates).
    4. Filtering by artist cap (if eliminate_duplicates is True).
    5. Pre-calculating and caching vectors and distances to the anchor.
    """
    from app_helper import get_score_data_by_ids
    
    # --- PERFORMANCE: Skip the slow O(N*K) _filter_by_distance. ---
    # The radius walk is *designed* to find a diverse path, making the
    # pre-filter redundant and slow. We only apply the name/artist filter.
    logger.info(f"Radius walk: Skipping slow pre-filter _filter_by_distance for {len(initial_results)} candidates.")

    # 1. Filter by name/artist (This is now O(N) thanks to optimization)
    # This filter now also removes the original song by checking against original_song_details
    unique_results_by_song = _deduplicate_and_filter_neighbors(initial_results, db_conn, original_song_details)
    
    # 2. Apply artist cap pre-walk if requested
    if eliminate_duplicates:
        item_ids_to_check = [r['item_id'] for r in unique_results_by_song]
        track_details_list = get_score_data_by_ids(item_ids_to_check)
        details_map = {d['item_id']: {'author': d['author']} for d in track_details_list}
        artist_counts = {}
        final_filtered = []
        for song in unique_results_by_song:
            song_id = song['item_id']
            author = details_map.get(song_id, {}).get('author')
            if not author:
                continue
            current_count = artist_counts.get(author, 0)
            if current_count < MAX_SONGS_PER_ARTIST:
                final_filtered.append(song)
                artist_counts[author] = current_count + 1
        unique_results_by_song = final_filtered
        logger.info(f"Radius walk: artist cap applied pre-walk. Candidate pool size reduced to {len(unique_results_by_song)}.")

    # 2.b Optional: apply distance-based duplicate filtering (use the same logic as non-radius path).
    # Prepend the original song so the distance filter can compare candidates against the anchor.
    try:
        if unique_results_by_song:
            original_for_filter = {"item_id": target_item_id, "distance": 0.0}
            # _filter_by_distance honors DUPLICATE_DISTANCE_CHECK_LOOKBACK and will be a no-op if disabled.
            filtered_after_distance = _filter_by_distance([original_for_filter] + unique_results_by_song, db_conn)
            # Remove the original entry we prepended
            filtered_after_distance = [s for s in filtered_after_distance if s['item_id'] != target_item_id]
            logger.info(f"Radius walk: distance-based filtering reduced candidates {len(unique_results_by_song)} -> {len(filtered_after_distance)}")
            unique_results_by_song = filtered_after_distance
    except Exception:
        # If distance filtering fails for any reason, continue with the previous candidate set.
        logger.exception("Radius walk: distance-based pre-filter failed, continuing without it.")
    
    # 2.c Apply mood-similarity filtering as a pre-walk candidate filter for radius mode.
    # This mirrors the non-radius behavior but runs before the greedy walk.
    try:
        if unique_results_by_song:
            before_mood = len(unique_results_by_song)
            unique_results_by_song = _filter_by_mood_similarity(unique_results_by_song, target_item_id, db_conn)
            after_mood = len(unique_results_by_song)
            logger.info(f"Radius walk: mood-based filtering reduced candidates {before_mood} -> {after_mood}")
    except Exception:
        logger.exception("Radius walk: mood-based pre-filter failed, continuing without it.")
    
    # 3. Fetch item details for remaining candidates and pre-calculate/cache data for the walk
    candidate_data = []
    if unique_results_by_song:
        item_ids_to_fetch = [r['item_id'] for r in unique_results_by_song]
        # Fetch details in batch (uses app_helper get_score_data_by_ids)
        try:
            track_details_list = get_score_data_by_ids(item_ids_to_fetch)
            details_map = {d['item_id']: {'title': d['title'], 'author': d['author']} for d in track_details_list}
        except Exception:
            details_map = {}

        for song in unique_results_by_song:
            item_id = song['item_id']
            vector = _get_cached_vector(item_id)
            if vector is not None:
                # Normalize vectors to float32 once to avoid repeated casting later
                try:
                    vector = vector.astype(np.float32)
                except Exception:
                    vector = np.array(vector, dtype=np.float32)
                dist_to_anchor = get_direct_distance(vector, anchor_vector)
                info = details_map.get(item_id, {'title': None, 'author': None})
                candidate_data.append({
                    "item_id": item_id,
                    "vector": vector,
                    "dist_anchor": dist_to_anchor,
                    "title": info.get('title'),
                    "author": info.get('author')
                })
            
    logger.info(f"Radius walk: pre-calculated vectors and distances for {len(candidate_data)} candidates.")
    return candidate_data


def _execute_radius_walk(
    target_item_id: str,
    n: int,
    candidate_data: list,
    original_song_details: dict | None = None
) -> list:
    """
    Executes the bucketed greedy walk based on the pre-filtered and pre-calculated candidate data.
    """
    if not candidate_data:
        logger.warning("Radius walk: No candidates available after filtering. Returning empty list.")
        return []

    # --- Parameters for the walk ---
    BUCKET_SIZE = 50
    
    # --- BUG FIX: Ensure we scan *at least* enough buckets to get N songs ---
    # The previous logic `max(2, ...)` was the bug.
    # This logic ensures we scan at least 3 buckets (for variety) OR
    # enough buckets to cover the requested N items, whichever is larger.
    BUCKETS_TO_SCAN = max(3, int(math.ceil(n / BUCKET_SIZE)))
    
    logger.info(f"Radius walk: N={n}, BUCKET_SIZE={BUCKET_SIZE}, BUCKETS_TO_SCAN={BUCKETS_TO_SCAN}")

    # --- Step 1: Sort candidates by distance to anchor and create buckets ---
    # We also precompute numpy arrays per-bucket to vectorize distance computations
    candidate_data.sort(key=lambda x: x['dist_anchor'])

    num_buckets = int(math.ceil(len(candidate_data) / BUCKET_SIZE))
    raw_buckets = [
        candidate_data[i * BUCKET_SIZE : (i + 1) * BUCKET_SIZE]
        for i in range(num_buckets)
    ]

    buckets = []
    for b in raw_buckets:
        ids = [c['item_id'] for c in b]
        # Stack vectors into an (m, dim) array for fast vectorized ops
        if b:
            # vectors are already cast to float32 in candidate_data; stack without further casting
            vecs = np.vstack([c['vector'] for c in b])
            dist_anchor_arr = np.array([c['dist_anchor'] for c in b], dtype=np.float32)
        else:
            vecs = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
            dist_anchor_arr = np.empty((0,), dtype=np.float32)

        buckets.append({
            'items': b,
            'ids': ids,
            'vecs': vecs,
            'dist_anchor': dist_anchor_arr,
        })

    logger.info(f"Radius walk: Created {len(buckets)} buckets of size {BUCKET_SIZE} (vectorized).")

    # --- Step 2: Initialize the walk ---
    
    # We will return n songs, so we need to collect n items.
    # The playlist_ids will store the final ordered list of item_ids.
    playlist_ids = []
    
    # used_ids keeps track of candidates already in the playlist for O(1) lookup.
    used_ids = set()

    # The walk starts from the song closest to the anchor (A).
    # This first song (B) is the start of our re-ordered playlist.
    try:
        first_song = candidate_data[0]
        playlist_ids.append(first_song['item_id'])
        used_ids.add(first_song['item_id'])
        current_song_vector = first_song['vector'].astype(np.float32)
        current_song_id = first_song['item_id']
        logger.info(f"Radius walk: Starting walk with song {current_song_id}.")
    except IndexError:
        logger.warning("Radius walk: Candidate data was empty, cannot start walk.")
        return []

    # Maintain a dict of selected item_id -> vector for distance checks during the walk
    selected_vectors = {playlist_ids[0]: current_song_vector}

    # Maintain a set of normalized (title, author) signatures for selected songs to prevent name duplicates
    selected_signatures = set()
    try:
        if original_song_details:
            sig = (_normalize_string(original_song_details.get('title')), _normalize_string(original_song_details.get('author')))
            selected_signatures.add(sig)
    except Exception:
        pass

    # If first_song contains title/author, add its signature
    try:
        first_title = first_song.get('title')
        first_author = first_song.get('author')
        if first_title or first_author:
            selected_signatures.add((_normalize_string(first_title), _normalize_string(first_author)))
    except Exception:
        pass

    except IndexError:
        logger.warning("Radius walk: Candidate data was empty, cannot start walk.")
        return []

    # --- Step 3: Iteratively build the playlist (Greedy, low-cost scanning) ---

    # Optimization: instead of computing distances for many candidates each iteration,
    # we maintain a per-bucket pointer and evaluate only the next un-used candidate
    # in the nearest BUCKETS_TO_SCAN buckets each loop. This drastically reduces the
    # number of direct distance calculations while keeping a greedy selection.

    num_buckets = len(buckets)
    # per-bucket next index pointer
    next_idx = [0] * num_buckets

    # Collect songs until we have n or run out
    while len(playlist_ids) < n:
        best_next_song = None
        best_bucket = None
        best_idx_in_bucket = None
        lowest_score = float('inf')

        # Limit scanning to the nearest buckets only
        buckets_to_check = min(num_buckets, BUCKETS_TO_SCAN)

        for bi in range(buckets_to_check):
            bucket = buckets[bi]
            if bucket['vecs'].size == 0:
                continue

            # Advance pointer to the next unused candidate in this bucket
            while next_idx[bi] < len(bucket['ids']) and bucket['ids'][next_idx[bi]] in used_ids:
                next_idx[bi] += 1

            if next_idx[bi] >= len(bucket['ids']):
                continue

            idx = next_idx[bi]
            try:
                # Prefer vectorized numpy norm for speed; fallback to get_direct_distance
                cand_vec = bucket['vecs'][idx]
                diffs = cand_vec - current_song_vector
                dist_prev = float(np.linalg.norm(diffs))
            except Exception:
                dist_prev = get_direct_distance(current_song_vector, bucket['items'][idx]['vector'])
            # Additional duplicate checks against songs already selected in the walk
            # 1) Name/artist duplicate
            candidate_item = bucket['items'][idx]
            cand_title = candidate_item.get('title')
            cand_author = candidate_item.get('author')
            cand_signature = (_normalize_string(cand_title), _normalize_string(cand_author))
            if cand_signature in selected_signatures:
                # Skip this candidate (treat as used)
                # Advance pointer for this bucket and continue to next bucket
                next_idx[bi] = idx + 1
                continue

            # 2) Distance duplicate against recent selected songs (lookback)
            # Use the configured threshold depending on metric
            threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE if VOYAGER_METRIC == 'angular' else DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
            lookback_ids = list(playlist_ids[-DUPLICATE_DISTANCE_CHECK_LOOKBACK:]) if DUPLICATE_DISTANCE_CHECK_LOOKBACK > 0 else []
            too_close = False
            for sel_id in lookback_ids:
                sel_vec = selected_vectors.get(sel_id)
                if sel_vec is None:
                    # Try to fetch and cache
                    try:
                        sel_vec = _get_cached_vector(sel_id)
                        if sel_vec is not None:
                            try:
                                sel_vec = sel_vec.astype(np.float32)
                            except Exception:
                                sel_vec = np.array(sel_vec, dtype=np.float32)
                            selected_vectors[sel_id] = sel_vec
                    except Exception:
                        sel_vec = None
                if sel_vec is None:
                    continue
                # compute direct distance
                try:
                    d = float(np.linalg.norm(cand_vec - sel_vec))
                except Exception:
                    d = get_direct_distance(cand_vec, sel_vec)
                if d < threshold:
                    too_close = True
                    break
            if too_close:
                # Skip this candidate and advance pointer
                next_idx[bi] = idx + 1
                continue

            cand_anchor = float(bucket['dist_anchor'][idx])
            score = 0.7 * dist_prev + 0.3 * cand_anchor

            if score < lowest_score:
                lowest_score = score
                best_next_song = bucket['items'][idx]
                best_bucket = bi
                best_idx_in_bucket = idx

        # If we didn't find any candidate in the scanned buckets, stop
        if best_next_song is None:
            logger.warning(f"Radius walk: Stopped early at {len(playlist_ids)} songs; no candidates found in top {buckets_to_check} buckets.")
            break

        # Accept the best candidate found among the checked bucket-tops
        playlist_ids.append(best_next_song['item_id'])
        used_ids.add(best_next_song['item_id'])
        current_song_vector = best_next_song['vector'].astype(np.float32)
        current_song_id = best_next_song['item_id']

        # Advance the pointer in the bucket we took from so next time we consider the next item
        if best_bucket is not None:
            next_idx[best_bucket] = best_idx_in_bucket + 1

    logger.info(f"Radius walk: Walk complete. Collected {len(playlist_ids)} songs.")

    # --- Step 5: Finalize and return ---
    
    # We now have the re-ordered list of item_ids.
    # We need to fetch their final distances to the anchor (A) for display.
    # We can re-use the pre-calculated `dist_anchor` for efficiency.
    
    # Create a map of item_id -> dist_anchor for all candidates
    dist_anchor_map = {cand['item_id']: cand['dist_anchor'] for cand in candidate_data}
    
    final_results = []
    for item_id in playlist_ids:
        # Get the original distance to the anchor song
        dist_anchor = dist_anchor_map.get(item_id)
        
        if dist_anchor is not None:
            final_results.append({
                "item_id": item_id,
                "distance": dist_anchor
            })
        else:
            # Fallback in case something went wrong (should not happen)
            vec = _get_cached_vector(item_id)
            anchor_vec = _get_cached_vector(target_item_id)
            if vec is not None and anchor_vec is not None:
                final_results.append({
                    "item_id": item_id,
                    "distance": get_direct_distance(vec, anchor_vec)
                })

    # The list is already ordered by the walk. We just return it.
    # No final sort by distance is needed.
    return final_results

# --- END: RADIUS SIMILARITY RE-IMPLEMENTATION ---


def find_nearest_neighbors_by_id(target_item_id: str, n: int = 10, eliminate_duplicates: bool = False, mood_similarity: bool = True, radius_similarity: bool | None = None):
    """
    Finds the N nearest neighbors for a given item_id using the globally cached Voyager index.
    If mood_similarity is True, filters results by mood feature similarity (danceability, aggressive, happy, party, relaxed, sad).
    If radius_similarity is True, re-orders results based on the 70/30 weighted score.
    """
    if voyager_index is None or id_map is None or reverse_id_map is None:
        raise RuntimeError("Voyager index is not loaded in memory. It may be missing, empty, or the server failed to load it on startup.")

    from app_helper import get_db, get_score_data_by_ids
    db_conn = get_db()

    target_song_details_list = get_score_data_by_ids([target_item_id])
    if not target_song_details_list:
        logger.error(f"Could not retrieve details for the target song {target_item_id}. Aborting neighbor search.")
        return []
    target_song_details = target_song_details_list[0]


    target_voyager_id = reverse_id_map.get(target_item_id)
    if target_voyager_id is None:
        logger.warning(f"Target item_id '{target_item_id}' not found in the loaded Voyager index map.")
        return []

    try:
        query_vector = voyager_index.get_vector(target_voyager_id)
    except Exception as e:
        logger.error(f"Could not retrieve vector for Voyager ID {target_voyager_id} (item_id: {target_item_id}): {e}")
        return []

    # If caller didn't supply radius_similarity explicitly (None), use the configured default.
    if radius_similarity is None:
        radius_similarity = SIMILARITY_RADIUS_DEFAULT

    # --- Increase search size to get a large candidate pool ---
    # We need a *much larger* pool for the radius walk to be effective.
    # The multiplier should be consistent for both modes to satisfy the user's requirement.
    if radius_similarity:
        # Radius walk needs a large pool to choose from.
        # Let's use a base multiplier of 20, same as old code.
        base_multiplier = 20
        k_increase = max(4000, int(n * base_multiplier)) # Get a large pool, e.g. 4000+
        num_to_query = n + k_increase + 1
        logger.info(f"Radius similarity enabled. Fetching a large candidate pool of {num_to_query} songs.")
    elif mood_similarity:
        base_multiplier = 8 if eliminate_duplicates else 4
        k_increase = max(20, int(n * base_multiplier))
        num_to_query = n + k_increase + 1
    elif eliminate_duplicates:
        k_increase = max(5, int(n * 4))
        num_to_query = n + k_increase + 1
    else:
        k_increase = max(5, int(n * 0.20))
        num_to_query = n + k_increase + 1

    original_num_to_query = num_to_query
    if num_to_query > len(voyager_index):
        logger.warning(
            f"Voyager query request for {n} final items was expanded to {original_num_to_query} neighbors for processing. "
            f"This exceeds the total items in the index ({len(voyager_index)}). "
            f"Capping the actual query to {len(voyager_index)} items."
        )
        num_to_query = len(voyager_index)

    try:
        if num_to_query <= 1:
             logger.warning(f"Number of neighbors to query ({num_to_query}) is too small. Skipping query.")
             neighbor_voyager_ids, distances = [], []
        else:
             neighbor_voyager_ids, distances = voyager_index.query(query_vector, k=num_to_query)
    except voyager.RecallError as e:
        logger.warning(f"Voyager RecallError for item '{target_item_id}': {e}. "
                       "This is expected with small or sparse datasets. Returning empty list.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during Voyager query for item '{target_item_id}': {e}", exc_info=True)
        return []

    # --- Initial list of neighbors ---
    initial_results = []
    for voyager_id, dist in zip(neighbor_voyager_ids, distances):
        item_id = id_map.get(voyager_id)
        if item_id and item_id != target_item_id:
            initial_results.append({"item_id": item_id, "distance": float(dist)})

    # --- Divert logic for Radius Similarity ---
    if radius_similarity:
        # Mood similarity is explicitly *disabled* for radius walk, as it's a different discovery mode.
        logger.info(f"Starting Radius Similarity walk for {n} songs...")
        
        # 1. Get the pre-filtered and pre-calculated candidate pool
        candidate_data = _radius_walk_get_candidates(
            target_item_id=target_item_id,
            anchor_vector=query_vector,
            initial_results=initial_results,
            db_conn=db_conn,
            original_song_details=target_song_details,
            eliminate_duplicates=eliminate_duplicates # Pass this flag to apply artist cap pre-walk
        )
        
        # 2. Execute the bucketed greedy walk
        # The walk itself will return exactly n items (or fewer if the pool is too small)
        final_results = _execute_radius_walk(
            target_item_id=target_item_id,
            n=n,
            candidate_data=candidate_data,
            original_song_details=target_song_details
        )
        
        # 3. Return the results. They are already in the correct "walk" order.
        # No further filtering or sorting is needed.
        return final_results

    # --- Standard Logic (No Radius) ---
    else:
        # Apply standard filters
        
        # 1. Prepend original song for distance filtering
        original_song_for_filtering = {"item_id": target_item_id, "distance": 0.0}
        results_with_original = [original_song_for_filtering] + initial_results
        
        # 2. Filter by distance
        temp_filtered_results = _filter_by_distance(results_with_original, db_conn)
        
        # 3. Remove original song and filter by name/artist
        distance_filtered_results = [song for song in temp_filtered_results if song['item_id'] != target_item_id]
        unique_results_by_song = _deduplicate_and_filter_neighbors(distance_filtered_results, db_conn, target_song_details)
        
        # 4. Apply mood similarity filtering if requested
        if mood_similarity:
            logger.info(f"Mood similarity filtering requested for target_item_id: {target_item_id}")
            unique_results_by_song = _filter_by_mood_similarity(unique_results_by_song, target_item_id, db_conn)
        else:
            logger.info(f"No mood similarity filtering requested (mood_similarity={mood_similarity})")
        
        # 5. Apply artist cap (eliminate_duplicates)
        if eliminate_duplicates:
            item_ids_to_check = [r['item_id'] for r in unique_results_by_song]
            
            track_details_list = get_score_data_by_ids(item_ids_to_check)
            details_map = {d['item_id']: {'author': d['author']} for d in track_details_list}

            artist_counts = {}
            final_results = []
            for song in unique_results_by_song:
                song_id = song['item_id']
                author = details_map.get(song_id, {}).get('author')

                if not author:
                    logger.warning(f"Could not find author for item_id {song_id} during artist deduplication. Skipping.")
                    continue

                current_count = artist_counts.get(author, 0)
                if current_count < MAX_SONGS_PER_ARTIST:
                    final_results.append(song)
                    artist_counts[author] = current_count + 1
        else:
            final_results = unique_results_by_song

        # 6. Return the top N results, sorted by original distance
        return final_results[:n]

def find_nearest_neighbors_by_vector(query_vector: np.ndarray, n: int = 100, eliminate_duplicates: bool = False):
    """
    Finds the N nearest neighbors for a given query vector.
    """
    if voyager_index is None or id_map is None:
        raise RuntimeError("Voyager index is not loaded in memory.")

    from app_helper import get_db, get_score_data_by_ids
    db_conn = get_db()

    if eliminate_duplicates:
        num_to_query = n + int(n * 4)
    else:
        num_to_query = n + int(n * 0.2)

    original_num_to_query = num_to_query
    if num_to_query > len(voyager_index):
        logger.warning(
            f"Voyager query request for {n} final items was expanded to {original_num_to_query} neighbors for processing. "
            f"This exceeds the total items in the index ({len(voyager_index)}). "
            f"Capping the actual query to {len(voyager_index)} items."
        )
        num_to_query = len(voyager_index)

    try:
        if num_to_query <= 0:
            logger.warning("Number of neighbors to query is zero or less. Skipping query.")
            neighbor_voyager_ids, distances = [], []
        else:
            neighbor_voyager_ids, distances = voyager_index.query(query_vector, k=num_to_query)
    except voyager.RecallError as e:
        logger.warning(f"Voyager RecallError for synthetic vector query: {e}. "
                       "This is expected with small or sparse datasets. Returning empty list.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during Voyager query for synthetic vector: {e}", exc_info=True)
        return []

    initial_results = [
        {"item_id": id_map.get(voyager_id), "distance": float(dist)}
        for voyager_id, dist in zip(neighbor_voyager_ids, distances)
        if id_map.get(voyager_id) is not None
    ]

    distance_filtered_results = _filter_by_distance(initial_results, db_conn)

    # Fetch item details in parallel batches
    item_ids = [r['item_id'] for r in distance_filtered_results]
    item_details = {}
    
    def fetch_details_batch(id_batch):
        batch_details = {}
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT item_id, title, author FROM score WHERE item_id = ANY(%s)", (id_batch,))
            rows = cur.fetchall()
            for row in rows:
                batch_details[row['item_id']] = {'title': row['title'], 'author': row['author']}
        return batch_details
    
    # Split item_ids into batches for parallel DB queries
    id_batches = [item_ids[i:i + BATCH_SIZE_DB_OPS] for i in range(0, len(item_ids), BATCH_SIZE_DB_OPS)]
    
    if len(id_batches) > 1:
        # Use parallel DB queries for large datasets
        executor = _get_thread_pool()
        future_to_batch = {executor.submit(fetch_details_batch, batch): batch for batch in id_batches}
        
        for future in as_completed(future_to_batch):
            batch_details = future.result()
            item_details.update(batch_details)
    else:
        # Use single query for small datasets
        item_details = fetch_details_batch(item_ids)
            
    unique_songs_by_content = []
    added_songs_details = []
    for song in distance_filtered_results:
        current_details = item_details.get(song['item_id'])
        if not current_details:
            continue

        is_duplicate = any(_is_same_song(current_details['title'], current_details['author'], added['title'], added['author']) for added in added_songs_details)
        
        if not is_duplicate:
            unique_songs_by_content.append(song)
            added_songs_details.append(current_details)

    if eliminate_duplicates:
        artist_counts = {}
        final_results = []
        for song in unique_songs_by_content:
            author = item_details.get(song['item_id'], {}).get('author')
            if not author:
                continue

            current_count = artist_counts.get(author, 0)
            if current_count < MAX_SONGS_PER_ARTIST:
                final_results.append(song)
                artist_counts[author] = current_count + 1
    else:
        final_results = unique_songs_by_content

    return final_results[:n]

def get_item_id_by_title_and_artist(title: str, artist: str):
    """
    Finds the item_id for an exact title and artist match.
    """
    from app_helper import get_db
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        query = "SELECT item_id FROM score WHERE title = %s AND author = %s LIMIT 1"
        cur.execute(query, (title, artist))
        result = cur.fetchone()
        if result:
            return result['item_id']
        return None
    except Exception as e:
        logger.error(f"Error fetching item_id for '{title}' by '{artist}': {e}", exc_info=True)
        return None
    finally:
        cur.close()

def search_tracks_by_title_and_artist(title_query: str, artist_query: str, limit: int = 15):
    """
    Searches for tracks using partial title and artist names for autocomplete.
    """
    from app_helper import get_db
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    results = []
    try:
        query_parts = []
        params = []
        
        if title_query and not artist_query:
            query_parts.append("(title ILIKE %s OR author ILIKE %s)")
            params.extend([f"%{title_query}%", f"%{title_query}%"])
        else:
            if artist_query:
                query_parts.append("author ILIKE %s")
                params.append(f"%{artist_query}%")
                
            if title_query:
                query_parts.append("title ILIKE %s")
                params.append(f"%{title_query}%")

        if not query_parts:
            return []

        where_clause = " AND ".join(query_parts)
        
        query = f"""
            SELECT item_id, title, author 
            FROM score 
            WHERE {where_clause}
            ORDER BY author, title 
            LIMIT %s
        """
        params.append(limit)
        
        cur.execute(query, tuple(params))
        results = [dict(row) for row in cur.fetchall()]

    except Exception as e:
        logger.error(f"Error searching tracks with query '{title_query}', '{artist_query}': {e}", exc_info=True)
    finally:
        cur.close()
    
    return results


def create_playlist_from_ids(playlist_name: str, track_ids: list, user_creds: dict = None):
    """
    Creates a new playlist on the configured media server with the provided name and track IDs.
    """
    try:
        # Use the mediaserver dispatcher (imported at module top) to create the playlist.
        # This avoids importing app_external which may not export the helper.
        created_playlist = create_instant_playlist(playlist_name, track_ids, user_creds=user_creds)
        
        if not created_playlist:
            raise Exception("Playlist creation failed. The media server did not return a playlist object.")

        playlist_id = created_playlist.get('Id')

        if not playlist_id:
            raise Exception("Media server API response did not include a playlist ID.")

        return playlist_id

    except Exception as e:
        raise e

def cleanup_resources():
    """
    Cleanup function to shutdown thread pool and clear caches.
    Call this when shutting down the application.
    """
    logger.info("Cleaning up voyager_manager resources...")
    _shutdown_thread_pool()
    _get_cached_vector.cache_clear()
    logger.info("Voyager manager cleanup complete.")

