import os
import time
import logging
import numpy as np
from collections import OrderedDict
from psycopg2.extras import DictCursor

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import EMBEDDING_DIMENSION, INDEX_NAME, IVF_METRIC, MAX_SONGS_PER_ARTIST, DUPLICATE_DISTANCE_THRESHOLD_COSINE, DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN, DUPLICATE_DISTANCE_CHECK_LOOKBACK, MOOD_SIMILARITY_THRESHOLD, SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT, SIMILARITY_RADIUS_DEFAULT, MOOD_SIMILARITY_ENABLE, IVF_RESULT_CACHE_SECONDS, IVF_RESULT_CACHE_MAX

logger = logging.getLogger(__name__)

# Optional instrumentation: enable with RADIUS_INSTRUMENTATION=True in env
INSTRUMENT_BUCKET_SKIPS = os.environ.get("RADIUS_INSTRUMENTATION", "False").lower() == 'true'

# --- Global cache for the loaded index ---
ivf_index = None
id_map = None # {vec_int_id: item_id_str}
reverse_id_map = None # {item_id_str: vec_int_id}


class _ResultCache:
    """Thread-safe TTL+LRU cache for fully-computed query results.

    The cell cache only removes Postgres reads; the per-request distance, dedup
    and metadata work still runs every time, so two identical similarity or
    max-distance queries take the same wall time. This caches the final result so
    repeats are instant. Entries expire after ttl seconds and are dropped on index
    rebuild via clear(). ttl <= 0 disables it.
    """

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


_neighbor_result_cache = _ResultCache(IVF_RESULT_CACHE_SECONDS, IVF_RESULT_CACHE_MAX)
_max_distance_cache = _ResultCache(IVF_RESULT_CACHE_SECONDS, IVF_RESULT_CACHE_MAX)


# --- Thread pool for parallel operations ---
_thread_pool = None
_thread_pool_lock = threading.Lock()

# --- Configuration for parallel processing ---
MAX_WORKER_THREADS = max(1, (os.cpu_count() or 1) - 1)  # Use cpu_count - 1, minimum 1
BATCH_SIZE_VECTOR_OPS = 50  # Process vectors in batches
BATCH_SIZE_DB_OPS = 100     # Process database operations in batches

def _get_thread_pool():
    """Get or create the global thread pool for parallel operations."""
    global _thread_pool
    with _thread_pool_lock:
        if _thread_pool is None:
            _thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS, thread_name_prefix="ivf")
        return _thread_pool

def _shutdown_thread_pool():
    """Shutdown the global thread pool."""
    global _thread_pool
    with _thread_pool_lock:
        if _thread_pool is not None:
            _thread_pool.shutdown(wait=True)
            _thread_pool = None

def _get_cached_vector(item_id: str) -> np.ndarray | None:
    """Return the stored vector for ``item_id`` from the loaded IVF index.

    The IVF backend keeps its own byte-bounded per-request cell cache, so no
    extra process-global cache is needed here.
    """
    if ivf_index is None or reverse_id_map is None:
        return None
    vec_id = reverse_id_map.get(item_id)
    if vec_id is None:
        return None
    try:
        return ivf_index.get_vector(vec_id)
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
    """Public helper that picks the metric according to IVF_METRIC."""
    if IVF_METRIC == 'angular':
        return _get_direct_cosine_distance(v1, v2)
    return _get_direct_euclidean_distance(v1, v2)


def load_ivf_index_for_querying(force_reload=False):
    """Load the disk-paged IVF audio index into the global in-memory cache.

    Kept under this name because the app startup and reload paths import it directly.
    """
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
        loaded = load_paged_ivf_index(conn, INDEX_NAME, EMBEDDING_DIMENSION, IVF_METRIC, label="audio")
    except Exception as e:
        logger.error("Failed to load audio IVF index: %s", e, exc_info=True)
        ivf_index, id_map, reverse_id_map = None, None, None
        return
    if loaded is None:
        logger.warning("Audio IVF index not found in the database. Cache will be empty (run analysis to build it).")
        ivf_index, id_map, reverse_id_map = None, None, None
        return
    ivf_index, id_map, reverse_id_map = loaded
    logger.info("Audio IVF index with %d items loaded successfully into memory.", len(id_map))


def build_and_store_ivf_index(db_conn=None):
    """Build the disk-paged IVF audio index from song embeddings and store it.

    Accepts an optional db_conn. If None, acquires one via app_helper.get_db().
    """
    if db_conn is None:
        try:
            from app_helper import get_db
            db_conn = get_db()
        except Exception:
            logger.error("build_and_store_ivf_index: no db_conn provided and get_db() failed.")
            return

    from .index_build_helpers import stream_embeddings_to_buffer
    from .paged_ivf import build_and_store_paged_ivf
    logger.info("Starting to build and store audio IVF index (disk-paged)...")
    try:
        buf, item_ids = stream_embeddings_to_buffer(
            table="embedding", column="embedding", dim=EMBEDDING_DIMENSION,
            where_clause="embedding IS NOT NULL",
        )
        if buf.shape[0] == 0:
            logger.warning("No valid audio embeddings found for IVF index build. Aborting.")
            return
        if build_and_store_paged_ivf(db_conn, INDEX_NAME, buf, item_ids, EMBEDDING_DIMENSION, IVF_METRIC):
            db_conn.commit()
            logger.info("Audio IVF index build and database storage complete.")
    except Exception as e:
        logger.error("An error occurred during audio IVF index build: %s", e, exc_info=True)
        try:
            db_conn.rollback()
        except Exception:
            pass


def get_vector_by_id(item_id: str) -> np.ndarray | None:
    """
    Retrieves the embedding vector for a given item_id from the loaded IVF index.
    Uses caching for better performance.
    """
    return _get_cached_vector(item_id)


def get_vectors_by_ids(item_ids) -> dict:
    """
    Batch-fetch embedding vectors for many item_ids in parallel.

    Returns a dict mapping str(item_id) -> np.ndarray (or None if missing).
    Repeat IDs collapse to a single fetch (LRU cache hit). The thread pool
    is the existing module-level _get_thread_pool() used elsewhere.
    """
    if not item_ids:
        return {}

    unique = []
    seen = set()
    for raw in item_ids:
        sid = str(raw)
        if sid not in seen:
            seen.add(sid)
            unique.append(sid)

    if voyager_index is None or reverse_id_map is None:
        return {sid: None for sid in unique}

    out = {}
    if len(unique) <= 4:
        for sid in unique:
            out[sid] = _get_cached_vector(sid)
        return out

    executor = _get_thread_pool()
    futures = {executor.submit(_get_cached_vector, sid): sid for sid in unique}
    for fut in as_completed(futures):
        sid = futures[fut]
        try:
            out[sid] = fut.result()
        except Exception:
            out[sid] = None
    return out


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
        # Build an effective comparison window that includes the provided lookback
        # plus any songs already accepted in this batch. This avoids the case where
        # two near-duplicate songs fall into the same batch and both slip through
        # because they weren't compared to each other.
        combined_recent = list(lookback_songs) + list(batch_results)
        for recent_song in combined_recent:
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

    threshold = DUPLICATE_DISTANCE_THRESHOLD_COSINE if IVF_METRIC == 'angular' else DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN
    metric_name = 'Angular' if IVF_METRIC == 'angular' else 'Euclidean'
    
    filtered_songs = []

    # For small datasets, use sequential processing. The larger-dataset branch
    # below runs its vector distance work on this same request thread (only the
    # metadata fetch above uses the pool), so the IVF backend takes the identical
    # code path as ivf and produces identical results.
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
            cur.execute("SELECT item_id, title, author, album, album_artist FROM score WHERE item_id = ANY(%s)", (id_batch,))
            rows = cur.fetchall()
            for row in rows:
                batch_details[row['item_id']] = {'title': row['title'], 'author': row['author'], 'album': row.get('album'), 'album_artist': row.get('album_artist')}
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
    eliminate_duplicates: bool,
    mood_similarity: bool | None = None,
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
    
    # Follow the same pre-filter order used by the non-radius path:
    # 1) Distance-based de-dup (prepend original)
    # 2) Name/artist dedupe
    # 3) Mood similarity filter
    # NOTE: Artist-cap is intentionally NOT applied here and will be enforced
    #       during the walk itself (in-walk) to preserve selection dynamics.

    # Early exit if no candidates
    if not initial_results:
        return []

    # 1) Distance-based filtering: prepend original so the filter can compare against the anchor
    try:
        original_for_filter = {"item_id": target_item_id, "distance": 0.0}
        results_with_original = [original_for_filter] + initial_results
        temp_filtered = _filter_by_distance(results_with_original, db_conn)
        # Remove the original we prepended
        distance_filtered_results = [s for s in temp_filtered if s['item_id'] != target_item_id]
        logger.info(f"Radius walk: distance-based filtering reduced candidates {len(initial_results)} -> {len(distance_filtered_results)}")
    except Exception:
        logger.exception("Radius walk: distance-based pre-filter failed, continuing with original candidate set.")
        distance_filtered_results = initial_results

    # 2) Name/artist deduplication (remove exact duplicates and the original song)
    try:
        unique_results_by_song = _deduplicate_and_filter_neighbors(distance_filtered_results, db_conn, original_song_details)
        logger.info(f"Radius walk: name-based dedupe reduced candidates to {len(unique_results_by_song)}")
    except Exception:
        logger.exception("Radius walk: name-based dedupe failed, continuing without it.")
        unique_results_by_song = distance_filtered_results

    # 3) Mood similarity filtering: only apply if globally enabled via config.
    try:
        # Determine effective mood filtering: caller preference takes precedence.
        effective_mood = MOOD_SIMILARITY_ENABLE if mood_similarity is None else mood_similarity
        if effective_mood:
            before_mood = len(unique_results_by_song)
            unique_results_by_song = _filter_by_mood_similarity(unique_results_by_song, target_item_id, db_conn)
            after_mood = len(unique_results_by_song)
            logger.info(f"Radius walk: mood-based filtering reduced candidates {before_mood} -> {after_mood}")
        else:
            logger.debug("Radius walk: mood-based pre-filter disabled by caller/config. Skipping.")
    except Exception:
        logger.exception("Radius walk: mood-based pre-filter failed, continuing without it.")
    
    # 3. Fetch item details for remaining candidates and pre-calculate/cache data for the walk
    candidate_data = []
    if unique_results_by_song:
        item_ids_to_fetch = [r['item_id'] for r in unique_results_by_song]
        # Fetch details in batch (uses app_helper get_score_data_by_ids)
        try:
            track_details_list = get_score_data_by_ids(item_ids_to_fetch)
            details_map = {d['item_id']: {'title': d.get('title'), 'author': d.get('author'), 'album': d.get('album'), 'album_artist': d.get('album_artist')} for d in track_details_list}
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
    n: int,
    candidate_data: list,
    eliminate_duplicates: bool = False
) -> list:
    """
    Executes the bucketed greedy walk based on the pre-filtered and pre-calculated
    candidate data.  Delegates to the shared ``radius_walk_helper`` module.
    """
    from .radius_walk_helper import execute_radius_walk as _shared_walk

    return _shared_walk(
        candidate_data=candidate_data,
        n=n,
        eliminate_duplicates=eliminate_duplicates,
        max_songs_per_artist=MAX_SONGS_PER_ARTIST,
        get_distance_fn=get_direct_distance,
    )

# --- END: RADIUS SIMILARITY RE-IMPLEMENTATION ---


def find_nearest_neighbors_by_id(target_item_id: str, n: int = 10, eliminate_duplicates: bool | None = None, mood_similarity: bool | None = None, radius_similarity: bool | None = None):
    """
    Finds the N nearest neighbors for a given item_id using the globally cached IVF index.
    If mood_similarity is True, filters results by mood feature similarity (danceability, aggressive, happy, party, relaxed, sad).
    If radius_similarity is True, re-orders results based on the 70/30 weighted score.
    """
    if ivf_index is None or id_map is None or reverse_id_map is None:
        raise RuntimeError("IVF index is not loaded in memory. It may be missing, empty, or the server failed to load it on startup.")

    eff_radius = SIMILARITY_RADIUS_DEFAULT if radius_similarity is None else radius_similarity
    eff_eliminate = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT if eliminate_duplicates is None else eliminate_duplicates
    eff_mood = MOOD_SIMILARITY_ENABLE if mood_similarity is None else mood_similarity
    _result_key = (target_item_id, int(n), bool(eff_eliminate), bool(eff_mood), bool(eff_radius))
    _cached = _neighbor_result_cache.get(_result_key)
    if _cached is not None:
        return [dict(r) for r in _cached]

    if ivf_index is not None:
        ivf_index.begin_request()

    from app_helper import get_db, get_score_data_by_ids
    db_conn = get_db()

    target_song_details_list = get_score_data_by_ids([target_item_id])
    if not target_song_details_list:
        logger.error(f"Could not retrieve details for the target song {target_item_id}. Aborting neighbor search.")
        return []
    target_song_details = target_song_details_list[0]


    target_vec_id = reverse_id_map.get(target_item_id)
    if target_vec_id is None:
        logger.warning(f"Target item_id '{target_item_id}' not found in the loaded IVF index map.")
        return []

    try:
        query_vector = ivf_index.get_vector(target_vec_id)
    except Exception as e:
        logger.error(f"Could not retrieve vector for IVF ID {target_vec_id} (item_id: {target_item_id}): {e}")
        return []


    # If caller didn't supply radius_similarity explicitly (None), use the configured default.
    if radius_similarity is None:
        radius_similarity = SIMILARITY_RADIUS_DEFAULT

    # If caller didn't supply eliminate_duplicates explicitly (None), use configured default
    if eliminate_duplicates is None:
        eliminate_duplicates = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT

    # If caller didn't supply mood_similarity explicitly (None), DO NOT force it here.
    # We will treat None as "use the config default". Caller-provided True must
    # override the config; caller-provided False disables it.

    # --- Increase search size to get a large candidate pool ---
    # We need a *much larger* pool for the radius walk to be effective.
    # The multiplier should be consistent for both modes to satisfy the user's requirement.
    if radius_similarity or eliminate_duplicates:
        # Radius walk needs a large pool to choose from.
        # Let's use a base multiplier of 20, same as old code.
        base_multiplier = 3
        k_increase = max(20, int(n * base_multiplier)) # Get a large pool, e.g. 4000+
        num_to_query = n + k_increase + 1
        logger.info(f"Radius similarity enabled. Fetching a large candidate pool of {num_to_query} songs.")
    else:
        k_increase = max(3, int(n * 0.20))
        num_to_query = n + k_increase + 1
    if mood_similarity:
        base_multiplier = 8 if eliminate_duplicates else 4
        k_increase = max(20, int(n * base_multiplier))
        num_to_query = n + k_increase + 1

    original_num_to_query = num_to_query
    if num_to_query > len(ivf_index):
        logger.warning(
            f"IVF query request for {n} final items was expanded to {original_num_to_query} neighbors for processing. "
            f"This exceeds the total items in the index ({len(ivf_index)}). "
            f"Capping the actual query to {len(ivf_index)} items."
        )
        num_to_query = len(ivf_index)

    try:
        if num_to_query <= 1:
             logger.warning(f"Number of neighbors to query ({num_to_query}) is too small. Skipping query.")
             neighbor_vec_ids, distances = [], []
        else:
             neighbor_vec_ids, distances = ivf_index.query(query_vector, k=num_to_query)
    except Exception as e:
        logger.error(f"An unexpected error occurred during IVF query for item '{target_item_id}': {e}", exc_info=True)
        return []

    # --- Initial list of neighbors ---
    initial_results = []
    for vec_id, dist in zip(neighbor_vec_ids, distances):
        item_id = id_map.get(vec_id)
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
            eliminate_duplicates=eliminate_duplicates, # Pass this flag to apply artist cap pre-walk
            mood_similarity=mood_similarity
        )
        
        # 2. Execute the bucketed greedy walk
        # The walk itself will return exactly n items (or fewer if the pool is too small)
        final_results = _execute_radius_walk(
            n=n,
            candidate_data=candidate_data,
            eliminate_duplicates=eliminate_duplicates
        )
        
        # 3. Return the results. They are already in the correct "walk" order.
        # No further filtering or sorting is needed.
        _neighbor_result_cache.put(_result_key, final_results)
        return [dict(r) for r in final_results]

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
        
        # 4. Apply mood similarity filtering: caller preference overrides config.
        effective_mood_nonradius = MOOD_SIMILARITY_ENABLE if mood_similarity is None else mood_similarity
        if effective_mood_nonradius:
            logger.info(f"Mood similarity filtering requested/enabled for target_item_id: {target_item_id}")
            unique_results_by_song = _filter_by_mood_similarity(unique_results_by_song, target_item_id, db_conn)
        else:
            logger.info(f"Mood filtering skipped (mood_similarity={mood_similarity}, MOOD_SIMILARITY_ENABLE={MOOD_SIMILARITY_ENABLE})")
        
        # 5. Apply artist cap (eliminate_duplicates)
        if eliminate_duplicates:
            # If MAX_SONGS_PER_ARTIST <= 0, treat as disabled and skip cap enforcement
            if MAX_SONGS_PER_ARTIST is None or MAX_SONGS_PER_ARTIST <= 0:
                final_results = unique_results_by_song
            else:
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
        final_results = final_results[:n]
        _neighbor_result_cache.put(_result_key, final_results)
        return [dict(r) for r in final_results]

def find_nearest_neighbors_by_vector(query_vector: np.ndarray, n: int = 100, eliminate_duplicates: bool | None = None):
    """
    Finds the N nearest neighbors for a given query vector.
    """
    if ivf_index is None or id_map is None:
        raise RuntimeError("IVF index is not loaded in memory.")

    if ivf_index is not None:
        ivf_index.begin_request()

    from app_helper import get_db
    db_conn = get_db()

    # If caller didn't supply eliminate_duplicates explicitly (None), use configured default
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
    except Exception as e:
        logger.error(f"An unexpected error occurred during IVF query for synthetic vector: {e}", exc_info=True)
        return []

    initial_results = [
        {"item_id": id_map.get(vec_id), "distance": float(dist)}
        for vec_id, dist in zip(neighbor_vec_ids, distances)
        if id_map.get(vec_id) is not None
    ]

    distance_filtered_results = _filter_by_distance(initial_results, db_conn)

    # Fetch item details in parallel batches
    item_ids = [r['item_id'] for r in distance_filtered_results]
    item_details = {}
    
    def fetch_details_batch(id_batch):
        batch_details = {}
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT item_id, title, author, album, album_artist FROM score WHERE item_id = ANY(%s)", (id_batch,))
            rows = cur.fetchall()
            for row in rows:
                batch_details[row['item_id']] = {'title': row['title'], 'author': row['author'], 'album': row.get('album'), 'album_artist': row.get('album_artist')}
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
        # If MAX_SONGS_PER_ARTIST <= 0, treat as disabled and skip cap enforcement
        if MAX_SONGS_PER_ARTIST is None or MAX_SONGS_PER_ARTIST <= 0:
            final_results = unique_songs_by_content
        else:
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


def get_max_distance_for_id(target_item_id: str):
    """
    Returns the maximum distance from the given item to any other item in the loaded ivf index.

    This is the APPROXIMATE farthest neighbour computed by
    ``PagedIvfIndex.get_max_distance`` (it ranks centroids and scans only the
    ``IVF_MAX_DISTANCE_NPROBE`` farthest cells), used as the UI "max distance"
    reference value. Returns a dict: { 'max_distance': float, 'farthest_item_id': str | None }
    Raises RuntimeError if the index is not loaded.
    """
    if ivf_index is None or id_map is None or reverse_id_map is None:
        raise RuntimeError("IVF index is not loaded in memory. It may be missing, empty, or the server failed to load it on startup.")

    target_vec_id = reverse_id_map.get(target_item_id)
    if target_vec_id is None:
        return None

    _cached = _max_distance_cache.get(target_item_id)
    if _cached is not None:
        return dict(_cached)

    ivf_index.begin_request()
    try:
        max_d, far_vec_id = ivf_index.get_max_distance(target_vec_id)
    except Exception as e:
        logger.error(f"Error computing IVF max distance for {target_item_id}: {e}", exc_info=True)
        return None
    if max_d is None:
        return None
    far_item_id = id_map.get(far_vec_id) if far_vec_id is not None else None
    result = {'max_distance': float(max_d), 'farthest_item_id': far_item_id}
    _max_distance_cache.put(target_item_id, result)
    return dict(result)

def get_item_id_by_title_and_artist(title: str, artist: str):
    """
    Finds the item_id for a title and artist match.
    Uses fuzzy matching (case-insensitive partial match) to handle variations.
    """
    from app_helper import get_db
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    try:
        # First try exact match (case-insensitive)
        query = "SELECT item_id FROM score WHERE LOWER(title) = LOWER(%s) AND LOWER(author) = LOWER(%s) LIMIT 1"
        cur.execute(query, (title, artist))
        result = cur.fetchone()
        if result:
            return result['item_id']
        
        # If no exact match, try fuzzy match (partial match on both title and artist)
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
            logger.info(f"Fuzzy matched '{title}' by '{artist}' to '{result['title']}' by '{result['author']}'")
            return result['item_id']
        
        return None
    except Exception as e:
        logger.error(f"Error fetching item_id for '{title}' by '{artist}': {e}", exc_info=True)
        return None
    finally:
        cur.close()

def search_tracks_unified(search_query: str, limit: int = 20, offset: int = 0,
                          item_id_filter: set = None):
    """
    Deterministic substring search over title, author and album.

    - Accent and case insensitive
    - Each token must match title, author or album
    - Ranking priority: title > author > album
    - item_id_filter: optional set of item_ids to restrict results to
    """

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

        # Filtering
        for token in tokens:
            like_pattern = f"%{token}%"
            where_clauses.append("search_u LIKE unaccent(%s)")
            params.append(like_pattern)

        # Weighted ordering
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

        # Optionally restrict to a specific set of item_ids (e.g. SemGrove index)
        # IMPORTANT: id_filter params must be inserted *after* the WHERE token params
        # but *before* the score/ORDER BY params so they match the SQL position.
        id_filter_sql = ""
        id_filter_params: list = []
        if item_id_filter:
            id_placeholders = ",".join(["%s"] * len(item_id_filter))
            id_filter_sql = f" AND item_id IN ({id_placeholders})"
            id_filter_params = list(item_id_filter)

        # Final param list order must mirror SQL: WHERE tokens → WHERE id filter → ORDER BY scores → LIMIT/OFFSET
        all_params = params[:len(tokens)] + id_filter_params + params[len(tokens):]

        query = f"""
            SELECT item_id, title, author, album, album_artist
            FROM score
            WHERE {where_sql}{id_filter_sql}
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

    except Exception as e:
        logger.error(
            f"Error searching tracks with query '{search_query}': {e}",
            exc_info=True
        )
    finally:
        cur.close()

    return results

def create_playlist_from_ids(playlist_name: str, track_ids: list, user_creds: dict = None):
    """
    Creates a new playlist on the configured media server with the provided name and track IDs.
    """
    try:
        from .mediaserver import create_instant_playlist
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
    Cleanup function to shutdown the thread pool.
    Call this when shutting down the application.
    """
    logger.info("Cleaning up similarity manager resources...")
    _shutdown_thread_pool()
    logger.info("Similarity manager cleanup complete.")

