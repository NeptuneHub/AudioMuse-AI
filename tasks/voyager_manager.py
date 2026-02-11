import os
import json
import logging
import tempfile
import numpy as np
import psycopg2 # type: ignore
from psycopg2.extras import DictCursor
import io 

# Attempt to import Voyager (may be missing on non-AVX systems)
try:
    import voyager  # type: ignore
    VOYAGER_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("Voyager library not found. HNSW-based features will be disabled (non-AVX CPU detected).")
    VOYAGER_AVAILABLE = False
except Exception as e:
    logging.getLogger(__name__).error(f"Error importing Voyager: {e}")
    VOYAGER_AVAILABLE = False
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
    , SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT, SIMILARITY_RADIUS_DEFAULT,
    MOOD_SIMILARITY_ENABLE
)
# Import from other project modules
from .mediaserver import create_instant_playlist

logger = logging.getLogger(__name__)

# Optional instrumentation: enable with RADIUS_INSTRUMENTATION=True in env
INSTRUMENT_BUCKET_SKIPS = os.environ.get("RADIUS_INSTRUMENTATION", "False").lower() == 'true'

# --- Global cache for the loaded Voyager index ---
voyager_index = None
id_map = None # {voyager_int_id: item_id_str}
reverse_id_map = None # {item_id_str: voyager_int_id}

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
def build_and_store_voyager_index(db_conn=None):
    """
    Fetches all song embeddings, builds a new Voyager index, and stores it
    atomically in the 'voyager_index_data' table in PostgreSQL.

    Accepts an optional db_conn (psycopg2 connection). If None, the function
    will acquire a connection via app_helper.get_db().
    """
    if not VOYAGER_AVAILABLE:
        logger.warning("Voyager not available - skipping index build")
        return

    # Acquire DB connection if not provided
    if db_conn is None:
        try:
            from app_helper import get_db
            db_conn = get_db()
        except Exception:
            logger.error("build_and_store_voyager_index: no db_conn provided and get_db() failed.")
            return

    logger.info("Starting to build and store Voyager index...")

    # Map the string metric from config to the voyager.Space enum
    metric_str = VOYAGER_METRIC.lower()
    if metric_str == 'angular':
        space = voyager.Space.Cosine
    elif metric_str == 'euclidean':
        space = voyager.Space.Euclidean
    elif metric_str == 'dot':
        space = voyager.Space.InnerProduct
    else:
        logger.warning(f"Unknown Voyager metric '{VOYAGER_METRIC}'. Defaulting to Cosine.")
        space = voyager.Space.Cosine

    cur = db_conn.cursor()
    try:
        logger.info("Fetching all embeddings from the database...")
        cur.execute("SELECT item_id, embedding FROM embedding")
        all_embeddings = cur.fetchall()

        if not all_embeddings:
            logger.warning("No embeddings found in DB. Voyager index will not be built.")
            return

        logger.info(f"Found {len(all_embeddings)} embeddings to index.")

        voyager_index_builder = voyager.Index(
            space=space,
            num_dimensions=EMBEDDING_DIMENSION,
            M=VOYAGER_M,
            ef_construction=VOYAGER_EF_CONSTRUCTION
        )

        local_id_map = {}
        voyager_item_index = 0
        vectors_to_add = []
        ids_to_add = []

        for item_id, embedding_blob in all_embeddings:
            if embedding_blob is None:
                logger.warning(f"Skipping item_id {item_id}: embedding data is NULL.")
                continue

            embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)

            if embedding_vector.shape[0] != EMBEDDING_DIMENSION:
                logger.warning(f"Skipping item_id {item_id}: embedding dimension mismatch. "
                               f"Expected {EMBEDDING_DIMENSION}, got {embedding_vector.shape[0]}.")
                continue

            vectors_to_add.append(embedding_vector)
            ids_to_add.append(voyager_item_index)
            local_id_map[voyager_item_index] = item_id
            voyager_item_index += 1

        if not vectors_to_add:
            logger.warning("No valid embeddings were found to add to the Voyager index. Aborting build process.")
            return

        logger.info(f"Adding {len(vectors_to_add)} items to the index...")
        voyager_index_builder.add_items(np.array(vectors_to_add), ids=np.array(ids_to_add))

        logger.info(f"Building index with {len(vectors_to_add)} items...")

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".voyager") as tmp:
                temp_file_path = tmp.name

            voyager_index_builder.save(temp_file_path)

            with open(temp_file_path, 'rb') as f:
                index_binary_data = f.read()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        logger.info(f"Voyager index binary data size to be stored: {len(index_binary_data)} bytes.")

        if not index_binary_data:
            logger.error("CRITICAL: Generated Voyager index file is empty. Aborting database storage.")
            return

        id_map_json = json.dumps(local_id_map)

        logger.info(f"Storing Voyager index '{INDEX_NAME}' in the database...")
        upsert_query = """
            INSERT INTO voyager_index_data (index_name, index_data, id_map_json, embedding_dimension, created_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (index_name) DO UPDATE SET
                index_data = EXCLUDED.index_data,
                id_map_json = EXCLUDED.id_map_json,
                embedding_dimension = EXCLUDED.embedding_dimension,
                created_at = CURRENT_TIMESTAMP;
        """
        # Use psycopg2.Binary to ensure proper bytea handling
        cur.execute(upsert_query, (INDEX_NAME, psycopg2.Binary(index_binary_data), id_map_json, EMBEDDING_DIMENSION))
        db_conn.commit()
        logger.info("Voyager index build and database storage complete.")

    except Exception as e:
        logger.error("An error occurred during Voyager index build: %s", e, exc_info=True)
        try:
            db_conn.rollback()
        except Exception:
            pass
    finally:
        try:
            cur.close()
        except Exception:
            pass

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
    target_item_id: str,
    n: int,
    candidate_data: list,
    original_song_details: dict | None = None,
    eliminate_duplicates: bool = False
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

    # Track per-artist counts during the walk (enforced only if eliminate_duplicates=True)
    artist_counts = {}
    try:
        # Initialize with the first selected song's artist
        if first_author:
            artist_counts[first_author] = artist_counts.get(first_author, 0) + 1
    except Exception:
        pass

    # Track how many different buckets each artist has been included in.
    # This enforces the "only two different bucket subpaths can have the same artist"
    # rule until the global per-artist cap (MAX_SONGS_PER_ARTIST) is reached.
    artist_bucket_counts = {}

    # --- Step 3: Iteratively build the playlist (Greedy, low-cost scanning) ---

    # Optimization: instead of computing distances for many candidates each iteration,
    # we maintain a per-bucket pointer and evaluate only the next un-used candidate
    # in the nearest BUCKETS_TO_SCAN buckets each loop. This drastically reduces the
    # number of direct distance calculations while keeping a greedy selection.

    num_buckets = len(buckets)

    # We'll process buckets sequentially (nearest -> farthest), building a
    # greedy subpath inside each bucket and appending it to the global playlist.
    # This satisfies the requirement to create tight per-bucket paths and then
    # stitch them together, while enforcing artist cap and duplicate avoidance
    # across the entire walk.
    buckets_to_check = min(num_buckets, BUCKETS_TO_SCAN)

    # Helper: perform a greedy walk constrained to a single bucket's items.
    def _walk_single_bucket(bucket_index, start_item_id=None):
        bucket = buckets[bucket_index]
        items = bucket['items']
        if not items:
            return []

        # Build arrays for candidate vectors and metadata
        cand_ids = [c['item_id'] for c in items]
        cand_vecs = bucket['vecs']
        cand_anchor = bucket['dist_anchor']

        # remaining mask (True = available)
        remaining = [True] * len(cand_ids)
        # Per-bucket artist set to avoid more than one song per artist within the same bucket
        bucket_artist_set = set()

        subpath = []

        # If a start_item_id is provided and present in this bucket, use it as the first
        # accepted song (if not already used); otherwise pick the nearest available.
        def _find_start_index():
            if start_item_id:
                try:
                    si = cand_ids.index(start_item_id)
                    if remaining[si] and cand_ids[si] not in used_ids:
                        return si
                except ValueError:
                    pass
            # fallback: first available candidate in bucket order
            for i, cid in enumerate(cand_ids):
                if remaining[i] and cid not in used_ids:
                    return i
            return None

        cur_idx = _find_start_index()
        if cur_idx is None:
            return []

        # Accept start if it passes name/artist/artist-cap checks
        def _accept_at_index(i):
            # In-walk name/title and distance duplicates are intentionally NOT applied here.
            # Those duplicates should have been filtered in the initial candidate prep.
            # Only enforce artist cap (eliminate_duplicates) and ensure the id is not already used.
            cid = cand_ids[i]
            if cid in used_ids:
                return False
            # Per-bucket: avoid more than one song per artist inside this bucket
            # This restriction is considered part of the artist-cap behavior.
            # Treat MAX_SONGS_PER_ARTIST <= 0 as DISABLED (no per-bucket restriction).
            try:
                if eliminate_duplicates and MAX_SONGS_PER_ARTIST is not None and MAX_SONGS_PER_ARTIST > 0:
                    author = items[i].get('author')
                    if author and author in bucket_artist_set:
                        return False
            except Exception:
                pass
            # Enforce global artist cap. Treat MAX_SONGS_PER_ARTIST <= 0 as DISABLED.
            if eliminate_duplicates and MAX_SONGS_PER_ARTIST is not None and MAX_SONGS_PER_ARTIST > 0:
                author = items[i].get('author')
                if author:
                    # If this artist has already appeared in two different buckets
                    # and still hasn't hit the global cap, don't allow them in a third bucket.
                    if artist_bucket_counts.get(author, 0) >= 2 and artist_counts.get(author, 0) < MAX_SONGS_PER_ARTIST:
                        return False
                    if artist_counts.get(author, 0) >= MAX_SONGS_PER_ARTIST:
                        return False
            return True

        # Try to accept start index, otherwise mark it unavailable and find next
        if _accept_at_index(cur_idx):
            remaining[cur_idx] = False
            cid = cand_ids[cur_idx]
            subpath.append(items[cur_idx])
            used_ids.add(cid)
            # Only add if we still need more songs
            if len(playlist_ids) < n:
                playlist_ids.append(cid)
            try:
                v = items[cur_idx]['vector'].astype(np.float32)
            except Exception:
                v = np.array(items[cur_idx]['vector'], dtype=np.float32)
            selected_vectors[cid] = v
            if eliminate_duplicates:
                a = items[cur_idx].get('author')
                if a:
                    # update global artist count
                    artist_counts[a] = artist_counts.get(a, 0) + 1
                    # register this artist as present in this bucket (only increment bucket count once)
                    if a not in bucket_artist_set:
                        bucket_artist_set.add(a)
                        artist_bucket_counts[a] = artist_bucket_counts.get(a, 0) + 1
        else:
            remaining[cur_idx] = False

        # Greedy selection inside the bucket: repeatedly pick best candidate among remaining
        while True:
            # build list of candidate indices still available
            avail_idxs = [i for i, r in enumerate(remaining) if r and cand_ids[i] not in used_ids]
            if not avail_idxs:
                break

            # compute distances from current song vector to each available candidate
            try:
                cur_vec = selected_vectors[playlist_ids[-1]]
            except Exception:
                break

            best_i = None
            best_score = float('inf')
            best_d = None

            for i in avail_idxs:
                meta = items[i]
                cid = cand_ids[i]
                # Skip if already used
                if cid in used_ids:
                    continue
                # Per-bucket: avoid more than one song per artist inside this bucket.
                # Treat MAX_SONGS_PER_ARTIST <= 0 as DISABLED (no per-bucket restriction).
                try:
                    if eliminate_duplicates and MAX_SONGS_PER_ARTIST is not None and MAX_SONGS_PER_ARTIST > 0:
                        auth = meta.get('author')
                        if auth and auth in bucket_artist_set:
                            if INSTRUMENT_BUCKET_SKIPS:
                                logger.debug(f"Bucket {bucket_index}: skipping idx={i} bucket-artist-limit {auth}")
                            continue
                except Exception:
                    pass
                # Global artist cap check (only enforce if positive cap configured)
                if eliminate_duplicates and MAX_SONGS_PER_ARTIST is not None and MAX_SONGS_PER_ARTIST > 0:
                    auth = meta.get('author')
                    if auth:
                        # if artist already occupies two different buckets and hasn't hit the cap, skip
                        if artist_bucket_counts.get(auth, 0) >= 2 and artist_counts.get(auth, 0) < MAX_SONGS_PER_ARTIST:
                            if INSTRUMENT_BUCKET_SKIPS:
                                logger.debug(f"Bucket {bucket_index}: skipping idx={i} bucket-count-limit {auth}")
                            continue
                        if artist_counts.get(auth, 0) >= MAX_SONGS_PER_ARTIST:
                            if INSTRUMENT_BUCKET_SKIPS:
                                logger.debug(f"Bucket {bucket_index}: skipping idx={i} artist-cap {auth}")
                            continue

                # compute dist_prev (distance to current song) using configured metric
                try:
                    dist_prev = get_direct_distance(cand_vecs[i], cur_vec)
                except Exception:
                    try:
                        dist_prev = float(np.linalg.norm(cand_vecs[i] - cur_vec))
                    except Exception:
                        dist_prev = float('inf')

                score = 0.7 * dist_prev + 0.3 * float(cand_anchor[i])
                if score < best_score:
                    best_score = score
                    best_i = i
                    best_d = dist_prev

            if best_i is None:
                break

            # Accept best_i
            remaining[best_i] = False
            cid = cand_ids[best_i]
            used_ids.add(cid)
            subpath.append(items[best_i])
            # Only add if we still need more songs
            if len(playlist_ids) < n:
                playlist_ids.append(cid)
            try:
                v = items[best_i]['vector'].astype(np.float32)
            except Exception:
                v = np.array(items[best_i]['vector'], dtype=np.float32)
            selected_vectors[cid] = v
            selected_signatures.add((_normalize_string(items[best_i].get('title')), _normalize_string(items[best_i].get('author'))))
            if eliminate_duplicates:
                a = items[best_i].get('author')
                if a:
                    # update global artist count
                    artist_counts[a] = artist_counts.get(a, 0) + 1
                    # register this artist as present in this bucket (only increment bucket count once)
                    if a not in bucket_artist_set:
                        bucket_artist_set.add(a)
                        artist_bucket_counts[a] = artist_bucket_counts.get(a, 0) + 1

            if INSTRUMENT_BUCKET_SKIPS:
                logger.debug(f"Bucket {bucket_index}: accepted idx={best_i} item_id={cid}")

        return subpath

    # Process buckets sequentially; expand if needed until we reach n or run out
    processed_buckets = 0
    while len(playlist_ids) < n and processed_buckets < num_buckets:
        # ensure we process at least the nearest buckets_to_check first
        target = min(num_buckets, buckets_to_check)
        # iterate over the next unprocessed bucket indices
        for bi in range(processed_buckets, target):
            # for bucket 0, we already selected the first_song (candidate_data[0])
            start_id = None
            if bi == 0:
                start_id = playlist_ids[0] if playlist_ids else None
            sub = _walk_single_bucket(bi, start_item_id=start_id)
            processed_buckets += 1
            if len(playlist_ids) >= n:
                break

        # if we still need more and there are more buckets, expand the window
        if len(playlist_ids) < n and buckets_to_check < num_buckets:
            prev = buckets_to_check
            buckets_to_check = min(num_buckets, max(prev + 1, prev * 2))
            logger.info(f"Radius walk: expanded bucket processing window to {buckets_to_check} buckets (needed more songs)")
            continue

    logger.info(f"Radius walk: Walk complete. Collected {len(playlist_ids)} songs.")

    # --- Post-processing: avoid undesirable adjacency ---
    # Ensure we don't have three songs from the same artist in a row after stitching
    def _avoid_triple_adjacent(ids):
        # Build a lightweight map of item_id -> author from candidate_data (fallbacks to None)
        id_to_author = {cand['item_id']: cand.get('author') for cand in candidate_data}
        i = 0
        while i <= len(ids) - 3:
            a1 = id_to_author.get(ids[i])
            a2 = id_to_author.get(ids[i+1])
            a3 = id_to_author.get(ids[i+2])
            if a1 and a1 == a2 == a3:
                # Try to find a later item with a different artist to swap with the 3rd element
                swapped = False
                for j in range(i+3, len(ids)):
                    if id_to_author.get(ids[j]) != a1:
                        ids[i+2], ids[j] = ids[j], ids[i+2]
                        swapped = True
                        break
                # If not found, try swapping the middle element instead
                if not swapped:
                    for j in range(i+3, len(ids)):
                        if id_to_author.get(ids[j]) != a1:
                            ids[i+1], ids[j] = ids[j], ids[i+1]
                            swapped = True
                            break
                # If we couldn't swap, advance to avoid infinite loop
                if not swapped:
                    i += 1
                else:
                    # Re-evaluate the current window in case swaps created new triples
                    continue
            else:
                i += 1
        return ids

    playlist_ids = _avoid_triple_adjacent(playlist_ids)

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
    # Ensure we return exactly n items requested by the caller. Trim if we collected extra.
    playlist_ids = playlist_ids[:n]
    # Rebuild final_results from the (possibly trimmed) playlist_ids
    final_results = []
    for item_id in playlist_ids:
        dist_anchor = dist_anchor_map.get(item_id)
        if dist_anchor is not None:
            final_results.append({"item_id": item_id, "distance": dist_anchor})
        else:
            vec = _get_cached_vector(item_id)
            anchor_vec = _get_cached_vector(target_item_id)
            if vec is not None and anchor_vec is not None:
                final_results.append({"item_id": item_id, "distance": get_direct_distance(vec, anchor_vec)})

    # No final sort by distance is needed.
    return final_results

# --- END: RADIUS SIMILARITY RE-IMPLEMENTATION ---


def find_nearest_neighbors_by_id(target_item_id: str, n: int = 10, eliminate_duplicates: bool | None = None, mood_similarity: bool | None = None, radius_similarity: bool | None = None):
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
            eliminate_duplicates=eliminate_duplicates, # Pass this flag to apply artist cap pre-walk
            mood_similarity=mood_similarity
        )
        
        # 2. Execute the bucketed greedy walk
        # The walk itself will return exactly n items (or fewer if the pool is too small)
        final_results = _execute_radius_walk(
            target_item_id=target_item_id,
            n=n,
            candidate_data=candidate_data,
            original_song_details=target_song_details,
            eliminate_duplicates=eliminate_duplicates
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
        return final_results[:n]

def find_nearest_neighbors_by_vector(query_vector: np.ndarray, n: int = 100, eliminate_duplicates: bool | None = None):
    """
    Finds the N nearest neighbors for a given query vector.
    """
    if voyager_index is None or id_map is None:
        raise RuntimeError("Voyager index is not loaded in memory.")

    from app_helper import get_db, get_score_data_by_ids
    db_conn = get_db()

    # If caller didn't supply eliminate_duplicates explicitly (None), use configured default
    if eliminate_duplicates is None:
        eliminate_duplicates = SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT

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
    Returns the exact maximum distance from the given item to any other item in the loaded voyager index.
    Returns a dict: { 'max_distance': float, 'farthest_item_id': str | None }
    Raises RuntimeError if the index is not loaded.
    """
    if voyager_index is None or id_map is None or reverse_id_map is None:
        raise RuntimeError("Voyager index is not loaded in memory. It may be missing, empty, or the server failed to load it on startup.")

    target_voyager_id = reverse_id_map.get(target_item_id)
    if target_voyager_id is None:
        return None

    try:
        query_vector = voyager_index.get_vector(target_voyager_id)
    except Exception as e:
        logger.error(f"Could not retrieve vector for Voyager ID {target_voyager_id} (item_id: {target_item_id}): {e}")
        return None

    # Query distances to all items in the index (includes self). This returns a list of neighbor ids and distances.
    try:
        nbrs, dists = voyager_index.query(query_vector, k=len(voyager_index))
    except Exception as e:
        logger.error(f"Error querying voyager index for max distance of {target_item_id}: {e}", exc_info=True)
        return None

    # Find the maximum distance excluding the item itself
    max_d = float('-inf')
    far_voy = None
    for vid, dist in zip(nbrs, dists):
        if vid == target_voyager_id:
            continue
        if dist is None:
            continue
        if dist > max_d:
            max_d = dist
            far_voy = vid

    if max_d == float('-inf'):
        # No other items in index (single-item index) -> distance 0.0
        return { 'max_distance': 0.0, 'farthest_item_id': None }

    return { 'max_distance': float(max_d), 'farthest_item_id': id_map.get(far_voy) }

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
            SELECT item_id, title, author, album, album_artist
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

