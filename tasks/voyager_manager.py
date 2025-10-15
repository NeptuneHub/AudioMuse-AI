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

from config import (
    EMBEDDING_DIMENSION, INDEX_NAME, VOYAGER_METRIC, VOYAGER_EF_CONSTRUCTION,
    VOYAGER_M, VOYAGER_QUERY_EF, MAX_SONGS_PER_ARTIST,
    DUPLICATE_DISTANCE_THRESHOLD_COSINE, DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN,
    DUPLICATE_DISTANCE_CHECK_LOOKBACK, MOOD_SIMILARITY_THRESHOLD
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
    if v1 is not None and v2 is not None:
        return np.linalg.norm(v1 - v2)
    return float('inf')

def _get_direct_angular_distance(v1, v2):
    if v1 is not None and v2 is not None and np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        cosine_similarity = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return np.arccos(cosine_similarity) / np.pi
    return float('inf')

def get_direct_distance(v1, v2):
    """Calculates direct distance between two vectors based on the VOYAGER_METRIC."""
    if VOYAGER_METRIC == 'angular':
        return _get_direct_angular_distance(v1, v2)
    else: # Default to euclidean
        return _get_direct_euclidean_distance(v1, v2)


def build_and_store_voyager_index(db_conn):
    """
    Fetches all song embeddings, builds a new Voyager index, and stores it
    atomically in the 'voyager_index_data' table in PostgreSQL.
    """
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
            space=space, # Use the mapped enum value
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
        
        # Use a more explicit approach for binary data storage
        upsert_query = """
            INSERT INTO voyager_index_data (index_name, index_data, id_map_json, embedding_dimension, created_at)
            VALUES (%s, %s::bytea, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (index_name) DO UPDATE SET
                index_data = EXCLUDED.index_data,
                id_map_json = EXCLUDED.id_map_json,
                embedding_dimension = EXCLUDED.embedding_dimension,
                created_at = CURRENT_TIMESTAMP;
        """
        # Use memoryview to ensure proper binary handling
        binary_data = memoryview(index_binary_data)
        cur.execute(upsert_query, (INDEX_NAME, binary_data, id_map_json, EMBEDDING_DIMENSION))
        db_conn.commit()
        logger.info("Voyager index build and database storage complete.")

    except Exception as e:
        logger.error("An error occurred during Voyager index build: %s", e, exc_info=True)
        db_conn.rollback()
    finally:
        cur.close()


def load_voyager_index_for_querying(force_reload=False):
    """
    Loads the Voyager index from the database into the global in-memory cache.
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
    added_songs_details = [{'title': original_song_details['title'], 'author': original_song_details['author']}] 

    for song in song_results:
        current_details = item_details.get(song['item_id'])
        if not current_details:
            logger.warning(f"Could not find details for item_id {song['item_id']} during deduplication. Skipping.")
            continue

        is_duplicate = False
        for added_detail in added_songs_details:
            if _is_same_song(
                current_details['title'], current_details['author'],
                added_detail['title'], added_detail['author']
            ):
                is_duplicate = True
                logger.info(f"Found duplicate (NAME FILTER): '{current_details['title']}' by '{current_details['author']}' (Distance from source: {song['distance']:.4f}).")
                break
        
        if not is_duplicate:
            unique_songs.append(song)
            added_songs_details.append(current_details)

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

def find_nearest_neighbors_by_id(target_item_id: str, n: int = 10, eliminate_duplicates: bool = False, mood_similarity: bool = True):
    """
    Finds the N nearest neighbors for a given item_id using the globally cached Voyager index.
    If mood_similarity is True, filters results by mood feature similarity (danceability, aggressive, happy, party, relaxed, sad).
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

    # Increase search size if we need mood filtering
    if mood_similarity:
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

    initial_results = []
    for voyager_id, dist in zip(neighbor_voyager_ids, distances):
        item_id = id_map.get(voyager_id)
        if item_id and item_id != target_item_id:
            initial_results.append({"item_id": item_id, "distance": float(dist)})

    # --- IMPLEMENTATION OF USER PROPOSAL ---
    # 1. Create a representation of the original song to prepend to the list.
    original_song_for_filtering = {"item_id": target_item_id, "distance": 0.0}
    
    # 2. Prepend the original song to the neighbor results.
    results_with_original = [original_song_for_filtering] + initial_results
    
    # 3. Pass the combined list to the distance filter.
    # The first song in the lookback window will now be the original song.
    temp_filtered_results = _filter_by_distance(results_with_original, db_conn)
    
    # 4. Remove the original song from the filtered list before proceeding.
    distance_filtered_results = [song for song in temp_filtered_results if song['item_id'] != target_item_id]
    # --- END OF IMPLEMENTATION ---

    unique_results_by_song = _deduplicate_and_filter_neighbors(distance_filtered_results, db_conn, target_song_details)
    
    # Apply mood similarity filtering if requested
    if mood_similarity:
        logger.info(f"Mood similarity filtering requested for target_item_id: {target_item_id}")
        unique_results_by_song = _filter_by_mood_similarity(unique_results_by_song, target_item_id, db_conn)
    else:
        logger.info(f"No mood similarity filtering requested (mood_similarity={mood_similarity})")
    
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
