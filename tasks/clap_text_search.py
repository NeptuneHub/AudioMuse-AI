"""
CLAP Text Search Manager
Provides in-memory caching and fast text-based music search using CLAP embeddings.
"""

import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time

import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from typing import List, Dict, Optional
import config

logger = logging.getLogger(__name__)

# Global in-memory cache state
_CLAP_CACHE = {
    'loaded': False
}

# Global in-memory CLAP index cache
_CLAP_INDEX_CACHE = {
    'index': None,
    'id_map': None,
    'reverse_id_map': None,
    'loaded': False
}

# Top queries cache (precomputed at startup)
_TOP_QUERIES_CACHE = {
    'queries': [],
    'ready': False,
    'computing': False
}

# Warm cache timer for text search (keeps model loaded)
_WARM_CACHE_TIMER = {
    'expiry_time': None,  # Unix timestamp when model should unload
    'timer_thread': None,  # Background thread for unloading
    'lock': threading.Lock(),
    'duration_seconds': None  # Loaded from config on first use
}


def get_clap_cache_size() -> int:
    """Return the number of items in the loaded CLAP index."""
    if _CLAP_INDEX_CACHE['loaded'] and _CLAP_INDEX_CACHE['id_map'] is not None:
        return len(_CLAP_INDEX_CACHE['id_map'])
    return 0


def _fetch_clap_metadata(item_ids: list) -> Dict[str, Dict[str, str]]:
    """Fetch metadata for CLAP result item_ids from the database."""
    metadata_map: Dict[str, Dict[str, str]] = {}
    if not item_ids:
        return metadata_map

    from app_helper import get_score_data_by_ids
    try:
        track_details_list = get_score_data_by_ids(item_ids)
        for row in track_details_list:
            item_id = row['item_id']
            metadata_map[item_id] = {
                'title': row.get('title', ''),
                'author': row.get('author', '')
            }
    except Exception:
        pass

    return metadata_map


def _split_bytes(data: bytes, part_size: int) -> list:
    """Split a bytes object into chunks no larger than part_size."""
    return [data[i:i + part_size] for i in range(0, len(data), part_size)]


def _load_clap_index_from_db() -> bool:
    """Load a persisted CLAP voyager index from the database."""
    global _CLAP_CACHE, _CLAP_INDEX_CACHE

    from app_helper import get_db
    from config import CLAP_EMBEDDING_DIMENSION, VOYAGER_QUERY_EF

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 0")
            cur.execute(
                "SELECT index_data, id_map_json, embedding_dimension FROM clap_index_data WHERE index_name = %s",
                ('clap_index',)
            )
            row = cur.fetchone()

            index_stream = None
            try:
                if row:
                    index_binary_data, id_map_json, db_embedding_dim = row
                    index_stream = tempfile.TemporaryFile()
                    index_stream.write(index_binary_data)
                    index_stream.seek(0)
                else:
                    seg_pattern = re.compile(r'^clap_index_(\d+)_(\d+)$')
                    parts = []
                    total_expected = None
                    id_map_json_candidate = None
                    with conn.cursor(name='clap_index_segments') as seg_cur:
                        seg_cur.itersize = 50
                        seg_cur.execute(
                            "SELECT index_name, index_data, id_map_json, embedding_dimension FROM clap_index_data WHERE index_name LIKE %s ESCAPE '\\'",
                            (r'clap_index\_%\_%',)
                        )
                        for name, part_data, part_id_map_json, part_dim in seg_cur:
                            m = seg_pattern.match(name)
                            if not m:
                                continue
                            part_no = int(m.group(1))
                            total = int(m.group(2))
                            if total_expected is None:
                                total_expected = total
                            elif total_expected != total:
                                logger.error(f"Segment total mismatch for CLAP index parts ({total_expected} vs {total}).")
                                return False
                            parts.append((part_no, part_data, part_id_map_json, part_dim))
                            if part_id_map_json and not id_map_json_candidate:
                                id_map_json_candidate = part_id_map_json

                    if total_expected is None or len(parts) != total_expected:
                        logger.error(f"Incomplete CLAP index segments: expected {total_expected}, found {len(parts)}.")
                        return False

                    parts.sort(key=lambda p: p[0])
                    for _, _, _, part_dim in parts:
                        if part_dim != CLAP_EMBEDDING_DIMENSION:
                            logger.error(f"CLAP index embedding_dimension mismatch in segmented parts: expected {CLAP_EMBEDDING_DIMENSION}, got {part_dim}.")
                            return False

                    if id_map_json_candidate is None:
                        logger.error("No id_map_json found in segmented CLAP index rows.")
                        return False

                    db_embedding_dim = parts[0][3]
                    index_stream = tempfile.TemporaryFile()
                    for _, part_data, _, _ in parts:
                        index_stream.write(part_data)
                    index_stream.seek(0)
                    id_map_json = id_map_json_candidate

                if index_stream is None:
                    logger.error("CLAP index binary data was empty.")
                    return False

                if db_embedding_dim != CLAP_EMBEDDING_DIMENSION:
                    logger.error(f"CLAP index dimension mismatch: db={db_embedding_dim} expected={CLAP_EMBEDDING_DIMENSION}")
                    index_stream.close()
                    return False

                try:
                    try:
                        import voyager  # type: ignore
                    except ImportError:
                        logger.warning("Voyager library is unavailable; cannot load persisted CLAP index.")
                        return False

                    loaded_index = voyager.Index.load(index_stream)
                    loaded_index.ef = VOYAGER_QUERY_EF
                finally:
                    if index_stream is not None:
                        try:
                            index_stream.close()
                        except Exception as close_error:
                            logger.warning("Failed to close CLAP index stream: %s", close_error, exc_info=True)

            except Exception:
                if index_stream is not None:
                    try:
                        index_stream.close()
                    except Exception:
                        pass
                raise

            id_map = {int(k): v for k, v in json.loads(id_map_json).items()}
            reverse_id_map = {v: k for k, v in id_map.items()}

            if not id_map:
                logger.error("CLAP index id_map is empty.")
                return False

            _CLAP_CACHE['loaded'] = True

            _CLAP_INDEX_CACHE['index'] = loaded_index
            _CLAP_INDEX_CACHE['id_map'] = id_map
            _CLAP_INDEX_CACHE['reverse_id_map'] = reverse_id_map
            _CLAP_INDEX_CACHE['loaded'] = True

            logger.info(f"CLAP index loaded from database with {len(id_map)} items.")
            return True
    except Exception as e:
        logger.error(f"Failed to load CLAP index from DB: {e}", exc_info=True)
        return False


def build_and_store_clap_index(db_conn=None):
    """Build a CLAP text search voyager index from stored CLAP embeddings and save it to the DB."""
    from app_helper import get_db
    from config import CLAP_EMBEDDING_DIMENSION, VOYAGER_METRIC, VOYAGER_M, VOYAGER_EF_CONSTRUCTION, VOYAGER_QUERY_EF, VOYAGER_MAX_PART_SIZE_MB

    try:
        import voyager  # type: ignore
    except ImportError:
        logger.warning("Voyager library is unavailable; cannot build CLAP index.")
        return False

    if db_conn is None:
        db_conn = get_db()

    max_part_size = VOYAGER_MAX_PART_SIZE_MB * 1024 * 1024

    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT item_id, embedding FROM clap_embedding")
            all_embeddings = cur.fetchall()

            if not all_embeddings:
                logger.warning("No CLAP embeddings found in DB to build CLAP index.")
                return False

            space = voyager.Space.Cosine if VOYAGER_METRIC == 'angular' else {
                'euclidean': voyager.Space.Euclidean,
                'dot': voyager.Space.InnerProduct
            }.get(VOYAGER_METRIC, voyager.Space.Cosine)

            logger.info(f"Building CLAP voyager index for {len(all_embeddings)} items...")
            index_builder = voyager.Index(space=space, num_dimensions=CLAP_EMBEDDING_DIMENSION, M=VOYAGER_M, ef_construction=VOYAGER_EF_CONSTRUCTION)

            id_map = {}
            vectors = []
            voyager_id = 0
            for item_id, embedding_blob in all_embeddings:
                if embedding_blob is None:
                    logger.warning(f"Skipping CLAP item {item_id} because embedding is NULL.")
                    continue
                embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)
                if embedding_vector.shape[0] != CLAP_EMBEDDING_DIMENSION:
                    logger.warning(f"Skipping CLAP item {item_id}: dimension {embedding_vector.shape[0]} != {CLAP_EMBEDDING_DIMENSION}")
                    continue
                vectors.append(embedding_vector)
                id_map[voyager_id] = item_id
                voyager_id += 1

            if not vectors:
                logger.warning("No valid CLAP embedding vectors found for CLAP index build.")
                return False

            index_builder.add_items(np.vstack(vectors), ids=np.array(list(id_map.keys())))

            with tempfile.NamedTemporaryFile(delete=False, suffix='.voyager') as tmp:
                temp_file_path = tmp.name
            try:
                index_builder.save(temp_file_path)
                with open(temp_file_path, 'rb') as f:
                    index_binary_data = f.read()
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            if not index_binary_data:
                logger.error("Generated CLAP index binary is empty. Aborting storage.")
                return False

            id_map_json = json.dumps(id_map)
            cur.execute(
                "DELETE FROM clap_index_data WHERE index_name = %s OR index_name LIKE %s ESCAPE '\\'",
                ('clap_index', r'clap_index\_%\_%')
            )

            if len(index_binary_data) <= max_part_size:
                cur.execute(
                    "INSERT INTO clap_index_data (index_name, index_data, id_map_json, embedding_dimension, created_at) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) ON CONFLICT (index_name) DO UPDATE SET index_data = EXCLUDED.index_data, id_map_json = EXCLUDED.id_map_json, embedding_dimension = EXCLUDED.embedding_dimension, created_at = EXCLUDED.created_at",
                    ('clap_index', psycopg2.Binary(index_binary_data), id_map_json, CLAP_EMBEDDING_DIMENSION)
                )
                logger.info("Stored CLAP index as single row in clap_index_data.")
            else:
                parts = _split_bytes(index_binary_data, max_part_size)
                num_parts = len(parts)
                insert_q = "INSERT INTO clap_index_data (index_name, index_data, id_map_json, embedding_dimension, created_at) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)"
                for idx, part in enumerate(parts, start=1):
                    part_name = f"clap_index_{idx}_{num_parts}"
                    part_id_map_json = id_map_json if idx == 1 else ''
                    cur.execute(insert_q, (part_name, psycopg2.Binary(part), part_id_map_json, CLAP_EMBEDDING_DIMENSION))
                logger.info(f"Stored CLAP index in {num_parts} segmented rows in clap_index_data.")

        db_conn.commit()
        logger.info("CLAP text search index build successful.")
        return True
    except Exception as e:
        logger.error(f"Failed to build and store CLAP index: {e}", exc_info=True)
        try:
            db_conn.rollback()
        except Exception:
            pass
        return False


def _unload_timer_worker():
    """Background thread that unloads CLAP text model after timer expires."""
    global _WARM_CACHE_TIMER
    
    while True:
        with _WARM_CACHE_TIMER['lock']:
            expiry = _WARM_CACHE_TIMER['expiry_time']
        
        if expiry is None:
            # Timer cancelled, exit thread
            break
        
        time_remaining = expiry - time.time()
        
        if time_remaining <= 0:
            # Timer expired - unload text model only
            from .clap_analyzer import unload_clap_model, is_clap_text_loaded
            
            if is_clap_text_loaded():
                logger.info("Warm cache timer expired - unloading CLAP text model")
                unload_clap_model()
            
            with _WARM_CACHE_TIMER['lock']:
                _WARM_CACHE_TIMER['expiry_time'] = None
                _WARM_CACHE_TIMER['timer_thread'] = None
            break
        
        # Sleep in 1-second chunks to check for cancellation
        time.sleep(min(1.0, time_remaining))


def warmup_text_search_model():
    """Preload CLAP text model (not audio model) and reset warmup timer.
    
    Returns:
        dict: Status with 'loaded' (bool) and 'expiry_seconds' (int)
    """
    global _WARM_CACHE_TIMER
    from .clap_analyzer import initialize_clap_text_model, is_clap_text_loaded
    
    # Load duration from config on first use
    if _WARM_CACHE_TIMER['duration_seconds'] is None:
        _WARM_CACHE_TIMER['duration_seconds'] = config.CLAP_TEXT_SEARCH_WARMUP_DURATION
    
    # Load text model only (not audio model - saves 268MB)
    if not is_clap_text_loaded():
        logger.info("Warming up CLAP text model for text search (not loading audio model)...")
        success = initialize_clap_text_model()
        if not success:
            return {'loaded': False, 'expiry_seconds': 0}
    
    # Reset timer
    with _WARM_CACHE_TIMER['lock']:
        _WARM_CACHE_TIMER['expiry_time'] = time.time() + _WARM_CACHE_TIMER['duration_seconds']
        
        # Start timer thread if not already running
        if _WARM_CACHE_TIMER['timer_thread'] is None or not _WARM_CACHE_TIMER['timer_thread'].is_alive():
            thread = threading.Thread(target=_unload_timer_worker, daemon=True)
            thread.start()
            _WARM_CACHE_TIMER['timer_thread'] = thread
            logger.info(f"Started warm cache timer ({_WARM_CACHE_TIMER['duration_seconds']}s)")
        else:
            logger.debug(f"Reset warm cache timer ({_WARM_CACHE_TIMER['duration_seconds']}s)")
    
    return {
        'loaded': True,
        'expiry_seconds': _WARM_CACHE_TIMER['duration_seconds']
    }


def get_warm_cache_status() -> Dict:
    """Get current warm cache status.
    
    Returns:
        dict: Status with 'active' (bool), 'seconds_remaining' (int)
    """
    global _WARM_CACHE_TIMER
    from .clap_analyzer import is_clap_model_loaded
    
    with _WARM_CACHE_TIMER['lock']:
        expiry = _WARM_CACHE_TIMER['expiry_time']
    
    if expiry is None or not is_clap_model_loaded():
        return {'active': False, 'seconds_remaining': 0}
    
    remaining = max(0, int(expiry - time.time()))
    return {'active': True, 'seconds_remaining': remaining}


def load_clap_cache_from_db():
    """
    Load the persisted CLAP Voyager index from the database.
    Returns True if successful, False otherwise.
    """
    global _CLAP_CACHE
    
    from app_helper import get_db
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
    """Force refresh of CLAP cache from database."""
    global _CLAP_CACHE
    old_count = get_clap_cache_size()
    logger.info(f"Refreshing CLAP cache... (current: {old_count} songs)")
    result = load_clap_cache_from_db()
    new_count = get_clap_cache_size()
    if result:
        logger.info(f"✓ CLAP cache refreshed: {old_count} → {new_count} songs ({new_count - old_count:+d})")
    else:
        logger.error(f"✗ CLAP cache refresh failed! Still at {new_count} songs")
    return result


def is_clap_cache_loaded() -> bool:
    """Check if CLAP cache is loaded and ready."""
    return _CLAP_CACHE['loaded']


def search_by_text(query_text: str, limit: int = 100) -> List[Dict]:
    """
    Search songs using natural language text query.
    
    Args:
        query_text: Natural language description (e.g., "upbeat summer songs")
        limit: Maximum number of results to return
        
    Returns:
        List of dicts with item_id, title, author, similarity
    """
    from .clap_analyzer import get_text_embedding
    from config import CLAP_ENABLED
    
    if not CLAP_ENABLED:
        return []
    
    # CLAP search must use the persisted index only
    if not _CLAP_INDEX_CACHE['loaded'] or _CLAP_INDEX_CACHE['index'] is None:
        logger.error("Cannot search: persisted CLAP index not loaded. Ensure Flask startup loaded the CLAP index.")
        return []
    
    try:
        # Auto-warmup: ensures model is loaded and resets timer
        warmup_text_search_model()
        
        # Get text embedding (model is now guaranteed loaded)
        text_embedding = get_text_embedding(query_text)
        if text_embedding is None:
            logger.error(f"Failed to generate text embedding for: {query_text}")
            return []

        from config import MAX_SONGS_PER_ARTIST
        artist_cap = MAX_SONGS_PER_ARTIST if MAX_SONGS_PER_ARTIST and MAX_SONGS_PER_ARTIST > 0 else 0
        fetch_size = (limit + max(20, limit * 4) + 1) if artist_cap else limit

        if _CLAP_INDEX_CACHE['loaded'] and _CLAP_INDEX_CACHE['index'] is not None:
            voyager_index = _CLAP_INDEX_CACHE['index']
            id_map = _CLAP_INDEX_CACHE['id_map'] or {}
            num_to_query = min(fetch_size, len(voyager_index))

            if num_to_query <= 0:
                logger.warning("CLAP index is loaded but contains no items.")
                return []

            neighbor_ids, distances = voyager_index.query(text_embedding, k=num_to_query)
            candidate_item_ids = [id_map.get(int(voyager_id)) for voyager_id in neighbor_ids]
            candidate_item_ids = [item_id for item_id in candidate_item_ids if item_id is not None]

            metadata_map = _fetch_clap_metadata(candidate_item_ids)

            results = []
            artist_counts: dict = {}
            for voyager_id, distance in zip(neighbor_ids, distances):
                if len(results) >= limit:
                    break
                item_id = id_map.get(int(voyager_id))
                if item_id is None:
                    continue

                metadata = metadata_map.get(item_id, {'title': '', 'author': ''})
                author = metadata.get('author', '')

                if artist_cap and author:
                    author_norm = author.strip().lower()
                    if artist_counts.get(author_norm, 0) >= artist_cap:
                        continue
                    artist_counts[author_norm] = artist_counts.get(author_norm, 0) + 1

                similarity = 1.0 - float(distance)
                results.append({
                    'item_id': item_id,
                    'title': metadata.get('title', ''),
                    'author': metadata.get('author', ''),
                    'similarity': similarity
                })

            logger.info(f"Text search '{query_text}': found {len(results)} results via CLAP index (artist cap: {artist_cap or 'disabled'})")
            return results
        
    except Exception as e:
        logger.error(f"Text search failed for '{query_text}': {e}")
        import traceback
        traceback.print_exc()
        return []


def get_cache_stats() -> Dict:
    """Get statistics about the CLAP cache."""
    if not _CLAP_INDEX_CACHE['loaded'] or _CLAP_INDEX_CACHE['index'] is None:
        return {
            'loaded': False,
            'song_count': 0,
            'embedding_dimension': 0,
            'memory_mb': 0
        }

    index_obj = _CLAP_INDEX_CACHE['index']
    index_size = sys.getsizeof(index_obj)
    if isinstance(index_obj, np.ndarray):
        index_size = index_obj.nbytes
    elif hasattr(index_obj, 'embeddings') and isinstance(index_obj.embeddings, np.ndarray):
        index_size = index_obj.embeddings.nbytes

    id_map_size = sys.getsizeof(_CLAP_INDEX_CACHE['id_map']) if _CLAP_INDEX_CACHE['id_map'] is not None else 0
    reverse_map_size = sys.getsizeof(_CLAP_INDEX_CACHE['reverse_id_map']) if _CLAP_INDEX_CACHE['reverse_id_map'] is not None else 0
    total_size_mb = (index_size + id_map_size + reverse_map_size) / (1024 * 1024)
    song_count = len(_CLAP_INDEX_CACHE['id_map']) if _CLAP_INDEX_CACHE['id_map'] is not None else 0

    return {
        'loaded': True,
        'song_count': song_count,
        'embedding_dimension': config.CLAP_EMBEDDING_DIMENSION,
        'memory_mb': round(total_size_mb, 2)
    }



def ensure_text_search_queries_table():
    """
    Create text_search_queries table if it doesn't exist.
    Called automatically at startup.
    """
    from app_helper import get_db
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(726354821)")
            try:
                cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", ('text_search_queries',))
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
    except Exception as e:
        logger.error(f"Failed to create text_search_queries table: {e}")
        if conn:
            conn.rollback()
        return False


def load_top_queries_from_db():
    """
    Load top queries from database into memory cache.
    Returns True if queries were loaded, False otherwise.
    On first startup (empty DB), this will return False and trigger generation.
    """
    global _TOP_QUERIES_CACHE
    from app_helper import get_db
    
    # Ensure table exists first
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
                # Insert default queries if table is empty
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
                    "pop energetic synthesizer"
                ]
                
                for rank, query in enumerate(default_queries, start=1):
                    cur.execute("""
                        INSERT INTO text_search_queries (query_text, score, rank, created_at)
                        VALUES (%s, %s, %s, NOW())
                    """, (query, 1.0, rank))
                
                conn.commit()
                
                # Load them into cache
                _TOP_QUERIES_CACHE['queries'] = default_queries
                _TOP_QUERIES_CACHE['ready'] = True
                logger.info(f"Inserted and loaded {len(default_queries)} default queries")
                return True
    except Exception as e:
        logger.warning(f"Could not load top queries from database: {e}")
        return False


def save_top_queries_to_db(queries: List[str], scores: List[float]):
    """
    Save top queries to database, replacing old ones atomically.
    This ensures users get old queries until new ones are ready.
    """
    from app_helper import get_db
    
    # Safety check: don't delete existing queries if new list is empty
    if not queries:
        logger.warning("Refusing to save empty query list to database")
        return False
    
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            # Delete old queries
            cur.execute("DELETE FROM text_search_queries")
            
            # Insert new queries
            for rank, (query, score) in enumerate(zip(queries, scores), start=1):
                cur.execute("""
                    INSERT INTO text_search_queries (query_text, score, rank, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (query, float(score), rank))
            
            conn.commit()
            logger.info(f"Saved {len(queries)} top queries to database")
            return True
    except Exception as e:
        logger.error(f"Failed to save top queries to database: {e}")
        if conn:
            conn.rollback()
        return False


def get_cached_top_queries() -> List[str]:
    """
    Get precomputed top queries from cache.
    Returns empty list if not ready yet.
    """
    if _TOP_QUERIES_CACHE['ready']:
        return _TOP_QUERIES_CACHE['queries']
    return []
