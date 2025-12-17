"""
MuLan Text Search Manager
Provides in-memory caching and fast text-based music search using MuLan embeddings.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from psycopg2.extras import DictCursor
import config
import threading
import time

logger = logging.getLogger(__name__)

# Global in-memory cache
_MULAN_CACHE = {
    'embeddings': None,  # NumPy array (N, embedding_dim)
    'metadata': None,    # List of dicts with item_id, title, author
    'item_ids': None,    # List of item_ids
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


def get_mulan_cache_size() -> int:
    """Return the number of embeddings in the MuLan cache."""
    global _MULAN_CACHE
    if _MULAN_CACHE['loaded'] and _MULAN_CACHE['embeddings'] is not None:
        return len(_MULAN_CACHE['embeddings'])
    return 0


def _unload_timer_worker():
    """Background thread that unloads MuLan model after timer expires."""
    global _WARM_CACHE_TIMER
    
    while True:
        with _WARM_CACHE_TIMER['lock']:
            expiry = _WARM_CACHE_TIMER['expiry_time']
        
        if expiry is None:
            # Timer cancelled, exit thread
            break
        
        time_remaining = expiry - time.time()
        
        if time_remaining <= 0:
            # Timer expired - unload model
            from .mulan_analyzer import unload_mulan_model, is_mulan_model_loaded
            
            if is_mulan_model_loaded():
                logger.info("Warm cache timer expired - unloading MuLan models")
                unload_mulan_model()
            
            with _WARM_CACHE_TIMER['lock']:
                _WARM_CACHE_TIMER['expiry_time'] = None
                _WARM_CACHE_TIMER['timer_thread'] = None
            break
        
        # Sleep in 1-second chunks to check for cancellation
        time.sleep(min(1.0, time_remaining))


def warmup_text_search_model():
    """Preload MuLan models and reset warmup timer.
    
    Returns:
        dict: Status with 'loaded' (bool) and 'expiry_seconds' (int)
    """
    global _WARM_CACHE_TIMER
    from .mulan_analyzer import initialize_mulan_model, is_mulan_model_loaded
    
    # Load duration from config on first use
    if _WARM_CACHE_TIMER['duration_seconds'] is None:
        _WARM_CACHE_TIMER['duration_seconds'] = config.MULAN_TEXT_SEARCH_WARMUP_DURATION
    
    # Load model if not already loaded
    if not is_mulan_model_loaded():
        logger.info("Warming up MuLan models for text search...")
        success = initialize_mulan_model()
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
    from .mulan_analyzer import is_mulan_model_loaded
    
    with _WARM_CACHE_TIMER['lock']:
        expiry = _WARM_CACHE_TIMER['expiry_time']
    
    if expiry is None or not is_mulan_model_loaded():
        return {'active': False, 'seconds_remaining': 0}
    
    remaining = max(0, int(expiry - time.time()))
    return {'active': True, 'seconds_remaining': remaining}


def load_mulan_cache_from_db():
    """
    Load all MuLan embeddings and metadata into memory for fast searching.
    Returns True if successful, False otherwise.
    """
    global _MULAN_CACHE
    
    from app_helper import get_db
    from config import MULAN_ENABLED, MULAN_EMBEDDING_DIMENSION
    
    if not MULAN_ENABLED:
        logger.info("MuLan is disabled, skipping cache load.")
        return False
    
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Fetch all MuLan embeddings with metadata from score table
        cur.execute("""
            SELECT 
                me.item_id,
                me.embedding,
                s.title,
                s.author
            FROM mulan_embedding me
            JOIN score s ON me.item_id = s.item_id
            ORDER BY me.item_id
        """)
        
        rows = cur.fetchall()
        cur.close()
        
        if not rows:
            logger.warning("No MuLan embeddings found in database.")
            _MULAN_CACHE['loaded'] = False
            return False
        
        # Build cache structures
        embeddings_list = []
        metadata_list = []
        item_ids_list = []
        
        for row in rows:
            item_id = row['item_id']
            embedding_blob = row['embedding']
            title = row['title']
            author = row['author']
            
            # Convert BYTEA to numpy array
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            if embedding.shape[0] != MULAN_EMBEDDING_DIMENSION:
                logger.warning(f"Skipping {item_id}: wrong dimension {embedding.shape[0]} (expected {MULAN_EMBEDDING_DIMENSION})")
                continue
            
            embeddings_list.append(embedding)
            metadata_list.append({
                'item_id': item_id,
                'title': title,
                'author': author
            })
            item_ids_list.append(item_id)
        
        if not embeddings_list:
            logger.error("No valid MuLan embeddings loaded.")
            _MULAN_CACHE['loaded'] = False
            return False
        
        # Convert to NumPy matrix for vectorized operations
        _MULAN_CACHE['embeddings'] = np.vstack(embeddings_list)
        _MULAN_CACHE['metadata'] = metadata_list
        _MULAN_CACHE['item_ids'] = item_ids_list
        _MULAN_CACHE['loaded'] = True
        
        logger.info(f"MuLan cache loaded: {len(metadata_list)} songs with {MULAN_EMBEDDING_DIMENSION}-dim embeddings in memory")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load MuLan cache: {e}")
        import traceback
        traceback.print_exc()
        _MULAN_CACHE['loaded'] = False
        return False


def refresh_mulan_cache():
    """Force refresh of MuLan cache from database."""
    return load_mulan_cache_from_db()


def is_mulan_cache_loaded() -> bool:
    """Check if MuLan cache is loaded and ready."""
    return _MULAN_CACHE['loaded']


def search_by_text(query_text: str, limit: int = 100) -> List[Dict]:
    """
    Search songs using natural language text query.
    
    Args:
        query_text: Natural language description (e.g., "upbeat summer songs")
        limit: Maximum number of results to return
        
    Returns:
        List of dicts with item_id, title, author, similarity
    """
    from .mulan_analyzer import get_text_embedding
    from config import MULAN_ENABLED
    
    if not MULAN_ENABLED:
        return []
    
    # Cache must be loaded at startup - no lazy loading
    if not _MULAN_CACHE['loaded']:
        logger.error("Cannot search: MuLan cache not loaded. Ensure Flask started successfully.")
        return []
    
    try:
        # Auto-warmup: ensures model is loaded and resets timer
        warmup_text_search_model()
        
        # Get text embedding (model is now guaranteed loaded)
        text_embedding = get_text_embedding(query_text)
        if text_embedding is None:
            logger.error(f"Failed to generate text embedding for: {query_text}")
            return []
        
        # Vectorized similarity computation (cosine similarity via dot product)
        # Both embeddings are already normalized
        similarities = _MULAN_CACHE['embeddings'] @ text_embedding
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            idx = int(idx)
            similarity = float(similarities[idx])
            metadata = _MULAN_CACHE['metadata'][idx]
            
            results.append({
                'item_id': metadata['item_id'],
                'title': metadata['title'],
                'author': metadata['author'],
                'similarity': similarity
            })
        
        logger.info(f"MuLan text search '{query_text}': found {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"MuLan text search failed for '{query_text}': {e}")
        import traceback
        traceback.print_exc()
        return []


def get_cache_stats() -> Dict:
    """Get statistics about the MuLan cache."""
    if not _MULAN_CACHE['loaded']:
        return {
            'loaded': False,
            'song_count': 0,
            'embedding_dimension': 0,
            'memory_mb': 0
        }
    
    embeddings_size = _MULAN_CACHE['embeddings'].nbytes if _MULAN_CACHE['embeddings'] is not None else 0
    metadata_size = sum(len(str(m)) for m in _MULAN_CACHE['metadata']) if _MULAN_CACHE['metadata'] else 0
    total_size_mb = (embeddings_size + metadata_size) / (1024 * 1024)
    
    return {
        'loaded': True,
        'song_count': len(_MULAN_CACHE['metadata']) if _MULAN_CACHE['metadata'] else 0,
        'embedding_dimension': _MULAN_CACHE['embeddings'].shape[1] if _MULAN_CACHE['embeddings'] is not None else 0,
        'memory_mb': round(total_size_mb, 2)
    }


def generate_top_queries(num_queries=None, top_n=50, return_scores=False):
    """
    Generate top N diverse queries by sampling from query.json.
    Uses probabilistic category selection to ensure balanced representation.
    Each query has exactly 3 terms from different categories.
    
    Args:
        num_queries: Number of random queries to generate (defaults to config.MULAN_TOP_QUERIES_COUNT)
        top_n: Number of top queries to return
        return_scores: If True, return list of dicts with 'query' and 'score' keys
    
    Returns:
        List of query strings (or dicts if return_scores=True) sorted by score and diversity.
    """
    import json
    import os
    import random
    from concurrent.futures import ThreadPoolExecutor
    from collections import defaultdict
    import multiprocessing
    
    # Use config default if not specified
    if num_queries is None:
        num_queries = config.MULAN_TOP_QUERIES_COUNT
    
    if not is_mulan_cache_loaded():
        logger.error("MuLan cache not loaded, cannot generate top queries")
        return []
    
    # Load query.json
    query_file = os.path.join(os.path.dirname(__file__), 'query.json')
    with open(query_file, 'r') as f:
        query_data = json.load(f)
    
    # Get category weights from config
    category_weights = config.MULAN_CATEGORY_WEIGHTS
    
    # Generate random queries
    def generate_random_query():
        """Generate a query with exactly 3 terms from different categories using weighted sampling."""
        categories = list(query_data.keys())
        weights = [category_weights.get(cat, 1.0) for cat in categories]
        
        # Sample 3 unique categories
        selected_categories = []
        available_categories = list(categories)
        available_weights = list(weights)
        
        for _ in range(3):
            if not available_categories:
                break
                
            # Normalize weights
            total_weight = sum(available_weights)
            normalized_weights = [w / total_weight for w in available_weights]
            
            # Choose category
            chosen_idx = random.choices(range(len(available_categories)), weights=normalized_weights)[0]
            selected_categories.append(available_categories[chosen_idx])
            
            # Remove selected category from pool
            available_categories.pop(chosen_idx)
            available_weights.pop(chosen_idx)
        
        # Sample one term from each selected category
        terms = [random.choice(query_data[cat]) for cat in selected_categories]
        # Shuffle terms
        random.shuffle(terms)
        return ' '.join(terms).lower()
    
    # Generate unique queries
    queries = list(set([generate_random_query() for _ in range(num_queries)]))
    logger.info(f"Generated {len(queries)} unique MuLan queries")
    
    # Compute embeddings
    from .mulan_analyzer import get_text_embedding
    
    try:
        # Use single core to prevent OOM
        num_cores = 1
        
        logger.info(f"Computing text embeddings for {len(queries)} queries...")
        
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            query_embeddings = list(executor.map(get_text_embedding, queries))
        
        # Filter out any None results
        valid_data = [(q, e) for q, e in zip(queries, query_embeddings) if e is not None]
        if not valid_data:
            logger.error("No valid embeddings generated")
            return []
        
        queries = [q for q, _ in valid_data]
        query_embeddings = np.vstack([e for _, e in valid_data])
        
        logger.info(f"Computed {len(query_embeddings)} text embeddings")
        
    except Exception as e:
        logger.error(f"Failed to compute text embeddings: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    # Score all queries using vectorized operations
    song_embeddings = _MULAN_CACHE['embeddings']
    
    logger.info(f"Scoring {len(queries)} queries against {len(song_embeddings)} songs...")
    
    # Compute similarity matrix
    similarity_matrix = np.dot(query_embeddings, song_embeddings.T)
    
    # For each query, get top 50 scores and sum them
    top_k = 50
    top_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :top_k]
    
    row_indices = np.arange(len(queries))[:, None]
    top_scores = similarity_matrix[row_indices, top_indices]
    
    # Sum top 50 scores for each query
    total_scores = np.sum(top_scores, axis=1)
    
    logger.info(f"Computed scores for all queries")
    
    # Process term tracking
    def analyze_query_terms(idx):
        """Track which terms are used in a query."""
        query = queries[idx]
        terms_used = defaultdict(set)
        query_words = set(query.lower().split())
        for category, terms_list in query_data.items():
            for term in terms_list:
                term_words = set(term.lower().split())
                if term_words.issubset(query_words):
                    terms_used[category].add(term)
        return (query, float(total_scores[idx]), dict(terms_used))
    
    num_cores = 1  # Use single core
    
    logger.info(f"Analyzing term diversity...")
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        scored_queries = list(executor.map(analyze_query_terms, range(len(queries))))
    
    # Sort by score
    scored_queries.sort(key=lambda x: x[1], reverse=True)
    
    # Multi-pass selection
    selected = []
    term_usage_count = defaultdict(int)
    category_usage_count = defaultdict(int)
    
    # Pass 1: Strict diversity
    target_pass1 = int(top_n * 0.4)
    for query, score, terms_used in scored_queries:
        if len(selected) >= target_pass1:
            break
        
        has_overused_term = False
        for category, terms_set in terms_used.items():
            for term in terms_set:
                if term_usage_count.get(term, 0) >= 1:
                    has_overused_term = True
                    break
            if has_overused_term:
                break
        
        if has_overused_term:
            continue
        
        selected.append(query)
        
        for category, terms_set in terms_used.items():
            category_usage_count[category] += 1
            for term in terms_set:
                term_usage_count[term] += 1
    
    # Pass 2: Relaxed diversity
    target_pass2 = int(top_n * 0.8)
    for query, score, terms_used in scored_queries:
        if len(selected) >= target_pass2:
            break
        
        if query in selected:
            continue
        
        has_overused_term = False
        for category, terms_set in terms_used.items():
            for term in terms_set:
                if term_usage_count.get(term, 0) >= 2:
                    has_overused_term = True
                    break
            if has_overused_term:
                break
        
        if has_overused_term:
            continue
        
        selected.append(query)
        
        for category, terms_set in terms_used.items():
            category_usage_count[category] += 1
            for term in terms_set:
                term_usage_count[term] += 1
    
    # Pass 3: Fill remaining slots
    for query, score, terms_used in scored_queries:
        if len(selected) >= top_n:
            break
        
        if query in selected:
            continue
        
        selected.append(query)
    
    logger.info(f"Selected {len(selected)} diverse MuLan queries")
    
    # Return with or without scores
    if return_scores:
        result = []
        for query in selected:
            score = next((s for q, s, _ in scored_queries if q == query), 0.0)
            result.append({'query': query, 'score': score})
        return result
    else:
        return selected


def ensure_text_search_queries_table():
    """
    Create mulan_text_search_queries table if it doesn't exist.
    Called automatically at startup.
    """
    from app_helper import get_db
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mulan_text_search_queries (
                    id SERIAL PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    score REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(rank)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mulan_text_search_queries_rank 
                ON mulan_text_search_queries(rank)
            """)
            conn.commit()
            logger.info("Ensured mulan_text_search_queries table exists")
            return True
    except Exception as e:
        logger.error(f"Failed to create mulan_text_search_queries table: {e}")
        if conn:
            conn.rollback()
        return False


def load_top_queries_from_db():
    """
    Load top queries from database into memory cache.
    Returns True if queries were loaded, False otherwise.
    """
    global _TOP_QUERIES_CACHE
    from app_helper import get_db
    
    ensure_text_search_queries_table()
    
    try:
        conn = get_db()
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT query_text, score, rank 
                FROM mulan_text_search_queries 
                ORDER BY rank ASC
            """)
            rows = cur.fetchall()
            
            if rows:
                _TOP_QUERIES_CACHE['queries'] = [row['query_text'] for row in rows]
                _TOP_QUERIES_CACHE['ready'] = True
                logger.info(f"Loaded {len(rows)} MuLan top queries from database")
                return True
            else:
                # Insert default queries if table is empty
                logger.info("No MuLan top queries found - inserting default queries")
                default_queries = [
                    "energetic electronic dance",
                    "relaxing acoustic guitar",
                    "heavy metal aggressive",
                    "smooth jazz saxophone",
                    "indie pop upbeat",
                    "classical piano peaceful",
                    "hip hop bass",
                    "rock guitar energetic",
                    "folk acoustic melancholic",
                    "ambient electronic atmospheric",
                    "blues guitar soulful",
                    "reggae rhythmic groovy",
                    "country acoustic storytelling",
                    "techno electronic fast",
                    "punk rock aggressive",
                    "soul vocal emotional",
                    "trap bass heavy",
                    "house electronic dance",
                    "alternative rock moody",
                    "r&b smooth vocals",
                    "metal guitar distorted",
                    "disco groovy funky",
                    "funk bass groovy",
                    "dubstep bass heavy",
                    "gospel choir uplifting",
                    "latin rhythmic percussion",
                    "experimental electronic weird",
                    "lofi hip hop relaxing",
                    "progressive rock complex",
                    "synthwave electronic retro",
                    "grunge guitar distorted",
                    "neo soul smooth",
                    "edm electronic energetic",
                    "shoegaze guitar dreamy",
                    "ska upbeat brass",
                    "trip hop moody electronic",
                    "indie folk acoustic intimate",
                    "hard rock guitar heavy",
                    "bossa nova smooth relaxing",
                    "drum and bass fast",
                    "psychedelic rock trippy",
                    "afrobeat rhythmic percussion",
                    "post punk dark moody",
                    "downtempo electronic chill",
                    "bluegrass acoustic banjo",
                    "new wave synth upbeat",
                    "darkwave electronic moody",
                    "world music percussion",
                    "garage rock raw",
                    "trance electronic uplifting"
                ]
                
                for rank, query in enumerate(default_queries, start=1):
                    cur.execute("""
                        INSERT INTO mulan_text_search_queries (query_text, score, rank, created_at)
                        VALUES (%s, %s, %s, NOW())
                    """, (query, 1.0, rank))
                
                conn.commit()
                
                _TOP_QUERIES_CACHE['queries'] = default_queries
                _TOP_QUERIES_CACHE['ready'] = True
                logger.info(f"Inserted and loaded {len(default_queries)} default MuLan queries")
                return True
    except Exception as e:
        logger.warning(f"Could not load MuLan top queries from database: {e}")
        return False


def save_top_queries_to_db(queries: List[str], scores: List[float]):
    """
    Save top queries to database, replacing old ones atomically.
    """
    from app_helper import get_db
    
    if not queries:
        logger.warning("Refusing to save empty MuLan query list to database")
        return False
    
    conn = None
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mulan_text_search_queries")
            
            for rank, (query, score) in enumerate(zip(queries, scores), start=1):
                cur.execute("""
                    INSERT INTO mulan_text_search_queries (query_text, score, rank, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (query, float(score), rank))
            
            conn.commit()
            logger.info(f"Saved {len(queries)} MuLan top queries to database")
            return True
    except Exception as e:
        logger.error(f"Failed to save MuLan top queries to database: {e}")
        if conn:
            conn.rollback()
        return False


def precompute_top_queries_background():
    """
    Precompute top queries in background thread.
    Saves to database and updates in-memory cache.
    """
    global _TOP_QUERIES_CACHE
    
    if _TOP_QUERIES_CACHE['computing']:
        logger.info("MuLan top queries already being computed")
        return
    
    _TOP_QUERIES_CACHE['computing'] = True
    logger.info("Starting background computation of MuLan top queries...")
    
    try:
        scored_queries = generate_top_queries(top_n=50, return_scores=True)
        
        if not scored_queries:
            logger.warning("MuLan query generation returned empty list - skipping save")
            return
        
        queries = [q['query'] for q in scored_queries]
        scores = [q['score'] for q in scored_queries]
        
        # Save to database needs Flask app context
        from app import app
        with app.app_context():
            if save_top_queries_to_db(queries, scores):
                _TOP_QUERIES_CACHE['queries'] = queries
                _TOP_QUERIES_CACHE['ready'] = True
                logger.info(f"MuLan top queries precomputed successfully: {len(queries)} queries ready")
            else:
                logger.error("Failed to save MuLan queries to database")
    except Exception as e:
        logger.error(f"Failed to precompute MuLan top queries: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _TOP_QUERIES_CACHE['computing'] = False


def get_cached_top_queries() -> List[str]:
    """
    Get precomputed top queries from cache.
    Returns empty list if not ready yet.
    """
    if _TOP_QUERIES_CACHE['ready']:
        return _TOP_QUERIES_CACHE['queries']
    return []
