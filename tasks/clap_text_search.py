"""
CLAP Text Search Manager
Provides in-memory caching and fast text-based music search using CLAP embeddings.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from psycopg2.extras import DictCursor
import config

logger = logging.getLogger(__name__)

# Global in-memory cache
_CLAP_CACHE = {
    'embeddings': None,  # NumPy array (N, 512)
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


def load_clap_cache_from_db():
    """
    Load all CLAP embeddings and metadata into memory for fast searching.
    Returns True if successful, False otherwise.
    """
    global _CLAP_CACHE
    
    from app_helper import get_db
    from config import CLAP_ENABLED, CLAP_EMBEDDING_DIMENSION
    
    if not CLAP_ENABLED:
        logger.info("CLAP is disabled, skipping cache load.")
        return False
    
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Fetch all CLAP embeddings with metadata from score table
        cur.execute("""
            SELECT 
                ce.item_id,
                ce.embedding,
                s.title,
                s.author
            FROM clap_embedding ce
            JOIN score s ON ce.item_id = s.item_id
            ORDER BY ce.item_id
        """)
        
        rows = cur.fetchall()
        cur.close()
        
        if not rows:
            logger.warning("No CLAP embeddings found in database.")
            _CLAP_CACHE['loaded'] = False
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
            
            if embedding.shape[0] != CLAP_EMBEDDING_DIMENSION:
                logger.warning(f"Skipping {item_id}: wrong dimension {embedding.shape[0]} (expected {CLAP_EMBEDDING_DIMENSION})")
                continue
            
            embeddings_list.append(embedding)
            metadata_list.append({
                'item_id': item_id,
                'title': title,
                'author': author
            })
            item_ids_list.append(item_id)
        
        if not embeddings_list:
            logger.error("No valid CLAP embeddings loaded.")
            _CLAP_CACHE['loaded'] = False
            return False
        
        # Convert to NumPy matrix for vectorized operations
        _CLAP_CACHE['embeddings'] = np.vstack(embeddings_list)
        _CLAP_CACHE['metadata'] = metadata_list
        _CLAP_CACHE['item_ids'] = item_ids_list
        _CLAP_CACHE['loaded'] = True
        
        logger.info(f"CLAP cache loaded: {len(metadata_list)} songs with 512-dim embeddings in memory")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load CLAP cache: {e}")
        import traceback
        traceback.print_exc()
        _CLAP_CACHE['loaded'] = False
        return False


def refresh_clap_cache():
    """Force refresh of CLAP cache from database."""
    return load_clap_cache_from_db()


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
    
    # Cache must be loaded at startup - no lazy loading
    if not _CLAP_CACHE['loaded']:
        logger.error("Cannot search: CLAP cache not loaded. Ensure Flask started successfully.")
        return []
    
    try:
        # Get text embedding
        text_embedding = get_text_embedding(query_text)
        if text_embedding is None:
            logger.error(f"Failed to generate text embedding for: {query_text}")
            return []
        
        # Vectorized similarity computation (cosine similarity via dot product)
        # Both embeddings are already normalized
        similarities = _CLAP_CACHE['embeddings'] @ text_embedding
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            idx = int(idx)
            similarity = float(similarities[idx])
            metadata = _CLAP_CACHE['metadata'][idx]
            
            results.append({
                'item_id': metadata['item_id'],
                'title': metadata['title'],
                'author': metadata['author'],
                'similarity': similarity
            })
        
        logger.info(f"Text search '{query_text}': found {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Text search failed for '{query_text}': {e}")
        import traceback
        traceback.print_exc()
        return []


def get_cache_stats() -> Dict:
    """Get statistics about the CLAP cache."""
    if not _CLAP_CACHE['loaded']:
        return {
            'loaded': False,
            'song_count': 0,
            'embedding_dimension': 0,
            'memory_mb': 0
        }
    
    embeddings_size = _CLAP_CACHE['embeddings'].nbytes if _CLAP_CACHE['embeddings'] is not None else 0
    metadata_size = sum(len(str(m)) for m in _CLAP_CACHE['metadata']) if _CLAP_CACHE['metadata'] else 0
    total_size_mb = (embeddings_size + metadata_size) / (1024 * 1024)
    
    return {
        'loaded': True,
        'song_count': len(_CLAP_CACHE['metadata']) if _CLAP_CACHE['metadata'] else 0,
        'embedding_dimension': _CLAP_CACHE['embeddings'].shape[1] if _CLAP_CACHE['embeddings'] is not None else 0,
        'memory_mb': round(total_size_mb, 2)
    }


def generate_top_queries(num_queries=None, top_n=50, return_scores=False):
    """
    Generate top N diverse queries by sampling from query.json.
    Uses probabilistic category selection to ensure balanced representation.
    Each query has exactly 3 terms from different categories.
    Avoids mixing Instrumentation and Voice_Type in same query.
    
    Args:
        num_queries: Number of random queries to generate (defaults to config.CLAP_TOP_QUERIES_COUNT)
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
        num_queries = config.CLAP_TOP_QUERIES_COUNT
    
    if not is_clap_cache_loaded():
        logger.error("CLAP cache not loaded, cannot generate top queries")
        return []
    
    # Load query.json
    query_file = os.path.join(os.path.dirname(__file__), 'query.json')
    with open(query_file, 'r') as f:
        query_data = json.load(f)
    
    # Get category weights from config (optimized for CLAP's strengths)
    category_weights = config.CLAP_CATEGORY_WEIGHTS
    
    # Mutually exclusive categories (can't appear together)
    conflicting_categories = [
        {'Instrumentation_Vocal', 'Voice_Type'}
    ]
    
    def categories_conflict(selected_categories):
        """Check if selected categories have conflicts."""
        selected_set = set(selected_categories)
        for conflict_set in conflicting_categories:
            if conflict_set.issubset(selected_set):
                return True
        return False
    
    # Generate random queries with smart category selection
    def generate_random_query():
        """Generate a query with exactly 3 terms from different categories using weighted sampling."""
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            # Weighted random sampling of 3 different categories
            categories = list(query_data.keys())
            weights = [category_weights[cat] for cat in categories]
            
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
            
            # Check for conflicts
            if not categories_conflict(selected_categories):
                # Sample one term from each selected category
                terms = [random.choice(query_data[cat]) for cat in selected_categories]
                # Shuffle terms so category order varies
                random.shuffle(terms)
                return ' '.join(terms).lower()
            
            attempts += 1
        
        # Fallback: simple safe selection
        safe_categories = ['Rhythm_Tempo', 'Genre_Style', 'Emotion_Mood']
        terms = [random.choice(query_data[cat]) for cat in safe_categories]
        random.shuffle(terms)
        return ' '.join(terms).lower()
    
    # Generate unique queries
    queries = list(set([generate_random_query() for _ in range(num_queries)]))
    logger.info(f"Generated {len(queries)} unique queries")
    
    # Compute embeddings in parallel using multithreading
    from .clap_analyzer import get_text_embedding
    
    try:
        num_cores = multiprocessing.cpu_count() // 2
        num_cores = max(1, num_cores)
        
        logger.info(f"Computing text embeddings for {len(queries)} queries using {num_cores} threads...")
        
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
    
    # Now score ALL queries at once using vectorized matrix multiplication
    song_embeddings = _CLAP_CACHE['embeddings']  # Shape: (N_songs, 512)
    
    logger.info(f"Scoring {len(queries)} queries against {len(song_embeddings)} songs using vectorized operations...")
    
    # Compute similarity matrix: (num_queries, N_songs)
    # This is a single matrix multiplication - FAST!
    similarity_matrix = np.dot(query_embeddings, song_embeddings.T)  # Shape: (500, N_songs)
    
    # For each query, get top 50 scores and sum them
    top_k = 50
    top_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :top_k]  # Top 50 per query
    
    # Get the actual scores for top 50
    row_indices = np.arange(len(queries))[:, None]
    top_scores = similarity_matrix[row_indices, top_indices]  # Shape: (num_queries, 50)
    
    # Sum top 50 scores for each query
    total_scores = np.sum(top_scores, axis=1)  # Shape: (num_queries,)
    
    logger.info(f"Computed scores for all queries in vectorized pass")
    
    # Now process term tracking in parallel
    def analyze_query_terms(idx):
        """Track which terms are used in a query."""
        query = queries[idx]
        terms_used = defaultdict(set)
        for category, terms_list in query_data.items():
            for term in terms_list:
                if term.lower() in query.lower():
                    terms_used[category].add(term)
        return (query, float(total_scores[idx]), dict(terms_used))
    
    # Use physical CPU cores for term analysis
    num_cores = multiprocessing.cpu_count() // 2  # Physical cores
    num_cores = max(1, num_cores)
    
    logger.info(f"Analyzing term diversity using {num_cores} threads...")
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        scored_queries = list(executor.map(analyze_query_terms, range(len(queries))))
    
    # Sort by score
    scored_queries.sort(key=lambda x: x[1], reverse=True)
    
    # Multi-pass selection to guarantee exactly top_n results
    selected = []
    term_usage_count = defaultdict(int)
    category_usage_count = defaultdict(int)
    
    # Pass 1: Strict diversity (aim for first 40%)
    target_pass1 = int(top_n * 0.4)
    for query, score, terms_used in scored_queries:
        if len(selected) >= target_pass1:
            break
        
        # Calculate diversity penalty
        term_penalty = 0
        category_penalty = 0
        
        for category, terms_set in terms_used.items():
            category_penalty += category_usage_count.get(category, 0) * 2
            for term in terms_set:
                term_penalty += term_usage_count.get(term, 0) * 5
        
        total_penalty = term_penalty + category_penalty
        
        # Skip if exact term appears more than once in pass 1
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
        
        # Update usage counts
        for category, terms_set in terms_used.items():
            category_usage_count[category] += 1
            for term in terms_set:
                term_usage_count[term] += 1
    
    # Pass 2: Relaxed diversity (next 40%, allow terms up to 2x)
    target_pass2 = int(top_n * 0.8)
    for query, score, terms_used in scored_queries:
        if len(selected) >= target_pass2:
            break
        
        if query in selected:
            continue
        
        # Allow terms to appear up to 2 times
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
        
        # Update usage counts
        for category, terms_set in terms_used.items():
            category_usage_count[category] += 1
            for term in terms_set:
                term_usage_count[term] += 1
    
    # Pass 3: Fill remaining slots with highest scores (no restrictions)
    for query, score, terms_used in scored_queries:
        if len(selected) >= top_n:
            break
        
        if query in selected:
            continue
        
        selected.append(query)
    
    logger.info(f"Selected {len(selected)} diverse queries from {len(scored_queries)} candidates")
    
    # Return with or without scores
    if return_scores:
        # Return list of dicts with query and score
        result = []
        for query in selected:
            # Find the score for this query
            score = next((s for q, s, _ in scored_queries if q == query), 0.0)
            result.append({'query': query, 'score': score})
        return result
    else:
        return selected


def ensure_text_search_queries_table():
    """
    Create text_search_queries table if it doesn't exist.
    Called automatically at startup.
    """
    from app_helper import get_db
    
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS text_search_queries (
                    id SERIAL PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    score REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(rank)
                )
            """)
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
                logger.info("No top queries found in database (first startup or after reset)")
                return False
    except Exception as e:
        logger.warning(f"Could not load top queries from database: {e}")
        return False


def save_top_queries_to_db(queries: List[str], scores: List[float]):
    """
    Save top queries to database, replacing old ones atomically.
    This ensures users get old queries until new ones are ready.
    """
    from app_helper import get_db
    
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


def precompute_top_queries_background():
    """
    Precompute top queries in background thread.
    Saves to database and updates in-memory cache.
    Users get old queries from DB until new ones are ready.
    """
    global _TOP_QUERIES_CACHE
    
    if _TOP_QUERIES_CACHE['computing']:
        logger.info("Top queries already being computed")
        return
    
    _TOP_QUERIES_CACHE['computing'] = True
    logger.info("Starting background computation of top queries...")
    
    try:
        # Generate queries with scores (uses config.CLAP_TOP_QUERIES_COUNT)
        scored_queries = generate_top_queries(top_n=50, return_scores=True)
        queries = [q['query'] for q in scored_queries]
        scores = [q['score'] for q in scored_queries]
        
        # Save to database needs Flask app context
        from app import app
        with app.app_context():
            # Save to database first (atomic replacement)
            if save_top_queries_to_db(queries, scores):
                # Update in-memory cache
                _TOP_QUERIES_CACHE['queries'] = queries
                _TOP_QUERIES_CACHE['ready'] = True
                logger.info(f"Top queries precomputed successfully: {len(queries)} queries ready")
            else:
                logger.error("Failed to save queries to database")
    except Exception as e:
        logger.error(f"Failed to precompute top queries: {e}")
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
