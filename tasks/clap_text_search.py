"""
CLAP Text Search Manager
Provides in-memory caching and fast text-based music search using CLAP embeddings.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)

# Global in-memory cache
_CLAP_CACHE = {
    'embeddings': None,  # NumPy array (N, 512)
    'metadata': None,    # List of dicts with item_id, title, author
    'item_ids': None,    # List of item_ids
    'loaded': False
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
