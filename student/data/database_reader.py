"""
Database reader for loading training data from PostgreSQL.
Reads embeddings, scores, and metadata from AudioMuse-AI database.
"""

import logging
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class DatabaseReader:
    """Reader for AudioMuse-AI PostgreSQL database."""
    
    def __init__(self, config: Dict):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration dictionary with keys:
                    host, port, user, password, dbname
        """
        self.config = config
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                user=self.config['user'],
                password=self.config['password'],
                dbname=self.config['dbname'],
                connect_timeout=30,
                keepalives_idle=600,
                keepalives_interval=30,
                keepalives_count=3
            )
            logger.info("Successfully connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_songs_with_embeddings(self) -> List[Dict]:
        """
        Get all songs that have both MusiCNN and CLAP embeddings.
        
        Returns:
            List of dictionaries containing song metadata and item_id
        """
        query = """
            SELECT 
                s.item_id,
                s.title,
                s.author,
                s.tempo,
                s.energy,
                s.key,
                s.scale,
                s.mood_vector,
                s.other_features
            FROM score s
            INNER JOIN embedding e ON s.item_id = e.item_id
            INNER JOIN clap_embedding ce ON s.item_id = ce.item_id
            WHERE e.embedding IS NOT NULL 
            AND ce.embedding IS NOT NULL
            ORDER BY s.item_id
        """
        
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query)
                results = cur.fetchall()
                
                songs = []
                for row in results:
                    songs.append({
                        'item_id': row['item_id'],
                        'title': row['title'],
                        'author': row['author'],
                        'tempo': row['tempo'],
                        'energy': row['energy'],
                        'key': row['key'],
                        'scale': row['scale'],
                        'mood_vector': row['mood_vector'],
                        'other_features': row['other_features']
                    })
                
                logger.info(f"Found {len(songs)} songs with both MusiCNN and CLAP embeddings")
                return songs
                
        except Exception as e:
            logger.error(f"Error fetching songs with embeddings: {e}")
            raise
    
    def get_musicnn_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Get MusiCNN embedding for a song.
        
        Args:
            item_id: Song item ID
            
        Returns:
            200-dimensional numpy array or None if not found
        """
        query = "SELECT embedding FROM embedding WHERE item_id = %s"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (item_id,))
                result = cur.fetchone()
                
                if result and result[0]:
                    # Convert BYTEA to numpy array
                    embedding_bytes = bytes(result[0])
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    return embedding
                else:
                    logger.warning(f"No MusiCNN embedding found for item_id: {item_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching MusiCNN embedding for {item_id}: {e}")
            return None
    
    def get_clap_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Get CLAP audio embedding for a song.
        
        Args:
            item_id: Song item ID
            
        Returns:
            512-dimensional numpy array or None if not found
        """
        query = "SELECT embedding FROM clap_embedding WHERE item_id = %s"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (item_id,))
                result = cur.fetchone()
                
                if result and result[0]:
                    # Convert BYTEA to numpy array
                    embedding_bytes = bytes(result[0])
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    return embedding
                else:
                    logger.warning(f"No CLAP embedding found for item_id: {item_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching CLAP embedding for {item_id}: {e}")
            return None
    
    def get_batch_embeddings(self, item_ids: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get both MusiCNN and CLAP embeddings for multiple songs.
        
        Args:
            item_ids: List of song item IDs
            
        Returns:
            Tuple of (musicnn_embeddings_dict, clap_embeddings_dict)
        """
        musicnn_embeddings = {}
        clap_embeddings = {}
        
        # Fetch MusiCNN embeddings
        musicnn_query = "SELECT item_id, embedding FROM embedding WHERE item_id = ANY(%s)"
        try:
            with self.conn.cursor() as cur:
                cur.execute(musicnn_query, (item_ids,))
                results = cur.fetchall()
                
                for item_id, embedding_bytes in results:
                    if embedding_bytes:
                        embedding = np.frombuffer(bytes(embedding_bytes), dtype=np.float32)
                        musicnn_embeddings[item_id] = embedding
                        
        except Exception as e:
            logger.error(f"Error fetching batch MusiCNN embeddings: {e}")
        
        # Fetch CLAP embeddings
        clap_query = "SELECT item_id, embedding FROM clap_embedding WHERE item_id = ANY(%s)"
        try:
            with self.conn.cursor() as cur:
                cur.execute(clap_query, (item_ids,))
                results = cur.fetchall()
                
                for item_id, embedding_bytes in results:
                    if embedding_bytes:
                        embedding = np.frombuffer(bytes(embedding_bytes), dtype=np.float32)
                        clap_embeddings[item_id] = embedding
                        
        except Exception as e:
            logger.error(f"Error fetching batch CLAP embeddings: {e}")
        
        logger.info(f"Fetched {len(musicnn_embeddings)} MusiCNN and {len(clap_embeddings)} CLAP embeddings")
        return musicnn_embeddings, clap_embeddings
