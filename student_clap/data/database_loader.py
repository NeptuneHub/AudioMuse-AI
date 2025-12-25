"""
Database Loader for Student CLAP Training

Loads teacher CLAP embeddings from PostgreSQL database.
"""

import os
import sys
import logging
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from typing import List, Dict, Tuple, Optional

# Add parent directory to path to import from main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logger = logging.getLogger(__name__)


class DatabaseLoader:
    """Loads CLAP embeddings from PostgreSQL database."""
    
    def __init__(self, config: dict):
        """
        Initialize database loader.
        
        Args:
            config: Database configuration dict with keys:
                - host: Database host
                - port: Database port
                - database: Database name
                - user: Database user
                - password: Database password
        """
        self.config = config
        self.conn = None
        
    def connect(self):
        """Establish database connection."""
        if self.conn is not None:
            return
            
        try:
            self.conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                connect_timeout=30,
                keepalives_idle=600,
                keepalives_interval=30,
                keepalives_count=3,
                options='-c statement_timeout=300000'
            )
            logger.info(f"Connected to database: {self.config['host']}:{self.config['port']}/{self.config['database']}")
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
            
    def close(self):
        """Close database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
            
    def load_embeddings(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Load all CLAP embeddings with metadata.
        
        Args:
            limit: Optional limit on number of embeddings to load
            
        Returns:
            List of dicts with keys:
                - item_id: Song ID
                - title: Song title
                - author: Song artist
                - embedding: 512-dim numpy array (float32)
        """
        self.connect()
        
        query = """
            SELECT ce.item_id, ce.embedding, s.title, s.author
            FROM clap_embedding ce
            JOIN score s ON ce.item_id = s.item_id
            WHERE ce.embedding IS NOT NULL
        """
        
        if limit is not None:
            query += f" LIMIT {limit}"
            
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
                
            results = []
            for row in rows:
                # Decode embedding from BYTEA
                embedding_bytes = bytes(row['embedding'])
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Verify embedding dimension
                if len(embedding) != 512:
                    logger.warning(f"Unexpected embedding dimension for {row['item_id']}: {len(embedding)} (expected 512)")
                    continue
                    
                results.append({
                    'item_id': row['item_id'],
                    'title': row['title'] or 'Unknown',
                    'author': row['author'] or 'Unknown Artist',
                    'embedding': embedding
                })
                
            logger.info(f"Loaded {len(results)} embeddings from database")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
            
    def get_embedding_stats(self) -> Dict:
        """
        Get statistics about embeddings in database.
        
        Returns:
            Dict with keys:
                - total_embeddings: Total number of embeddings
                - total_songs: Total number of songs
                - coverage: Percentage of songs with embeddings
        """
        self.connect()
        
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                # Count embeddings
                cur.execute("SELECT COUNT(*) as count FROM clap_embedding WHERE embedding IS NOT NULL")
                total_embeddings = cur.fetchone()['count']
                
                # Count total songs
                cur.execute("SELECT COUNT(*) as count FROM score")
                total_songs = cur.fetchone()['count']
                
            coverage = (total_embeddings / total_songs * 100) if total_songs > 0 else 0
            
            stats = {
                'total_embeddings': total_embeddings,
                'total_songs': total_songs,
                'coverage': coverage
            }
            
            logger.info(f"Database stats: {total_embeddings} embeddings, {total_songs} songs, {coverage:.1f}% coverage")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            raise
            
    def verify_embedding(self, item_id: str) -> bool:
        """
        Verify that an embedding exists for a given item_id.
        
        Args:
            item_id: Song item ID
            
        Returns:
            True if embedding exists and is valid
        """
        self.connect()
        
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    "SELECT embedding FROM clap_embedding WHERE item_id = %s",
                    (item_id,)
                )
                row = cur.fetchone()
                
            if row is None:
                return False
                
            # Decode and verify
            embedding_bytes = bytes(row['embedding'])
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            return len(embedding) == 512
            
        except Exception as e:
            logger.error(f"Failed to verify embedding for {item_id}: {e}")
            return False


def verify_database_connection(config: dict) -> bool:
    """
    Verify database connection and check for required tables.
    
    Args:
        config: Database configuration dict
        
    Returns:
        True if connection successful and tables exist
    """
    try:
        loader = DatabaseLoader(config)
        loader.connect()
        
        # Check for required tables
        with loader.conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('clap_embedding', 'score')
            """)
            tables = [row[0] for row in cur.fetchall()]
            
        loader.close()
        
        if 'clap_embedding' not in tables:
            logger.error("Table 'clap_embedding' not found in database")
            return False
            
        if 'score' not in tables:
            logger.error("Table 'score' not found in database")
            return False
            
        logger.info("Database connection verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False


if __name__ == '__main__':
    """Test database loader functionality."""
    import yaml
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test database loader')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    parser.add_argument('--verify', action='store_true',
                        help='Verify database connection')
    parser.add_argument('--limit', type=int, default=10,
                        help='Number of embeddings to load for testing')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Expand environment variables
    db_config = {}
    for key, value in config['database'].items():
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            db_config[key] = os.environ.get(env_var, value)
        else:
            db_config[key] = value
    
    if args.verify:
        # Verify connection
        if verify_database_connection(db_config):
            print("✓ Database connection verified")
        else:
            print("✗ Database connection failed")
            sys.exit(1)
    
    # Load and display stats
    loader = DatabaseLoader(db_config)
    
    stats = loader.get_embedding_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total embeddings: {stats['total_embeddings']}")
    print(f"  Total songs: {stats['total_songs']}")
    print(f"  Coverage: {stats['coverage']:.1f}%")
    
    # Load sample embeddings
    print(f"\nLoading {args.limit} sample embeddings...")
    embeddings = loader.load_embeddings(limit=args.limit)
    
    if embeddings:
        print(f"✓ Successfully loaded {len(embeddings)} embeddings")
        print(f"\nSample embedding:")
        sample = embeddings[0]
        print(f"  Item ID: {sample['item_id']}")
        print(f"  Title: {sample['title']}")
        print(f"  Author: {sample['author']}")
        print(f"  Embedding shape: {sample['embedding'].shape}")
        print(f"  Embedding norm: {np.linalg.norm(sample['embedding']):.4f}")
    else:
        print("✗ No embeddings loaded")
        sys.exit(1)
    
    loader.close()
