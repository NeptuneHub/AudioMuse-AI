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
            
    def load_embeddings(self, limit: Optional[int] = None, sample_size: Optional[int] = None, 
                       balanced_genres: Optional[List[str]] = None) -> List[Dict]:
        """
        Load CLAP embeddings with metadata.
        
        Args:
            limit: Optional limit on number of embeddings to load (legacy, use sample_size instead)
            sample_size: Total number of songs to sample (e.g., 10000)
            balanced_genres: List of genres to balance equally across sample_size
            
        Returns:
            List of dicts with keys:
                - item_id: Song ID
                - title: Song title
                - author: Song artist
                - embedding: 512-dim numpy array (float32)
        """
        self.connect()
        
        # If balanced genre sampling is requested
        if sample_size and balanced_genres:
            return self._load_balanced_genre_sample(sample_size, balanced_genres)
        
        # Legacy path: load all or limited
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
    
    def _load_balanced_genre_sample(self, sample_size: int, genres: List[str]) -> List[Dict]:
        """
        Load embeddings with balanced genre sampling.
        
        Each song appears ONCE with its TOP genre (highest score in mood_vector).
        Uses mood_vector column which stores genres as "genre:score,genre2:score".
        
        Args:
            sample_size: Total number of songs to sample (e.g., 10000)
            genres: List of genre names to balance
            
        Returns:
            List of embedding dicts (each song appears exactly once)
        """
        songs_per_genre = sample_size // len(genres)
        logger.info(f"🎯 Balanced genre sampling: {sample_size} total songs, "
                   f"~{songs_per_genre} per genre across {len(genres)} genres")
        logger.info(f"   📊 Each song will appear ONCE with its TOP genre only")
        
        # Sort target genres by length (longest first) to prioritize specific genres
        # This ensures "hard rock" is checked before "rock", "indie pop" before "indie", etc.
        sorted_genres = sorted(genres, key=lambda g: len(g), reverse=True)
        
        # First, load ALL songs that match any of the requested genres
        genre_filter = " OR ".join([f"s.mood_vector ILIKE '%{genre}%'" for genre in genres])
        
        query = f"""
            SELECT 
                ce.item_id, 
                ce.embedding, 
                s.title, 
                s.author,
                s.mood_vector
            FROM clap_embedding ce
            JOIN score s ON ce.item_id = s.item_id
            WHERE ce.embedding IS NOT NULL
              AND ({genre_filter})
        """
        
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                logger.info(f"   ⏱️  Loading all songs with target genres...")
                cur.execute(query)
                rows = cur.fetchall()
            
            logger.info(f"   ✅ Found {len(rows)} songs matching target genres")
            
            # DEBUG: Sample a few mood_vectors to see the format
            logger.info(f"   🔍 Sample mood_vectors from database:")
            for i, row in enumerate(rows[:3]):
                logger.info(f"      Song {i+1}: {row['mood_vector'][:100]}...")
            
            # Group songs by their TOP genre
            genre_songs = {genre: [] for genre in genres}
            skipped = 0
            
            for row in rows:
                # Decode embedding
                embedding_bytes = bytes(row['embedding'])
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                if len(embedding) != 512:
                    logger.warning(f"Unexpected embedding dimension for {row['item_id']}: {len(embedding)}")
                    continue
                
                # Parse mood_vector to find TOP genre that matches our target list
                # Format: "female vocalists:0.577,jazz:0.544,folk:0.537,pop:0.535,rock:0.528"
                mood_vector = row['mood_vector'] or ""
                
                # Parse all genres with scores, sorted by score (highest first)
                song_genres = []
                for pair in mood_vector.split(','):
                    pair = pair.strip()
                    if ':' not in pair:
                        continue
                    genre_name, score_str = pair.split(':', 1)
                    genre_name = genre_name.strip().lower()
                    
                    try:
                        score = float(score_str)
                        song_genres.append((genre_name, score))
                    except ValueError:
                        continue
                
                # Sort by score descending
                song_genres.sort(key=lambda x: x[1], reverse=True)
                
                # Find the HIGHEST scored genre that matches our target list
                top_genre = None
                for genre_name, score in song_genres:
                    # Check if this genre EXACTLY matches any target genre (no substring matching!)
                    for target_genre in sorted_genres:
                        target_lower = target_genre.lower()
                        # EXACT match only - "rock" should NOT match "hard rock"
                        if target_lower == genre_name:
                            top_genre = target_genre
                            break
                    
                    # If we found a match, stop checking
                    if top_genre:
                        break
                
                # Add song to its top genre bucket
                if top_genre:
                    genre_songs[top_genre].append({
                        'item_id': row['item_id'],
                        'title': row['title'] or 'Unknown',
                        'author': row['author'] or 'Unknown Artist',
                        'embedding': embedding
                    })
                else:
                    skipped += 1
            
            if skipped > 0:
                logger.info(f"   📝 {skipped} songs not assigned yet (will use for fallback)")
            
            # FIRST: Sample from each genre bucket (only take songs_per_genre from each)
            results = []
            genre_counts = {}
            sampled_item_ids = set()
            
            for genre in genres:
                available = len(genre_songs[genre])
                to_sample = min(songs_per_genre, available)
                
                if to_sample > 0:
                    # Randomly sample (deterministic with numpy seed already set)
                    indices = np.random.choice(available, size=to_sample, replace=False)
                    sampled = [genre_songs[genre][i] for i in indices]
                    results.extend(sampled)
                    genre_counts[genre] = len(sampled)
                    # Track which songs were actually sampled
                    for song in sampled:
                        sampled_item_ids.add(song['item_id'])
                else:
                    genre_counts[genre] = 0
            
            logger.info(f"   ✅ PASS 1 sampled: {len(results)} songs")
            
            # PASS 2: Fill underrepresented genres with UN-SAMPLED songs (including leftovers from other genres!)
            # Collect all songs that were NOT sampled
            unsampled_songs = []
            for row in rows:
                if row['item_id'] not in sampled_item_ids:
                    embedding_bytes = bytes(row['embedding'])
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    if len(embedding) == 512:
                        unsampled_songs.append({
                            'item_id': row['item_id'],
                            'title': row['title'] or 'Unknown',
                            'author': row['author'] or 'Unknown Artist',
                            'embedding': embedding,
                            'mood_vector': row['mood_vector'] or ""
                        })
            
            logger.info(f"   🔄 PASS 2 FALLBACK: {len(unsampled_songs)} unsampled songs available (includes leftovers from popular genres)")
            
            # For each genre that needs more songs, check unsampled pool
            for target_genre in sorted_genres:
                current_count = genre_counts.get(target_genre, 0)
                if current_count < songs_per_genre and unsampled_songs:
                    needed = songs_per_genre - current_count
                    target_lower = target_genre.lower()
                    
                    # Find unsampled songs that have this genre EXACTLY in their mood_vector
                    candidates = []
                    for song in unsampled_songs:
                        mood_vector = song['mood_vector']
                        # Check if target genre appears EXACTLY anywhere in mood_vector
                        for pair in mood_vector.split(','):
                            if ':' not in pair:
                                continue
                            genre_name = pair.split(':', 1)[0].strip().lower()
                            
                            # EXACT match only - "hard rock" must match "hard rock" exactly
                            if target_lower == genre_name:
                                candidates.append(song)
                                break
                    
                    # Take what we need from candidates
                    if candidates:
                        to_add = min(needed, len(candidates))
                        for i in range(to_add):
                            song = candidates[i]
                            results.append({
                                'item_id': song['item_id'],
                                'title': song['title'],
                                'author': song['author'],
                                'embedding': song['embedding']
                            })
                            genre_counts[target_genre] = genre_counts.get(target_genre, 0) + 1
                            # Remove from unsampled
                            unsampled_songs.remove(song)
                        logger.info(f"   ✅ FALLBACK: Added {to_add} songs to '{target_genre}' (was {current_count}, now {genre_counts[target_genre]})")
            
            # Log distribution
            logger.info(f"   ✅ Loaded {len(results)} UNIQUE songs with balanced distribution:")
            for genre in sorted(genre_counts.keys()):
                logger.info(f"      {genre}: {genre_counts[genre]} songs")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to load balanced genre sample: {e}")
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
