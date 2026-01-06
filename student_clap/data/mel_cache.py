"""
Mel Spectrogram Cache using SQLite

Caches computed mel spectrograms to avoid recomputing them in subsequent epochs.
First epoch: compute and save to SQLite
Later epochs: load directly from SQLite (massive speedup!)
"""

import os
import sqlite3
import logging
import numpy as np
import io
import zlib
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class MelSpectrogramCache:
    """SQLite-based cache for mel spectrograms."""
    
    def __init__(self, db_path: str = "./cache/mel_spectrograms.db"):
        """
        Initialize mel spectrogram cache.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        
        # Create table if not exists
        self._create_table()
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.migrations_completed = 0
        
        logger.info(f"Mel spectrogram cache initialized: {self.db_path}")
        
    def _create_table(self):
        """Create mel spectrogram cache table."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mel_spectrograms (
                item_id TEXT PRIMARY KEY,
                num_segments INTEGER NOT NULL,
                mel_shape_time INTEGER NOT NULL,
                mel_shape_mels INTEGER NOT NULL,
                mel_data BLOB NOT NULL,
                compressed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        
        # Create song_embeddings table for teacher CLAP embeddings (averaged)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS song_embeddings (
                item_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                file_path TEXT NOT NULL,
                compressed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        
        # Create segment_embeddings table for per-segment teacher CLAP embeddings
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS segment_embeddings (
                item_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                compressed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (item_id, segment_index)
            )
        """)
        self.conn.commit()
        
        # Add compressed column to existing tables (migration)
        try:
            self.conn.execute("ALTER TABLE song_embeddings ADD COLUMN compressed INTEGER DEFAULT 0")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            self.conn.execute("ALTER TABLE segment_embeddings ADD COLUMN compressed INTEGER DEFAULT 0")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Add compressed column to existing tables (migration)
        try:
            self.conn.execute("ALTER TABLE mel_spectrograms ADD COLUMN compressed INTEGER DEFAULT 0")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Create index for faster lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_item_id ON mel_spectrograms(item_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_item_id ON song_embeddings(item_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_segment_item_id ON segment_embeddings(item_id)
        """)
        self.conn.commit()
        
    def get(self, item_id: str) -> Optional[np.ndarray]:
        """
        Get mel spectrogram from cache.
        
        Args:
            item_id: Song item ID
            
        Returns:
            Mel spectrogram array of shape (num_segments, 1, n_mels, time) or None if not cached
        """
        cursor = self.conn.execute(
            "SELECT num_segments, mel_shape_time, mel_shape_mels, mel_data, compressed FROM mel_spectrograms WHERE item_id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            self.cache_misses += 1
            return None
            
        # Deserialize mel spectrogram
        num_segments, mel_time, mel_mels, mel_data_bytes, is_compressed = row
        
        # Decompress if compressed (background thread handles migration)
        if is_compressed:
            mel_data_bytes = zlib.decompress(mel_data_bytes)
        # If uncompressed, use as-is (background thread will compress it later)
        
        mel_data = np.frombuffer(mel_data_bytes, dtype=np.float32)
        # Reshape to (num_segments, 1, n_mels, time)
        mel_data = mel_data.reshape(num_segments, 1, mel_mels, mel_time)
        
        self.cache_hits += 1
        logger.debug(f"Cache HIT for {item_id}: {mel_data.shape}")
        return mel_data
    
    def get_with_compression_status(self, item_id: str) -> Optional[tuple]:
        """
        Get mel and compression status (for parallel batch loading).
        Returns: (mel_array, is_compressed, original_bytes_if_uncompressed) or None
        """
        cursor = self.conn.execute(
            "SELECT num_segments, mel_shape_time, mel_shape_mels, mel_data, compressed FROM mel_spectrograms WHERE item_id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            self.cache_misses += 1
            return None
            
        num_segments, mel_time, mel_mels, mel_data_bytes, is_compressed = row
        
        if is_compressed:
            # Decompress and return
            decompressed = zlib.decompress(mel_data_bytes)
            mel_data = np.frombuffer(decompressed, dtype=np.float32)
            mel_data = mel_data.reshape(num_segments, 1, mel_mels, mel_time)
            self.cache_hits += 1
            return (mel_data, True, None)  # Already compressed
        else:
            # Return uncompressed data + original bytes for compression
            mel_data = np.frombuffer(mel_data_bytes, dtype=np.float32)
            mel_data = mel_data.reshape(num_segments, 1, mel_mels, mel_time)
            self.cache_hits += 1
            return (mel_data, False, mel_data_bytes)  # Needs compression
    
    def compress_and_update(self, item_id: str, original_bytes: bytes):
        """Compress and update entry (called from worker thread)."""
        try:
            compressed = zlib.compress(original_bytes, level=6)
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("UPDATE mel_spectrograms SET mel_data = ?, compressed = 1 WHERE item_id = ?",
                        (compressed, item_id))
            conn.commit()
            conn.close()
            self.migrations_completed += 1
        except Exception as e:
            logger.debug(f"Compression failed for {item_id}: {e}")
        
    def put(self, item_id: str, mel_spectrogram: np.ndarray):
        """
        Store mel spectrogram in cache (IMMEDIATE commit for crash safety).
        
        Args:
            item_id: Song item ID
            mel_spectrogram: Mel spectrogram array of shape (num_segments, 1, n_mels, time)
        """
        # Serialize mel spectrogram
        if mel_spectrogram.dtype != np.float32:
            mel_spectrogram = mel_spectrogram.astype(np.float32)
            
        mel_data_bytes = mel_spectrogram.tobytes()
        # Shape is (num_segments, 1, n_mels, time_frames)
        num_segments = mel_spectrogram.shape[0]
        mel_channels = mel_spectrogram.shape[1]  # Always 1
        mel_n_mels = mel_spectrogram.shape[2]     # 128
        mel_time = mel_spectrogram.shape[3]       # time frames
        
        # Compress with zlib (level 6 for good balance)
        compressed_bytes = zlib.compress(mel_data_bytes, level=6)
        compression_ratio = len(mel_data_bytes) / len(compressed_bytes)
        
        try:
            # Insert or replace (atomic operation)
            self.conn.execute("""
                INSERT OR REPLACE INTO mel_spectrograms 
                (item_id, num_segments, mel_shape_time, mel_shape_mels, mel_data, compressed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (item_id, num_segments, mel_time, mel_n_mels, compressed_bytes, 1))
            
            # IMMEDIATE commit - ensures data is saved even if process crashes!
            self.conn.commit()
            
            logger.debug(f"💾 Cache SAVED (compressed {compression_ratio:.1f}x): {item_id} {mel_spectrogram.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save mel cache for {item_id}: {e}")
            self.conn.rollback()
            raise
        
    def has(self, item_id: str) -> bool:
        """
        Check if item is in cache.
        
        Args:
            item_id: Song item ID
            
        Returns:
            True if cached, False otherwise
        """
        cursor = self.conn.execute(
            "SELECT 1 FROM mel_spectrograms WHERE item_id = ? LIMIT 1",
            (item_id,)
        )
        return cursor.fetchone() is not None
        
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM mel_spectrograms")
        total_cached = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT SUM(LENGTH(mel_data)) FROM mel_spectrograms")
        total_size_bytes = cursor.fetchone()[0] or 0
        
        # Get compression stats
        cursor = self.conn.execute("SELECT COUNT(*) FROM mel_spectrograms WHERE compressed = 1")
        compressed_count = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM mel_spectrograms WHERE compressed = 0 OR compressed IS NULL")
        uncompressed_count = cursor.fetchone()[0]
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        compression_rate = (compressed_count / total_cached * 100) if total_cached > 0 else 0
        
        return {
            'total_cached': total_cached,
            'cache_size_mb': total_size_bytes / (1024 * 1024),
            'cache_size_gb': total_size_bytes / (1024 * 1024 * 1024),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'compressed_count': compressed_count,
            'uncompressed_count': uncompressed_count,
            'compression_rate_percent': compression_rate
        }
    
    def get_cache_size_gb(self) -> float:
        """
        Get total cache size in GB.
        
        Returns:
            Cache size in gigabytes
        """
        cursor = self.conn.execute("SELECT SUM(LENGTH(mel_data)) as total_bytes FROM mel_spectrograms")
        row = cursor.fetchone()
        total_bytes = row[0] or 0
        return total_bytes / (1024 * 1024 * 1024)
    
    def get_cached_item_ids(self) -> list:
        """
        Get list of all cached item IDs.
        
        Returns:
            List of item_id strings
        """
        cursor = self.conn.execute("SELECT item_id FROM mel_spectrograms")
        return [row[0] for row in cursor.fetchall()]
        
    def clear(self):
        """Clear all cached mel spectrograms."""
        self.conn.execute("DELETE FROM mel_spectrograms")
        self.conn.commit()
        logger.info("Cleared all mel spectrogram cache")
    
    def put_embedding(self, item_id: str, embedding: np.ndarray, file_path: str):
        """
        Store teacher CLAP embedding for a song (compressed).
        
        Args:
            item_id: Song item ID
            embedding: Teacher embedding array (512-dim)
            file_path: Full path to audio file
        """
        # Serialize embedding
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        embedding_bytes = embedding.tobytes()
        
        # Compress with zlib (embeddings are ~2KB each, but good for storage)
        compressed_bytes = zlib.compress(embedding_bytes, level=6)
        
        self.conn.execute(
            "INSERT OR REPLACE INTO song_embeddings (item_id, embedding, file_path, compressed) VALUES (?, ?, ?, ?)",
            (item_id, compressed_bytes, file_path, 1)
        )
        self.conn.commit()
    
    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Get teacher CLAP embedding from cache (with decompression).
        
        Args:
            item_id: Song item ID
            
        Returns:
            Embedding array (512-dim) or None if not cached
        """
        cursor = self.conn.execute(
            "SELECT embedding, compressed FROM song_embeddings WHERE item_id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        embedding_bytes, is_compressed = row
        
        # Decompress if needed
        if is_compressed:
            embedding_bytes = zlib.decompress(embedding_bytes)
        
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        return embedding
    
    def has_embedding(self, item_id: str) -> bool:
        """
        Check if teacher embedding is cached for a song.
        
        Args:
            item_id: Song item ID
            
        Returns:
            True if embedding is cached
        """
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM song_embeddings WHERE item_id = ?",
            (item_id,)
        )
        count = cursor.fetchone()[0]
        return count > 0
    
    def put_segment_embeddings(self, item_id: str, segment_embeddings: list):
        """
        Store per-segment teacher CLAP embeddings for a song (compressed).
        
        Args:
            item_id: Song item ID
            segment_embeddings: List of embedding arrays (one per segment, each 512-dim)
        """
        # Delete existing segments first
        self.conn.execute("DELETE FROM segment_embeddings WHERE item_id = ?", (item_id,))
        
        # Insert all segments with compression
        for segment_idx, embedding in enumerate(segment_embeddings):
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
            embedding_bytes = embedding.tobytes()
            
            # Compress
            compressed_bytes = zlib.compress(embedding_bytes, level=6)
            
            self.conn.execute(
                "INSERT INTO segment_embeddings (item_id, segment_index, embedding, compressed) VALUES (?, ?, ?, ?)",
                (item_id, segment_idx, compressed_bytes, 1)
            )
        
        self.conn.commit()
        logger.debug(f"💾 Cached {len(segment_embeddings)} compressed segment embeddings for {item_id}")
    
    def get_segment_embeddings(self, item_id: str) -> Optional[list]:
        """
        Get per-segment teacher CLAP embeddings from cache (with decompression).
        
        Args:
            item_id: Song item ID
            
        Returns:
            List of embedding arrays (512-dim each) or None if not cached
        """
        cursor = self.conn.execute(
            "SELECT segment_index, embedding, compressed FROM segment_embeddings WHERE item_id = ? ORDER BY segment_index",
            (item_id,)
        )
        rows = cursor.fetchall()
        
        if not rows:
            return None
        
        embeddings = []
        for segment_idx, embedding_bytes, is_compressed in rows:
            # Decompress if needed
            if is_compressed:
                embedding_bytes = zlib.decompress(embedding_bytes)
            
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding)
        
        return embeddings
    
    def has_segment_embeddings(self, item_id: str) -> bool:
        """
        Check if per-segment teacher embeddings are cached for a song.
        
        Args:
            item_id: Song item ID
            
        Returns:
            True if segment embeddings are cached
        """
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM segment_embeddings WHERE item_id = ?",
            (item_id,)
        )
        count = cursor.fetchone()[0]
        return count > 0
    
    def get_song_info(self, item_id: str) -> Optional[Dict]:
        """
        Get song information from cache.
        
        Args:
            item_id: Song item ID
            
        Returns:
            Dict with file_path and created_at, or None if not cached
        """
        cursor = self.conn.execute(
            "SELECT file_path, created_at FROM song_embeddings WHERE item_id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return {
            'file_path': row[0],
            'created_at': row[1]
        }
        
    def clear(self):
        """Clear all cached data."""
        self.conn.execute("DELETE FROM mel_spectrograms")
        self.conn.execute("DELETE FROM song_embeddings")
        self.conn.commit()
        logger.info("Cleared all mel spectrogram and embedding cache")
        
    def close(self):
        """Close database connection."""
        stats = self.get_stats()
        logger.info(f"Mel cache stats: {stats['total_cached']} items, "
                   f"{stats['cache_size_gb']:.1f}GB, "
                   f"hit rate: {stats['hit_rate_percent']:.1f}%")
        logger.info(f"Compression: {stats['compressed_count']} compressed, "
                   f"{stats['uncompressed_count']} uncompressed "
                   f"({stats['compression_rate_percent']:.1f}% migrated)")
        if self.migrations_completed > 0:
            logger.info(f"🔄 On-demand compressions: {self.migrations_completed}")
        self.conn.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == '__main__':
    """Test mel cache functionality."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test cache
    print("Testing mel spectrogram cache...")
    
    with MelSpectrogramCache("./cache/test_mel_cache.db") as cache:
        # Create fake mel spectrogram
        test_id = "test_song_123"
        test_mel = np.random.randn(5, 1000, 128).astype(np.float32)  # 5 segments, 1000 time, 128 mels
        
        print(f"\n1. Testing PUT:")
        print(f"   Storing mel spectrogram: {test_mel.shape}")
        cache.put(test_id, test_mel)
        
        print(f"\n2. Testing HAS:")
        has_it = cache.has(test_id)
        print(f"   Item exists: {has_it}")
        
        print(f"\n3. Testing GET:")
        retrieved = cache.get(test_id)
        print(f"   Retrieved mel spectrogram: {retrieved.shape}")
        print(f"   Data matches: {np.allclose(test_mel, retrieved)}")
        
        print(f"\n4. Testing MISS:")
        missing = cache.get("nonexistent_id")
        print(f"   Missing item returns: {missing}")
        
        print(f"\n5. Cache Stats:")
        stats = cache.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    print("\n✓ Mel cache test complete")
