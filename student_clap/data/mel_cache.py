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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        
        # Create index for faster lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_item_id ON mel_spectrograms(item_id)
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
            "SELECT num_segments, mel_shape_time, mel_shape_mels, mel_data FROM mel_spectrograms WHERE item_id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            self.cache_misses += 1
            return None
            
        # Deserialize mel spectrogram
        num_segments, mel_time, mel_mels, mel_data_bytes = row
        mel_data = np.frombuffer(mel_data_bytes, dtype=np.float32)
        # Reshape to (num_segments, 1, n_mels, time)
        mel_data = mel_data.reshape(num_segments, 1, mel_mels, mel_time)
        
        self.cache_hits += 1
        logger.debug(f"Cache HIT for {item_id}: {mel_data.shape}")
        return mel_data
        
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
        
        try:
            # Insert or replace (atomic operation)
            self.conn.execute("""
                INSERT OR REPLACE INTO mel_spectrograms 
                (item_id, num_segments, mel_shape_time, mel_shape_mels, mel_data)
                VALUES (?, ?, ?, ?, ?)
            """, (item_id, num_segments, mel_time, mel_n_mels, mel_data_bytes))
            
            # IMMEDIATE commit - ensures data is saved even if process crashes!
            self.conn.commit()
            
            logger.debug(f"💾 Cache SAVED (crash-safe): {item_id} {mel_spectrogram.shape}")
            
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
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_cached': total_cached,
            'cache_size_mb': total_size_bytes / (1024 * 1024),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate
        }
        
    def clear(self):
        """Clear all cached mel spectrograms."""
        self.conn.execute("DELETE FROM mel_spectrograms")
        self.conn.commit()
        logger.info("Cleared all mel spectrogram cache")
        
    def close(self):
        """Close database connection."""
        stats = self.get_stats()
        logger.info(f"Mel cache stats: {stats['total_cached']} items, "
                   f"{stats['cache_size_mb']:.1f}MB, "
                   f"hit rate: {stats['hit_rate_percent']:.1f}%")
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
