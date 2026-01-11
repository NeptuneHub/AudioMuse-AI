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
        
        logger.info(f"Mel spectrogram cache initialized: {self.db_path}")
        
    def _create_table(self):
        """Create mel spectrogram cache table (NEW FORMAT ONLY - stores full spectrograms)."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mel_spectrograms (
                item_id TEXT PRIMARY KEY,
                mel_shape_time INTEGER NOT NULL,
                mel_shape_mels INTEGER NOT NULL,
                mel_data BLOB NOT NULL,
                audio_length INTEGER NOT NULL,
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (item_id, segment_index)
            )
        """)
        self.conn.commit()
        
        # Create indexes for faster lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_item_id ON mel_spectrograms(item_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_segment_item_id ON segment_embeddings(item_id)
        """)
        self.conn.commit()
        
    def get(self, item_id: str) -> Optional[np.ndarray]:
        """
        Get full mel spectrogram from cache.
        
        Args:
            item_id: Song item ID
            
        Returns:
            Full mel spectrogram array of shape (1, n_mels, time) or None if not cached
        """
        cursor = self.conn.execute(
            "SELECT mel_shape_time, mel_shape_mels, mel_data, audio_length FROM mel_spectrograms WHERE item_id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            self.cache_misses += 1
            return None
            
        # Deserialize mel spectrogram
        mel_time, mel_mels, mel_data_bytes, audio_length = row
        
        # Decompress
        mel_data_bytes = zlib.decompress(mel_data_bytes)
        
        mel_data = np.frombuffer(mel_data_bytes, dtype=np.float32)
        # Reshape to (1, n_mels, time)
        mel_data = mel_data.reshape(1, mel_mels, mel_time)
        
        self.cache_hits += 1
        logger.debug(f"Cache HIT for {item_id}: {mel_data.shape}")
        return mel_data
    
    def get_with_audio_length(self, item_id: str) -> Optional[tuple]:
        """
        Get full mel spectrogram with audio length info.
        Returns: (mel_array, audio_length) or None
        """
        cursor = self.conn.execute(
            "SELECT mel_shape_time, mel_shape_mels, mel_data, audio_length FROM mel_spectrograms WHERE item_id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            self.cache_misses += 1
            return None
            
        mel_time, mel_mels, mel_data_bytes, audio_length = row
        
        # Decompress
        decompressed = zlib.decompress(mel_data_bytes)
        mel_data = np.frombuffer(decompressed, dtype=np.float32)
        mel_data = mel_data.reshape(1, mel_mels, mel_time)
        
        self.cache_hits += 1
        return (mel_data, audio_length)
    

    def put(self, item_id: str, mel_spectrogram: np.ndarray, audio_length: int):
        """
        Store FULL mel spectrogram in cache (compressed, immediate commit for crash safety).
        
        Args:
            item_id: Song item ID
            mel_spectrogram: Full mel spectrogram array of shape (1, n_mels, time)
            audio_length: Length of original audio in samples
        """
        # Validate shape
        if len(mel_spectrogram.shape) != 3 or mel_spectrogram.shape[0] != 1:
            raise ValueError(f"Expected full mel spectrogram shape (1, n_mels, time), got {mel_spectrogram.shape}")
        
        # Serialize mel spectrogram
        if mel_spectrogram.dtype != np.float32:
            mel_spectrogram = mel_spectrogram.astype(np.float32)
            
        mel_data_bytes = mel_spectrogram.tobytes()
        
        # Extract shape: (1, n_mels, time_frames)
        mel_n_mels = mel_spectrogram.shape[1]  # 128
        mel_time = mel_spectrogram.shape[2]    # time frames
        
        # Compress with zlib (level 6 for good balance)
        compressed_bytes = zlib.compress(mel_data_bytes, level=6)
        compression_ratio = len(mel_data_bytes) / len(compressed_bytes)
        
        try:
            # Insert or replace (atomic operation)
            self.conn.execute("""
                INSERT OR REPLACE INTO mel_spectrograms 
                (item_id, mel_shape_time, mel_shape_mels, mel_data, audio_length)
                VALUES (?, ?, ?, ?, ?)
            """, (item_id, mel_time, mel_n_mels, compressed_bytes, audio_length))
            
            # IMMEDIATE commit - ensures data is saved even if process crashes!
            self.conn.commit()
            
            logger.debug(f"💾 Cache SAVED (FULL, compressed {compression_ratio:.1f}x): {item_id} {mel_spectrogram.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save mel cache for {item_id}: {e}")
            self.conn.rollback()
            raise
        
    def extract_overlapped_segments(self, full_mel: np.ndarray, audio_length: int, 
                                   segment_length: int = 480000, hop_length: int = 240000,
                                   sample_rate: int = 48000, hop_length_stft: int = 480) -> np.ndarray:
        """
        Extract overlapped segments from full mel spectrogram at runtime.
        
        This is the KEY OPTIMIZATION: instead of storing overlapped segments,
        we store the full spectrogram once and extract segments on-demand.
        
        Args:
            full_mel: Full mel spectrogram, shape (1, n_mels, time_frames)
            audio_length: Original audio length in samples
            segment_length: Audio segment length in samples (default: 480000 = 10s at 48kHz)
            hop_length: Audio hop length in samples (default: 240000 = 5s)
            sample_rate: Audio sample rate (default: 48000)
            hop_length_stft: STFT hop length used for mel computation (default: 480)
            
        Returns:
            Segmented mel spectrogram of shape (num_segments, 1, n_mels, segment_time_frames)
        """
        # Calculate time frames per segment
        segment_time_frames = segment_length // hop_length_stft
        hop_time_frames = hop_length // hop_length_stft
        
        # Extract shape info
        n_mels = full_mel.shape[1]
        total_time_frames = full_mel.shape[2]
        
        segments = []
        
        if audio_length <= segment_length:
            # Short audio: pad to segment length
            if total_time_frames < segment_time_frames:
                # Pad with zeros
                padded = np.pad(full_mel, ((0, 0), (0, 0), (0, segment_time_frames - total_time_frames)), 
                               mode='constant', constant_values=0)
                segments.append(padded[:, :, :segment_time_frames])
            else:
                segments.append(full_mel[:, :, :segment_time_frames])
        else:
            # Create overlapping segments
            start_frame = 0
            while start_frame + segment_time_frames <= total_time_frames:
                segment = full_mel[:, :, start_frame:start_frame + segment_time_frames]
                segments.append(segment)
                start_frame += hop_time_frames
            
            # Add final segment if needed
            if start_frame < total_time_frames:
                # Take last segment_time_frames from the end
                final_segment = full_mel[:, :, -segment_time_frames:]
                segments.append(final_segment)
        
        # Stack into batch: (num_segments, 1, n_mels, time)
        segmented_mel = np.stack(segments, axis=0)
        
        logger.debug(f"Extracted {len(segments)} segments from full mel (shape {full_mel.shape} -> {segmented_mel.shape})")
        return segmented_mel
    
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
            'cache_size_gb': total_size_bytes / (1024 * 1024 * 1024),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate
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
    
    def get_averaged_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Compute averaged teacher embedding from segment embeddings (no storage needed).
        
        Args:
            item_id: Song item ID
            
        Returns:
            Averaged embedding array (512-dim) or None if segments not cached
        """
        segment_embeddings = self.get_segment_embeddings(item_id)
        if segment_embeddings is None:
            return None
        
        # Average and normalize
        avg_embedding = np.mean(segment_embeddings, axis=0).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding
    
    def put_segment_embeddings(self, item_id: str, segment_embeddings: list):
        """
        Store per-segment teacher CLAP embeddings for a song (always compressed).
        
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
                "INSERT INTO segment_embeddings (item_id, segment_index, embedding) VALUES (?, ?, ?)",
                (item_id, segment_idx, compressed_bytes)
            )
        
        self.conn.commit()
        logger.debug(f"💾 Cached {len(segment_embeddings)} compressed segment embeddings for {item_id}")
    
    def get_segment_embeddings(self, item_id: str) -> Optional[list]:
        """
        Get per-segment teacher CLAP embeddings from cache (always compressed).
        
        Args:
            item_id: Song item ID
            
        Returns:
            List of embedding arrays (512-dim each) or None if not cached
        """
        cursor = self.conn.execute(
            "SELECT segment_index, embedding FROM segment_embeddings WHERE item_id = ? ORDER BY segment_index",
            (item_id,)
        )
        rows = cursor.fetchall()
        
        if not rows:
            return None
        
        embeddings = []
        for segment_idx, embedding_bytes in rows:
            # Decompress
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
        # Note: Now only uses segment_embeddings since song_embeddings table removed
        cursor = self.conn.execute(
            "SELECT MIN(created_at) FROM segment_embeddings WHERE item_id = ?",
            (item_id,)
        )
        row = cursor.fetchone()
        
        if row is None or row[0] is None:
            return None
        
        return {
            'file_path': 'N/A',  # Not stored in segment_embeddings
            'created_at': row[0]
        }
        
    def clear(self):
        """Clear all cached data."""
        self.conn.execute("DELETE FROM mel_spectrograms")
        self.conn.execute("DELETE FROM segment_embeddings")
        self.conn.commit()
        logger.info("Cleared all mel spectrogram and embedding cache")
        
    def close(self):
        """Close database connection."""
        stats = self.get_stats()
        logger.info(f"Mel cache stats: {stats['total_cached']} items, "
                   f"{stats['cache_size_gb']:.1f}GB, "
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
