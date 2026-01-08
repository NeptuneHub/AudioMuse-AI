#!/usr/bin/env python3
"""
Migrate existing mel_cache.db to use zlib compression.

This script processes the database in batches to avoid memory issues.
It reads uncompressed entries, compresses them with zlib, and updates them in place.
"""

import sqlite3
import zlib
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_to_compressed(db_path: str, batch_size: int = 100):
    """
    Migrate mel cache database to use zlib compression.
    
    Args:
        db_path: Path to mel_spectrograms.db
        batch_size: Number of records to process per batch
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        logger.error(f"❌ Database not found: {db_path}")
        return
    
    logger.info(f"🔄 Starting compression migration for: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    
    # Add compressed column if it doesn't exist
    try:
        conn.execute("ALTER TABLE mel_spectrograms ADD COLUMN compressed INTEGER DEFAULT 0")
        conn.commit()
        logger.info("✅ Added 'compressed' column to table")
    except sqlite3.OperationalError:
        logger.info("📌 'compressed' column already exists")
    
    # Get total count of uncompressed entries
    cursor = conn.execute("SELECT COUNT(*) FROM mel_spectrograms WHERE compressed = 0 OR compressed IS NULL")
    total_uncompressed = cursor.fetchone()[0]
    
    if total_uncompressed == 0:
        logger.info("✅ All entries already compressed! Nothing to do.")
        conn.close()
        return
    
    logger.info(f"📊 Found {total_uncompressed} uncompressed entries")
    
    # Get database size before compression
    cursor = conn.execute("SELECT SUM(LENGTH(mel_data)) FROM mel_spectrograms")
    size_before = cursor.fetchone()[0] or 0
    size_before_gb = size_before / (1024 * 1024 * 1024)
    logger.info(f"💾 Database size before: {size_before_gb:.2f} GB")
    
    # Process in batches
    total_processed = 0
    total_saved_bytes = 0
    
    with tqdm(total=total_uncompressed, desc="Compressing") as pbar:
        while True:
            # Fetch batch of uncompressed entries
            cursor = conn.execute("""
                SELECT item_id, mel_data 
                FROM mel_spectrograms 
                WHERE compressed = 0 OR compressed IS NULL
                LIMIT ?
            """, (batch_size,))
            
            batch = cursor.fetchall()
            
            if not batch:
                break  # No more uncompressed entries
            
            # Process batch
            for item_id, mel_data_bytes in batch:
                try:
                    # Compress with zlib level 6
                    compressed_bytes = zlib.compress(mel_data_bytes, level=6)
                    bytes_saved = len(mel_data_bytes) - len(compressed_bytes)
                    total_saved_bytes += bytes_saved
                    
                    # Update in database
                    conn.execute("""
                        UPDATE mel_spectrograms 
                        SET mel_data = ?, compressed = 1 
                        WHERE item_id = ?
                    """, (compressed_bytes, item_id))
                    
                    total_processed += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to compress {item_id}: {e}")
                    continue
            
            # Commit batch
            conn.commit()
            pbar.update(len(batch))
    
    # Get database size after compression
    cursor = conn.execute("SELECT SUM(LENGTH(mel_data)) FROM mel_spectrograms")
    size_after = cursor.fetchone()[0] or 0
    size_after_gb = size_after / (1024 * 1024 * 1024)
    
    # Calculate savings
    size_saved_gb = (size_before - size_after) / (1024 * 1024 * 1024)
    compression_ratio = size_before / size_after if size_after > 0 else 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Migration complete!")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Processed: {total_processed} entries")
    logger.info(f"💾 Size before: {size_before_gb:.2f} GB")
    logger.info(f"💾 Size after: {size_after_gb:.2f} GB")
    logger.info(f"✨ Space saved: {size_saved_gb:.2f} GB ({compression_ratio:.2f}x compression)")
    logger.info(f"{'='*60}")
    
    # Vacuum database to reclaim space
    logger.info(f"\n🧹 Running VACUUM to reclaim disk space...")
    conn.execute("VACUUM")
    conn.commit()
    logger.info(f"✅ VACUUM complete - disk space reclaimed!")
    
    conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate mel cache to use zlib compression")
    parser.add_argument(
        "--db-path",
        type=str,
        default="/Volumes/audiomuse/student_clap_cache/mel_spectrograms.db",
        help="Path to mel_spectrograms.db"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records to process per batch (default: 100)"
    )
    
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  Mel Cache Compression Migration                         ║
╚═══════════════════════════════════════════════════════════╝

Database: {args.db_path}
Batch size: {args.batch_size}

This will compress all uncompressed mel spectrograms in place.
The database will be modified - make a backup if desired.

Press Ctrl+C to cancel, or wait 5 seconds to continue...
""")
    
    import time
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n❌ Migration cancelled")
        exit(0)
    
    migrate_to_compressed(args.db_path, args.batch_size)
