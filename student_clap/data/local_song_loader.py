"""
Local Song Loader for Student CLAP Training

Loads audio files from local FMA directory.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LocalSongLoader:
    """Loads audio files from local FMA directory."""
    
    def __init__(self, fma_path: str):
        """
        Initialize local song loader.
        
        Args:
            fma_path: Path to FMA directory (e.g., /Users/guidocolangiuli/Music/FMA)
        """
        self.fma_path = Path(fma_path)
        if not self.fma_path.exists():
            raise ValueError(f"FMA path does not exist: {fma_path}")
        
        logger.info(f"Local song loader initialized: {self.fma_path}")
        
    def load_songs(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Load audio files from FMA directory.
        
        Args:
            limit: Optional limit on number of songs to load
            
        Returns:
            List of dicts with keys:
                - item_id: Song ID (derived from filename/path)
                - file_path: Full path to audio file
                - title: Song title (filename without extension)
        """
        logger.info(f"Scanning for audio files in {self.fma_path}...")
        
        # Supported audio formats
        audio_extensions = {'.mp3', '.flac', '.wav', '.m4a', '.ogg', '.opus'}
        
        # Find all audio files recursively (sorted for consistency)
        audio_files = []
        for root, dirs, files in os.walk(self.fma_path):
            # Sort directories and files for deterministic order
            dirs.sort()
            files.sort()
            for file in files:
                if Path(file).suffix.lower() in audio_extensions:
                    full_path = Path(root) / file
                    audio_files.append(full_path)
        
        # Sort final list for consistent ordering across runs
        audio_files.sort()
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Convert to dict format
        songs = []
        for audio_file in audio_files:
            # Generate unique item_id from relative path
            rel_path = audio_file.relative_to(self.fma_path)
            item_id = str(rel_path).replace('/', '_').replace('\\', '_')
            
            # Remove extension from item_id
            item_id = Path(item_id).stem
            
            songs.append({
                'item_id': item_id,
                'file_path': str(audio_file),
                'title': audio_file.stem,  # Filename without extension
            })
        
        # Apply limit if specified (0 means no limit)
        if limit is not None and limit > 0:
            songs = songs[:limit]
            logger.info(f"Limited to {len(songs)} songs")
        
        logger.info(f"Loaded {len(songs)} songs from local FMA directory")
        return songs
