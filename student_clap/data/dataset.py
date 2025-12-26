"""
Training Dataset for Student CLAP

Loads audio files, segments them, and pairs with teacher embeddings
for knowledge distillation training.
"""

import os
import sys
import logging
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from student_clap.data.database_loader import DatabaseLoader
from student_clap.data.jellyfin_downloader import JellyfinDownloader
from student_clap.preprocessing.audio_segmentation import (
    segment_audio, SAMPLE_RATE, SEGMENT_LENGTH, HOP_LENGTH
)
from student_clap.preprocessing.mel_spectrogram import compute_mel_spectrogram_batch

logger = logging.getLogger(__name__)


class StudentCLAPDataset:
    """Dataset for training student CLAP model."""
    
    def __init__(self, 
                 config: dict,
                 split: str = 'train',
                 validation_split: float = 0.15):
        """
        Initialize dataset.
        
        Args:
            config: Full configuration dict
            split: 'train' or 'val'
            validation_split: Fraction of data for validation
        """
        self.config = config
        self.split = split
        self.validation_split = validation_split
        
        # Extract configs
        self.db_config = config['database']
        self.jellyfin_config = config['jellyfin']
        self.audio_config = config['audio']
        self.paths_config = config['paths']
        
        # Initialize loaders
        self.db_loader = DatabaseLoader(self.db_config)
        self.jellyfin_downloader = JellyfinDownloader(
            self.jellyfin_config,
            cache_dir=self.paths_config['audio_cache']
        )
        
        # Load embeddings
        logger.info("Loading embeddings from database...")
        all_items = self.db_loader.load_embeddings()
        logger.info(f"Loaded {len(all_items)} total items")
        
        # Split into train/val
        np.random.seed(42)  # Reproducible split
        indices = np.random.permutation(len(all_items))
        split_idx = int(len(all_items) * (1 - validation_split))
        
        if split == 'train':
            self.items = [all_items[i] for i in indices[:split_idx]]
        else:
            self.items = [all_items[i] for i in indices[split_idx:]]
            
        logger.info(f"Dataset split '{split}': {len(self.items)} items")
        
    def __len__(self) -> int:
        """Return number of items in dataset."""
        return len(self.items)
        
    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with keys:
                - item_id: Song ID
                - title: Song title
                - author: Song artist
                - audio_path: Path to cached audio file
                - audio_segments: List of audio segments (each 10s)
                - mel_spectrograms: Batch of mel-specs (num_segments, time, mels)
                - teacher_embedding: 512-dim teacher embedding
            Returns None if loading fails
        """
        item = self.items[idx]
        item_id = item['item_id']
        
        try:
            # Download audio (with caching)
            audio_path = self.jellyfin_downloader.download(item_id)
            if audio_path is None:
                logger.error(f"Failed to download audio for {item_id}")
                return None
                
            # Load audio at 48kHz
            audio_data, sr = librosa.load(
                audio_path,
                sr=self.audio_config['sample_rate'],
                mono=True
            )
            
            # Quantize to int16 and back (match teacher preprocessing)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767.0).astype(np.int16)
            audio_data = (audio_data / 32767.0).astype(np.float32)
            
            # Segment audio (10s segments, 5s hop)
            segments = segment_audio(
                audio_data,
                sample_rate=self.audio_config['sample_rate'],
                segment_length=self.audio_config['segment_length'],
                hop_length=self.audio_config['hop_length']
            )
            
            # Compute mel-spectrograms for all segments
            mel_specs = compute_mel_spectrogram_batch(
                segments,
                sr=self.audio_config['sample_rate'],
                n_mels=self.audio_config['n_mels'],
                n_fft=self.audio_config['n_fft'],
                hop_length=self.audio_config['hop_length_stft'],
                fmin=self.audio_config['fmin'],
                fmax=self.audio_config['fmax']
            )
            
            return {
                'item_id': item_id,
                'title': item['title'],
                'author': item['author'],
                'audio_path': audio_path,
                'audio_segments': segments,
                'mel_spectrograms': mel_specs,
                'teacher_embedding': item['embedding'],
                'num_segments': len(segments)
            }
            
        except Exception as e:
            logger.error(f"Failed to load item {item_id}: {e}")
            return None
            
    def get_batch(self, batch_size: int) -> List[Dict]:
        """
        Get a batch of samples.
        
        Args:
            batch_size: Number of samples per batch
            
        Returns:
            List of sample dicts
        """
        indices = np.random.choice(len(self), batch_size, replace=False)
        batch = []
        
        for idx in indices:
            sample = self[idx]
            if sample is not None:
                batch.append(sample)
                
        return batch
        
    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """
        Iterate over dataset in batches.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            
        Yields:
            Batch of samples (list of dicts)
        """
        indices = np.arange(len(self))
        
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, len(self), batch_size):
            end_idx = min(start_idx + batch_size, len(self))
            batch_indices = indices[start_idx:end_idx]
            
            batch = []
            for idx in batch_indices:
                sample = self[idx]
                if sample is not None:
                    batch.append(sample)
                    
            if batch:  # Only yield non-empty batches
                yield batch
                
    def get_dataset_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dict with dataset statistics
        """
        stats = {
            'split': self.split,
            'total_items': len(self.items),
            'sample_rate': self.audio_config['sample_rate'],
            'segment_length': self.audio_config['segment_length'],
            'hop_length': self.audio_config['hop_length'],
            'embedding_dim': 512
        }
        
        # Add downloader stats
        downloader_stats = self.jellyfin_downloader.get_stats()
        stats.update({
            'cached_files': downloader_stats['cached_files'],
            'cache_size_mb': downloader_stats['cache_size_mb']
        })
        
        return stats
        
    def close(self):
        """Clean up resources."""
        self.db_loader.close()


def collate_batch(batch: List[Dict]) -> Dict:
    """
    Collate a batch of samples into tensors.
    
    Since songs have different numbers of segments, we process each song
    independently during training (not true batching across songs).
    
    Args:
        batch: List of sample dicts
        
    Returns:
        Dict with collated data (still as lists, not tensors)
    """
    # Extract common fields
    item_ids = [item['item_id'] for item in batch]
    titles = [item['title'] for item in batch]
    authors = [item['author'] for item in batch]
    
    # Keep segment-level data as lists (varying lengths)
    mel_spectrograms = [item['mel_spectrograms'] for item in batch]
    teacher_embeddings = np.stack([item['teacher_embedding'] for item in batch])
    num_segments = [item['num_segments'] for item in batch]
    
    return {
        'item_ids': item_ids,
        'titles': titles,
        'authors': authors,
        'mel_spectrograms': mel_spectrograms,  # List of arrays
        'teacher_embeddings': teacher_embeddings,  # (batch_size, 512)
        'num_segments': num_segments
    }


if __name__ == '__main__':
    """Test dataset functionality."""
    import yaml
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test dataset')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to test')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Expand environment variables in config
    def expand_env_vars(cfg):
        """Recursively expand environment variables in config."""
        if isinstance(cfg, dict):
            return {k: expand_env_vars(v) for k, v in cfg.items()}
        elif isinstance(cfg, str) and cfg.startswith('${') and cfg.endswith('}'):
            env_var = cfg[2:-1]
            return os.environ.get(env_var, cfg)
        else:
            return cfg
    
    config = expand_env_vars(config)
    
    # Create dataset
    print("Creating dataset...")
    dataset = StudentCLAPDataset(config, split='train')
    
    # Print stats
    stats = dataset.get_dataset_stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test loading samples
    print(f"\nTesting {args.num_samples} samples:")
    for i in range(min(args.num_samples, len(dataset))):
        print(f"\nSample {i}:")
        sample = dataset[i]
        
        if sample:
            print(f"  Item ID: {sample['item_id']}")
            print(f"  Title: {sample['title']}")
            print(f"  Author: {sample['author']}")
            print(f"  Num segments: {sample['num_segments']}")
            print(f"  Mel-specs shape: {sample['mel_spectrograms'].shape}")
            print(f"  Teacher embedding shape: {sample['teacher_embedding'].shape}")
            print(f"  Teacher embedding norm: {np.linalg.norm(sample['teacher_embedding']):.4f}")
        else:
            print(f"  ✗ Failed to load")
    
    # Test batch iteration
    print(f"\nTesting batch iteration (batch_size=2):")
    batch_count = 0
    for batch in dataset.iterate_batches(batch_size=2):
        batch_count += 1
        print(f"  Batch {batch_count}: {len(batch)} samples")
        if batch_count >= 3:
            break
    
    # Test collation
    if batch:
        print(f"\nTesting batch collation:")
        collated = collate_batch(batch)
        print(f"  Item IDs: {len(collated['item_ids'])}")
        print(f"  Teacher embeddings shape: {collated['teacher_embeddings'].shape}")
        print(f"  Num segments: {collated['num_segments']}")
    
    dataset.close()
    print("\n✓ Dataset test complete")
