"""
Training Dataset for Student CLAP

Loads audio files, segments them, and pairs with teacher embeddings
for knowledge distillation training.
"""

import os
import sys
import logging
import numpy as np
import torch
import torchaudio
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from student_clap.data.database_loader import DatabaseLoader
from student_clap.data.jellyfin_downloader import JellyfinDownloader
from student_clap.data.mel_cache import MelSpectrogramCache
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
                 validation_split: float = 0.15,
                 epoch: int = 1):
        """
        Initialize dataset.
        
        Args:
            config: Full configuration dict
            split: 'train' or 'val'
            validation_split: Fraction of data for validation
            epoch: Current epoch number (1 = cache building, 2+ = cache reuse only)
        """
        self.config = config
        self.split = split
        self.validation_split = validation_split
        self.epoch = epoch
        
        # Extract configs
        self.db_config = config['database']
        self.jellyfin_config = config['jellyfin']
        self.audio_config = config['audio']
        self.paths_config = config['paths']
        self.dataset_config = config.get('dataset', {})
        
        # Initialize loaders
        self.db_loader = DatabaseLoader(self.db_config)
        
        # Jellyfin downloader (temporary audio storage - files deleted after processing)
        self.jellyfin_downloader = JellyfinDownloader(
            self.jellyfin_config,
            cache_dir=self.paths_config['audio_cache']
        )
        
        # Initialize mel spectrogram cache (MASSIVE speedup for epoch 2+!)
        mel_cache_path = self.paths_config.get('mel_cache', './cache/mel_spectrograms.db')
        self.mel_cache = MelSpectrogramCache(mel_cache_path)
        logger.info(f"🔧 MEL CACHE PATH: {mel_cache_path}")
        logger.info(f"🔧 MEL CACHE: No size limit - will cache all songs")
        
        # Check existing cache FIRST (to prioritize cached songs in sampling)
        cached_item_ids = set(self.mel_cache.get_cached_item_ids())
        
        # Load embeddings (with optional balanced sampling)
        logger.info("Loading embeddings from database...")
        sample_size = self.dataset_config.get('sample_size')
        balanced_genres = self.dataset_config.get('balanced_genres')
        
        if sample_size and balanced_genres:
            logger.info(f"🎯 Using balanced genre sampling: {sample_size} songs across {len(balanced_genres)} genres")
            if cached_item_ids:
                logger.info(f"   📦 Will PRIORITIZE {len(cached_item_ids)} already cached songs to avoid re-downloading!")
            all_items = self.db_loader.load_embeddings(
                sample_size=sample_size,
                balanced_genres=balanced_genres,
                cached_item_ids=cached_item_ids  # Pass cached IDs for prioritization
            )
        else:
            logger.info("📚 Loading all available embeddings (no sampling)")
            all_items = self.db_loader.load_embeddings()
        
        logger.info(f"Loaded {len(all_items)} total items")
        if cached_item_ids:
            cache_size_gb = self.mel_cache.get_cache_size_gb()
            logger.info(f"📦 Found existing mel cache: {len(cached_item_ids)} songs, {cache_size_gb:.1f}GB")
            logger.info(f"🔨 CACHE BUILDING MODE: Will cache ALL songs (no size limit)")
            logger.info(f"   Already cached: {len(cached_item_ids)} songs ({cache_size_gb:.1f}GB)")
            logger.info(f"   Total songs available: {len(all_items)}")
        else:
            logger.info(f"🚀 Building mel cache from scratch (no size limit)")
        
        # Split into train/val
        np.random.seed(42)  # Reproducible split
        indices = np.random.permutation(len(all_items))
        split_idx = int(len(all_items) * (1 - validation_split))
        
        if split == 'train':
            self.items = [all_items[i] for i in indices[:split_idx]]
        else:
            self.items = [all_items[i] for i in indices[split_idx:]]
            
        logger.info(f"Dataset split '{split}': {len(self.items)} items")
        
        # 🔍 VALIDATE: Check which songs can actually be accessed
        if epoch == 1:
            logger.info(f"🔍 Validating song availability in Jellyfin for {split} split...")
            valid_items = []
            cached_count = 0
            available_count = 0
            failed_items = []
            
            # Get actual cached item IDs directly from database (most reliable)
            cached_item_ids = set(self.mel_cache.get_cached_item_ids())
            logger.info(f"   📊 Mel cache has {len(cached_item_ids)} songs total")
            
            for item in self.items:
                item_id = item['item_id']
                # Check if cached using direct set lookup
                if item_id in cached_item_ids:
                    valid_items.append(item)
                    cached_count += 1
                else:
                    # In epoch 1: assume all non-cached songs will be downloaded (skip expensive checks)
                    # Jellyfin check is too slow and unreliable for thousands of songs
                    valid_items.append(item)
                    available_count += 1
            
            logger.info(f"✅ {split} split ready: {len(valid_items)} songs")
            logger.info(f"   📦 Already cached: {cached_count}")
            logger.info(f"   🆕 Will download in epoch 1: {available_count}")
            
            self.items = valid_items
        
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
        
    def iterate_batches_streaming(self, batch_size: int, shuffle: bool = True):
        """
        STREAMING batch iteration - downloads ONLY current batch.
        
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
            
            logger.info(f"📥 Loading batch {start_idx//batch_size + 1}: songs {start_idx+1}-{end_idx}")
            
            # Prepare batch items
            batch_items = [self.items[idx] for idx in batch_indices]
            
            # Check cache status and categorize
            import threading
            
            tasks_to_download = []
            tasks_to_compress = []
            tasks_cached = []
            
            for item in batch_items:
                result = self.mel_cache.get_with_compression_status(item['item_id'])
                if result is None:
                    if self.epoch == 1:
                        tasks_to_download.append(item)
                else:
                    mel_data, is_compressed, original_bytes = result
                    if is_compressed:
                        tasks_cached.append((item, mel_data))
                    else:
                        tasks_to_compress.append((item, mel_data, original_bytes))
            
            logger.info(f"   📊 {len(tasks_cached)} compressed, "
                       f"{len(tasks_to_compress)} uncompressed, "
                       f"{len(tasks_to_download)} to download")
            
            batch = []
            
            # 1. Process cached (fast, sequential is fine)
            for item, mel_data in tasks_cached:
                mel_tensor = torch.from_numpy(mel_data).float()
                batch.append({
                    'item_id': item['item_id'],
                    'title': item['title'],
                    'author': item['author'],
                    'audio_path': 'cached',
                    'audio_segments': mel_tensor,
                    'teacher_embedding': item['embedding'],
                    'num_segments': len(mel_data)
                })
            
            # 2. Process uncompressed (spawn compression threads)
            for item, mel_data, original_bytes in tasks_to_compress:
                # Spawn compression thread
                threading.Thread(
                    target=self.mel_cache.compress_and_update,
                    args=(item['item_id'], original_bytes),
                    daemon=True
                ).start()
                
                mel_tensor = torch.from_numpy(mel_data).float()
                batch.append({
                    'item_id': item['item_id'],
                    'title': item['title'],
                    'author': item['author'],
                    'audio_path': 'cached (compressing)',
                    'audio_segments': mel_tensor,
                    'teacher_embedding': item['embedding'],
                    'num_segments': len(mel_data)
                })
            
            # 3. Download in parallel (use existing download_batch logic)
            if tasks_to_download:
                download_ids = [item['item_id'] for item in tasks_to_download]
                download_results = self.jellyfin_downloader.download_batch(download_ids, max_workers=batch_size)
                
                # Process downloads sequentially (to avoid nested parallelism)
                for item in tasks_to_download:
                    audio_path = download_results.get(item['item_id'])
                    if not audio_path:
                        continue
                    
                    try:
                        import torchaudio
                        import os
                        from student_clap.preprocessing.audio_segmentation import segment_audio
                        from student_clap.preprocessing.mel_spectrogram import compute_mel_spectrogram_batch
                        
                        audio_tensor, sr = torchaudio.load(audio_path)
                        if audio_tensor.shape[0] > 1:
                            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
                        if sr != self.audio_config['sample_rate']:
                            resampler = torchaudio.transforms.Resample(sr, self.audio_config['sample_rate'])
                            audio_tensor = resampler(audio_tensor)
                        
                        audio_data = audio_tensor.squeeze(0).numpy()
                        segments = segment_audio(audio_data, self.audio_config['sample_rate'],
                                                self.audio_config['segment_length'], self.audio_config['hop_length'])
                        
                        segments_array = np.array(segments) if isinstance(segments, list) else segments
                        mel_specs = compute_mel_spectrogram_batch(
                            segments_array, sr=self.audio_config['sample_rate'],
                            n_mels=self.audio_config['n_mels'], n_fft=self.audio_config['n_fft'],
                            hop_length=self.audio_config['hop_length_stft'],
                            fmin=self.audio_config['fmin'], fmax=self.audio_config['fmax']
                        )
                        
                        self.mel_cache.put(item['item_id'], mel_specs)
                        
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                        
                        mel_tensor = torch.from_numpy(mel_specs).float()
                        batch.append({
                            'item_id': item['item_id'],
                            'title': item['title'],
                            'author': item['author'],
                            'audio_path': audio_path,
                            'audio_segments': mel_tensor,
                            'teacher_embedding': item['embedding'],
                            'num_segments': len(mel_specs)
                        })
                    except Exception as e:
                        logger.error(f"Failed to process {item['title']}: {e}")
            
            logger.info(f"   ✅ Batch ready: {len(batch)} samples")
            
            if len(batch) == 0:
                continue
                
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
        
        # Add mel cache stats
        mel_cache_stats = self.mel_cache.get_stats()
        stats.update({
            'mel_cache_items': mel_cache_stats['total_cached'],
            'mel_cache_size_mb': mel_cache_stats['cache_size_mb'],
            'mel_cache_hit_rate': mel_cache_stats['hit_rate_percent']
        })
        
        return stats
        
    def _segment_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Segment audio using the same strategy as teacher CLAP.
        This is critical for compatibility - must match exactly.
        
        Args:
            audio_data: Raw audio samples at 48kHz
            
        Returns:
            segments: Array of shape (num_segments, segment_length) 
                     where segment_length = 480,000 (10 seconds at 48kHz)
        """
        sample_rate = self.audio_config['sample_rate']  # 48000
        segment_length = self.audio_config['segment_length']  # 480000 (10s)
        hop_length = self.audio_config['hop_length']  # 240000 (5s)
        
        total_length = len(audio_data)
        
        # If audio is shorter than 10 seconds, pad to 10 seconds
        if total_length <= segment_length:
            padded = np.pad(audio_data, (0, segment_length - total_length), mode='constant')
            return padded.reshape(1, -1)  # (1, segment_length)
        
        # For longer audio: create overlapping segments (10s segments, 5s hop)
        segments = []
        for start in range(0, total_length - segment_length + 1, hop_length):
            segment = audio_data[start:start + segment_length]
            segments.append(segment)
        
        # Add final segment if needed (to capture the end of the audio)
        last_start = len(segments) * hop_length
        if last_start < total_length:
            final_segment = audio_data[-segment_length:]
            segments.append(final_segment)
        
        return np.array(segments)  # (num_segments, segment_length)
        
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
