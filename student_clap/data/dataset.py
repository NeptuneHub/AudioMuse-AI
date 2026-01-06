"""
Training Dataset for Student CLAP

Loads local audio files from FMA, analyzes with CLAP for teacher embeddings,
and pairs with student mel-spectrograms for knowledge distillation training.
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

from student_clap.data.local_song_loader import LocalSongLoader
from student_clap.data.clap_embedder import CLAPEmbedder
from student_clap.data.mel_cache import MelSpectrogramCache
from student_clap.preprocessing.audio_segmentation import (
    segment_audio, SAMPLE_RATE, SEGMENT_LENGTH, HOP_LENGTH
)
from student_clap.preprocessing.mel_spectrogram import compute_mel_spectrogram_batch

logger = logging.getLogger(__name__)


class StudentCLAPDataset:
    """Dataset for training student CLAP model using local FMA files."""
    
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
        self.audio_config = config['audio']
        self.paths_config = config['paths']
        self.dataset_config = config.get('dataset', {})
        
        # Initialize local song loader (replaces database + Jellyfin)
        fma_path = self.dataset_config['fma_path']
        self.song_loader = LocalSongLoader(fma_path)
        
        # Initialize CLAP embedder for teacher embeddings
        teacher_model_path = self.paths_config['teacher_model']
        self.clap_embedder = CLAPEmbedder(teacher_model_path)
        
        # Initialize mel spectrogram cache (stores both mel specs and embeddings)
        mel_cache_path = self.paths_config.get('mel_cache', './cache/mel_spectrograms.db')
        self.mel_cache = MelSpectrogramCache(mel_cache_path)
        logger.info(f"🔧 MEL CACHE PATH: {mel_cache_path}")
        logger.info(f"🔧 MEL CACHE: No size limit - will cache all songs")
        
        # Load songs from local FMA directory
        logger.info("Loading songs from local FMA directory...")
        sample_size = self.dataset_config.get('sample_size')
        all_songs = self.song_loader.load_songs(limit=sample_size)
        
        # Show big label with song count
        logger.info("="*80)
        logger.info("="*80)
        if sample_size == 0 or sample_size is None:
            logger.info(f"🎵 LOADING ALL SONGS: {len(all_songs)} AUDIO FILES FOUND 🎵")
        else:
            logger.info(f"🎵 LOADING SAMPLE: {len(all_songs)} AUDIO FILES (limited from dataset) 🎵")
        logger.info("="*80)
        logger.info("="*80)
        
        # Check what's already cached
        cached_item_ids = set(self.mel_cache.get_cached_item_ids())
        if cached_item_ids:
            cache_size_gb = self.mel_cache.get_cache_size_gb()
            logger.info(f"📦 Found existing mel cache: {len(cached_item_ids)} songs, {cache_size_gb:.1f}GB")
        
        # Split into train/val
        np.random.seed(42)  # Reproducible split
        indices = np.random.permutation(len(all_songs))
        split_idx = int(len(all_songs) * (1 - validation_split))
        
        if split == 'train':
            self.items = [all_songs[i] for i in indices[:split_idx]]
        else:
            self.items = [all_songs[i] for i in indices[split_idx:]]
            
        logger.info(f"Dataset split '{split}': {len(self.items)} items")
        
    def __len__(self) -> int:
        """Return number of items in dataset."""
        return len(self.items)
    
    def iterate_batches_streaming(self, batch_size: int, shuffle: bool = True):
        """
        STREAMING batch iteration - processes songs from local FMA directory.
        
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
            
            tasks_to_process = []
            tasks_to_compress = []
            tasks_cached = []
            
            # Track cache statistics
            mel_cached_count = 0
            embedding_cached_count = 0
            
            for item in batch_items:
                mel_result = self.mel_cache.get_with_compression_status(item['item_id'])
                has_embedding = self.mel_cache.has_embedding(item['item_id'])
                
                # Update cache counts
                if mel_result is not None:
                    mel_cached_count += 1
                if has_embedding:
                    embedding_cached_count += 1
                
                # Decide what to do with this item
                if mel_result is None:
                    # No mel cache - need to process
                    tasks_to_process.append(item)
                else:
                    mel_data, is_compressed, original_bytes = mel_result
                    if is_compressed:
                        tasks_cached.append((item, mel_data))
                    else:
                        tasks_to_compress.append((item, mel_data, original_bytes))
            
            logger.info(f"   📊 Cache Status: Mel={mel_cached_count}/{len(batch_items)} | Embedding={embedding_cached_count}/{len(batch_items)}")
            logger.info(f"   📦 Processing: {len(tasks_cached)} fully cached, "
                       f"{len(tasks_to_compress)} need compression, "
                       f"{len(tasks_to_process)} need analysis")
            
            batch = []
            
            # 1. Process cached (fast, sequential is fine)
            for item, mel_data in tasks_cached:
                # Get teacher embedding from cache
                teacher_embedding = self.mel_cache.get_embedding(item['item_id'])
                if teacher_embedding is None:
                    logger.warning(f"Missing teacher embedding for {item['item_id']}, will reprocess")
                    tasks_to_process.append(item)
                    continue
                    
                # Copy to make array writable for PyTorch
                mel_tensor = torch.from_numpy(mel_data.copy()).float()
                batch.append({
                    'item_id': item['item_id'],
                    'title': item['title'],
                    'author': item.get('author', 'Unknown'),
                    'audio_path': 'cached',
                    'audio_segments': mel_tensor,
                    'teacher_embedding': teacher_embedding,
                    'num_segments': len(mel_data)
                })
            
            # 2. Process uncompressed (spawn compression threads)
            for item, mel_data, original_bytes in tasks_to_compress:
                # Get teacher embedding from cache
                teacher_embedding = self.mel_cache.get_embedding(item['item_id'])
                if teacher_embedding is None:
                    logger.warning(f"Missing teacher embedding for {item['item_id']}, will reprocess")
                    tasks_to_process.append(item)
                    continue
                
                # Spawn compression thread
                threading.Thread(
                    target=self.mel_cache.compress_and_update,
                    args=(item['item_id'], original_bytes),
                    daemon=True
                ).start()
                
                # Copy to make array writable for PyTorch
                mel_tensor = torch.from_numpy(mel_data.copy()).float()
                batch.append({
                    'item_id': item['item_id'],
                    'title': item['title'],
                    'author': item.get('author', 'Unknown'),
                    'audio_path': 'cached (compressing)',
                    'audio_segments': mel_tensor,
                    'teacher_embedding': teacher_embedding,
                    'num_segments': len(mel_data)
                })
            
            # 3. Process new songs from local files
            if tasks_to_process:
                from student_clap.preprocessing.audio_segmentation import segment_audio
                from student_clap.preprocessing.mel_spectrogram import compute_mel_spectrogram_batch
                
                for item in tasks_to_process:
                    audio_path = item['file_path']
                    
                    try:
                        # Check what's already cached
                        cached_mel = self.mel_cache.get(item['item_id'])
                        cached_embedding = self.mel_cache.get_embedding(item['item_id'])
                        
                        # Get teacher embedding (compute if not cached)
                        if cached_embedding is not None:
                            teacher_embedding = cached_embedding
                        else:
                            teacher_embedding, duration_sec, num_segments = self.clap_embedder.analyze_audio(audio_path)
                            if teacher_embedding is None:
                                logger.error(f"CLAP analysis failed for {item['title']}")
                                continue
                            # Cache the embedding
                            self.mel_cache.put_embedding(item['item_id'], teacher_embedding, audio_path)
                        
                        # Get mel spectrograms (compute if not cached)
                        if cached_mel is not None:
                            mel_specs = cached_mel
                        else:
                            # Load audio at 48kHz
                            import torchaudio
                            audio_tensor, sr = torchaudio.load(audio_path)
                            if audio_tensor.shape[0] > 1:
                                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
                            if sr != self.audio_config['sample_rate']:
                                resampler = torchaudio.transforms.Resample(sr, self.audio_config['sample_rate'])
                                audio_tensor = resampler(audio_tensor)
                            
                            audio_data = audio_tensor.squeeze(0).numpy()
                            
                            # Segment audio (10s segments, 5s hop)
                            segments = segment_audio(
                                audio_data, 
                                self.audio_config['sample_rate'],
                                self.audio_config['segment_length'], 
                                self.audio_config['hop_length']
                            )
                            
                            # Compute mel-spectrograms for all segments
                            segments_array = np.array(segments) if isinstance(segments, list) else segments
                            mel_specs = compute_mel_spectrogram_batch(
                                segments_array, 
                                sr=self.audio_config['sample_rate'],
                                n_mels=self.audio_config['n_mels'], 
                                n_fft=self.audio_config['n_fft'],
                                hop_length=self.audio_config['hop_length_stft'],
                                fmin=self.audio_config['fmin'], 
                                fmax=self.audio_config['fmax']
                            )
                            
                            # Cache the mel spectrograms
                            self.mel_cache.put(item['item_id'], mel_specs)
                        
                        mel_tensor = torch.from_numpy(mel_specs).float()
                        batch.append({
                            'item_id': item['item_id'],
                            'title': item['title'],
                            'author': item.get('author', 'Unknown'),
                            'audio_path': audio_path,
                            'audio_segments': mel_tensor,
                            'teacher_embedding': teacher_embedding,
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
        self.mel_cache.close()


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
    
    # Test batch iteration
    print(f"\nTesting streaming batch iteration (batch_size=2):")
    batch_count = 0
    for batch in dataset.iterate_batches_streaming(batch_size=2, shuffle=False):
        batch_count += 1
        print(f"\n  Batch {batch_count}: {len(batch)} samples")
        
        # Show first sample in batch
        if batch:
            sample = batch[0]
            print(f"    First sample:")
            print(f"      Item ID: {sample['item_id']}")
            print(f"      Title: {sample['title']}")
            print(f"      Author: {sample['author']}")
            print(f"      Num segments: {sample['num_segments']}")
            print(f"      Mel-specs shape: {sample['audio_segments'].shape}")
            print(f"      Teacher embedding shape: {sample['teacher_embedding'].shape}")
            print(f"      Teacher embedding norm: {np.linalg.norm(sample['teacher_embedding']):.4f}")
        
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
