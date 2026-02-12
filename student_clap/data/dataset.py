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
                 epoch: int = 1,
                 **kwargs):
        """
        Initialize dataset.

        Args:
            config: Full configuration dict
            split: 'train' or 'val' (loads from fma_path or validation_path respectively)
            epoch: Current epoch number (1 = cache building, 2+ = cache reuse only)
        """
        self.config = config
        self.split = split
        self.epoch = epoch
        
        # Extract configs
        self.audio_config = config['audio']
        self.paths_config = config['paths']
        self.dataset_config = config.get('dataset', {})
        
        # Initialize local song loader from the appropriate directory
        if split == 'train':
            data_path = self.dataset_config['fma_path']
        else:
            data_path = self.dataset_config['validation_path']
        self.song_loader = LocalSongLoader(data_path)

        # Initialize CLAP embedder for teacher embeddings
        teacher_model_path = self.paths_config['teacher_model']
        self.clap_embedder = CLAPEmbedder(teacher_model_path)

        # Initialize mel spectrogram cache (stores both mel specs and embeddings)
        mel_cache_path = self.paths_config.get('mel_cache', './cache/mel_spectrograms.db')
        self.mel_cache = MelSpectrogramCache(mel_cache_path)
        logger.info(f"🔧 MEL CACHE PATH: {mel_cache_path}")
        logger.info(f"🔧 MEL CACHE: No size limit - will cache all songs")
        # Log teacher embedding cache setting
        tcache = self.config.get('training', {}).get('use_teacher_embedding_cache', True)
        logger.info(f"🔧 Teacher embedding cache enabled: {tcache}")

        # Load songs from directory
        logger.info(f"Loading songs for '{split}' from {data_path}...")
        sample_size = self.dataset_config.get('sample_size')
        self.items = self.song_loader.load_songs(limit=sample_size)

        # Show big label with song count
        logger.info("="*80)
        logger.info("="*80)
        if sample_size == 0 or sample_size is None:
            logger.info(f"🎵 [{split.upper()}] LOADING ALL SONGS: {len(self.items)} AUDIO FILES FOUND 🎵")
        else:
            logger.info(f"🎵 [{split.upper()}] LOADING SAMPLE: {len(self.items)} AUDIO FILES (limited from dataset) 🎵")
        logger.info("="*80)
        logger.info("="*80)

        # Check cache size without loading all IDs into memory (memory optimization!)
        cache_size_gb = self.mel_cache.get_cache_size_gb()
        if cache_size_gb > 0:
            # Count cached items efficiently without loading all IDs
            cursor = self.mel_cache.conn.execute("SELECT COUNT(*) FROM mel_spectrograms")
            cached_count = cursor.fetchone()[0]
            logger.info(f"📦 Found existing mel cache: {cached_count} songs, {cache_size_gb:.1f}GB")

        logger.info(f"Dataset '{split}': {len(self.items)} items from {data_path}")
        
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
            tasks_to_process = []
            tasks_cached = []
            use_teacher_emb_cache = self.config.get('training', {}).get('use_teacher_embedding_cache', True)

            # Track cache statistics
            mel_cached_count = 0
            embedding_cached_count = 0

            for item in batch_items:
                mel_result = self.mel_cache.get_with_audio_length(item['item_id'])
                # Skip embedding check when cache is off — embeddings will be recomputed
                if use_teacher_emb_cache or self.split != 'train':
                    has_embedding = self.mel_cache.has_segment_embeddings(item['item_id'])
                else:
                    has_embedding = True  # don't care, will recompute

                # Update cache counts
                if mel_result is not None:
                    mel_cached_count += 1
                if has_embedding:
                    embedding_cached_count += 1

                # Categorize items
                if mel_result is None or not has_embedding:
                    # Need to process from scratch
                    tasks_to_process.append(item)
                else:
                    # Fully cached (or mel cached + embeddings will be recomputed)
                    tasks_cached.append((item, mel_result))
            
            logger.info(f"   📊 Cache Status: Mel={mel_cached_count}/{len(batch_items)} | Embedding={embedding_cached_count}/{len(batch_items)}")
            logger.info(f"   📦 Processing: {len(tasks_cached)} fully cached, "
                       f"{len(tasks_to_process)} need analysis")
            
            batch = []
            
            # 1. Process cached (extract segments from full mel at runtime)
            for item, mel_result in tasks_cached:
                full_mel, audio_length = mel_result
                
                # Get teacher embeddings from cache (skip if cache off — will recompute after augmentation)
                if use_teacher_emb_cache or self.split != 'train':
                    teacher_segment_embeddings = self.mel_cache.get_segment_embeddings(item['item_id'])
                    if teacher_segment_embeddings is not None:
                        avg_emb = np.mean(teacher_segment_embeddings, axis=0).astype(np.float32)
                        norm = np.linalg.norm(avg_emb)
                        teacher_embedding = avg_emb / norm if norm > 0 else avg_emb
                    else:
                        teacher_embedding = None
                else:
                    teacher_segment_embeddings = None
                    teacher_embedding = None
                
                # Extract overlapped segments from full mel spectrogram at runtime
                mel_specs = self.mel_cache.extract_overlapped_segments(
                    full_mel,
                    audio_length=audio_length,
                    segment_length=self.audio_config['segment_length'],
                    hop_length=self.audio_config['hop_length'],
                    sample_rate=self.audio_config['sample_rate'],
                    hop_length_stft=self.audio_config['hop_length_stft']
                )
                
                # --- Gold standard spectrogram augmentation ---
                mel_aug = mel_specs.copy()
                augmentation_enabled = self.config.get('training', {}).get('augmentation_enabled', True)
                if self.split == 'train' and augmentation_enabled:
                    # Random gain
                    gain = np.random.uniform(0.8, 1.2)
                    mel_aug *= gain
                    # Additive noise
                    if np.random.rand() < 0.5:
                        noise_level = np.random.uniform(0.001, 0.01)
                        mel_aug += np.random.normal(0, noise_level, mel_aug.shape)
                    # Time shifting
                    if np.random.rand() < 0.5:
                        shift = np.random.randint(-mel_aug.shape[1] // 20, mel_aug.shape[1] // 20)
                        mel_aug = np.roll(mel_aug, shift, axis=1)
                    # Frequency masking (SpecAugment)
                    freq_masked = False
                    if np.random.rand() < 0.5:
                        num_masks = np.random.randint(1, 3)
                        for _ in range(num_masks):
                            f = np.random.randint(0, mel_aug.shape[0])
                            f_width = np.random.randint(0, mel_aug.shape[0] // 8 + 1)
                            mel_aug[max(0, f - f_width // 2):min(mel_aug.shape[0], f + f_width // 2), :] = 0
                        freq_masked = True
                    # Time masking (SpecAugment)
                    time_masked = False
                    if np.random.rand() < 0.5:
                        num_masks = np.random.randint(1, 3)
                        for _ in range(num_masks):
                            t = np.random.randint(0, mel_aug.shape[1])
                            t_width = np.random.randint(0, mel_aug.shape[1] // 10 + 1)
                            mel_aug[:, max(0, t - t_width // 2):min(mel_aug.shape[1], t + t_width // 2)] = 0
                        time_masked = True
                    # Logging
                    import logging
                    logging.getLogger(__name__).info(
                        f"[AUGMENT] Epoch {self.epoch} (train): gain={gain:.3f}, noise={'yes' if 'noise_level' in locals() else 'no'}, shift={'yes' if 'shift' in locals() and shift != 0 else 'no'}, freq_mask={'yes' if freq_masked else 'no'}, time_mask={'yes' if time_masked else 'no'}"
                    )
                mel_tensor = torch.from_numpy(mel_aug).float()

                # If user disabled teacher embedding cache, recompute teacher embeddings from
                # the augmented mel so teacher receives identical augmentations as student.
                if not use_teacher_emb_cache and self.split == 'train':
                    try:
                        teacher_emb, teacher_seg_embs = self.clap_embedder.compute_embeddings_from_mel(mel_aug)
                        teacher_embedding = teacher_emb
                        teacher_segment_embeddings = teacher_seg_embs
                    except Exception as e:
                        logger.error(f"[CACHE-OFF] Failed to compute teacher embeddings from mel for {item['item_id']}: {e}")

                batch.append({
                    'item_id': item['item_id'],
                    'title': item['title'],
                    'author': item.get('author', 'Unknown'),
                    'audio_path': 'cached',
                    'audio_segments': mel_tensor,
                    'teacher_embedding': teacher_embedding,
                    'teacher_segment_embeddings': teacher_segment_embeddings,
                    'num_segments': len(mel_specs)
                })
            
            # 2. Process new songs from local files
            if tasks_to_process:
                from student_clap.preprocessing.mel_spectrogram import compute_full_mel_spectrogram
                
                for item in tasks_to_process:
                    audio_path = item['file_path']
                    
                    try:
                        # Check what's already cached
                        cached_mel_result = self.mel_cache.get_with_audio_length(item['item_id'])
                        cached_segment_embeddings = self.mel_cache.get_segment_embeddings(item['item_id'])
                        
                        # Get teacher embeddings (compute if not cached)
                        use_teacher_emb_cache = self.config.get('training', {}).get('use_teacher_embedding_cache', True)

                        if cached_segment_embeddings is not None and (use_teacher_emb_cache or self.split != 'train'):
                            # Use cached segments and compute average on-the-fly
                            teacher_segment_embeddings = cached_segment_embeddings
                            teacher_embedding = self.mel_cache.get_averaged_embedding(item['item_id'])
                        else:
                            # Compute from audio if cache disabled or not present
                            teacher_embedding, duration_sec, num_segments, teacher_segment_embeddings = self.clap_embedder.analyze_audio(audio_path)
                            if teacher_embedding is None:
                                logger.error(f"CLAP analysis failed for {item['title']}")
                                continue
                            # Cache only per-segment embeddings if allowed
                            if teacher_segment_embeddings and (use_teacher_emb_cache or self.split != 'train'):
                                self.mel_cache.put_segment_embeddings(item['item_id'], teacher_segment_embeddings)

                        # Get mel spectrograms (compute if not cached)
                        if cached_mel_result is not None:
                            full_mel, audio_length = cached_mel_result
                            
                            # Extract overlapped segments from full spectrogram at runtime
                            mel_specs = self.mel_cache.extract_overlapped_segments(
                                full_mel,
                                audio_length=audio_length,
                                segment_length=self.audio_config['segment_length'],
                                hop_length=self.audio_config['hop_length'],
                                sample_rate=self.audio_config['sample_rate'],
                                hop_length_stft=self.audio_config['hop_length_stft']
                            )
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
                            audio_length = len(audio_data)
                            
                            # OPTIMIZED APPROACH: Compute FULL mel spectrogram once (SAVES SPACE!)
                            logger.debug(f"Computing full mel spectrogram for {item['title']} ({audio_length} samples)")
                            full_mel_spec = compute_full_mel_spectrogram(
                                audio_data,
                                sr=self.audio_config['sample_rate'],
                                n_mels=self.audio_config['n_mels'],
                                n_fft=self.audio_config['n_fft'],
                                hop_length=self.audio_config['hop_length_stft'],
                                fmin=self.audio_config['fmin'],
                                fmax=self.audio_config['fmax']
                            )
                            
                            # Cache the FULL mel spectrogram (not segmented - SAVES SPACE!)
                            self.mel_cache.put(item['item_id'], full_mel_spec, audio_length=audio_length)
                            
                            # Extract overlapped segments at runtime
                            mel_specs = self.mel_cache.extract_overlapped_segments(
                                full_mel_spec,
                                audio_length=audio_length,
                                segment_length=self.audio_config['segment_length'],
                                hop_length=self.audio_config['hop_length'],
                                sample_rate=self.audio_config['sample_rate'],
                                hop_length_stft=self.audio_config['hop_length_stft']
                            )

                        # Apply augmentations to the newly computed mel_specs as well
                        mel_aug = mel_specs.copy()
                        augmentation_enabled = self.config.get('training', {}).get('augmentation_enabled', True)
                        if self.split == 'train' and augmentation_enabled:
                            # Random gain
                            gain = np.random.uniform(0.8, 1.2)
                            mel_aug *= gain
                            # Additive noise
                            if np.random.rand() < 0.5:
                                noise_level = np.random.uniform(0.001, 0.01)
                                mel_aug += np.random.normal(0, noise_level, mel_aug.shape)
                            # Time shifting
                            if np.random.rand() < 0.5:
                                shift = np.random.randint(-mel_aug.shape[1] // 20, mel_aug.shape[1] // 20)
                                mel_aug = np.roll(mel_aug, shift, axis=1)
                            # Frequency masking (SpecAugment)
                            freq_masked = False
                            if np.random.rand() < 0.5:
                                num_masks = np.random.randint(1, 3)
                                for _ in range(num_masks):
                                    f = np.random.randint(0, mel_aug.shape[0])
                                    f_width = np.random.randint(0, mel_aug.shape[0] // 8 + 1)
                                    mel_aug[max(0, f - f_width // 2):min(mel_aug.shape[0], f + f_width // 2), :] = 0
                                freq_masked = True
                            # Time masking (SpecAugment)
                            time_masked = False
                            if np.random.rand() < 0.5:
                                num_masks = np.random.randint(1, 3)
                                for _ in range(num_masks):
                                    t = np.random.randint(0, mel_aug.shape[1])
                                    t_width = np.random.randint(0, mel_aug.shape[1] // 10 + 1)
                                    mel_aug[:, max(0, t - t_width // 2):min(mel_aug.shape[1], t + t_width // 2)] = 0
                                time_masked = True
                            # Logging
                            import logging
                            logging.getLogger(__name__).info(
                                f"[AUGMENT] Epoch {self.epoch} (train): gain={gain:.3f}, noise={'yes' if 'noise_level' in locals() else 'no'}, shift={'yes' if 'shift' in locals() and shift != 0 else 'no'}, freq_mask={'yes' if freq_masked else 'no'}, time_mask={'yes' if time_masked else 'no'}"
                            )

                        # If user disabled teacher embedding cache, recompute teacher embeddings from
                        # the augmented mel so teacher receives identical augmentations as student.
                        if not use_teacher_emb_cache:
                            try:
                                teacher_emb, teacher_seg_embs = self.clap_embedder.compute_embeddings_from_mel(mel_aug)
                                teacher_embedding = teacher_emb
                                teacher_segment_embeddings = teacher_seg_embs
                            except Exception as e:
                                logger.error(f"[CACHE-OFF] Failed to compute teacher embeddings from mel for {item['item_id']}: {e}")

                        mel_tensor = torch.from_numpy(mel_aug).float()
                        batch.append({
                            'item_id': item['item_id'],
                            'title': item['title'],
                            'author': item.get('author', 'Unknown'),
                            'audio_path': audio_path,
                            'audio_segments': mel_tensor,
                            'teacher_embedding': teacher_embedding,
                            'teacher_segment_embeddings': teacher_segment_embeddings,
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
