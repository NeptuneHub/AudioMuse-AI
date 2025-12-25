"""
Feature extractor for converting audio to mel-spectrograms.
Follows CLAP preprocessing standards for consistency with teacher model.
"""

import logging
import numpy as np
import librosa
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract mel-spectrogram features from audio."""
    
    def __init__(self, config: Dict):
        """
        Initialize feature extractor.
        
        Args:
            config: Audio configuration dictionary with keys:
                    sample_rate, n_mels, n_fft, hop_length, window_size
        """
        self.sample_rate = config.get('sample_rate', 48000)
        self.n_mels = config.get('n_mels', 128)
        self.n_fft = config.get('n_fft', 2048)
        self.hop_length = config.get('hop_length', 512)
        self.window_size = config.get('window_size', 2048)
        
        # Frequency range for mel filterbank
        self.fmin = 0
        self.fmax = self.sample_rate // 2
        
        logger.info(f"Initialized feature extractor: n_mels={self.n_mels}, "
                   f"n_fft={self.n_fft}, hop_length={self.hop_length}")
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio.
        
        Args:
            audio: Audio waveform array (mono, sample_rate Hz)
            
        Returns:
            Mel-spectrogram array of shape (n_mels, time_steps)
        """
        try:
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.window_size,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=2.0  # Power spectrogram
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            logger.debug(f"Extracted mel-spectrogram: shape={mel_spec_db.shape}")
            return mel_spec_db
            
        except Exception as e:
            logger.error(f"Failed to extract mel-spectrogram: {e}")
            raise
    
    def normalize_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Normalize mel-spectrogram to [0, 1] range.
        
        Args:
            mel_spec: Mel-spectrogram array
            
        Returns:
            Normalized mel-spectrogram
        """
        # Min-max normalization
        min_val = mel_spec.min()
        max_val = mel_spec.max()
        
        if max_val > min_val:
            mel_spec_normalized = (mel_spec - min_val) / (max_val - min_val)
        else:
            mel_spec_normalized = mel_spec - min_val
        
        return mel_spec_normalized
    
    def prepare_for_model(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Prepare mel-spectrogram for model input.
        Adds channel dimension for CNN: (1, n_mels, time_steps)
        
        Args:
            mel_spec: Mel-spectrogram array (n_mels, time_steps)
            
        Returns:
            Model-ready array with shape (1, n_mels, time_steps)
        """
        # Add channel dimension
        mel_spec_with_channel = np.expand_dims(mel_spec, axis=0)
        return mel_spec_with_channel.astype(np.float32)
    
    def extract_features(self, audio: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Full feature extraction pipeline.
        
        Args:
            audio: Audio waveform array
            normalize: Whether to normalize the spectrogram
            
        Returns:
            Model-ready mel-spectrogram with shape (1, n_mels, time_steps)
        """
        # Extract mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Normalize if requested
        if normalize:
            mel_spec = self.normalize_spectrogram(mel_spec)
        
        # Prepare for model
        mel_spec_ready = self.prepare_for_model(mel_spec)
        
        return mel_spec_ready
    
    def batch_extract(self, audio_list: list) -> np.ndarray:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_list: List of audio waveform arrays
            
        Returns:
            Batch of mel-spectrograms with shape (batch_size, 1, n_mels, max_time_steps)
        """
        features_list = []
        max_time_steps = 0
        
        # Extract features for each audio
        for audio in audio_list:
            features = self.extract_features(audio, normalize=True)
            features_list.append(features)
            max_time_steps = max(max_time_steps, features.shape[2])
        
        # Pad all spectrograms to same time length
        padded_features = []
        for features in features_list:
            if features.shape[2] < max_time_steps:
                # Pad on the time dimension
                padding = max_time_steps - features.shape[2]
                features = np.pad(features, ((0, 0), (0, 0), (0, padding)), mode='constant')
            padded_features.append(features)
        
        # Stack into batch
        batch = np.stack(padded_features, axis=0)
        logger.debug(f"Created feature batch: shape={batch.shape}")
        
        return batch
