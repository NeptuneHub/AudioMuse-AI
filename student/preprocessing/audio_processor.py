"""
Audio processor for loading and preprocessing audio files.
Loads audio at 48kHz (CLAP's native resolution) and converts to mono.
"""

import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio loader and preprocessor for student training."""
    
    def __init__(self, config: Dict):
        """
        Initialize audio processor.
        
        Args:
            config: Audio configuration dictionary with keys:
                    sample_rate, duration
        """
        self.sample_rate = config.get('sample_rate', 48000)
        self.duration = config.get('duration', 30.0)  # 0 for full song
        
        logger.info(f"Initialized audio processor: sr={self.sample_rate}Hz, duration={self.duration}s")
    
    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load audio file at target sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio waveform as numpy array (mono, target sample rate) or None if failed
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True, duration=self.duration if self.duration > 0 else None)
            
            logger.debug(f"Loaded audio: {file_path}, shape={audio.shape}, sr={sr}")
            return audio
            
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            return None
    
    def ensure_length(self, audio: np.ndarray, target_length: Optional[int] = None) -> np.ndarray:
        """
        Ensure audio has target length by padding or truncating.
        
        Args:
            audio: Input audio array
            target_length: Target length in samples (if None, use duration * sample_rate)
            
        Returns:
            Audio array with target length
        """
        if target_length is None:
            if self.duration > 0:
                target_length = int(self.duration * self.sample_rate)
            else:
                return audio  # No length enforcement for full songs
        
        current_length = len(audio)
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio = np.pad(audio, (0, padding), mode='constant')
            logger.debug(f"Padded audio from {current_length} to {target_length}")
        elif current_length > target_length:
            # Truncate
            audio = audio[:target_length]
            logger.debug(f"Truncated audio from {current_length} to {target_length}")
        
        return audio
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def process_audio_file(self, file_path: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Load and process audio file (full pipeline).
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize audio
            
        Returns:
            Processed audio array or None if failed
        """
        audio = self.load_audio(file_path)
        
        if audio is None:
            return None
        
        # Ensure length if duration is specified
        if self.duration > 0:
            audio = self.ensure_length(audio)
        
        # Normalize if requested
        if normalize:
            audio = self.normalize(audio)
        
        return audio
