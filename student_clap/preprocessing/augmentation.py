"""
Audio Augmentation for Student CLAP Training

Optional augmentation techniques to improve generalization.
Includes time stretching, pitch shifting, and noise injection.
"""

import numpy as np
import librosa
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def time_stretch_audio(audio: np.ndarray,
                       rate_range: tuple = (0.95, 1.05)) -> np.ndarray:
    """
    Apply time stretching to audio.
    
    Args:
        audio: Audio waveform
        rate_range: Range of stretch rates (min, max)
        
    Returns:
        Time-stretched audio
    """
    rate = np.random.uniform(rate_range[0], rate_range[1])
    
    if abs(rate - 1.0) < 0.01:  # Skip if very close to 1.0
        return audio
    
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return stretched


def pitch_shift_audio(audio: np.ndarray,
                      sr: int = 48000,
                      n_steps_range: tuple = (-1, 1)) -> np.ndarray:
    """
    Apply pitch shifting to audio.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        n_steps_range: Range of pitch shift in semitones
        
    Returns:
        Pitch-shifted audio
    """
    n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
    
    if abs(n_steps) < 0.1:  # Skip if very small shift
        return audio
    
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return shifted


def add_gaussian_noise(audio: np.ndarray,
                       noise_std: float = 0.005) -> np.ndarray:
    """
    Add Gaussian noise to audio.
    
    Args:
        audio: Audio waveform
        noise_std: Standard deviation of noise
        
    Returns:
        Noisy audio
    """
    noise = np.random.normal(0, noise_std, audio.shape)
    noisy = audio + noise.astype(audio.dtype)
    
    # Clip to valid range
    noisy = np.clip(noisy, -1.0, 1.0)
    
    return noisy


class AudioAugmenter:
    """Applies random augmentations to audio."""
    
    def __init__(self, config: dict):
        """
        Initialize augmenter.
        
        Args:
            config: Augmentation configuration dict with keys:
                - enabled: Whether augmentation is enabled
                - time_stretch_range: Range for time stretching
                - pitch_shift_range: Range for pitch shifting (semitones)
                - gaussian_noise_std: Std deviation for Gaussian noise
        """
        self.enabled = config.get('enabled', False)
        self.time_stretch_range = config.get('time_stretch_range', [0.95, 1.05])
        self.pitch_shift_range = config.get('pitch_shift_range', [-1, 1])
        self.gaussian_noise_std = config.get('gaussian_noise_std', 0.005)
        
        if self.enabled:
            logger.info("Audio augmentation enabled")
            logger.info(f"  Time stretch: {self.time_stretch_range}")
            logger.info(f"  Pitch shift: {self.pitch_shift_range} semitones")
            logger.info(f"  Gaussian noise: std={self.gaussian_noise_std}")
        else:
            logger.info("Audio augmentation disabled")
            
    def augment(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """
        Apply random augmentations to audio.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Augmented audio
        """
        if not self.enabled:
            return audio
        
        try:
            # Randomly apply augmentations
            # Each has 50% chance of being applied
            
            # Time stretching
            if np.random.rand() < 0.5:
                audio = time_stretch_audio(audio, self.time_stretch_range)
            
            # Pitch shifting
            if np.random.rand() < 0.5:
                audio = pitch_shift_audio(audio, sr, self.pitch_shift_range)
            
            # Gaussian noise (always apply if enabled)
            audio = add_gaussian_noise(audio, self.gaussian_noise_std)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, returning original audio")
            return audio


if __name__ == '__main__':
    """Test augmentation functionality."""
    import argparse
    import matplotlib.pyplot as plt
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test audio augmentation')
    parser.add_argument('--audio', type=str,
                        help='Path to audio file to test')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration to process (seconds)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot waveforms')
    args = parser.parse_args()
    
    # Load audio
    if args.audio:
        print(f"Loading audio: {args.audio}")
        audio, sr = librosa.load(args.audio, sr=48000, mono=True, duration=args.duration)
    else:
        print(f"Creating synthetic audio ({args.duration}s)")
        sr = 48000
        duration = args.duration
        t = np.linspace(0, duration, int(duration * sr))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # A440 tone
    
    print(f"Original audio: {len(audio)} samples at {sr} Hz")
    
    # Test individual augmentations
    print("\nTesting time stretching...")
    stretched = time_stretch_audio(audio, rate_range=(0.9, 1.1))
    print(f"  Stretched length: {len(stretched)} (ratio: {len(stretched)/len(audio):.3f})")
    
    print("\nTesting pitch shifting...")
    shifted = pitch_shift_audio(audio, sr=sr, n_steps_range=(-2, 2))
    print(f"  Shifted length: {len(shifted)}")
    
    print("\nTesting Gaussian noise...")
    noisy = add_gaussian_noise(audio, noise_std=0.01)
    snr = 10 * np.log10(np.mean(audio**2) / np.mean((noisy - audio)**2))
    print(f"  SNR: {snr:.2f} dB")
    
    # Test augmenter
    print("\nTesting augmenter...")
    config = {
        'enabled': True,
        'time_stretch_range': [0.95, 1.05],
        'pitch_shift_range': [-1, 1],
        'gaussian_noise_std': 0.005
    }
    augmenter = AudioAugmenter(config)
    
    # Apply multiple times
    for i in range(3):
        augmented = augmenter.augment(audio, sr)
        print(f"  Augmentation {i+1}: {len(augmented)} samples")
    
    # Plot if requested
    if args.plot:
        print("\nPlotting waveforms...")
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        time = np.arange(len(audio)) / sr
        
        axes[0].plot(time, audio)
        axes[0].set_title('Original')
        axes[0].set_ylabel('Amplitude')
        
        axes[1].plot(np.arange(len(stretched)) / sr, stretched)
        axes[1].set_title('Time Stretched')
        axes[1].set_ylabel('Amplitude')
        
        axes[2].plot(time, shifted)
        axes[2].set_title('Pitch Shifted')
        axes[2].set_ylabel('Amplitude')
        
        axes[3].plot(time, noisy)
        axes[3].set_title('With Gaussian Noise')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig('/tmp/augmentation_test.png', dpi=150)
        print("Saved plot to /tmp/augmentation_test.png")
    
    print("\n✓ Augmentation test complete")
