"""
Mel-Spectrogram Computation for Student CLAP

Computes mel-spectrograms compatible with CLAP preprocessing.
Uses different parameters than teacher (128 mel bands vs 64) since
the student model has its own architecture.
"""

import numpy as np
import librosa
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default parameters for student model
# (Different from teacher which uses 64 mel bands)
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 480
DEFAULT_FMIN = 0
DEFAULT_FMAX = 14000


def compute_mel_spectrogram(audio_data: np.ndarray,
                             sr: int = DEFAULT_SAMPLE_RATE,
                             n_mels: int = DEFAULT_N_MELS,
                             n_fft: int = DEFAULT_N_FFT,
                             hop_length: int = DEFAULT_HOP_LENGTH,
                             fmin: int = DEFAULT_FMIN,
                             fmax: int = DEFAULT_FMAX) -> np.ndarray:
    """
    Compute log mel-spectrogram from audio waveform.
    
    This creates mel-spectrograms for the student model input.
    Note: Parameters differ from teacher CLAP (128 vs 64 mel bands)
    since the student has its own architecture.
    
    Args:
        audio_data: Audio waveform (mono, 48kHz), shape (n_samples,)
        sr: Sample rate (should be 48000)
        n_mels: Number of mel bands (128 for student)
        n_fft: FFT window size
        hop_length: STFT hop length
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Log mel-spectrogram of shape (1, time_frames, n_mels)
    """
    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale (dB)
    mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    
    # Transpose to (time_frames, mel_bins)
    mel = mel.T
    
    # Add batch dimension: (1, time_frames, n_mels)
    mel = mel[np.newaxis, :, :]
    
    return mel.astype(np.float32)


def compute_mel_spectrogram_batch(audio_segments: list,
                                   sr: int = DEFAULT_SAMPLE_RATE,
                                   n_mels: int = DEFAULT_N_MELS,
                                   n_fft: int = DEFAULT_N_FFT,
                                   hop_length: int = DEFAULT_HOP_LENGTH,
                                   fmin: int = DEFAULT_FMIN,
                                   fmax: int = DEFAULT_FMAX) -> np.ndarray:
    """
    Compute mel-spectrograms for a batch of audio segments.
    
    Args:
        audio_segments: List of audio segments, each of shape (n_samples,)
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: STFT hop length
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Batch of mel-spectrograms, shape (batch_size, time_frames, n_mels)
    """
    mel_specs = []
    
    for segment in audio_segments:
        mel = compute_mel_spectrogram(
            segment, sr, n_mels, n_fft, hop_length, fmin, fmax
        )
        mel_specs.append(mel)
    
    # Stack into batch (remove individual batch dims and create new one)
    mel_batch = np.concatenate(mel_specs, axis=0)
    
    return mel_batch


def normalize_mel_spectrogram(mel: np.ndarray,
                               mean: Optional[np.ndarray] = None,
                               std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize mel-spectrogram using mean and std.
    
    Optional normalization that can be applied if computed from training set.
    
    Args:
        mel: Mel-spectrogram of shape (batch, time, mels) or (1, time, mels)
        mean: Mean values, shape (mels,) or None
        std: Std values, shape (mels,) or None
        
    Returns:
        Normalized mel-spectrogram
    """
    if mean is None or std is None:
        return mel
    
    # Add dimensions for broadcasting: (1, 1, mels)
    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)
    
    normalized = (mel - mean) / (std + 1e-8)
    return normalized


def get_mel_spectrogram_shape(audio_length: int,
                               sr: int = DEFAULT_SAMPLE_RATE,
                               n_mels: int = DEFAULT_N_MELS,
                               n_fft: int = DEFAULT_N_FFT,
                               hop_length: int = DEFAULT_HOP_LENGTH) -> tuple:
    """
    Calculate expected mel-spectrogram shape without computing it.
    
    Args:
        audio_length: Audio length in samples
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: STFT hop length
        
    Returns:
        Tuple of (time_frames, n_mels)
    """
    # librosa pads with n_fft // 2 on each side when center=True
    padded_length = audio_length + n_fft
    
    # Calculate number of frames
    time_frames = (padded_length - n_fft) // hop_length + 1
    
    return (time_frames, n_mels)


def validate_mel_spectrogram(mel: np.ndarray,
                              expected_mels: int = DEFAULT_N_MELS,
                              min_time_frames: int = 1) -> bool:
    """
    Validate mel-spectrogram shape and values.
    
    Args:
        mel: Mel-spectrogram array
        expected_mels: Expected number of mel bands
        min_time_frames: Minimum expected time frames
        
    Returns:
        True if valid
    """
    # Check shape
    if mel.ndim != 3:
        logger.error(f"Mel-spectrogram has wrong number of dimensions: {mel.ndim} (expected 3)")
        return False
    
    batch, time, mels = mel.shape
    
    if mels != expected_mels:
        logger.error(f"Mel-spectrogram has wrong number of mel bands: {mels} (expected {expected_mels})")
        return False
    
    if time < min_time_frames:
        logger.error(f"Mel-spectrogram has too few time frames: {time} (expected >= {min_time_frames})")
        return False
    
    # Check for NaN or Inf
    if not np.isfinite(mel).all():
        logger.error("Mel-spectrogram contains NaN or Inf values")
        return False
    
    logger.debug(f"Mel-spectrogram validation passed: shape {mel.shape}")
    return True


if __name__ == '__main__':
    """Test mel-spectrogram computation."""
    import argparse
    import matplotlib.pyplot as plt
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test mel-spectrogram computation')
    parser.add_argument('--audio', type=str,
                        help='Path to audio file to test')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration to process (seconds)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the mel-spectrogram')
    args = parser.parse_args()
    
    if args.audio:
        # Load real audio
        print(f"Loading audio: {args.audio}")
        audio_data, sr = librosa.load(args.audio, sr=DEFAULT_SAMPLE_RATE, 
                                      mono=True, duration=args.duration)
        print(f"Loaded: {len(audio_data)} samples at {sr} Hz")
    else:
        # Create synthetic audio (chirp)
        print(f"Creating synthetic chirp audio ({args.duration}s)")
        duration = args.duration
        sr = DEFAULT_SAMPLE_RATE
        t = np.linspace(0, duration, int(duration * sr))
        # Chirp from 100 Hz to 8000 Hz
        audio_data = np.sin(2 * np.pi * (100 + (8000 - 100) * t / duration) * t)
        audio_data = audio_data.astype(np.float32)
    
    # Compute mel-spectrogram
    print("\nComputing mel-spectrogram...")
    mel = compute_mel_spectrogram(audio_data, sr=sr)
    
    print(f"Mel-spectrogram shape: {mel.shape}")
    print(f"  Batch: {mel.shape[0]}")
    print(f"  Time frames: {mel.shape[1]}")
    print(f"  Mel bands: {mel.shape[2]}")
    print(f"  Value range: [{mel.min():.2f}, {mel.max():.2f}] dB")
    
    # Validate
    is_valid = validate_mel_spectrogram(mel)
    print(f"Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    
    # Expected shape
    expected_shape = get_mel_spectrogram_shape(len(audio_data), sr)
    print(f"\nExpected shape: {expected_shape}")
    print(f"Actual shape:   {mel.shape[1:]}")
    print(f"Match: {'✓' if mel.shape[1:] == expected_shape else '✗'}")
    
    # Test batch processing
    print("\nTesting batch processing...")
    segments = [audio_data, audio_data, audio_data]  # 3 identical segments
    mel_batch = compute_mel_spectrogram_batch(segments, sr=sr)
    print(f"Batch shape: {mel_batch.shape}")
    print(f"  Expected: (3, {expected_shape[0]}, {expected_shape[1]})")
    
    # Plot if requested
    if args.plot:
        print("\nPlotting mel-spectrogram...")
        plt.figure(figsize=(12, 6))
        plt.imshow(mel[0].T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='dB')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Band')
        plt.title('Log Mel-Spectrogram')
        plt.tight_layout()
        plt.savefig('/tmp/mel_spectrogram_test.png', dpi=150)
        print("Saved plot to /tmp/mel_spectrogram_test.png")
