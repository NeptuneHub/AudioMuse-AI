"""
Audio Segmentation Module

Implements the exact segmentation strategy used by the teacher CLAP model
in production (10-second segments with 5-second hop, 50% overlap).

CRITICAL: This must match tasks/clap_analyzer.py segmentation exactly!
"""

import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Constants (MUST match teacher CLAP analyzer)
SAMPLE_RATE = 48000
SEGMENT_LENGTH = 480000  # 10 seconds at 48kHz
HOP_LENGTH = 240000      # 5 seconds (50% overlap)


def segment_audio(audio_data: np.ndarray, 
                  sample_rate: int = SAMPLE_RATE,
                  segment_length: int = SEGMENT_LENGTH,
                  hop_length: int = HOP_LENGTH) -> List[np.ndarray]:
    """
    Segment audio into overlapping windows.
    
    This function replicates the exact segmentation logic from
    tasks/clap_analyzer.py (lines 614-628) to ensure the student
    model processes audio identically to the teacher.
    
    Args:
        audio_data: Audio waveform (mono, 48kHz)
        sample_rate: Sample rate (must be 48000)
        segment_length: Length of each segment in samples (480000 = 10s)
        hop_length: Hop between segments in samples (240000 = 5s)
        
    Returns:
        List of audio segments, each of length segment_length
    """
    # Validate sample rate
    if sample_rate != SAMPLE_RATE:
        logger.warning(f"Sample rate {sample_rate} != {SAMPLE_RATE}, resampling needed")
        
    segments = []
    total_length = len(audio_data)
    
    if total_length <= segment_length:
        # Pad short audio to segment_length
        padded = np.pad(audio_data, (0, segment_length - total_length), mode='constant')
        segments.append(padded)
    else:
        # Create overlapping segments
        for start in range(0, total_length - segment_length + 1, hop_length):
            segment = audio_data[start:start + segment_length]
            segments.append(segment)
        
        # Add final segment if needed
        last_start = len(segments) * hop_length
        if last_start < total_length:
            last_segment = audio_data[-segment_length:]
            segments.append(last_segment)
    
    logger.debug(f"Segmented audio: {total_length} samples -> {len(segments)} segments")
    return segments


def compute_segment_positions(total_length: int,
                               segment_length: int = SEGMENT_LENGTH,
                               hop_length: int = HOP_LENGTH) -> List[Tuple[int, int]]:
    """
    Compute segment start/end positions without loading audio.
    
    Useful for pre-computing segment boundaries and understanding
    how many segments will be created.
    
    Args:
        total_length: Total audio length in samples
        segment_length: Length of each segment in samples
        hop_length: Hop between segments in samples
        
    Returns:
        List of (start, end) tuples for each segment
    """
    positions = []
    
    if total_length <= segment_length:
        # Single segment (will be padded)
        positions.append((0, segment_length))
    else:
        # Overlapping segments
        for start in range(0, total_length - segment_length + 1, hop_length):
            end = start + segment_length
            positions.append((start, end))
        
        # Final segment
        last_start = len(positions) * hop_length
        if last_start < total_length:
            positions.append((total_length - segment_length, total_length))
    
    return positions


def get_num_segments(duration_sec: float, 
                     sample_rate: int = SAMPLE_RATE,
                     segment_length: int = SEGMENT_LENGTH,
                     hop_length: int = HOP_LENGTH) -> int:
    """
    Calculate number of segments for a given duration.
    
    Args:
        duration_sec: Audio duration in seconds
        sample_rate: Sample rate
        segment_length: Length of each segment in samples
        hop_length: Hop between segments in samples
        
    Returns:
        Number of segments that will be created
    """
    total_length = int(duration_sec * sample_rate)
    positions = compute_segment_positions(total_length, segment_length, hop_length)
    return len(positions)


def validate_segmentation(audio_data: np.ndarray,
                          segments: List[np.ndarray],
                          sample_rate: int = SAMPLE_RATE,
                          segment_length: int = SEGMENT_LENGTH) -> bool:
    """
    Validate that segmentation was performed correctly.
    
    Args:
        audio_data: Original audio data
        segments: List of segments
        sample_rate: Sample rate
        segment_length: Expected segment length
        
    Returns:
        True if segmentation is valid
    """
    # Check all segments have correct length
    for i, seg in enumerate(segments):
        if len(seg) != segment_length:
            logger.error(f"Segment {i} has incorrect length: {len(seg)} != {segment_length}")
            return False
    
    # Check expected number of segments
    expected_num = get_num_segments(len(audio_data) / sample_rate, sample_rate)
    if len(segments) != expected_num:
        logger.error(f"Unexpected number of segments: {len(segments)} != {expected_num}")
        return False
    
    logger.debug("Segmentation validation passed")
    return True


def average_segment_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Average embeddings from multiple segments.
    
    This replicates the averaging performed by the teacher model
    (tasks/clap_analyzer.py line 713).
    
    Args:
        embeddings: Array of shape (num_segments, embedding_dim)
        
    Returns:
        Averaged embedding of shape (embedding_dim,)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
    
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Normalize (teacher normalizes after averaging)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    
    return avg_embedding


if __name__ == '__main__':
    """Test segmentation functionality."""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test audio segmentation')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Test audio duration in seconds')
    args = parser.parse_args()
    
    # Create synthetic audio
    duration_sec = args.duration
    total_samples = int(duration_sec * SAMPLE_RATE)
    audio_data = np.random.randn(total_samples).astype(np.float32)
    
    print(f"Test audio: {duration_sec:.1f} seconds ({total_samples} samples)")
    
    # Segment
    segments = segment_audio(audio_data)
    
    print(f"\nSegmentation results:")
    print(f"  Number of segments: {len(segments)}")
    print(f"  Segment length: {len(segments[0])} samples ({len(segments[0])/SAMPLE_RATE:.1f}s)")
    
    # Validate
    is_valid = validate_segmentation(audio_data, segments)
    print(f"  Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    
    # Show segment positions
    positions = compute_segment_positions(total_samples)
    print(f"\nSegment positions:")
    for i, (start, end) in enumerate(positions):
        start_sec = start / SAMPLE_RATE
        end_sec = end / SAMPLE_RATE
        print(f"  Segment {i}: {start:8d} - {end:8d} ({start_sec:6.2f}s - {end_sec:6.2f}s)")
    
    # Test averaging
    print(f"\nTesting embedding averaging:")
    fake_embeddings = np.random.randn(len(segments), 512).astype(np.float32)
    avg_embedding = average_segment_embeddings(fake_embeddings)
    print(f"  Input shape: {fake_embeddings.shape}")
    print(f"  Output shape: {avg_embedding.shape}")
    print(f"  Output norm: {np.linalg.norm(avg_embedding):.4f} (should be ~1.0)")
