"""
Loss Functions for Student CLAP Training

Implements MSE and cosine similarity losses for embedding matching.
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def mse_loss(student_embedding: np.ndarray,
             teacher_embedding: np.ndarray) -> float:
    """
    Compute Mean Squared Error loss between embeddings.
    
    Args:
        student_embedding: Student embedding (batch, 512) or (512,)
        teacher_embedding: Teacher embedding (batch, 512) or (512,)
        
    Returns:
        MSE loss value
    """
    diff = student_embedding - teacher_embedding
    mse = np.mean(diff ** 2)
    return float(mse)


def cosine_similarity_loss(student_embedding: np.ndarray,
                           teacher_embedding: np.ndarray) -> float:
    """
    Compute cosine similarity loss (1 - cosine_similarity).
    
    Args:
        student_embedding: Student embedding (batch, 512) or (512,)
        teacher_embedding: Teacher embedding (batch, 512) or (512,)
        
    Returns:
        Cosine loss value (0 = perfect match, 2 = opposite)
    """
    # Normalize embeddings
    student_norm = student_embedding / (np.linalg.norm(student_embedding, axis=-1, keepdims=True) + 1e-8)
    teacher_norm = teacher_embedding / (np.linalg.norm(teacher_embedding, axis=-1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    cosine_sim = np.sum(student_norm * teacher_norm, axis=-1)
    
    # Convert to loss (1 - similarity)
    cosine_loss = 1.0 - np.mean(cosine_sim)
    
    return float(cosine_loss)


def l2_distance(student_embedding: np.ndarray,
                teacher_embedding: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) distance between embeddings.
    
    Args:
        student_embedding: Student embedding (batch, 512) or (512,)
        teacher_embedding: Teacher embedding (batch, 512) or (512,)
        
    Returns:
        Average L2 distance
    """
    diff = student_embedding - teacher_embedding
    l2_dist = np.linalg.norm(diff, axis=-1)
    return float(np.mean(l2_dist))


def combined_loss(student_embedding: np.ndarray,
                  teacher_embedding: np.ndarray,
                  mse_weight: float = 0.6,
                  cosine_weight: float = 0.4) -> Tuple[float, dict]:
    """
    Compute combined loss (MSE + cosine similarity).
    
    Args:
        student_embedding: Student embedding (batch, 512) or (512,)
        teacher_embedding: Teacher embedding (batch, 512) or (512,)
        mse_weight: Weight for MSE loss
        cosine_weight: Weight for cosine loss
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Compute individual losses
    mse = mse_loss(student_embedding, teacher_embedding)
    cosine = cosine_similarity_loss(student_embedding, teacher_embedding)
    
    # Weighted combination
    total = mse_weight * mse + cosine_weight * cosine
    
    # Return detailed breakdown
    loss_dict = {
        'total_loss': total,
        'mse_loss': mse,
        'cosine_loss': cosine,
        'mse_weight': mse_weight,
        'cosine_weight': cosine_weight
    }
    
    return total, loss_dict


def compute_batch_loss(student_embeddings: list,
                       teacher_embeddings: np.ndarray,
                       mse_weight: float = 0.6,
                       cosine_weight: float = 0.4) -> Tuple[float, dict]:
    """
    Compute loss for a batch where each sample may have different numbers of segments.
    
    Since each song is segmented independently and then averaged, we:
    1. Average each song's student segment embeddings
    2. Compute loss between averaged student and teacher embeddings
    
    Args:
        student_embeddings: List of segment embeddings per song,
                           each of shape (num_segments, 512)
        teacher_embeddings: Teacher embeddings, shape (batch_size, 512)
        mse_weight: Weight for MSE loss
        cosine_weight: Weight for cosine loss
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    batch_size = len(student_embeddings)
    
    if batch_size != len(teacher_embeddings):
        raise ValueError(f"Batch size mismatch: {batch_size} vs {len(teacher_embeddings)}")
    
    # Average student embeddings for each song
    student_averaged = []
    for segments in student_embeddings:
        # Average across segments
        avg_embedding = np.mean(segments, axis=0)
        # Normalize
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        student_averaged.append(avg_embedding)
    
    student_averaged = np.array(student_averaged)
    
    # Compute combined loss
    total, loss_dict = combined_loss(
        student_averaged,
        teacher_embeddings,
        mse_weight,
        cosine_weight
    )
    
    # Add batch info
    loss_dict['batch_size'] = batch_size
    loss_dict['avg_segments'] = np.mean([len(seg) for seg in student_embeddings])
    
    return total, loss_dict


if __name__ == '__main__':
    """Test loss functions."""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test loss functions')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Test batch size')
    args = parser.parse_args()
    
    print("Testing loss functions...")
    
    # Create synthetic embeddings
    batch_size = args.batch_size
    embedding_dim = 512
    
    # Teacher embeddings (already normalized)
    teacher = np.random.randn(batch_size, embedding_dim).astype(np.float32)
    teacher = teacher / np.linalg.norm(teacher, axis=1, keepdims=True)
    
    # Student embeddings (close to teacher with some noise)
    noise_level = 0.1
    student = teacher + np.random.randn(batch_size, embedding_dim).astype(np.float32) * noise_level
    student = student / np.linalg.norm(student, axis=1, keepdims=True)
    
    print(f"\nTest setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Noise level: {noise_level}")
    
    # Test individual losses
    print("\n1. MSE Loss:")
    mse = mse_loss(student, teacher)
    print(f"   Value: {mse:.6f}")
    
    print("\n2. Cosine Similarity Loss:")
    cosine = cosine_similarity_loss(student, teacher)
    print(f"   Value: {cosine:.6f}")
    print(f"   Cosine similarity: {1 - cosine:.6f}")
    
    print("\n3. L2 Distance:")
    l2 = l2_distance(student, teacher)
    print(f"   Value: {l2:.6f}")
    
    print("\n4. Combined Loss:")
    total, loss_dict = combined_loss(student, teacher, mse_weight=0.6, cosine_weight=0.4)
    print(f"   Total: {total:.6f}")
    print(f"   MSE component: {loss_dict['mse_loss']:.6f} (weight: {loss_dict['mse_weight']})")
    print(f"   Cosine component: {loss_dict['cosine_loss']:.6f} (weight: {loss_dict['cosine_weight']})")
    
    # Test batch loss with segments
    print("\n5. Batch Loss with Segments:")
    # Simulate multiple segments per song
    student_segments = []
    for i in range(batch_size):
        num_segments = np.random.randint(3, 8)
        # Each segment is close to teacher with noise
        segments = teacher[i:i+1].repeat(num_segments, axis=0)
        segments += np.random.randn(num_segments, embedding_dim).astype(np.float32) * noise_level
        segments = segments / np.linalg.norm(segments, axis=1, keepdims=True)
        student_segments.append(segments)
    
    batch_loss, batch_dict = compute_batch_loss(student_segments, teacher)
    print(f"   Total: {batch_loss:.6f}")
    print(f"   Batch size: {batch_dict['batch_size']}")
    print(f"   Avg segments per song: {batch_dict['avg_segments']:.1f}")
    
    # Test perfect match
    print("\n6. Perfect Match Test:")
    perfect_loss, perfect_dict = combined_loss(teacher, teacher)
    print(f"   Total: {perfect_loss:.6f} (should be ~0)")
    print(f"   MSE: {perfect_dict['mse_loss']:.6f}")
    print(f"   Cosine: {perfect_dict['cosine_loss']:.6f}")
    
    # Test opposite embeddings
    print("\n7. Opposite Embeddings Test:")
    opposite = -teacher
    worst_loss, worst_dict = combined_loss(opposite, teacher)
    print(f"   Total: {worst_loss:.6f} (should be high)")
    print(f"   MSE: {worst_dict['mse_loss']:.6f}")
    print(f"   Cosine: {worst_dict['cosine_loss']:.6f} (should be ~2)")
    
    print("\n✓ Loss function tests complete")
