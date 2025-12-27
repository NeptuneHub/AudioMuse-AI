"""
Model Utilities

Shared utilities for model operations.
"""

import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


def l2_normalize(embeddings: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    L2 normalize embeddings.
    
    Args:
        embeddings: Embeddings to normalize
        axis: Axis along which to normalize
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return normalized


def average_and_normalize(segment_embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Average segment embeddings and normalize.
    
    This replicates the exact averaging performed by teacher CLAP.
    
    Args:
        segment_embeddings: List of embeddings, each (num_segments, 512)
        
    Returns:
        Averaged and normalized embeddings, shape (batch_size, 512)
    """
    averaged = []
    
    for segments in segment_embeddings:
        # Average across segments
        avg = np.mean(segments, axis=0)
        # L2 normalize
        avg_norm = l2_normalize(avg.reshape(1, -1))[0]
        averaged.append(avg_norm)
    
    return np.array(averaged)


def calculate_embedding_stats(embeddings: np.ndarray) -> dict:
    """
    Calculate statistics for embeddings.
    
    Args:
        embeddings: Embeddings array (n_samples, embedding_dim)
        
    Returns:
        Dict with statistics
    """
    norms = np.linalg.norm(embeddings, axis=1)
    
    stats = {
        'shape': embeddings.shape,
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'min_norm': float(np.min(norms)),
        'max_norm': float(np.max(norms)),
        'mean_value': float(np.mean(embeddings)),
        'std_value': float(np.std(embeddings))
    }
    
    return stats


def count_parameters(model_config: dict) -> int:
    """
    Estimate total number of parameters in model.
    
    Args:
        model_config: Model configuration dict
        
    Returns:
        Estimated parameter count
    """
    # This is a simplified estimation
    # Full implementation would calculate based on layer sizes
    
    # CNN parameters
    cnn_params = 0
    in_channels = 1
    for out_channels in model_config['cnn_channels']:
        cnn_params += in_channels * out_channels * 9  # 3x3 conv
        cnn_params += out_channels * 2  # BatchNorm
        in_channels = out_channels
    
    # Transformer parameters (rough estimate)
    d_model = model_config['hidden_dim']
    n_layers = model_config['transformer_layers']
    transformer_params = n_layers * (
        4 * d_model * d_model +  # Q, K, V, O projections
        4 * d_model * d_model +  # FFN
        4 * d_model  # Layer norms
    )
    
    # Projection head
    proj_params = d_model * 512 + 512 * 512 + 512 * 4
    
    total = cnn_params + transformer_params + proj_params
    
    return int(total)


if __name__ == '__main__':
    """Test model utilities."""
    
    print("Testing model utilities...")
    
    # Test L2 normalization
    print("\n1. L2 Normalization:")
    embeddings = np.random.randn(5, 512).astype(np.float32)
    normalized = l2_normalize(embeddings)
    norms = np.linalg.norm(normalized, axis=1)
    print(f"  Input shape: {embeddings.shape}")
    print(f"  Output norms: {norms}")
    print(f"  All norms ≈ 1.0: {np.allclose(norms, 1.0)}")
    
    # Test averaging
    print("\n2. Average and Normalize:")
    segment_embeddings = [
        np.random.randn(3, 512).astype(np.float32),
        np.random.randn(5, 512).astype(np.float32),
        np.random.randn(4, 512).astype(np.float32)
    ]
    averaged = average_and_normalize(segment_embeddings)
    print(f"  Input segments: {[s.shape for s in segment_embeddings]}")
    print(f"  Output shape: {averaged.shape}")
    avg_norms = np.linalg.norm(averaged, axis=1)
    print(f"  Output norms: {avg_norms}")
    print(f"  All norms ≈ 1.0: {np.allclose(avg_norms, 1.0)}")
    
    # Test embedding stats
    print("\n3. Embedding Statistics:")
    stats = calculate_embedding_stats(averaged)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test parameter count
    print("\n4. Parameter Count:")
    model_config = {
        'cnn_channels': [32, 64, 128],
        'transformer_layers': 2,
        'hidden_dim': 256
    }
    param_count = count_parameters(model_config)
    size_mb = (param_count * 4) / (1024 * 1024)
    print(f"  Config: {model_config}")
    print(f"  Estimated parameters: {param_count:,}")
    print(f"  Estimated size: {size_mb:.2f} MB")
    
    print("\n✓ Model utilities test complete")
