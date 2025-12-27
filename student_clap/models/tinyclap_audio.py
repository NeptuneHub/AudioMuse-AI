"""
TinyCLAP Audio Encoder Architecture

Lightweight audio encoder inspired by Microsoft's TinyCLAP project.
This is a placeholder/reference implementation - actual ONNX graph
construction would happen in build_student_onnx.py

Architecture:
1. Efficient CNN Stem (reduces spatial dimensions)
2. Lightweight Transformer (2 layers, 4 heads)
3. Projection Head (to 512-dim L2-normalized embeddings)

Target size: ~20-40MB ONNX model
"""

import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class TinyCLAPConfig:
    """Configuration for TinyCLAP audio encoder."""
    
    def __init__(self, config: dict):
        """
        Initialize from config dict.
        
        Args:
            config: Model configuration dict
        """
        self.architecture = config.get('architecture', 'tinyclap')
        self.embedding_dim = config.get('embedding_dim', 512)
        
        # CNN stem
        self.cnn_channels = config.get('cnn_channels', [32, 64, 128])
        
        # Transformer
        self.transformer_layers = config.get('transformer_layers', 2)
        self.attention_heads = config.get('attention_heads', 4)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Dropout
        self.dropout = config.get('dropout', 0.1)
        
        # Validate
        assert self.embedding_dim == 512, "Embedding dim must be 512 for CLAP compatibility"
        assert len(self.cnn_channels) == 3, "CNN must have 3 stages"
        
    def get_summary(self) -> Dict:
        """Get configuration summary."""
        return {
            'architecture': self.architecture,
            'embedding_dim': self.embedding_dim,
            'cnn_channels': self.cnn_channels,
            'transformer_layers': self.transformer_layers,
            'attention_heads': self.attention_heads,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout
        }


def calculate_model_size(config: TinyCLAPConfig, 
                         input_shape: Tuple[int, int, int]) -> Dict:
    """
    Estimate model size and parameter count.
    
    Args:
        config: Model configuration
        input_shape: Input shape (batch, time, mels)
        
    Returns:
        Dict with size estimates
    """
    batch, time, mels = input_shape
    
    # CNN stem parameters
    cnn_params = 0
    in_channels = 1
    for out_channels in config.cnn_channels:
        # Conv2D: kernel_size=3x3
        cnn_params += in_channels * out_channels * 9
        # BatchNorm
        cnn_params += out_channels * 2
        in_channels = out_channels
    
    # Estimate spatial dimensions after CNN
    # Each stage: Conv(stride=2) + MaxPool(2x2) = 4x reduction
    # So 3 stages = 4^3 = 64x reduction in each dimension
    reduced_time = max(1, time // 64)
    reduced_mels = max(1, mels // 64)
    sequence_length = reduced_time * reduced_mels
    feature_dim = config.cnn_channels[-1]
    
    # Transformer parameters
    d_model = config.hidden_dim
    
    # Input projection
    proj_params = feature_dim * d_model + d_model
    
    # Transformer layers
    transformer_params = 0
    for _ in range(config.transformer_layers):
        # Multi-head attention
        # Q, K, V projections
        transformer_params += 3 * (d_model * d_model + d_model)
        # Output projection
        transformer_params += d_model * d_model + d_model
        
        # Feed-forward network (4x expansion)
        transformer_params += d_model * (4 * d_model) + 4 * d_model
        transformer_params += (4 * d_model) * d_model + d_model
        
        # Layer norms (2 per layer)
        transformer_params += 2 * (d_model * 2)
    
    # Projection head
    # Dense(512) -> Dense(512)
    proj_head_params = d_model * 512 + 512
    proj_head_params += 512 * 512 + 512
    # BatchNorm
    proj_head_params += 512 * 2
    
    # Total
    total_params = (cnn_params + proj_params + 
                   transformer_params + proj_head_params)
    
    # Estimate size in MB (float32 = 4 bytes per parameter)
    size_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        'total_parameters': int(total_params),
        'cnn_parameters': int(cnn_params),
        'transformer_parameters': int(transformer_params),
        'projection_parameters': int(proj_head_params),
        'estimated_size_mb': round(size_mb, 2),
        'sequence_length': sequence_length,
        'feature_dim': feature_dim
    }


def get_architecture_description(config: TinyCLAPConfig) -> str:
    """
    Get detailed architecture description.
    
    Args:
        config: Model configuration
        
    Returns:
        Formatted string describing architecture
    """
    desc = []
    desc.append("=" * 60)
    desc.append("TinyCLAP Audio Encoder Architecture")
    desc.append("=" * 60)
    desc.append("")
    
    desc.append("Input: Mel-spectrogram (batch, time_frames, 128)")
    desc.append("")
    
    desc.append("1. CNN Stem (Efficient Feature Extraction)")
    desc.append("   " + "-" * 50)
    for i, channels in enumerate(config.cnn_channels):
        desc.append(f"   Stage {i+1}:")
        desc.append(f"     Conv2D(out={channels}, kernel=3x3, stride=2)")
        desc.append(f"     BatchNorm2D({channels})")
        desc.append(f"     GELU()")
        if i < 2:
            desc.append(f"     MaxPool2D(2x2)")
    desc.append("")
    
    desc.append("2. Lightweight Transformer")
    desc.append("   " + "-" * 50)
    desc.append(f"   Flatten spatial dims -> sequence")
    desc.append(f"   {config.transformer_layers} Transformer layers:")
    desc.append(f"     - Multi-head attention ({config.attention_heads} heads)")
    desc.append(f"     - Hidden dim: {config.hidden_dim}")
    desc.append(f"     - Feed-forward network (4x expansion)")
    desc.append(f"     - Layer normalization")
    desc.append(f"     - Dropout: {config.dropout}")
    desc.append("")
    
    desc.append("3. Projection Head")
    desc.append("   " + "-" * 50)
    desc.append(f"   Global average pooling")
    desc.append(f"   Dense({config.hidden_dim} -> 512)")
    desc.append(f"   BatchNorm(512)")
    desc.append(f"   GELU()")
    desc.append(f"   Dense(512 -> 512)")
    desc.append(f"   L2 Normalization")
    desc.append("")
    
    desc.append("Output: 512-dim L2-normalized embedding")
    desc.append("=" * 60)
    
    return "\n".join(desc)


if __name__ == '__main__':
    """Test architecture configuration."""
    import yaml
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test TinyCLAP architecture')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model config
    model_config = TinyCLAPConfig(config['model'])
    
    # Print configuration
    print("\nModel Configuration:")
    summary = model_config.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Print architecture
    print("\n" + get_architecture_description(model_config))
    
    # Estimate size
    print("\nModel Size Estimation:")
    # Assume 10-second audio segment at 48kHz
    # With n_fft=1024, hop=480: time_frames ≈ 1000
    input_shape = (1, 1000, 128)
    size_info = calculate_model_size(model_config, input_shape)
    
    for key, value in size_info.items():
        print(f"  {key}: {value}")
    
    if size_info['estimated_size_mb'] > 40:
        print("\n⚠️  Warning: Model size exceeds 40MB target")
        print("   Consider reducing hidden_dim or transformer_layers")
    else:
        print(f"\n✓ Model size within target (< 40MB)")
