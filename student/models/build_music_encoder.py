"""
Build lightweight CNN-based music encoder as ONNX graph.
Architecture: Conv2D layers → BatchNorm → ReLU → MaxPool → GlobalAvgPool → Dense → L2 Norm
"""

import logging
import onnx
from onnx import helper, TensorProto
import numpy as np
from typing import Dict, List
from models.student_utils import (
    make_conv2d_node, make_batchnorm_node, make_relu_node,
    make_maxpool_node, make_global_avgpool_node, make_dense_node,
    make_l2_normalize_node, make_flatten_node, make_dropout_node,
    initialize_weights, create_initializer_tensor, make_value_info
)

logger = logging.getLogger(__name__)


def build_music_encoder(config: Dict, output_path: str = None) -> onnx.ModelProto:
    """
    Build the music encoder ONNX model programmatically.
    
    Architecture:
    Input (batch, 1, 128, time) → Conv2D (32) → BN → ReLU → MaxPool 
    → Conv2D (64) → BN → ReLU → MaxPool → Conv2D (128) → BN → ReLU → GlobalAvgPool
    → Flatten → Dense (256) → L2 Normalize → Output (batch, 256)
    
    Args:
        config: Model configuration dictionary
        output_path: Optional path to save the model
        
    Returns:
        ONNX ModelProto
    """
    logger.info("Building music encoder ONNX model")
    
    # Extract configuration
    model_config = config.get('music_encoder', {})
    conv_filters = model_config.get('conv_filters', [32, 64, 128])
    kernel_sizes = model_config.get('kernel_sizes', [[3, 3], [3, 3], [3, 3]])
    pool_sizes = model_config.get('pool_sizes', [[2, 2], [2, 2], [2, 2]])
    embedding_dim = model_config.get('embedding_dim', 256)
    dropout = model_config.get('dropout', 0.1)
    input_channels = model_config.get('input_channels', 1)
    
    # Model inputs
    # Shape: (batch_size, channels, n_mels, time_steps)
    # Use dynamic dimensions for batch_size and time_steps
    input_name = "mel_spectrogram"
    output_name = "music_embedding"
    
    # Create nodes and initializers
    nodes = []
    initializers = []
    
    # Input shape: (None, 1, 128, None) - dynamic batch and time
    current_input = input_name
    current_channels = input_channels
    
    # Build convolutional layers
    for i, (filters, kernel_size, pool_size) in enumerate(zip(conv_filters, kernel_sizes, pool_sizes)):
        layer_prefix = f"conv{i+1}"
        
        # Conv2D weights and bias
        conv_weight_name = f"{layer_prefix}_weight"
        conv_bias_name = f"{layer_prefix}_bias"
        conv_output = f"{layer_prefix}_conv_out"
        
        # Initialize Conv2D weights: (out_channels, in_channels, kernel_h, kernel_w)
        conv_weight_shape = (filters, current_channels, kernel_size[0], kernel_size[1])
        conv_weight = initialize_weights(conv_weight_shape, conv_weight_name, "he")
        conv_bias = initialize_weights((filters,), conv_bias_name, "zeros")
        
        initializers.append(create_initializer_tensor(conv_weight_name, conv_weight))
        initializers.append(create_initializer_tensor(conv_bias_name, conv_bias))
        
        # Conv2D node
        conv_node = make_conv2d_node(
            name=f"{layer_prefix}_conv",
            input_name=current_input,
            output_name=conv_output,
            weight_name=conv_weight_name,
            bias_name=conv_bias_name,
            in_channels=current_channels,
            out_channels=filters,
            kernel_size=tuple(kernel_size),
            stride=(1, 1),
            padding="SAME_UPPER"
        )
        nodes.append(conv_node)
        
        # BatchNorm
        bn_output = f"{layer_prefix}_bn_out"
        bn_scale_name = f"{layer_prefix}_bn_scale"
        bn_bias_name = f"{layer_prefix}_bn_bias"
        bn_mean_name = f"{layer_prefix}_bn_mean"
        bn_var_name = f"{layer_prefix}_bn_var"
        
        # Initialize BatchNorm parameters
        bn_scale = np.ones((filters,), dtype=np.float32)
        bn_bias = np.zeros((filters,), dtype=np.float32)
        bn_mean = np.zeros((filters,), dtype=np.float32)
        bn_var = np.ones((filters,), dtype=np.float32)
        
        initializers.extend([
            create_initializer_tensor(bn_scale_name, bn_scale),
            create_initializer_tensor(bn_bias_name, bn_bias),
            create_initializer_tensor(bn_mean_name, bn_mean),
            create_initializer_tensor(bn_var_name, bn_var)
        ])
        
        bn_node = make_batchnorm_node(
            name=f"{layer_prefix}_bn",
            input_name=conv_output,
            output_name=bn_output,
            scale_name=bn_scale_name,
            bias_name=bn_bias_name,
            mean_name=bn_mean_name,
            var_name=bn_var_name
        )
        nodes.append(bn_node)
        
        # ReLU
        relu_output = f"{layer_prefix}_relu_out"
        relu_node = make_relu_node(
            name=f"{layer_prefix}_relu",
            input_name=bn_output,
            output_name=relu_output
        )
        nodes.append(relu_node)
        
        # MaxPool
        pool_output = f"{layer_prefix}_pool_out"
        pool_node = make_maxpool_node(
            name=f"{layer_prefix}_pool",
            input_name=relu_output,
            output_name=pool_output,
            kernel_size=tuple(pool_size),
            stride=tuple(pool_size)
        )
        nodes.append(pool_node)
        
        # Update for next layer
        current_input = pool_output
        current_channels = filters
    
    # GlobalAveragePool
    gap_output = "gap_out"
    gap_node = make_global_avgpool_node(
        name="global_avg_pool",
        input_name=current_input,
        output_name=gap_output
    )
    nodes.append(gap_node)
    
    # Flatten (GlobalAvgPool output is (batch, channels, 1, 1))
    flatten_output = "flatten_out"
    flatten_node = make_flatten_node(
        name="flatten",
        input_name=gap_output,
        output_name=flatten_output,
        axis=1
    )
    nodes.append(flatten_node)
    
    # Dense layer to embedding_dim
    dense_output = "dense_out"
    dense_weight_name = "dense_weight"
    dense_bias_name = "dense_bias"
    
    # Initialize Dense weights: (input_dim, output_dim) for MatMul
    dense_weight = initialize_weights((current_channels, embedding_dim), dense_weight_name, "xavier")
    dense_bias = initialize_weights((embedding_dim,), dense_bias_name, "zeros")
    
    initializers.append(create_initializer_tensor(dense_weight_name, dense_weight))
    initializers.append(create_initializer_tensor(dense_bias_name, dense_bias))
    
    dense_nodes = make_dense_node(
        name="dense",
        input_name=flatten_output,
        output_name=dense_output,
        weight_name=dense_weight_name,
        bias_name=dense_bias_name
    )
    nodes.extend(dense_nodes)
    
    # L2 Normalization
    l2_nodes, l2_initializers = make_l2_normalize_node(
        name="l2_normalize",
        input_name=dense_output,
        output_name=output_name,
        axis=-1
    )
    nodes.extend(l2_nodes)
    initializers.extend(l2_initializers)
    
    # Create graph
    graph_inputs = [make_value_info(input_name, [None, input_channels, 128, None], TensorProto.FLOAT)]
    graph_outputs = [make_value_info(output_name, [None, embedding_dim], TensorProto.FLOAT)]
    
    graph_def = helper.make_graph(
        nodes=nodes,
        name="MusicEncoder",
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=initializers
    )
    
    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name="AudioMuse-AI Student Training",
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    # Check model
    try:
        onnx.checker.check_model(model_def)
        logger.info("Music encoder model created successfully")
    except Exception as e:
        logger.error(f"Model check failed: {e}")
        raise
    
    # Save if path provided
    if output_path:
        onnx.save(model_def, output_path)
        logger.info(f"Music encoder saved to {output_path}")
    
    return model_def


if __name__ == "__main__":
    # Test model building
    logging.basicConfig(level=logging.INFO)
    
    test_config = {
        'music_encoder': {
            'input_channels': 1,
            'conv_filters': [32, 64, 128],
            'kernel_sizes': [[3, 3], [3, 3], [3, 3]],
            'pool_sizes': [[2, 2], [2, 2], [2, 2]],
            'embedding_dim': 256,
            'dropout': 0.1
        }
    }
    
    model = build_music_encoder(test_config, "test_music_encoder.onnx")
    print(f"Model created with {len(model.graph.node)} nodes")
