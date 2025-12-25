"""
Build lightweight text encoder as ONNX graph.
Simplified architecture using embeddings and averaging for initial implementation.
A full transformer implementation would be more complex and is left for future enhancement.
"""

import logging
import onnx
from onnx import helper, TensorProto
import numpy as np
from typing import Dict
from models.student_utils import (
    make_dense_node, make_l2_normalize_node, make_flatten_node,
    initialize_weights, create_initializer_tensor, make_value_info
)

logger = logging.getLogger(__name__)


def build_text_encoder(config: Dict, output_path: str = None) -> onnx.ModelProto:
    """
    Build the text encoder ONNX model programmatically.
    
    Simplified Architecture (for initial implementation):
    Input (batch, seq_len) → Embedding → Mean Pooling → Dense → L2 Normalize → Output (batch, 256)
    
    Note: A full transformer implementation would require significantly more complexity.
    This simplified version provides a working baseline that can be enhanced later.
    
    Args:
        config: Model configuration dictionary
        output_path: Optional path to save the model
        
    Returns:
        ONNX ModelProto
    """
    logger.info("Building text encoder ONNX model (simplified version)")
    
    # Extract configuration
    model_config = config.get('text_encoder', {})
    vocab_size = model_config.get('vocab_size', 30000)
    embedding_dim = model_config.get('embedding_dim', 256)
    max_seq_length = model_config.get('max_seq_length', 128)
    
    # Model inputs and outputs
    input_name = "input_ids"  # Tokenized text (batch, seq_len)
    output_name = "text_embedding"  # Text embedding (batch, embedding_dim)
    
    # Create nodes and initializers
    nodes = []
    initializers = []
    
    # Note: ONNX doesn't have a native Embedding layer, so we use Gather
    # Embedding table
    embedding_table_name = "embedding_table"
    embedding_table = initialize_weights((vocab_size, embedding_dim), embedding_table_name, "normal")
    initializers.append(create_initializer_tensor(embedding_table_name, embedding_table))
    
    # Gather (embedding lookup)
    gather_output = "embedded"
    gather_node = helper.make_node(
        'Gather',
        inputs=[embedding_table_name, input_name],
        outputs=[gather_output],
        name="embedding_lookup",
        axis=0
    )
    nodes.append(gather_node)
    
    # Mean pooling over sequence dimension
    # ReduceMean over axis=1 (sequence dimension)
    mean_pool_output = "mean_pooled"
    mean_pool_node = helper.make_node(
        'ReduceMean',
        inputs=[gather_output],
        outputs=[mean_pool_output],
        name="mean_pool",
        axes=[1],
        keepdims=0
    )
    nodes.append(mean_pool_node)
    
    # Dense layer (projection)
    dense_output = "dense_out"
    dense_weight_name = "dense_weight"
    dense_bias_name = "dense_bias"
    
    # Initialize Dense weights
    dense_weight = initialize_weights((embedding_dim, embedding_dim), dense_weight_name, "xavier")
    dense_bias = initialize_weights((embedding_dim,), dense_bias_name, "zeros")
    
    initializers.append(create_initializer_tensor(dense_weight_name, dense_weight))
    initializers.append(create_initializer_tensor(dense_bias_name, dense_bias))
    
    dense_nodes = make_dense_node(
        name="dense",
        input_name=mean_pool_output,
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
    graph_inputs = [make_value_info(input_name, [None, None], TensorProto.INT64)]
    graph_outputs = [make_value_info(output_name, [None, embedding_dim], TensorProto.FLOAT)]
    
    graph_def = helper.make_graph(
        nodes=nodes,
        name="TextEncoder",
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
        logger.info("Text encoder model created successfully")
    except Exception as e:
        logger.error(f"Model check failed: {e}")
        raise
    
    # Save if path provided
    if output_path:
        onnx.save(model_def, output_path)
        logger.info(f"Text encoder saved to {output_path}")
    
    return model_def


if __name__ == "__main__":
    # Test model building
    logging.basicConfig(level=logging.INFO)
    
    test_config = {
        'text_encoder': {
            'vocab_size': 30000,
            'embedding_dim': 256,
            'max_seq_length': 128
        }
    }
    
    model = build_text_encoder(test_config, "test_text_encoder.onnx")
    print(f"Model created with {len(model.graph.node)} nodes")
