"""
Shared utilities for building ONNX models programmatically.
Provides helper functions for creating common layers and operations.
"""

import logging
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def make_conv2d_node(name: str,
                     input_name: str,
                     output_name: str,
                     weight_name: str,
                     bias_name: str,
                     in_channels: int,
                     out_channels: int,
                     kernel_size: Tuple[int, int],
                     stride: Tuple[int, int] = (1, 1),
                     padding: str = "SAME_UPPER") -> onnx.NodeProto:
    """
    Create a Conv2D node.
    
    Args:
        name: Node name
        input_name: Input tensor name
        output_name: Output tensor name
        weight_name: Weight tensor name
        bias_name: Bias tensor name
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size (height, width)
        stride: Stride (height, width)
        padding: Padding mode ("SAME_UPPER", "VALID")
        
    Returns:
        ONNX Conv node
    """
    return helper.make_node(
        'Conv',
        inputs=[input_name, weight_name, bias_name],
        outputs=[output_name],
        name=name,
        kernel_shape=list(kernel_size),
        strides=list(stride),
        pads=[0, 0, 0, 0] if padding == "VALID" else None,
        auto_pad=padding if padding != "VALID" else None
    )


def make_batchnorm_node(name: str,
                       input_name: str,
                       output_name: str,
                       scale_name: str,
                       bias_name: str,
                       mean_name: str,
                       var_name: str,
                       epsilon: float = 1e-5) -> onnx.NodeProto:
    """
    Create a BatchNormalization node.
    
    Args:
        name: Node name
        input_name: Input tensor name
        output_name: Output tensor name
        scale_name: Scale (gamma) tensor name
        bias_name: Bias (beta) tensor name
        mean_name: Running mean tensor name
        var_name: Running variance tensor name
        epsilon: Epsilon for numerical stability
        
    Returns:
        ONNX BatchNormalization node
    """
    return helper.make_node(
        'BatchNormalization',
        inputs=[input_name, scale_name, bias_name, mean_name, var_name],
        outputs=[output_name],
        name=name,
        epsilon=epsilon
    )


def make_relu_node(name: str, input_name: str, output_name: str) -> onnx.NodeProto:
    """Create a ReLU activation node."""
    return helper.make_node(
        'Relu',
        inputs=[input_name],
        outputs=[output_name],
        name=name
    )


def make_maxpool_node(name: str,
                     input_name: str,
                     output_name: str,
                     kernel_size: Tuple[int, int],
                     stride: Tuple[int, int] = None) -> onnx.NodeProto:
    """
    Create a MaxPool node.
    
    Args:
        name: Node name
        input_name: Input tensor name
        output_name: Output tensor name
        kernel_size: Pool size (height, width)
        stride: Stride (height, width), defaults to kernel_size
        
    Returns:
        ONNX MaxPool node
    """
    if stride is None:
        stride = kernel_size
    
    return helper.make_node(
        'MaxPool',
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        kernel_shape=list(kernel_size),
        strides=list(stride)
    )


def make_global_avgpool_node(name: str, input_name: str, output_name: str) -> onnx.NodeProto:
    """Create a GlobalAveragePool node."""
    return helper.make_node(
        'GlobalAveragePool',
        inputs=[input_name],
        outputs=[output_name],
        name=name
    )


def make_dense_node(name: str,
                   input_name: str,
                   output_name: str,
                   weight_name: str,
                   bias_name: str) -> List[onnx.NodeProto]:
    """
    Create a fully connected (dense) layer using MatMul + Add.
    
    Args:
        name: Base name for nodes
        input_name: Input tensor name
        output_name: Output tensor name
        weight_name: Weight tensor name
        bias_name: Bias tensor name
        
    Returns:
        List of ONNX nodes (MatMul, Add)
    """
    matmul_output = f"{name}_matmul_out"
    
    matmul_node = helper.make_node(
        'MatMul',
        inputs=[input_name, weight_name],
        outputs=[matmul_output],
        name=f"{name}_matmul"
    )
    
    add_node = helper.make_node(
        'Add',
        inputs=[matmul_output, bias_name],
        outputs=[output_name],
        name=f"{name}_add"
    )
    
    return [matmul_node, add_node]


def make_l2_normalize_node(name: str, input_name: str, output_name: str, axis: int = -1) -> List[onnx.NodeProto]:
    """
    Create L2 normalization nodes.
    
    Args:
        name: Base name for nodes
        input_name: Input tensor name
        output_name: Output tensor name
        axis: Axis to normalize over
        
    Returns:
        List of ONNX nodes implementing L2 normalization
    """
    # L2 norm: x / ||x||_2
    # Implementation: x / sqrt(sum(x^2) + epsilon)
    
    square_output = f"{name}_square"
    reducesum_output = f"{name}_reducesum"
    sqrt_output = f"{name}_sqrt"
    epsilon_output = f"{name}_epsilon"
    add_eps_output = f"{name}_add_eps"
    
    # Square
    square_node = helper.make_node(
        'Mul',
        inputs=[input_name, input_name],
        outputs=[square_output],
        name=f"{name}_square"
    )
    
    # ReduceSum - In ONNX 13, axes is passed as input, not attribute
    axes_name = f"{name}_axes"
    axes_initializer = numpy_helper.from_array(np.array([axis], dtype=np.int64), axes_name)
    
    reducesum_node = helper.make_node(
        'ReduceSum',
        inputs=[square_output, axes_name],
        outputs=[reducesum_output],
        name=f"{name}_reducesum",
        keepdims=1
    )
    
    # Add epsilon for numerical stability
    epsilon_name = f"{name}_eps_const"
    epsilon_initializer = numpy_helper.from_array(np.array([1e-8], dtype=np.float32), epsilon_name)
    
    add_eps_node = helper.make_node(
        'Add',
        inputs=[reducesum_output, epsilon_name],
        outputs=[add_eps_output],
        name=f"{name}_add_eps"
    )
    
    # Sqrt
    sqrt_node = helper.make_node(
        'Sqrt',
        inputs=[add_eps_output],
        outputs=[sqrt_output],
        name=f"{name}_sqrt"
    )
    
    # Divide
    div_node = helper.make_node(
        'Div',
        inputs=[input_name, sqrt_output],
        outputs=[output_name],
        name=f"{name}_div"
    )
    
    # Return nodes and initializers
    return [square_node, reducesum_node, add_eps_node, sqrt_node, div_node], [axes_initializer, epsilon_initializer]


def make_flatten_node(name: str, input_name: str, output_name: str, axis: int = 1) -> onnx.NodeProto:
    """Create a Flatten node."""
    return helper.make_node(
        'Flatten',
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        axis=axis
    )


def make_dropout_node(name: str, input_name: str, output_name: str, ratio: float = 0.5) -> onnx.NodeProto:
    """
    Create a Dropout node.
    
    Args:
        name: Node name
        input_name: Input tensor name
        output_name: Output tensor name
        ratio: Dropout probability
        
    Returns:
        ONNX Dropout node
    """
    return helper.make_node(
        'Dropout',
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        ratio=ratio
    )


def initialize_weights(shape: Tuple[int, ...], name: str, initializer: str = "xavier") -> np.ndarray:
    """
    Initialize weights with specified strategy.
    
    Args:
        shape: Weight tensor shape
        name: Weight name (for logging)
        initializer: Initialization strategy ("xavier", "he", "normal", "zeros")
        
    Returns:
        Initialized weight array
    """
    if initializer == "xavier":
        # Xavier/Glorot initialization
        fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
        fan_out = shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        weights = np.random.uniform(-limit, limit, shape).astype(np.float32)
    
    elif initializer == "he":
        # He initialization
        fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
        std = np.sqrt(2.0 / fan_in)
        weights = np.random.normal(0, std, shape).astype(np.float32)
    
    elif initializer == "normal":
        weights = np.random.normal(0, 0.01, shape).astype(np.float32)
    
    elif initializer == "zeros":
        weights = np.zeros(shape, dtype=np.float32)
    
    else:
        raise ValueError(f"Unknown initializer: {initializer}")
    
    logger.debug(f"Initialized {name} with shape {shape} using {initializer}")
    return weights


def create_initializer_tensor(name: str, tensor_array: np.ndarray) -> onnx.TensorProto:
    """
    Create an ONNX initializer tensor from numpy array.
    
    Args:
        name: Tensor name
        tensor_array: Numpy array
        
    Returns:
        ONNX TensorProto
    """
    return numpy_helper.from_array(tensor_array, name=name)


def make_value_info(name: str, shape: List[Optional[int]], dtype: int = TensorProto.FLOAT) -> onnx.ValueInfoProto:
    """
    Create a ValueInfo for model input/output.
    
    Args:
        name: Tensor name
        shape: Tensor shape (use None for dynamic dimensions)
        dtype: Data type (TensorProto constant)
        
    Returns:
        ONNX ValueInfoProto
    """
    return helper.make_tensor_value_info(
        name,
        dtype,
        shape
    )
