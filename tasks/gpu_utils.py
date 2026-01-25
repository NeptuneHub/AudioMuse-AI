# tasks/gpu_utils.py
"""
Unified GPU detection and provider selection for AudioMuse-AI.

This module provides centralized GPU backend detection supporting:
- NVIDIA CUDA (via libcuda.so.1)
- AMD ROCm (via libamdhip64.so)

Environment override: GPU_BACKEND=cuda|rocm|none

Usage:
    from tasks.gpu_utils import detect_gpu_backend, get_onnx_providers, is_gpu_clustering_available, GPUBackend

    backend = detect_gpu_backend()
    providers = get_onnx_providers()
"""

import ctypes
import logging
import os
from enum import Enum
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Cached GPU backend detection result
_cached_backend = None
_detection_done = False


class GPUBackend(Enum):
    """Supported GPU backends."""
    NONE = "none"
    CUDA = "cuda"
    ROCM = "rocm"


def _check_cuda_driver_available() -> bool:
    """
    Check if CUDA driver is available using low-level libcuda.so.
    This MUST be called before importing cupy/cuml to avoid corrupting CUDA state.
    """
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        # Must initialize CUDA driver first
        init_result = cuda.cuInit(0)
        if init_result != 0:
            return False
        # Then get device count
        device_count = ctypes.c_int()
        result = cuda.cuDeviceGetCount(ctypes.byref(device_count))
        return result == 0 and device_count.value > 0
    except Exception:
        return False


def _check_rocm_driver_available() -> bool:
    """
    Check if ROCm/HIP driver is available using libamdhip64.so.
    """
    try:
        hip = ctypes.CDLL('libamdhip64.so')
        # Get device count
        device_count = ctypes.c_int()
        result = hip.hipGetDeviceCount(ctypes.byref(device_count))
        # hipSuccess = 0
        return result == 0 and device_count.value > 0
    except Exception:
        return False


def detect_gpu_backend() -> GPUBackend:
    """
    Detect the available GPU backend.

    Checks environment variable GPU_BACKEND first, then probes for hardware:
    1. CUDA (NVIDIA) via libcuda.so.1
    2. ROCm (AMD) via libamdhip64.so

    Results are cached after first detection.

    Returns:
        GPUBackend enum value (CUDA, ROCM, or NONE)
    """
    global _cached_backend, _detection_done

    if _detection_done:
        return _cached_backend

    # Check for environment override
    env_backend = os.environ.get('GPU_BACKEND', '').lower().strip()
    if env_backend:
        if env_backend == 'cuda':
            if _check_cuda_driver_available():
                _cached_backend = GPUBackend.CUDA
                logger.info("GPU backend: CUDA (set via GPU_BACKEND environment variable)")
            else:
                logger.warning("GPU_BACKEND=cuda but CUDA driver not available, falling back to NONE")
                _cached_backend = GPUBackend.NONE
        elif env_backend == 'rocm':
            if _check_rocm_driver_available():
                _cached_backend = GPUBackend.ROCM
                logger.info("GPU backend: ROCm (set via GPU_BACKEND environment variable)")
            else:
                logger.warning("GPU_BACKEND=rocm but ROCm driver not available, falling back to NONE")
                _cached_backend = GPUBackend.NONE
        elif env_backend == 'none':
            _cached_backend = GPUBackend.NONE
            logger.info("GPU backend: NONE (set via GPU_BACKEND environment variable)")
        else:
            logger.warning(f"Unknown GPU_BACKEND value: {env_backend}, auto-detecting...")
            env_backend = None  # Fall through to auto-detection

    # Auto-detect if no valid environment override
    if not env_backend:
        if _check_cuda_driver_available():
            _cached_backend = GPUBackend.CUDA
            logger.info("GPU backend: CUDA (auto-detected via libcuda.so.1)")
        elif _check_rocm_driver_available():
            _cached_backend = GPUBackend.ROCM
            logger.info("GPU backend: ROCm (auto-detected via libamdhip64.so)")
        else:
            _cached_backend = GPUBackend.NONE
            logger.info("GPU backend: NONE (no GPU driver detected)")

    _detection_done = True
    return _cached_backend


def get_onnx_providers() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Get the appropriate ONNX Runtime execution providers based on detected GPU backend.

    Returns:
        List of (provider_name, options) tuples suitable for ort.InferenceSession()
    """
    import onnxruntime as ort

    backend = detect_gpu_backend()
    available_providers = ort.get_available_providers()

    if backend == GPUBackend.CUDA and 'CUDAExecutionProvider' in available_providers:
        gpu_device_id = 0
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible != '-1':
            gpu_device_id = 0  # CUDA_VISIBLE_DEVICES remaps to device 0

        cuda_options = {
            'device_id': gpu_device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'DEFAULT',
        }
        logger.info(f"ONNX providers: CUDAExecutionProvider (device_id={gpu_device_id}), CPUExecutionProvider")
        return [('CUDAExecutionProvider', cuda_options), ('CPUExecutionProvider', {})]

    elif backend == GPUBackend.ROCM and 'ROCmExecutionProvider' in available_providers:
        gpu_device_id = 0
        rocm_visible = os.environ.get('ROCR_VISIBLE_DEVICES', '')
        if rocm_visible and rocm_visible != '-1':
            gpu_device_id = 0  # ROCR_VISIBLE_DEVICES remaps to device 0

        rocm_options = {
            'device_id': gpu_device_id,
        }
        logger.info(f"ONNX providers: ROCmExecutionProvider (device_id={gpu_device_id}), CPUExecutionProvider")
        return [('ROCmExecutionProvider', rocm_options), ('CPUExecutionProvider', {})]

    else:
        logger.info("ONNX providers: CPUExecutionProvider only")
        return [('CPUExecutionProvider', {})]


def get_onnx_provider_names() -> List[str]:
    """
    Get just the provider names (without options) for ONNX Runtime.

    Returns:
        List of provider name strings
    """
    providers = get_onnx_providers()
    return [p[0] for p in providers]


def get_onnx_provider_options() -> List[Dict[str, Any]]:
    """
    Get just the provider options for ONNX Runtime.

    Returns:
        List of provider option dicts
    """
    providers = get_onnx_providers()
    return [p[1] for p in providers]


def is_gpu_clustering_available() -> bool:
    """
    Check if GPU-accelerated clustering (cuML/RAPIDS) is available.

    cuML is only available for NVIDIA CUDA. ROCm users must use sklearn fallback.

    Returns:
        True only if CUDA backend is detected (cuML may be available)
    """
    backend = detect_gpu_backend()

    if backend == GPUBackend.ROCM:
        logger.info("GPU clustering not available: cuML (RAPIDS) is NVIDIA-only. Using sklearn fallback for ROCm.")
        return False

    if backend == GPUBackend.NONE:
        return False

    # CUDA detected - cuML might be available (actual availability checked in clustering_gpu.py)
    return True


def get_active_provider_name() -> str:
    """
    Get a human-readable name for the currently active GPU backend.

    Returns:
        String like "CUDA", "ROCm", or "CPU"
    """
    backend = detect_gpu_backend()
    if backend == GPUBackend.CUDA:
        return "CUDA"
    elif backend == GPUBackend.ROCM:
        return "ROCm"
    else:
        return "CPU"


def reset_detection_cache() -> None:
    """
    Reset the cached GPU detection result.
    Useful for testing or when GPU configuration changes.
    """
    global _cached_backend, _detection_done
    _cached_backend = None
    _detection_done = False
    logger.debug("GPU detection cache reset")
