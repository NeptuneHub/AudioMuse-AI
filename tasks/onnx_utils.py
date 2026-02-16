"""ONNX Runtime provider utilities.

Centralizes provider selection logic so the rest of the codebase can
consistently try GPU providers in the preferred order (CUDA -> MPS -> CPU).

Functions:
- get_preferred_onnx_provider_options() -> list[tuple[str, dict]]

This keeps provider-selection logic in one place and makes it easy to
extend in the future (e.g. add CoreMLExecutionProvider, DirectML, etc.).
"""
from __future__ import annotations

import os
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def get_preferred_onnx_provider_options() -> List[Tuple[str, Dict]]:
    """Return ONNX provider options in preferred order.

    Preferred order:
      1. CUDAExecutionProvider (if available)
      2. MPSExecutionProvider (if available)
      3. CPUExecutionProvider (always present as fallback)

    Each entry is a tuple (provider_name, provider_options_dict).
    """
    try:
        import onnxruntime as ort
    except Exception:
        # If onnxruntime isn't importable the caller will soon fail when
        # trying to create sessions; return CPU fallback to keep behaviour
        logger.debug("onnxruntime not available when selecting providers; falling back to CPU only")
        return [("CPUExecutionProvider", {})]

    available = ort.get_available_providers() or []
    opts: List[Tuple[str, Dict]] = []

    # 1) CUDA
    if 'CUDAExecutionProvider' in available:
        gpu_device_id = 0
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible != '-1':
            gpu_device_id = 0

        cuda_options = {
            'device_id': gpu_device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }
        opts.append(('CUDAExecutionProvider', cuda_options))

    # 2) Apple GPU: prefer MPSExecutionProvider, fallback to CoreMLExecutionProvider
    if 'MPSExecutionProvider' in available:
        opts.append(('MPSExecutionProvider', {}))
    elif 'CoreMLExecutionProvider' in available:
        # Some ONNX Runtime mac builds expose CoreMLExecutionProvider instead
        opts.append(('CoreMLExecutionProvider', {}))

    # 3) Always include CPU as final fallback
    opts.append(('CPUExecutionProvider', {}))

    logger.debug(f"Preferred ONNX providers: {[p for p, _ in opts]}")
    return opts
