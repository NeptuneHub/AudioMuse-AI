"""Centralized ONNX Runtime execution-provider selection.

All MusiCNN / CLAP / MuLan / memory-pool callers go through this helper so
provider preference (ROCm > CUDA > CPU) is defined in exactly one place.

The CUDA provider options dict matches the values previously hard-coded in
``tasks.analysis_helper.get_provider_options`` so NVIDIA behavior is
preserved byte-for-byte. CLAP / MuLan previously used a slightly different
``cudnn_conv_algo_search`` value ("DEFAULT") in their inline blocks; we keep
the MusiCNN value ("EXHAUSTIVE") because it is the documented preference for
this workload and the per-model difference looked accidental.
"""

import logging
from typing import List, Tuple, Dict

import onnxruntime as ort

logger = logging.getLogger(__name__)

ProviderSpec = Tuple[str, Dict]

_CUDA_OPTIONS: Dict = {
    'device_id': 0,
    'arena_extend_strategy': 'kSameAsRequested',
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}

_ROCM_OPTIONS: Dict = {
    'device_id': 0,
}


def select_providers(model_label: str = "") -> List[ProviderSpec]:
    """Return ordered ``[(provider_name, options), ...]`` for InferenceSession.

    Order: ROCm > CUDA > CPU. CPU is always appended as a fallback when a
    GPU provider is selected.
    """
    available = ort.get_available_providers()
    label = f" for {model_label}" if model_label else ""

    if 'ROCMExecutionProvider' in available:
        logger.info(f"ROCm provider available - using AMD GPU{label}")
        return [('ROCMExecutionProvider', _ROCM_OPTIONS), ('CPUExecutionProvider', {})]

    if 'CUDAExecutionProvider' in available:
        logger.info(f"CUDA provider available - using NVIDIA GPU{label}")
        return [('CUDAExecutionProvider', _CUDA_OPTIONS), ('CPUExecutionProvider', {})]

    logger.info(f"No GPU provider available - using CPU{label}")
    return [('CPUExecutionProvider', {})]


def preferred_provider_name() -> str:
    """Return the first provider that ``select_providers`` would pick."""
    return select_providers()[0][0]


def log_provider_summary() -> None:
    """Emit a single INFO line summarizing ONNX provider availability.

    Intended to be called once at process startup (after logging is
    configured) so operators can see at a glance which provider will be
    used without grepping per-model log lines.
    """
    available = ort.get_available_providers()
    preferred = preferred_provider_name()
    logger.info(
        f"ONNX Runtime providers available: {available}; preferred: {preferred}"
    )
