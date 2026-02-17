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
      1. CUDAExecutionProvider (if available and not blacklisted)
      2. MPSExecutionProvider (if available and not blacklisted)
      3. CoreMLExecutionProvider (if available and not blacklisted)
      4. CPUExecutionProvider (always present as fallback)

    Providers can be disabled at runtime via `disable_onnx_provider()` or by
    setting the `ONNX_PROVIDER_BLACKLIST` environment variable. This allows
    automatic disabling when providers fail (e.g. CoreML dynamic-resize).

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

    # Support runtime blacklisting via env var or in-memory blacklist
    env_blacklist = set()
    env_val = os.environ.get('ONNX_PROVIDER_BLACKLIST', '')
    if env_val:
        env_blacklist = set([p.strip() for p in env_val.split(',') if p.strip()])

    # Internal in-memory blacklist (set by runtime when providers fail)
    global _disabled_providers
    try:
        _ = _disabled_providers  # referenced below
    except NameError:
        _disabled_providers = set()

    def provider_allowed(name: str) -> bool:
        return name not in env_blacklist and name not in _disabled_providers

    # 1) CUDA
    if 'CUDAExecutionProvider' in available and provider_allowed('CUDAExecutionProvider'):
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
    if 'MPSExecutionProvider' in available and provider_allowed('MPSExecutionProvider'):
        opts.append(('MPSExecutionProvider', {}))
    elif 'CoreMLExecutionProvider' in available and provider_allowed('CoreMLExecutionProvider'):
        # Some ONNX Runtime mac builds expose CoreMLExecutionProvider instead
        opts.append(('CoreMLExecutionProvider', {}))

    # 3) Always include CPU as final fallback
    if provider_allowed('CPUExecutionProvider'):
        opts.append(('CPUExecutionProvider', {}))

    logger.debug(f"Preferred ONNX providers: {[p for p, _ in opts]}")
    return opts


# Runtime control helpers --------------------------------------------------
def disable_onnx_provider(provider_name: str) -> None:
    """Disable an ONNX provider at runtime (affects subsequent calls).

    Example: disable_onnx_provider('CoreMLExecutionProvider')
    """
    global _disabled_providers
    try:
        _ = _disabled_providers
    except NameError:
        _disabled_providers = set()
    _disabled_providers.add(provider_name)
    logger.warning(f"ONNX provider disabled at runtime: {provider_name}")


def is_onnx_provider_disabled(provider_name: str) -> bool:
    try:
        return provider_name in _disabled_providers
    except Exception:
        return False
