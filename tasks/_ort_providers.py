"""Canonical ONNX Runtime provider selection.

Single source of truth for "what providers should we hand to
``ort.InferenceSession``?". Used by:

* ``tasks.analysis_helper`` (MusiCNN + analysis pipeline)
* ``lyrics.embeddings`` (e5)
* ``lyrics.silero_onnx`` (VAD)
* ``lyrics.translation_onnx`` (Marian)
* ``lyrics.whisper_onnx`` (Whisper)

On GPU images (``onnxruntime-gpu`` installed via ``requirements/gpu.txt``)
sessions transparently use CUDA with a CPU fallback. On CPU images they
keep using the CPU provider as before.

The ``CUDAExecutionProvider`` options mirror what the MusiCNN pipeline has
historically used (``arena_extend_strategy=kSameAsRequested`` to keep
fragmentation down, ``EXHAUSTIVE`` cudnn convolution search, etc.) so all
GPU-bound sessions get consistent tuning.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import onnxruntime as ort


_CUDA_OPTIONS: Dict[str, object] = {
    'device_id': 0,
    'arena_extend_strategy': 'kSameAsRequested',
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}


def pick_provider_options() -> List[Tuple[str, Dict]]:
    """Return ordered ``(provider, options)`` pairs.

    Prefers ``CUDAExecutionProvider`` when ``onnxruntime-gpu`` reports it
    available and always appends ``CPUExecutionProvider`` as a safety
    fallback so a failed CUDA init doesn't crash the load.

    Recomputed every call (no module-level cache) so unit tests can
    monkey-patch ``ort.get_available_providers`` without leaking state
    across test cases.
    """
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        return [
            ('CUDAExecutionProvider', dict(_CUDA_OPTIONS)),
            ('CPUExecutionProvider', {}),
        ]
    return [('CPUExecutionProvider', {})]


def pick_providers() -> List[str]:
    """Return ordered provider name list (no per-provider options).

    Use this when the InferenceSession constructor is called without the
    ``provider_options`` argument (most lyrics ONNX modules).
    """
    return [name for name, _ in pick_provider_options()]
