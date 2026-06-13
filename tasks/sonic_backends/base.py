# tasks/sonic_backends/base.py
"""Pluggable sonic-analysis backend interface.

Each backend owns the entire ``(audio, sr) -> (track_embedding, moods)``
pipeline for a single track. The default backend (``musicnn``) reproduces
the historical Essentia-lineage path bit-for-bit so existing Voyager
indexes, ``embedding`` rows and ``score.mood_vector`` strings stay valid.
Alternative backends (e.g. ``mert``) can substitute the embedding (and
optionally the tagging head) without the rest of the analysis pipeline
needing to care.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SonicBackend(ABC):
    """Backend for sonic embedding + mood-tag prediction.

    Lifecycle:
      * ``load_sessions()`` is called once per album (or once per track
        when ``PER_SONG_MODEL_RELOAD`` is set). The returned dict is
        opaque to callers — backends can stash ONNX sessions, torch
        modules, tokenizers, anything they need.
      * ``analyze()`` is called per track and returns the per-track
        embedding + a ``{mood_label: score}`` dict (or ``None`` on
        failure).
      * ``cleanup_sessions()`` releases the resources allocated by
        ``load_sessions()``.

    The ``target_sr`` class attribute tells the orchestrator at what
    sample rate to load audio. Backends that need a different rate
    internally should resample inside ``analyze()`` rather than forcing
    the orchestrator to round-trip to disk.
    """

    name: str = ""
    embedding_dim: int = 0
    target_sr: int = 16000

    @abstractmethod
    def load_sessions(self) -> Dict[str, Any]:
        """Allocate model resources. May raise on permanent failure."""

    @abstractmethod
    def cleanup_sessions(self, sessions: Dict[str, Any], context: str = "") -> None:
        """Release model resources. Must never raise."""

    @abstractmethod
    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        sessions: Optional[Dict[str, Any]],
        *,
        file_basename: str,
        mood_labels: List[str],
    ) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        """Run the full per-track pipeline.

        Returns ``(track_embedding, moods)`` where ``track_embedding`` is
        a 1-D float32 array of length ``embedding_dim`` and ``moods`` is
        ``{mood_label: probability}`` aligned with ``mood_labels``.

        Returns ``None`` on per-track failures the caller should skip.
        Hard failures (model load errors, etc.) should raise.
        """


_REGISTRY: Dict[str, SonicBackend] = {}


def register(backend: SonicBackend) -> SonicBackend:
    """Register a backend instance under its ``name``."""
    if not backend.name:
        raise ValueError("Backend must define a non-empty .name")
    _REGISTRY[backend.name] = backend
    return backend


def get_backend(name: str) -> SonicBackend:
    """Return the singleton backend instance for ``name``.

    Importing this module triggers backend module imports lazily via
    ``tasks.sonic_backends`` so callers do not need to worry about
    registration order.
    """
    if name not in _REGISTRY:
        from . import _ensure_loaded  # noqa: WPS433 — lazy load on demand
        _ensure_loaded()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise ValueError(
            f"Unknown SONIC_BACKEND '{name}'. Available backends: {available}"
        )
    return _REGISTRY[name]


def available_backends() -> List[str]:
    from . import _ensure_loaded
    _ensure_loaded()
    return sorted(_REGISTRY)
