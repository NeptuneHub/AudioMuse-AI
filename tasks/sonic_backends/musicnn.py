# tasks/sonic_backends/musicnn.py
"""Default sonic backend: MusiCNN embedding + MusiCNN-prediction tag head.

This is the historical AudioMuse path lifted out of ``tasks/analysis.py``
so existing Voyager indexes, ``embedding`` table rows and
``score.mood_vector`` strings remain valid when this backend is
selected. It is the default — set ``SONIC_BACKEND=musicnn`` or leave the
variable unset to keep behavior unchanged.

The OOM-fallback dance (drop the OOM'd GPU session, allocate a CPU
session, rewire the shared album-level dict so subsequent tracks pick
up the fallback) lives here rather than in the orchestrator because it
is MusiCNN-specific: MERT-style backends do their fallback inside
``torch`` instead.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import analysis_helper as _ah
from .base import SonicBackend, register

logger = logging.getLogger(__name__)


class MusiCNNBackend(SonicBackend):
    name = "musicnn"
    embedding_dim = 200
    target_sr = 16000

    def __init__(self) -> None:
        # Defer config import to instance time so unit tests that stub
        # ``config`` after import still pick up the active values.
        from config import EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH
        self._model_paths = {
            "embedding": EMBEDDING_MODEL_PATH,
            "prediction": PREDICTION_MODEL_PATH,
        }

    # --- session management ------------------------------------------------

    def load_sessions(self) -> Dict[str, Any]:
        sessions = _ah.load_musicnn_sessions(self._model_paths)
        if sessions is None:
            raise RuntimeError("Failed to load MusiCNN ONNX sessions")
        return sessions

    def cleanup_sessions(self, sessions: Dict[str, Any], context: str = "") -> None:
        try:
            _ah.cleanup_musicnn_sessions(sessions, context=context)
        except Exception as e:  # noqa: BLE001
            logger.warning("MusiCNN cleanup raised (suppressed): %s", e)

    # --- per-track analysis ------------------------------------------------

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        sessions: Optional[Dict[str, Any]],
        *,
        file_basename: str,
        mood_labels: List[str],
    ) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        embeddings_per_patch = _run_musicnn(
            audio=audio, sr=sr,
            sessions=sessions, model_paths=self._model_paths,
            file_basename=file_basename,
        )
        if embeddings_per_patch is None:
            return None
        embeddings_per_patch, mood_logits = embeddings_per_patch
        # Replicate the historical "double sigmoid" mood pipeline (see
        # tasks.analysis comment block — old Essentia ONNX baked sigmoid
        # in; the new prediction ONNX outputs raw logits, and downstream
        # code is calibrated for sigmoid(mean(sigmoid(logits)))).
        mood_probs_per_patch = _ah.sigmoid(mood_logits)
        final_mood_predictions = _ah.sigmoid(np.mean(mood_probs_per_patch, axis=0))
        moods = {
            label: float(score)
            for label, score in zip(mood_labels, final_mood_predictions)
        }
        track_embedding = np.mean(embeddings_per_patch, axis=0).astype(np.float32, copy=False)
        return track_embedding, moods


def _run_musicnn(
    *,
    audio: np.ndarray,
    sr: int,
    sessions: Optional[Dict[str, Any]],
    model_paths: Dict[str, str],
    file_basename: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Inner MusiCNN inference: ``audio -> (embeddings_per_patch, mood_logits)``.

    Returns ``None`` when the track is too short to form a single
    spectrogram patch (caller should skip the track silently). Raises
    on hard model failures.

    Exposed as a module-level function (rather than a method) so the
    MERT backend can reuse it to keep MusiCNN-quality mood tags while
    substituting only the stored similarity embedding.
    """
    try:
        final_patches = _ah.prepare_spectrogram_patches(audio, sr)
    except Exception as e:
        logger.exception("Spectrogram creation failed for %s: %s", file_basename, e)
        return None
    if final_patches is None:
        logger.warning("Track too short to create spectrogram patches: %s", file_basename)
        return None

    owns_sessions = sessions is None
    if owns_sessions:
        provider_options = _ah.get_provider_options()
        sessions = {
            "embedding": _ah.create_onnx_session(model_paths["embedding"], provider_options, label="embedding"),
            "prediction": _ah.create_onnx_session(model_paths["prediction"], provider_options, label="prediction"),
        }

    try:
        embedding_sess = sessions["embedding"]
        prediction_sess = sessions["prediction"]

        original_embedding_sess = embedding_sess
        original_prediction_sess = prediction_sess

        embedding_feed = {_ah.DEFINED_TENSOR_NAMES["embedding"]["input"]: final_patches}
        embeddings_per_patch, embedding_sess = _ah.run_inference_with_oom_fallback(
            embedding_sess, embedding_feed,
            _ah.DEFINED_TENSOR_NAMES["embedding"]["output"],
            model_paths["embedding"], "embedding",
            owns_sessions, file_basename,
        )
        if embedding_sess is not original_embedding_sess:
            sessions["embedding"] = embedding_sess
            original_embedding_sess = None  # release the OOM'd GPU buffer NOW

        prediction_feed = {_ah.DEFINED_TENSOR_NAMES["prediction"]["input"]: embeddings_per_patch}
        mood_logits, prediction_sess = _ah.run_inference_with_oom_fallback(
            prediction_sess, prediction_feed,
            _ah.DEFINED_TENSOR_NAMES["prediction"]["output"],
            model_paths["prediction"], "prediction",
            owns_sessions, file_basename,
        )
        if prediction_sess is not original_prediction_sess:
            sessions["prediction"] = prediction_sess
            original_prediction_sess = None

        return embeddings_per_patch, mood_logits
    finally:
        if owns_sessions:
            try:
                _ah.cleanup_musicnn_sessions(sessions, context=file_basename)
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning("Error during single-track MusiCNN cleanup: %s", cleanup_error)


register(MusiCNNBackend())
