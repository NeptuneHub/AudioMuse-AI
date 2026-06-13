# tasks/sonic_backends/mert.py
"""MERT-based sonic backend (self-supervised music foundation model).

MERT (`m-a-p/MERT-v1-95M` / `-330M`) replaces the MusiCNN CNN as the
source of the *similarity* embedding stored in the ``embedding`` table
and indexed by Voyager. Mood tagging is intentionally still handled by
the existing MusiCNN-prediction ONNX head: it consumes MusiCNN
embeddings, so swapping it requires either training a new head on MERT
features (out of scope for this backend) or using a zero-shot text
scorer like CLAP. Keeping the legacy tag head means ``mood_vector`` /
``score.mood_vector`` rows stay consistent with what the rest of the
codebase expects (Song Alchemy mood priors, CLAP "other features"
zero-shot scoring, the UI filters, etc.).

Backend selection is controlled by the ``SONIC_BACKEND`` env var. When
``mert`` is selected:

  * The ``embedding`` column stores a 768-dim (MERT-95M) or 1024-dim
    (MERT-330M) float32 vector instead of 200-dim MusiCNN.
  * ``EMBEDDING_DIMENSION`` in :mod:`config` resolves to the MERT dim,
    so the Voyager index is built at the correct size.
  * Existing 200-dim Voyager indexes / ``embedding`` rows are NOT
    forwards-compatible — switching backends requires re-running
    analysis from scratch. See the release notes pattern used for the
    v2.0 lyrics model swap.

Audio is resampled inside this backend (MERT wants 24 kHz mono); the
orchestrator still loads at ``target_sr`` = 24000 to skip the
re-resample, but ``analyze()`` also accepts 16 kHz audio and resamples
on the fly so the same audio buffer can feed both the MERT embedding
and the (16 kHz) MusiCNN tag head without a second decode.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import SonicBackend, register
from .musicnn import _run_musicnn

logger = logging.getLogger(__name__)

# Imported at use time so the module can register itself even when torch
# / transformers are not installed (selection-time error is friendlier
# than import-time crash).
_torch = None
_transformers = None
_librosa = None


def _lazy_imports() -> None:
    global _torch, _transformers, _librosa
    if _torch is None:
        import torch  # noqa: WPS433
        _torch = torch
    if _transformers is None:
        import transformers  # noqa: WPS433
        _transformers = transformers
    if _librosa is None:
        import librosa  # noqa: WPS433
        _librosa = librosa


# Dimensions of the public MERT-v1 checkpoints, used to pre-declare
# ``embedding_dim`` before any model weights have been downloaded.
_KNOWN_DIMS = {
    "m-a-p/MERT-v1-95M": 768,
    "m-a-p/MERT-v1-330M": 1024,
}


class MERTBackend(SonicBackend):
    name = "mert"

    def __init__(self) -> None:
        from config import (
            EMBEDDING_MODEL_PATH, PREDICTION_MODEL_PATH,
            MERT_MODEL_ID, MERT_LAYER, MERT_DEVICE,
            MERT_HF_CACHE_DIR, MERT_TARGET_SR,
            MERT_CHUNK_SECONDS, MERT_MIN_CHUNK_SECONDS,
        )
        self._musicnn_paths = {
            "embedding": EMBEDDING_MODEL_PATH,
            "prediction": PREDICTION_MODEL_PATH,
        }
        self._model_id = MERT_MODEL_ID
        self._layer = MERT_LAYER
        self._device_pref = MERT_DEVICE
        self._cache_dir = MERT_HF_CACHE_DIR or None
        self.target_sr = int(MERT_TARGET_SR)
        self._chunk_seconds = float(MERT_CHUNK_SECONDS)
        self._min_chunk_seconds = float(MERT_MIN_CHUNK_SECONDS)
        if self._model_id in _KNOWN_DIMS:
            self.embedding_dim = _KNOWN_DIMS[self._model_id]
        else:
            # Unknown checkpoint — fall back to MERT-95M-style 768. Hard
            # error happens at first inference if the real dim differs.
            self.embedding_dim = 768
            logger.warning(
                "Unknown MERT model id '%s' — assuming embedding_dim=768. "
                "Set MERT_MODEL_ID to one of %s for an explicit value.",
                self._model_id, sorted(_KNOWN_DIMS),
            )

    # --- session management ------------------------------------------------

    def load_sessions(self) -> Dict[str, Any]:
        _lazy_imports()
        device = self._resolve_device()
        logger.info(
            "Loading MERT model '%s' on %s (layer=%s)",
            self._model_id, device, self._layer,
        )
        kwargs = {"trust_remote_code": True}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir
        processor = _transformers.Wav2Vec2FeatureExtractor.from_pretrained(
            self._model_id, **kwargs,
        )
        model = _transformers.AutoModel.from_pretrained(
            self._model_id, **kwargs,
        )
        model.eval()
        model.to(device)
        # Validate that the configured layer index is actually reachable.
        # MERT-95M ships 12 hidden layers + the conv output; -1 (last
        # layer) is always valid.
        num_layers = getattr(model.config, "num_hidden_layers", None)
        if num_layers is not None and isinstance(self._layer, int):
            if self._layer < -1 or self._layer >= num_layers:
                logger.warning(
                    "MERT_LAYER=%s out of range for %s (num_hidden_layers=%s); "
                    "clamping to last layer.",
                    self._layer, self._model_id, num_layers,
                )
                self._layer = -1
        # MusiCNN sessions for the tag head. Reused via the same shared
        # dict semantics as the MusiCNN backend so album-level reuse
        # works identically.
        from .. import analysis_helper as _ah
        musicnn = _ah.load_musicnn_sessions(self._musicnn_paths)
        if musicnn is None:
            raise RuntimeError("Failed to load MusiCNN tag head for MERT backend")

        return {
            "mert_model": model,
            "mert_processor": processor,
            "mert_device": device,
            "mert_sr": processor.sampling_rate,
            "musicnn": musicnn,
        }

    def cleanup_sessions(self, sessions: Dict[str, Any], context: str = "") -> None:
        try:
            from .. import analysis_helper as _ah
            _ah.cleanup_musicnn_sessions(sessions.get("musicnn"), context=context)
        except Exception as e:  # noqa: BLE001
            logger.warning("MusiCNN tag-head cleanup raised (suppressed): %s", e)

        try:
            model = sessions.pop("mert_model", None)
            if model is not None:
                # Move to CPU before dropping refs so any CUDA buffers
                # held by parameters are freed deterministically.
                try:
                    model.to("cpu")
                except Exception:
                    pass
                del model
            sessions.pop("mert_processor", None)
            if _torch is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            logger.warning("MERT cleanup raised (suppressed): %s", e)

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
        if sessions is None:
            sessions = self.load_sessions()
            owns_sessions = True
        else:
            owns_sessions = False

        try:
            mert_emb = self._embed_with_mert(audio, sr, sessions, file_basename)
            if mert_emb is None:
                return None

            # MusiCNN expects 16 kHz; we either have it (orchestrator
            # passes 16 kHz when MusicNN is the only backend) or have to
            # resample down from the MERT-target rate.
            if sr == 16000:
                musicnn_audio, musicnn_sr = audio, sr
            else:
                _lazy_imports()
                musicnn_audio = _librosa.resample(
                    audio.astype(np.float32, copy=False), orig_sr=sr, target_sr=16000,
                )
                musicnn_sr = 16000

            tagging = _run_musicnn(
                audio=musicnn_audio, sr=musicnn_sr,
                sessions=sessions["musicnn"], model_paths=self._musicnn_paths,
                file_basename=file_basename,
            )
            if tagging is None:
                return None
            _, mood_logits = tagging
            from .. import analysis_helper as _ah
            mood_probs_per_patch = _ah.sigmoid(mood_logits)
            final_mood_predictions = _ah.sigmoid(np.mean(mood_probs_per_patch, axis=0))
            moods = {
                label: float(score)
                for label, score in zip(mood_labels, final_mood_predictions)
            }
            return mert_emb.astype(np.float32, copy=False), moods
        finally:
            if owns_sessions:
                self.cleanup_sessions(sessions, context=file_basename)

    # --- internals ---------------------------------------------------------

    def _resolve_device(self) -> str:
        _lazy_imports()
        pref = (self._device_pref or "auto").lower()
        if pref == "cuda" or (pref == "auto" and _torch.cuda.is_available()):
            return "cuda" if _torch.cuda.is_available() else "cpu"
        return "cpu"

    def _embed_with_mert(
        self,
        audio: np.ndarray,
        sr: int,
        sessions: Dict[str, Any],
        file_basename: str,
    ) -> Optional[np.ndarray]:
        _lazy_imports()
        torch = _torch
        processor = sessions["mert_processor"]
        model = sessions["mert_model"]
        device = sessions["mert_device"]
        target_sr = sessions.get("mert_sr") or self.target_sr

        if sr != target_sr:
            audio = _librosa.resample(
                audio.astype(np.float32, copy=False),
                orig_sr=sr, target_sr=target_sr,
            )

        if audio.size == 0:
            logger.warning("MERT got empty audio for %s", file_basename)
            return None

        # Chunk the audio into fixed-length windows and mean-pool the
        # per-chunk track embeddings. MERT was pretrained on short
        # clips (HuBERT-style), so chunking keeps inference in
        # distribution and bounds per-track latency to a constant
        # ``ceil(track_seconds / chunk_seconds)`` forward passes. A
        # trailing chunk shorter than ``min_chunk_samples`` is dropped
        # rather than padded, since its mean would just be noise from
        # the tail end. If the whole track is shorter than the min
        # chunk threshold, we still embed it as one short pass so
        # very short clips don't silently fail to produce a vector.
        chunk_samples = int(self._chunk_seconds * target_sr)
        min_chunk_samples = int(self._min_chunk_seconds * target_sr)
        layer = self._layer if isinstance(self._layer, int) else -1

        if audio.size <= chunk_samples:
            chunks = [audio]
        else:
            chunks = [
                audio[i:i + chunk_samples]
                for i in range(0, audio.size, chunk_samples)
            ]
            if len(chunks) > 1 and chunks[-1].size < min_chunk_samples:
                chunks.pop()

        chunk_vecs = []
        for chunk in chunks:
            inputs = processor(
                chunk, sampling_rate=target_sr, return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            layer_out = outputs.hidden_states[layer]  # (1, T, D)
            chunk_vecs.append(
                layer_out.mean(dim=1).squeeze(0).detach().cpu().numpy()
            )

        if not chunk_vecs:
            return None
        # Mean-pool across chunks. Equal weighting is the standard
        # MIR choice — both volume-weighted and energy-weighted
        # variants exist but don't measurably help similarity recall
        # in the published benchmarks.
        return np.mean(np.stack(chunk_vecs, axis=0), axis=0)


register(MERTBackend())
