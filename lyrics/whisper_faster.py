# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""faster-whisper (CTranslate2) backend for lyrics ASR.

Drop-in alternative to whisper_onnx, used on the AMD/ROCm image since MIGraphX
can't parse the Whisper decoder's dynamic If/KV-cache subgraphs. CTranslate2
has a native ROCm HIP backend instead. Selected via LYRICS_WHISPER_BACKEND=faster
(lyrics/_asr_backend.py); mirrors whisper_onnx's public surface.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# CTranslate2 mirrors the CUDA API on ROCm, so "cuda" targets the AMD GPU here.
_DEVICE = os.environ.get("LYRICS_WHISPER_FASTER_DEVICE", "cuda").strip() or "cuda"
_COMPUTE_TYPE = os.environ.get("LYRICS_WHISPER_FASTER_COMPUTE_TYPE", "float16").strip() or "float16"
_MODEL_DIR = os.environ.get(
    "LYRICS_WHISPER_FASTER_MODEL_DIR", "/app/model/faster-whisper-small"
).strip()
_BEAM_SIZE = int(os.environ.get("LYRICS_WHISPER_BEAM_SIZE", "5"))

_model = None
_model_dir: Optional[str] = None
_load_lock = threading.Lock()


class WhisperLoadRefused(RuntimeError):
    """Raised when the model cannot be loaded; transcribe() degrades to empty."""


def load_whisper_model():
    global _model, _model_dir
    if _model is not None and _model_dir == _MODEL_DIR:
        return _model
    with _load_lock:
        if _model is not None and _model_dir == _MODEL_DIR:
            return _model
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # library missing on non-ROCm images
            raise WhisperLoadRefused(f"faster_whisper import failed: {exc}") from exc

        model_src = _MODEL_DIR if os.path.isdir(_MODEL_DIR) else "small"
        device = _DEVICE
        compute_type = _COMPUTE_TYPE
        try:
            _model = WhisperModel(model_src, device=device, compute_type=compute_type)
        except Exception as exc:
            # Fall back to CPU/int8 rather than failing the whole lyrics run.
            logger.warning(
                "faster-whisper GPU load failed (device=%s, compute=%s): %s - "
                "falling back to CPU/int8",
                device,
                compute_type,
                exc,
            )
            try:
                _model = WhisperModel(model_src, device="cpu", compute_type="int8")
            except Exception as exc2:
                raise WhisperLoadRefused(
                    f"faster_whisper load failed on GPU and CPU: {exc2}"
                ) from exc2
        _model_dir = _MODEL_DIR
        logger.info(
            "faster-whisper loaded (src=%s, device=%s, compute=%s)",
            model_src,
            device,
            compute_type,
        )
        return _model


def transcribe(
    wav: np.ndarray, sr: int, language: Optional[str] = None
) -> Dict[str, object]:
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    audio = np.ascontiguousarray(wav, dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE

    try:
        model = load_whisper_model()
    except WhisperLoadRefused as exc:
        logger.warning("faster-whisper load refused: %s", exc)
        return {"text": "", "language": "", "duration": duration, "avg_logprob": float("-inf")}

    # language=None lets faster-whisper auto-detect; we do our own VAD upstream
    # (silero_onnx), so leave faster-whisper's vad_filter off.
    segments, info = model.transcribe(
        audio,
        language=language or None,
        beam_size=_BEAM_SIZE,
        vad_filter=False,
    )

    texts = []
    logprobs = []
    for seg in segments:  # generator - consuming it runs the decode
        t = (seg.text or "").strip()
        if t:
            texts.append(t)
        lp = getattr(seg, "avg_logprob", None)
        if lp is not None:
            logprobs.append(float(lp))

    full_text = " ".join(texts).strip()
    detected = getattr(info, "language", "") or ""
    avg_logprob = float(np.mean(logprobs)) if logprobs else float("-inf")
    info_dur = getattr(info, "duration", None)
    logger.info(
        "faster-whisper: %.1fs audio (lang=%r, beam=%d, avg_logprob=%.2f)",
        info_dur if info_dur else duration,
        detected,
        _BEAM_SIZE,
        avg_logprob,
    )
    return {
        "text": full_text,
        "language": detected,
        "duration": float(info_dur) if info_dur else duration,
        "avg_logprob": avg_logprob,
    }


def is_loaded() -> bool:
    return _model is not None


def unload() -> bool:
    global _model, _model_dir
    if _model is None and _model_dir is None:
        return False
    model = _model
    _model = None
    _model_dir = None
    try:
        del model
    except Exception:
        logger.exception("Error dropping faster-whisper model")
    try:
        import gc

        gc.collect()
    except Exception:
        logger.exception("Error during faster-whisper GC")
    try:
        from tasks.memory_utils import comprehensive_memory_cleanup

        comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
    except Exception:
        logger.exception("Error during memory cleanup on faster-whisper unload")
    logger.info("faster-whisper: model unloaded")
    return True


def reset_session() -> None:
    unload()
