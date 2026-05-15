from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
LYRICS_WHISPER_MIN_FREE_RAM_GB = float(os.environ.get(
    "LYRICS_WHISPER_MIN_FREE_RAM_GB", "2.5"))

_pipeline = None
_pipeline_dir = None
_load_lock = threading.Lock()


class WhisperLoadRefused(RuntimeError):
    pass


def _check_free_ram_or_raise() -> None:
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        if free_gb < LYRICS_WHISPER_MIN_FREE_RAM_GB:
            raise RuntimeError(
                f"Insufficient RAM for Whisper: {free_gb:.1f} GB free, "
                f"need {LYRICS_WHISPER_MIN_FREE_RAM_GB} GB"
            )
    except ImportError:
        pass


class _FasterWhisperPipeline:
    def __init__(self, model_dir: str, device: str = "cpu",
                 compute_type: str = "int8", num_workers: int = 1):
        from faster_whisper import WhisperModel
        logger.info(
            "faster-whisper: loading model from %s (device=%s compute_type=%s)",
            model_dir, device, compute_type,
        )
        self._model = WhisperModel(
            model_dir,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
            local_files_only=True,
        )
        logger.info("faster-whisper: model ready")

    def transcribe(self, wav: np.ndarray,
                   language: Optional[str] = None) -> Dict[str, object]:
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        # Normalise to [-1, 1] if needed
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = wav / peak

        segments_gen, info = self._model.transcribe(
            wav,
            language=language,
            beam_size=5,
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        text_parts = []
        total_logprob = 0.0
        seg_count = 0
        for seg in segments_gen:
            text_parts.append(seg.text)
            total_logprob += seg.avg_logprob
            seg_count += 1

        text = " ".join(text_parts).strip()
        avg_logprob = (total_logprob / seg_count) if seg_count else float("-inf")

        logger.info(
            "faster-whisper: transcribed %.1fs audio → %d chars  lang=%s  "
            "avg_logprob=%.3f",
            info.duration, len(text), info.language, avg_logprob,
        )
        return {
            "text": text,
            "language": info.language,
            "duration": info.duration,
            "avg_logprob": avg_logprob,
        }


def load_whisper_model() -> _FasterWhisperPipeline:
    global _pipeline, _pipeline_dir

    # Env var takes priority over config.py (which may point to old ONNX path)
    _env_dir = os.environ.get("LYRICS_WHISPER_MODEL_DIR", "")
    if _env_dir:
        LYRICS_WHISPER_MODEL_DIR = _env_dir
    else:
        try:
            from config import LYRICS_WHISPER_MODEL_DIR
        except Exception:
            LYRICS_WHISPER_MODEL_DIR = "/app/model/faster-whisper-small"

    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

    if _pipeline is not None and _pipeline_dir == LYRICS_WHISPER_MODEL_DIR:
        return _pipeline

    with _load_lock:
        if _pipeline is not None and _pipeline_dir == LYRICS_WHISPER_MODEL_DIR:
            return _pipeline
        try:
            _check_free_ram_or_raise()
        except RuntimeError as exc:
            raise WhisperLoadRefused(str(exc)) from exc

        _pipeline = _FasterWhisperPipeline(
            LYRICS_WHISPER_MODEL_DIR,
            device=device,
            compute_type=compute_type,
        )
        _pipeline_dir = LYRICS_WHISPER_MODEL_DIR
        return _pipeline


def transcribe(wav: np.ndarray, sr: int, language: Optional[str] = None,
               num_threads: Optional[int] = None) -> Dict[str, object]:
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr,
                               target_sr=SAMPLE_RATE)
    try:
        pipeline = load_whisper_model()
    except WhisperLoadRefused as exc:
        logger.warning("faster-whisper load refused: %s", exc)
        return {"text": "", "language": "",
                "duration": len(wav) / SAMPLE_RATE,
                "avg_logprob": float("-inf")}
    return pipeline.transcribe(wav, language=language)


def is_loaded() -> bool:
    return _pipeline is not None


def unload() -> bool:
    global _pipeline, _pipeline_dir
    pipeline = _pipeline
    if pipeline is None and _pipeline_dir is None:
        return False
    _pipeline = None
    _pipeline_dir = None
    try:
        pipeline._model = None
    except Exception:
        logger.exception("Error dropping faster-whisper model ref")
    try:
        import gc
        del pipeline
        gc.collect()
    except Exception:
        logger.exception("Error during faster-whisper GC")
    logger.info("faster-whisper: pipeline unloaded")
    return True


def reset_session() -> None:
    unload()
