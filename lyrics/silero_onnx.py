"""Torch-free Silero VAD using raw ONNX Runtime.

Replaces the ``silero-vad`` PyPI package (which pulls torch + torchaudio).
Loads the official ``silero_vad.onnx`` model file from
``/app/model/silero_vad.onnx`` (downloaded at Docker build time from
snakers4/silero-vad releases) and exposes a single helper:

    get_speech_timestamps(audio_int16_or_float, sample_rate=16000)
        -> list[{'start': int_samples, 'end': int_samples}]

The implementation is a faithful but minimal port of silero's reference
``get_speech_timestamps`` — enough for our use case: keep voiced regions
of a song before feeding them to whisper.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_PATH = '/app/model/silero_vad.onnx'
_WINDOW_SAMPLES_16K = 512
_WINDOW_SAMPLES_8K = 256

_session = None
_session_path: Optional[str] = None
_session_lock = threading.Lock()


def _load_session(model_path: Optional[str] = None):
    """Load and cache the silero ONNX session."""
    global _session, _session_path

    path = model_path or os.environ.get('SILERO_VAD_ONNX_PATH', _DEFAULT_MODEL_PATH)

    if _session is not None and _session_path == path:
        return _session

    with _session_lock:
        if _session is not None and _session_path == path:
            return _session
        if not os.path.isfile(path):
            raise RuntimeError(f'silero_vad.onnx not found at {path}')

        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        _session = ort.InferenceSession(
            path, sess_options=opts, providers=['CPUExecutionProvider'])
        _session_path = path
        logger.info('Silero VAD ONNX session ready (path=%s)', path)
        return _session


def _voice_probabilities(audio: np.ndarray, sample_rate: int,
                         session) -> np.ndarray:
    """Run silero on every window of ``audio``; return per-window voice prob."""
    if sample_rate not in (8000, 16000):
        raise ValueError('Silero VAD requires 8000 or 16000 Hz input.')

    window = _WINDOW_SAMPLES_16K if sample_rate == 16000 else _WINDOW_SAMPLES_8K
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)

    n_windows = len(audio) // window
    if n_windows == 0:
        return np.zeros(0, dtype=np.float32)

    probs = np.zeros(n_windows, dtype=np.float32)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr_arg = np.array(sample_rate, dtype=np.int64)
    input_names = {inp.name for inp in session.get_inputs()}

    for i in range(n_windows):
        chunk = audio[i * window: (i + 1) * window]
        feed = {
            'input': chunk.reshape(1, window),
            'sr': sr_arg,
        }
        # Newer silero exports use 'state'; older use 'h' + 'c' separately.
        if 'state' in input_names:
            feed['state'] = state
        elif 'h' in input_names and 'c' in input_names:
            feed['h'] = state[0]
            feed['c'] = state[1]
        outputs = session.run(None, feed)
        probs[i] = float(outputs[0].squeeze())
        # Output ordering: [voice_prob, state] (newer) or
        # [voice_prob, h, c] (older). Pick whatever matches our feed.
        if 'state' in input_names and len(outputs) > 1:
            state = outputs[1]
        elif 'h' in input_names and len(outputs) >= 3:
            state = np.stack([outputs[1], outputs[2]], axis=0)
    return probs


def get_speech_timestamps(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    model_path: Optional[str] = None,
) -> List[Dict[str, int]]:
    """Return speech segments as ``[{'start': int_samples, 'end': int_samples}]``.

    Mirrors the public API of ``silero_vad.get_speech_timestamps`` for our
    use case (single-channel float32, 16 kHz).
    """
    if audio.size == 0:
        return []
    session = _load_session(model_path)
    window = _WINDOW_SAMPLES_16K if sample_rate == 16000 else _WINDOW_SAMPLES_8K

    probs = _voice_probabilities(audio, sample_rate, session)
    if probs.size == 0:
        return []

    min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
    min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
    speech_pad = int(sample_rate * speech_pad_ms / 1000)

    is_speech = probs >= threshold
    segments: List[Dict[str, int]] = []
    in_segment = False
    seg_start = 0
    silence_count = 0

    for i, voiced in enumerate(is_speech):
        sample_pos = i * window
        if voiced:
            if not in_segment:
                in_segment = True
                seg_start = sample_pos
            silence_count = 0
        else:
            if in_segment:
                silence_count += window
                if silence_count >= min_silence_samples:
                    seg_end = sample_pos - silence_count + window
                    if seg_end - seg_start >= min_speech_samples:
                        segments.append({
                            'start': max(0, seg_start - speech_pad),
                            'end':   min(len(audio), seg_end + speech_pad),
                        })
                    in_segment = False
                    silence_count = 0
    if in_segment:
        seg_end = len(audio)
        if seg_end - seg_start >= min_speech_samples:
            segments.append({
                'start': max(0, seg_start - speech_pad),
                'end':   seg_end,
            })

    return segments


def reset_session() -> None:
    """Drop the cached session (for memory cleanup)."""
    global _session, _session_path
    with _session_lock:
        _session = None
        _session_path = None
