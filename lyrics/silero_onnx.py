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
_CONTEXT_SAMPLES_16K = 64
_CONTEXT_SAMPLES_8K = 32

_session = None
_session_path: Optional[str] = None
_session_lock = threading.Lock()

def _load_session(model_path: Optional[str] = None):
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
        opts.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            path, sess_options=opts, providers=['CPUExecutionProvider'])
        _session_path = path
        logger.info('Silero VAD ONNX session ready (path=%s)', path)
        return _session

def _voice_probabilities(audio: np.ndarray, sample_rate: int,
                         session) -> np.ndarray:
    if sample_rate not in (8000, 16000):
        raise ValueError('Silero VAD requires 8000 or 16000 Hz input.')

    window = _WINDOW_SAMPLES_16K if sample_rate == 16000 else _WINDOW_SAMPLES_8K
    context_size = (_CONTEXT_SAMPLES_16K if sample_rate == 16000
                    else _CONTEXT_SAMPLES_8K)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)

    n_windows = len(audio) // window
    if n_windows == 0:
        return np.zeros(0, dtype=np.float32)

    probs = np.zeros(n_windows, dtype=np.float32)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(context_size, dtype=np.float32)
    sr_arg = np.array(sample_rate, dtype=np.int64)
    input_names = {inp.name for inp in session.get_inputs()}

    for i in range(n_windows):
        chunk = audio[i * window: (i + 1) * window]
        x = np.concatenate([context, chunk]).astype(np.float32, copy=False)
        feed = {
            'input': x.reshape(1, -1),
            'sr': sr_arg,
        }
        if 'state' in input_names:
            feed['state'] = state
        elif 'h' in input_names and 'c' in input_names:
            feed['h'] = state[0]
            feed['c'] = state[1]
        outputs = session.run(None, feed)
        probs[i] = float(outputs[0].squeeze())
        if 'state' in input_names and len(outputs) > 1:
            state = outputs[1]
        elif 'h' in input_names and len(outputs) >= 3:
            state = np.stack([outputs[1], outputs[2]], axis=0)
        context = x[-context_size:]
    return probs

def _segments_from_probs(
    probs: np.ndarray,
    audio_len: int,
    sample_rate: int,
    window: int,
    threshold: float,
    min_speech_samples: int,
    min_silence_samples: int,
    speech_pad: int,
    neg_threshold: Optional[float] = None,
) -> List[Dict[str, int]]:
    if neg_threshold is None:
        neg_threshold = max(0.01, threshold - 0.15)
    segments: List[Dict[str, int]] = []
    in_segment = False
    seg_start = 0
    silence_count = 0

    for i, prob in enumerate(probs):
        sample_pos = i * window
        if in_segment:
            if prob >= neg_threshold:
                silence_count = 0
            else:
                silence_count += window
                if silence_count >= min_silence_samples:
                    seg_end = sample_pos - silence_count + window
                    if seg_end - seg_start >= min_speech_samples:
                        segments.append({
                            'start': max(0, seg_start - speech_pad),
                            'end':   min(audio_len, seg_end + speech_pad),
                        })
                    in_segment = False
                    silence_count = 0
        else:
            if prob >= threshold:
                in_segment = True
                seg_start = sample_pos
                silence_count = 0
    if in_segment:
        seg_end = audio_len
        if seg_end - seg_start >= min_speech_samples:
            segments.append({
                'start': max(0, seg_start - speech_pad),
                'end':   seg_end,
            })
    return segments

def analyze_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    neg_threshold: Optional[float] = None,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    model_path: Optional[str] = None,
) -> Dict[str, object]:
    if audio.size == 0:
        return {'segments': [], 'max_prob': 0.0, 'mean_prob': 0.0,
                'n_windows': 0, 'threshold': threshold,
                'probs': np.zeros(0, dtype=np.float32)}
    session = _load_session(model_path)
    window = _WINDOW_SAMPLES_16K if sample_rate == 16000 else _WINDOW_SAMPLES_8K

    probs = _voice_probabilities(audio, sample_rate, session)
    if probs.size == 0:
        return {'segments': [], 'max_prob': 0.0, 'mean_prob': 0.0,
                'n_windows': 0, 'threshold': threshold, 'probs': probs}

    min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
    min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
    speech_pad = int(sample_rate * speech_pad_ms / 1000)
    segments = _segments_from_probs(
        probs, len(audio), sample_rate, window, threshold,
        min_speech_samples, min_silence_samples, speech_pad,
        neg_threshold=neg_threshold,
    )
    return {
        'segments': segments,
        'max_prob': float(np.max(probs)),
        'mean_prob': float(np.mean(probs)),
        'n_windows': int(probs.size),
        'threshold': float(threshold),
        'probs': probs,
    }

def get_speech_timestamps(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    neg_threshold: Optional[float] = None,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    model_path: Optional[str] = None,
) -> List[Dict[str, int]]:
    result = analyze_audio(
        audio, sample_rate=sample_rate, threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms, model_path=model_path,
    )
    return result['segments']

def threshold_segments(probs: np.ndarray, audio_len: int,
                       sample_rate: int = 16000,
                       threshold: float = 0.5,
                       neg_threshold: Optional[float] = None,
                       min_speech_duration_ms: int = 250,
                       min_silence_duration_ms: int = 100,
                       speech_pad_ms: int = 30) -> List[Dict[str, int]]:
    window = _WINDOW_SAMPLES_16K if sample_rate == 16000 else _WINDOW_SAMPLES_8K
    min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
    min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
    speech_pad = int(sample_rate * speech_pad_ms / 1000)
    return _segments_from_probs(
        probs, audio_len, sample_rate, window, threshold,
        min_speech_samples, min_silence_samples, speech_pad,
        neg_threshold=neg_threshold,
    )

def is_loaded() -> bool:
    return _session is not None

def reset_session() -> None:
    global _session, _session_path
    with _session_lock:
        _session = None
        _session_path = None
