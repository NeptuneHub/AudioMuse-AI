"""Whisper-small speech-to-text backed by HuggingFace optimum.

Loads the ``openai/whisper-small`` ONNX bundle (encoder +
``decoder_model_merged.onnx`` — a single decoder file that handles both the
first-step and KV-cache-step paths — plus the WhisperProcessor) via
``optimum.onnxruntime.ORTModelForSpeechSeq2Seq`` and exposes the
``_load`` / ``transcribe`` / ``reset_session`` contract the rest of the
lyrics pipeline expects.

The bundle lives at ``${LYRICS_WHISPER_ONNX_DIR}`` (default
``/app/model/whisper-small-onnx``) and is produced once at Docker build
time by ``scripts/onnx_export/export_whisper_to_onnx.py``. At runtime
nothing is downloaded — ``local_files_only=True`` is passed everywhere.

Why optimum instead of a hand-rolled mel + greedy decode loop:

* Whisper's decoder is exported with KV-cache inputs; without them the
  per-token cost is quadratic in sequence length and KV-state mismatch
  can produce empty / corrupted output (the same class of bug we hit
  on Marian).
* WhisperProcessor handles the mel filterbank + log-mel transform with
  the exact constants the model was trained on. Hand-rolling that is a
  source of subtle accuracy regressions.
* ``model.generate()`` handles the language-detection / task / start-of-
  transcript token assembly that whisper requires.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = '/app/model/whisper-small-onnx'

# Whisper's encoder always operates on 30-second 16-kHz windows.
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

# Per-chunk decode cap. Whisper-small's positional embedding tops out at
# 448 tokens; cap at 440 so generation never bumps the limit mid-token.
MAX_NEW_TOKENS_PER_CHUNK = 440

_state_lock = threading.Lock()
_state = {
    'model_dir': None,    # type: Optional[str]
    'processor': None,    # WhisperProcessor (tokenizer + mel feature extractor)
    'model':     None,    # ORTModelForSpeechSeq2Seq
}


def _load(model_dir: Optional[str] = None) -> None:
    """Idempotently load the processor + model. Safe to call from any thread."""
    target_dir = model_dir or os.environ.get(
        'LYRICS_WHISPER_ONNX_DIR', _DEFAULT_MODEL_DIR)

    if (_state['model_dir'] == target_dir
            and _state['model'] is not None):
        return

    with _state_lock:
        if (_state['model_dir'] == target_dir
                and _state['model'] is not None):
            return

        if not os.path.isdir(target_dir):
            raise RuntimeError(
                f'Whisper ONNX directory not found at {target_dir}. '
                f'Re-run scripts/onnx_export/export_whisper_to_onnx.py.')

        from transformers import WhisperProcessor
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from tasks._ort_providers import pick_providers

        processor = WhisperProcessor.from_pretrained(
            target_dir, local_files_only=True)
        provider = pick_providers()[0]
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            target_dir,
            provider=provider,
            local_files_only=True,
        )

        _state.update({
            'model_dir': target_dir,
            'processor': processor,
            'model': model,
        })
        logger.info('Whisper ONNX ready (dir=%s, provider=%s)', target_dir, provider)


def _chunks(audio: np.ndarray) -> Iterable[np.ndarray]:
    """Yield 30-second chunks (zero-padded if the last is short)."""
    n = len(audio)
    if n == 0:
        return
    pos = 0
    while pos < n:
        chunk = audio[pos: pos + N_SAMPLES]
        if len(chunk) < N_SAMPLES:
            pad = np.zeros(N_SAMPLES, dtype=np.float32)
            pad[: len(chunk)] = chunk
            chunk = pad
        yield chunk
        pos += N_SAMPLES


def transcribe(audio: np.ndarray,
               language: Optional[str] = None,
               task: str = 'transcribe',
               model_dir: Optional[str] = None) -> Dict:
    """Transcribe ``audio`` (float32, 16 kHz, mono) with whisper-small.

    ``language`` may be a 2-letter code (e.g. ``'en'``) or ``None`` for
    auto-detect. ``task`` is ``'transcribe'`` (source-language output) or
    ``'translate'`` (English output regardless of source).

    Returns ``{'text': str, 'language': str, 'duration': float}``.
    """
    _load(model_dir)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return {'text': '', 'language': language or 'en', 'duration': 0.0}

    duration = float(len(audio)) / SAMPLE_RATE
    processor = _state['processor']
    model = _state['model']

    pieces = []
    detected_lang = language
    for chunk in _chunks(audio):
        # Mel-spectrogram via the processor (uses the exact constants the
        # model was trained on — no reason to hand-roll this).
        features = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors='pt',
        ).input_features

        gen_kwargs = {
            'max_new_tokens': MAX_NEW_TOKENS_PER_CHUNK,
            'num_beams': 1,
            'do_sample': False,
            'task': task,
        }
        # ``language=None`` lets whisper auto-detect from the audio.
        if detected_lang is not None:
            gen_kwargs['language'] = detected_lang

        try:
            predicted_ids = model.generate(features, **gen_kwargs)
        except Exception as exc:
            logger.warning('Whisper inference failed on a chunk (%s); skipping', exc)
            continue

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        if text and text.strip():
            pieces.append(text.strip())

        # Pin the detected language after the first non-empty chunk so the
        # next chunks reuse it instead of paying for re-detection.
        if detected_lang is None:
            detected_lang = _peek_language(predicted_ids[0], processor)

    text = ' '.join(pieces).strip()
    return {
        'text': text,
        'language': detected_lang or 'en',
        'duration': duration,
    }


def _peek_language(token_ids, processor) -> Optional[str]:
    """Pull the language code out of whisper's prefix tokens.

    Whisper prepends ``<|startoftranscript|><|<lang>|><|task|><|notimestamps|>``
    to its output. We scan the first few tokens for ``<|<2-letter>|>``.
    Returns ``None`` if no language token was emitted (older whispers or
    edge cases).
    """
    try:
        head = processor.tokenizer.convert_ids_to_tokens(
            [int(t) for t in token_ids[:6]])
    except Exception:
        return None
    for tok in head:
        if (isinstance(tok, str) and tok.startswith('<|') and tok.endswith('|>')
                and 4 <= len(tok) <= 6):
            inner = tok[2:-2]
            if len(inner) == 2 and inner.isalpha():
                return inner.lower()
    return None


def reset_session() -> None:
    """Drop the cached processor + model to free RAM (idle-unload helper)."""
    with _state_lock:
        _state.update({
            'model_dir': None,
            'processor': None,
            'model': None,
        })
