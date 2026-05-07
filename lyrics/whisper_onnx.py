"""Torch-free Whisper transcription via raw ONNX Runtime.

Loads ``encoder_model.onnx`` + ``decoder_model.onnx`` (exported once by
``scripts/onnx_export/export_whisper_to_onnx.py`` from ``openai/whisper-small``)
plus the bundled tokenizer files, and runs:

    audio (np.float32, 16 kHz)
        -> log-mel spectrogram (80 bins, 3000 frames per 30 s window)
        -> encoder ONNX → encoder_hidden_states (1, 1500, 768)
        -> [optional] language probe via single decoder step
        -> greedy decode through decoder ONNX
        -> token IDs → text (transformers tokenizer, torch-free)

The public entry point is::

    transcribe(audio: np.ndarray, language: Optional[str] = None,
               task: str = 'transcribe') -> dict
        # returns {'text': str, 'language': str, 'duration': float}

Long inputs are processed as a sequence of independent 30-second chunks; each
chunk is padded/trimmed to exactly 480 000 samples (whisper's fixed input
window). This is simpler than openai-whisper's overlapping/sliding-window
strategy and slightly worse on segment boundaries, but adequate for lyrics —
we cap audio at 4 minutes upstream so at most 8 chunks per song.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Whisper's preprocessing constants (from openai/whisper preprocessor_config.json).
SAMPLE_RATE     = 16000
N_FFT           = 400
HOP_LENGTH      = 160
N_MELS          = 80
CHUNK_SECONDS   = 30
N_SAMPLES       = SAMPLE_RATE * CHUNK_SECONDS  # 480 000
N_FRAMES        = N_SAMPLES // HOP_LENGTH      # 3 000
MAX_DECODE_LEN  = 448                          # whisper limit per chunk

# Whisper language codes (multilingual variant). Order matches openai/whisper.
LANGUAGE_CODES: Tuple[str, ...] = (
    'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca',
    'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms',
    'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la',
    'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn',
    'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw',
    'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be',
    'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn',
    'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha',
    'ba', 'jw', 'su', 'yue',
)

_state = {
    'model_dir':       None,    # type: Optional[str]
    'tokenizer':       None,
    'encoder_session': None,
    'decoder_session': None,
    'mel_filters':     None,    # type: Optional[np.ndarray]
    'eot_id':          None,    # type: Optional[int]
    'sot_id':          None,    # type: Optional[int]
    'no_timestamps':   None,    # type: Optional[int]
    'transcribe_id':   None,    # type: Optional[int]
    'translate_id':    None,    # type: Optional[int]
    'language_token_ids': None, # type: Optional[Dict[str, int]]
    'enc_input_name':  None,    # type: Optional[str]
    'dec_input_names': (),
}
_state_lock = threading.Lock()


def _hann_window(n: int) -> np.ndarray:
    """Numpy-only Hann window matching whisper's torch.hann_window."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _build_mel_filters() -> np.ndarray:
    """Return the (n_mels, n_fft//2 + 1) mel filterbank whisper uses.

    librosa's filterbank with ``htk=False`` matches whisper. Computed once
    on first use.
    """
    import librosa  # lazy: keep module import light
    return librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, htk=False,
    ).astype(np.float32)


def _log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute the (N_MELS, N_FRAMES) log-mel spectrogram for one whisper chunk.

    ``audio`` must be a float32 array of exactly ``N_SAMPLES`` samples.
    """
    if _state['mel_filters'] is None:
        _state['mel_filters'] = _build_mel_filters()

    window = _hann_window(N_FFT).astype(np.float32)
    # STFT via numpy: pad reflect so the centered windows match whisper.
    pad = N_FFT // 2
    padded = np.pad(audio, (pad, pad), mode='reflect')
    n_frames_out = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(N_FFT, n_frames_out),
        strides=(padded.strides[0], padded.strides[0] * HOP_LENGTH),
    ).copy()
    frames = frames * window[:, None]
    spec = np.fft.rfft(frames, axis=0)
    magnitudes = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    mel_spec = _state['mel_filters'] @ magnitudes  # (N_MELS, frames)
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    # Pad/trim to exactly N_FRAMES so the encoder graph sees a fixed shape.
    if log_spec.shape[1] < N_FRAMES:
        log_spec = np.pad(log_spec, ((0, 0), (0, N_FRAMES - log_spec.shape[1])))
    elif log_spec.shape[1] > N_FRAMES:
        log_spec = log_spec[:, :N_FRAMES]
    return log_spec.astype(np.float32, copy=False)


def _resolve_paths(model_dir: Optional[str]) -> str:
    target = model_dir or os.environ.get(
        'LYRICS_WHISPER_ONNX_DIR', '/app/model/whisper-small-onnx')
    if not os.path.isdir(target):
        raise RuntimeError(f'whisper ONNX directory not found at {target}')
    return target


def _load(model_dir: Optional[str] = None):
    target = _resolve_paths(model_dir)
    if (_state['model_dir'] == target
            and _state['encoder_session'] is not None
            and _state['decoder_session'] is not None):
        return

    with _state_lock:
        if (_state['model_dir'] == target
                and _state['encoder_session'] is not None):
            return

        encoder_path = os.path.join(target, 'encoder_model.onnx')
        decoder_path = os.path.join(target, 'decoder_model.onnx')
        for path in (encoder_path, decoder_path):
            if not os.path.isfile(path):
                raise RuntimeError(f'whisper ONNX file missing: {path}')

        import onnxruntime as ort
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        encoder_session = ort.InferenceSession(
            encoder_path, sess_options=opts, providers=['CPUExecutionProvider'])
        decoder_session = ort.InferenceSession(
            decoder_path, sess_options=opts, providers=['CPUExecutionProvider'])

        # Find the encoder's input name (optimum sometimes uses 'input_features').
        enc_inputs = encoder_session.get_inputs()
        enc_input_name = enc_inputs[0].name if enc_inputs else 'input_features'

        # Resolve special tokens.
        sot_id          = tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
        eot_id          = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        no_timestamps   = tokenizer.convert_tokens_to_ids('<|notimestamps|>')
        transcribe_id   = tokenizer.convert_tokens_to_ids('<|transcribe|>')
        translate_id    = tokenizer.convert_tokens_to_ids('<|translate|>')

        language_token_ids = {}
        for code in LANGUAGE_CODES:
            tok = f'<|{code}|>'
            tid = tokenizer.convert_tokens_to_ids(tok)
            # Some codes may be absent in older variants; skip those.
            if tid is not None and tid != tokenizer.unk_token_id:
                language_token_ids[code] = tid

        _state.update({
            'model_dir':          target,
            'tokenizer':          tokenizer,
            'encoder_session':    encoder_session,
            'decoder_session':    decoder_session,
            'sot_id':             sot_id,
            'eot_id':             eot_id,
            'no_timestamps':      no_timestamps,
            'transcribe_id':      transcribe_id,
            'translate_id':       translate_id,
            'language_token_ids': language_token_ids,
            'enc_input_name':     enc_input_name,
            'dec_input_names':    tuple(i.name for i in decoder_session.get_inputs()),
        })
        logger.info(
            'Whisper ONNX session ready (dir=%s, sot=%s, eot=%s, lang_tokens=%d)',
            target, sot_id, eot_id, len(language_token_ids))


def _encode(audio_chunk: np.ndarray) -> np.ndarray:
    """Run mel + encoder forward, return ``(1, 1500, 768)`` hidden states."""
    mel = _log_mel_spectrogram(audio_chunk)              # (80, 3000)
    mel = mel[np.newaxis, ...]                           # (1, 80, 3000)
    enc_input_name = _state['enc_input_name']
    encoder_outputs = _state['encoder_session'].run(
        None, {enc_input_name: mel})
    return encoder_outputs[0]


def _decoder_feed(decoder_input_ids: np.ndarray,
                  encoder_hidden: np.ndarray) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {}
    for name in _state['dec_input_names']:
        if name in ('input_ids', 'decoder_input_ids'):
            feed[name] = decoder_input_ids
        elif name == 'encoder_hidden_states':
            feed[name] = encoder_hidden
        elif name in ('encoder_attention_mask', 'attention_mask'):
            # Whisper encoder always emits 1500 frames; build a full-attention mask.
            feed[name] = np.ones(
                (encoder_hidden.shape[0], encoder_hidden.shape[1]),
                dtype=np.int64,
            )
    return feed


def _detect_language(encoder_hidden: np.ndarray) -> str:
    """Single-step language probe — argmax over the language token logits."""
    sot_id = _state['sot_id']
    decoder_input_ids = np.array([[sot_id]], dtype=np.int64)
    outputs = _state['decoder_session'].run(
        None, _decoder_feed(decoder_input_ids, encoder_hidden))
    logits = outputs[0][0, -1, :]
    lang_ids = _state['language_token_ids']
    if not lang_ids:
        return 'en'
    codes  = list(lang_ids.keys())
    ids    = np.array(list(lang_ids.values()), dtype=np.int64)
    scores = logits[ids]
    best   = int(np.argmax(scores))
    return codes[best]


def _greedy_decode(encoder_hidden: np.ndarray,
                   initial_tokens: List[int],
                   max_length: int) -> List[int]:
    eot = _state['eot_id']
    decoder_input_ids = np.array([initial_tokens], dtype=np.int64)
    generated: List[int] = []
    for _ in range(max_length):
        outputs = _state['decoder_session'].run(
            None, _decoder_feed(decoder_input_ids, encoder_hidden))
        logits = outputs[0][:, -1, :]
        next_token = int(np.argmax(logits, axis=-1)[0])
        if next_token == eot:
            break
        generated.append(next_token)
        decoder_input_ids = np.concatenate(
            [decoder_input_ids, np.array([[next_token]], dtype=np.int64)],
            axis=1,
        )
    return generated


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
    """Transcribe ``audio`` (float32, 16 kHz, mono) with whisper-small ONNX.

    ``language`` may be a 2-letter code (e.g. ``'en'``) or ``None`` for
    auto-detect (cheap single-step probe on the first chunk).
    ``task`` is ``'transcribe'`` (source-language output) or ``'translate'``
    (English output regardless of source).

    Returns ``{'text': str, 'language': str, 'duration': float}``.
    """
    _load(model_dir)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return {'text': '', 'language': language or 'en', 'duration': 0.0}

    duration = float(len(audio)) / SAMPLE_RATE
    task_id = (_state['translate_id'] if task == 'translate'
               else _state['transcribe_id'])

    pieces: List[str] = []
    detected_lang = language
    for idx, chunk in enumerate(_chunks(audio)):
        encoder_hidden = _encode(chunk)
        if detected_lang is None:
            detected_lang = _detect_language(encoder_hidden)

        lang_id = _state['language_token_ids'].get(detected_lang)
        if lang_id is None:
            # Unknown code (e.g. 'iw' alias) — fall back to English.
            lang_id = _state['language_token_ids'].get('en')
            detected_lang = 'en'

        initial = [
            _state['sot_id'],
            lang_id,
            task_id,
            _state['no_timestamps'],
        ]
        token_ids = _greedy_decode(encoder_hidden, initial, MAX_DECODE_LEN)
        if token_ids:
            pieces.append(_state['tokenizer'].decode(
                token_ids, skip_special_tokens=True))

    text = ' '.join(s.strip() for s in pieces if s and s.strip()).strip()
    return {
        'text': text,
        'language': detected_lang or 'en',
        'duration': duration,
    }


def reset_session() -> None:
    """Drop cached sessions to free RAM."""
    with _state_lock:
        for k in (
            'tokenizer', 'encoder_session', 'decoder_session',
            'mel_filters', 'sot_id', 'eot_id', 'no_timestamps',
            'transcribe_id', 'translate_id', 'language_token_ids',
            'enc_input_name',
        ):
            _state[k] = None
        _state['model_dir'] = None
        _state['dec_input_names'] = ()
