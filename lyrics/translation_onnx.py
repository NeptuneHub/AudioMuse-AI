"""Torch-free Marian-style translator using raw ONNX Runtime.

Replaces the ``transformers.AutoModelForSeq2SeqLM`` + ``model.generate()``
path (which requires torch) with a small numpy + onnxruntime greedy-decode
loop against a pre-exported ``Helsinki-NLP/opus-mt-mul-en`` model. That single
model handles ~70 source languages → English, so we don't have to bundle
per-language pairs.

The model is exported once at Docker build time by
``scripts/onnx_export/export_marian_to_onnx.py`` into
``/app/model/opus-mt-mul-en-onnx/`` and contains:

    encoder_model.onnx
    decoder_model.onnx
    config.json, generation_config.json
    tokenizer files (sentencepiece)

The translator reuses ``transformers.AutoTokenizer`` for tokenization (the
tokenizer is torch-free) and ``onnxruntime`` for the encoder/decoder
forward passes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = '/app/model/opus-mt-mul-en-onnx'
_DEFAULT_MAX_LENGTH = 512

_translator_lock = threading.Lock()
_translator_state = {
    'model_dir':       None,    # type: Optional[str]
    'tokenizer':       None,
    'encoder_session': None,
    'decoder_session': None,
    'pad_token_id':    None,    # type: Optional[int]
    'eos_token_id':    None,    # type: Optional[int]
    'decoder_start':   None,    # type: Optional[int]
    'enc_input_names': (),
    'dec_input_names': (),
}


def _load_translator(model_dir: Optional[str] = None):
    """Cache and return ``(tokenizer, encoder, decoder)`` for the translator."""
    target_dir = model_dir or os.environ.get(
        'LYRICS_TRANSLATOR_ONNX_DIR', _DEFAULT_MODEL_DIR)

    if (_translator_state['model_dir'] == target_dir
            and _translator_state['tokenizer'] is not None):
        return (
            _translator_state['tokenizer'],
            _translator_state['encoder_session'],
            _translator_state['decoder_session'],
        )

    with _translator_lock:
        if (_translator_state['model_dir'] == target_dir
                and _translator_state['tokenizer'] is not None):
            return (
                _translator_state['tokenizer'],
                _translator_state['encoder_session'],
                _translator_state['decoder_session'],
            )

        if not os.path.isdir(target_dir):
            raise RuntimeError(
                f'Translator ONNX directory not found at {target_dir}. '
                f'Re-run scripts/onnx_export/export_marian_to_onnx.py.')

        encoder_path = os.path.join(target_dir, 'encoder_model.onnx')
        decoder_path = os.path.join(target_dir, 'decoder_model.onnx')
        for path in (encoder_path, decoder_path):
            if not os.path.isfile(path):
                raise RuntimeError(f'Translator ONNX file missing: {path}')

        import onnxruntime as ort
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        encoder_session = ort.InferenceSession(
            encoder_path, sess_options=opts, providers=['CPUExecutionProvider'])
        decoder_session = ort.InferenceSession(
            decoder_path, sess_options=opts, providers=['CPUExecutionProvider'])

        # decoder_start_token_id lives in config.json (Marian usually uses pad)
        decoder_start = None
        try:
            with open(os.path.join(target_dir, 'config.json'), 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            decoder_start = cfg.get('decoder_start_token_id')
        except Exception:
            decoder_start = None
        if decoder_start is None:
            decoder_start = tokenizer.pad_token_id

        _translator_state.update({
            'model_dir':       target_dir,
            'tokenizer':       tokenizer,
            'encoder_session': encoder_session,
            'decoder_session': decoder_session,
            'pad_token_id':    tokenizer.pad_token_id,
            'eos_token_id':    tokenizer.eos_token_id,
            'decoder_start':   decoder_start,
            'enc_input_names': tuple(i.name for i in encoder_session.get_inputs()),
            'dec_input_names': tuple(i.name for i in decoder_session.get_inputs()),
        })
        logger.info(
            'Translator ONNX ready (dir=%s, decoder_start=%s, eos=%s)',
            target_dir, decoder_start, _translator_state['eos_token_id'])
        return tokenizer, encoder_session, decoder_session


def _greedy_decode(
    encoder_hidden_states: np.ndarray,
    encoder_attention_mask: np.ndarray,
    decoder_session,
    decoder_start_id: int,
    eos_token_id: int,
    max_length: int,
    dec_input_names: Tuple[str, ...],
) -> list:
    """Standard greedy autoregressive decode against a Marian-style decoder."""
    decoder_input_ids = np.array([[decoder_start_id]], dtype=np.int64)
    generated: list = []

    for _ in range(max_length):
        feed = {}
        for name in dec_input_names:
            if name in ('input_ids', 'decoder_input_ids'):
                feed[name] = decoder_input_ids
            elif name in ('encoder_attention_mask', 'attention_mask'):
                feed[name] = encoder_attention_mask
            elif name == 'encoder_hidden_states':
                feed[name] = encoder_hidden_states
        outputs = decoder_session.run(None, feed)
        logits = outputs[0]                 # (1, seq, vocab)
        next_token_logits = logits[:, -1, :]
        next_token = int(np.argmax(next_token_logits, axis=-1)[0])
        if next_token == eos_token_id:
            break
        generated.append(next_token)
        decoder_input_ids = np.concatenate(
            [decoder_input_ids, np.array([[next_token]], dtype=np.int64)],
            axis=1,
        )

    return generated


def translate_to_english(text: str, max_length: int = _DEFAULT_MAX_LENGTH) -> str:
    """Translate ``text`` (any of the languages opus-mt-mul-en supports) to English.

    Returns ``''`` on any failure so the caller can treat the song as
    instrumental, matching the existing ``_translate_to_english`` contract.
    """
    if not text or not text.strip():
        return ''

    try:
        tokenizer, encoder_session, decoder_session = _load_translator()
    except Exception as exc:
        logger.warning('Translator not ready (%s); dropping lyrics', exc)
        return ''

    eos_id = _translator_state['eos_token_id']
    decoder_start = _translator_state['decoder_start']
    enc_input_names = _translator_state['enc_input_names']
    dec_input_names = _translator_state['dec_input_names']
    if eos_id is None or decoder_start is None:
        logger.warning(
            'Translator missing eos/decoder_start (eos=%s, start=%s); dropping lyrics',
            eos_id, decoder_start)
        return ''

    try:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='np',
        )
        input_ids      = encoded['input_ids'].astype(np.int64, copy=False)
        attention_mask = encoded['attention_mask'].astype(np.int64, copy=False)

        feed = {}
        for name in enc_input_names:
            if name == 'input_ids':
                feed[name] = input_ids
            elif name == 'attention_mask':
                feed[name] = attention_mask
        encoder_outputs = encoder_session.run(None, feed)
        encoder_hidden_states = encoder_outputs[0]  # (1, seq, hidden)

        # Decoder typically expects encoder_attention_mask as int64
        token_ids = _greedy_decode(
            encoder_hidden_states,
            attention_mask,
            decoder_session,
            decoder_start_id=int(decoder_start),
            eos_token_id=int(eos_id),
            max_length=max_length,
            dec_input_names=dec_input_names,
        )
    except Exception as exc:
        logger.warning('Translator inference failed: %s; dropping lyrics', exc)
        return ''

    if not token_ids:
        return ''

    try:
        translated = tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception as exc:
        logger.warning('Translator detokenize failed: %s; dropping lyrics', exc)
        return ''
    return (translated or '').strip()


def reset_session() -> None:
    """Drop the cached sessions (memory cleanup helper)."""
    with _translator_lock:
        _translator_state.update({
            'model_dir':       None,
            'tokenizer':       None,
            'encoder_session': None,
            'decoder_session': None,
            'pad_token_id':    None,
            'eos_token_id':    None,
            'decoder_start':   None,
            'enc_input_names': (),
            'dec_input_names': (),
        })
