"""Marian-style multilingual-to-English translator backed by HuggingFace optimum.

Loads the ``Helsinki-NLP/opus-mt-mul-en`` ONNX bundle (encoder +
``decoder_model_merged.onnx`` — a single decoder file that handles both the
first-step and KV-cache-step paths) via
``optimum.onnxruntime.ORTModelForSeq2SeqLM`` and exposes the same
``translate_to_english`` / ``reset_session`` contract the rest of the
lyrics pipeline expects.

The bundle lives at ``${LYRICS_TRANSLATOR_ONNX_DIR}`` (default
``/app/model/opus-mt-mul-en-onnx``) and is produced once at Docker build
time by ``scripts/onnx_export/export_marian_to_onnx.py``. At runtime
nothing is downloaded — ``local_files_only=True`` is passed everywhere.

Why optimum instead of a hand-rolled greedy decode loop:

* opus-mt-mul-en's decoder is exported with KV-cache inputs. The previous
  hand-rolled loop never fed ``past_key_values`` to the decoder, which
  produced empty translations on non-trivial source languages (e.g.
  Italian → English silently returned ``""``).
* ``model.generate()`` from optimum handles all the Marian-specific
  generation quirks (decoder_start_token_id, target-language token
  prefix, KV-cache plumbing, EOS/length termination).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = '/app/model/opus-mt-mul-en-onnx'
_DEFAULT_MAX_LENGTH = 512

_translator_lock = threading.Lock()
_translator_state = {
    'model_dir': None,   # type: Optional[str]
    'tokenizer': None,
    'model':     None,
}


def _load_translator(model_dir: Optional[str] = None):
    """Cache and return ``(tokenizer, model)`` for the translator."""
    target_dir = model_dir or os.environ.get(
        'LYRICS_TRANSLATOR_ONNX_DIR', _DEFAULT_MODEL_DIR)

    if (_translator_state['model_dir'] == target_dir
            and _translator_state['model'] is not None):
        return _translator_state['tokenizer'], _translator_state['model']

    with _translator_lock:
        if (_translator_state['model_dir'] == target_dir
                and _translator_state['model'] is not None):
            return _translator_state['tokenizer'], _translator_state['model']

        if not os.path.isdir(target_dir):
            raise RuntimeError(
                f'Translator ONNX directory not found at {target_dir}. '
                f'Re-run scripts/onnx_export/export_marian_to_onnx.py.')

        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from tasks._ort_providers import pick_providers

        tokenizer = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)
        # ``provider`` accepts a single name; ORTModelForSeq2SeqLM internally
        # falls back to CPU on CUDA init failure. Use the first entry from
        # the centralized helper so analysis honors the same GPU/CPU choice
        # MusiCNN + CLAP do.
        provider = pick_providers()[0]
        model = ORTModelForSeq2SeqLM.from_pretrained(
            target_dir,
            provider=provider,
            local_files_only=True,
        )

        _translator_state.update({
            'model_dir': target_dir,
            'tokenizer': tokenizer,
            'model': model,
        })
        logger.info('Translator ONNX ready (dir=%s, provider=%s)', target_dir, provider)
        return tokenizer, model


def translate_to_english(text: str, max_length: int = _DEFAULT_MAX_LENGTH) -> str:
    """Translate ``text`` to English. Returns ``''`` on any failure so the
    caller can treat the song as instrumental."""
    if not text or not text.strip():
        return ''

    try:
        tokenizer, model = _load_translator()
    except Exception as exc:
        logger.warning('Translator not ready (%s); dropping lyrics', exc)
        return ''

    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
        )
        # Greedy decode (num_beams=1) is deterministic and ~4× faster than
        # beam search; beam search adds quality on hard sentences but lyrics
        # are usually short and direct.
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=1,
            do_sample=False,
        )
        translated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    except Exception as exc:
        logger.warning('Translator inference failed: %s; dropping lyrics', exc)
        return ''

    return (translated or '').strip()


def reset_session() -> None:
    """Drop the cached model + tokenizer to free RAM."""
    with _translator_lock:
        _translator_state.update({
            'model_dir': None,
            'tokenizer': None,
            'model': None,
        })
