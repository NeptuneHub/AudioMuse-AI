# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Multilingual GTE text-embedding runtime (INT8 ONNX) for lyrics vectors.

Loads the quantized gte-multilingual-base model on onnxruntime and turns raw
lyric or query text into 768-dim CLS-pooled embeddings, the text side of the
lyrics pipeline that complements whisper_onnx (audio) and silero_onnx (VAD).
Consumed by lyrics_transcriber for both stored lyric vectors and search queries.

Main Features:
* Lazy, thread-safe session and tokenizer load with model/tokenizer paths
  overridable via env, adapting to the concrete model input names.
* embed_text returns an L2-normalized float32 vector (or None on failure) plus
  is_loaded / reset_session hooks so the models can be released for memory.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ONNX_PATH = '/app/model/gte-multilingual-base-int8.onnx'
_DEFAULT_TOKENIZER_DIR = '/app/model/gte-multilingual-base'

_session = None
_tokenizer = None
_loaded_onnx_path: Optional[str] = None
_input_names: Tuple[str, ...] = ()
_output_name: Optional[str] = None
_load_lock = threading.Lock()


def _resolve_onnx_path() -> str:
    return os.environ.get('LYRICS_GTE_ONNX_PATH', _DEFAULT_ONNX_PATH)


def _resolve_tokenizer_dir() -> str:
    return os.environ.get('LYRICS_GTE_TOKENIZER_DIR', _DEFAULT_TOKENIZER_DIR)


def load_gte_model():
    global _session, _tokenizer, _loaded_onnx_path, _input_names, _output_name

    onnx_path = _resolve_onnx_path()
    tokenizer_dir = _resolve_tokenizer_dir()

    if _session is not None and _tokenizer is not None and _loaded_onnx_path == onnx_path:
        return _tokenizer, _session

    with _load_lock:
        if _session is not None and _tokenizer is not None and _loaded_onnx_path == onnx_path:
            return _tokenizer, _session

        if not os.path.isfile(onnx_path):
            raise RuntimeError(
                f'gte-multilingual-base ONNX weights not found at {onnx_path}. '
                'Expected from lyrics_model_gte_vnni.tar.gz (NeptuneHub release); '
                'override with LYRICS_GTE_ONNX_PATH.'
            )

        tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.json')
        if not os.path.isfile(tokenizer_path):
            raise RuntimeError(
                f'gte tokenizer.json not found at {tokenizer_path}. '
                'Override the directory with LYRICS_GTE_TOKENIZER_DIR.'
            )

        import onnxruntime as ort
        from tokenizers import Tokenizer

        logger.info('Loading gte tokenizer from %s', tokenizer_path)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        from config import LYRICS_GTE_MAX_TOKENS

        try:
            tokenizer.enable_truncation(max_length=LYRICS_GTE_MAX_TOKENS)
            tokenizer.no_padding()
        except Exception as exc:
            logger.warning('Could not configure gte tokenizer padding/truncation: %s', exc)

        logger.info('Loading gte ONNX session from %s', onnx_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        sess_options.inter_op_num_threads = 1
        try:
            from tasks.analysis_helper import create_onnx_session

            session = create_onnx_session(
                onnx_path,
                provider_options=[('CPUExecutionProvider', {})],
                sess_options=sess_options,
                label='gte',
            )
        except Exception as exc:
            logger.warning('gte: provider helper unavailable (%s) - CPU only', exc)
            session = ort.InferenceSession(
                onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider']
            )
        logger.info('gte active provider: %s', session.get_providers()[0])

        _tokenizer = tokenizer
        _session = session
        _loaded_onnx_path = onnx_path
        _input_names = tuple(inp.name for inp in session.get_inputs())
        _output_name = session.get_outputs()[0].name
        logger.info('gte ONNX session ready (inputs=%s, output=%s)', _input_names, _output_name)
        return _tokenizer, _session


def embed_text(text: str, tokenizer=None, session=None) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    if tokenizer is None or session is None:
        tokenizer, session = load_gte_model()

    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    type_ids = np.array([encoded.type_ids], dtype=np.int64)

    feed: dict = {}
    for name in _input_names:
        if name == 'input_ids':
            feed[name] = input_ids
        elif name == 'attention_mask':
            feed[name] = attention_mask
        elif name == 'token_type_ids':
            feed[name] = type_ids

    outputs = session.run([_output_name], feed)
    last_hidden = outputs[0]

    pooled = last_hidden[:, 0, :].squeeze(0)

    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32, copy=False)


def is_loaded() -> bool:
    return _session is not None or _tokenizer is not None


def reset_session() -> None:
    global _session, _tokenizer, _loaded_onnx_path, _input_names, _output_name
    with _load_lock:
        _session = None
        _tokenizer = None
        _loaded_onnx_path = None
        _input_names = ()
        _output_name = None
