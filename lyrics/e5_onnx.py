"""Torch-free e5-base-v2 sentence embedding via ONNX Runtime.

Uses the project-hosted ONNX bundle ``lyrics_model_e5.tar.gz`` (NeptuneHub
GitHub release ``v4.0.0-model``). The bundle ships the ONNX file flat at
the archive root and the tokenizer files under a sibling directory:

    /app/model/
    ├── e5-base-v2.onnx              # ~436 MB ONNX weights
    └── e5-base-v2/                  # tokenizer
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── vocab.txt
        ├── config.json
        └── special_tokens_map.json

Override the locations with ``LYRICS_E5_ONNX_PATH`` (full path to the .onnx
file) and ``LYRICS_E5_TOKENIZER_DIR`` (directory containing tokenizer.json).
The runtime needs only ``onnxruntime`` + the bare ``tokenizers`` package —
no torch, no transformers.

Replaces the ``transformers.AutoModel.from_pretrained`` + torch tensor path
that used to live in ``lyrics_transcriber._embed_text``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_ONNX_PATH = '/app/model/e5-base-v2.onnx'
_DEFAULT_TOKENIZER_DIR = '/app/model/e5-base-v2'

_session = None
_tokenizer = None
_loaded_onnx_path: Optional[str] = None
_input_names: Tuple[str, ...] = ()
_load_lock = threading.Lock()


def _resolve_onnx_path() -> str:
    return os.environ.get('LYRICS_E5_ONNX_PATH', _DEFAULT_ONNX_PATH)


def _resolve_tokenizer_dir() -> str:
    return os.environ.get('LYRICS_E5_TOKENIZER_DIR', _DEFAULT_TOKENIZER_DIR)


def load_e5_model():
    """Load and cache the e5 tokenizer + ONNX session.

    Returns ``(tokenizer, ort_session)``. The tokenizer is a bare
    ``tokenizers.Tokenizer`` (no transformers / torch dependency).
    """
    global _session, _tokenizer, _loaded_onnx_path, _input_names

    onnx_path = _resolve_onnx_path()
    tokenizer_dir = _resolve_tokenizer_dir()

    if (_session is not None and _tokenizer is not None
            and _loaded_onnx_path == onnx_path):
        return _tokenizer, _session

    with _load_lock:
        if (_session is not None and _tokenizer is not None
                and _loaded_onnx_path == onnx_path):
            return _tokenizer, _session

        if not os.path.isfile(onnx_path):
            raise RuntimeError(
                f'e5-base-v2 ONNX weights not found at {onnx_path}. '
                'Expected from lyrics_model_e5.tar.gz (NeptuneHub release '
                'v4.0.0-model); override with LYRICS_E5_ONNX_PATH.')

        tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.json')
        if not os.path.isfile(tokenizer_path):
            raise RuntimeError(
                f'e5 tokenizer.json not found at {tokenizer_path}. '
                'Override the directory with LYRICS_E5_TOKENIZER_DIR.')

        import onnxruntime as ort
        from tokenizers import Tokenizer

        logger.info('Loading e5 tokenizer from %s', tokenizer_path)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        # e5 was trained with max sequence length 512; we cap at 128 anyway.
        try:
            tokenizer.enable_truncation(max_length=128)
            tokenizer.enable_padding(length=128)
        except Exception as exc:
            logger.warning('Could not configure e5 tokenizer padding/truncation: %s', exc)

        logger.info('Loading e5 ONNX session from %s', onnx_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Keep idle RSS down — disable transient buffer caching between calls.
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            onnx_path, sess_options=sess_options,
            providers=['CPUExecutionProvider'])

        _tokenizer = tokenizer
        _session = session
        _loaded_onnx_path = onnx_path
        _input_names = tuple(inp.name for inp in session.get_inputs())
        logger.info('e5 ONNX session ready (inputs=%s)', _input_names)
        return _tokenizer, _session


def embed_text(text: str, tokenizer=None, session=None) -> Optional[np.ndarray]:
    """Return a normalized float32 vector for ``text``, or ``None`` on empty input.

    Mean-pools the encoder's last_hidden_state across the attention mask
    (matching the e5 reference recipe), then L2-normalizes.
    """
    if not text or not text.strip():
        return None
    if tokenizer is None or session is None:
        tokenizer, session = load_e5_model()

    encoded = tokenizer.encode(text)
    # encoded.ids / .attention_mask / .type_ids are Python lists (length 128
    # after enable_padding/enable_truncation in load_e5_model).
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

    outputs = session.run(['last_hidden_state'], feed)
    last_hidden = outputs[0]  # (1, seq, hidden)

    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (1, seq, 1)
    summed = (last_hidden * mask).sum(axis=1)  # (1, hidden)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = (summed / counts).squeeze(0)  # (hidden,)

    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32, copy=False)


def is_loaded() -> bool:
    """True when the e5 ONNX session + tokenizer are currently cached."""
    return _session is not None or _tokenizer is not None


def reset_session() -> None:
    """Drop the cached session + tokenizer."""
    global _session, _tokenizer, _loaded_onnx_path, _input_names
    with _load_lock:
        _session = None
        _tokenizer = None
        _loaded_onnx_path = None
        _input_names = ()
