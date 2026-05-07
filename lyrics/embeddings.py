"""Single-owner module for the e5 lyrics embedding model.

Both ``analyze_lyrics`` (worker side, batch) and ``search_by_text`` (web side,
per-request) need the same e5-base-v2 tokenizer + model and the same axis
description embeddings. Keeping them here avoids loading the model twice in
RAM when the same Python process serves both roles.

Backend
-------
Runtime uses ``onnxruntime`` against ``/app/model/e5-base-v2.onnx``, which is
exported once at Docker build time by ``scripts/onnx_export/export_e5_to_onnx.py``.
This means the CPU image does not need ``torch`` at runtime — only the
transformers tokenizer (which is torch-free) plus ``onnxruntime`` and ``numpy``.

The heavy ``onnxruntime.InferenceSession`` and tokenizer loads happen lazily
inside ``load_topic_embedding_model`` so that ``from lyrics.embeddings import
embed_query_text`` at app startup remains cheap.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from .axes import MUSIC_ANALYSIS_AXES

logger = logging.getLogger(__name__)

_embedding_tokenizer = None
_embedding_session = None
_embedding_model_name: Optional[str] = None
_embedding_input_names: Tuple[str, ...] = ()
_embedding_load_lock = threading.Lock()
_axis_label_map: Optional[Dict[str, List[Tuple[str, str]]]] = None
_axis_embeddings: Optional[Dict[str, np.ndarray]] = None


def _resolve_paths(model_name: Optional[str]) -> Tuple[str, str]:
    """Return ``(tokenizer_dir, onnx_path)``.

    Tokenizer directory: where ``tokenizer.json`` / ``vocab.txt`` live.
    ONNX path: where the exported encoder weights live.
    """
    if model_name is None:
        try:
            from config import LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL
            model_name = LYRICS_DEFAULT_TOPIC_EMBEDDING_MODEL
        except Exception:
            model_name = 'intfloat/e5-base-v2'

    try:
        from config import LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR
        tokenizer_dir = LYRICS_DEFAULT_TOPIC_EMBEDDING_CACHE_DIR
    except Exception:
        tokenizer_dir = '/app/model/e5-base-v2'

    onnx_path = os.environ.get('LYRICS_E5_ONNX_PATH', '/app/model/e5-base-v2.onnx')
    return (tokenizer_dir if os.path.isdir(tokenizer_dir) else model_name), onnx_path


def load_topic_embedding_model(model_name: Optional[str] = None):
    """Load the e5 tokenizer + ONNX inference session.

    Returns ``(tokenizer, ort_session)``. The tuple shape is preserved (with
    ``ort_session`` standing in for the old PyTorch ``model``) so callers
    that destructure ``tokenizer, model = load_topic_embedding_model()`` keep
    working — the second element is now consumed by ``_embed_text`` below.
    """
    global _embedding_tokenizer, _embedding_session, _embedding_model_name
    global _embedding_input_names

    # Heavy imports stay inside the function so the bare module import is cheap.
    import onnxruntime as ort
    from transformers import AutoTokenizer

    target_dir, onnx_path = _resolve_paths(model_name)

    if (_embedding_tokenizer is not None
            and _embedding_session is not None
            and _embedding_model_name == onnx_path):
        return _embedding_tokenizer, _embedding_session

    with _embedding_load_lock:
        if (_embedding_tokenizer is not None
                and _embedding_session is not None
                and _embedding_model_name == onnx_path):
            return _embedding_tokenizer, _embedding_session

        if not os.path.isfile(onnx_path):
            raise RuntimeError(
                f'e5 ONNX model not found at {onnx_path}. '
                f'Re-run scripts/onnx_export/export_e5_to_onnx.py.'
            )

        logger.info('Loading e5 tokenizer from %s', target_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)

        logger.info('Loading e5 ONNX session from %s', onnx_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable ORT's pre-allocated CPU memory arena and run-to-run pattern
        # cache. The arena holds ~150-200 MB of resident pages that no glibc
        # / jemalloc tuning can reclaim. Per-query inference still completes
        # in tens of ms; the small allocation overhead is invisible to users
        # but the memory savings are real.
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        try:
            cpu_threads = max(1, (os.cpu_count() or 2) // 2)
            sess_options.intra_op_num_threads = cpu_threads
        except Exception:
            pass
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider'],
        )

        _embedding_tokenizer = tokenizer
        _embedding_session = session
        _embedding_model_name = onnx_path
        _embedding_input_names = tuple(inp.name for inp in session.get_inputs())
        logger.info('e5 ONNX session ready (inputs=%s)', _embedding_input_names)
        return _embedding_tokenizer, _embedding_session


def _embed_text(text: str, tokenizer, session) -> Optional[np.ndarray]:
    """Encode ``text`` into a normalized float32 vector via the ONNX session."""
    if not text or not text.strip():
        return None
    encoded = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='np',
    )
    feed = {}
    for name in _embedding_input_names:
        if name in encoded:
            value = encoded[name]
            # ONNX expects int64 for *_ids / *_mask tensors; transformers may
            # return int32 on some platforms.
            if value.dtype != np.int64:
                value = value.astype(np.int64, copy=False)
            feed[name] = value

    outputs = session.run(['last_hidden_state'], feed)
    last_hidden = outputs[0]  # (1, seq, hidden)

    mask = encoded['attention_mask'].astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (1, seq, 1)
    summed = (last_hidden * mask).sum(axis=1)  # (1, hidden)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = (summed / counts).squeeze(0)  # (hidden,)

    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32, copy=False)


def _get_axis_embeddings():
    global _axis_label_map, _axis_embeddings
    if _axis_label_map is not None and _axis_embeddings is not None:
        return _axis_label_map, _axis_embeddings

    tokenizer, session = load_topic_embedding_model()
    label_map: Dict[str, List[Tuple[str, str]]] = {}
    embeddings: Dict[str, np.ndarray] = {}
    for axis_name, axis_meta in MUSIC_ANALYSIS_AXES.items():
        labels = list(axis_meta.get('labels', {}).items())
        label_map[axis_name] = labels
        vectors = []
        for _, description in labels:
            vec = _embed_text(description, tokenizer, session)
            if vec is not None:
                vectors.append(vec)
        embeddings[axis_name] = (np.stack(vectors)
                                 if vectors else np.zeros((0, 0), dtype=np.float32))

    _axis_label_map = label_map
    _axis_embeddings = embeddings
    return _axis_label_map, _axis_embeddings


def embed_query_text(text: str) -> Optional[np.ndarray]:
    """Embed a free-form user query with the same e5 model used at analysis time.

    Returns a normalized float32 vector or ``None`` if the model is not yet
    ready or the embedding pass fails. Callers should treat ``None`` as
    "index/cache not ready, retry later" rather than as a fatal error.
    """
    if not text or not text.strip():
        return None
    try:
        tokenizer, session = load_topic_embedding_model()
    except Exception as exc:
        logger.warning('Embedding model not ready (%s); returning no query vector', exc)
        return None
    try:
        vec = _embed_text(text.strip(), tokenizer, session)
    except Exception as exc:
        logger.warning('Embedding pass failed for query %r: %s', text, exc)
        return None
    if vec is None:
        return None
    return vec.astype(np.float32, copy=False)


def reset_session() -> None:
    """Drop the cached e5 tokenizer + ONNX session + axis embeddings.

    Called by ``tasks.lyrics_manager._unload_lyrics_caches`` when the lyrics
    idle window elapses, so the ~440 MB ONNX session + the axis-embedding
    matrix don't survive forever after the user stops searching. The next
    search lazy-loads everything again.
    """
    global _embedding_tokenizer, _embedding_session, _embedding_model_name
    global _embedding_input_names, _axis_label_map, _axis_embeddings
    with _embedding_load_lock:
        _embedding_tokenizer = None
        _embedding_session = None
        _embedding_model_name = None
        _embedding_input_names = ()
        _axis_label_map = None
        _axis_embeddings = None


__all__ = [
    'embed_query_text',
    'reset_session',
    'load_topic_embedding_model',
]
