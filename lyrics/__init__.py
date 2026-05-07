"""Lyrics package entry point.

The heavy ML imports inside ``lyrics_transcriber`` (whisper-onnx, silero-onnx,
translation-onnx via onnxruntime + transformers tokenizer) are gated behind
``LYRICS_ENABLED`` so the no-AVX2 image — which intentionally does not ship
the bundled ONNX artifacts — can boot cleanly.
"""

import logging as _logging

try:
    from config import LYRICS_ENABLED as _LYRICS_ENABLED
except Exception:
    _LYRICS_ENABLED = True

_logger = _logging.getLogger(__name__)


def _disabled(*_args, **_kwargs):
    raise RuntimeError(
        "Lyrics analysis is disabled (LYRICS_ENABLED=false) or its dependencies "
        "are not installed in this image."
    )


from .axes import MUSIC_ANALYSIS_AXES, axis_columns
from .embeddings import embed_query_text, load_topic_embedding_model as _load_topic_embedding_model

_analyze_lyrics = _disabled
_load_whisper_model = _disabled

if _LYRICS_ENABLED:
    def _load_transcriber_exports() -> None:
        global _analyze_lyrics, _load_whisper_model
        if _analyze_lyrics is not _disabled:
            return
        try:
            from .lyrics_transcriber import (
                analyze_lyrics as _transcriber_analyze_lyrics,
                load_whisper_model as _transcriber_load_whisper_model,
            )
            _analyze_lyrics = _transcriber_analyze_lyrics
            _load_whisper_model = _transcriber_load_whisper_model
        except Exception as _exc:  # pragma: no cover - defensive
            _logger.warning(
                "Lyrics module failed to load (%s); disabling lyrics features.",
                _exc,
            )
            _analyze_lyrics = _disabled
            _load_whisper_model = _disabled
else:
    _logger.info("Lyrics features are disabled (LYRICS_ENABLED=false).")


def analyze_lyrics(*args, **kwargs):
    _load_transcriber_exports()
    return _analyze_lyrics(*args, **kwargs)


def load_topic_embedding_model(*args, **kwargs):
    return _load_topic_embedding_model(*args, **kwargs)


def load_whisper_model(*args, **kwargs):
    _load_transcriber_exports()
    return _load_whisper_model(*args, **kwargs)


__all__ = [
    'MUSIC_ANALYSIS_AXES',
    'analyze_lyrics',
    'axis_columns',
    'embed_query_text',
    'load_topic_embedding_model',
    'load_whisper_model',
]
