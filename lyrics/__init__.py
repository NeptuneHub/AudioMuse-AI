"""Lyrics package entry point.

The heavy ML imports inside ``lyrics_transcriber`` (whisper, torch, silero-vad,
llama-cpp, transformers) are gated behind ``LYRICS_ENABLED`` so the no-AVX2
image — which intentionally does not ship those wheels — can boot cleanly.
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


if _LYRICS_ENABLED:
    try:
        from .lyrics_transcriber import (
            MUSIC_ANALYSIS_AXES,
            analyze_lyrics,
            axis_columns,
            embed_query_text,
            load_llama_model,
            load_topic_embedding_model,
            load_whisper_model,
        )
    except Exception as _exc:  # pragma: no cover - defensive
        _logger.warning(
            "Lyrics module failed to load (%s); disabling lyrics features.",
            _exc,
        )
        MUSIC_ANALYSIS_AXES = {}
        analyze_lyrics = _disabled
        axis_columns = _disabled
        embed_query_text = _disabled
        load_llama_model = _disabled
        load_topic_embedding_model = _disabled
        load_whisper_model = _disabled
else:
    _logger.info("Lyrics features are disabled (LYRICS_ENABLED=false).")
    MUSIC_ANALYSIS_AXES = {}
    analyze_lyrics = _disabled
    axis_columns = _disabled
    embed_query_text = _disabled
    load_llama_model = _disabled
    load_topic_embedding_model = _disabled
    load_whisper_model = _disabled


__all__ = [
    'MUSIC_ANALYSIS_AXES',
    'analyze_lyrics',
    'axis_columns',
    'embed_query_text',
    'load_llama_model',
    'load_topic_embedding_model',
    'load_whisper_model',
]
