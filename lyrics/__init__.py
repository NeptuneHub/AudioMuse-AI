"""Lyrics package entry point.

The heavy ML imports inside ``lyrics_transcriber`` (Qwen3-ASR ONNX, silero-vad
ONNX, e5 ONNX, MarianMT ONNX) are gated behind ``LYRICS_ENABLED`` so the
no-AVX2 image — which intentionally does not ship those wheels — can boot
cleanly.
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
            load_topic_embedding_model,
            load_asr_model,
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
        load_topic_embedding_model = _disabled
        load_asr_model = _disabled
else:
    _logger.info("Lyrics features are disabled (LYRICS_ENABLED=false).")
    MUSIC_ANALYSIS_AXES = {}
    analyze_lyrics = _disabled
    axis_columns = _disabled
    embed_query_text = _disabled
    load_topic_embedding_model = _disabled
    load_asr_model = _disabled


# ── Album-lifecycle release (mirrors tasks.clap_analyzer) ───────────────
# Each submodule is queried/released independently. The CLAP pattern at
# tasks.analysis_helper._OPTIONAL_MODELS expects this pair; registering
# ('lyrics', 'lyrics', 'is_lyrics_loaded', 'unload_lyrics_models') there
# is what wires us into the existing per-album cleanup at
# tasks.analysis.analyze_album_task's `finally`.

def _safe_call(label, fn):
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("Lyrics %s: %s", label, exc)
        return None


def is_lyrics_loaded() -> bool:
    """True if ANY of the four lyrics ONNX models are currently cached."""
    if not _LYRICS_ENABLED:
        return False
    try:
        from . import qwen_asr, e5_onnx, silero_onnx, translation_onnx
    except Exception:
        return False
    for mod in (qwen_asr, e5_onnx, silero_onnx, translation_onnx):
        try:
            if mod.is_loaded():
                return True
        except Exception:
            # If a submodule's is_loaded throws, assume yes so we still try
            # to release whatever state it might be holding.
            return True
    return False


def unload_lyrics_models() -> bool:
    """Release every lyrics ONNX session held by this worker process.

    Called from ``tasks.analysis_helper.cleanup_optional_models`` at album
    end (and in the surrounding ``finally`` of ``analyze_album_task``), so
    a 4 GB resident footprint is bounded to the lifetime of the one album
    that triggered ASR — not the whole worker process.

    Every release runs in its own try/except inside an overarching
    try/finally so a failure in one submodule cannot leak the others, and
    the final ``comprehensive_memory_cleanup`` always runs.
    """
    if not _LYRICS_ENABLED:
        return False
    released_any = False
    try:
        try:
            from . import qwen_asr
            if qwen_asr.is_loaded():
                released_any = bool(_safe_call('qwen_asr.unload', qwen_asr.unload))
        except Exception as exc:
            _logger.warning("Lyrics qwen_asr release failed: %s", exc)

        try:
            from . import translation_onnx
            if translation_onnx.is_loaded():
                _safe_call('translation_onnx.reset_session',
                           translation_onnx.reset_session)
                released_any = True
        except Exception as exc:
            _logger.warning("Lyrics translation_onnx release failed: %s", exc)

        try:
            from . import e5_onnx
            if e5_onnx.is_loaded():
                _safe_call('e5_onnx.reset_session', e5_onnx.reset_session)
                released_any = True
        except Exception as exc:
            _logger.warning("Lyrics e5_onnx release failed: %s", exc)

        try:
            from . import silero_onnx
            if silero_onnx.is_loaded():
                _safe_call('silero_onnx.reset_session', silero_onnx.reset_session)
                released_any = True
        except Exception as exc:
            _logger.warning("Lyrics silero_onnx release failed: %s", exc)
    finally:
        # Always run the GC + ONNX memory-pool reset, even if every
        # per-submodule release above raised. ``force_cuda=False`` because
        # the lyrics ONNX sessions are all CPUExecutionProvider — CUDA
        # cleanup is a no-op for them and would just slow shutdown.
        try:
            import gc
            gc.collect()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            from tasks.memory_utils import comprehensive_memory_cleanup
            comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=True)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("Lyrics final memory cleanup failed: %s", exc)
    if released_any:
        _logger.info("Lyrics models unloaded (~4 GB freed)")
    return released_any


__all__ = [
    'MUSIC_ANALYSIS_AXES',
    'analyze_lyrics',
    'axis_columns',
    'embed_query_text',
    'load_topic_embedding_model',
    'load_asr_model',
    'is_lyrics_loaded',
    'unload_lyrics_models',
]
