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
    except Exception as _exc:
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


def _safe_call(label, fn):
    try:
        return fn()
    except Exception as exc:
        _logger.warning("Lyrics %s: %s", label, exc)
        return None


def is_lyrics_loaded() -> bool:
    if not _LYRICS_ENABLED:
        return False
    try:
        from . import whisper_onnx, gte_onnx, silero_onnx
    except Exception:
        return False
    for mod in (whisper_onnx, gte_onnx, silero_onnx):
        try:
            if mod.is_loaded():
                return True
        except Exception:
            return True
    return False


def unload_lyrics_models() -> bool:
    if not _LYRICS_ENABLED:
        return False
    released_any = False
    try:
        try:
            from . import whisper_onnx

            if whisper_onnx.is_loaded():
                released_any = bool(_safe_call('whisper_onnx.unload', whisper_onnx.unload))
        except Exception as exc:
            _logger.warning("Lyrics whisper_onnx release failed: %s", exc)

        try:
            from . import gte_onnx

            if gte_onnx.is_loaded():
                _safe_call('gte_onnx.reset_session', gte_onnx.reset_session)
                released_any = True
        except Exception as exc:
            _logger.warning("Lyrics gte_onnx release failed: %s", exc)

        try:
            from . import silero_onnx

            if silero_onnx.is_loaded():
                _safe_call('silero_onnx.reset_session', silero_onnx.reset_session)
                released_any = True
        except Exception as exc:
            _logger.warning("Lyrics silero_onnx release failed: %s", exc)
    finally:
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        try:
            from tasks.memory_utils import comprehensive_memory_cleanup

            comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=True)
        except Exception as exc:
            _logger.warning("Lyrics final memory cleanup failed: %s", exc)
    if released_any:
        _logger.info("Lyrics models unloaded (~2 GB freed)")
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
