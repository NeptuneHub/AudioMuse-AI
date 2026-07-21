# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""CLAP audio/text embedding backend used during analysis and text search.

Loads the CLAP ONNX audio and text encoders and produces the shared embedding
space that links a track's audio to natural-language descriptions. The analysis
pipeline calls analyze_audio_file per track; tasks.clap_text_search calls the
text-encoder side to embed queries against the stored audio embeddings.

Main Features:
* initialize/unload/get accessors that load the audio and text models lazily and
  free them independently to keep worker RSS low between jobs.
* analyze_audio_file: segment audio, build mel spectrograms, and embed each track.
* get_text_embedding(_batch) plus other-feature label embeddings for CLAP-derived
  scalar features, cached in Redis to avoid re-encoding fixed label sets.
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import config

try:
    from config import AUDIO_LOAD_TIMEOUT
except Exception:
    AUDIO_LOAD_TIMEOUT = None
from tasks.memory_utils import (
    cleanup_cuda_memory,
    handle_onnx_memory_error,
    comprehensive_memory_cleanup,
)

logger = logging.getLogger(__name__)

_audio_session = None
_text_session = None
_tokenizer = None
_label_text_embeddings_cache = None

_SEGMENT_LENGTH_SAMPLES = 480000

# Providers whose graph compiler needs the CLAP audio model's symbolic
# time-frame axis pinned to a fixed value before it can compile the graph
# (neither compiles a dynamic dim).
_PREPARED_MODEL_PROVIDERS = {'CoreMLExecutionProvider', 'MIGraphXExecutionProvider'}


def _prepared_model_bytes(model_path):
    """Return the audio model with its symbolic time axis pinned to a static
    shape, or None when the model has no symbolic dims. See _PREPARED_MODEL_PROVIDERS."""
    import onnx
    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

    hop = config.CLAP_AUDIO_HOP_LENGTH
    frames = 1 + _SEGMENT_LENGTH_SAMPLES // hop

    model = onnx.load(model_path, load_external_data=True)
    symbolic = {
        d.dim_param
        for inp in model.graph.input
        for d in inp.type.tensor_type.shape.dim
        if d.HasField('dim_param')
    }
    if not symbolic:
        return None
    for name in symbolic:
        make_dim_param_fixed(model.graph, name, frames)
    return model.SerializeToString()


def _clap_session_options(label):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    if config.CLAP_PYTHON_MULTITHREADS:
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        logger.info("CLAP %s: Using Python threading, ONNX single-threaded", label)
    else:
        logger.info("CLAP %s: Using ONNX Runtime automatic thread management", label)
    return sess_options


def _load_audio_model():
    import onnxruntime as ort
    import gc

    model_path = config.CLAP_AUDIO_MODEL_PATH
    logger.info(f"Loading CLAP audio model from {model_path}...")

    data_file = model_path + ".data"
    if not os.path.exists(data_file):
        data_file = os.path.splitext(model_path)[0] + ".data"
    has_external_data = os.path.exists(data_file)
    if has_external_data:
        logger.info(f"External data file detected: {data_file}")

    sess_options = _clap_session_options("Audio")

    session = None

    from tasks.analysis.song import resolve_providers

    provider_options = resolve_providers(
        allow_coreml=True,
        cuda_options={
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'DEFAULT',
        },
        label='clap',
    )

    def _create_session(model_input, providers, provider_opts):
        return ort.InferenceSession(
            model_input,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_opts,
        )

    preferred_providers = [p[0] for p in provider_options]
    preferred_opts = [p[1] for p in provider_options]
    cpu_providers = ['CPUExecutionProvider']
    cpu_opts = [{}]

    preferred_model_input = model_path
    if _PREPARED_MODEL_PROVIDERS.intersection(preferred_providers):
        try:
            prepared_bytes = _prepared_model_bytes(model_path)
            if prepared_bytes is not None:
                preferred_model_input = prepared_bytes
                logger.info("Pinned CLAP audio time axis to a static shape for graph compilation")
        except Exception as e:
            logger.warning(f"Could not build static-shape model ({e}); using dynamic model")

    try:
        session = _create_session(preferred_model_input, preferred_providers, preferred_opts)
        logger.info("OK CLAP audio model loaded successfully (direct path)")

    except Exception as direct_err:
        logger.warning(f"Direct path load failed: {direct_err}")

        if has_external_data:
            logger.info("Trying in-memory external-data fallback…")
            try:
                import onnx as _onnx

                _model_proto = _onnx.load(model_path, load_external_data=True)
                _model_bytes = _model_proto.SerializeToString()
                del _model_proto
                gc.collect()
                session = _create_session(_model_bytes, preferred_providers, preferred_opts)
                logger.info("OK CLAP audio model loaded (in-memory external data)")
            except Exception as mem_err:
                logger.warning(f"In-memory fallback failed: {mem_err}")
                session = None
        else:
            session = None

        if session is None:
            logger.info("Attempting final CPU-only fallback…")
            try:
                session = _create_session(model_path, cpu_providers, cpu_opts)
                logger.info("OK CLAP audio model loaded (CPU fallback, direct path)")
            except Exception:
                logger.exception("Failed to load ONNX audio model even with CPU")
                raise

    if session is None:
        raise RuntimeError("Failed to create audio ONNX session")

    gc.collect()
    return session


def _load_text_model():
    import onnxruntime as ort
    import gc

    model_path = config.CLAP_TEXT_MODEL_PATH
    logger.info(f"Loading CLAP text model from {model_path}...")

    sess_options = _clap_session_options("Text")

    session = None
    available_providers = ort.get_available_providers()
    _is_worker = os.environ.get('AUDIOMUSE_ROLE') == 'worker'

    if not _is_worker:
        provider_options = [('CPUExecutionProvider', {})]
        logger.info(
            "CLAP text model: CPU only (Flask process) - thread-safe across request threads"
        )
    elif 'CUDAExecutionProvider' in available_providers:
        gpu_device_id = 0
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible != '-1':
            gpu_device_id = 0

        cuda_options = {
            'device_id': gpu_device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'DEFAULT',
        }
        provider_options = [('CUDAExecutionProvider', cuda_options), ('CPUExecutionProvider', {})]
        logger.info(
            f"CUDA provider available - will attempt to use GPU (device_id={gpu_device_id})"
        )
    else:
        provider_options = [('CPUExecutionProvider', {})]
        logger.info("CUDA provider not available - using CPU only")

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[p[0] for p in provider_options],
            provider_options=[p[1] for p in provider_options],
        )

        logger.info("OK CLAP text model loaded successfully (~478MB)")

    except Exception as e:
        logger.warning(f"Failed to load with preferred providers: {e}")
        logger.info("Attempting final CPU-only fallback...")
        try:
            session = ort.InferenceSession(
                model_path, sess_options=sess_options, providers=['CPUExecutionProvider']
            )
            logger.info("OK CLAP text model loaded successfully (CPU fallback)")
        except Exception:
            logger.exception("Failed to load ONNX text model even with CPU")
            raise

    if session is None:
        raise RuntimeError("Failed to create text ONNX session")

    gc.collect()
    return session


def _load_tokenizer():
    from transformers import AutoTokenizer

    logger.info("Loading RoBERTa tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", local_files_only=True)

    logger.info("OK Tokenizer loaded successfully")
    return tokenizer


def initialize_clap_audio_model():
    global _audio_session

    if not config.CLAP_ENABLED:
        logger.info("CLAP is disabled in config. Skipping audio model initialization.")
        return False

    if _audio_session is not None:
        logger.debug("CLAP audio model already initialized.")
        return True

    if not os.path.exists(config.CLAP_AUDIO_MODEL_PATH):
        logger.error(f"CLAP audio model not found at {config.CLAP_AUDIO_MODEL_PATH}")
        return False

    try:
        _audio_session = _load_audio_model()
        logger.info("OK CLAP audio model initialized successfully (for music analysis)")
        return True
    except Exception:
        logger.exception("Failed to initialize CLAP audio model")
        return False


def initialize_clap_text_model():
    global _text_session, _tokenizer

    if not config.CLAP_ENABLED:
        logger.info("CLAP is disabled in config. Skipping text model initialization.")
        return False

    if _text_session is not None:
        logger.debug("CLAP text model already initialized.")
        return True

    if not os.path.exists(config.CLAP_TEXT_MODEL_PATH):
        logger.error(f"CLAP text model not found at {config.CLAP_TEXT_MODEL_PATH}")
        return False

    try:
        _tokenizer = _load_tokenizer()
        _text_session = _load_text_model()
        logger.info("OK CLAP text model initialized successfully (for text search)")
        return True
    except Exception:
        logger.exception("Failed to initialize CLAP text model")
        return False


def unload_clap_audio_only():
    global _audio_session
    if _audio_session is None:
        return False
    try:
        _audio_session = None
        import gc

        gc.collect()
        from .memory_utils import cleanup_cuda_memory

        cleanup_cuda_memory(force=True)
        logger.info("OK CLAP audio model unloaded (~268MB freed), text cache preserved")
        return True
    except Exception:
        logger.exception("Error unloading CLAP audio model")
        return False


def unload_clap_model():
    global _audio_session, _text_session, _tokenizer

    if _audio_session is None and _text_session is None:
        return False

    try:
        freed_mb = 0
        if _audio_session is not None:
            _audio_session = None
            freed_mb += 268
        if _text_session is not None:
            _text_session = None
            freed_mb += 478

        _tokenizer = None

        import gc

        gc.collect()

        from .memory_utils import comprehensive_memory_cleanup

        comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)

        logger.info(
            f"OK CLAP model(s) unloaded from memory (~{freed_mb}MB freed + GPU memory released)"
        )
        return True
    except Exception:
        logger.exception("Error unloading CLAP model")
        return False


def is_clap_model_loaded():
    return _audio_session is not None or _text_session is not None


def is_clap_text_loaded():
    return _text_session is not None


def get_clap_audio_model():
    if _audio_session is None:
        logger.info("Lazy-loading CLAP audio model on first use...")
        if not initialize_clap_audio_model():
            raise RuntimeError("Failed to initialize CLAP audio model")
    return _audio_session


def get_clap_text_model():
    if _text_session is None:
        logger.info("Lazy-loading CLAP text model on first use...")
        if not initialize_clap_text_model():
            raise RuntimeError("Failed to initialize CLAP text model")
    return _text_session


def get_tokenizer():
    if _tokenizer is None and not initialize_clap_text_model():
        raise RuntimeError("CLAP tokenizer could not be initialized")
    return _tokenizer


def compute_mel_spectrogram(audio_data: np.ndarray, sr: int = 48000) -> np.ndarray:
    import librosa

    n_fft = config.CLAP_AUDIO_N_FFT
    hop_length = config.CLAP_AUDIO_HOP_LENGTH
    n_mels = config.CLAP_AUDIO_N_MELS
    f_min = config.CLAP_AUDIO_FMIN
    f_max = config.CLAP_AUDIO_FMAX
    transpose = config.CLAP_AUDIO_MEL_TRANSPOSE

    mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )

    mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)

    if transpose:
        mel = mel.T
        mel = mel[np.newaxis, np.newaxis, :, :]
    else:
        mel = mel[np.newaxis, np.newaxis, :, :]

    return mel.astype(np.float32)


def analyze_audio_file(audio_path: str) -> Tuple[Optional[np.ndarray], float, int]:
    if not config.CLAP_ENABLED:
        return None, 0, 0

    try:
        session = get_clap_audio_model()

        SAMPLE_RATE = 48000
        SEGMENT_LENGTH = _SEGMENT_LENGTH_SAMPLES
        HOP_LENGTH = 240000

        from tasks.analysis import robust_load_audio_with_fallback

        audio_data, sr = robust_load_audio_with_fallback(audio_path, target_sr=SAMPLE_RATE)

        if audio_data is None or audio_data.size == 0:
            logger.warning(f"Could not load audio for CLAP analysis: {audio_path}")
            return None, 0, 0

        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767.0).astype(np.int16)
        audio_data = (audio_data / 32767.0).astype(np.float32)

        duration_sec = len(audio_data) / SAMPLE_RATE

        segments = []
        total_length = len(audio_data)

        if total_length <= SEGMENT_LENGTH:
            padded = np.pad(audio_data, (0, SEGMENT_LENGTH - total_length), mode='constant')
            segments.append(padded)
        else:
            for start in range(0, total_length - SEGMENT_LENGTH + 1, HOP_LENGTH):
                segments.append(audio_data[start : start + SEGMENT_LENGTH])
            last_start = len(segments) * HOP_LENGTH
            if last_start < total_length:
                segments.append(audio_data[-SEGMENT_LENGTH:])

        num_segments = len(segments)
        logger.info(f"CLAP: Processing {num_segments} segments ({duration_sec:.1f}s audio)")

        all_embs = []
        for seg_idx, seg in enumerate(segments):
            mel_spec = compute_mel_spectrogram(seg, SAMPLE_RATE)
            onnx_inputs = {'mel_spectrogram': mel_spec}
            try:
                outputs = session.run(None, onnx_inputs)
                emb = outputs[0]
            except Exception as e:

                def cleanup_fn():
                    cleanup_cuda_memory(force=True)

                def retry_fn(onnx_inputs=onnx_inputs):
                    return session.run(None, onnx_inputs)

                result = handle_onnx_memory_error(
                    e,
                    f"CLAP segment {seg_idx}/{num_segments}",
                    cleanup_func=cleanup_fn,
                    retry_func=retry_fn,
                )
                if result is not None:
                    emb = result[0]
                else:
                    raise
            all_embs.append(emb)

        if all_embs:
            audio_embeddings = np.vstack(all_embs)
        else:
            audio_embeddings = np.zeros((0, config.CLAP_EMBEDDING_DIMENSION), dtype=np.float32)

        num_segments = audio_embeddings.shape[0]
        if num_segments > 0:
            audio_embedding = np.mean(audio_embeddings, axis=0)
            audio_embedding = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-9)
        else:
            audio_embedding = np.zeros((config.CLAP_EMBEDDING_DIMENSION,), dtype=np.float32)

        return audio_embedding, duration_sec, num_segments

    except Exception:
        logger.exception(f"CLAP analysis failed for {audio_path}")
        comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
        return None, 0, 0
    finally:
        import gc

        gc.collect()


def get_text_embedding(query_text: str) -> Optional[np.ndarray]:
    if not config.CLAP_ENABLED:
        return None

    try:
        session = get_clap_text_model()
        tokenizer = get_tokenizer()

        encoded = tokenizer(
            query_text, max_length=77, padding='max_length', truncation=True, return_tensors='np'
        )

        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)

        onnx_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

        outputs = session.run(None, onnx_inputs)
        text_embedding = outputs[0]

        text_embedding = text_embedding[0]

        text_embedding = text_embedding / np.linalg.norm(text_embedding)

        return text_embedding

    except Exception:
        logger.exception(f"Failed to get text embedding for '{query_text}'")
        return None


def get_text_embeddings_batch(query_texts: list) -> Optional[np.ndarray]:
    if not config.CLAP_ENABLED:
        return None

    if not query_texts:
        return None

    try:
        session = get_clap_text_model()
        tokenizer = get_tokenizer()

        encoded = tokenizer(
            query_texts, max_length=77, padding='max_length', truncation=True, return_tensors='np'
        )

        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)

        onnx_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

        outputs = session.run(None, onnx_inputs)
        text_embeddings = outputs[0]

        norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / norms

        return text_embeddings

    except Exception:
        logger.exception("Failed to get batch text embeddings")
        return None


def is_clap_available() -> bool:
    if not config.CLAP_ENABLED:
        return False

    return os.path.exists(config.CLAP_AUDIO_MODEL_PATH) and os.path.exists(
        config.CLAP_TEXT_MODEL_PATH
    )


def get_or_cache_other_feature_text_embeddings(redis_conn) -> Optional[dict]:
    if not config.CLAP_ENABLED:
        logger.warning("CLAP is disabled, cannot compute other feature text embeddings")
        return None

    global _label_text_embeddings_cache
    if _label_text_embeddings_cache is not None:
        logger.debug("Using in-process cached CLAP text embeddings")
        return _label_text_embeddings_cache

    cache_key = config.CLAP_OTHER_FEATURES_REDIS_KEY

    try:
        cached_blob = redis_conn.get(cache_key)
        if cached_blob is not None:
            import io

            buf = io.BytesIO(cached_blob)
            npz = np.load(buf)
            result = {label: npz[label] for label in npz.files}
            missing = [lbl for lbl in config.OTHER_FEATURE_LABELS if lbl not in result]
            if not missing:
                logger.info(f"Loaded CLAP text embeddings from Redis cache ({len(result)} labels)")
                _label_text_embeddings_cache = result
                return result
            else:
                logger.warning(f"Cached embeddings missing labels: {missing}. Recomputing...")
    except Exception as e:
        logger.warning(f"Failed to read CLAP text embeddings from Redis: {e}")

    logger.info(
        f"Computing CLAP text embeddings for config.OTHER_FEATURE_LABELS: {config.OTHER_FEATURE_LABELS}"
    )
    try:
        embeddings = get_text_embeddings_batch(config.OTHER_FEATURE_LABELS)
        if embeddings is None:
            logger.error("Failed to compute CLAP text embeddings")
            return None

        result = {label: embeddings[i] for i, label in enumerate(config.OTHER_FEATURE_LABELS)}
        _label_text_embeddings_cache = result
        try:
            import io

            buf = io.BytesIO()
            np.savez_compressed(buf, **result)
            buf.seek(0)
            redis_conn.set(cache_key, buf.read())
            logger.info(f"Cached CLAP text embeddings in Redis ({buf.tell()} bytes)")
        except Exception as e:
            logger.warning(f"Failed to write text embeddings to Redis: {e}")
        return result
    except Exception:
        logger.exception("Failed to compute CLAP text embeddings for other features")
        return None
    finally:
        global _text_session, _tokenizer
        if _text_session is not None:
            _text_session = None
            _tokenizer = None
            import gc

            gc.collect()
            logger.info("Unloaded CLAP text model after computing other feature embeddings")


def compute_other_features_from_clap(audio_embedding: np.ndarray, label_embeddings: dict) -> dict:
    result = {}
    for label, text_emb in label_embeddings.items():
        similarity = float(np.dot(audio_embedding, text_emb))
        result[label] = (similarity + 1.0) / 2.0
    return result
