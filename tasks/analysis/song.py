# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-song analysis: audio decode, MusiCNN/CLAP/lyrics models and their persistence.

Everything that happens to ONE audio file lives here: loading it (librosa with a
PyAV fallback), the MusiCNN analyze_track pipeline with ONNX session management
and OOM-to-CPU retry, the CLAP and lyrics stages, and the DB writes that store
each result under the canonical catalogue id.

Main Features:
* analyze_track / robust_load_audio_with_fallback: decode a file and produce the
  MusiCNN moods + embedding; a track that cannot be decoded returns None.
* run_clap_for_track / run_lyrics_for_track: the optional per-song stages; every
  failure is recorded through the central error registry and never raised past
  the stage (a DB outage is the one exception: it re-raises so the album retries).
* persist_musicnn_results / persist_clap_embedding / refresh_other_features:
  the writes; other_features starts as zeros and is refreshed when CLAP lands.
* run_song_analyzed_hook: the plugin hook, carrying the server the song was
  analyzed from (server_id / server_name).
"""

import gc
import importlib
import logging
import os

import numpy as np
import librosa
import onnxruntime as ort

from cpu_budget import usable_cpu_count

from config import (
    AUDIO_LOAD_TIMEOUT,
    MUSICNN_BATCH_SIZE,
    OTHER_FEATURE_LABELS,
    PER_SONG_MODEL_RELOAD,
)
from database import (
    get_db,
    save_track_analysis_and_embedding,
    save_clap_embedding,
    save_lyrics_embedding,
)
from psycopg2 import OperationalError

from sanitization import sanitize_string_for_db

from error import error_manager
from error.error_dictionary import (
    ERR_DB_QUERY,
    ERR_LYRICS_TRANSCRIPTION,
    ERR_MODEL_INFERENCE,
)

from ..memory_utils import cleanup_cuda_memory, cleanup_onnx_session, comprehensive_memory_cleanup


logger = logging.getLogger(__name__)


DEFINED_TENSOR_NAMES = {
    'embedding': {'input': 'model/Placeholder:0', 'output': 'model/dense/BiasAdd:0'},
    'prediction': {'input': 'serving_default_model_Placeholder:0', 'output': 'PartitionedCall:0'},
}


def _find_onnx_name(candidate, names):
    if not names:
        return None
    stripped = candidate.split(':')[0]
    for cand in (candidate, stripped, stripped.split('/')[-1], stripped.replace('/', '_')):
        if cand in names:
            return cand
    return names[0]


def run_inference(session, feed_dict, output_tensor_name=None):
    input_names = [i.name for i in session.get_inputs()]
    mapped = {}
    for k, v in feed_dict.items():
        name = _find_onnx_name(k, input_names)
        if name is None:
            logger.error(f"Could not map input '{k}' to ONNX inputs {input_names}")
            return None
        mapped[name] = v
    output_names = [o.name for o in session.get_outputs()]
    default_output = output_names[0] if output_names else None
    out = (
        _find_onnx_name(output_tensor_name, output_names)
        if output_tensor_name
        else default_output
    )
    if out is None:
        logger.error("No ONNX output name available to run inference.")
        return None
    result = session.run([out], mapped)
    return result[0] if isinstance(result, list) and len(result) > 0 else result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The session labels a plugin can scope an ONNX provider to. Every session that
# calls resolve_providers passes one of these, so anything else in a plugin's
# only_models/exclude_models is a typo that would silently match nothing.
MODEL_LABELS = frozenset({
    'musicnn',
    'clap',
    'clap_text',
    'whisper_encoder',
    'whisper_decoder',
    'gte',
    'silero_vad',
})

def _scoped_labels(provider, key):
    """Return a provider's model scope as a list, warning about unknown labels."""
    value = provider.get(key)
    if not value:
        return None
    labels = [value] if isinstance(value, str) else list(value)
    unknown = [name for name in labels if name not in MODEL_LABELS]
    if unknown:
        logger.warning(
            "ONNX provider %s: unknown %s %s - known model labels are %s",
            provider.get('name'), key, unknown, sorted(MODEL_LABELS),
        )
    return labels


def _add_plugin_provider(chain, provider, entry):
    position = provider.get('position') or 'before_cpu'
    if position == 'before_cuda':
        chain.insert(0, entry)
        return
    if position != 'before_cpu':
        logger.warning(
            "ONNX provider %s: unknown position %r - using 'before_cpu'",
            provider.get('name'), position,
        )
    chain.append(entry)


def resolve_providers(allow_coreml=False, cuda_options=None, label=None,
                      cpu_only_default=False):
    """Build the ONNX provider chain for one session.

    ``label`` names the model (see MODEL_LABELS) so a plugin can offer its
    accelerator for only the graphs it can compile. ``cpu_only_default`` marks a
    session that core keeps on CPU: the built-in GPU providers are skipped and a
    plugin provider is used only when it names the label in ``only_models``.
    """
    available = ort.get_available_providers()
    chain = []

    if not cpu_only_default and 'CUDAExecutionProvider' in available:
        chain.append(
            (
                'CUDAExecutionProvider',
                cuda_options
                or {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                },
            )
        )

    if not cpu_only_default and allow_coreml and 'CoreMLExecutionProvider' in available:
        chain.append(
            (
                'CoreMLExecutionProvider',
                {
                    'MLComputeUnits': 'ALL',
                    'ModelFormat': 'MLProgram',
                },
            )
        )

    for provider in _plugin_onnx_providers():
        name = provider.get('name')
        if not name or name not in available or name in [p[0] for p in chain]:
            continue
        # Providers can be scoped to specific models by their session label, so a
        # plugin can offer an accelerator for the graphs it handles
        only = _scoped_labels(provider, 'only_models')
        exclude = _scoped_labels(provider, 'exclude_models')
        if only and label not in only:
            continue
        if exclude and label in exclude:
            continue
        # A CPU-by-default session is opt-in: the plugin has to name it.
        if cpu_only_default and not only:
            continue
        _add_plugin_provider(chain, provider, (name, provider.get('options') or {}))

    chain.append(('CPUExecutionProvider', {}))
    logger.info("ONNX provider chain for %s: %s", label or 'unlabelled', [p[0] for p in chain])
    return chain


def _plugin_onnx_providers():
    try:
        from plugin.manager import plugin_manager
        return plugin_manager.get_onnx_providers()
    except Exception:
        return []


def analysis_server_identity():
    try:
        from tasks.mediaserver import context, registry

        server = context.active_server()
        if server is None:
            server = registry.get_default_server()
        if not server:
            return None, None
        server_id = server.get('server_id')
        return (str(server_id) if server_id else None), server.get('name')
    except Exception:
        logger.debug("Could not resolve the analysis server identity", exc_info=True)
        return None, None


def run_song_analyzed_hook(item, audio_path, musicnn_analysis, musicnn_embedding,
                           clap_embedding, top_moods, album_id, album_name, run_id):
    try:
        from plugin.manager import plugin_manager
        if not plugin_manager.enabled() or not plugin_manager.song_analyzed_hooks():
            return
        server_id, server_name = analysis_server_identity()
        payload = {
            'item_id': str(item.get('Id')),
            'run_id': run_id,
            'server_id': server_id,
            'server_name': server_name,
            'audio_path': audio_path,
            'metadata': {
                'title': item.get('Name'),
                'artist': item.get('AlbumArtist'),
                'album': item.get('Album'),
                'album_artist': item.get('OriginalAlbumArtist') or item.get('AlbumArtist'),
                'year': item.get('Year'),
                'rating': item.get('Rating'),
                'file_path': item.get('FilePath'),
                'album_id': album_id,
                'album_name': album_name,
            },
            'media_item': item,
            'analysis': musicnn_analysis,
            'top_moods': top_moods,
            'musicnn_embedding': musicnn_embedding,
            'clap_embedding': clap_embedding,
        }
        plugin_manager.run_song_analyzed(payload)
    except Exception:
        logger.exception('Plugin song-analyzed hook dispatch failed')


def _default_sess_options():
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    threads = usable_cpu_count()
    if threads is not None:
        opts.intra_op_num_threads = threads
    return opts


def create_onnx_session(
    model_path, provider_options=None, label="", sess_options=None, allow_coreml=False
):
    opts = provider_options or resolve_providers(allow_coreml=allow_coreml, label=label)
    if sess_options is None:
        sess_options = _default_sess_options()
    try:
        return ort.InferenceSession(
            model_path,
            providers=[p[0] for p in opts],
            provider_options=[p[1] for p in opts],
            sess_options=sess_options,
        )
    except Exception:
        logger.warning(f"Failed to load {label or model_path} with GPU - falling back to CPU")
        return ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            sess_options=sess_options,
        )


def load_musicnn_sessions(model_paths):
    opts = resolve_providers(allow_coreml=False, label='musicnn')
    try:
        sessions = {n: create_onnx_session(p, opts, label=n) for n, p in model_paths.items()}
        logger.info(f"OK Loaded {len(sessions)} MusiCNN models for album reuse")
        return sessions
    except Exception:
        logger.exception("Failed to load MusiCNN models")
        return None


def cleanup_musicnn_sessions(onnx_sessions, context=""):
    if not onnx_sessions:
        return
    suffix = f" ({context})" if context else ""
    logger.info(f"Cleaning up {len(onnx_sessions)} MusiCNN model sessions{suffix}")
    for name in list(onnx_sessions.keys()):
        session = onnx_sessions.pop(name, None)
        try:
            cleanup_onnx_session(session, name)
        except Exception:
            logger.exception(f"Error cleaning up {name} session")
        session = None
    gc.collect()


_OPTIONAL_MODELS = (
    ('clap', 'tasks.clap_analyzer', 'is_clap_model_loaded', 'unload_clap_model'),
    ('lyrics', 'lyrics', 'is_lyrics_loaded', 'unload_lyrics_models'),
)


def cleanup_optional_models(context=""):
    suffix = f" ({context})" if context else ""
    for label, mod, is_loaded_fn, unload_fn in _OPTIONAL_MODELS:
        try:
            module = importlib.import_module(mod)
            if getattr(module, is_loaded_fn)():
                logger.info(f"Cleaning up {label.upper()} model{suffix}")
                getattr(module, unload_fn)()
        except Exception as e:
            logger.warning(f"Error cleaning up {label.upper()} model: {e}")


def run_inference_with_oom_fallback(
    session, feed_dict, output_tensor_name, model_path, label, file_basename
):
    try:
        return run_inference(session, feed_dict, output_tensor_name), session
    except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
        if "Failed to allocate memory" not in str(e):
            raise
        logger.warning(
            f"GPU OOM for {file_basename} during {label} inference - falling back to CPU"
        )
        try:
            try:
                cleanup_onnx_session(session, label)
            except Exception:
                logger.exception("Error cleaning up OOM'd %s session before CPU fallback", label)
            try:
                comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
            except Exception:
                logger.exception("Error during memory cleanup before %s CPU fallback", label)

            cpu_session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider'],
                sess_options=_default_sess_options(),
            )
            result = run_inference(cpu_session, feed_dict, output_tensor_name)
            if result is None:
                raise RuntimeError(
                    f"CPU fallback inference returned None for {label} ({file_basename})"
                )
            logger.info(f"Successfully completed {label} inference on CPU after OOM")
            return result, cpu_session
        finally:
            del session


_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


_MAJOR = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])


_MINOR = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])


def extract_basic_features(audio, sr):
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    energy = float(np.mean(librosa.feature.rms(y=audio)))
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
    maj = np.array([np.corrcoef(chroma_mean, np.roll(_MAJOR, i))[0, 1] for i in range(12)])
    mnr = np.array([np.corrcoef(chroma_mean, np.roll(_MINOR, i))[0, 1] for i in range(12)])
    mi, ni = int(np.argmax(maj)), int(np.argmax(mnr))
    if maj[mi] > mnr[ni]:
        return float(tempo), energy, _KEYS[mi], 'major'
    return float(tempo), energy, _KEYS[ni], 'minor'


def prepare_spectrogram_patches(audio, sr):
    n_mels, hop, n_fft, frame = 96, 256, 512, 187
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        window='hann',
        center=False,
        power=2.0,
        norm='slaney',
        htk=False,
    )
    log_mel = np.log10(1 + 10000 * np.maximum(mel, 0.0))
    patches = [log_mel[:, i : i + frame] for i in range(0, log_mel.shape[1] - frame + 1, frame)]
    if not patches:
        return None
    return np.array(patches).transpose(0, 2, 1).astype(np.float32)


def _decode_audio_with_pyav(file_path, target_sr):
    import av

    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=target_sr)
    max_samples = int(AUDIO_LOAD_TIMEOUT * target_sr) if (AUDIO_LOAD_TIMEOUT and target_sr) else None
    chunks = []
    total = 0
    actual_sr = target_sr
    with av.open(file_path) as container:
        if not container.streams.audio:
            return np.array([], dtype=np.float32), actual_sr
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for rframe in resampler.resample(frame):
                if actual_sr is None:
                    actual_sr = rframe.sample_rate
                    if AUDIO_LOAD_TIMEOUT:
                        max_samples = int(AUDIO_LOAD_TIMEOUT * actual_sr)
                arr = rframe.to_ndarray().reshape(-1)
                if arr.size:
                    chunks.append(arr)
                    total += arr.size
            if max_samples and total >= max_samples:
                break
        for rframe in resampler.resample(None):
            if actual_sr is None:
                actual_sr = rframe.sample_rate
                if AUDIO_LOAD_TIMEOUT:
                    max_samples = int(AUDIO_LOAD_TIMEOUT * actual_sr)
            arr = rframe.to_ndarray().reshape(-1)
            if arr.size:
                chunks.append(arr)
    if not chunks:
        return np.array([], dtype=np.float32), actual_sr
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if max_samples:
        audio = audio[:max_samples]
    return audio, actual_sr


def robust_load_audio_with_fallback(file_path, target_sr=16000):
    name = os.path.basename(file_path)
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True, duration=AUDIO_LOAD_TIMEOUT)
        if audio is None or audio.size == 0:
            raise ValueError("Librosa returned an empty audio signal.")
        return audio, sr
    except Exception as e:
        logger.warning(f"Direct librosa load failed for {name}: {e}. Attempting PyAV fallback.")

    try:
        audio, sr = _decode_audio_with_pyav(file_path, target_sr)
        if audio is None or audio.size == 0 or not np.any(audio):
            logger.error(f"PyAV fallback resulted in empty/silent audio for {name}.")
            return None, None
        return audio, sr
    except Exception:
        logger.exception(f"PyAV fallback loading also failed for {name}")
        return None, None


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type='soxr_hq')


def decode_audio_once(file_path):
    return robust_load_audio_with_fallback(file_path, target_sr=None)


def _patches_for_track(audio, sr, name):
    try:
        patches = prepare_spectrogram_patches(audio, sr)
        if patches is None:
            logger.warning(f"Track too short to create spectrogram patches: {name}")
        return patches
    except Exception:
        logger.exception(f"Spectrogram creation failed for {name}")
        return None


def _sessions_for_track(onnx_sessions, model_paths):
    if onnx_sessions:
        return onnx_sessions['embedding'], onnx_sessions['prediction'], False
    provider_options = resolve_providers(label='musicnn')
    embedding_sess = create_onnx_session(
        model_paths['embedding'], provider_options, label='embedding'
    )
    prediction_sess = create_onnx_session(
        model_paths['prediction'], provider_options, label='prediction'
    )
    return embedding_sess, prediction_sess, True


def _run_musicnn_models(final_patches, mood_labels_list, model_paths, onnx_sessions, name):
    embedding_sess = prediction_sess = None
    own_sessions = False
    try:
        embedding_sess, prediction_sess, own_sessions = _sessions_for_track(
            onnx_sessions, model_paths
        )

        # Chunked so peak memory stays flat: a whole-track batch needs several
        # GB of convolution activations, well past small worker memory caps.
        batch = MUSICNN_BATCH_SIZE if MUSICNN_BATCH_SIZE > 0 else len(final_patches)
        embedding_chunks = []
        for start in range(0, len(final_patches), batch):
            chunk, new_embedding_sess = run_inference_with_oom_fallback(
                embedding_sess,
                {DEFINED_TENSOR_NAMES['embedding']['input']: final_patches[start:start + batch]},
                DEFINED_TENSOR_NAMES['embedding']['output'],
                model_paths['embedding'],
                'embedding',
                name,
            )
            if new_embedding_sess is not embedding_sess and onnx_sessions:
                onnx_sessions['embedding'] = new_embedding_sess
            embedding_sess = new_embedding_sess
            embedding_chunks.append(chunk)
        embeddings_per_patch = np.concatenate(embedding_chunks, axis=0)

        mood_logits, new_prediction_sess = run_inference_with_oom_fallback(
            prediction_sess,
            {DEFINED_TENSOR_NAMES['prediction']['input']: embeddings_per_patch},
            DEFINED_TENSOR_NAMES['prediction']['output'],
            model_paths['prediction'],
            'prediction',
            name,
        )
        if new_prediction_sess is not prediction_sess and onnx_sessions:
            onnx_sessions['prediction'] = new_prediction_sess
        prediction_sess = new_prediction_sess

        final_mood_predictions = sigmoid(np.mean(sigmoid(mood_logits), axis=0))
        moods = {
            label: float(score)
            for label, score in zip(mood_labels_list, final_mood_predictions)
        }
        return np.mean(embeddings_per_patch, axis=0), moods
    except Exception:
        logger.exception(f"Main model inference failed for {name}")
        return None, None
    finally:
        if own_sessions:
            try:
                cleanup_onnx_session(embedding_sess, "embedding")
                cleanup_onnx_session(prediction_sess, "prediction")
                cleanup_cuda_memory(force=True)
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")


class AudioNotDecodableError(RuntimeError):
    pass


def _analyze_track(file_path, mood_labels_list, model_paths, onnx_sessions=None,
                   return_audio=False, raise_on_unreadable=False,
                   native_audio=None, native_sr=None):
    name = os.path.basename(file_path)
    logger.info(f"Starting analysis for: {name}")
    nothing = (None, None, None, None) if return_audio else (None, None)

    if native_audio is not None and native_sr is not None:
        audio, sr = resample_audio(native_audio, native_sr, 16000), 16000
    else:
        audio, sr = robust_load_audio_with_fallback(file_path, target_sr=16000)
    if audio is None or not np.any(audio) or audio.size == 0:
        logger.warning(
            f"Could not load a valid audio signal for {name} after all attempts. Skipping track."
        )
        if raise_on_unreadable:
            raise AudioNotDecodableError(f"no decodable audio for {name}")
        return nothing

    tempo, average_energy, musical_key, scale = extract_basic_features(audio, sr)

    final_patches = _patches_for_track(audio, sr, name)
    if final_patches is None:
        return nothing

    embedding, moods = _run_musicnn_models(
        final_patches, mood_labels_list, model_paths, onnx_sessions, name
    )
    if embedding is None:
        return nothing

    analysis_result = {
        "tempo": tempo,
        "key": musical_key,
        "scale": scale,
        "moods": moods,
        "energy": average_energy,
        "duration_seconds": float(audio.size) / float(sr) if sr else None,
    }
    return_values = (
        (analysis_result, embedding, audio, sr)
        if return_audio
        else (analysis_result, embedding)
    )
    gc.collect()
    comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
    return return_values


def analyze_track(file_path, mood_labels_list, model_paths, onnx_sessions=None,
                  return_audio=False, native_audio=None, native_sr=None):
    return _analyze_track(
        file_path, mood_labels_list, model_paths, onnx_sessions=onnx_sessions,
        return_audio=return_audio, native_audio=native_audio, native_sr=native_sr,
    )


def analyze_track_for_album(file_path, mood_labels_list, model_paths,
                            onnx_sessions=None, return_audio=False,
                            native_audio=None, native_sr=None):
    return _analyze_track(
        file_path, mood_labels_list, model_paths, onnx_sessions=onnx_sessions,
        return_audio=return_audio, raise_on_unreadable=True,
        native_audio=native_audio, native_sr=native_sr,
    )


def catalog_item_id(item):
    return sanitize_string_for_db(
        str(item.get('_catalog_item_id') or item.get('Id') or item.get('id'))
    )


def provider_item_id(item):
    return sanitize_string_for_db(str(item.get('Id') or item.get('id')))


def ensure_musicnn_sessions(onnx_sessions, model_paths, session_recycler, album_name):
    if onnx_sessions is None:
        logger.info(f"Lazy-loading MusiCNN models for album: {album_name}")
        return load_musicnn_sessions(model_paths)
    if not session_recycler.should_recycle():
        return onnx_sessions
    logger.info(
        f"Recycling ONNX sessions after {session_recycler.get_use_count()} tracks"
    )
    cleanup_musicnn_sessions(onnx_sessions, context="recycle")
    comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
    session_recycler.mark_recycled()
    return load_musicnn_sessions(model_paths)


def run_clap_for_track(path, track_name_full, native_audio=None, native_sr=None):
    logger.info(f"  - Starting CLAP analysis for {track_name_full}...")
    try:
        from ..clap_analyzer import analyze_audio_file

        emb, _, _ = analyze_audio_file(path, native_audio=native_audio, native_sr=native_sr)
        if PER_SONG_MODEL_RELOAD:
            try:
                from ..clap_analyzer import unload_clap_audio_only

                unload_clap_audio_only()
            except Exception as e:
                logger.debug(f"  - CLAP audio unload skipped: {e}")
        return emb
    except OperationalError:
        raise
    except Exception as e:
        error_manager.record(
            error_manager.classify(e, ERR_MODEL_INFERENCE),
            f"CLAP analysis failed for {track_name_full}: {e}",
            exc=e, logger=logger, level=logging.WARNING,
        )
        return None


def compute_other_features_str(clap_embedding, label_embeddings, labels):
    zero = zero_other_features(labels)
    if label_embeddings is None or clap_embedding is None:
        return zero
    try:
        from ..clap_analyzer import compute_other_features_from_clap

        d = compute_other_features_from_clap(clap_embedding, label_embeddings)
        return ",".join(f"{k}:{d.get(k, 0.0):.2f}" for k in labels)
    except Exception as e:
        logger.warning(f"  - Failed to compute other_features from CLAP: {e}")
        return zero


def persist_musicnn_results(item, analysis, top_moods, embedding, other_features_str):
    save_track_analysis_and_embedding(
        catalog_item_id(item),
        item['Name'],
        item.get('AlbumArtist', 'Unknown'),
        analysis['tempo'],
        analysis['key'],
        analysis['scale'],
        top_moods,
        embedding,
        energy=analysis['energy'],
        other_features=other_features_str,
        album=item.get('Album') or item.get('album'),
        album_artist=item.get('OriginalAlbumArtist')
        or item.get('originalAlbumArtist')
        or item.get('album_artist'),
        year=item.get('Year'),
        rating=item.get('Rating'),
        duration=analysis.get('duration_seconds'),
    )


def zero_other_features(labels):
    return ",".join(f"{label}:0.00" for label in labels)


ZERO_OTHER_FEATURES = zero_other_features(OTHER_FEATURE_LABELS)


def refresh_other_features(item_id, other_features_str):
    if not other_features_str:
        return False
    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE score SET other_features = %s WHERE item_id = %s",
                (other_features_str, str(item_id)),
            )
            updated = cur.rowcount
            conn.commit()
        return bool(updated)
    except OperationalError:
        raise
    except Exception as e:
        error_manager.record(
            ERR_DB_QUERY, f"Could not refresh other_features for {item_id}: {e}",
            exc=e, logger=logger, level=logging.WARNING,
        )
        return False


def persist_clap_embedding(item_id, embedding):
    if embedding is None:
        return False
    try:
        save_clap_embedding(item_id, embedding)
        logger.info("  - CLAP embedding saved (512-dim)")
        return True
    except OperationalError:
        raise
    except Exception as e:
        logger.warning(f"  - Failed to save CLAP embedding: {e}")
        return False


def _make_lyrics_audio_loader(robust_load_fn, download_fn):
    def audio_loader():
        p = download_fn() if download_fn is not None else None
        if not p:
            raise RuntimeError("Failed to download audio for lyrics ASR")
        a, s = robust_load_fn(str(p), target_sr=16000)
        if a is None or a.size == 0 or s is None:
            raise RuntimeError("Failed to load audio for lyrics ASR")
        return a, s, str(p)

    return audio_loader


def _prepare_lyrics_audio(path, track_audio, track_sr, robust_load_fn, download_fn):
    if track_audio is not None and track_sr is not None:
        return track_audio, track_sr, None
    if path is not None:
        logger.info("  - Loading audio from file for lyrics analysis")
        track_audio, track_sr = robust_load_fn(str(path), target_sr=16000)
        if track_audio is None or track_audio.size == 0 or track_sr is None:
            raise RuntimeError("Failed to load audio for lyrics analysis")
        return track_audio, track_sr, None
    return track_audio, track_sr, _make_lyrics_audio_loader(robust_load_fn, download_fn)


def run_lyrics_for_track(
    item,
    path,
    track_audio,
    track_sr,
    track_name_full,
    robust_load_fn,
    top_moods=None,
    download_fn=None,
):
    logger.info(f"  - Starting lyrics analysis for {track_name_full}...")
    try:
        from lyrics.lyrics_transcriber import analyze_lyrics

        track_audio, track_sr, audio_loader = _prepare_lyrics_audio(
            path, track_audio, track_sr, robust_load_fn, download_fn
        )

        result = analyze_lyrics(
            audio=track_audio,
            sr=track_sr,
            source_path=str(path) if path is not None else None,
            artist=item.get('AlbumArtist') or item.get('Artist'),
            track=item.get('Name'),
            track_id=str(item.get('Id') or item.get('id') or catalog_item_id(item)),
            top_moods=top_moods,
            audio_loader=audio_loader,
        )
        emb = result.get('embedding')
        if emb is None or getattr(emb, 'size', 0) == 0:
            logger.warning(f"  - Lyrics analysis produced no embedding for {track_name_full}")
            return False
        save_lyrics_embedding(catalog_item_id(item), emb, result.get('axis_vector'))
        logger.info("  - Lyrics embedding saved")
        return True
    except OperationalError:
        raise
    except Exception as e:
        error_manager.record(
            error_manager.classify(e, ERR_LYRICS_TRANSCRIPTION),
            str(e),
            exc=e,
            logger=logger,
            level=logging.WARNING,
        )
        return False
