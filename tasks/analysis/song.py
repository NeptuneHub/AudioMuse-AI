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
* musicnn_embedding_for_audio: the MusiCNN embedding of an in-memory 16 kHz
  clip through the same patches and sessions analysis uses; None when the clip
  is too short for one patch, RuntimeError when inference itself fails.
* analyze_track / robust_load_audio_with_fallback: decode a file and produce the
  MusiCNN moods + embedding; a track that yields no audio at all returns None,
  while one whose packets are only partly corrupt returns however much decoded
  cleanly.
* The PyAV fallback (tasks.pyav_decode) runs in a short-lived process, so a native
  FFmpeg crash ends that process, never the worker or Flask. A PyAV error, or a crash
  or hang repeated in a fresh process, comes back empty and the album marks the file
  not analyzable, unless a clean test clip fails too (a broken decoder); a kill or a
  temporary error raise AudioDecodeInterrupted: nothing marked, a later run retries.
* duration_seconds is MEASURED on the audio that actually decoded, never read from
  the container header, and two consequences follow. AUDIO_LOAD_TIMEOUT caps the
  decode at 600s, so a track longer than that already stores a truncated duration
  today. And a partly corrupt file stores a short duration, which makes the
  identity duration veto (DURATION_TOLERANCE_SECONDS, 1s) split it from an intact
  copy of the same song on another server. That split is DELIBERATE: a damaged
  file must not share a catalogue row with a good one, otherwise replacing it with
  a working copy would reuse the damaged embedding instead of re-analyzing it.
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
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import wave

import numpy as np
import librosa

import onnxruntime as ort  # noqa: F401

from config import (
    AUDIO_LOAD_TIMEOUT,
    AUDIO_MIN_DECODED_FRACTION,
    MUSICNN_BATCH_SIZE,
    OTHER_FEATURE_LABELS,
    PER_SONG_MODEL_RELOAD,
    TEMPO_MAX_BPM,
    TEMPO_MIN_BPM,
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

from ..pyav_decode import (
    CHILD_FLAG,
    STATUS_OK,
    STATUS_TRANSIENT,
    STATUS_UNREADABLE,
    decode_with_pyav,
    wall_seconds,
)

from ..memory_utils import cleanup_cuda_memory, cleanup_onnx_session, comprehensive_memory_cleanup

from ..onnx_utils import create_onnx_session, resolve_providers, run_inference_with_oom_fallback


logger = logging.getLogger(__name__)


DEFINED_TENSOR_NAMES = {
    'embedding': {'input': 'model/Placeholder:0', 'output': 'model/dense/BiasAdd:0'},
    'prediction': {'input': 'serving_default_model_Placeholder:0', 'output': 'PartitionedCall:0'},
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _estimate_tempo(audio, sr):
    if audio is None or audio.size == 0:
        return 0.0
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(np.ravel(tempo)[0])
    if tempo <= 0:
        return 0.0
    if TEMPO_MIN_BPM <= 0 or TEMPO_MAX_BPM < TEMPO_MIN_BPM:
        return tempo
    while tempo < TEMPO_MIN_BPM:
        tempo *= 2.0
    while tempo > TEMPO_MAX_BPM:
        tempo /= 2.0
    return tempo


def _estimate_energy(audio):
    if audio is None or audio.size == 0:
        return 0.0
    rms = librosa.feature.rms(y=audio)
    if rms is None or rms.size == 0:
        return 0.0
    rms_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)
    energy = np.clip((rms_db + 60.0) / 60.0, 0.0, 1.0)
    return float(np.mean(energy))


def _estimate_key_scale(audio, sr):
    if audio is None or audio.size == 0:
        return 'C', 'major'
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    if chroma is None or chroma.size == 0:
        return 'C', 'major'
    chroma_mean = np.mean(chroma, axis=1)
    if chroma_mean.sum() <= 0:
        return 'C', 'major'
    c = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    maj = np.array([np.dot(c, np.roll(_KS_MAJOR, i)) for i in range(12)])
    mnr = np.array([np.dot(c, np.roll(_KS_MINOR, i)) for i in range(12)])
    maj = (maj - maj.mean()) / (maj.std() + 1e-9)
    mnr = (mnr - mnr.mean()) / (mnr.std() + 1e-9)
    mi, ni = int(np.argmax(maj)), int(np.argmax(mnr))
    if maj[mi] > mnr[ni]:
        return _KEYS[mi], 'major'
    return _KEYS[ni], 'minor'


def extract_basic_features(audio, sr):
    tempo = _estimate_tempo(audio, sr)
    energy = _estimate_energy(audio)
    musical_key, scale = _estimate_key_scale(audio, sr)
    return tempo, energy, musical_key, scale


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
    return decode_with_pyav(file_path, target_sr, AUDIO_LOAD_TIMEOUT, AUDIO_MIN_DECODED_FRACTION)


class AudioDecodeInterrupted(RuntimeError):
    pass


_DECODE_ATTEMPTS = 2
_CHILD_LOG_TAIL_CHARS = 4000
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_POSIX_CRASH_SIGNALS = frozenset(
    getattr(signal, name) for name in ('SIGSEGV', 'SIGBUS', 'SIGABRT', 'SIGFPE', 'SIGTRAP')
    if hasattr(signal, name)
)
_WINDOWS_CRASH_CODES = frozenset({
    3, 0x80000003, 0xC0000005, 0xC0000094, 0xC00000FD, 0xC0000374, 0xC0000409,
})
_CHILD_LOG_LEVELS = (
    ('[CRITICAL]', logging.ERROR), ('[ERROR]', logging.ERROR),
    ('[WARNING]', logging.WARNING), ('[INFO]', logging.INFO),
)
_decoder_proven = False


def _decode_command(file_path, target_sr):
    args = [
        os.path.abspath(file_path),
        str(int(target_sr or 0)),
        str(int(AUDIO_LOAD_TIMEOUT or 0)),
        repr(float(AUDIO_MIN_DECODED_FRACTION)),
    ]
    if getattr(sys, 'frozen', False):
        return [sys.executable, CHILD_FLAG, *args], None
    return [sys.executable, '-m', decode_with_pyav.__module__, *args], _PROJECT_ROOT


def _exit_label(returncode):
    if returncode < 0:
        try:
            return signal.Signals(-returncode).name
        except ValueError:
            return f"signal {-returncode}"
    return f"exit code {returncode} (0x{returncode & 0xFFFFFFFF:08X})"


def _relay_child_log(stderr, name):
    text = (stderr or b'').decode('utf-8', 'replace').strip()
    level = logging.WARNING
    for line in text[-_CHILD_LOG_TAIL_CHARS:].splitlines():
        level = next((lvl for tag, lvl in _CHILD_LOG_LEVELS if line.startswith(tag)), level)
        logger.log(level, "PyAV decode process for %s: %s", name, line)


def _decode_child_crashed(returncode):
    if sys.platform == 'win32':
        return (returncode & 0xFFFFFFFF) in _WINDOWS_CRASH_CODES
    return returncode < 0 and -returncode in _POSIX_CRASH_SIGNALS


def _decoded_reply(stdout, name):
    start = stdout.find(b'\n') + 1
    try:
        reply = json.loads(stdout[:start])
        status, sr, samples = reply['status'], reply['sr'], int(reply['samples'])
    except (ValueError, KeyError, TypeError) as exc:
        raise AudioDecodeInterrupted(
            f"The PyAV decode process for {name} sent an unreadable reply; the file is not "
            "marked and is retried on the next run."
        ) from exc
    if status == STATUS_UNREADABLE:
        return np.array([], dtype=np.float32), sr
    if status != STATUS_OK:
        state = 'a temporary error' if status == STATUS_TRANSIENT else f'status {status!r}'
        raise AudioDecodeInterrupted(
            f"The PyAV decode process reported {state} reading {name}; the file is not "
            "marked and is retried on the next run."
        )
    body = len(stdout) - start
    if body != samples * np.dtype(np.float32).itemsize:
        raise AudioDecodeInterrupted(
            f"The PyAV decode process for {name} sent {body} bytes for {samples} samples; "
            "the file is not marked and is retried on the next run."
        )
    return np.frombuffer(stdout, dtype=np.float32, count=samples, offset=start).copy(), sr


def _run_child(command, cwd):
    return subprocess.run(  # nosec B603
        command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=wall_seconds(AUDIO_LOAD_TIMEOUT),
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )


def _decoder_works(directory):
    global _decoder_proven
    if _decoder_proven:
        return True
    rate = 8000
    tone = (np.sin(np.linspace(0, 440 * 2 * np.pi, rate)) * 12000).astype('<i2')
    try:
        handle, probe = tempfile.mkstemp(suffix='.wav', dir=directory)
        os.close(handle)
    except OSError:
        logger.exception("Could not write the clean test clip for the PyAV decode process")
        return False
    try:
        with wave.open(probe, 'wb') as clip:
            clip.setnchannels(1)
            clip.setsampwidth(2)
            clip.setframerate(rate)
            clip.writeframes(tone.tobytes())
        child = _run_child(*_decode_command(probe, None))
        _decoder_proven = (
            child.returncode == 0 and _decoded_reply(child.stdout, 'probe.wav')[0].size > 0
        )
        if not _decoder_proven:
            logger.error(
                "The PyAV decode process failed on a clean test clip (%s).",
                _exit_label(child.returncode),
            )
    except (OSError, subprocess.SubprocessError, AudioDecodeInterrupted):
        logger.exception("The PyAV decode process could not decode a clean test clip")
    finally:
        os.remove(probe)
    return _decoder_proven


def _decode_audio_isolated(file_path, target_sr):
    name = os.path.basename(file_path)
    command, cwd = _decode_command(file_path, target_sr)
    for attempt in range(1, _DECODE_ATTEMPTS + 1):
        try:
            child = _run_child(command, cwd)
        except subprocess.TimeoutExpired as exc:
            _relay_child_log(exc.stderr, name)
            logger.error(
                "The PyAV decode process ran past %ss while reading %s (attempt %d of %d).",
                exc.timeout, name, attempt, _DECODE_ATTEMPTS,
            )
            continue
        except (OSError, subprocess.SubprocessError) as exc:
            _relay_child_log(getattr(exc, 'stderr', None), name)
            raise AudioDecodeInterrupted(
                f"The PyAV decode process for {name} could not finish ({exc}); the file is "
                "not marked and is retried on the next run."
            ) from exc
        _relay_child_log(child.stderr, name)
        if child.returncode == 0:
            return _decoded_reply(child.stdout, name)
        label = _exit_label(child.returncode)
        if not _decode_child_crashed(child.returncode):
            raise AudioDecodeInterrupted(
                f"The PyAV decode process for {name} ended with {label}, which is not a crash "
                "the file causes; the file is not marked and is retried on the next run."
            )
        logger.error(
            "The PyAV decode process crashed with %s while reading %s (attempt %d of %d).",
            label, name, attempt, _DECODE_ATTEMPTS,
        )
    if not _decoder_works(os.path.dirname(os.path.abspath(file_path))):
        raise AudioDecodeInterrupted(
            f"The PyAV decode process also fails on a clean test clip, so the decoder is "
            f"broken rather than {name}; the file is not marked and is retried on the next run."
        )
    logger.error(
        "The PyAV decode process crashed or hung on every attempt for %s; it is treated as "
        "not decodable.", name,
    )
    return np.array([], dtype=np.float32), None


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
        audio, sr = _decode_audio_isolated(file_path, target_sr)
        if audio is None or audio.size == 0 or not np.any(audio):
            logger.error(f"PyAV fallback resulted in empty/silent audio for {name}.")
            return None, None
        return audio, sr
    except AudioDecodeInterrupted:
        logger.exception(f"PyAV fallback for {name} was interrupted")
        raise
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
    except Exception as e:
        error_manager.record(
            error_manager.classify(e, ERR_MODEL_INFERENCE),
            f"MusiCNN inference failed for {name}",
            exc=e, logger=logger,
        )
        return None, None
    finally:
        if own_sessions:
            try:
                cleanup_onnx_session(embedding_sess, "embedding")
                cleanup_onnx_session(prediction_sess, "prediction")
                cleanup_cuda_memory(force=True)
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")


def musicnn_embedding_for_audio(audio, sr, mood_labels_list, model_paths, onnx_sessions=None,
                                name='clip'):
    final_patches = _patches_for_track(audio, sr, name)
    if final_patches is None:
        return None
    embedding, _moods = _run_musicnn_models(
        final_patches, mood_labels_list, model_paths, onnx_sessions, name
    )
    if embedding is None:
        raise RuntimeError(f"MusiCNN inference failed for {name}")
    return embedding


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


def refresh_base_features(item_id, tempo, energy, musical_key, scale):
    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE score SET tempo = %s, energy = %s, key = %s, scale = %s "
                "WHERE item_id = %s",
                (tempo, energy, musical_key, scale, str(item_id)),
            )
            updated = cur.rowcount
            conn.commit()
        return bool(updated)
    except OperationalError:
        raise
    except Exception as e:
        error_manager.record(
            ERR_DB_QUERY, f"Could not refresh base features for {item_id}: {e}",
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
            raise AudioNotDecodableError(f"no decodable audio for lyrics ASR of {p}")
        return a, s, str(p)

    return audio_loader


def _prepare_lyrics_audio(path, track_audio, track_sr, robust_load_fn, download_fn):
    if track_audio is not None and track_sr is not None:
        return track_audio, track_sr, None
    if path is not None:
        logger.info("  - Loading audio from file for lyrics analysis")
        track_audio, track_sr = robust_load_fn(str(path), target_sr=16000)
        if track_audio is None or track_audio.size == 0 or track_sr is None:
            raise AudioNotDecodableError(f"no decodable audio for lyrics analysis of {path}")
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
    except (OperationalError, AudioNotDecodableError):
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
