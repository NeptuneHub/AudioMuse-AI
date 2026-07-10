# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Analysis helpers: ONNX inference, feature extraction and per-track persistence.

Support code factored out of tasks.analysis so the album loop stays readable.
Owns ONNX Runtime session creation and provider selection (CPU/CUDA/CoreML), the
MusiCNN inference path, spectrogram/feature extraction, and the "what does this
track still need" decisions plus the DB upserts that store each result.

Main Features:
* create_onnx_session / load_musicnn_sessions / run_inference_with_oom_fallback:
  build sessions, resolve execution providers, and retry inference on OOM.
* compute_album_needs / decide_track_needs: per-track dedup deciding which of
  musicnn, CLAP and lyrics embeddings are missing (the real analysis dedup).
* persist_* helpers upsert mood tags, embeddings, CLAP and lyrics vectors.
"""

import gc
import importlib
import logging

import numpy as np
import librosa
import onnxruntime as ort

from .memory_utils import cleanup_onnx_session, comprehensive_memory_cleanup

from database import (
    get_db,
    get_clap_embedding,
    save_track_analysis_and_embedding,
    save_clap_embedding,
    save_lyrics_embedding,
)
from app_helper_artist import upsert_artist_mappings
from psycopg2 import sql as pgsql

from error import error_manager
from error.error_dictionary import ERR_LYRICS_TRANSCRIPTION

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


def resolve_providers(allow_coreml=False, role=None, cuda_options=None):
    available = ort.get_available_providers()
    chain = []
    accel_ok = role != 'flask'

    if accel_ok and 'CUDAExecutionProvider' in available:
        chain.append(
            (
                'CUDAExecutionProvider',
                cuda_options
                or {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                },
            )
        )

    if accel_ok and allow_coreml and 'CoreMLExecutionProvider' in available:
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
        if name and name in available and name not in [p[0] for p in chain]:
            chain.append((name, provider.get('options') or {}))

    chain.append(('CPUExecutionProvider', {}))
    logger.info("ONNX provider chain: %s", [p[0] for p in chain])
    return chain


def _plugin_onnx_providers():
    try:
        from plugin.manager import plugin_manager
        return plugin_manager.get_onnx_providers()
    except Exception:
        return []


def run_song_analyzed_hook(item, audio_path, musicnn_analysis, musicnn_embedding,
                           clap_embedding, top_moods, album_id, album_name, run_id):
    """Fire plugin on_song_analyzed hooks for a finished song; guarded no-op when no plugin listens.

    Fully wrapped so it can never raise into the analysis loop, and it builds the
    payload only when a worker plugin actually registered a listener. ``run_id`` is
    the analysis run's task id, shared by every song of one run, so a listener can
    count or group per run.
    """
    try:
        from plugin.manager import plugin_manager
        if not plugin_manager.enabled() or not plugin_manager.song_analyzed_hooks():
            return
        payload = {
            'item_id': str(item.get('Id')),
            'run_id': run_id,
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


def get_provider_options(allow_coreml=False, role=None):
    return resolve_providers(allow_coreml=allow_coreml, role=role)


def _default_sess_options():
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return opts


def create_onnx_session(
    model_path, provider_options=None, label="", sess_options=None, allow_coreml=False
):
    opts = provider_options or resolve_providers(allow_coreml=allow_coreml)
    if sess_options is None:
        sess_options = _default_sess_options()
    extra = {'sess_options': sess_options}
    try:
        return ort.InferenceSession(
            model_path,
            providers=[p[0] for p in opts],
            provider_options=[p[1] for p in opts],
            **extra,
        )
    except Exception:
        logger.warning(f"Failed to load {label or model_path} with GPU - falling back to CPU")
        return ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            **extra,
        )


def load_musicnn_sessions(model_paths):
    opts = resolve_providers(allow_coreml=False)
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
    for name, session in onnx_sessions.items():
        try:
            cleanup_onnx_session(session, name)
        except Exception as e:
            logger.warning(f"Error cleaning up {name} session: {e}")
    gc.collect()


_OPTIONAL_MODELS = (
    ('clap', '.clap_analyzer', 'is_clap_model_loaded', 'unload_clap_model'),
    ('lyrics', 'lyrics', 'is_lyrics_loaded', 'unload_lyrics_models'),
)


def cleanup_optional_models(context=""):
    suffix = f" ({context})" if context else ""
    for label, mod, is_loaded_fn, unload_fn in _OPTIONAL_MODELS:
        try:
            module = importlib.import_module(mod, package=__package__)
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
        cpu_session = None
        try:
            try:
                cleanup_onnx_session(session, label)
            except Exception:
                logger.exception("Error cleaning up OOM'd %s session before CPU fallback", label)
            session = None
            try:
                comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
            except Exception:
                logger.exception("Error during memory cleanup before %s CPU fallback", label)

            cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            result = run_inference(cpu_session, feed_dict, output_tensor_name)
            if result is None:
                raise RuntimeError(
                    f"CPU fallback inference returned None for {label} ({file_basename})"
                )
            logger.info(f"Successfully completed {label} inference on CPU after OOM")
            return result, cpu_session
        finally:
            session = None


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


def _str_ids(ids):
    return [str(i) for i in ids]


def get_existing_track_ids(track_ids):
    if not track_ids:
        return set()
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT s.item_id FROM score s JOIN embedding e ON s.item_id = e.item_id "
            "WHERE s.item_id IN %s AND s.other_features IS NOT NULL "
            "AND s.energy IS NOT NULL AND s.mood_vector IS NOT NULL "
            "AND s.tempo IS NOT NULL",
            (tuple(_str_ids(track_ids)),),
        )
        return {row[0] for row in cur.fetchall()}


def fetch_existing_top_moods(track_ids, top_n_moods):
    if not track_ids or not top_n_moods or top_n_moods <= 0:
        return {}
    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT item_id, mood_vector FROM score "
                "WHERE item_id IN %s AND mood_vector IS NOT NULL AND mood_vector <> ''",
                (tuple(_str_ids(track_ids)),),
            )
            rows = cur.fetchall()
    except Exception as exc:
        logger.warning(f"Failed to fetch prior moods from score table: {exc}")
        return {}

    result = {}
    for item_id, mv in rows:
        pairs = []
        for part in mv.split(','):
            k, _, v = part.partition(':')
            k = k.strip()
            if not k:
                continue
            try:
                pairs.append((k, float(v)))
            except ValueError:
                continue
        if pairs:
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            result[str(item_id)] = dict(pairs[:top_n_moods])
    return result


def get_missing_ids_in_table(table_name, track_ids):
    if not track_ids:
        return set()
    ids = _str_ids(track_ids)
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            pgsql.SQL("SELECT item_id FROM {} WHERE item_id IN %s").format(
                pgsql.Identifier(table_name)
            ),
            (tuple(ids),),
        )
        existing = {row[0] for row in cur.fetchall()}
    return set(ids) - existing


_REFRESH_FIELDS = ('album', 'album_artist', 'year', 'rating', 'file_path')


def refresh_track_metadata(item, album_name):
    values = (
        album_name,
        item.get('OriginalAlbumArtist'),
        item.get('Year'),
        item.get('Rating'),
        item.get('FilePath'),
    )
    if not any(v is not None for v in values):
        return False
    set_parts = pgsql.SQL(", ").join(
        pgsql.SQL("{} = COALESCE(%s, {})").format(pgsql.Identifier(f), pgsql.Identifier(f))
        for f in _REFRESH_FIELDS
    )
    where_parts = pgsql.SQL(" OR ").join(
        pgsql.SQL("(%s IS NOT NULL AND {} IS DISTINCT FROM %s)").format(pgsql.Identifier(f))
        for f in _REFRESH_FIELDS
    )
    query = pgsql.SQL("UPDATE score SET {} WHERE item_id = %s AND ({})").format(
        set_parts, where_parts
    )
    params = (*values, str(item['Id']), *(p for v in values for p in (v, v)))
    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            changed = cur.rowcount
            conn.commit()
        return bool(changed)
    except Exception as e:
        logger.warning(f"[refresh_track_metadata] Failed to update '{item.get('Name')}': {e}")
        return False


def upsert_artist_mappings_for_tracks(tracks, album_name=None):
    last_id_by_name = {}
    for t in tracks:
        name, aid = t.get('AlbumArtist'), t.get('ArtistId')
        if name and aid:
            last_id_by_name[name] = aid
        elif name:
            last_id_by_name.setdefault(name, None)
    upsert_artist_mappings((n, a) for n, a in last_id_by_name.items() if a)
    for name, aid in last_id_by_name.items():
        if not aid:
            scope = f" in album '{album_name}'" if album_name else ""
            logger.warning(f"No artist_id for '{name}'{scope}")


def decide_track_needs(track_id, existing, missing_clap, missing_lyrics, lyrics_enabled):
    return (
        track_id not in existing,
        track_id in missing_clap,
        lyrics_enabled and track_id in missing_lyrics,
    )


def compute_album_needs(tracks, clap_available, lyrics_enabled):
    ids = [str(t['Id']) for t in tracks]
    existing = len(get_existing_track_ids(ids))

    def needs_in(flag, table):
        return flag and bool(get_missing_ids_in_table(table, ids))

    return (
        existing,
        needs_in(clap_available, 'clap_embedding'),
        needs_in(lyrics_enabled, 'lyrics_embedding'),
    )


def build_feature_status_parts(clap_available, lyrics_enabled, include_check_marks=False):
    parts = ["MusiCNN"]
    if clap_available:
        parts.append("CLAP")
    if lyrics_enabled:
        parts.append("Lyrics")
    if include_check_marks:
        return [f"{p}: OK" for p in parts]
    return parts


def run_clap_for_track(path, track_name_full, needs_clap, clap_available, per_song_reload):
    if not (needs_clap and clap_available):
        return None
    logger.info(f"  - Starting CLAP analysis for {track_name_full}...")
    try:
        from .clap_analyzer import analyze_audio_file

        emb, _, _ = analyze_audio_file(path)
        if per_song_reload:
            try:
                from .clap_analyzer import unload_clap_audio_only

                unload_clap_audio_only()
            except Exception as e:
                logger.debug(f"  - CLAP audio unload skipped: {e}")
        return emb
    except Exception as e:
        logger.warning(f"  - CLAP analysis failed: {e}")
        return None


def compute_other_features_str(clap_embedding, needs_clap, label_embeddings, item_id, labels):
    zero = ",".join(f"{k}:0.00" for k in labels)
    if label_embeddings is None:
        return zero
    try:
        from .clap_analyzer import compute_other_features_from_clap

        emb = clap_embedding
        if emb is None and not needs_clap:
            emb = get_clap_embedding(item_id)
        if emb is None:
            return zero
        d = compute_other_features_from_clap(emb, label_embeddings)
        return ",".join(f"{k}:{d.get(k, 0.0):.2f}" for k in labels)
    except Exception as e:
        logger.warning(f"  - Failed to compute other_features from CLAP: {e}")
        return zero


def persist_musicnn_results(item, analysis, top_moods, embedding, other_features_str):
    save_track_analysis_and_embedding(
        item['Id'],
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
        file_path=item.get('FilePath'),
    )


def persist_clap_embedding(item_id, embedding, needs_clap):
    if embedding is None or not needs_clap:
        return False
    try:
        save_clap_embedding(item_id, embedding)
        logger.info("  - CLAP embedding saved (512-dim)")
        return True
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
    needs_lyrics,
    lyrics_enabled,
    robust_load_fn,
    top_moods=None,
    download_fn=None,
):
    if not (needs_lyrics and lyrics_enabled):
        if lyrics_enabled:
            logger.info("  - Lyrics analysis already exists or skipped")
        return False
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
            track_id=item.get('Id') or item.get('id'),
            top_moods=top_moods,
            audio_loader=audio_loader,
        )
        emb = result.get('embedding')
        if emb is None or getattr(emb, 'size', 0) == 0:
            logger.warning(f"  - Lyrics analysis produced no embedding for {track_name_full}")
            return False
        save_lyrics_embedding(item['Id'], emb, result.get('axis_vector'))
        logger.info("  - Lyrics embedding saved")
        return True
    except Exception as e:
        error_manager.record(
            error_manager.classify(e, ERR_LYRICS_TRANSCRIPTION),
            str(e),
            exc=e,
            logger=logger,
            level=logging.WARNING,
        )
        return False
