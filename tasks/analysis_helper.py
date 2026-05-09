# tasks/analysis_helper.py
"""Reusable building blocks extracted from tasks.analysis."""

import gc
import importlib
import logging

import numpy as np
import librosa
import onnxruntime as ort

from .memory_utils import cleanup_onnx_session, comprehensive_memory_cleanup
from .onnx_providers import select_providers

# `app_helper` and `app_helper_artist` are safe at module top: they have no
# import cycle back into this module. Optional ML modules
# (.clap_analyzer / .mulan_analyzer / lyrics.lyrics_transcriber) stay inline
# inside the per-feature helpers so workers without those models can still
# import this module.
from app_helper import (
    get_db,
    get_clap_embedding,
    save_track_analysis_and_embedding,
    save_clap_embedding,
    save_lyrics_embedding,
    save_mulan_embedding,
)
from app_helper_artist import upsert_artist_mapping
from psycopg2 import sql as pgsql

logger = logging.getLogger(__name__)


# --- ONNX -------------------------------------------------------------------

DEFINED_TENSOR_NAMES = {
    'embedding': {'input': 'model/Placeholder:0', 'output': 'model/dense/BiasAdd:0'},
    'prediction': {'input': 'serving_default_model_Placeholder:0', 'output': 'PartitionedCall:0'},
}


def _find_onnx_name(candidate, names):
    """Match a TF-style tensor name to one of the ONNX tensor names."""
    if not names:
        return None
    stripped = candidate.split(':')[0]
    for cand in (candidate, stripped, stripped.split('/')[-1], stripped.replace('/', '_')):
        if cand in names:
            return cand
    return names[0]


def run_inference(session, feed_dict, output_tensor_name=None):
    """Run inference on an ONNX Runtime session, mapping TF-style names if needed."""
    input_names = [i.name for i in session.get_inputs()]
    mapped = {}
    for k, v in feed_dict.items():
        name = _find_onnx_name(k, input_names)
        if name is None:
            logger.error(f"Could not map input '{k}' to ONNX inputs {input_names}")
            return None
        mapped[name] = v
    output_names = [o.name for o in session.get_outputs()]
    out = _find_onnx_name(output_tensor_name, output_names) if output_tensor_name else (output_names[0] if output_names else None)
    if out is None:
        logger.error("No ONNX output name available to run inference.")
        return None
    result = session.run([out], mapped)
    return result[0] if isinstance(result, list) and len(result) > 0 else result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_provider_options():
    """Return [(provider_name, options), ...] preferring ROCm > CUDA > Vulkan > CPU."""
    return select_providers("musicnn")


def create_onnx_session(model_path, provider_options=None, label=""):
    """Create an InferenceSession; falls back to CPU if the preferred providers fail."""
    opts = provider_options or get_provider_options()
    try:
        return ort.InferenceSession(
            model_path,
            providers=[p[0] for p in opts],
            provider_options=[p[1] for p in opts],
        )
    except Exception:
        logger.warning(f"Failed to load {label or model_path} with GPU - falling back to CPU")
        return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


def load_musicnn_sessions(model_paths):
    """Build a {name: InferenceSession} dict for the MusiCNN models, or None on failure."""
    opts = get_provider_options()
    try:
        sessions = {n: create_onnx_session(p, opts, label=n) for n, p in model_paths.items()}
        logger.info(f"✓ Loaded {len(sessions)} MusiCNN models for album reuse")
        return sessions
    except Exception as e:
        logger.error(f"Failed to load MusiCNN models: {e}")
        return None


def cleanup_musicnn_sessions(onnx_sessions, context=""):
    """Close every MusiCNN session and run gc."""
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


# (loader, is_loaded, unloader, label) for each optional model.
_OPTIONAL_MODELS = (
    ('clap', '.clap_analyzer', 'is_clap_model_loaded', 'unload_clap_model'),
    ('mulan', '.mulan_analyzer', 'is_mulan_model_loaded', 'unload_mulan_model'),
)


def cleanup_optional_models(context=""):
    """Unload CLAP / MuLan models if currently loaded."""
    suffix = f" ({context})" if context else ""
    for label, mod, is_loaded_fn, unload_fn in _OPTIONAL_MODELS:
        try:
            module = importlib.import_module(mod, package=__package__)
            if getattr(module, is_loaded_fn)():
                logger.info(f"Cleaning up {label.upper()} model{suffix}")
                getattr(module, unload_fn)()
        except Exception as e:
            logger.warning(f"Error cleaning up {label.upper()} model: {e}")


def run_inference_with_oom_fallback(session, feed_dict, output_tensor_name,
                                    model_path, label, owns_session, file_basename):
    """Run inference; on GPU OOM, recreate the session on CPU and retry. Returns (result, session)."""
    try:
        return run_inference(session, feed_dict, output_tensor_name), session
    except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
        if "Failed to allocate memory" not in str(e):
            raise
        logger.warning(f"GPU OOM for {file_basename} during {label} inference - falling back to CPU")
        if owns_session:
            cleanup_onnx_session(session, label)
        comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
        cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        result = run_inference(cpu_session, feed_dict, output_tensor_name)
        if result is None:
            raise RuntimeError(f"CPU fallback inference returned None for {label} ({file_basename})")
        logger.info(f"Successfully completed {label} inference on CPU after OOM")
        return result, cpu_session


# --- Audio features ---------------------------------------------------------

_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_MAJOR = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
_MINOR = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])


def extract_basic_features(audio, sr):
    """Return (tempo, energy, key, scale)."""
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
    """Build the (N, 187, 96) float32 patch tensor MusiCNN expects, or None if too short."""
    n_mels, hop, n_fft, frame = 96, 256, 512, 187
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        window='hann', center=False, power=2.0, norm='slaney', htk=False,
    )
    log_mel = np.log10(1 + 10000 * mel)
    patches = [log_mel[:, i:i + frame] for i in range(0, log_mel.shape[1] - frame + 1, frame)]
    if not patches:
        return None
    return np.array(patches).transpose(0, 2, 1).astype(np.float32)


# --- DB helpers -------------------------------------------------------------

def _str_ids(ids):
    return [str(i) for i in ids]


def get_existing_track_ids(track_ids):
    """Return the subset of track_ids already fully analyzed by MusiCNN."""
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


def get_missing_ids_in_table(table_name, track_ids):
    """Return the subset of track_ids (as strings) with no row in `table_name`."""
    if not track_ids:
        return set()
    ids = _str_ids(track_ids)
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            pgsql.SQL("SELECT item_id FROM {} WHERE item_id IN %s").format(pgsql.Identifier(table_name)),
            (tuple(ids),),
        )
        existing = {row[0] for row in cur.fetchall()}
    return set(ids) - existing


_REFRESH_FIELDS = ('album', 'album_artist', 'year', 'rating', 'file_path')


def refresh_track_metadata(item, album_name):
    """COALESCE-update score metadata; never overwrite a non-null value with NULL."""
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
    query = pgsql.SQL("UPDATE score SET {} WHERE item_id = %s AND ({})")\
        .format(set_parts, where_parts)
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
    """Bulk-store artist_name → artist_id for a list of tracks. Errors are logged, never raised."""
    for t in tracks:
        name, aid = t.get('AlbumArtist'), t.get('ArtistId')
        if name and aid:
            try:
                upsert_artist_mapping(name, aid)
            except Exception as e:
                logger.error(f"Failed to upsert artist mapping for '{name}': {e}")
        elif name:
            scope = f" in album '{album_name}'" if album_name else ""
            logger.warning(f"✗ No artist_id for '{name}'{scope}")


# --- Per-track decision / status --------------------------------------------

def decide_track_needs(track_id, existing, missing_clap, missing_mulan, missing_lyrics, lyrics_enabled):
    """Return (needs_musicnn, needs_clap, needs_mulan, needs_lyrics) for a single track."""
    return (
        track_id not in existing,
        track_id in missing_clap,
        track_id in missing_mulan,
        lyrics_enabled and track_id in missing_lyrics,
    )


def compute_album_needs(tracks, clap_available, mulan_enabled, lyrics_enabled):
    """Return (existing_count, needs_clap, needs_mulan, needs_lyrics) for an album."""
    ids = [str(t['Id']) for t in tracks]
    existing = len(get_existing_track_ids(ids))
    needs_in = lambda flag, table: flag and bool(get_missing_ids_in_table(table, ids))
    return (
        existing,
        needs_in(clap_available, 'clap_embedding'),
        needs_in(mulan_enabled, 'mulan_embedding'),
        needs_in(lyrics_enabled, 'lyrics_embedding'),
    )


def build_feature_status_parts(clap_available, mulan_enabled, lyrics_enabled, include_check_marks=False):
    """Build the list of enabled-feature labels for skip log messages.

    The two callers historically used different orderings: per-track skip logs
    list Lyrics before MuLan; album skip logs list MuLan before Lyrics.
    """
    parts = ["MusiCNN"]
    if clap_available:
        parts.append("CLAP")
    if include_check_marks:
        if lyrics_enabled:
            parts.append("Lyrics")
        if mulan_enabled:
            parts.append("MuLan")
        return [f"{p}: ✓" for p in parts]
    if mulan_enabled:
        parts.append("MuLan")
    if lyrics_enabled:
        parts.append("Lyrics")
    return parts


# --- CLAP / MuLan / Lyrics per-track sub-tasks ------------------------------

def run_clap_for_track(path, track_name_full, needs_clap, clap_available, per_song_reload):
    """Run CLAP audio analysis; returns the embedding or None."""
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
    """Return a 'label:0.42,label2:0.55' string from CLAP. Falls back to all-zero."""
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
    """Save MusiCNN analysis + embedding via app_helper."""
    save_track_analysis_and_embedding(
        item['Id'], item['Name'], item.get('AlbumArtist', 'Unknown'),
        analysis['tempo'], analysis['key'], analysis['scale'], top_moods, embedding,
        energy=analysis['energy'],
        other_features=other_features_str,
        album=item.get('Album') or item.get('album'),
        album_artist=item.get('OriginalAlbumArtist') or item.get('originalAlbumArtist') or item.get('album_artist'),
        year=item.get('Year'),
        rating=item.get('Rating'),
        file_path=item.get('FilePath'),
    )


def persist_clap_embedding(item_id, embedding, needs_clap):
    """Save CLAP embedding (after the score row exists). Returns True on success."""
    if embedding is None or not needs_clap:
        return False
    try:
        save_clap_embedding(item_id, embedding)
        logger.info("  - CLAP embedding saved (512-dim)")
        return True
    except Exception as e:
        logger.warning(f"  - Failed to save CLAP embedding: {e}")
        return False


def run_lyrics_for_track(item, path, track_audio, track_sr, track_name_full,
                         needs_lyrics, lyrics_enabled, robust_load_fn):
    """Run lyrics analysis and persist embeddings. Returns True on save."""
    if not (needs_lyrics and lyrics_enabled):
        if lyrics_enabled:
            logger.info("  - Lyrics analysis already exists or skipped")
        return False
    logger.info(f"  - Starting lyrics analysis for {track_name_full}...")
    try:
        from lyrics.lyrics_transcriber import analyze_lyrics
        if track_audio is None or track_sr is None:
            logger.info("  - Loading audio from file for lyrics analysis")
            track_audio, track_sr = robust_load_fn(str(path), target_sr=16000)
            if track_audio is None or track_audio.size == 0 or track_sr is None:
                raise RuntimeError("Failed to load audio for lyrics analysis")
        result = analyze_lyrics(
            audio=track_audio, sr=track_sr, source_path=str(path),
            artist=item.get('AlbumArtist') or item.get('Artist'),
            track=item.get('Name'), track_id=item.get('Id') or item.get('id'),
        )
        emb = result.get('embedding')
        if emb is None or getattr(emb, 'size', 0) == 0:
            logger.warning(f"  - Lyrics analysis produced no embedding for {track_name_full}")
            return False
        save_lyrics_embedding(item['Id'], emb, result.get('axis_vector'))
        logger.info("  - Lyrics embedding saved")
        return True
    except Exception as e:
        logger.warning(f"  - Lyrics analysis failed: {e}", exc_info=True)
        return False


def run_mulan_for_track(path, item, track_name_full, needs_mulan, mulan_enabled):
    """Run MuLan analysis and persist embedding. Returns True on save."""
    if not (needs_mulan and mulan_enabled):
        if mulan_enabled and not needs_mulan:
            logger.info("  - MuLan embedding already exists, skipping")
        return False
    logger.info(f"  - Starting MuLan analysis for {track_name_full}...")
    try:
        from .mulan_analyzer import analyze_audio_file
        emb, duration, _ = analyze_audio_file(path)
        if emb is None:
            return False
        save_mulan_embedding(item['Id'], emb)
        logger.info(f"  - MuLan embedding saved (512-dim, duration: {duration:.1f}s)")
        return True
    except Exception as e:
        logger.warning(f"  - MuLan analysis failed: {e}")
        return False
