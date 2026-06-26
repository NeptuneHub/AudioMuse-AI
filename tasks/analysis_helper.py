# tasks/analysis_helper.py
"""Reusable building blocks extracted from tasks.analysis."""

import gc
import importlib
import logging

import numpy as np
import librosa
import onnxruntime as ort

from .memory_utils import cleanup_onnx_session, comprehensive_memory_cleanup

# `database` and `app_helper_artist` are safe at module top: they have no
# import cycle back into this module, and importing the DB primitives directly
# (rather than via the app_helper facade) keeps this helper decoupled from the
# blueprint layer. Optional ML modules (.clap_analyzer / lyrics.lyrics_transcriber)
# stay inline inside the per-feature helpers so workers without those models can
# still import this module.
from database import (
    get_db,
    get_clap_embedding,
    save_track_analysis_and_embedding,
    save_clap_embedding,
    save_lyrics_embedding,
)
from app_helper_artist import upsert_artist_mappings
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


def resolve_providers(allow_coreml=False, role=None, cuda_options=None):
    """Centralized ONNX provider selection.

    Returns an ordered ``[(provider_name, options), ...]`` chain following the
    priority NVIDIA CUDA → Apple CoreML (M1-M4) → CPU. Providers that are not
    available on the current machine are skipped, and CPU is always appended
    last as the universal fallback.

    Hardware accelerators are gated off when ``role == 'flask'`` because both
    CUDA and CoreML sessions are thread-affine and the Flask web process serves
    requests on short-lived per-request threads (see ``_load_text_model``).

    ``cuda_options`` overrides the default CUDA provider options for callers
    that need model-specific tuning (e.g. CLAP audio historically used
    ``cudnn_conv_algo_search='DEFAULT'`` rather than the ``EXHAUSTIVE`` default
    MusiCNN uses). CUDA-only — it has no effect on the macOS CoreML/CPU path.

    CoreML is only attempted when ``allow_coreml`` is True. It ships in the
    macOS ``onnxruntime`` wheel by default (no extra dependency), but it is not
    a safe blanket default: it requires the ``MLProgram`` format for the dynamic
    batch dimension our exports rely on, and unsupported ops (e.g. attention)
    get partitioned back to CPU. So we opt in per-model only where it helps.
    """
    available = ort.get_available_providers()
    chain = []
    accel_ok = role != 'flask'

    if accel_ok and 'CUDAExecutionProvider' in available:
        chain.append(('CUDAExecutionProvider', cuda_options or {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }))

    if accel_ok and allow_coreml and 'CoreMLExecutionProvider' in available:
        chain.append(('CoreMLExecutionProvider', {
            'MLComputeUnits': 'ALL',       # CPU + GPU + Apple Neural Engine
            'ModelFormat': 'MLProgram',    # required for our dynamic batch dim
        }))

    chain.append(('CPUExecutionProvider', {}))
    logger.info("ONNX provider chain: %s", [p[0] for p in chain])
    return chain


def get_provider_options(allow_coreml=False, role=None):
    """Backwards-compatible alias for :func:`resolve_providers`."""
    return resolve_providers(allow_coreml=allow_coreml, role=role)


def _default_sess_options():
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return opts


def create_onnx_session(model_path, provider_options=None, label="", sess_options=None, allow_coreml=False):
    """Create an InferenceSession; falls back to CPU if the preferred providers fail."""
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
    """Build a {name: InferenceSession} dict for the MusiCNN models, or None on failure."""
    # CoreML stays OFF here: MusiCNN takes a variable-length (N, 187, 96) patch
    # tensor and CoreML can compile but NOT execute that dynamic shape — it
    # raises "Unable to compute the prediction (error code: -1)" at run time,
    # which our load-time/OOM fallbacks don't catch, so every track gets skipped.
    opts = resolve_providers(allow_coreml=False)
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


# (label, module_path, is_loaded_fn, unload_fn) for each optional model.
# ``module_path`` is fed straight into ``importlib.import_module`` (relative
# paths use ``__package__`` = ``tasks``; absolute paths like ``'lyrics'``
# resolve from the project root).
_OPTIONAL_MODELS = (
    ('clap', '.clap_analyzer', 'is_clap_model_loaded', 'unload_clap_model'),
    ('lyrics', 'lyrics', 'is_lyrics_loaded', 'unload_lyrics_models'),
)


def cleanup_optional_models(context=""):
    """Unload every optional model currently held by this worker.

    Each entry is released inside its own try/except so a failure to
    release one (e.g. import error, partial state) cannot prevent the
    others from being freed. Called from ``analyze_album_task`` at album
    end *and* in its surrounding ``finally`` clause — both call sites
    expect this function to never raise.
    """
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
    """Run inference; on GPU OOM, recreate the session on CPU and retry.

    Returns ``(result, session)`` — the returned session is either the
    original (no fallback) or a fresh CPU session (fallback occurred).

    On OOM, the OOM'd GPU session's GPU buffers are freed BEFORE the CPU
    session is allocated — this is critical when the caller passes a
    shared session (``owns_session=False``), because creating the CPU
    session itself allocates memory and can re-OOM if we don't reclaim
    the GPU buffers first. The cleanup runs inside ``try/finally`` so
    the OOM'd session is dropped even when CPU-session creation raises.

    Note: dropping the local ``session`` reference here is necessary but
    not sufficient — the caller must also drop *its* references (e.g. a
    captured ``original_session`` local, or the shared session dict slot)
    before GC can actually reclaim the GPU memory. See
    ``tasks.analysis.analyze_track`` for the matching caller-side cleanup.
    """
    try:
        return run_inference(session, feed_dict, output_tensor_name), session
    except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
        if "Failed to allocate memory" not in str(e):
            raise
        logger.warning(f"GPU OOM for {file_basename} during {label} inference - falling back to CPU")
        cpu_session = None
        try:
            # ALWAYS drop our local reference to the OOM'd session and reset the
            # ONNX/CUDA memory pools before allocating the CPU session — even
            # when ``owns_session=False``. The previous behavior skipped this
            # cleanup for shared sessions, which (a) leaked the OOM'd GPU
            # buffers for the rest of the album and (b) made the CPU session
            # allocation more likely to re-OOM under memory pressure.
            try:
                cleanup_onnx_session(session, label)
            except Exception:
                logger.exception(
                    "Error cleaning up OOM'd %s session before CPU fallback", label)
            session = None  # break this frame's reference explicitly
            try:
                comprehensive_memory_cleanup(force_cuda=True, reset_onnx_pool=True)
            except Exception:
                logger.exception(
                    "Error during memory cleanup before %s CPU fallback", label)

            cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            result = run_inference(cpu_session, feed_dict, output_tensor_name)
            if result is None:
                raise RuntimeError(f"CPU fallback inference returned None for {label} ({file_basename})")
            logger.info(f"Successfully completed {label} inference on CPU after OOM")
            return result, cpu_session
        finally:
            # Belt-and-suspenders: if anything above raised between the
            # cleanup_onnx_session() call and the return, make sure our local
            # references are gone so the OOM'd buffers can be reclaimed when
            # the exception unwinds the frame.
            session = None


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
    log_mel = np.log10(1 + 10000 * np.maximum(mel, 0.0))
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


def fetch_existing_top_moods(track_ids, top_n_moods):
    """Return {track_id: {label: score}} top-N moods for already-analyzed tracks.

    Tight DB-side fetch of just ``item_id`` + ``mood_vector`` from ``score``.
    DB errors are logged and an empty dict is returned; malformed rows are
    silently skipped. Callers must treat a missing key as 'no prior available'
    and degrade to running the lyrics pipeline without the prior.
    """
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
    """Store distinct artist_name -> artist_id for a list of tracks. Errors are logged, never raised."""
    # One id per name (last wins, as a per-track upsert would leave it); write the
    # whole list in a single batch instead of one commit per track.
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


# --- Per-track decision / status --------------------------------------------

def decide_track_needs(track_id, existing, missing_clap, missing_lyrics, lyrics_enabled):
    """Return (needs_musicnn, needs_clap, needs_lyrics) for a single track."""
    return (
        track_id not in existing,
        track_id in missing_clap,
        lyrics_enabled and track_id in missing_lyrics,
    )


def compute_album_needs(tracks, clap_available, lyrics_enabled):
    """Return (existing_count, needs_clap, needs_lyrics) for an album."""
    ids = [str(t['Id']) for t in tracks]
    existing = len(get_existing_track_ids(ids))
    needs_in = lambda flag, table: flag and bool(get_missing_ids_in_table(table, ids))
    return (
        existing,
        needs_in(clap_available, 'clap_embedding'),
        needs_in(lyrics_enabled, 'lyrics_embedding'),
    )


def build_feature_status_parts(clap_available, lyrics_enabled, include_check_marks=False):
    """Build the list of enabled-feature labels for skip log messages."""
    parts = ["MusiCNN"]
    if clap_available:
        parts.append("CLAP")
    if lyrics_enabled:
        parts.append("Lyrics")
    if include_check_marks:
        return [f"{p}: ✓" for p in parts]
    return parts


# --- CLAP / Lyrics per-track sub-tasks --------------------------------------

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
                         needs_lyrics, lyrics_enabled, robust_load_fn,
                         top_moods=None, download_fn=None):
    """Run lyrics analysis and persist embeddings. Returns True on save.

    ``top_moods`` is the MusicNN top-N moods dict (label → score). When it
    includes 'instrumental', analyze_lyrics short-circuits the entire pipeline
    (skips Whisper-small ASR + gte embedding) and writes the instrumental
    sentinel directly. When it includes 'female vocalists' / 'male vocalists'
    the VAD pre-pass is bypassed so quiet/low-voiced singers are not dropped.

    Callers should pass the freshly computed top_moods on a full analysis
    pass, or the moods reloaded from the ``score`` table on a MusicNN-skipped
    pass (via fetch_existing_top_moods). Passing None is also valid and
    simply runs the lyrics pipeline without these priors.
    """
    if not (needs_lyrics and lyrics_enabled):
        if lyrics_enabled:
            logger.info("  - Lyrics analysis already exists or skipped")
        return False
    logger.info(f"  - Starting lyrics analysis for {track_name_full}...")
    try:
        from lyrics.lyrics_transcriber import analyze_lyrics
        audio_loader = None
        if track_audio is None or track_sr is None:
            if path is not None:
                logger.info("  - Loading audio from file for lyrics analysis")
                track_audio, track_sr = robust_load_fn(str(path), target_sr=16000)
                if track_audio is None or track_audio.size == 0 or track_sr is None:
                    raise RuntimeError("Failed to load audio for lyrics analysis")
            else:
                def audio_loader():  # noqa: F811
                    p = download_fn() if download_fn is not None else None
                    if not p:
                        raise RuntimeError("Failed to download audio for lyrics ASR")
                    a, s = robust_load_fn(str(p), target_sr=16000)
                    if a is None or a.size == 0 or s is None:
                        raise RuntimeError("Failed to load audio for lyrics ASR")
                    return a, s, str(p)
        result = analyze_lyrics(
            audio=track_audio, sr=track_sr,
            source_path=str(path) if path is not None else None,
            artist=item.get('AlbumArtist') or item.get('Artist'),
            track=item.get('Name'), track_id=item.get('Id') or item.get('id'),
            top_moods=top_moods, audio_loader=audio_loader,
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
