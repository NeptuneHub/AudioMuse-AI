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
* load_server_work_map: ONE keyset-paginated scan per server returning what each
  of its provider tracks already has (musicnn / CLAP / lyrics bit mask), so the
  album loop decides skip-or-launch from memory instead of querying per album.
* decide_track_needs: per-track dedup deciding which of musicnn, CLAP and lyrics
  embeddings are missing (the real analysis dedup).
* load_fingerprint_index: Hamming-tolerant catalogue index over the canonical
  embedding-hash ids, consulted after MusiCNN to dedup cross-server content.
* persist_* helpers upsert mood tags, embeddings, CLAP and lyrics vectors.
"""

import gc
import importlib
import logging
import time

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


def resolve_providers(allow_coreml=False, cuda_options=None):
    available = ort.get_available_providers()
    chain = []

    if 'CUDAExecutionProvider' in available:
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

    if allow_coreml and 'CoreMLExecutionProvider' in available:
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


def _default_sess_options():
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return opts


def create_onnx_session(model_path, provider_options=None, label="", sess_options=None):
    opts = provider_options or resolve_providers()
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


def catalog_item_id(item):
    """Return the canonical catalogue id attached to a provider track."""
    return str(item.get('_catalog_item_id') or item.get('Id') or item.get('id'))


def attach_catalog_item_ids(tracks, server_id=None):
    """Attach canonical ids resolved from this server's provider ids.

    Unknown provider tracks retain their provider id temporarily. After their
    first analysis canonicalization rewrites that id and records the server map.
    """
    if not tracks:
        return tracks
    from tasks.mediaserver import context, registry

    provider_ids = [str(t.get('Id') or t.get('id')) for t in tracks]
    active_server_id = server_id or context.active_server_id()
    mapped = registry.reverse_translate_ids(provider_ids, active_server_id)
    for item, provider_id in zip(tracks, provider_ids):
        item['_catalog_item_id'] = str(mapped.get(provider_id, provider_id))
    return tracks


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


def upsert_artist_mappings_for_tracks(tracks, album_name=None):
    """Record this server's artist links for the album being analyzed.

    Only ``artist_server_map`` is written. The legacy ``artist_mapping`` table is
    the sweep's to refresh, from the server's whole catalogue in one pass, so a
    per-album analysis run is not a second writer of it.
    """
    last_id_by_name = {}
    for t in tracks:
        name, aid = t.get('AlbumArtist'), t.get('ArtistId')
        if name and aid:
            last_id_by_name[name] = aid
        elif name:
            last_id_by_name.setdefault(name, None)
    from tasks.mediaserver import context, registry

    valid = {name: artist_id for name, artist_id in last_id_by_name.items() if artist_id}
    server_id = context.active_server_id() or registry.get_default_server_id()
    if valid and server_id:
        registry.upsert_artist_maps(server_id, valid)
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


WORK_MUSICNN = 1
WORK_CLAP = 2
WORK_LYRICS = 4


def work_done_bits(clap_available, lyrics_enabled):
    """The mask a track must carry to need no work at all in this configuration."""
    return (
        WORK_MUSICNN
        | (WORK_CLAP if clap_available else 0)
        | (WORK_LYRICS if lyrics_enabled else 0)
    )


def _work_map_scan(cur, sql, params, work_map, chunk_size):
    last = ''
    while True:
        cur.execute(sql, (*params, last, chunk_size))
        rows = cur.fetchall()
        if not rows:
            return
        for provider_id, has_musicnn, has_clap, has_lyrics in rows:
            key = str(provider_id)
            mask = WORK_MUSICNN if has_musicnn else 0
            if has_clap:
                mask |= WORK_CLAP
            if has_lyrics:
                mask |= WORK_LYRICS
            work_map[key] = work_map.get(key, 0) | mask
        last = str(rows[-1][0])


def load_server_work_map(server_id, is_default, clap_available, lyrics_enabled,
                         chunk_size=20000):
    """What one server's tracks already have, keyed by PROVIDER track id.

    ONE keyset-paginated scan per server instead of a handful of DB round-trips
    per album: the album loop then decides skip-or-launch from memory alone, so
    a settled library costs one query rather than one query set per album. The
    value is a bit mask (musicnn / clap / lyrics) rather than a plain "done"
    flag, because the loop reports how many albums need each feature.

    The default server gets a second scan: an unknown provider id resolves to
    ITSELF there (``registry.reverse_translate_ids``), so a legacy pre-canonical
    ``score`` row counts as analyzed while owning no map row at all. A secondary
    server has no such fallback and must not see them.
    """
    def feature_parts(key_column):
        selects, joins = [], []
        for enabled, table, alias in (
            (clap_available, 'clap_embedding', 'c'),
            (lyrics_enabled, 'lyrics_embedding', 'l'),
        ):
            if enabled:
                selects.append(f"({alias}.item_id IS NOT NULL)")
                joins.append(
                    f"LEFT JOIN {table} {alias} ON {alias}.item_id = {key_column}"
                )
            else:
                selects.append("TRUE")
        return selects, " ".join(joins)

    analyzed = (
        "s.other_features IS NOT NULL AND s.energy IS NOT NULL "
        "AND s.mood_vector IS NOT NULL AND s.tempo IS NOT NULL"
    )

    mapped_selects, mapped_joins = feature_parts('m.item_id')
    mapped_sql = (
        "SELECT m.provider_track_id, "
        f"(e.item_id IS NOT NULL AND {analyzed}), {', '.join(mapped_selects)} "
        "FROM track_server_map m "
        "JOIN score s ON s.item_id = m.item_id "
        "LEFT JOIN embedding e ON e.item_id = m.item_id "
        f"{mapped_joins} "
        "WHERE m.server_id = %s AND m.provider_track_id > %s "
        "ORDER BY m.provider_track_id LIMIT %s"
    )

    legacy_selects, legacy_joins = feature_parts('s.item_id')
    legacy_sql = (
        f"SELECT s.item_id, TRUE, {', '.join(legacy_selects)} "
        "FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        f"{legacy_joins} "
        f"WHERE s.item_id NOT LIKE 'fp\\_%%' AND {analyzed} "
        "AND s.item_id > %s ORDER BY s.item_id LIMIT %s"
    )

    work_map = {}
    with get_db() as conn, conn.cursor() as cur:
        if server_id:
            _work_map_scan(cur, mapped_sql, (server_id,), work_map, chunk_size)
        if is_default:
            _work_map_scan(cur, legacy_sql, (), work_map, chunk_size)
    return work_map


def _fetch_embedding_blob(item_id):
    with get_db() as conn, conn.cursor() as cur:
        cur.execute("SELECT embedding FROM embedding WHERE item_id = %s", (str(item_id),))
        row = cur.fetchone()
    return bytes(row[0]) if row and row[0] is not None else None


_FINGERPRINT_INDEX_TTL_SECONDS = 300.0
_fingerprint_index_cache = {'built': 0.0, 'resolver': None}


def load_fingerprint_index():
    """Identity resolver over every canonical row already in the catalogue.

    Built from item_id strings alone (the id encodes the 200-bit signature), so
    loading it never reads embedding blobs; a candidate row's raw embedding is
    fetched lazily only when the signature proposes it, and the exact cosine
    (the Similar Songs duplicate rule) takes the final same/different decision.
    Cached per worker process for a few minutes: resolve() registers every id
    this worker mints, so the cache stays current between refreshes and album
    jobs stop rebuilding the full index each time.
    """
    from tasks.simhash import CANONICAL_ID_LEN, CatalogResolver

    now = time.monotonic()
    cached = _fingerprint_index_cache
    if (
        cached['resolver'] is not None
        and now - cached['built'] < _FINGERPRINT_INDEX_TTL_SECONDS
    ):
        return cached['resolver']

    resolver = CatalogResolver(embedding_fetcher=_fetch_embedding_blob)
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT item_id FROM score "
            "WHERE item_id LIKE 'fp\\_2%%' AND length(item_id) = %s",
            (CANONICAL_ID_LEN,),
        )
        for (item_id,) in cur.fetchall():
            resolver.register(item_id)
    cached['built'] = now
    cached['resolver'] = resolver
    return resolver


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


def persist_musicnn_results(
    item, analysis, top_moods, embedding, other_features_str, is_default_server=True
):
    """Store one track's MusiCNN results under its canonical catalogue id.

    ``file_path`` is written ONLY from the default server. The catalogue row is
    shared by every server holding the track but carries a single path, and that
    path is the top-priority tier of the matcher that onboards the NEXT server.
    Letting a secondary stamp its own layout onto the shared row would silently
    demote the track to the weaker metadata tiers forever. The upsert COALESCEs
    it, so passing None never erases a path the default server already wrote.
    """
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
        file_path=item.get('FilePath') if is_default_server else None,
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
    except Exception as e:
        error_manager.record(
            error_manager.classify(e, ERR_LYRICS_TRANSCRIPTION),
            str(e),
            exc=e,
            logger=logger,
            level=logging.WARNING,
        )
        return False
