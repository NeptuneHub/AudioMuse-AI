# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared analysis planning, identity resolution and task reporting.

The decision layer between the orchestrator (tasks.analysis), the album job
(tasks.analysis.album) and the per-song stages (tasks.analysis.song): what does
each track still need, which catalogue row is this audio, and how does a task
report its progress.

Main Features:
* TrackPlan / plan_track_stages / build_album_plan: which of MusiCNN, CLAP and
  lyrics still need to run, per track and per album.
* load_server_work_map: ONE keyset-paginated scan per server (provider id ->
  work bit mask), so the phase loop decides skip-or-launch from memory.
* resolve_track_identity / claim_new_canonical_id / load_fingerprint_index:
  content identity via the embedding signature, confirmed by exact cosine plus
  track-duration agreement and settled against the DB so concurrent workers
  converge on one catalogue row per recording.
* make_task_reporter: the one task_status reporter every analysis task uses
  (capped log, job.meta mirror, optional progress rescaling and DB throttling).
* flush_pending_track_maps: per-track map-row flush, so a killed worker cannot
  strand an analyzed track without its server mapping.
"""

import logging
import time
from typing import NamedTuple

import numpy as np

from config import (
    ANALYSIS_MONITOR_DB_INTERVAL,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_FAILURE,
)
from database import get_db, save_task_status
from psycopg2 import OperationalError
from psycopg2 import sql as pgsql

from error import error_manager
from error.error_dictionary import ERR_DB_CONNECTION

_SONG_EXPORTS = frozenset((
    'analysis_server_identity', 'catalog_item_id', 'provider_item_id',
    'compute_other_features_str', 'ensure_musicnn_sessions',
    'load_musicnn_sessions', 'persist_clap_embedding', 'persist_musicnn_results',
    'refresh_other_features', 'run_clap_for_track', 'run_lyrics_for_track',
    'run_song_analyzed_hook', 'zero_other_features', 'ZERO_OTHER_FEATURES',
))


def __getattr__(name):
    if name in _SONG_EXPORTS:
        from . import song

        return getattr(song, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _bind_server_context(server_id):
    if not server_id:
        return None
    from flask_app import app
    from ..mediaserver import registry

    with app.app_context():
        return registry.context_for(server_id)


logger = logging.getLogger(__name__)


def make_task_reporter(task_id, task_type, job, initial_message,
                        parent_task_id=None, sub_type_identifier=None,
                        base_details=None, log_cap=200, prefix=None,
                        progress_base=0.0, progress_span=100.0,
                        downgrade_terminal=False, min_db_interval=0.0):
    logs = [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {initial_message}"]
    base = dict(base_details or {})
    state = {'progress': 0, 'last_db': float('-inf')}
    label = prefix or f"{task_type}-{task_id}"

    try:
        save_task_status(
            task_id, task_type, TASK_STATUS_STARTED,
            parent_task_id=parent_task_id, sub_type_identifier=sub_type_identifier,
            progress=int(progress_base), details={**base, "message": initial_message,
                                                  "log": list(logs)},
        )
    except OperationalError as e:
        error_manager.from_exception(e, code=ERR_DB_CONNECTION, logger=logger)
        raise

    def report(message, progress, **kwargs):
        state['progress'] = progress
        logger.info(f"[{label}] {message}")
        task_state = kwargs.get('task_state', TASK_STATUS_PROGRESS)
        details = {**base, **kwargs, "message": message, "status_message": message}
        if downgrade_terminal and task_state in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE):
            task_state = TASK_STATUS_PROGRESS
        scaled = int(progress_base + (progress or 0) * progress_span / 100.0)
        if task_state == TASK_STATUS_SUCCESS:
            details["log"] = [f"Task completed successfully. Final status: {message}"]
        else:
            logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
            if len(logs) > log_cap:
                del logs[:-log_cap]
            details["log"] = logs
        if job:
            job.meta.update(
                {'progress': scaled, 'status_message': message, 'details': details}
            )
            job.save_meta()
        now = time.monotonic()
        throttled = (
            min_db_interval
            and task_state == TASK_STATUS_PROGRESS
            and 'task_state' not in kwargs
            and now - state['last_db'] < min_db_interval
        )
        if throttled:
            return
        state['last_db'] = now
        save_task_status(
            task_id, task_type, task_state,
            parent_task_id=parent_task_id, sub_type_identifier=sub_type_identifier,
            progress=scaled, details=details,
        )

    report.state = state
    return report


def _str_ids(ids):
    return [str(i) for i in ids]


def attach_catalog_item_ids(tracks, server_id=None):
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
            f"WHERE s.item_id IN %s AND {_WORK_ANALYZED}",
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


class TrackPlan(NamedTuple):
    musicnn: bool
    clap: bool
    lyrics: bool

    @property
    def any_stage(self):
        return self.musicnn or self.clap or self.lyrics

    @property
    def needs_audio(self):
        return self.musicnn or self.clap

    def describe(self):
        wanted = [
            name
            for name, on in (('MusiCNN', self.musicnn), ('CLAP', self.clap), ('Lyrics', self.lyrics))
            if on
        ]
        return ' + '.join(wanted) if wanted else 'nothing'


def plan_track_stages(track_id, existing_ids, missing_clap_ids, missing_lyrics_ids,
                      lyrics_enabled):
    return TrackPlan(
        track_id not in existing_ids,
        track_id in missing_clap_ids,
        bool(lyrics_enabled) and track_id in missing_lyrics_ids,
    )


def replan_for_catalogue_row(plan, item_id):
    return TrackPlan(
        False,
        plan.clap and bool(get_missing_ids_in_table('clap_embedding', [item_id])),
        plan.lyrics and bool(get_missing_ids_in_table('lyrics_embedding', [item_id])),
    )


def resolve_track_identity(fingerprint_index, embedding, item, source_server_id,
                           duration=None):
    from tasks import simhash

    from .song import provider_item_id

    provider_id = provider_item_id(item)
    refresh_fingerprint_index(fingerprint_index)
    kind, resolved_id = fingerprint_index.resolve(
        embedding, duration=duration, path=item.get('FilePath'),
        fingerprint=item.get('_chromaprint'),
    )
    if kind == 'new' and resolved_id is not None:
        kind, resolved_id = claim_new_canonical_id(
            fingerprint_index, resolved_id, embedding, duration=duration,
            fingerprint=item.get('_chromaprint'),
        )
    if resolved_id is None:
        resolved_id = simhash.unsignable_canonical_id(source_server_id, provider_id)
        kind = 'unsignable'
    item['_catalog_item_id'] = str(resolved_id)
    return kind, str(resolved_id), provider_id


def top_moods_from(musicnn_analysis, top_n_moods):
    moods = (musicnn_analysis or {}).get('moods') or {}
    ranked = sorted(moods.items(), key=lambda pair: pair[1], reverse=True)
    return dict(ranked[:top_n_moods])


def _album_clap_label_embeddings(track_total, existing_ids, missing_clap_ids, redis_conn):
    from .. import clap_analyzer

    any_track_needs_musicnn = len(existing_ids) < track_total
    if (any_track_needs_musicnn or missing_clap_ids) and clap_analyzer.is_clap_available():
        try:
            labels = clap_analyzer.get_or_cache_other_feature_text_embeddings(redis_conn)
            if labels:
                logger.info(f"OK CLAP other feature text embeddings ready ({len(labels)} labels)")
            else:
                logger.warning("Could not load CLAP text embeddings - other_features will be zeros")
            return labels
        except Exception as e:
            logger.warning(f"Failed to load CLAP text embeddings: {e}")
            return None
    if not any_track_needs_musicnn:
        logger.info("No track in this album needs MusiCNN - skipping CLAP text embedding load")
    else:
        logger.info("CLAP not available - other_features will be zeros")
    return None


def _prior_moods_for_lyrics(track_ids, existing_ids, missing_lyrics_ids, top_n_moods,
                            lyrics_enabled, album_name):
    if not (lyrics_enabled and existing_ids and missing_lyrics_ids):
        return {}
    already_analyzed_needing_lyrics = [
        tid for tid in track_ids if tid in existing_ids and tid in missing_lyrics_ids
    ]
    if not already_analyzed_needing_lyrics:
        return {}
    prior = fetch_existing_top_moods(already_analyzed_needing_lyrics, top_n_moods)
    logger.info(
        f"Prefetched prior moods for {len(prior)}/{len(already_analyzed_needing_lyrics)} "
        f"already-analyzed tracks in '{album_name}' (used as lyrics-pipeline prior)"
    )
    return prior


def build_album_plan(album_name, tracks, top_n_moods, redis_conn, lyrics_enabled):
    from .. import clap_analyzer

    attach_catalog_item_ids(tracks)
    from .song import catalog_item_id

    track_ids = [catalog_item_id(t) for t in tracks]
    existing_ids = get_existing_track_ids(track_ids)
    missing_clap_ids = (
        get_missing_ids_in_table('clap_embedding', track_ids)
        if clap_analyzer.is_clap_available()
        else set()
    )
    missing_lyrics_ids = (
        get_missing_ids_in_table('lyrics_embedding', track_ids) if lyrics_enabled else set()
    )
    logger.info(
        "Feature plan for album '%s': MusiCNN=%d, DCLAP=%d, Lyrics=%d of %d tracks.",
        album_name,
        len(tracks) - len(existing_ids),
        len(missing_clap_ids),
        len(missing_lyrics_ids),
        len(tracks),
    )
    clap_label_embeddings = _album_clap_label_embeddings(
        len(tracks), existing_ids, missing_clap_ids, redis_conn
    )
    prior_moods = _prior_moods_for_lyrics(
        track_ids, existing_ids, missing_lyrics_ids, top_n_moods, lyrics_enabled, album_name
    )
    return existing_ids, missing_clap_ids, missing_lyrics_ids, clap_label_embeddings, prior_moods


def flush_pending_track_maps(pending_track_maps, map_flush_errors, album_name):
    from tasks.mediaserver import registry

    drained = []
    for map_server_id, pending in pending_track_maps.items():
        if not pending:
            continue
        try:
            ready_ids = get_existing_track_ids([v[0] for v in pending.values()])
            filtered = {pid: v for pid, v in pending.items() if v[0] in ready_ids}
            if filtered:
                registry.upsert_track_maps(map_server_id, filtered)
            drained.append(map_server_id)
        except Exception:
            logger.exception(
                "Failed to persist %d pending track map(s) for server %s in album '%s'; "
                "will retry on the next flush",
                len(pending), map_server_id, album_name,
            )
            if str(map_server_id) not in map_flush_errors:
                map_flush_errors.append(str(map_server_id))
    for map_server_id in drained:
        pending_track_maps[map_server_id] = {}


def raise_album_failures(failed_tracks, map_flush_errors, total_tracks_in_album):
    failure_reasons = []
    if failed_tracks:
        preview = "; ".join(failed_tracks[:3])
        failure_reasons.append(
            f"{len(failed_tracks)}/{total_tracks_in_album} tracks failed analysis; "
            f"first failures: {preview}"
        )
    if map_flush_errors:
        failure_reasons.append(
            f"track-server map flush failed for server(s): {', '.join(map_flush_errors)}"
        )
    if failure_reasons:
        raise RuntimeError(" | ".join(failure_reasons))


def album_feature_needs(masks, done_bits, clap_available, lyrics_enabled):
    album_done = sum(1 for m in masks if m & done_bits == done_bits)
    needs_musicnn = any(not m & WORK_MUSICNN for m in masks)
    needs_clap = clap_available and any(not m & WORK_CLAP for m in masks)
    needs_lyrics = lyrics_enabled and any(not m & WORK_LYRICS for m in masks)
    return album_done, needs_musicnn, needs_clap, needs_lyrics


WORK_MUSICNN = 1


WORK_CLAP = 2


WORK_LYRICS = 4


def work_done_bits(clap_available, lyrics_enabled):
    return (
        WORK_MUSICNN
        | (WORK_CLAP if clap_available else 0)
        | (WORK_LYRICS if lyrics_enabled else 0)
    )


_WORK_ANALYZED = (
    "s.other_features IS NOT NULL AND s.energy IS NOT NULL "
    "AND s.mood_vector IS NOT NULL AND s.tempo IS NOT NULL"
)


def _work_feature_parts(clap_available, lyrics_enabled, key_column):
    selects, joins = [], []
    for enabled, table, alias in (
        (clap_available, 'clap_embedding', 'c'),
        (lyrics_enabled, 'lyrics_embedding', 'l'),
    ):
        if enabled:
            selects.append(f"({alias}.item_id IS NOT NULL)")
            joins.append(f"LEFT JOIN {table} {alias} ON {alias}.item_id = {key_column}")
        else:
            selects.append("TRUE")
    return selects, " ".join(joins)


def _apply_work_bits(work_map, provider_id, has_musicnn, has_clap, has_lyrics):
    key = str(provider_id)
    mask = WORK_MUSICNN if has_musicnn else 0
    if has_clap:
        mask |= WORK_CLAP
    if has_lyrics:
        mask |= WORK_LYRICS
    work_map[key] = work_map.get(key, 0) | mask


def _work_map_scan(cur, sql, params, work_map, chunk_size):
    last = ''
    while True:
        cur.execute(sql, (*params, last, chunk_size))
        rows = cur.fetchall()
        if not rows:
            return
        for provider_id, has_musicnn, has_clap, has_lyrics in rows:
            _apply_work_bits(work_map, provider_id, has_musicnn, has_clap, has_lyrics)
        last = str(rows[-1][0])


def _work_sql(clap_available, lyrics_enabled):
    mapped_selects, mapped_joins = _work_feature_parts(clap_available, lyrics_enabled, 'm.item_id')
    mapped_sql = (
        "SELECT m.provider_track_id, "
        f"(e.item_id IS NOT NULL AND {_WORK_ANALYZED}), {', '.join(mapped_selects)} "
        "FROM track_server_map m "
        "JOIN score s ON s.item_id = m.item_id "
        "LEFT JOIN embedding e ON e.item_id = m.item_id "
        f"{mapped_joins} "
        "WHERE m.server_id = %s"
    )
    legacy_selects, legacy_joins = _work_feature_parts(clap_available, lyrics_enabled, 's.item_id')
    legacy_sql = (
        f"SELECT s.item_id, TRUE, {', '.join(legacy_selects)} "
        "FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        f"{legacy_joins} "
        f"WHERE s.item_id NOT LIKE 'fp\\_%%' AND {_WORK_ANALYZED}"
    )
    return mapped_sql, legacy_sql


def _is_default_server(server_id):
    from tasks.mediaserver import registry

    return server_id is None or str(server_id) == str(registry.get_default_server_id() or '')


def load_server_work_map(server_id, clap_available, lyrics_enabled, chunk_size=20000):
    mapped_sql, legacy_sql = _work_sql(clap_available, lyrics_enabled)
    work_map = {}
    with get_db() as conn, conn.cursor() as cur:
        if server_id:
            _work_map_scan(
                cur,
                mapped_sql + " AND m.provider_track_id > %s "
                "ORDER BY m.provider_track_id LIMIT %s",
                (server_id,), work_map, chunk_size,
            )
        if _is_default_server(server_id):
            _work_map_scan(
                cur,
                legacy_sql + " AND s.item_id > %s ORDER BY s.item_id LIMIT %s",
                (), work_map, chunk_size,
            )
    return work_map


def album_work_masks(provider_ids, server_id, clap_available, lyrics_enabled):
    ids = _str_ids(provider_ids)
    if not ids:
        return {}
    mapped_sql, legacy_sql = _work_sql(clap_available, lyrics_enabled)
    work_map = {}
    with get_db() as conn, conn.cursor() as cur:
        if server_id:
            cur.execute(
                mapped_sql + " AND m.provider_track_id = ANY(%s)", (server_id, ids)
            )
            for row in cur.fetchall():
                _apply_work_bits(work_map, *row)
        if _is_default_server(server_id):
            cur.execute(legacy_sql + " AND s.item_id = ANY(%s)", (ids,))
            for row in cur.fetchall():
                _apply_work_bits(work_map, *row)
    return work_map


def _fetch_embedding_blob(item_id):
    with get_db() as conn, conn.cursor() as cur:
        cur.execute("SELECT embedding FROM embedding WHERE item_id = %s", (str(item_id),))
        row = cur.fetchone()
    return bytes(row[0]) if row and row[0] is not None else None


def _fetch_row_duration(item_id):
    with get_db() as conn, conn.cursor() as cur:
        cur.execute("SELECT duration FROM score WHERE item_id = %s", (str(item_id),))
        row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _fetch_row_fingerprint(item_id):
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT c.fingerprint FROM chromaprint c "
            "JOIN track_server_map m "
            "ON m.server_id = c.server_id AND m.provider_track_id = c.provider_track_id "
            "WHERE m.item_id = %s AND c.fingerprint IS NOT NULL "
            "LIMIT 1",
            (str(item_id),),
        )
        row = cur.fetchone()
    return bytes(row[0]) if row and row[0] is not None else None


def _fetch_row_paths(item_id):
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT file_path FROM track_server_map "
            "WHERE item_id = %s AND file_path IS NOT NULL",
            (str(item_id),),
        )
        return [row[0] for row in cur.fetchall()]


_FINGERPRINT_INDEX_TTL_SECONDS = 300.0


_fingerprint_index_cache = {'built': 0.0, 'resolver': None, 'watermark': None}


def refresh_fingerprint_index(resolver, force=False):
    if resolver is None:
        return resolver
    cached = _fingerprint_index_cache
    if cached.get('resolver') is not resolver:
        return resolver
    now = time.monotonic()
    if not force and now - cached.get('refreshed', 0.0) < ANALYSIS_MONITOR_DB_INTERVAL:
        return resolver
    cached['refreshed'] = now
    try:
        from tasks.simhash import CANONICAL_ID_LEN

        with get_db() as conn, conn.cursor() as cur:
            if cached.get('watermark') is None:
                cur.execute("SELECT now()")
                cached['watermark'] = cur.fetchone()[0]
                return resolver
            cur.execute(
                "SELECT item_id, created_at, duration FROM score "
                "WHERE created_at > %s AND item_id LIKE 'fp\\_%%' "
                "AND length(item_id) = %s "
                "AND substring(item_id from 4 for 1) BETWEEN '1' AND '9' "
                "ORDER BY created_at",
                (cached['watermark'], CANONICAL_ID_LEN),
            )
            rows = cur.fetchall()
        for item_id, created_at, duration in rows:
            resolver.register(item_id, duration=duration)
            cached['watermark'] = created_at
        if rows:
            logger.info(
                "Fingerprint index caught up with %d canonical row(s) another worker "
                "committed since this one last looked.", len(rows),
            )
    except Exception:
        logger.exception("Could not refresh the fingerprint index; using the snapshot")
    return resolver


def load_fingerprint_index():
    from tasks.simhash import CANONICAL_ID_LEN, CatalogResolver

    now = time.monotonic()
    cached = _fingerprint_index_cache
    if cached['resolver'] is not None:
        if now - cached['built'] >= _FINGERPRINT_INDEX_TTL_SECONDS:
            cached['resolver'].drop_cached_embeddings()
            refresh_fingerprint_index(cached['resolver'], force=True)
            cached['built'] = now
        return cached['resolver']

    resolver = CatalogResolver(
        embedding_fetcher=_fetch_embedding_blob,
        duration_fetcher=_fetch_row_duration,
        path_fetcher=_fetch_row_paths,
        fingerprint_fetcher=_fetch_row_fingerprint,
    )
    with get_db() as conn, conn.cursor() as cur:
        cur.execute("SELECT now()")
        watermark = cur.fetchone()[0]
        cur.execute(
            "SELECT item_id, duration FROM score "
            "WHERE item_id LIKE 'fp\\_%%' AND length(item_id) = %s "
            "AND substring(item_id from 4 for 1) BETWEEN '1' AND '9'",
            (CANONICAL_ID_LEN,),
        )
        for item_id, duration in cur.fetchall():
            resolver.register(item_id, duration=duration)
    cached['built'] = now
    cached['resolver'] = resolver
    cached['watermark'] = watermark
    return resolver


def catalogue_embedding(item_id):
    blob = _fetch_embedding_blob(item_id)
    if blob is None:
        return None
    vector = np.frombuffer(blob, dtype=np.float32)
    return vector if vector.size else None


def _is_same_recording(embedding, other, duration=None, other_duration_fn=None,
                       fingerprint=None, other_fingerprint_fn=None):
    from tasks.simhash import cosine_distance, durations_compatible
    from tasks.chromaprint import chromaprints_agree
    from config import DUPLICATE_DISTANCE_THRESHOLD_COSINE, CHROMAPRINT_GATE_ENABLED

    if other is None:
        return False
    if cosine_distance(embedding, other) > DUPLICATE_DISTANCE_THRESHOLD_COSINE:
        return False
    other_duration = other_duration_fn() if other_duration_fn is not None else None
    if not durations_compatible(duration, other_duration):
        return False
    if CHROMAPRINT_GATE_ENABLED and fingerprint:
        other_fp = other_fingerprint_fn() if other_fingerprint_fn is not None else None
        if chromaprints_agree(fingerprint, other_fp) is False:
            return False
    return True


def claim_new_canonical_id(resolver, minted_id, embedding, duration=None, fingerprint=None):
    from tasks.simhash import mint_canonical_id, signature_from_canonical_id

    if not minted_id:
        return ('new', minted_id)

    taken = set()
    candidate = minted_id
    while True:
        if not get_existing_track_ids([candidate]):
            if candidate != minted_id:
                resolver.register(
                    candidate,
                    embedding=embedding,
                    signature=signature_from_canonical_id(candidate),
                    duration=duration,
                    fingerprint=fingerprint,
                )
            return ('new', candidate)

        stored = catalogue_embedding(candidate)
        if _is_same_recording(
            embedding, stored, duration=duration,
            other_duration_fn=lambda candidate=candidate: _fetch_row_duration(candidate),
            fingerprint=fingerprint,
            other_fingerprint_fn=lambda candidate=candidate: _fetch_row_fingerprint(candidate),
        ):
            logger.info(
                "Canonical id %s was minted concurrently by another worker for the "
                "same recording; adopting it instead of persisting a duplicate.",
                candidate,
            )
            return ('existing', candidate)

        logger.warning(
            "Canonical id %s already belongs to a track this one cannot be proven "
            "identical to (different audio, different duration, or unknown "
            "duration); minting the next free id rather than overwriting it.",
            candidate,
        )
        if stored is not None:
            resolver.register(candidate, embedding=stored)
        taken.add(candidate)
        signature = signature_from_canonical_id(minted_id)
        if signature is None:
            return ('new', minted_id)
        candidate = mint_canonical_id(signature, taken)


def build_feature_status_parts(clap_available, lyrics_enabled, include_check_marks=False):
    parts = ["MusiCNN"]
    if clap_available:
        parts.append("CLAP")
    if lyrics_enabled:
        parts.append("Lyrics")
    if include_check_marks:
        return [f"{p}: OK" for p in parts]
    return parts
