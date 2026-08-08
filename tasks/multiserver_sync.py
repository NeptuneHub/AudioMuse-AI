# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The sweep: alignment of the analyzed catalogue with media servers.

The ONLY place that walks a server's whole catalogue to reconcile it with the
analyzed database. A sweep is a pure metadata pass, never a download or an
analysis: track mappings via normalized path, path tail and metadata tiers, plus
the server's artist links and a set-based catalogue metadata refresh. It runs
automatically when a server is added or its matching settings change, and from
the Align button.

Runs as a high-priority queue job reporting progress into task_status and is
cancellable. Unmatched tracks are left unmapped; re-sweeps are incremental.
Full-refresh sweeps prune mappings whose provider track is gone (only map rows,
never analyzed tracks), skipping the prune when a fetch looks partial.

Main Features:
* sweep_server / sweep_all_secondary_servers entry points with live
  progress and cooperative cancellation; helpers shared with the cleaning task so
  the two can never drift apart.
* Zero-download alignment, artist link upserts, and batch metadata refresh.
* Lean memory: fetched catalogue is condensed into a slim CandidateIndex and the
  local side streams through it in keyset-paginated chunks.
* A sweep whose worker died is restarted by the queue's own reclaim; an
  empty-catalogue guard makes first-install sweeps instant no-ops.
"""

import logging
import time
import uuid

from psycopg2 import sql as pgsql
from psycopg2.extras import execute_values

from config import SWEEP_PRUNE_MIN_FETCH_RATIO
from database import (
    connect_raw,
    stage_pending_task_row,
)
from sanitization import sanitize_string_for_db
from tasks import provider_probe
from tasks.mediaserver import context as ms_context, registry
from tasks.provider_migration_matcher import CandidateIndex

logger = logging.getLogger(__name__)

SWEEP_TASK_TYPE = 'server_sweep'


class SweepCancelled(Exception):
    pass


def insert_pending_sweep_row(cur, task_id, message):
    return stage_pending_task_row(
        cur,
        task_id,
        SWEEP_TASK_TYPE,
        {'message': message, 'status_message': message},
    )


def enqueue_server_alignment(server_id=None, message=None, task_id=None,
                             parent_task_id=None):
    import config
    import taskqueue

    task_id = task_id or str(uuid.uuid4())
    text = message or 'Server alignment queued.'
    db = connect_raw()
    try:
        target = str(server_id) if server_id else registry.get_default_server_id(db)
        if not target:
            return None
        with db.cursor() as cur:
            cur.execute(
                "SELECT task_id FROM task_status WHERE task_type = %s "
                "AND status = ANY(%s) "
                "ORDER BY timestamp DESC",
                (SWEEP_TASK_TYPE, list(config.TASK_STATUS_LIVE)),
            )
            other_active = [row[0] for row in cur.fetchall() if row[0] != task_id]
        if other_active:
            logger.info(
                "Alignment %s coalesced into already-live sweep %s", task_id, other_active[0],
            )
            db.commit()
            return other_active[0]
        try:
            taskqueue.enqueue(
                'tasks.multiserver_sync.sweep_server',
                args=(target,),
                kwargs={'task_id': task_id, 'parent_task_id': parent_task_id},
                task_id=task_id,
                task_type=SWEEP_TASK_TYPE,
                queue=taskqueue.QUEUE_HIGH,
                parent_task_id=parent_task_id,
                details={'message': text, 'status_message': text},
                conn=db,
            )
        except taskqueue.TaskAlreadyRunning:
            db.rollback()
            logger.info("Alignment %s lost the race to another live sweep.", task_id)
            return None
        except taskqueue.TaskNotQueued:
            db.rollback()
            logger.info("Alignment %s was already queued by an earlier attempt.", task_id)
            return task_id
        db.commit()
        return task_id
    except Exception:
        db.rollback()
        raise
    finally:
        try:
            db.close()
        except Exception:
            logger.debug("Alignment enqueue connection close failed", exc_info=True)


def _make_reporter(task_id, label, parent_task_id=None):
    try:
        from flask_app import app
        from app_helper import save_task_status
        from config import TASK_STATUS_PROGRESS
    except Exception:
        app = None
    last = {'pct': -1}

    def report(message, progress, task_state=None):
        pct = max(0, min(100, int(progress)))
        logger.info("[Sweep-%s] %s (%d%%)", label, message, pct)
        if app is None:
            return
        if task_state is None and pct == last['pct']:
            return
        last['pct'] = pct
        details = {'status_message': message, 'message': message}
        try:
            with app.app_context():
                save_task_status(
                    task_id,
                    SWEEP_TASK_TYPE,
                    task_state or TASK_STATUS_PROGRESS,
                    parent_task_id=parent_task_id,
                    progress=pct,
                    details=details,
                )
        except Exception:
            logger.debug("Sweep status update failed (ignored)", exc_info=True)

    return report


def make_cancel_check(task_id):
    import config

    try:
        check_conn = connect_raw()
        check_conn.autocommit = True
    except Exception:
        check_conn = None
    state = {'last': 0.0}

    def check():
        if check_conn is None:
            return
        now = time.monotonic()
        if now - state['last'] < 2.0:
            return
        state['last'] = now
        try:
            cur = check_conn.cursor()
            try:
                cur.execute("SELECT status FROM task_status WHERE task_id = %s", (task_id,))
                row = cur.fetchone()
            finally:
                cur.close()
        except Exception:
            logger.debug("Sweep cancel check failed (ignored)", exc_info=True)
            return
        if row is None or row[0] == config.TASK_STATUS_REVOKED:
            raise SweepCancelled()

    def close():
        if check_conn is not None:
            try:
                check_conn.close()
            except Exception:
                logger.debug("Sweep cancel-check connection close failed", exc_info=True)

    return check, close


_make_cancel_check = make_cancel_check


def _resolve_task_id(task_id):
    if task_id:
        return task_id
    import taskqueue

    return taskqueue.current_task_id() or str(uuid.uuid4())


def unmapped_local_count(conn, server_id):
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM score s WHERE NOT EXISTS ("
            "SELECT 1 FROM track_server_map m WHERE m.item_id = s.item_id AND m.server_id = %s)",
            (server_id,),
        )
        return cur.fetchone()[0]
    finally:
        cur.close()


def _iter_unmapped_local_rows(conn, server_id, chunk_size=20000):
    last_id = ''
    while True:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT s.item_id, s.title, s.author, s.album, s.album_artist, "
                "s.file_path, ARRAY(SELECT DISTINCT p.file_path FROM track_server_map p "
                "WHERE p.item_id = s.item_id AND p.file_path IS NOT NULL) "
                "FROM score s WHERE s.item_id > %s AND NOT EXISTS ("
                "SELECT 1 FROM track_server_map m WHERE m.item_id = s.item_id AND m.server_id = %s) "
                "ORDER BY s.item_id LIMIT %s",
                (last_id, server_id, chunk_size),
            )
            rows = cur.fetchall()
        finally:
            cur.close()
        if not rows:
            return
        last_id = rows[-1][0]
        yield [
            {
                'item_id': r[0],
                'title': r[1],
                'author': r[2],
                'album': r[3],
                'album_artist': r[4],
                'file_path': r[5],
                'file_paths': [p for p in (list(r[6] or []) + [r[5]]) if p],
            }
            for r in rows
        ]


def _local_track_count(conn):
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM score")
        return cur.fetchone()[0]
    finally:
        cur.close()


def _already_mapped_ids(db, server_id):
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT provider_track_id FROM track_server_map WHERE server_id = %s", (server_id,)
        )
        return {str(r[0]) for r in cur.fetchall()}
    finally:
        cur.close()


def _write_matches(db, server_id, result, path_by_id=None):
    paths = path_by_id or {}
    mapping = {
        new_id: (
            item_id,
            result['match_tiers'].get(item_id),
            paths.get(str(new_id)),
        )
        for item_id, new_id in result['matches'].items()
    }
    return registry.upsert_track_maps(server_id, mapping, conn=db)


def prune_stale_mappings(db, server_id, present_ids, refused=None):
    present_set = {_strip_nul(str(pid)) for pid in present_ids if pid is not None}
    present_set.discard('')
    present = [(pid,) for pid in present_set]
    if not present:
        return 0
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM track_server_map WHERE server_id = %s", (server_id,)
        )
        current = cur.fetchone()[0]
        if current > 0 and len(present) < current * SWEEP_PRUNE_MIN_FETCH_RATIO:
            logger.warning(
                "Multi-server sweep for server %s: fetch returned %d tracks but %d "
                "mappings exist; fetch looks partial, pruning skipped",
                server_id, len(present), current,
            )
            if refused is not None:
                refused.append((len(present), current))
            return 0
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS sweep_present_ids "
            "(provider_track_id TEXT PRIMARY KEY)"
        )
        cur.execute("DELETE FROM sweep_present_ids")
        execute_values(
            cur,
            "INSERT INTO sweep_present_ids (provider_track_id) VALUES %s "
            "ON CONFLICT DO NOTHING",
            present,
            page_size=5000,
        )
        cur.execute(
            "DELETE FROM track_server_map t WHERE t.server_id = %s "
            "AND NOT EXISTS (SELECT 1 FROM sweep_present_ids p "
            "WHERE p.provider_track_id = t.provider_track_id)",
            (server_id,),
        )
        removed = cur.rowcount
        cur.execute("DROP TABLE sweep_present_ids")
        if removed:
            cur.execute(
                "UPDATE music_servers SET updated_at = now() WHERE server_id = %s",
                (server_id,),
            )
        db.commit()
        if removed:
            try:
                from tasks.paged_ivf import invalidate_availability_cache
                invalidate_availability_cache(server_id)
            except Exception:
                logger.debug("Availability-cache invalidation failed", exc_info=True)
        return removed
    finally:
        cur.close()


def _store_server_track_count(db, server_id, track_count):
    cur = db.cursor()
    try:
        cur.execute(
            "UPDATE music_servers SET track_count = %s WHERE server_id = %s",
            (int(track_count), server_id),
        )
        db.commit()
    except Exception:
        logger.debug("Could not persist track count for server %s", server_id, exc_info=True)
        try:
            db.rollback()
        except Exception:
            logger.debug("Track-count rollback failed", exc_info=True)
    finally:
        cur.close()


def _strip_nul(value):
    if isinstance(value, str):
        return sanitize_string_for_db(value)
    return value


def _collect_artist_maps(tracks):
    maps = {}
    for t in tracks:
        name = t.get('artist') or t.get('album_artist')
        artist_id = t.get('artist_id')
        if name and artist_id:
            maps[_strip_nul(str(name))] = _strip_nul(str(artist_id))
    return maps


def _write_artist_maps(db, server, artist_maps):
    if not artist_maps:
        return 0
    try:
        return registry.upsert_artist_maps(server['server_id'], artist_maps, conn=db)
    except Exception:
        logger.exception("Artist map upsert failed for server %s", server['server_id'])
        return 0


_META_FIELDS = ('album', 'album_artist', 'year', 'rating')


def _stage_track_metadata(db, tracks):
    rows = {}
    for t in tracks:
        provider_id = t.get('id')
        if not provider_id:
            continue
        clean_id = _strip_nul(str(provider_id))
        if not clean_id:
            continue
        rows[clean_id] = (
            clean_id, _strip_nul(t.get('album')),
            _strip_nul(t.get('album_artist')),
            t.get('year'), t.get('rating'), _strip_nul(t.get('path')),
        )
    tracks = None
    cur = db.cursor()
    try:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS sweep_track_meta "
            "(provider_track_id TEXT PRIMARY KEY, album TEXT, album_artist TEXT, "
            "year INTEGER, rating INTEGER, file_path TEXT)"
        )
        cur.execute("DELETE FROM sweep_track_meta")
        if rows:
            execute_values(
                cur,
                "INSERT INTO sweep_track_meta VALUES %s",
                list(rows.values()),
                page_size=5000,
            )
        db.commit()
    except Exception:
        logger.exception("Could not stage catalogue metadata for the sweep refresh")
        try:
            db.rollback()
        except Exception:
            logger.debug("Metadata staging rollback failed", exc_info=True)
    finally:
        cur.close()


def _refresh_mapped_metadata(db, server_id):
    cur = db.cursor()
    try:
        cur.execute("SELECT to_regclass('sweep_track_meta')")
        if cur.fetchone()[0] is None:
            return 0
    finally:
        cur.close()
    fields = _META_FIELDS
    set_parts = pgsql.SQL(", ").join(
        pgsql.SQL("{0} = COALESCE(i.{0}, s.{0})").format(pgsql.Identifier(f))
        for f in fields
    )
    changed_parts = pgsql.SQL(" OR ").join(
        pgsql.SQL("(i.{0} IS NOT NULL AND s.{0} IS DISTINCT FROM i.{0})").format(
            pgsql.Identifier(f)
        )
        for f in fields
    )
    query = pgsql.SQL(
        "UPDATE score s SET {} FROM ("
        "  SELECT DISTINCT ON (m.item_id) m.item_id AS item_id, i.* "
        "  FROM track_server_map m "
        "  JOIN sweep_track_meta i ON i.provider_track_id = m.provider_track_id "
        "  WHERE m.server_id = %s "
        "  ORDER BY m.item_id, m.provider_track_id"
        ") i WHERE s.item_id = i.item_id AND ({})"
    ).format(set_parts, changed_parts)
    cur = db.cursor()
    try:
        cur.execute(query, (server_id,))
        refreshed = cur.rowcount
        cur.execute(
            "UPDATE track_server_map m SET file_path = i.file_path "
            "FROM sweep_track_meta i "
            "WHERE m.provider_track_id = i.provider_track_id AND m.server_id = %s "
            "AND i.file_path IS NOT NULL AND m.file_path IS DISTINCT FROM i.file_path",
            (server_id,),
        )
        cur.execute("DROP TABLE IF EXISTS sweep_track_meta")
        db.commit()
        return refreshed
    except Exception:
        logger.exception("Catalogue metadata refresh failed for server %s", server_id)
        try:
            db.rollback()
        except Exception:
            logger.debug("Metadata refresh rollback failed", exc_info=True)
        return 0
    finally:
        cur.close()


def fetch_server_catalogue(server):
    import config

    stype = server['server_type'] if server else config.MEDIASERVER_TYPE
    creds = server['creds'] if server else None
    with ms_context.use_server(server):
        return provider_probe.fetch_all_tracks(stype, creds, apply_filter=True)


def _sweep_one(server, db, report, base, span, cancel, full_refresh=False):
    stype = server['server_type']
    server_id = server['server_id']
    total_local = _local_track_count(db)
    if not total_local:
        report(
            f"Nothing analyzed yet; {server['name']} aligns automatically during the first analysis.",
            base + span,
        )
        return {
            'server_id': server_id, 'name': server['name'], 'server_type': stype,
            'target_tracks': 0, 'local_tracks': 0, 'unmapped': 0,
            'matched': 0, 'aligned': True, 'empty_catalogue': True, 'tier_counts': {},
        }
    unmapped_count = unmapped_local_count(db, server_id)
    if not unmapped_count and not full_refresh:
        report(
            f"{server['name']} is already aligned ({total_local} tracks mapped); nothing to do.",
            base + span,
        )
        return {
            'server_id': server_id, 'name': server['name'], 'server_type': stype,
            'target_tracks': 0, 'local_tracks': total_local, 'unmapped': 0,
            'matched': 0, 'aligned': True, 'tier_counts': {},
        }

    report(f"Fetching catalogue from {server['name']} ({stype})...", base + span * 0.1)
    target_tracks = fetch_server_catalogue(server)
    cancel()

    target_total = len(target_tracks)
    present_ids = {str(t['id']) for t in target_tracks if t.get('id')}
    artist_maps = _collect_artist_maps(target_tracks)
    _stage_track_metadata(db, target_tracks)
    already_mapped = _already_mapped_ids(db, server_id)

    def _drain_candidates(tracks):
        while tracks:
            track = tracks.pop()
            if track.get('id') and _strip_nul(str(track.get('id'))) not in already_mapped:
                yield track

    index = CandidateIndex(_drain_candidates(target_tracks))
    target_tracks = None
    _store_server_track_count(db, server_id, target_total)
    pruned = 0
    prune_refused = []
    if full_refresh:
        pruned = prune_stale_mappings(db, server_id, present_ids, refused=prune_refused)
        if prune_refused:
            fetched, mapped = prune_refused[0]
            report(
                f"{server['name']}: only {fetched} of the {mapped} tracks it has mapped "
                "came back, so stale mappings were NOT pruned. Re-run the alignment if "
                "the library really shrank that much.",
                base + span * 0.5,
            )
        if pruned:
            logger.info(
                "Multi-server sweep for '%s': pruned %d stale mappings no longer on the server",
                server['name'], pruned,
            )
            unmapped_count = unmapped_local_count(db, server_id)
    report(
        f"Aligning {server['name']}: {unmapped_count} tracks to match "
        f"({total_local - unmapped_count} already aligned)...",
        base + span * 0.5,
    )

    written = 0
    processed = 0
    tier_counts = {}
    claimed = {}
    if index.size:
        for chunk in _iter_unmapped_local_rows(db, server_id):
            cancel()
            result = index.match_chunk(chunk, claimed)
            written += _write_matches(db, server_id, result, index.path_by_id)
            processed += len(chunk)
            for tier, count in result['tier_counts'].items():
                if count:
                    tier_counts[tier] = tier_counts.get(tier, 0) + count
            if unmapped_count:
                pct = base + span * (0.5 + 0.45 * min(1.0, processed / unmapped_count))
                report(
                    f"Aligning {server['name']}: {min(processed, unmapped_count)}/"
                    f"{unmapped_count} checked, {written} matched...",
                    pct,
                )
    refreshed = _refresh_mapped_metadata(db, server_id)
    artists_written = _write_artist_maps(db, server, artist_maps)
    logger.info(
        "Multi-server sweep for '%s': mapped %d/%d unmapped tracks "
        "(target=%d, tiers=%s), %d artist links, %d metadata rows refreshed",
        server['name'], written, unmapped_count, target_total, tier_counts,
        artists_written, refreshed,
    )
    return {
        'server_id': server_id,
        'name': server['name'],
        'server_type': stype,
        'target_tracks': target_total,
        'local_tracks': total_local,
        'unmapped': unmapped_count,
        'matched': written,
        'pruned': pruned,
        'prune_refused': bool(prune_refused),
        'artists': artists_written,
        'refreshed': refreshed,
        'tier_counts': tier_counts,
    }


def sweep_server(server_id, task_id=None, conn=None, parent_task_id=None):
    import config

    task_id = _resolve_task_id(task_id)
    own_conn = conn is None
    db = conn or connect_raw()
    report = None
    cancel, close_cancel = _make_cancel_check(task_id)
    try:
        from config import TASK_STATUS_STARTED, TASK_STATUS_SUCCESS

        cancel()
        report = _make_reporter(task_id, server_id, parent_task_id=parent_task_id)
        server = registry.get_server(server_id, conn=db)
        if server is None:
            report("Server no longer exists; nothing to align.", 100, task_state=TASK_STATUS_SUCCESS)
            return {'server_id': server_id, 'skipped': 'deleted', 'matched': 0}

        report(f"Starting alignment for {server['name']}...", 2, task_state=TASK_STATUS_STARTED)
        cancel()
        summary = _sweep_one(server, db, report, 5, 95, cancel, full_refresh=True)
        if summary.get('empty_catalogue'):
            message = "Nothing analyzed yet; alignment runs automatically during the first analysis."
        elif summary.get('aligned'):
            message = f"{server['name']} is already aligned; nothing to do."
        else:
            message = (
                f"Alignment complete: {summary['matched']}/{summary['unmapped']} pending tracks "
                f"matched on {server['name']}"
                + (f", {summary['pruned']} stale mappings removed." if summary.get('pruned')
                   else ".")
            )
        report(message, 100, task_state=TASK_STATUS_SUCCESS)
        return summary
    except SweepCancelled:
        if report is not None:
            report("Alignment cancelled; matches found so far are kept.", 100,
                   task_state=config.TASK_STATUS_REVOKED)
        return {'server_id': server_id, 'cancelled': True}
    except Exception:
        logger.exception("Multi-server sweep failed for server %s", server_id)
        try:
            db.rollback()
        except Exception:
            logger.debug("Rollback after failed sweep failed", exc_info=True)
        report(
            "Alignment failed; check the container logs for details.",
            100,
            task_state=config.TASK_STATUS_FAILURE,
        )
        return {'server_id': server_id, 'error': 'sweep failed'}
    finally:
        close_cancel()
        if own_conn:
            db.close()


def sweep_all_secondary_servers(task_id=None, conn=None, server_ids=None, full_refresh=None):
    import config

    if full_refresh is None:
        full_refresh = server_ids is None

    task_id = _resolve_task_id(task_id)
    own_conn = conn is None
    db = conn or connect_raw()
    report = None
    cancel, close_cancel = _make_cancel_check(task_id)
    try:
        from config import TASK_STATUS_STARTED, TASK_STATUS_SUCCESS

        cancel()
        report = _make_reporter(task_id, 'all')
        selected = {str(server_id) for server_id in server_ids} if server_ids is not None else None
        servers = [
            s for s in registry.list_servers(conn=db)
            if selected is None or s['server_id'] in selected
        ]
        report(
            f"Starting alignment for {len(servers)} selected server(s)...",
            2, task_state=TASK_STATUS_STARTED,
        )
        cancel()
        if not servers:
            report("No selected servers to align.", 100,
                   task_state=TASK_STATUS_SUCCESS)
            return []

        span = 95 / len(servers)
        results = []
        for i, server in enumerate(servers):
            try:
                results.append(
                    _sweep_one(
                        server, db, report, 5 + i * span, span, cancel,
                        full_refresh=full_refresh,
                    )
                )
            except SweepCancelled:
                report("Alignment cancelled; matches found so far are kept.", 100,
                       task_state=config.TASK_STATUS_REVOKED)
                return results
            except Exception:
                logger.exception("Multi-server sweep failed for server %s", server['server_id'])
                try:
                    db.rollback()
                except Exception:
                    logger.debug("Rollback after failed server sweep failed", exc_info=True)
                results.append({'server_id': server['server_id'], 'error': 'sweep failed'})
        if all(r.get('empty_catalogue') for r in results):
            report(
                "Nothing analyzed yet; alignment runs automatically during the first analysis.",
                100, task_state=TASK_STATUS_SUCCESS,
            )
            return results
        matched = sum(r.get('matched', 0) for r in results)
        report(
            f"Alignment complete for {len(servers)} server(s); {matched} track mappings written.",
            100, task_state=TASK_STATUS_SUCCESS,
        )
        return results
    except SweepCancelled:
        if report is not None:
            report("Alignment cancelled; matches found so far are kept.", 100,
                   task_state=config.TASK_STATUS_REVOKED)
        return []
    except Exception:
        logger.exception("Multi-server alignment failed")
        report(
            "Alignment failed; check the container logs for details.",
            100,
            task_state=config.TASK_STATUS_FAILURE,
        )
        return []
    finally:
        close_cancel()
        if own_conn:
            db.close()
