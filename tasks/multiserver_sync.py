# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The sweep: alignment of the analyzed catalogue with media servers.

The ONLY place that walks a server's whole catalogue to reconcile it with the
analyzed database - analysis never sweeps, it only aligns the tracks it
actually analyzes. A sweep is a pure metadata pass, NEVER a download or an
analysis: track mappings via normalized path, path tail, and metadata tiers,
plus the server's artist links and a set-based catalogue metadata refresh. It
runs automatically the moment a server is added or its matching-relevant
settings change, and from the Align button.

Runs as an RQ job on the high-priority queue (a main task, like analysis and
clustering coordinators, so album jobs on the default queue can never starve
it) and reports progress into ``task_status`` (task type ``server_sweep``);
cancellable via the standard /api/cancel endpoint (cooperative checks).
Unmatched tracks are simply left unmapped - the analyzed catalogue is never
touched or reduced. Already-mapped tracks are skipped, so re-sweeps are
incremental. Catalogue fetches run bound to the target server so its own
library filter applies, and a full-refresh sweep (the add-server and manual
align actions) prunes mappings whose provider track is no longer on that
server - only map rows are removed, never analyzed tracks.

Main Features:
* ``sweep_server`` / ``sweep_all_secondary_servers`` RQ entry points with live
  percentage progress, one-line status, and cooperative cancellation.
* ``fetch_server_catalogue`` / ``prune_stale_mappings`` / ``unmapped_local_count``
  are the public helpers this module owns; the cleaning task reuses them instead
  of re-implementing the fetch and the prune, so the two can never drift apart.
* Zero-download alignment: matching from catalogue metadata only.
* Artist links: each swept server's ``artist_server_map`` rows are upserted
  from its fetched catalogue (legacy ``artist_mapping`` too for the default).
* Catalogue metadata refresh: album, album artist, year and rating are
  batch-updated for every track mapped to the swept server; ``file_path``
  only from the default server (it is the matcher's top-priority tier).
* Lean memory: the fetched target catalogue is condensed into a slim
  CandidateIndex right after the fetch (its metadata staged into a temp
  table), and the local catalogue streams through it in keyset-paginated
  chunks with per-chunk upserts, so the local side is never fully
  materialized.
* ``recover_abandoned_sweeps`` (run by the RQ janitor) revokes sweeps whose RQ
  job died mid-run - e.g. killed by the worker restart a default-server change
  publishes - and enqueues one replacement alignment of all servers, at most
  once per 10 minutes; rows with no RQ job at all (enqueue failures) are left
  to the batch-start cleanup.
* Full-refresh sweeps re-fetch even aligned servers and prune stale mappings so
  per-server counts stay truthful; pruning is skipped when the fetch looks
  partial so a transient provider error never mass-deletes valid mappings.
"""

import json
import logging
import time
import uuid

from psycopg2 import sql as pgsql
from psycopg2.extras import execute_values

from database import connect_raw
from tasks import provider_probe
from tasks.mediaserver import context as ms_context, registry
from tasks.provider_migration_matcher import CandidateIndex

logger = logging.getLogger(__name__)

SWEEP_TASK_TYPE = 'server_sweep'
_PRUNE_MIN_FETCH_RATIO = 0.5
_RQ_ALIVE_STATUSES = ('queued', 'started', 'deferred', 'scheduled')


class SweepCancelled(Exception):
    pass


def _sweep_job_state(task_id):
    """Classify a sweep's RQ job as 'alive', 'dead', or 'missing'.

    'missing' means no RQ job exists under that id at all (a row whose enqueue
    failed); 'dead' means the job exists but is no longer queued or running.
    """
    from rq.job import Job
    from app_helper import redis_conn

    try:
        job = Job.fetch(task_id, connection=redis_conn)
        status = job.get_status(refresh=True)
    except Exception:
        return 'missing'
    value = getattr(status, 'value', None) or str(status)
    return 'alive' if value in _RQ_ALIVE_STATUSES else 'dead'


_recovery_state = {'last': None}


def recover_abandoned_sweeps():
    """Replace alignment sweeps whose RQ job died before finishing.

    A worker restart (for example the one published right after changing the
    default server) can kill a queued or running sweep; RQ later parks the job
    as failed/abandoned while its task_status row stays stuck in PROGRESS and
    the servers it covered are never aligned. Called periodically by the RQ
    janitor: every non-terminal sweep row whose RQ job exists but is dead is
    marked REVOKED and one fresh alignment covering all servers is
    enqueued in their place. Rows with no RQ job at all are skipped:
    enqueue-failed PENDING rows are archived by the batch-start cleanup.
    Recovery is throttled to once per 10 minutes after a replacement is
    enqueued, so a replacement that itself keeps dying (for example OOM during
    the index rebuild) is not revoked and re-enqueued in a tight loop. Returns
    the replacement task id, or None when nothing was recovered. Uses its own
    raw connection so it needs no Flask app context.
    """
    import config
    from app_helper import rq_queue_high

    last = _recovery_state['last']
    if last is not None and time.monotonic() - last < 600:
        return None

    db = connect_raw()
    db.autocommit = True
    try:
        cur = db.cursor()
        try:
            cur.execute(
                "SELECT task_id FROM task_status WHERE task_type = %s "
                "AND status NOT IN (%s, %s, %s)",
                (SWEEP_TASK_TYPE, config.TASK_STATUS_SUCCESS,
                 config.TASK_STATUS_FAILURE, config.TASK_STATUS_REVOKED),
            )
            candidates = [r[0] for r in cur.fetchall()]
        finally:
            cur.close()
        stale = [task_id for task_id in candidates if _sweep_job_state(task_id) == 'dead']
        if not stale:
            return None

        now = time.time()
        message = (
            "Alignment was interrupted (worker restarted); "
            "a fresh alignment of all servers was enqueued."
        )
        details = json.dumps({'message': message, 'status_message': message})
        cur = db.cursor()
        try:
            cur.execute(
                "UPDATE task_status SET status = %s, progress = 100, details = %s, "
                "timestamp = NOW(), end_time = COALESCE(end_time, %s) "
                "WHERE task_id = ANY(%s)",
                (config.TASK_STATUS_REVOKED, details, now, stale),
            )
            new_task_id = str(uuid.uuid4())
            queued = json.dumps({
                'message': 'Server alignment queued for all servers.',
            })
            cur.execute(
                "INSERT INTO task_status "
                "(task_id, task_type, status, progress, details, timestamp, start_time) "
                "VALUES (%s, %s, %s, 0, %s, NOW(), %s) "
                "ON CONFLICT (task_id) DO NOTHING",
                (new_task_id, SWEEP_TASK_TYPE, config.TASK_STATUS_PENDING, queued, now),
            )
        finally:
            cur.close()
        rq_queue_high.enqueue(
            'tasks.multiserver_sync.sweep_all_secondary_servers',
            kwargs={'task_id': new_task_id},
            job_id=new_task_id,
            job_timeout=-1,
        )
        _recovery_state['last'] = time.monotonic()
        logger.warning(
            "Recovered %d interrupted alignment sweep(s); enqueued replacement %s",
            len(stale), new_task_id,
        )
        return new_task_id
    finally:
        try:
            db.close()
        except Exception:
            logger.debug("Recovery connection close failed", exc_info=True)


def _make_reporter(task_id, label):
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
        try:
            with app.app_context():
                save_task_status(
                    task_id,
                    SWEEP_TASK_TYPE,
                    task_state or TASK_STATUS_PROGRESS,
                    progress=pct,
                    details={
                        'status_message': message,
                        'message': message,
                        'log': [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"],
                    },
                )
        except Exception:
            logger.debug("Sweep status update failed (ignored)", exc_info=True)

    return report


def _make_cancel_check(task_id):
    """Cooperative cancellation: raises SweepCancelled once /api/cancel marked
    the task REVOKED. Uses its own autocommit connection so it always sees the
    latest status, throttled to one DB read every 2 seconds."""
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
        if row and row[0] == config.TASK_STATUS_REVOKED:
            raise SweepCancelled()

    def close():
        if check_conn is not None:
            try:
                check_conn.close()
            except Exception:
                logger.debug("Sweep cancel-check connection close failed", exc_info=True)

    return check, close


def _resolve_task_id(task_id):
    if task_id:
        return task_id
    try:
        from rq import get_current_job
        job = get_current_job()
        if job is not None:
            return job.id
    except Exception:
        logger.debug("No RQ job context for sweep task id", exc_info=True)
    return str(uuid.uuid4())


def unmapped_local_count(conn, server_id):
    """How many analyzed tracks still lack a mapping for ``server_id``.

    Already-mapped tracks are aligned and never reconsidered, so a sweep over an
    aligned server is a no-op and the end-of-analysis alignment only processes
    the newly analyzed songs.
    """
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
    """Yield the still-unmapped analyzed tracks in bounded-memory chunks.

    Keyset pagination on item_id keeps each page cheap and survives the
    per-chunk commits the caller performs between pages, so the whole local
    catalogue is never materialized at once.
    """
    last_id = ''
    while True:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT s.item_id, s.title, s.author, s.album, s.album_artist, s.file_path "
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


def _write_matches(db, server_id, result):
    mapping = {
        new_id: (item_id, result['match_tiers'].get(item_id))
        for item_id, new_id in result['matches'].items()
    }
    return registry.upsert_track_maps(server_id, mapping, conn=db)


def prune_stale_mappings(db, server_id, present_ids):
    """Remove map rows whose provider track is no longer on (or is filtered out
    of) the server. Only track_server_map shrinks; the catalogue never does.
    Skipped entirely when the fetch produced nothing or looks partial (fewer
    tracks than half the existing mappings), so a fetch problem can never wipe
    a server's mappings."""
    present = [(pid,) for pid in present_ids]
    if not present:
        return 0
    cur = db.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM track_server_map WHERE server_id = %s", (server_id,)
        )
        current = cur.fetchone()[0]
        if current > 0 and len(present) < current * _PRUNE_MIN_FETCH_RATIO:
            logger.warning(
                "Multi-server sweep for server %s: fetch returned %d tracks but %d "
                "mappings exist; fetch looks partial, pruning skipped",
                server_id, len(present), current,
            )
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
    """Persist the server's own catalogue size (from the sweep fetch) so the
    dashboard can report alignment against the server's real library instead of
    the union catalogue."""
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


def _collect_artist_maps(tracks):
    maps = {}
    for t in tracks:
        name = t.get('artist') or t.get('album_artist')
        artist_id = t.get('artist_id')
        if name and artist_id:
            maps[str(name)] = str(artist_id)
    return maps


def _write_artist_maps(db, server, artist_maps):
    """Upsert the server's artist links from its fetched catalogue.

    The same registry write path analysis uses at analyze time; for the default
    server the legacy ``artist_mapping`` fallback table is refreshed too.
    """
    if not artist_maps:
        return 0
    try:
        written = registry.upsert_artist_maps(server['server_id'], artist_maps, conn=db)
    except Exception:
        logger.exception("Artist map upsert failed for server %s", server['server_id'])
        return 0
    if server.get('is_default'):
        try:
            from flask_app import app
            from app_helper_artist import upsert_artist_mappings

            with app.app_context():
                upsert_artist_mappings(artist_maps.items())
        except Exception:
            logger.exception("Legacy artist mapping refresh failed for the default server")
    return written


_META_FIELDS = ('album', 'album_artist', 'year', 'rating')


def _stage_track_metadata(db, tracks):
    """Stage the fetched catalogue's metadata in a temp table for the batch
    refresh that runs after matching, so nothing is retained in Python."""
    rows = {}
    for t in tracks:
        provider_id = t.get('id')
        if not provider_id:
            continue
        rows[str(provider_id)] = (
            str(provider_id), t.get('album'), t.get('album_artist'),
            t.get('year'), t.get('rating'), t.get('path'),
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


def _refresh_mapped_metadata(db, server_id, is_default):
    """Batch-refresh catalogue metadata for every track mapped to this server.

    ``file_path`` is written ONLY by the default server: the shared row holds a
    single path and that path is the matcher's top-priority tier, so a secondary
    server must never stamp its own layout onto it.
    """
    fields = _META_FIELDS + (('file_path',) if is_default else ())
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
    # N provider tracks may map to one item_id (duplicate files); collapse to one
    # metadata source per item so the UPDATE is deterministic (matters only for
    # file_path - all duplicates are the same audio).
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
    """Every track one server exposes, bound to it so its library filter applies.

    The ONE full-catalogue enumeration: the sweep matches against it and cleaning
    prunes against it, so the two can never disagree about what a server holds.
    ``server`` may be None (the legacy config default), which binds nothing.

    Binds the registry row it was handed rather than re-resolving it through
    ``registry.bind``: this runs on a worker with no Flask app context, and
    ``context_for`` would go to ``get_db()`` for a row the caller already has.
    """
    import config

    stype = server['server_type'] if server else config.MEDIASERVER_TYPE
    creds = server['creds'] if server else None
    with ms_context.use_server(server):
        return provider_probe.fetch_all_tracks(stype, creds, apply_filter=True)


def _sweep_one(server, db, report, base, span, cancel, full_refresh=False):
    stype = server['server_type']
    server_id = server['server_id']
    total_local = _local_track_count(db)
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
    index = CandidateIndex(
        t for t in target_tracks
        if t.get('id') and str(t.get('id')) not in already_mapped
    )
    target_tracks = None
    _store_server_track_count(db, server_id, target_total)
    pruned = 0
    if full_refresh:
        pruned = prune_stale_mappings(db, server_id, present_ids)
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
    # {provider_track_id: tier_rank} - shared across chunks so one provider track
    # never maps to two canonical rows, and a later, stronger match can take it.
    claimed = {}
    if index.size:
        for chunk in _iter_unmapped_local_rows(db, server_id):
            cancel()
            result = index.match_chunk(chunk, claimed)
            written += _write_matches(db, server_id, result)
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
    refreshed = _refresh_mapped_metadata(db, server_id, bool(server.get('is_default')))
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
        'artists': artists_written,
        'refreshed': refreshed,
        'tier_counts': tier_counts,
    }


def sweep_server(server_id, task_id=None, conn=None):
    """Match the local library against any configured server and store mappings."""
    import config

    task_id = _resolve_task_id(task_id)
    own_conn = conn is None
    db = conn or connect_raw()
    report = _make_reporter(task_id, server_id)
    cancel, close_cancel = _make_cancel_check(task_id)
    try:
        from config import TASK_STATUS_STARTED, TASK_STATUS_SUCCESS

        server = registry.get_server(server_id, conn=db)
        if server is None:
            report("Server no longer exists; nothing to align.", 100, task_state=TASK_STATUS_SUCCESS)
            return {'server_id': server_id, 'skipped': 'deleted', 'matched': 0}

        report(f"Starting alignment for {server['name']}...", 2, task_state=TASK_STATUS_STARTED)
        cancel()
        summary = _sweep_one(server, db, report, 5, 95, cancel, full_refresh=True)
        if summary.get('aligned'):
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
    """Align configured servers, optionally limited to ``server_ids``.

    ``server_ids=None`` means every server; an explicit EMPTY list is a no-op,
    never a sweep-everything. ``full_refresh`` defaults to True for unfiltered
    (manual/setup) sweeps so aligned servers are still re-fetched and their
    stale mappings pruned; callers passing an explicit ``server_ids`` subset
    get matching only.
    """
    import config

    if full_refresh is None:
        full_refresh = server_ids is None

    task_id = _resolve_task_id(task_id)
    own_conn = conn is None
    db = conn or connect_raw()
    report = _make_reporter(task_id, 'all')
    cancel, close_cancel = _make_cancel_check(task_id)
    try:
        from config import TASK_STATUS_STARTED, TASK_STATUS_SUCCESS

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
        matched = sum(r.get('matched', 0) for r in results)
        report(
            f"Alignment complete for {len(servers)} server(s); {matched} track mappings written.",
            100, task_state=TASK_STATUS_SUCCESS,
        )
        return results
    except SweepCancelled:
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
