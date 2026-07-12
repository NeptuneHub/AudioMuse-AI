# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Align the analyzed catalogue with secondary media servers.

Runs as an RQ job and reports progress into ``task_status`` (task type
``server_sweep``) so the setup wizard can show a live bar; it is cancellable via
the standard /api/cancel endpoint (cooperative checks). The sweep NEVER analyzes
or downloads songs. It first ensures the catalogue uses the canonical
Chromaprint-hash ids (a database relabel after analysis has backfilled legacy
Chromaprints, so a
legacy install is fixed the moment a second server is added), then matches each
secondary server's catalogue against the still-unmapped analyzed tracks by
normalized path, path tail, and metadata, storing confident pairs in
track_server_map. Unmatched secondary tracks are simply left unmapped - the
default server's catalogue is never touched or reduced. Already-mapped tracks
are skipped, so re-sweeps are incremental and an aligned server is a no-op.

Main Features:
* ``sweep_server`` / ``sweep_all_secondary_servers`` RQ entry points with live
  percentage progress, one-line status, and cooperative cancellation.
* Zero-download alignment: canonical ids from stored embeddings, matching from
  catalogue metadata only.
"""

import logging
import time
import uuid

from database import connect_raw
from tasks import provider_probe
from tasks.mediaserver import registry
from tasks.provider_migration_matcher import match_tracks

logger = logging.getLogger(__name__)

SWEEP_TASK_TYPE = 'server_sweep'


class SweepCancelled(Exception):
    pass


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


def _canonicalize_catalog(report, base, span):
    """Ensure item_ids with stored Chromaprints use canonical hash ids.

    The relabel itself is pure DB work (seconds), but the first run on a legacy
    install also rebuilds every similarity index, which takes a while on a large
    library - so each canonicalization/rebuild step advances the progress bar
    through the ``base``..``base+span`` window."""
    from tasks.fingerprint_canonicalize import canonicalize_fingerprinted_ids

    step = {'n': 0}

    def log_fn(message, _progress=None):
        step['n'] += 1
        pct = min(base + span * 0.1 + step['n'] * (span * 0.85 / 12), base + span * 0.95)
        report(message, pct)

    report("Checking catalogue ids (embedding hash)...", base)
    try:
        result = canonicalize_fingerprinted_ids(rebuild=True, log_fn=log_fn)
        relabelled = result.get('relabelled', 0) if isinstance(result, dict) else 0
        if relabelled:
            report(
                f"Relabelled {relabelled} tracks to canonical ids and rebuilt the indexes.",
                base + span,
            )
        else:
            report("Catalogue ids already canonical.", base + span)
    except Exception:
        logger.exception("Canonicalization before sweep failed; matching continues with current ids")


def _unmapped_local_rows(conn, server_id):
    """Analyzed tracks with no mapping for ``server_id`` yet - the only work left.

    Already-mapped tracks are aligned and never reconsidered, so a sweep over an
    aligned server is a no-op and the end-of-analysis alignment only processes
    the newly analyzed songs.
    """
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT s.item_id, s.title, s.author, s.album, s.album_artist, s.file_path "
            "FROM score s WHERE NOT EXISTS ("
            "SELECT 1 FROM track_server_map m WHERE m.item_id = s.item_id AND m.server_id = %s)",
            (server_id,),
        )
        return [
            {
                'item_id': r[0],
                'title': r[1],
                'author': r[2],
                'album': r[3],
                'album_artist': r[4],
                'file_path': r[5],
            }
            for r in cur.fetchall()
        ]
    finally:
        cur.close()


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
        item_id: (new_id, result['match_tiers'].get(item_id))
        for item_id, new_id in result['matches'].items()
    }
    return registry.upsert_track_maps(server_id, mapping, conn=db)


def _sweep_one(server, db, report, base, span, cancel):
    stype = server['server_type']
    server_id = server['server_id']
    total_local = _local_track_count(db)
    unmapped = _unmapped_local_rows(db, server_id)
    if not unmapped:
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
    target_tracks = provider_probe.fetch_all_tracks(stype, server['creds'])
    cancel()

    already_mapped = _already_mapped_ids(db, server_id)
    candidates = [
        t for t in target_tracks if t.get('id') and str(t.get('id')) not in already_mapped
    ]
    report(
        f"Aligning {server['name']}: {len(unmapped)} tracks to match "
        f"({total_local - len(unmapped)} already aligned)...",
        base + span * 0.6,
    )
    result = match_tracks(unmapped, candidates)
    written = _write_matches(db, server_id, result)
    logger.info(
        "Multi-server sweep for '%s': mapped %d/%d unmapped tracks (target=%d, tiers=%s)",
        server['name'], written, len(unmapped), len(target_tracks), result['tier_counts'],
    )
    return {
        'server_id': server_id,
        'name': server['name'],
        'server_type': stype,
        'target_tracks': len(target_tracks),
        'local_tracks': total_local,
        'unmapped': len(unmapped),
        'matched': written,
        'tier_counts': result['tier_counts'],
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
        if not server['enabled']:
            report("Server is disabled; skipping sweep.", 100, task_state=TASK_STATUS_SUCCESS)
            return {'server_id': server_id, 'skipped': 'disabled', 'matched': 0}

        report(f"Starting alignment for {server['name']}...", 2, task_state=TASK_STATUS_STARTED)
        _canonicalize_catalog(report, 5, 55)
        cancel()
        summary = _sweep_one(server, db, report, 60, 40, cancel)
        if summary.get('aligned'):
            message = f"{server['name']} is already aligned; nothing to do."
        else:
            message = (
                f"Alignment complete: {summary['matched']}/{summary['unmapped']} pending tracks "
                f"matched on {server['name']}."
            )
        report(message, 100, task_state=TASK_STATUS_SUCCESS)
        return summary
    except SweepCancelled:
        report("Alignment cancelled; matches found so far are kept.", 100,
               task_state=config.TASK_STATUS_REVOKED)
        return {'server_id': server_id, 'cancelled': True}
    except Exception:
        logger.exception("Multi-server sweep failed for server %s", server_id)
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


def sweep_all_secondary_servers(task_id=None, conn=None, server_ids=None):
    """Align enabled servers, optionally limited to ``server_ids``.

    The optional filter lets union analysis align only sources that have not had
    their own analysis phase yet. Existing callers with no filter still sweep all.
    """
    import config

    task_id = _resolve_task_id(task_id)
    own_conn = conn is None
    db = conn or connect_raw()
    report = _make_reporter(task_id, 'all')
    cancel, close_cancel = _make_cancel_check(task_id)
    try:
        from config import TASK_STATUS_STARTED, TASK_STATUS_SUCCESS

        selected = {str(server_id) for server_id in server_ids} if server_ids else None
        servers = [
            s for s in registry.list_servers(conn=db)
            if s['enabled'] and (selected is None or s['server_id'] in selected)
        ]
        report(
            f"Starting alignment for {len(servers)} selected server(s)...",
            2, task_state=TASK_STATUS_STARTED,
        )
        _canonicalize_catalog(report, 5, 55)
        cancel()
        if not servers:
            report("Catalogue ids up to date; no selected servers to align.", 100,
                   task_state=TASK_STATUS_SUCCESS)
            return []

        span = 40 / len(servers)
        results = []
        for i, server in enumerate(servers):
            try:
                results.append(_sweep_one(server, db, report, 60 + i * span, span, cancel))
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
    finally:
        close_cancel()
        if own_conn:
            db.close()
