# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for the dashboard landing page.

Serves the `/` home page and its summary API, showing recent activity, content
metrics, index counts, active workers, and scheduled tasks.

Main Features:
* Routes: `/` dashboard page and `/api/dashboard/summary`.
* Heavy library aggregates are NOT recomputed per request: they are refreshed
  by ``refresh_dashboard_stats()`` at startup and hourly into the singleton
  ``dashboard_stats`` row, which the summary reads alongside the cheap live bits
  (workers, recent tasks, cron).
"""

import json
import logging
import time
import psycopg2
from flask import Blueprint, render_template, jsonify
from psycopg2.extras import DictCursor

from database import get_db
from taskqueue import redis_conn
from tz_helper import LOCAL_TZ_FMT, UTC_NOW_SQL, to_local_str

logger = logging.getLogger(__name__)
dashboard_bp = Blueprint('dashboard_bp', __name__)

# Short-lived memo for the per-server alignment metrics: every open dashboard
# tab polls /api/dashboard/summary every ~30s and the metrics cost seconds of
# GROUP BY / NOT EXISTS work on large libraries, so one computation is shared
# instead of recomputed per request. The TTL is above the 30s poll so two
# consecutive polls of a single tab actually hit the memo.
_MUSIC_SERVER_METRICS_MEMO = {'ts': 0.0, 'data': None}
_MUSIC_SERVER_METRICS_TTL_SECONDS = 35.0
# Per-server flag set once the default server's legacy NOT-EXISTS count hits 0:
# it only ever shrinks after canonicalization, so the scan can be skipped then.
_LEGACY_UNMAPPED_DONE = {}


@dashboard_bp.route('/')
def dashboard_page():
    """
    Dashboard home page.
    ---
    tags:
      - Dashboard
    summary: HTML landing page rendering the AudioMuse-AI dashboard.
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template('dashboard.html', title='AudioMuse-AI - Dashboard', active='dashboard')


def _safe_rollback(cur):
    """Best-effort rollback on the connection backing this cursor so the next
    query doesn't fail with 'current transaction is aborted'."""
    try:
        cur.connection.rollback()
    except Exception:
        pass


def _safe_count(cur, sql, params=None):
    try:
        cur.execute(sql, params or ())
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception as e:
        logger.debug(f"dashboard count failed for [{sql}]: {e}")
        _safe_rollback(cur)
        return 0


def _counted_or_none(cur, sql, params=None):
    """Like ``_safe_count`` but returns None on failure instead of 0, so a
    caller can tell "counted zero" apart from "the query did not run"."""
    try:
        cur.execute(sql, params or ())
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception as e:
        logger.debug(f"dashboard count failed for [{sql}]: {e}")
        _safe_rollback(cur)
        return None


def _table_exists(cur, name):
    try:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (name,),
        )
        row = cur.fetchone()
        return bool(row and row[0])
    except Exception:
        _safe_rollback(cur)
        return False


def _get_gmm_index_count():
    try:
        from tasks.artist_gmm_manager import artist_map, artist_index

        if artist_map is not None:
            return len(artist_map)
        if artist_index is not None:
            return getattr(artist_index, 'num_elements', 0)
    except Exception:
        pass
    return 0


def _collect_workers():
    """Return basic info about RQ workers. Only the columns rendered in
    the Workers table of the dashboard are populated."""
    workers_info = []
    try:
        from rq import Worker

        workers = Worker.all(connection=redis_conn)
        for w in workers:
            try:
                state = w.get_state()
            except Exception:
                state = 'unknown'
            try:
                current_job = w.get_current_job()
                current_job_id = current_job.id if current_job else None
            except Exception:
                current_job_id = None
            workers_info.append(
                {
                    'hostname': getattr(w, 'hostname', None),
                    'queues': [q.name for q in getattr(w, 'queues', [])],
                    'state': state,
                    'current_job_id': current_job_id,
                    'successful_jobs': getattr(w, 'successful_job_count', 0),
                    'failed_jobs': getattr(w, 'failed_job_count', 0),
                }
            )
    except Exception as e:
        logger.warning(f"dashboard: failed to enumerate RQ workers: {e}")
    return workers_info


def _collect_task_metrics(cur):
    """Return the 10 most recent main tasks for the Recent Activity table."""
    recent = []
    if _table_exists(cur, 'task_history'):
        try:
            cur.execute("""
                SELECT task_id, task_type, status, duration_seconds, note, recorded_at
                FROM task_history
                WHERE task_type IS NOT NULL
                  AND task_type <> ''
                  AND task_type <> 'unknown'
                ORDER BY recorded_at DESC, id DESC
                LIMIT 10
            """)
            for r in cur.fetchall():
                recent.append(
                    {
                        'task_id': r['task_id'],
                        'task_type': r['task_type'],
                        'status': r['status'],
                        'duration_seconds': float(r['duration_seconds'])
                        if r['duration_seconds'] is not None
                        else None,
                        'note': r['note'] or '',
                        'timestamp': to_local_str(r['recorded_at']),
                    }
                )
        except Exception as e:
            logger.debug(f"dashboard: task_history query failed: {e}")
            _safe_rollback(cur)
    return recent


def _collect_music_server_metrics(cur):
    """Per-configured-server alignment status for the Music Server Status
    section. Servers hold DIFFERENT catalogues that may only partially overlap,
    so each server is measured against its OWN library size (``track_count``,
    captured by the last alignment sweep), never against the union catalogue.

    A server may hold several duplicate FILES of one song (many provider tracks,
    one canonical id), so the counts are split:
      ``unique_songs``     - distinct catalogue songs on the server
                             (COUNT(DISTINCT item_id); the default also counts
                             legacy provider-keyed rows predating canonicalization).
      ``duplicate_copies`` - extra provider files that collapse onto a song
                             already counted (COUNT(*) - COUNT(DISTINCT item_id)).
      ``resolved``         - provider tracks with a map row (the coverage numerator).
    ``server_songs`` is None until a sweep has fetched the server's catalogue at
    least once. ``catalogue_songs`` carries the union total for the section caption.
    Not part of the hourly stats cache: ``dashboard_summary`` refreshes it via a
    short memo so new servers and running sweeps show up quickly without redoing
    the aggregate per request. Empty list when the registry tables do not exist
    yet or only one server is configured (the section is hidden then)."""
    servers = []
    try:
        if not _table_exists(cur, 'music_servers'):
            return servers
        catalogue_total = _safe_count(cur, "SELECT COUNT(*) FROM score")
        cur.execute(
            "SELECT ms.server_id, ms.name, ms.server_type, ms.is_default, "
            "ms.track_count, COALESCE(m.rows_total, 0), COALESCE(m.unique_songs, 0) "
            "FROM music_servers ms LEFT JOIN "
            "(SELECT server_id, COUNT(*) AS rows_total, "
            "COUNT(DISTINCT item_id) AS unique_songs "
            "FROM track_server_map GROUP BY server_id) m "
            "ON m.server_id = ms.server_id "
            "ORDER BY ms.is_default DESC, ms.name ASC"
        )
        for r in cur.fetchall():
            rows_total = int(r[5] or 0)
            unique_songs = int(r[6] or 0)
            duplicate_copies = max(rows_total - unique_songs, 0)
            legacy = 0
            if r[3] and not _LEGACY_UNMAPPED_DONE.get(r[0]):
                # Legacy rows keep their provider id and are implicitly on the
                # default server until canonicalization maps them explicitly.
                # They are distinct items with no map row, so they add to both
                # unique_songs and resolved without adding duplicates.
                counted = _counted_or_none(
                    cur,
                    "SELECT COUNT(*) FROM score s "
                    "WHERE s.item_id NOT LIKE 'fp\\_%%' AND NOT EXISTS ("
                    "SELECT 1 FROM track_server_map m "
                    "WHERE m.item_id = s.item_id AND m.server_id = %s)",
                    (r[0],),
                )
                # Only a real zero retires the scan. A failed query must not be
                # read as "no legacy rows left", or the count stays wrong for
                # the life of the process.
                if counted == 0:
                    _LEGACY_UNMAPPED_DONE[r[0]] = True
                legacy = counted or 0
            unique_songs += legacy
            resolved = rows_total + legacy
            server_songs = r[4]
            if server_songs is None and r[3] and resolved:
                # The default server's library defined the catalogue before the
                # first sweep stored its real size, so resolved = its library.
                server_songs = resolved
            servers.append(
                {
                    'name': r[1],
                    'server_type': r[2],
                    'is_default': bool(r[3]),
                    'server_songs': int(server_songs) if server_songs is not None else None,
                    'unique_songs': unique_songs,
                    'duplicate_copies': duplicate_copies,
                    'resolved': resolved,
                    'catalogue_songs': catalogue_total,
                }
            )
    except Exception as e:
        logger.debug(f"dashboard: music server metrics failed: {e}")
        _safe_rollback(cur)
    return servers


def _collect_content_metrics(cur):
    # Core library counts use _counted_or_none so a transient DB failure is a
    # None (not a 0): refresh_dashboard_stats skips the whole upsert rather than
    # persisting zeros over the last good snapshot for an hour.
    metrics = {
        'total_songs': _counted_or_none(cur, "SELECT COUNT(*) FROM score"),
        'distinct_artists': _counted_or_none(
            cur,
            "SELECT COUNT(DISTINCT author) FROM score "
            "WHERE author IS NOT NULL AND author <> ''",
        ),
        # Album identity is (album_artist, album), matching the migration wizard
        # and idx_score_album_artist_album; a bare title collapses "Greatest Hits"
        # across artists into one. Fall back to author when album_artist is unset
        # (rows written before the column existed).
        'distinct_albums': _counted_or_none(
            cur,
            "SELECT COUNT(*) FROM (SELECT DISTINCT "
            "COALESCE(NULLIF(album_artist, ''), author) AS aa, album FROM score "
            "WHERE album IS NOT NULL AND album <> '') t",
        ),
        # Analyzed-row counts from the DB, not in-process index globals: the
        # searchable index lives in the worker, so the web process's copy is
        # stale or zero, and it is what these two tables are built from anyway.
        'musicnn_indexed': _safe_count(
            cur, "SELECT COUNT(*) FROM embedding WHERE embedding IS NOT NULL"
        ),
        'clap_indexed': _safe_count(cur, "SELECT COUNT(*) FROM clap_embedding"),
        'gmm_indexed': _get_gmm_index_count(),
    }
    # Cleared on any query failure below so the caller can refuse to publish a
    # partial snapshot. Popped before serialization.
    metrics['_complete'] = not any(
        metrics[k] is None
        for k in ('total_songs', 'distinct_artists', 'distinct_albums')
    )

    # Parse mood vectors into the two signals the dashboard renders:
    #  - mood_dominant_counts: per-song dominant-label counts -> Genres chart.
    #  - other_feature_totals: emotional mood scores summed across songs
    #    (other_features column) -> Moods Coverage pie.
    # Both columns are the plain `key:value,key:value` text produced by
    # save_track_analysis_and_embedding(), parsed directly (never JSON). A NAMED
    # server-side cursor streams the whole table in chunks so the web process
    # never buffers all ~180k rows at once (an unnamed cursor would).
    mood_dominant_counts = {}
    other_feature_totals = {}
    try:
        with cur.connection.cursor(name='dash_mood_scan') as scan:
            scan.itersize = 20000
            scan.execute(
                "SELECT mood_vector, other_features FROM score "
                "WHERE mood_vector IS NOT NULL AND mood_vector <> ''"
            )
            for mv, of in scan:
                if not mv:
                    continue
                parsed = _parse_keyval(mv)
                if not parsed:
                    continue
                dom = max(parsed.items(), key=lambda kv: kv[1])[0]
                mood_dominant_counts[dom] = mood_dominant_counts.get(dom, 0) + 1

                if of:
                    of_parsed = _parse_keyval(of)
                    for k, s in of_parsed.items():
                        if k in ('tempo_normalized', 'energy_normalized'):
                            continue
                        other_feature_totals[k] = other_feature_totals.get(k, 0.0) + s
    except Exception as e:
        logger.debug(f"dashboard: mood aggregation failed: {e}")
        _safe_rollback(cur)
        metrics['_complete'] = False

    # Genre breakdown: dominant-mood counts from mood_vector (genre-like labels).
    top_genre = sorted(mood_dominant_counts.items(), key=lambda kv: kv[1], reverse=True)
    metrics['top_genre'] = [{'label': k, 'count': int(v)} for k, v in top_genre]
    # Moods Coverage: emotional mood vector (other_features):
    # danceable / aggressive / happy / party / relaxed / sad.
    emotional = sorted(other_feature_totals.items(), key=lambda kv: kv[1], reverse=True)
    metrics['moods_coverage'] = [{'label': k, 'score': round(v, 2)} for k, v in emotional]

    # Tempo profile: bucket songs into slow/medium/fast/very-fast. Always
    # populate the key so the UI can render a real (possibly-zero) chart
    # rather than the "still collecting" placeholder when no songs have
    # a tempo yet.
    metrics['tempo_profile'] = {
        'slow': 0,
        'medium': 0,
        'fast': 0,
        'very_fast': 0,
        'avg_tempo': None,
    }
    try:
        cur.execute(
            "SELECT "
            "  COUNT(*) FILTER (WHERE tempo > 0 AND tempo < 85) AS slow, "
            "  COUNT(*) FILTER (WHERE tempo >= 85 AND tempo < 110) AS medium, "
            "  COUNT(*) FILTER (WHERE tempo >= 110 AND tempo < 140) AS fast, "
            "  COUNT(*) FILTER (WHERE tempo >= 140) AS very_fast, "
            "  AVG(tempo) FILTER (WHERE tempo > 0) AS avg_tempo "
            "FROM score WHERE tempo IS NOT NULL"
        )
        r = cur.fetchone()
        if r:
            metrics['tempo_profile'] = {
                'slow': int(r[0] or 0),
                'medium': int(r[1] or 0),
                'fast': int(r[2] or 0),
                'very_fast': int(r[3] or 0),
                'avg_tempo': round(float(r[4]), 1) if r[4] is not None else None,
            }
    except Exception as e:
        logger.warning(f"dashboard: tempo profile query failed: {e}", exc_info=True)
        _safe_rollback(cur)
        metrics['_complete'] = False

    return metrics


def _parse_keyval(s):
    """Parse a ``key:value,key:value`` string (as stored in the ``score``
    table's ``mood_vector`` / ``other_features`` columns) into a dict of
    ``{label: float}``. Invalid pairs are silently skipped. Designed to
    be fast on large libraries: no JSON parsing, no per-pair try/except
    on the hot path for well-formed values.
    """
    out = {}
    if not s:
        return out
    for part in s.split(','):
        # Use partition (fast, no regex) and tolerate leading/trailing
        # whitespace on the key only.
        k, sep, v = part.partition(':')
        if not sep:
            continue
        k = k.strip()
        if not k:
            continue
        try:
            out[k] = float(v)
        except (ValueError, TypeError):
            # Malformed numeric field - skip silently.
            continue
    return out


def _collect_cron(cur):
    rows = []
    try:
        cur.execute("""
            SELECT id, name, task_type, cron_expr, enabled, last_run
            FROM cron
            ORDER BY enabled DESC, id ASC
        """)
        for r in cur.fetchall():
            last_run_iso = None
            try:
                if r['last_run']:
                    last_run_iso = time.strftime(LOCAL_TZ_FMT, time.localtime(float(r['last_run'])))
            except Exception:
                pass
            rows.append(
                {
                    'id': r['id'],
                    'name': r['name'],
                    'task_type': r['task_type'],
                    'cron_expr': r['cron_expr'],
                    'enabled': bool(r['enabled']),
                    'last_run': last_run_iso,
                }
            )
    except Exception as e:
        logger.debug(f"dashboard: cron query failed: {e}")
        _safe_rollback(cur)
    return rows


@dashboard_bp.route('/api/dashboard/summary', methods=['GET'])
def dashboard_summary():
    """
    Dashboard summary payload.
    ---
    tags:
      - Dashboard
    summary: Aggregated dashboard data - library stats, worker status, recent tasks, cron entries.
    description: |
      Heavy library aggregates (the `content` block) are read from the
      precomputed `dashboard_stats` singleton row and NOT recomputed on each
      request. Everything else (workers, recent tasks, cron) is cheap and
      stays live.
    responses:
      200:
        description: Dashboard payload.
        content:
          application/json:
            schema:
              type: object
              properties:
                generated_at:
                  type: string
                stats_updated_at:
                  type: string
                workers:
                  type: array
                  items:
                    type: object
                content:
                  type: object
                recent_tasks:
                  type: array
                  items:
                    type: object
                cron:
                  type: array
                  items:
                    type: object
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    try:
        recent = _collect_task_metrics(cur)
        cron_rows = _collect_cron(cur)
        content, stats_updated_at = _load_dashboard_stats(cur)
        # Server alignment counts change while sweeps run, so they bypass the
        # hourly cache; a short memo still shares one computation across the
        # 30s auto-refresh of every open tab instead of redoing the aggregates
        # per request. The frontend hides the whole section for fewer than two
        # servers, so single-server installs skip the aggregate entirely.
        content = dict(content or {})
        if _safe_count(cur, "SELECT COUNT(*) FROM music_servers") <= 1:
            content['music_servers'] = []
        else:
            now = time.monotonic()
            memo = _MUSIC_SERVER_METRICS_MEMO
            if memo['data'] is not None and now - memo['ts'] < _MUSIC_SERVER_METRICS_TTL_SECONDS:
                content['music_servers'] = memo['data']
            else:
                metrics = _collect_music_server_metrics(cur)
                memo['ts'] = now
                memo['data'] = metrics
                content['music_servers'] = metrics
    finally:
        cur.close()

    workers = _collect_workers()

    return jsonify(
        {
            'generated_at': time.strftime(LOCAL_TZ_FMT),
            'stats_updated_at': stats_updated_at,
            'workers': workers,
            'recent_tasks': recent,
            'content': content,
            'cron': cron_rows,
        }
    )


def _load_dashboard_stats(cur):
    """Read the singleton dashboard_stats row. Returns (content, updated_at_iso)."""
    try:
        cur.execute("SELECT updated_at, content FROM dashboard_stats WHERE id = 1")
        row = cur.fetchone()
        if not row:
            return {}, None
        content = row['content'] or {}
        return content, to_local_str(row['updated_at'])
    except Exception as e:
        logger.debug(f"dashboard: load_dashboard_stats failed: {e}")
        _safe_rollback(cur)
        return {}, None


def refresh_dashboard_stats(app):
    """Recompute content metrics and upsert them into the
    ``dashboard_stats`` singleton row. Intended to be called from
    a background thread at app startup and then once per hour.

    Runs inside an app context so ``get_db()`` works, and commits the
    result so the new values are visible to subsequent requests.
    """
    started = time.time()
    try:
        with app.app_context():
            db = get_db()
            cur = db.cursor(cursor_factory=DictCursor)
            try:
                content = _collect_content_metrics(cur)
            finally:
                cur.close()

            if not content.pop('_complete', True):
                logger.warning(
                    "dashboard_stats refresh skipped: a core count or scan failed; "
                    "keeping the previous snapshot"
                )
                return

            cur2 = db.cursor()
            try:
                try:
                    cur2.execute(
                        f"INSERT INTO dashboard_stats (id, updated_at, content) "
                        f"VALUES (1, {UTC_NOW_SQL}, %s::jsonb) "
                        f"ON CONFLICT (id) DO UPDATE SET "
                        f"updated_at = EXCLUDED.updated_at, "
                        f"content = EXCLUDED.content",
                        (json.dumps(content),),
                    )
                except psycopg2.Error as e:
                    if getattr(e, 'pgcode', None) == '42P10' or 'ON CONFLICT' in str(e):
                        logger.warning(
                            "dashboard_stats upsert fallback due missing unique constraint: %s", e
                        )
                        _safe_rollback(cur2)
                        cur2.execute("DELETE FROM dashboard_stats WHERE id = 1")
                        cur2.execute(
                            f"INSERT INTO dashboard_stats (id, updated_at, content) "
                            f"VALUES (1, {UTC_NOW_SQL}, %s::jsonb)",
                            (json.dumps(content),),
                        )
                    else:
                        raise
                db.commit()
            finally:
                cur2.close()
        elapsed = time.time() - started
        logger.info(f"dashboard_stats refreshed in {elapsed:.1f}s")
    except Exception:
        logger.exception("refresh_dashboard_stats failed")
