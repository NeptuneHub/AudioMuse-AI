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

Every number the dashboard publishes is classified on two axes:

* SCOPE - CATALOG (the deduplicated union of every server, i.e. the ``score``
  table) or SERVER (one music server, i.e. ``music_servers`` + the
  ``track_server_map`` rows pointing at it). Nothing is both.
* CADENCE - SNAPSHOT (precomputed into the ``dashboard_stats`` singleton) or
  LIVE (cheap enough to recompute on every poll). Nothing is in between. The
  SNAPSHOT itself refreshes on two timers that merge into the one blob: the
  cheap FAST metrics every 60s, and the single heavy mood scan hourly.

Main Features:
* Routes: `/` dashboard page and `/api/dashboard/summary`.
* SNAPSHOT tier: the FAST block (CATALOG counts, per-SERVER counts, tempo) via
  ``refresh_dashboard_stats()`` every 60s; the MOOD block (Genres + Moods
  Coverage) via ``refresh_dashboard_mood_stats()`` hourly.
* LIVE tier: workers (Redis), recent tasks and cron (tiny tables) only.
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


def _counted_or_none(cur, sql, params=None):
    # Run a COUNT and return it, or None if the query failed. Never 0-on-error: a
    # transient failure that returned 0 would be published into the snapshot as a
    # real "nothing analyzed" and stay on screen until the next refresh.
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
    # Per-configured-server view of the ANALYZED catalogue: how many analyzed
    # songs are mapped to each server, split into distinct songs and the extra
    # duplicate files that collapse onto a song already counted. Entirely local
    # (a single GROUP BY over track_server_map); it never walks a media server
    # and never scans score. Empty list when the registry table does not exist.
    servers = []
    try:
        if not _table_exists(cur, 'music_servers'):
            return servers
        cur.execute(
            "SELECT ms.server_id, ms.name, ms.server_type, ms.is_default, "
            "COALESCE(m.rows_total, 0), COALESCE(m.unique_songs, 0) "
            "FROM music_servers ms LEFT JOIN "
            "(SELECT server_id, COUNT(*) AS rows_total, "
            "COUNT(DISTINCT item_id) AS unique_songs "
            "FROM track_server_map GROUP BY server_id) m "
            "ON m.server_id = ms.server_id "
            "ORDER BY ms.is_default DESC, ms.name ASC"
        )
        for r in cur.fetchall():
            rows_total = int(r[4] or 0)
            unique_songs = int(r[5] or 0)
            duplicate_copies = max(rows_total - unique_songs, 0)
            servers.append(
                {
                    'name': r[1],
                    'server_type': r[2],
                    'is_default': bool(r[3]),
                    'unique_songs': unique_songs,
                    'duplicate_copies': duplicate_copies,
                    'resolved': rows_total,
                }
            )
    except Exception as e:
        logger.debug(f"dashboard: music server metrics failed: {e}")
        _safe_rollback(cur)
    return servers


def _collect_fast_metrics(cur):
    # The FAST tier: every aggregate here is a single indexed count/GROUP BY that
    # stays under a few seconds even on a 1M-song library, so it is cheap enough
    # to recompute every 60 seconds. The one expensive scan (mood vectors) lives
    # in _collect_mood_metrics on its own hourly cadence.
    #
    # Every count uses _counted_or_none so a transient DB failure is a None (not
    # a 0): the caller then skips the whole upsert rather than persisting zeros
    # over the last good snapshot.
    #
    # There is deliberately no "musicnn analyzed %" here: a song only enters
    # `score` at analysis time and save_track_analysis_and_embedding writes its
    # `embedding` row in the same transaction, so that ratio is ~100% by
    # construction. The catalogue IS the set of analyzed songs. The per-SERVER
    # breakdown of those analyzed songs lives in _collect_music_server_metrics.
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
        # A genuine subset of the catalogue: CLAP is a separate pass that runs
        # after analysis, so its percentage can really be < 100.
        'clap_indexed': _counted_or_none(cur, "SELECT COUNT(*) FROM clap_embedding"),
    }
    metrics['music_servers'] = _collect_music_server_metrics(cur)
    # Cleared on any query failure below so the caller can refuse to publish a
    # partial snapshot. Popped before serialization.
    metrics['_complete'] = not any(
        metrics[k] is None
        for k in (
            'total_songs', 'distinct_artists', 'distinct_albums',
            'clap_indexed',
        )
    )

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


def _collect_mood_metrics(cur):
    # The SLOW tier: the ONE full-table scan on the dashboard. It streams every
    # score row and parses the mood_vector / other_features text in Python, which
    # is ~tens of seconds on a 1M-song library - far too heavy for the 60s fast
    # cadence, so it runs on its own hourly timer and is merged into the same
    # snapshot blob. Genre and mood distributions barely move minute to minute,
    # so hourly is plenty.
    #
    # Two signals are derived:
    #  - mood_dominant_counts: per-song dominant-label counts -> Genres chart.
    #  - other_feature_totals: emotional mood scores summed across songs
    #    (other_features column) -> Moods Coverage pie.
    # Both columns are the plain `key:value,key:value` text produced by
    # save_track_analysis_and_embedding(), parsed directly (never JSON). A NAMED
    # server-side cursor streams the whole table in chunks so the web process
    # never buffers all rows at once (an unnamed cursor would).
    mood_dominant_counts = {}
    other_feature_totals = {}
    complete = True
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
        complete = False

    # Genre breakdown: dominant-mood counts from mood_vector (genre-like labels).
    top_genre = sorted(mood_dominant_counts.items(), key=lambda kv: kv[1], reverse=True)
    # Moods Coverage: emotional mood vector (other_features):
    # danceable / aggressive / happy / party / relaxed / sad.
    emotional = sorted(other_feature_totals.items(), key=lambda kv: kv[1], reverse=True)
    return {
        'top_genre': [{'label': k, 'count': int(v)} for k, v in top_genre],
        'moods_coverage': [{'label': k, 'score': round(v, 2)} for k, v in emotional],
        '_complete': complete,
    }


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
    """Scheduled rows. Every schedule is CATALOGUE scope: batch work always runs
    against every configured music server, one server at a time, so there is no
    per-row target to report."""
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
      Two tiers, and only two. SNAPSHOT: the whole `content` block (every
      CATALOG aggregate plus the per-SERVER alignment counts) is read from the
      precomputed `dashboard_stats` singleton and is NEVER recomputed on a
      request; `stats_updated_at` says when it was taken. LIVE: workers, recent
      tasks and cron are cheap enough to recompute per request; `generated_at`
      says when. A client must not present a LIVE timestamp over SNAPSHOT data.
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
        # LIVE tier only: three cheap reads. Everything heavy (every CATALOG
        # aggregate and the per-SERVER alignment counts) comes precomputed out of
        # the dashboard_stats singleton, so no scan of `score` can ever land on
        # the request path.
        recent = _collect_task_metrics(cur)
        cron_rows = _collect_cron(cur)
        content, stats_updated_at = _load_dashboard_stats(cur)
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


# The dashboard snapshot refreshes on two cadences that merge into the same
# dashboard_stats blob:
#  - FAST (every 60s): the cheap counts, per-server, and tempo. All stay under a
#    few seconds even on a 1M-song library.
#  - MOOD (hourly): the one full-table mood scan behind the Genres and Moods
#    Coverage charts (~tens of seconds on 1M rows); its distributions barely move
#    minute to minute, so it never needs the fast cadence.
DASHBOARD_REFRESH_INTERVAL_SECONDS = 60
DASHBOARD_MOOD_REFRESH_INTERVAL_SECONDS = 3600


def dashboard_refresh_interval(app):
    return DASHBOARD_REFRESH_INTERVAL_SECONDS


def _merge_dashboard_content(db, content):
    # Merge the given keys into the dashboard_stats singleton with `content ||`,
    # so the fast refresh never clobbers the hourly mood keys and vice versa. The
    # jsonb || is applied inside a single UPDATE, so concurrent fast/mood writes
    # each merge onto the latest committed blob (row lock serializes them).
    cur = db.cursor()
    try:
        try:
            cur.execute(
                f"INSERT INTO dashboard_stats (id, updated_at, content) "
                f"VALUES (1, {UTC_NOW_SQL}, %s::jsonb) "
                f"ON CONFLICT (id) DO UPDATE SET "
                f"updated_at = EXCLUDED.updated_at, "
                f"content = COALESCE(dashboard_stats.content, '{{}}'::jsonb) "
                f"|| EXCLUDED.content",
                (json.dumps(content),),
            )
        except psycopg2.Error as e:
            if getattr(e, 'pgcode', None) == '42P10' or 'ON CONFLICT' in str(e):
                logger.warning(
                    "dashboard_stats upsert fallback due missing unique constraint: %s", e
                )
                _safe_rollback(cur)
                cur.execute("SELECT content FROM dashboard_stats WHERE id = 1")
                row = cur.fetchone()
                merged = dict(row[0]) if row and row[0] else {}
                merged.update(content)
                cur.execute("DELETE FROM dashboard_stats WHERE id = 1")
                cur.execute(
                    f"INSERT INTO dashboard_stats (id, updated_at, content) "
                    f"VALUES (1, {UTC_NOW_SQL}, %s::jsonb)",
                    (json.dumps(merged),),
                )
            else:
                raise
        db.commit()
    finally:
        cur.close()


def _refresh_dashboard_block(app, collect, label):
    # Shared driver for both cadences: collect a block of metrics, refuse to
    # publish a partial one, and merge it into the snapshot blob.
    started = time.time()
    try:
        with app.app_context():
            db = get_db()
            cur = db.cursor(cursor_factory=DictCursor)
            try:
                content = collect(cur)
            finally:
                cur.close()

            if not content.pop('_complete', True):
                logger.warning(
                    "%s refresh skipped: a query failed; keeping the previous snapshot",
                    label,
                )
                return

            _merge_dashboard_content(db, content)
        logger.info("%s refreshed in %.1fs", label, time.time() - started)
    except Exception:
        logger.exception("%s refresh failed", label)


def refresh_dashboard_stats(app):
    # FAST cadence (every 60s): cheap counts, per-server, tempo.
    _refresh_dashboard_block(app, _collect_fast_metrics, "dashboard_stats")


def refresh_dashboard_mood_stats(app):
    # MOOD cadence (hourly): the full-table mood scan behind the Genres and Moods
    # Coverage charts.
    _refresh_dashboard_block(app, _collect_mood_metrics, "dashboard mood stats")
