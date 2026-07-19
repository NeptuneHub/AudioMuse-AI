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
  cheap FAST metrics every 60s, and the heavier distribution charts hourly.

Main Features:
* Routes: `/` dashboard page and `/api/dashboard/summary`.
* SNAPSHOT tier: the FAST block (CATALOG counts, per-SERVER counts) via
  ``refresh_dashboard_stats()`` every 60s; the CHARTS block (Genres, Moods
  Coverage, Tempo) via ``refresh_dashboard_charts_stats()`` hourly, carrying its
  own ``charts_updated_at`` stamp so the UI can label it honestly.
* LIVE tier: workers (Redis), recent tasks and cron (tiny tables) only.
"""

import json
import logging
import time
import psycopg2
from flask import Blueprint, render_template, jsonify, request
from psycopg2.extras import DictCursor

import config
from database import get_db
from taskqueue import redis_conn
from tasks.mediaserver import registry
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


# --- Browse view: paginated Song / Artist / Album listing --------------------
# A generic listing the dashboard numbers link into. Every query is LIMIT-bounded
# (page_size + 1, to derive has_more) so NO request can ever return the whole
# catalogue, and the response carries METADATA ONLY (title/author/album, plus the
# per-copy file PATHS for the duplicates view) - never the internal fp_ item_id -
# so there is nothing to leak. Deep pages are clamped by DASHBOARD_BROWSE_MAX_OFFSET
# so a 1M-row table can never be walked end to end.

_BROWSE_KINDS = ('songs', 'artists', 'albums')
_BROWSE_FILTERS = ('all', 'unique', 'duplicates', 'orphan')
_BROWSE_MIN_QUERY = 2


@dashboard_bp.route('/browse')
def browse_page():
    try:
        servers = [
            {'name': s['name'], 'is_default': bool(s['is_default'])}
            for s in registry.list_servers()
        ]
    except Exception:
        logger.debug("browse: could not list servers", exc_info=True)
        servers = []
    return render_template(
        'browse.html', title='AudioMuse-AI - Browse', active='browse',
        browse_servers=servers, page_size=config.DASHBOARD_BROWSE_PAGE_SIZE,
        max_offset=config.DASHBOARD_BROWSE_MAX_OFFSET,
    )


def _browse_like(value):
    # ILIKE contains-pattern with the user's own % / _ / \ escaped, so a search for
    # "50%" matches a literal percent instead of acting as a wildcard.
    escaped = value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    return '%' + escaped + '%'


def _browse_songs_sql(server_id, filt, q):
    params = []
    select = ("SELECT s.title, s.author, s.album, s.album_artist, s.year, "
              "NULL::bigint AS copies")
    frm = " FROM score s"
    where = []
    if filt == 'orphan':
        where.append("NOT EXISTS (SELECT 1 FROM track_server_map t "
                     "WHERE t.item_id = s.item_id)")
    elif filt == 'duplicates':
        # Songs with more than one provider FILE on this server (same
        # duplicate_copies definition the dashboard uses). The per-copy FILE PATHS
        # are returned so the user can see WHAT each copy is and judge whether the
        # merge is a real duplicate or a wrong one - never a provider/fp id, only
        # the on-disk path. Most-duplicated on top. Server is required.
        select = ("SELECT s.title, s.author, s.album, s.album_artist, s.year, "
                  "d.copies, d.files")
        frm = (" FROM (SELECT item_id, COUNT(*) AS copies, "
               "array_agg(COALESCE(NULLIF(file_path, ''), '(no file path)') "
               "ORDER BY file_path NULLS LAST) AS files "
               "FROM track_server_map WHERE server_id = %s "
               "GROUP BY item_id HAVING COUNT(*) > 1) d "
               "JOIN score s ON s.item_id = d.item_id")
        params.append(server_id)
    elif server_id:
        where.append("EXISTS (SELECT 1 FROM track_server_map t "
                     "WHERE t.item_id = s.item_id AND t.server_id = %s)")
        params.append(server_id)
    if q:
        where.append("s.search_u ILIKE %s")
        params.append(_browse_like(q))
    sql = select + frm
    if where:
        sql += " WHERE " + " AND ".join(where)
    # item_id is only a stable tiebreaker for deterministic paging; it is never
    # selected, so it cannot appear in the response.
    if filt == 'duplicates':
        sql += " ORDER BY d.copies DESC, s.author ASC, s.title ASC, s.item_id ASC"
    else:
        sql += " ORDER BY s.author ASC, s.title ASC, s.item_id ASC"
    return sql, params


def _browse_artists_sql(server_id, q):
    params = []
    if server_id:
        author_col = "s.author"
        sql = ("SELECT s.author FROM track_server_map m "
               "JOIN score s ON s.item_id = m.item_id "
               "WHERE m.server_id = %s AND s.author IS NOT NULL AND s.author <> ''")
        params.append(server_id)
    else:
        author_col = "author"
        sql = "SELECT author FROM score WHERE author IS NOT NULL AND author <> ''"
    if q:
        sql += " AND " + author_col + " ILIKE %s"
        params.append(_browse_like(q))
    sql += " GROUP BY 1 ORDER BY 1 ASC"
    return sql, params


def _browse_albums_sql(server_id, q):
    params = []
    aa = "COALESCE(NULLIF(s.album_artist, ''), s.author)"
    if server_id:
        frm = ("FROM track_server_map m JOIN score s ON s.item_id = m.item_id "
               "WHERE m.server_id = %s AND s.album IS NOT NULL AND s.album <> ''")
        params.append(server_id)
    else:
        frm = "FROM score s WHERE s.album IS NOT NULL AND s.album <> ''"
    sql = "SELECT " + aa + " AS aa, s.album " + frm
    if q:
        sql += " AND (s.album ILIKE %s OR " + aa + " ILIKE %s)"
        params.append(_browse_like(q))
        params.append(_browse_like(q))
    sql += " GROUP BY 1, 2 ORDER BY 1 ASC, 2 ASC"
    return sql, params


def _browse_serialize(kind, rows):
    out = []
    if kind == 'artists':
        for r in rows:
            out.append({'artist': r[0]})
    elif kind == 'albums':
        for r in rows:
            out.append({'album_artist': r[0], 'album': r[1]})
    else:
        for r in rows:
            item = {
                'title': r[0], 'author': r[1], 'album': r[2],
                'album_artist': r[3], 'year': r[4],
                'copies': int(r[5]) if r[5] is not None else None,
            }
            # The duplicates query adds a 7th column: the per-copy file paths.
            if len(r) > 6:
                item['files'] = list(r[6]) if r[6] is not None else []
            out.append(item)
    return out


def _browse_total(content, kind, server_id, server_name, filt, has_q):
    # Cheap pager total from the 60s snapshot (the exact number the user clicked);
    # None means "unknown, use has_more" - the case for search and for the few
    # filters whose live count would need its own scan.
    if has_q:
        return None
    servers = content.get('music_servers') or []

    def _server_row():
        for s in servers:
            if s.get('name') == server_name and not s.get('is_orphan') \
                    and not s.get('is_overlap'):
                return s
        return None

    if kind == 'songs':
        if filt == 'orphan':
            for s in servers:
                if s.get('is_orphan'):
                    return s.get('unique_songs')
            return None
        if filt == 'duplicates':
            return None
        if server_id is None:
            return content.get('total_songs')
        row = _server_row()
        return row.get('unique_songs') if row else None
    if kind == 'artists' and server_id is None:
        return content.get('distinct_artists')
    if kind == 'albums' and server_id is None:
        return content.get('distinct_albums')
    return None


@dashboard_bp.route('/api/dashboard/browse', methods=['GET'])
def browse_api():
    kind = (request.args.get('kind') or 'songs').strip().lower()
    if kind not in _BROWSE_KINDS:
        kind = 'songs'
    filt = (request.args.get('filter') or 'all').strip().lower()
    if filt not in _BROWSE_FILTERS:
        filt = 'all'
    q = (request.args.get('q') or '').strip()
    if len(q) < _BROWSE_MIN_QUERY:
        q = ''
    page = request.args.get('page', 1, type=int) or 1
    if page < 1:
        page = 1

    # Resolve an explicit server (id OR display name); empty / 'all' = catalogue.
    # This endpoint reads the server from its OWN control, not the global selector
    # (server_selector.js deliberately does not scope /api/dashboard/).
    raw_server = (request.args.get('server') or '').strip()
    server_id = None
    server_name = None
    if raw_server and raw_server.lower() != 'all':
        server = registry.get_server(raw_server) or registry.get_server_by_name(raw_server)
        if not server:
            return jsonify({'error': 'Invalid server selection.'}), 400
        server_id = server['server_id']
        server_name = server['name']

    if kind == 'songs' and filt == 'duplicates' and not server_id:
        return jsonify({'error': 'The duplicates filter needs a server.'}), 400
    if kind != 'songs' and filt in ('duplicates', 'orphan'):
        filt = 'all'
    if filt == 'orphan':
        server_id = None
        server_name = None

    page_size = config.DASHBOARD_BROWSE_PAGE_SIZE
    offset = (page - 1) * page_size
    if offset > config.DASHBOARD_BROWSE_MAX_OFFSET:
        return jsonify({
            'kind': kind, 'filter': filt, 'server': server_name, 'page': page,
            'page_size': page_size, 'results': [], 'has_more': False,
            'total': None, 'capped': True,
        })

    if kind == 'artists':
        sql, params = _browse_artists_sql(server_id, q)
    elif kind == 'albums':
        sql, params = _browse_albums_sql(server_id, q)
    else:
        sql, params = _browse_songs_sql(server_id, filt, q)

    try:
        with get_db() as conn, conn.cursor(cursor_factory=DictCursor) as cur:
            content = _load_dashboard_stats(cur)[0]
            # Listing orphans is a score anti-join; when the snapshot already knows
            # the count is 0, skip it rather than scan the whole table to confirm
            # an empty set.
            if (kind == 'songs' and filt == 'orphan'
                    and content.get('orphan_songs') == 0):
                rows, has_more = [], False
            else:
                cur.execute(sql + " LIMIT %s OFFSET %s",
                            params + [page_size + 1, offset])
                rows = cur.fetchall()
                has_more = len(rows) > page_size
                rows = rows[:page_size]
            # The NEXT page would exceed the offset cap, so there is nothing more to
            # page to even if this page filled: never advertise an unreachable page.
            if offset + page_size > config.DASHBOARD_BROWSE_MAX_OFFSET:
                has_more = False
    except Exception:
        logger.exception("dashboard browse query failed")
        return jsonify({'error': 'Browse query failed; check the container logs.'}), 500

    return jsonify({
        'kind': kind, 'filter': filt, 'server': server_name, 'page': page,
        'page_size': page_size, 'results': _browse_serialize(kind, rows),
        'has_more': has_more,
        'total': _browse_total(content or {}, kind, server_id, server_name, filt, bool(q)),
        'capped': False,
    })
# --- end Browse view ---------------------------------------------------------


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


def _collect_music_server_metrics(cur, total_songs=None):
    # Per-configured-server view of the ANALYZED catalogue: how many analyzed
    # songs are mapped to each server, split into distinct songs and the extra
    # duplicate files that collapse onto a song already counted. Reads only
    # track_server_map (the per-server GROUP BY plus one overall COUNT(DISTINCT)
    # for the overlap adjustment) and derives the orphan count by arithmetic from
    # the caller's already-counted total_songs, so it never scans score itself.
    # Empty list when the registry table does not exist.
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
        # On multiple servers: a song present on N servers is counted once in each
        # server row above, so summing those rows over-counts it by N-1. This is a
        # NEGATIVE adjustment removing that excess, measured against the distinct
        # mapped ids so it is exact for any N (a song on 3 servers needs -2).
        cur.execute("SELECT COUNT(DISTINCT item_id) FROM track_server_map")
        mapped_row = cur.fetchone()
        distinct_mapped = int(mapped_row[0] or 0) if mapped_row else 0
        overlap_excess = sum(s['unique_songs'] for s in servers) - distinct_mapped
        if overlap_excess > 0:
            servers.append(
                {
                    'name': 'On multiple servers',
                    'server_type': None,
                    'is_default': False,
                    'unique_songs': -overlap_excess,
                    'duplicate_copies': 0,
                    'resolved': 0,
                    'is_overlap': True,
                }
            )
        # Orphans: analyzed songs bound to NO server (score is append-only, so a
        # removed server / cleaned file / gone track leaves its score row behind).
        # Every mapped id references a score row, so orphans = total songs minus the
        # distinct mapped ones - pure arithmetic from the caller's count, no anti-join
        # scan of score on this 60s path. With the overlap adjustment above, each song
        # is counted once, so the "Unique in catalogue" column sums to the total.
        orphan_count = (
            max(0, int(total_songs) - distinct_mapped)
            if total_songs is not None else 0
        )
        if orphan_count > 0:
            servers.append(
                {
                    'name': 'Orphan',
                    'server_type': None,
                    'is_default': False,
                    'unique_songs': orphan_count,
                    'duplicate_copies': 0,
                    'resolved': orphan_count,
                    'is_orphan': True,
                }
            )
    except Exception as e:
        logger.debug(f"dashboard: music server metrics failed: {e}")
        _safe_rollback(cur)
    return servers


def _collect_fast_metrics(cur):
    # The FAST tier (every 60s): each aggregate here is a single indexed
    # count/GROUP BY that stays under a few seconds even on a 1M-song library.
    # The whole-library DISTRIBUTION charts (Genres, Moods, Tempo) do not move
    # minute to minute and one of them needs a full-table scan, so they all live
    # in _collect_charts_metrics on the hourly cadence instead.
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
    metrics['music_servers'] = _collect_music_server_metrics(
        cur, total_songs=metrics['total_songs']
    )
    # The exact orphan tally (songs bound to no server), published so the Browse
    # view can skip its score anti-join entirely when there are none. A synthetic
    # orphan row exists only when the count is > 0, so its absence in a populated
    # server list means zero; an empty list (no snapshot / no servers) is unknown.
    _orphan_row = next(
        (s for s in metrics['music_servers'] if s.get('is_orphan')), None
    )
    metrics['orphan_songs'] = (
        _orphan_row['unique_songs'] if _orphan_row
        else (0 if metrics['music_servers'] else None)
    )
    # Cleared on any query failure so the caller can refuse to publish a partial
    # snapshot. Popped before serialization.
    metrics['_complete'] = not any(
        metrics[k] is None
        for k in (
            'total_songs', 'distinct_artists', 'distinct_albums',
            'clap_indexed',
        )
    )
    return metrics


def _collect_charts_metrics(cur):
    # The SLOW tier (hourly): the whole-library DISTRIBUTION charts. None of them
    # move minute to minute, and Genres/Moods need the ONE heavy full-table scan
    # (streaming every score row and parsing the mood_vector / other_features text
    # in Python is ~tens of seconds on a 1M-song library), so all three share the
    # hourly cadence and one `charts_updated_at` stamp, kept off the 60s path.
    #
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
    metrics = {
        'top_genre': [{'label': k, 'count': int(v)} for k, v in top_genre],
        'moods_coverage': [{'label': k, 'score': round(v, 2)} for k, v in emotional],
    }

    # Tempo profile: bucket songs into slow/medium/fast/very-fast. Always populate
    # the key so the UI renders a real (possibly-zero) chart rather than the "still
    # collecting" placeholder when no songs have a tempo yet.
    metrics['tempo_profile'] = {
        'slow': 0, 'medium': 0, 'fast': 0, 'very_fast': 0, 'avg_tempo': None,
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
        complete = False

    # The charts' OWN timestamp, so the UI can honestly say these refresh hourly
    # rather than borrowing the fast tier's every-minute stamp.
    metrics['charts_updated_at'] = time.strftime(LOCAL_TZ_FMT)
    metrics['_complete'] = complete
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
#  - FAST (every 60s): the cheap counts and per-server rows. All stay under a few
#    seconds even on a 1M-song library.
#  - CHARTS (hourly): the whole-library distribution charts (Genres, Moods,
#    Tempo). One of them needs a full-table scan (~tens of seconds on 1M rows) and
#    none move minute to minute, so they carry their own hourly timestamp.
DASHBOARD_REFRESH_INTERVAL_SECONDS = 60
DASHBOARD_CHARTS_REFRESH_INTERVAL_SECONDS = 3600


def dashboard_refresh_interval():
    return DASHBOARD_REFRESH_INTERVAL_SECONDS


def _merge_dashboard_content(db, content):
    # Merge the given keys into the dashboard_stats singleton with `content ||`,
    # so the fast refresh never clobbers the hourly chart keys and vice versa. The
    # jsonb || is applied inside a single UPDATE, so concurrent fast/chart writes
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
    # FAST cadence (every 60s): cheap counts and per-server rows.
    _refresh_dashboard_block(app, _collect_fast_metrics, "dashboard_stats")


def refresh_dashboard_charts_stats(app):
    # CHARTS cadence (hourly): the whole-library distribution charts (Genres,
    # Moods Coverage, Tempo), one of which needs a full-table mood scan.
    _refresh_dashboard_block(app, _collect_charts_metrics, "dashboard charts stats")
