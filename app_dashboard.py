"""Dashboard blueprint: landing page with recent activity, content metrics,
index counts, workers and scheduled tasks.

Heavy library aggregates (content metrics) are NOT recomputed
on each request. They are refreshed by ``refresh_dashboard_stats()`` at app
startup and then once per hour, and persisted in the singleton
``dashboard_stats`` table. The summary endpoint only reads that row and
combines it with the cheap, always-live bits (workers, recent tasks, cron).
"""
import json
import logging
import time
import psycopg2
from flask import Blueprint, render_template, jsonify
from psycopg2.extras import DictCursor

from app_helper import get_db, redis_conn

logger = logging.getLogger(__name__)
dashboard_bp = Blueprint('dashboard_bp', __name__)


@dashboard_bp.route('/')
def dashboard_page():
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


def _get_musicnn_index_count():
    try:
        from tasks.voyager_manager import voyager_index, id_map
        if id_map is not None:
            return len(id_map)
        if voyager_index is not None:
            return getattr(voyager_index, 'num_elements', 0)
    except Exception:
        pass
    return 0


def _get_clap_index_count():
    try:
        from tasks.clap_text_search import is_clap_cache_loaded, get_clap_cache_size
        if is_clap_cache_loaded():
            return get_clap_cache_size()
    except Exception:
        pass
    return 0


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
            workers_info.append({
                'hostname': getattr(w, 'hostname', None),
                'queues': [q.name for q in getattr(w, 'queues', [])],
                'state': state,
                'current_job_id': current_job_id,
                'successful_jobs': getattr(w, 'successful_job_count', 0),
                'failed_jobs': getattr(w, 'failed_job_count', 0),
            })
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
                recent.append({
                    'task_id': r['task_id'],
                    'task_type': r['task_type'],
                    'status': r['status'],
                    'duration_seconds': float(r['duration_seconds']) if r['duration_seconds'] is not None else None,
                    'note': r['note'] or '',
                    'timestamp': r['recorded_at'].isoformat() if r['recorded_at'] else None,
                })
        except Exception as e:
            logger.debug(f"dashboard: task_history query failed: {e}")
            _safe_rollback(cur)
    return recent


def _collect_content_metrics(cur):
    metrics = {
        'total_songs': _safe_count(cur, "SELECT COUNT(*) FROM score"),
        'distinct_artists': _safe_count(cur, "SELECT COUNT(DISTINCT author) FROM score WHERE author IS NOT NULL"),
        'distinct_albums': _safe_count(cur, "SELECT COUNT(DISTINCT album) FROM score WHERE album IS NOT NULL"),
        'musicnn_indexed': _get_musicnn_index_count(),
        'clap_indexed': _get_clap_index_count(),
        'gmm_indexed': _get_gmm_index_count(),
    }

    # Parse mood vectors to collect the two signals actually rendered by
    # the dashboard:
    #  - mood_dominant_counts: per-song dominant-label counts, feeds the
    #    Genres chart.
    #  - other_feature_totals: emotional mood scores summed across songs
    #    (from the `other_features` column), feeds the Moods Coverage pie.
    #
    # Both columns are stored as plain text in the `key:value,key:value`
    # format produced by save_track_analysis_and_embedding() in
    # app_helper.py, so we parse that directly. We intentionally do NOT
    # call json.loads on every row: the column is never JSON, so the
    # exception-handling overhead would dominate the loop for large
    # libraries. We also iterate the cursor row-by-row instead of
    # fetchall() to keep memory usage low.
    mood_dominant_counts = {}
    other_feature_totals = {}
    try:
        cur.execute(
            "SELECT mood_vector, other_features FROM score "
            "WHERE mood_vector IS NOT NULL AND mood_vector <> ''"
        )
        for row in cur:
            mv = row[0]
            of = row[1]
            if not mv:
                continue
            parsed = _parse_keyval(mv)
            if not parsed:
                continue
            dom = max(parsed.items(), key=lambda kv: kv[1])[0]
            mood_dominant_counts[dom] = mood_dominant_counts.get(dom, 0) + 1

            # --- emotional mood vector (other_features) ---
            if of:
                of_parsed = _parse_keyval(of)
                for k, s in of_parsed.items():
                    # Skip non-emotional scalar helpers
                    if k in ('tempo_normalized', 'energy_normalized'):
                        continue
                    other_feature_totals[k] = other_feature_totals.get(k, 0.0) + s
    except Exception as e:
        logger.debug(f"dashboard: mood aggregation failed: {e}")
        _safe_rollback(cur)

    # Genre breakdown: dominant-mood counts from mood_vector (genre-like labels).
    top_genre = sorted(mood_dominant_counts.items(), key=lambda kv: kv[1], reverse=True)
    metrics['top_genre'] = [{'label': k, 'count': int(v)} for k, v in top_genre]
    # Moods Coverage: emotional mood vector (other_features):
    # danceable / aggressive / happy / party / relaxed / sad.
    emotional = sorted(other_feature_totals.items(), key=lambda kv: kv[1], reverse=True)
    metrics['moods_coverage'] = [
        {'label': k, 'score': round(v, 2)} for k, v in emotional
    ]

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
            # Malformed numeric field — skip silently.
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
                    last_run_iso = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(r['last_run'])))
            except Exception:
                pass
            rows.append({
                'id': r['id'],
                'name': r['name'],
                'task_type': r['task_type'],
                'cron_expr': r['cron_expr'],
                'enabled': bool(r['enabled']),
                'last_run': last_run_iso,
            })
    except Exception as e:
        logger.debug(f"dashboard: cron query failed: {e}")
        _safe_rollback(cur)
    return rows


@dashboard_bp.route('/api/dashboard/summary', methods=['GET'])
def dashboard_summary():
    """Return the payload rendered by templates/dashboard.html.

    Heavy library aggregates (``content``) are read from the precomputed
    ``dashboard_stats`` singleton row and NOT recomputed on each request.
    Everything else (workers, recent tasks, cron) is cheap and stays live.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    try:
        recent = _collect_task_metrics(cur)
        cron_rows = _collect_cron(cur)
        content, stats_updated_at = _load_dashboard_stats(cur)
    finally:
        cur.close()

    workers = _collect_workers()

    return jsonify({
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'stats_updated_at': stats_updated_at,
        'workers': workers,
        'recent_tasks': recent,
        'content': content,
        'cron': cron_rows,
    })


def _load_dashboard_stats(cur):
    """Read the singleton dashboard_stats row. Returns (content, updated_at_iso)."""
    try:
        cur.execute("SELECT updated_at, content FROM dashboard_stats WHERE id = 1")
        row = cur.fetchone()
        if not row:
            return {}, None
        updated_at = row['updated_at']
        updated_iso = (
            updated_at.strftime('%Y-%m-%d %H:%M:%S')
            if hasattr(updated_at, 'strftime') else (str(updated_at) if updated_at else None)
        )
        content = row['content'] or {}
        return content, updated_iso
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

            cur2 = db.cursor()
            try:
                try:
                    cur2.execute(
                        "INSERT INTO dashboard_stats (id, updated_at, content) "
                        "VALUES (1, NOW(), %s::jsonb) "
                        "ON CONFLICT (id) DO UPDATE SET "
                        "updated_at = EXCLUDED.updated_at, "
                        "content = EXCLUDED.content",
                        (json.dumps(content),),
                    )
                except psycopg2.Error as e:
                    if getattr(e, 'pgcode', None) == '42P10' or 'ON CONFLICT' in str(e):
                        logger.warning("dashboard_stats upsert fallback due missing unique constraint: %s", e)
                        _safe_rollback(cur2)
                        cur2.execute("DELETE FROM dashboard_stats WHERE id = 1")
                        cur2.execute(
                            "INSERT INTO dashboard_stats (id, updated_at, content) "
                            "VALUES (1, NOW(), %s::jsonb)",
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

