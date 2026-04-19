"""Dashboard blueprint: landing page with system health, recent activity,
content metrics, performance, indexes counts and node info."""
import json
import logging
import socket
import time
from urllib.parse import urlparse
from flask import Blueprint, render_template, jsonify
from psycopg2.extras import DictCursor

from app_helper import get_db, redis_conn, rq_queue_high, rq_queue_default

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


def _collect_workers():
    """Return basic info about RQ workers, plus optional CPU/RAM if psutil is available."""
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
            try:
                birth = w.birth_date.isoformat() if getattr(w, 'birth_date', None) else None
            except Exception:
                birth = None
            workers_info.append({
                'name': w.name,
                'hostname': getattr(w, 'hostname', None),
                'pid': getattr(w, 'pid', None),
                'queues': [q.name for q in getattr(w, 'queues', [])],
                'state': state,
                'current_job_id': current_job_id,
                'successful_jobs': getattr(w, 'successful_job_count', 0),
                'failed_jobs': getattr(w, 'failed_job_count', 0),
                'birth_date': birth,
            })
    except Exception as e:
        logger.warning(f"dashboard: failed to enumerate RQ workers: {e}")

    node = {}
    try:
        import psutil
        try:
            hostname = psutil.os.uname().nodename
        except Exception:
            import socket
            hostname = socket.gethostname()
        node = {
            'hostname': hostname,
            'cpu_percent': psutil.cpu_percent(interval=None),
            'cpu_count': psutil.cpu_count(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': round(psutil.virtual_memory().used / (1024 * 1024), 1),
            'memory_total_mb': round(psutil.virtual_memory().total / (1024 * 1024), 1),
        }
    except Exception as e:
        logger.debug(f"dashboard: psutil not available or failed: {e}")
    return workers_info, node


def _collect_queue_health():
    info = {'queues': [], 'redis_ok': False}
    try:
        redis_conn.ping()
        info['redis_ok'] = True
    except Exception as e:
        logger.warning(f"dashboard: redis ping failed: {e}")

    for q in (rq_queue_high, rq_queue_default):
        entry = {'name': q.name, 'queued': 0, 'started': 0, 'failed': 0, 'finished': 0, 'deferred': 0}
        try:
            entry['queued'] = q.count
        except Exception:
            pass
        try:
            entry['started'] = q.started_job_registry.count
        except Exception:
            pass
        try:
            entry['failed'] = q.failed_job_registry.count
        except Exception:
            pass
        try:
            entry['finished'] = q.finished_job_registry.count
        except Exception:
            pass
        try:
            entry['deferred'] = q.deferred_job_registry.count
        except Exception:
            pass
        info['queues'].append(entry)
    return info


def _collect_task_metrics(cur):
    """Return recent main tasks, daily history, distributions, and average
    duration per task type using start_time/end_time columns."""
    recent = []
    # Recent activity comes exclusively from the persistent task_history table
    # (populated when a main task ends — naturally, by failure, or via the
    # global Cancel button). This avoids duplicating rows that would appear in
    # both task_status and task_history, and prevents the synthetic
    # placeholder rows (task_type='unknown') from showing up.
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

    history = []
    try:
        cur.execute("""
            SELECT DATE(timestamp) AS day,
                   COUNT(*) FILTER (WHERE status = 'SUCCESS') AS success,
                   COUNT(*) FILTER (WHERE status = 'FAILURE') AS failure,
                   COUNT(*) AS total
            FROM task_status
            WHERE parent_task_id IS NULL
              AND timestamp >= NOW() - INTERVAL '7 days'
            GROUP BY day
            ORDER BY day ASC
        """)
        for r in cur.fetchall():
            history.append({
                'day': r['day'].isoformat() if r['day'] else None,
                'success': int(r['success'] or 0),
                'failure': int(r['failure'] or 0),
                'total': int(r['total'] or 0),
            })
    except Exception as e:
        logger.debug(f"dashboard: task history query failed: {e}")
        _safe_rollback(cur)

    status_dist = {}
    try:
        cur.execute("""
            SELECT status, COUNT(*) AS n
            FROM task_status
            WHERE parent_task_id IS NULL
              AND timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY status
        """)
        for r in cur.fetchall():
            status_dist[r['status'] or 'UNKNOWN'] = int(r['n'] or 0)
    except Exception as e:
        logger.debug(f"dashboard: status dist query failed: {e}")
        _safe_rollback(cur)

    type_dist = {}
    try:
        cur.execute("""
            SELECT task_type, COUNT(*) AS n
            FROM task_status
            WHERE parent_task_id IS NULL
              AND timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY task_type
            ORDER BY n DESC
            LIMIT 8
        """)
        for r in cur.fetchall():
            type_dist[r['task_type'] or 'unknown'] = int(r['n'] or 0)
    except Exception as e:
        logger.debug(f"dashboard: type dist query failed: {e}")
        _safe_rollback(cur)

    # Average duration uses the dedicated start_time / end_time columns
    # (also covers main tasks that ended in FAILURE so the figure isn't empty
    # while everything is still running for the first time).
    avg_duration = []
    try:
        cur.execute("""
            SELECT task_type,
                   COUNT(*) AS runs,
                   AVG(end_time - start_time) AS avg_seconds
            FROM task_status
            WHERE parent_task_id IS NULL
              AND status IN ('SUCCESS', 'FAILURE')
              AND start_time IS NOT NULL
              AND end_time IS NOT NULL
              AND end_time > start_time
              AND timestamp >= NOW() - INTERVAL '30 days'
            GROUP BY task_type
            ORDER BY runs DESC
            LIMIT 8
        """)
        for r in cur.fetchall():
            avg_duration.append({
                'task_type': r['task_type'],
                'runs': int(r['runs'] or 0),
                'avg_seconds': float(r['avg_seconds']) if r['avg_seconds'] is not None else None,
            })
    except Exception as e:
        logger.debug(f"dashboard: avg duration query failed: {e}")
        _safe_rollback(cur)

    return recent, history, status_dist, type_dist, avg_duration


def _collect_analysis_perf(cur):
    """Per-album analysis subtask metrics, projected onto a 3-minute song.

    `album_analysis` rows are subtasks (one per album). We compute average
    duration per album and, when possible, per song using a count from the
    JSON `details` column. Then project to a 3-minute song using the
    configured target audio length.
    """
    out = {
        'sample_size': 0,
        'avg_album_seconds': None,
        'avg_per_song_seconds': None,
        'estimated_for_3min_song_seconds': None,
        'per_song_basis': None,
        'reference_audio_length_s': None,
    }
    try:
        from config import TARGET_AUDIO_LENGTH_S as _ref
        out['reference_audio_length_s'] = float(_ref)
    except Exception:
        try:
            from config import MAX_AUDIO_LENGTH_S as _ref
            out['reference_audio_length_s'] = float(_ref)
        except Exception:
            out['reference_audio_length_s'] = None

    try:
        cur.execute("""
            SELECT details, start_time, end_time
            FROM task_status
            WHERE task_type = 'album_analysis'
              AND status = 'SUCCESS'
              AND start_time IS NOT NULL
              AND end_time IS NOT NULL
              AND end_time > start_time
              AND timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY timestamp DESC
            LIMIT 200
        """)
        rows = cur.fetchall()
    except Exception as e:
        logger.debug(f"dashboard: analysis perf query failed: {e}")
        _safe_rollback(cur)
        rows = []

    total_album_time = 0.0
    total_songs = 0
    album_count = 0
    for r in rows:
        try:
            duration = float(r['end_time']) - float(r['start_time'])
            if duration <= 0:
                continue
            total_album_time += duration
            album_count += 1
            songs_in_album = None
            if r['details']:
                try:
                    d = json.loads(r['details'])
                    if isinstance(d, dict):
                        for key in ('songs_in_album', 'tracks_count', 'song_count', 'num_songs'):
                            v = d.get(key)
                            if isinstance(v, (int, float)) and v > 0:
                                songs_in_album = int(v)
                                break
                except Exception:
                    pass
            if songs_in_album:
                total_songs += songs_in_album
        except Exception:
            continue

    if album_count > 0:
        out['sample_size'] = album_count
        out['avg_album_seconds'] = round(total_album_time / album_count, 2)
        if total_songs > 0:
            per_song = total_album_time / total_songs
            out['avg_per_song_seconds'] = round(per_song, 2)
            out['per_song_basis'] = 'songs_in_album'
            ref = out['reference_audio_length_s']
            if ref and ref > 0:
                out['estimated_for_3min_song_seconds'] = round(per_song * (180.0 / ref), 2)
            else:
                out['estimated_for_3min_song_seconds'] = round(per_song, 2)
        else:
            per_song = total_album_time / album_count
            out['avg_per_song_seconds'] = round(per_song, 2)
            out['per_song_basis'] = 'fallback_albums_only'
            out['estimated_for_3min_song_seconds'] = round(per_song, 2)
    return out


def _collect_content_metrics(cur):
    metrics = {
        'total_songs': _safe_count(cur, "SELECT COUNT(*) FROM score"),
        'analyzed_songs': _safe_count(
            cur, "SELECT COUNT(*) FROM score WHERE mood_vector IS NOT NULL AND mood_vector <> ''"
        ),
        'distinct_artists': _safe_count(cur, "SELECT COUNT(DISTINCT author) FROM score WHERE author IS NOT NULL"),
        'distinct_albums': _safe_count(cur, "SELECT COUNT(DISTINCT album) FROM score WHERE album IS NOT NULL"),
        'total_playlists': _safe_count(cur, "SELECT COUNT(DISTINCT playlist_name) FROM playlist"),
        'total_playlist_items': _safe_count(cur, "SELECT COUNT(*) FROM playlist"),
        'indexed_songs': 0,
        'artists_with_gmm': 0,
    }
    if _table_exists(cur, 'embedding'):
        metrics['indexed_songs'] = _safe_count(
            cur, "SELECT COUNT(*) FROM embedding WHERE embedding IS NOT NULL"
        )
    # artist_index_data stores one row per index segment, each with a JSON
    # artist_map. We parse it and union the artist keys across all rows so we
    # get the true number of artists actually covered by the GMM index.
    if _table_exists(cur, 'artist_index_data'):
        try:
            cur.execute("SELECT artist_map_json FROM artist_index_data")
            artists = set()
            for (mp,) in cur.fetchall():
                if not mp:
                    continue
                try:
                    obj = json.loads(mp)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    artists.update(obj.keys())
                elif isinstance(obj, list):
                    artists.update(str(x) for x in obj)
            metrics['artists_with_gmm'] = len(artists)
        except Exception as e:
            logger.debug(f"dashboard: artist_map_json parse failed: {e}")
            _safe_rollback(cur)

    # Parse mood vectors to collect multiple signals:
    #  - mood_totals: sum of scores per mood label (overall prevalence)
    #  - mood_dominant_counts: number of songs where this mood scored highest
    #    (used internally for the "Top Genre" view)
    #  - mood_presence_counts: number of songs in which this mood is present
    #    at all (score > 0) — used for the "Moods Coverage" view
    mood_totals = {}
    mood_dominant_counts = {}
    mood_presence_counts = {}
    sampled_songs = 0
    moods_per_song_sum = 0
    distinct_moods = set()
    # `other_features` is the column that holds the *emotional* moods
    # (danceable, aggressive, happy, party, relaxed, sad). `mood_vector`
    # holds genre-like labels. We aggregate both in the same pass.
    other_feature_totals = {}
    other_feature_songs = 0
    try:
        cur.execute(
            "SELECT mood_vector, other_features FROM score "
            "WHERE mood_vector IS NOT NULL AND mood_vector <> '' LIMIT 10000"
        )
        for row in cur.fetchall():
            mv = row[0]
            of = row[1]
            if not mv:
                continue
            parsed = {}
            try:
                obj = json.loads(mv)
                if isinstance(obj, dict):
                    parsed = {str(k): float(v) for k, v in obj.items() if _is_number(v)}
            except Exception:
                parsed = {}
            if not parsed:
                for part in str(mv).split(','):
                    if ':' in part:
                        k, _, v = part.partition(':')
                        k = k.strip()
                        try:
                            parsed[k] = float(v)
                        except Exception:
                            continue
            if not parsed:
                continue
            sampled_songs += 1
            present = {k: s for k, s in parsed.items() if s > 0.0}
            moods_per_song_sum += len(present)
            distinct_moods.update(present.keys())
            for k, s in parsed.items():
                mood_totals[k] = mood_totals.get(k, 0.0) + s
            for k in present.keys():
                mood_presence_counts[k] = mood_presence_counts.get(k, 0) + 1
            dom = max(parsed.items(), key=lambda kv: kv[1])[0]
            mood_dominant_counts[dom] = mood_dominant_counts.get(dom, 0) + 1

            # --- emotional mood vector (other_features) ---
            if of:
                of_parsed = {}
                try:
                    obj2 = json.loads(of)
                    if isinstance(obj2, dict):
                        of_parsed = {str(k): float(v) for k, v in obj2.items() if _is_number(v)}
                except Exception:
                    of_parsed = {}
                if not of_parsed:
                    for part in str(of).split(','):
                        if ':' in part:
                            k, _, v = part.partition(':')
                            k = k.strip()
                            try:
                                of_parsed[k] = float(v)
                            except Exception:
                                continue
                if of_parsed:
                    other_feature_songs += 1
                    for k, s in of_parsed.items():
                        # Skip non-emotional scalar helpers
                        if k in ('tempo_normalized', 'energy_normalized'):
                            continue
                        other_feature_totals[k] = other_feature_totals.get(k, 0.0) + s
    except Exception as e:
        logger.debug(f"dashboard: mood aggregation failed: {e}")
        _safe_rollback(cur)

    # Top Genre: dominant-mood counts from mood_vector (genre-like labels).
    top_genre = sorted(mood_dominant_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    metrics['top_genre'] = [{'label': k, 'count': int(v)} for k, v in top_genre]
    # Keep the score-based ranking for completeness.
    top_moods = sorted(mood_totals.items(), key=lambda kv: kv[1], reverse=True)[:10]
    metrics['top_moods'] = [{'label': k, 'score': round(v, 2)} for k, v in top_moods]
    # Moods Coverage now reads the emotional mood vector (other_features):
    # danceable / aggressive / happy / party / relaxed / sad.
    emotional = sorted(other_feature_totals.items(), key=lambda kv: kv[1], reverse=True)
    metrics['moods_coverage'] = [
        {'label': k, 'score': round(v, 2)} for k, v in emotional
    ]
    metrics['moods_coverage_sample'] = other_feature_songs
    metrics['mood_stats'] = {
        'sampled_songs': sampled_songs,
        'distinct_moods': len(distinct_moods),
        'avg_moods_per_song': round(moods_per_song_sum / sampled_songs, 2) if sampled_songs else 0,
    }
    metrics['mood_stats'] = {
        'sampled_songs': sampled_songs,
        'distinct_moods': len(distinct_moods),
        'avg_moods_per_song': round(moods_per_song_sum / sampled_songs, 2) if sampled_songs else 0,
    }

    # Tempo profile: bucket songs into slow/medium/fast/very-fast
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
        logger.debug(f"dashboard: tempo profile query failed: {e}")
        _safe_rollback(cur)

    return metrics


def _is_number(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def _collect_indexes(cur):
    """Return row counts for the various indexes / caches.

    These counts are surfaced as Key Number cards on the dashboard so the
    user can compare them against the total song count.
    """
    items = []

    def _check_table(label, table, ts_col='created_at', count_sql=None):
        if not _table_exists(cur, table):
            return None
        rows = _safe_count(cur, count_sql or f"SELECT COUNT(*) FROM {table}")
        last = None
        try:
            cur.execute(f"SELECT MAX({ts_col}) FROM {table}")
            r = cur.fetchone()
            if r and r[0]:
                last = r[0].isoformat() if hasattr(r[0], 'isoformat') else str(r[0])
        except Exception:
            _safe_rollback(cur)
        return {'name': label, 'rows': rows, 'updated_at': last}

    def _add(item):
        if item is not None:
            items.append(item)

    # Voyager song index count is surfaced inside the "Total Songs" card,
    # GMM/Map/Artist-Component are not shown as separate Key Numbers — we only
    # list MuLan here (when the table actually exists in this deployment).
    _add(_check_table(
        'MuLan Index', 'mulan_embedding',
        count_sql="SELECT COUNT(*) FROM mulan_embedding WHERE embedding IS NOT NULL",
    ))
    return items


def _resolve_host(host):
    if not host:
        return None
    try:
        return socket.gethostbyname(host)
    except Exception:
        return None


def _collect_services():
    """Return health info for each service in the cluster: web, postgres, redis,
    plus RQ workers (which we already collect separately)."""
    services = []

    # Web (this Flask process)
    try:
        web_host = socket.gethostname()
    except Exception:
        web_host = None
    services.append({
        'role': 'web',
        'name': 'Flask web',
        'host': web_host,
        'ip': _resolve_host(web_host),
        'port': None,
        'status': 'ok',
        'detail': 'serving this dashboard',
    })

    # Postgres
    try:
        from config import DATABASE_URL
        pg_url = urlparse(DATABASE_URL)
        pg_host = pg_url.hostname
        pg_port = pg_url.port or 5432
        pg_db = (pg_url.path or '').lstrip('/') or None
    except Exception:
        pg_host, pg_port, pg_db = None, None, None
    pg_status, pg_detail = 'unknown', None
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT version()")
        ver = cur.fetchone()[0]
        cur.close()
        pg_status = 'ok'
        pg_detail = ver.split(' on ')[0] if ver else None
    except Exception as e:
        pg_status = 'down'
        pg_detail = str(e)[:120]
    services.append({
        'role': 'postgres',
        'name': 'PostgreSQL' + (f' / {pg_db}' if pg_db else ''),
        'host': pg_host,
        'ip': _resolve_host(pg_host),
        'port': pg_port,
        'status': pg_status,
        'detail': pg_detail,
    })

    # Redis
    try:
        from config import REDIS_URL
        r_url = urlparse(REDIS_URL)
        r_host = r_url.hostname
        r_port = r_url.port or 6379
    except Exception:
        r_host, r_port = None, None
    r_status, r_detail = 'unknown', None
    try:
        info = redis_conn.info('server')
        r_status = 'ok'
        r_detail = 'redis ' + str(info.get('redis_version', '?'))
    except Exception as e:
        r_status = 'down'
        r_detail = str(e)[:120]
    services.append({
        'role': 'redis',
        'name': 'Redis',
        'host': r_host,
        'ip': _resolve_host(r_host),
        'port': r_port,
        'status': r_status,
        'detail': r_detail,
    })
    return services


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


def _recent_failures(db):
    try:
        cur = db.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM task_status
            WHERE status = 'FAILURE'
              AND parent_task_id IS NULL
              AND timestamp >= NOW() - INTERVAL '24 hours'
        """)
        n = cur.fetchone()[0]
        cur.close()
        return int(n or 0)
    except Exception:
        return 0


@dashboard_bp.route('/api/dashboard/summary', methods=['GET'])
def dashboard_summary():
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    try:
        recent, history, status_dist, type_dist, avg_duration = _collect_task_metrics(cur)
        analysis_perf = _collect_analysis_perf(cur)
        content = _collect_content_metrics(cur)
        indexes = _collect_indexes(cur)
        cron_rows = _collect_cron(cur)
    finally:
        cur.close()

    workers, node = _collect_workers()
    queue_health = _collect_queue_health()
    services = _collect_services()

    total_30d = sum(status_dist.values())
    success_30d = status_dist.get('SUCCESS', 0)
    failure_30d = status_dist.get('FAILURE', 0)
    success_rate = round(100.0 * success_30d / total_30d, 1) if total_30d else None

    return jsonify({
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_health': {
            'redis_ok': queue_health['redis_ok'],
            'queues': queue_health['queues'],
            'workers_count': len(workers),
            'enabled_cron_jobs': sum(1 for c in cron_rows if c['enabled']),
            'recent_failures_24h': _recent_failures(db),
        },
        'workers': workers,
        'services': services,
        'node': node,
        'recent_tasks': recent,
        'task_history_7d': history,
        'task_status_distribution_30d': status_dist,
        'task_type_distribution_30d': type_dist,
        'avg_duration_30d': avg_duration,
        'analysis_performance': analysis_perf,
        'content': content,
        'indexes': indexes,
        'cron': cron_rows,
        'quality': {
            'success_rate_30d': success_rate,
            'success_30d': success_30d,
            'failure_30d': failure_30d,
            'total_30d': total_30d,
        },
    })
