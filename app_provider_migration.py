"""Provider migration tool — Flask blueprint + runtime config override.

This module is the single add-on entry point for switching the active media
server provider on a running AudioMuse-AI install. It adds a wizard page at
``/provider-migration`` plus the backing REST API under ``/api/migration/*``.

It also exposes two top-level functions that are wired into ``app.py`` and
the RQ workers so every process hot-reloads the active provider when the
migration finishes:

    * ``apply_provider_overrides_from_db()`` — mutates ``config`` globals
      from the ``app_settings`` row written by the migration.
    * ``subscribe_to_provider_migrated_channel()`` — background thread that
      listens on Redis pub/sub so other processes pick up the override
      without a restart.

The tool itself does NOT touch ``config.py``. Credentials for the *target*
provider stay in ``migration_session.target_creds`` and are passed explicitly
to ``tasks.provider_probe`` (which also never reads ``config``), so the
current live provider keeps working throughout the dry-run and manual matching
steps of the wizard.
"""
import json
import logging
import os
import sys
import threading

from flask import Blueprint, jsonify, render_template, request

# App-level singletons (DB connection, Redis, RQ queues). Importing here keeps
# the blueprint file self-contained — the rest of the app doesn't need to hand
# anything in.
from app_helper import get_db, redis_conn, rq_queue_high

logger = logging.getLogger(__name__)

migration_bp = Blueprint('migration_bp', __name__)


# ---------------------------------------------------------------------------
# Lazy provider_probe import — keeps the _import_module bypass test happy
# because we don't trigger ``tasks/__init__.py`` at module-load time.
# ---------------------------------------------------------------------------

class _LazyProbe:
    """Lazy-imports ``tasks.provider_probe`` on first attribute access.

    Tests replace ``provider_probe`` on the module directly with a MagicMock,
    so the lazy loader never fires during tests.
    """
    _real = None

    def _load(self):
        if self._real is None:
            import importlib
            self._real = importlib.import_module('tasks.provider_probe')
        return self._real

    def __getattr__(self, name):
        return getattr(self._load(), name)


provider_probe = _LazyProbe()


# ---------------------------------------------------------------------------
# Supported target providers (what the tool knows how to talk to)
# ---------------------------------------------------------------------------

_SUPPORTED_TARGETS = frozenset({'jellyfin', 'navidrome', 'emby', 'lyrion', 'mpd'})


# ---------------------------------------------------------------------------
# Runtime config override — called at app/worker startup from app.py
# ---------------------------------------------------------------------------

def apply_provider_overrides_from_db():
    """Read ``migration.active_provider`` from ``app_settings`` and mutate
    the ``config`` module's globals.

    Safe no-op when:
      * the row doesn't exist (fresh install, no migration run yet),
      * the JSON is malformed,
      * the DB isn't reachable (returns silently with a warning log).

    Called from ``app.py``, ``rq_worker.py``, and ``rq_worker_high_priority.py``
    immediately after ``init_db()``.
    """
    try:
        import config
        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                "SELECT value FROM app_settings WHERE key = 'migration.active_provider'"
            )
            row = cur.fetchone()
        if not row:
            return
        raw = row[0]
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        t = (data or {}).get('type')
        creds = (data or {}).get('credentials') or {}
        if not t:
            return

        config.MEDIASERVER_TYPE = t
        if t == 'jellyfin':
            config.JELLYFIN_URL     = creds.get('url')     or config.JELLYFIN_URL
            config.JELLYFIN_USER_ID = creds.get('user_id') or config.JELLYFIN_USER_ID
            config.JELLYFIN_TOKEN   = creds.get('token')   or config.JELLYFIN_TOKEN
            config.HEADERS          = {"X-Emby-Token": config.JELLYFIN_TOKEN}
        elif t == 'emby':
            config.EMBY_URL     = creds.get('url')     or config.EMBY_URL
            config.EMBY_USER_ID = creds.get('user_id') or config.EMBY_USER_ID
            config.EMBY_TOKEN   = creds.get('token')   or config.EMBY_TOKEN
            config.HEADERS      = {"X-Emby-Token": config.EMBY_TOKEN}
        elif t == 'navidrome':
            config.NAVIDROME_URL      = creds.get('url')      or config.NAVIDROME_URL
            config.NAVIDROME_USER     = creds.get('user')     or config.NAVIDROME_USER
            config.NAVIDROME_PASSWORD = creds.get('password') or config.NAVIDROME_PASSWORD
            config.HEADERS            = {}
        elif t == 'lyrion':
            config.LYRION_URL = creds.get('url') or config.LYRION_URL
        elif t == 'mpd':
            config.MPD_HOST            = creds.get('host')             or config.MPD_HOST
            config.MPD_PORT            = int(creds.get('port') or config.MPD_PORT)
            config.MPD_PASSWORD        = creds.get('password')         or config.MPD_PASSWORD
            config.MPD_MUSIC_DIRECTORY = creds.get('music_directory')  or config.MPD_MUSIC_DIRECTORY

        logger.info(
            "provider migration: runtime override applied (type=%s)", t
        )
    except Exception as e:
        logger.warning(
            "provider migration: apply_provider_overrides_from_db failed "
            "(ignored): %s", e
        )


def subscribe_to_provider_migrated_channel():
    """Subscribe to Redis ``provider-migrated`` in a daemon thread.

    When the migration tool publishes a message after a successful execute,
    other Flask/worker processes pick it up here and re-run
    ``apply_provider_overrides_from_db()`` to hot-reload their config without
    a container restart. Failures inside the listener are logged and the
    thread exits cleanly.
    """
    def _listen():
        try:
            ps = redis_conn.pubsub(ignore_subscribe_messages=True)
            ps.subscribe('provider-migrated')
            for msg in ps.listen():
                if msg.get('type') == 'message':
                    logger.info(
                        "provider migration: received provider-migrated "
                        "pub/sub, reloading config"
                    )
                    apply_provider_overrides_from_db()
        except Exception as e:
            logger.warning(
                "provider migration: pub/sub subscriber died: %s", e
            )

    t = threading.Thread(
        target=_listen, daemon=True, name='provider-migrated-sub'
    )
    t.start()


# ---------------------------------------------------------------------------
# Routes — wizard page
# ---------------------------------------------------------------------------

@migration_bp.route('/provider-migration')
def provider_migration_page():
    return render_template(
        'provider_migration.html',
        title='Provider Migration',
        active='provider_migration',
    )


# ---------------------------------------------------------------------------
# Routes — session CRUD
# ---------------------------------------------------------------------------

@migration_bp.route('/api/migration/session/start', methods=['POST'])
def session_start():
    payload = request.get_json(silent=True) or {}
    target_type = (payload.get('target_type') or '').lower()
    target_creds = payload.get('target_creds') or {}

    if target_type not in _SUPPORTED_TARGETS:
        return jsonify({'error': f'target_type must be one of {sorted(_SUPPORTED_TARGETS)}'}), 400

    import config
    source_type = getattr(config, 'MEDIASERVER_TYPE', '') or ''

    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO migration_session "
            "(source_type, target_type, target_creds, state, status) "
            "VALUES (%s, %s, %s, %s, 'in_progress') RETURNING id",
            (source_type, target_type, json.dumps(target_creds), json.dumps({})),
        )
        row = cur.fetchone()
    db.commit()
    return jsonify({'session_id': row[0]})


@migration_bp.route('/api/migration/session/<int:session_id>', methods=['GET'])
def session_get(session_id):
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, source_type, target_type, status, state "
            "FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return jsonify({'error': 'session not found'}), 404
    _id, source_type, target_type, status, state = row
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {}
    return jsonify({
        'id': _id,
        'source_type': source_type,
        'target_type': target_type,
        'status': status,
        'state': state,
    })


# ---------------------------------------------------------------------------
# Routes — probe (delegates to tasks.provider_probe, passes creds explicitly)
# ---------------------------------------------------------------------------

@migration_bp.route('/api/migration/probe/test', methods=['POST'])
def probe_test():
    payload = request.get_json(silent=True) or {}
    t = (payload.get('type') or '').lower()
    creds = payload.get('creds') or {}
    try:
        result = provider_probe.test_connection(t, creds)
    except NotImplementedError as e:
        return jsonify({'ok': False, 'error': str(e), 'path_format': 'none',
                        'sample_count': 0, 'warnings': []}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'path_format': 'none',
                        'sample_count': 0, 'warnings': []}), 200
    return jsonify(result)


@migration_bp.route('/api/migration/search-albums', methods=['POST'])
def search_albums():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    query = payload.get('query') or ''

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404
    target_type, creds = session
    try:
        albums = provider_probe.search_albums(target_type, creds, query)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'albums': albums})


# ---------------------------------------------------------------------------
# Routes — dry run, manual match, finalize
# ---------------------------------------------------------------------------

@migration_bp.route('/api/migration/dry-run', methods=['POST'])
def dry_run():
    """Fetch all tracks from the target provider, match them against score,
    persist the result in ``migration_session.state->'dry_run'``."""
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404
    target_type, creds = session

    try:
        new_tracks = provider_probe.fetch_all_tracks(target_type, creds)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    old_rows = _load_score_rows_as_dicts()

    # Lazy import of matcher — same reasoning as provider_probe
    import importlib
    matcher = importlib.import_module('tasks.provider_migration_matcher')
    result = matcher.match_tracks(old_rows, new_tracks)

    # Serialize only what we need for persistence (no unmatched row dicts in state —
    # keep it light; unmatched_by_album is reconstructed from unmatched on demand)
    state_dry_run = {
        'matches':           result['matches'],
        'tier_counts':       result['tier_counts'],
        'unmatched_albums':  _albums_payload(result['unmatched_by_album']),
    }
    # Also snapshot new track metadata keyed by new_id for the post-execute
    # score refresh (file_path, title, artist, album, year).
    new_meta = {
        n['id']: {
            'path':   n.get('path'),
            'title':  n.get('title'),
            'artist': n.get('album_artist') or n.get('artist'),
            'album':  n.get('album'),
            'year':   n.get('year'),
        }
        for n in new_tracks if n.get('id')
    }

    _update_state(session_id, dry_run=state_dry_run, new_meta=new_meta,
                  manual_matches={}, final_counts=None)

    return jsonify({
        'tier_counts': result['tier_counts'],
        'matched':     len(result['matches']),
        'unmatched':   len(result['unmatched']),
        'unmatched_albums_count': len(result['unmatched_by_album']),
    })


@migration_bp.route('/api/migration/match-album', methods=['POST'])
def match_album():
    """User picked a target album for one of the unmatched old albums. We
    fetch the target album's tracks and auto-match inside it by title.

    Tracks we can't auto-match are left as orphans — the user sees them in
    the UI and can explicitly skip the album or re-run.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    old_album_key = payload.get('old_album_key')  # [album_artist, album]
    new_album_id = payload.get('new_album_id')

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404
    target_type, creds = session

    try:
        new_tracks = provider_probe.get_album_tracks(target_type, creds, new_album_id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    import importlib
    matcher = importlib.import_module('tasks.provider_migration_matcher')

    # Load the unmatched tracks for this album
    old_album_tuple = tuple(old_album_key) if isinstance(old_album_key, list) else old_album_key
    old_rows = _load_unmatched_for_album(session_id, old_album_tuple)

    # Match within the album: exact title, then normalized title
    by_title = {}
    by_norm_title = {}
    for n in new_tracks:
        t = (n.get('title') or '').lower()
        if t and t not in by_title:
            by_title[t] = n['id']
        nt = matcher.normalize_meta(n.get('title'))
        if nt and nt not in by_norm_title:
            by_norm_title[nt] = n['id']

    newly_matched = {}
    still_unmatched = []
    for old in old_rows:
        title_l = (old.get('title') or '').lower()
        nt = matcher.normalize_meta(old.get('title'))
        if title_l in by_title:
            newly_matched[old['item_id']] = by_title[title_l]
        elif nt and nt in by_norm_title:
            newly_matched[old['item_id']] = by_norm_title[nt]
        else:
            still_unmatched.append(old['item_id'])

    _merge_manual_matches(session_id, newly_matched)
    return jsonify({
        'matched':   len(newly_matched),
        'unmatched': len(still_unmatched),
        'unmatched_item_ids': still_unmatched,
    })


@migration_bp.route('/api/migration/skip-album', methods=['POST'])
def skip_album():
    """Mark an unmatched album as explicitly orphaned — those rows will be
    deleted by execute. Achieved by NOT adding anything to manual_matches
    (execute deletes everything not in the merged mapping)."""
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    # Record the skip in state so the UI can show progress; execution doesn't
    # care (orphans = anything not in mapping).
    _mark_album_skipped(session_id, payload.get('old_album_key'))
    return jsonify({'ok': True})


@migration_bp.route('/api/migration/finalize-dry-run', methods=['POST'])
def finalize_dry_run():
    """Compute final counts and transition session.status to 'dry_run_ready'."""
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')

    state = _load_state(session_id)
    if state is None:
        return jsonify({'error': 'session not found'}), 404

    dry = state.get('dry_run') or {}
    manual = state.get('manual_matches') or {}

    merged = {}
    merged.update(dry.get('matches') or {})
    merged.update(manual)

    total_score = _count_score_rows()
    matched = len(merged)
    orphans = max(0, total_score - matched)

    final_counts = {
        'matched':          matched,
        'orphans':          orphans,
        'tier_counts':      dry.get('tier_counts') or {},
    }

    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "UPDATE migration_session SET "
            "  state = jsonb_set(state, '{final_counts}', %s::jsonb, true), "
            "  status = 'dry_run_ready' "
            "WHERE id = %s",
            (json.dumps(final_counts), session_id),
        )
    db.commit()
    return jsonify(final_counts)


# ---------------------------------------------------------------------------
# Routes — execute gate + status
# ---------------------------------------------------------------------------

@migration_bp.route('/api/migration/execute', methods=['POST'])
def execute():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    backup_confirmed = bool(payload.get('backup_confirmed'))
    confirmation_text = payload.get('confirmation_text') or ''

    if not backup_confirmed:
        return jsonify({'error': 'You must confirm the backup checkbox'}), 400

    # Look up session target_type + current status for the gate check
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT target_type, status FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return jsonify({'error': 'session not found'}), 404
    target_type, status = row[0], row[1]

    expected = f"I want to migrate to {target_type} and delete unmatched tracks"
    if confirmation_text != expected:
        return jsonify({
            'error': f'Confirmation text does not match. Expected exactly: "{expected}"'
        }), 400
    if status != 'dry_run_ready':
        return jsonify({
            'error': f'Dry run must be finalized first. Session status is "{status}", '
                     f'expected "dry_run_ready".'
        }), 400

    # Enqueue the execute job
    from rq.job import Job  # noqa: F401  (used by enqueue internals)
    job = rq_queue_high.enqueue(
        'tasks.provider_migration_tasks.execute_provider_migration',
        session_id,
        job_timeout=3600,
    )
    return jsonify({'task_id': job.get_id()})


@migration_bp.route('/api/migration/status/<task_id>', methods=['GET'])
def job_status(task_id):
    try:
        from rq.job import Job
        job = Job.fetch(task_id, connection=redis_conn)
        return jsonify({
            'id': job.get_id(),
            'status': job.get_status(),
            'result': job.result if job.is_finished else None,
            'error': str(job.exc_info) if job.is_failed else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@migration_bp.route('/api/migration/dry-run-report/<int:session_id>', methods=['GET'])
def dry_run_report(session_id):
    """Download the full session state as a JSON file for audit."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT state FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return jsonify({'error': 'session not found'}), 404
    state = row[0]
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            pass
    from flask import Response
    pretty = json.dumps(state, indent=2, default=str)
    return Response(
        pretty,
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename=migration_session_{session_id}.json'},
    )


# ---------------------------------------------------------------------------
# Small DB helpers (kept near the routes that use them so behavior + SQL live
# together; these are also why the test suite patches ``get_db``).
# ---------------------------------------------------------------------------

def _fetch_session_creds(session_id):
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT target_type, target_creds FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    target_type, creds_raw = row
    try:
        creds = json.loads(creds_raw) if isinstance(creds_raw, str) else (creds_raw or {})
    except Exception:
        creds = {}
    return target_type, creds


def _load_score_rows_as_dicts():
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT item_id, file_path, title, author, album, album_artist FROM score"
        )
        rows = cur.fetchall() or []
    return [
        {
            'item_id':      r[0],
            'file_path':    r[1],
            'title':        r[2],
            'author':       r[3],
            'album':        r[4],
            'album_artist': r[5],
        }
        for r in rows
    ]


def _load_unmatched_for_album(session_id, album_key):
    """Return the set of old rows that live in the given (album_artist, album)
    and were NOT matched by the dry run."""
    state = _load_state(session_id) or {}
    matched_ids = set((state.get('dry_run') or {}).get('matches', {}).keys())
    matched_ids |= set((state.get('manual_matches') or {}).keys())
    rows = _load_score_rows_as_dicts()
    target_artist, target_album = (album_key[0] if album_key else None,
                                   album_key[1] if album_key and len(album_key) > 1 else None)
    out = []
    for r in rows:
        if r['item_id'] in matched_ids:
            continue
        ra = r.get('album_artist') or r.get('author')
        if ra == target_artist and r.get('album') == target_album:
            out.append(r)
    return out


def _albums_payload(unmatched_by_album):
    """Serialize ``{(album_artist, album): [rows]}`` into a JSON-safe list
    suitable for the wizard UI."""
    out = []
    for key, rows in unmatched_by_album.items():
        album_artist, album = key[0], key[1] if len(key) > 1 else None
        out.append({
            'album_artist': album_artist,
            'album':        album,
            'track_count':  len(rows),
            'sample_titles': [r.get('title') for r in rows[:5]],
        })
    return out


def _load_state(session_id):
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT state FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    state = row[0]
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {}
    return state or {}


def _update_state(session_id, **patch):
    """Shallow merge the given keys into migration_session.state."""
    state = _load_state(session_id) or {}
    for k, v in patch.items():
        if v is None and k in state:
            del state[k]
        elif v is not None:
            state[k] = v
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "UPDATE migration_session SET state = %s::jsonb, status = 'in_progress' "
            "WHERE id = %s",
            (json.dumps(state), session_id),
        )
    db.commit()


def _merge_manual_matches(session_id, new_matches):
    state = _load_state(session_id) or {}
    manual = state.get('manual_matches') or {}
    manual.update(new_matches)
    state['manual_matches'] = manual
    # Invalidate final_counts so the user must re-finalize
    state.pop('final_counts', None)
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "UPDATE migration_session SET state = %s::jsonb WHERE id = %s",
            (json.dumps(state), session_id),
        )
    db.commit()


def _mark_album_skipped(session_id, old_album_key):
    state = _load_state(session_id) or {}
    skipped = state.get('skipped_albums') or []
    if old_album_key and old_album_key not in skipped:
        skipped.append(old_album_key)
    state['skipped_albums'] = skipped
    state.pop('final_counts', None)
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "UPDATE migration_session SET state = %s::jsonb WHERE id = %s",
            (json.dumps(state), session_id),
        )
    db.commit()


def _count_score_rows():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM score")
        row = cur.fetchone()
    return int(row[0] or 0) if row else 0
