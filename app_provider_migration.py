"""Provider migration tool — Flask blueprint.

This module is the single add-on entry point for switching the active media
server provider on a running AudioMuse-AI install. It adds a wizard page at
``/provider-migration`` plus the backing REST API under ``/api/migration/*``.

Credentials for the *target* provider stay in ``migration_session.target_creds``
and are passed explicitly to ``tasks.provider_probe`` (which never reads
``config``), so the current live provider keeps working throughout the dry-run
and manual matching steps of the wizard. On successful execution the migration
task writes the new provider settings to ``app_config`` and triggers a config
reload + process restart via ``restart_manager``.
"""
import csv
import io
import json
import logging
import os
import sys

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
        'match_tiers':       result['match_tiers'],
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
                  manual_matches={}, manual_unmatches=[], final_counts=None)

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

    With ``rematch: true`` in the payload, the endpoint also reprocesses
    rows that were already auto-matched for this album — any auto-match
    for this album is discarded and replaced by the new target (with rows
    that don't match in the new target becoming explicit orphans via
    ``manual_unmatches``). This is how step 4 lets the user correct an
    incorrect automatic match.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    old_album_key = payload.get('old_album_key')  # [album_artist, album]
    new_album_id = payload.get('new_album_id')
    rematch = bool(payload.get('rematch'))

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

    old_album_tuple = tuple(old_album_key) if isinstance(old_album_key, list) else old_album_key
    if rematch:
        old_rows = _load_rows_for_album(old_album_tuple)
    else:
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

    if rematch:
        _rematch_album_rows(session_id, newly_matched, still_unmatched)
    else:
        _merge_manual_matches(session_id, newly_matched)
    return jsonify({
        'matched':   len(newly_matched),
        'unmatched': len(still_unmatched),
        'unmatched_item_ids': still_unmatched,
    })


@migration_bp.route('/api/migration/skip-album', methods=['POST'])
def skip_album():
    """Mark an album as explicitly orphaned — those rows will be deleted by
    execute.

    For first-time skips (unmatched albums), a ledger note is enough because
    the merged mapping already doesn't contain these rows. For rematch skips
    (album was already auto-matched), we also have to push every row in the
    album into ``manual_unmatches`` so finalize overrides the existing
    auto-match.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    old_album_key = payload.get('old_album_key')
    rematch = bool(payload.get('rematch'))

    if rematch:
        album_tuple = tuple(old_album_key) if isinstance(old_album_key, list) else old_album_key
        old_rows = _load_rows_for_album(album_tuple)
        all_ids = [r['item_id'] for r in old_rows]
        _rematch_album_rows(session_id, newly_matched={}, newly_unmatched=all_ids)

    _mark_album_skipped(session_id, old_album_key)
    return jsonify({'ok': True})


@migration_bp.route('/api/migration/finalize-dry-run', methods=['POST'])
def finalize_dry_run():
    """Compute final counts and transition session.status to 'dry_run_ready'.

    Final counts expose the same one-to-one dedup logic as execute so the
    user sees any collisions (multiple source rows fighting for the same
    target track) before they type the confirmation phrase. Without this the
    execute transaction would trip ``UNIQUE(new_id)`` on the temp rewrite
    table and roll back with an opaque Postgres error.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')

    state = _load_state(session_id)
    if state is None:
        return jsonify({'error': 'session not found'}), 404

    dry = state.get('dry_run') or {}
    new_meta = state.get('new_meta') or {}

    import importlib
    mig_tasks = importlib.import_module('tasks.provider_migration_tasks')
    merged, dropped = mig_tasks.build_mapping(state)

    total_score = _count_score_rows()
    matched = len(merged)
    collisions = len(dropped)
    # Rows with no match at all = total - (rows that were matched) - (rows
    # dropped by collision dedup). Both collision losers and no-match rows
    # get deleted on execute; showing them separately lets the user decide
    # whether to go back to step 4 and fix the duplicates.
    orphans = max(0, total_score - matched - collisions)

    # Build human-readable collision details so the UI can tell the user
    # exactly which albums to rematch. Only hit the DB for score rows if
    # there's anything to report.
    collision_details = []
    if dropped:
        old_by_id = {r['item_id']: r for r in _load_score_rows_as_dicts()}
        for loser_old_id, new_id, winner_old_id in dropped:
            loser = old_by_id.get(loser_old_id) or {}
            winner = old_by_id.get(winner_old_id) or {}
            tgt = new_meta.get(str(new_id)) or new_meta.get(new_id) or {}
            collision_details.append({
                'loser_title':   loser.get('title') or '',
                'loser_artist':  loser.get('album_artist') or loser.get('author') or '',
                'loser_album':   loser.get('album') or '',
                'loser_path':    loser.get('file_path') or '',
                'winner_title':  winner.get('title') or '',
                'winner_artist': winner.get('album_artist') or winner.get('author') or '',
                'winner_album':  winner.get('album') or '',
                'winner_path':   winner.get('file_path') or '',
                'target_title':  tgt.get('title') or '',
                'target_artist': tgt.get('artist') or '',
                'target_album':  tgt.get('album') or '',
                'target_path':   tgt.get('path') or '',
            })

    final_counts = {
        'matched':            matched,
        'orphans':            orphans,
        'collisions':         collisions,
        'collision_details':  collision_details,
        'tier_counts':        dry.get('tier_counts') or {},
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
    return jsonify({'task_id': job.id})


@migration_bp.route('/api/migration/status/<task_id>', methods=['GET'])
def job_status(task_id):
    try:
        from rq.job import Job
        job = Job.fetch(task_id, connection=redis_conn)
        return jsonify({
            'id': job.id,
            'status': job.get_status(),
            'result': job.result if job.is_finished else None,
            'error': str(job.exc_info) if job.is_failed else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@migration_bp.route('/api/migration/dry-run-report/<int:session_id>', methods=['GET'])
def dry_run_report(session_id):
    """Download the dry-run result as a CSV with both-provider columns.

    Row set = every ``score`` row at dry-run time. Matched rows get both
    old-side and new-side columns populated; orphans (rows that will be
    deleted on execute) have blank new-side columns so the user can see
    exactly what is about to disappear.
    """
    state = _load_state(session_id)
    if state is None:
        return jsonify({'error': 'session not found'}), 404

    dry_run = state.get('dry_run') or {}
    auto_matches     = dry_run.get('matches') or {}
    manual_matches   = state.get('manual_matches') or {}
    manual_unmatches = set(state.get('manual_unmatches') or [])
    new_meta         = state.get('new_meta') or {}

    # Same effective-merge logic as finalize: drop auto rows the user
    # force-orphaned, then manual_matches wins on any remaining conflict.
    matches = {}
    for old_id, new_id in auto_matches.items():
        if old_id not in manual_unmatches:
            matches[old_id] = new_id
    matches.update(manual_matches)

    old_rows = _load_score_rows_as_dicts()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'old_id', 'old_artist', 'old_album', 'old_track', 'old_path',
        'new_id', 'new_artist', 'new_album', 'new_track', 'new_path',
        'match_source',
    ])
    for old in old_rows:
        old_id = old.get('item_id')
        new_id = matches.get(old_id)
        meta = new_meta.get(new_id) if new_id else None
        if new_id and manual_matches.get(old_id):
            source = 'manual'
        elif new_id:
            source = 'auto'
        else:
            source = 'orphan'
        writer.writerow([
            old_id,
            old.get('album_artist') or old.get('author') or '',
            old.get('album') or '',
            old.get('title') or '',
            old.get('file_path') or '',
            new_id or '',
            (meta or {}).get('artist') or '',
            (meta or {}).get('album') or '',
            (meta or {}).get('title') or '',
            (meta or {}).get('path') or '',
            source,
        ])

    from flask import Response
    return Response(
        buf.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                f'attachment; filename=migration_session_{session_id}_dry_run.csv',
        },
    )


@migration_bp.route('/api/migration/matched-albums/<int:session_id>', methods=['GET'])
def matched_albums(session_id):
    """Return all currently-matched albums grouped by old (album_artist, album).

    Powers the step-4 inspection/correction view: the wizard paginates this
    list client-side and lets the user click any row to re-target an album
    whose automatic match was wrong. New-side columns use the most common
    target album across the matched tracks in the group (they're almost
    always unanimous).
    """
    state = _load_state(session_id)
    if state is None:
        return jsonify({'error': 'session not found'}), 404

    dry = state.get('dry_run') or {}
    auto_matches     = dry.get('matches') or {}
    match_tiers      = dry.get('match_tiers') or {}
    manual_matches   = state.get('manual_matches') or {}
    manual_unmatches = set(state.get('manual_unmatches') or [])
    new_meta         = state.get('new_meta') or {}

    merged = {}
    for old_id, new_id in auto_matches.items():
        if old_id not in manual_unmatches:
            merged[old_id] = new_id
    merged.update(manual_matches)

    if not merged:
        return jsonify({'albums': []})

    old_rows = _load_score_rows_as_dicts()
    groups = {}  # (old_artist, old_album) -> {'count', 'new_ids', 'tiers'}
    for r in old_rows:
        old_id = r['item_id']
        new_id = merged.get(old_id)
        if new_id is None:
            continue
        key = (r.get('album_artist') or r.get('author') or '', r.get('album') or '')
        g = groups.setdefault(key, {'count': 0, 'new_ids': [], 'tiers': []})
        g['count'] += 1
        g['new_ids'].append(new_id)
        # Manual rematch wins over the original auto tier if the user changed it.
        if old_id in manual_matches:
            g['tiers'].append('manual')
        else:
            g['tiers'].append(match_tiers.get(old_id) or 'unknown')

    albums = []
    for (old_artist, old_album), g in groups.items():
        tally = {}  # (new_artist, new_album) -> count
        for new_id in g['new_ids']:
            meta = new_meta.get(new_id) or {}
            tally_key = (meta.get('artist') or '', meta.get('album') or '')
            tally[tally_key] = tally.get(tally_key, 0) + 1
        if tally:
            (new_artist, new_album), _ = max(tally.items(), key=lambda kv: kv[1])
        else:
            new_artist, new_album = '', ''
        tier_tally = {}
        for t in g['tiers']:
            tier_tally[t] = tier_tally.get(t, 0) + 1
        dominant_tier = max(tier_tally.items(), key=lambda kv: kv[1])[0] if tier_tally else 'unknown'
        albums.append({
            'old_album_artist': old_artist,
            'old_album':        old_album,
            'track_count':      g['count'],
            'new_album_artist': new_artist,
            'new_album':        new_album,
            'tier':             dominant_tier,
        })

    albums.sort(key=lambda a: (
        (a['old_album_artist'] or '').lower(),
        (a['old_album'] or '').lower(),
    ))
    return jsonify({'albums': albums})


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


def _load_rows_for_album(album_key):
    """Return all old rows in the given (album_artist, album) regardless of
    whether they were matched. Used by the step-4 re-match flow, which needs
    to overwrite existing match state for the whole album at once."""
    target_artist, target_album = (album_key[0] if album_key else None,
                                   album_key[1] if album_key and len(album_key) > 1 else None)
    rows = _load_score_rows_as_dicts()
    out = []
    for r in rows:
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


def _rematch_album_rows(session_id, newly_matched, newly_unmatched):
    """Atomically replace match state for a re-targeted album.

    For each row we found in the new target: put it in manual_matches (which
    wins over dry.matches at finalize time) and make sure it's not stuck in
    manual_unmatches from a previous rematch.

    For each row we could NOT find in the new target: drop any stale
    manual_matches entry and add it to manual_unmatches so finalize treats
    it as an orphan regardless of what dry.matches said.
    """
    state = _load_state(session_id) or {}
    manual = dict(state.get('manual_matches') or {})
    unmatches = set(state.get('manual_unmatches') or [])
    for old_id, new_id in newly_matched.items():
        manual[old_id] = new_id
        unmatches.discard(old_id)
    for old_id in newly_unmatched:
        manual.pop(old_id, None)
        unmatches.add(old_id)
    state['manual_matches']   = manual
    state['manual_unmatches'] = sorted(unmatches)
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
