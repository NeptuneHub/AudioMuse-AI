"""Provider migration tool - Flask blueprint.

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

from flask import Blueprint, jsonify, render_template, request
from psycopg2 import sql as pgsql

# App-level singletons (DB connection, Redis, RQ queues). Importing here keeps
# the blueprint file self-contained - the rest of the app doesn't need to hand
# anything in.
from database import get_db
from taskqueue import redis_conn, rq_queue_high
from ssrf_guard import validate_outbound_url
from tasks.mediaserver.helper import detect_path_format as _detect_path_format

logger = logging.getLogger(__name__)

migration_bp = Blueprint('migration_bp', __name__)


# ---------------------------------------------------------------------------
# Lazy provider_probe import - keeps the _import_module bypass test happy
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

_SUPPORTED_TARGETS = frozenset({'jellyfin', 'navidrome', 'emby', 'lyrion'})


# ---------------------------------------------------------------------------
# SSRF guard for the user-supplied media-server URL. Delegates to the shared
# ``ssrf_guard.validate_outbound_url`` (allows LAN/loopback, blocks non-HTTP(S)
# schemes and link-local/cloud-metadata). A missing url is allowed and left to
# the downstream probe.
# ---------------------------------------------------------------------------


def _validate_probe_url(creds):
    """Return (True, None) if ``creds['url']`` is safe to fetch, else (False, reason)."""
    url = (creds or {}).get('url')
    if not url:
        return True, None
    return validate_outbound_url(url)


# ---------------------------------------------------------------------------
# Source path sanity check - matching tiers 1 (path) and 2 (path tail) need
# absolute filesystem paths in ``score.file_path``. If the user's current
# provider stored garbage (Navidrome without Report Real Path, Lyrion stream
# URIs, etc.), we can re-probe the current provider to get real paths and
# apply them to ``old_rows`` before matching.
# ---------------------------------------------------------------------------

_SOURCE_PATH_SAMPLE_SIZE = 100


def _sample_score_file_paths(limit=_SOURCE_PATH_SAMPLE_SIZE):
    """Return up to ``limit`` ``file_path`` values from the score table."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT file_path FROM score WHERE file_path IS NOT NULL LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall() or []
    return [r[0] for r in rows]


def _detect_source_path_format():
    """Classify ``score.file_path`` values by sampling and running
    the shared path-format helper. Returns one of
    ``'absolute' | 'relative' | 'none' | 'mixed'``.
    """
    samples = _sample_score_file_paths()
    tracks = [{'path': p} for p in samples]
    return _detect_path_format(tracks)


def _current_provider_creds():
    """Build a creds dict from ``config`` for the currently active provider.

    Returns ``(provider_type, creds_dict)`` or ``(None, {})`` when the
    provider isn't one we can re-probe.
    """
    import config as cfg

    t = (getattr(cfg, 'MEDIASERVER_TYPE', '') or '').lower()
    if t == 'jellyfin':
        return t, {
            'url': getattr(cfg, 'JELLYFIN_URL', ''),
            'user_id': getattr(cfg, 'JELLYFIN_USER_ID', ''),
            'token': getattr(cfg, 'JELLYFIN_TOKEN', ''),
        }
    if t == 'emby':
        return t, {
            'url': getattr(cfg, 'EMBY_URL', ''),
            'user_id': getattr(cfg, 'EMBY_USER_ID', ''),
            'token': getattr(cfg, 'EMBY_TOKEN', ''),
        }
    if t == 'navidrome':
        return t, {
            'url': getattr(cfg, 'NAVIDROME_URL', ''),
            'user': getattr(cfg, 'NAVIDROME_USER', ''),
            'password': getattr(cfg, 'NAVIDROME_PASSWORD', ''),
        }
    if t == 'lyrion':
        return t, {'url': getattr(cfg, 'LYRION_URL', '')}
    return None, {}


def _apply_source_path_overrides(old_rows, overrides):
    """Patch ``old_rows[i]['file_path']`` from the overrides dict in place.

    Pure function: the caller runs it before handing ``old_rows`` to the
    matcher, so matcher tests don't need to know about overrides at all.
    """
    if not overrides:
        return old_rows
    for r in old_rows:
        real = overrides.get(r.get('item_id'))
        if real:
            r['file_path'] = real
    return old_rows


# ---------------------------------------------------------------------------
# Routes - wizard page
# ---------------------------------------------------------------------------


@migration_bp.route('/provider-migration')
def provider_migration_page():
    """
    Provider migration wizard page.
    ---
    tags:
      - Provider Migration
    summary: HTML wizard for migrating analysis state between media-server providers (Jellyfin/Emby/Navidrome/Lyrion).
    description: Resumes any in-flight session so a page refresh lands on the right step.
    responses:
      200:
        description: Wizard HTML rendered with `active_session_id` if a non-terminal session exists.
    """
    # Look up an in-flight migration so a page refresh can resume the wizard
    # at the right step instead of creating a brand new session.
    active_session_id = None
    try:
        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                "SELECT id FROM migration_session "
                "WHERE status NOT IN ('completed', 'failed') "
                "ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
        if row:
            active_session_id = row[0]
    except Exception as e:
        logger.warning(
            "provider_migration_page: failed to look up active session: %s",
            e,
            exc_info=True,
        )
        active_session_id = None

    return render_template(
        'provider_migration.html',
        title='Provider Migration',
        active='provider_migration',
        active_session_id=active_session_id,
    )


# ---------------------------------------------------------------------------
# Routes - session CRUD
# ---------------------------------------------------------------------------


@migration_bp.route('/api/migration/session/start', methods=['POST'])
def session_start():
    """
    Start a new migration session.
    ---
    tags:
      - Provider Migration
    summary: Create a `migration_session` row and prune any already-terminal sessions.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [target_type, target_creds]
            properties:
              target_type:
                type: string
                enum: [jellyfin, emby, navidrome, lyrion]
              target_creds:
                type: object
                additionalProperties: true
    responses:
      200:
        description: Session id returned.
        content:
          application/json:
            schema:
              type: object
              properties:
                session_id:
                  type: integer
      400:
        description: Unsupported target_type.
    """
    payload = request.get_json(silent=True) or {}
    target_type = (payload.get('target_type') or '').lower()
    target_creds = payload.get('target_creds') or {}

    if target_type not in _SUPPORTED_TARGETS:
        return jsonify({'error': f'target_type must be one of {sorted(_SUPPORTED_TARGETS)}'}), 400

    ok, reason = _validate_probe_url(target_creds)
    if not ok:
        return jsonify({'error': f'target_creds url is not allowed: {reason}'}), 400

    import config

    source_type = getattr(config, 'MEDIASERVER_TYPE', '') or ''

    db = get_db()
    with db.cursor() as cur:
        # Prune terminal rows so the table does not grow unboundedly.
        # Safe: never touches in-flight sessions (in_progress / dry_run_ready).
        cur.execute("DELETE FROM migration_session WHERE status IN ('completed', 'failed')")
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
    """
    Inspect a migration session.
    ---
    tags:
      - Provider Migration
    summary: Return current status and JSON state for a session.
    parameters:
      - name: session_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Session summary.
        content:
          application/json:
            schema:
              type: object
              properties:
                id:
                  type: integer
                source_type:
                  type: string
                target_type:
                  type: string
                status:
                  type: string
                  enum: [in_progress, dry_run_ready, completed, failed]
                state:
                  type: object
      404:
        description: Session not found.
    """
    db = get_db()
    with db.cursor() as cur:
        # Strip the two heavy keys the wizard UI never reads (the per-row
        # auto-match map and the source-path override map). Server-side ``#-``
        # keeps a 100k-entry, tens-of-MB blob from being shipped to the browser
        # on every step-4 render.
        cur.execute(
            "SELECT id, source_type, target_type, status, "
            "(state #- '{dry_run,matches}' #- '{source_path_overrides}') "
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
    return jsonify(
        {
            'id': _id,
            'source_type': source_type,
            'target_type': target_type,
            'status': status,
            'state': state,
        }
    )


# ---------------------------------------------------------------------------
# Routes - probe (delegates to tasks.provider_probe, passes creds explicitly)
# ---------------------------------------------------------------------------


@migration_bp.route('/api/migration/session/<int:session_id>', methods=['DELETE'])
def session_discard(session_id):
    """
    Discard an in-flight migration session.
    ---
    tags:
      - Provider Migration
    summary: Delete a non-terminal session row (used by the wizard's Discard button).
    description: |
      Refuses to touch sessions in `completed` or `failed` status - those are
      pruned automatically on the next `session_start`.
    parameters:
      - name: session_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Session deleted.
      400:
        description: Session is already in a terminal state.
      404:
        description: Session not found.
    """
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM migration_session WHERE id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return jsonify({'error': 'session not found'}), 404
        if row[0] in ('completed', 'failed'):
            return jsonify({'error': 'cannot discard a finished session'}), 400
        cur.execute("DELETE FROM migration_session WHERE id = %s", (session_id,))
    db.commit()
    return jsonify({'ok': True})


@migration_bp.route('/api/migration/probe/test', methods=['POST'])
def probe_test():
    """
    Test a target-provider connection.
    ---
    tags:
      - Provider Migration
    summary: Probe a media-server provider with given credentials and report path quality.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [type, creds]
            properties:
              type:
                type: string
                enum: [jellyfin, emby, navidrome, lyrion]
              creds:
                type: object
                additionalProperties: true
    responses:
      200:
        description: Probe result (always 200; check `ok` for success).
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                error:
                  type: string
                path_format:
                  type: string
                  enum: [absolute, relative, virtual, none]
                sample_count:
                  type: integer
                warnings:
                  type: array
                  items:
                    type: string
    """
    payload = request.get_json(silent=True) or {}
    t = (payload.get('type') or '').lower()
    creds = payload.get('creds') or {}
    ok, reason = _validate_probe_url(creds)
    if not ok:
        return jsonify(
            {'ok': False, 'error': reason, 'path_format': 'none', 'sample_count': 0, 'warnings': []}
        ), 200
    try:
        result = provider_probe.test_connection(t, creds)
    except NotImplementedError:
        logger.warning("test_connection not supported for provider type %s", t)
        return jsonify(
            {
                'ok': False,
                'error': 'Connection testing is not supported for this provider.',
                'path_format': 'none',
                'sample_count': 0,
                'warnings': [],
            }
        ), 200
    except Exception:
        logger.warning("test_connection failed for provider type %s", t, exc_info=True)
        return jsonify(
            {
                'ok': False,
                'error': 'Connection test failed. Check the container logs for details.',
                'path_format': 'none',
                'sample_count': 0,
                'warnings': [],
            }
        ), 200
    return jsonify(result)


@migration_bp.route('/api/migration/libraries', methods=['POST'])
def libraries_list():
    """
    List target-provider music libraries.
    ---
    tags:
      - Provider Migration
    summary: Step 2 - return the target provider's libraries plus the user's prior checkbox selection.
    description: Uses session-stored credentials, never `config`, so the live provider keeps working.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
    responses:
      200:
        description: Library list (always 200; check `error` for failures).
        content:
          application/json:
            schema:
              type: object
              properties:
                libraries:
                  type: array
                  items:
                    type: object
                unsupported:
                  type: boolean
                selected_libraries:
                  type: array
                  items:
                    type: string
                error:
                  type: string
      400:
        description: Missing session_id.
      404:
        description: Session not found.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    if session_id is None:
        return jsonify({'error': 'session_id is required'}), 400

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404
    target_type, creds = session
    state = _load_state(session_id) or {}
    selected = state.get('selected_libraries')
    try:
        result = provider_probe.list_libraries(target_type, creds)
    except Exception as e:
        logger.warning("libraries_list failed for session %s: %s", session_id, e, exc_info=True)
        return jsonify(
            {
                'libraries': [],
                'unsupported': False,
                'selected_libraries': selected,
                'error': 'Failed to list libraries. Check the container logs for details.',
            }
        ), 200
    return jsonify(
        {
            'libraries': result.get('libraries', []),
            'unsupported': bool(result.get('unsupported', False)),
            'selected_libraries': selected,
        }
    ), 200


@migration_bp.route('/api/migration/libraries/select', methods=['POST'])
def libraries_select():
    """
    Persist library selection into session state.
    ---
    tags:
      - Provider Migration
    summary: Step 2 - save the user's library checkbox selection (null = no filter, [] = normalized to null).
    description: |
      Library names cannot contain commas because `MUSIC_LIBRARIES` is stored
      as a comma-separated string and split at scan time.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
              libraries:
                type: array
                nullable: true
                items:
                  type: string
    responses:
      200:
        description: Selection saved.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                selected_libraries:
                  type: array
                  nullable: true
                  items:
                    type: string
      400:
        description: Missing session_id, libraries not a list, or comma-containing library name.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    if session_id is None:
        return jsonify({'error': 'session_id is required'}), 400

    libraries = payload.get('libraries')
    if libraries is not None and not isinstance(libraries, list):
        return jsonify({'error': 'libraries must be a list of names or null'}), 400

    if isinstance(libraries, list):
        cleaned = [str(name).strip() for name in libraries if str(name).strip()]
        # MUSIC_LIBRARIES is stored as a comma-separated string and split on
        # ',' at scan time, so a name containing a comma would silently
        # corrupt the round-trip into multiple bogus fragments.
        if any(',' in name for name in cleaned):
            return jsonify({'error': 'Library names cannot contain commas.'}), 400
        selected = cleaned or None
    else:
        selected = None

    _update_state(session_id, selected_libraries=selected)
    return jsonify({'ok': True, 'selected_libraries': selected}), 200


@migration_bp.route('/api/migration/search-albums', methods=['POST'])
def search_albums():
    """
    Search target-provider albums.
    ---
    tags:
      - Provider Migration
    summary: Free-text album search against the target provider (used by step 4 manual matching).
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
              query:
                type: string
    responses:
      200:
        description: Album candidates.
        content:
          application/json:
            schema:
              type: object
              properties:
                albums:
                  type: array
                  items:
                    type: object
      404:
        description: Session not found.
      500:
        description: Provider error during search.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    query = payload.get('query') or ''

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404
    target_type, creds = session
    try:
        albums = provider_probe.search_albums(target_type, creds, query)
    except Exception:
        logger.warning("search_albums failed for session %s", session_id, exc_info=True)
        return jsonify({'error': 'Album search failed. Check the container logs for details.'}), 500
    return jsonify({'albums': albums})


# ---------------------------------------------------------------------------
# Routes - dry run, manual match, finalize
# ---------------------------------------------------------------------------


@migration_bp.route('/api/migration/source-paths/refresh', methods=['POST'])
def source_paths_refresh():
    """
    Refresh source-provider real paths.
    ---
    tags:
      - Provider Migration
    summary: Re-probe the currently active provider to build a {item_id -> real_path} override map.
    description: |
      Called when `score.file_path` is unusable (e.g. Navidrome analyzed
      without "Report Real Path"). After refresh, the dry-run can use the
      fresh paths for matcher tiers 1 and 2 without rebuilding analysis
      from scratch.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
    responses:
      200:
        description: Refresh result with override count and any warnings.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
                source_type:
                  type: string
                path_format:
                  type: string
                overrides_count:
                  type: integer
                warnings:
                  type: array
                  items:
                    type: string
      400:
        description: Missing session_id, or current provider doesn't support path refresh.
      500:
        description: Provider probe failed.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    if session_id is None:
        return jsonify({'error': 'session_id is required'}), 400

    # Cheap support check (reads config) stays in the request; the full
    # source-catalog fetch is offloaded to an RQ worker like the dry-run.
    source_type, _ = _current_provider_creds()
    if not source_type:
        return jsonify(
            {
                'ok': False,
                'error': 'The current provider does not support path refresh.',
            }
        ), 400

    job = rq_queue_high.enqueue(
        'tasks.provider_migration_tasks.source_refresh_provider_migration',
        session_id,
        job_timeout=3600,
    )
    _patch_state_keys(session_id, source_refresh_task_id=job.id)
    return jsonify({'task_id': job.id, 'async': True})


def run_source_refresh_core(session_id):
    """Heavy source-path refresh, run in an RQ worker (see
    :func:`source_paths_refresh`). Re-probes the current provider, builds the
    {item_id -> real_path} override map, and stores it in ``state``."""
    source_type, creds = _current_provider_creds()
    if not source_type:
        raise RuntimeError('The current provider does not support path refresh.')

    tracks = provider_probe.fetch_all_tracks(source_type, creds)

    path_format = _detect_path_format(tracks)
    overrides = {t['id']: t['path'] for t in tracks if t.get('id') and t.get('path')}

    warnings = []
    if path_format != 'absolute':
        warnings.append(
            f'{source_type} is still not returning absolute paths. '
            'Double-check that "Report Real Path" (Navidrome) or the '
            'equivalent setting is enabled, then refresh again. You can '
            'also proceed with metadata-only matching.'
        )

    _update_state(session_id, source_path_overrides=overrides, source_refresh_task_id=None)
    return {
        'ok': True,
        'source_type': source_type,
        'path_format': path_format,
        'overrides_count': len(overrides),
        'warnings': warnings,
    }


@migration_bp.route('/api/migration/dry-run', methods=['POST'])
def dry_run():
    """
    Run the migration matcher (dry-run).
    ---
    tags:
      - Provider Migration
    summary: Step 3 - match score rows against the target provider's tracks and persist the result.
    description: |
      Source `score.file_path` values are sanity-checked first. If they don't
      look like absolute filesystem paths, the endpoint returns **409** with
      `needs_source_refresh=true` so the UI can prompt the user to enable
      "Report Real Path" and call `/source-paths/refresh`. Pass
      `bypass_source_check=true` to skip the gate and use metadata-only
      matching.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
              bypass_source_check:
                type: boolean
                default: false
              allow_title_artist_only:
                type: boolean
                default: false
                description: Allow the matcher to fall back to title+artist when album metadata differs.
    responses:
      200:
        description: Dry-run summary.
        content:
          application/json:
            schema:
              type: object
              properties:
                tier_counts:
                  type: object
                matched:
                  type: integer
                unmatched:
                  type: integer
                unmatched_albums_count:
                  type: integer
      404:
        description: Session not found.
      409:
        description: Source paths look unusable; refresh required.
      500:
        description: Target provider error.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')
    bypass_source_check = bool(payload.get('bypass_source_check'))
    allow_title_artist_only = bool(payload.get('allow_title_artist_only'))

    session = _fetch_session_creds(session_id)
    if session is None:
        return jsonify({'error': 'session not found'}), 404

    # Gate on source path quality (cheap - samples 100 rows). Stays in the
    # request so the UI can prompt for a refresh. Skip if the user has already
    # refreshed (overrides present) or opted into metadata-only matching.
    state = _load_state(session_id) or {}
    source_overrides = state.get('source_path_overrides') or {}
    if not source_overrides and not bypass_source_check:
        source_format = _detect_source_path_format()
        if source_format != 'absolute':
            source_type, _ = _current_provider_creds()
            return jsonify(
                {
                    'needs_source_refresh': True,
                    'current_source_type': source_type,
                    'path_format': source_format,
                    'hint': (
                        'Your score.file_path values are not absolute filesystem '
                        'paths. Automatic path-based matching will fall back to '
                        'metadata only. Refresh source paths, or proceed with '
                        'metadata-only matching.'
                    ),
                }
            ), 409

    # The heavy work (fetch the whole target catalog + match every score row +
    # persist) can take minutes on 100k+ libraries - far past the gunicorn
    # request timeout - so it runs in an RQ worker; the UI polls the status.
    job = rq_queue_high.enqueue(
        'tasks.provider_migration_tasks.dry_run_provider_migration',
        session_id,
        allow_title_artist_only,
        job_timeout=3600,
    )
    _patch_state_keys(session_id, dry_run_task_id=job.id)
    return jsonify({'task_id': job.id, 'async': True})


def run_dry_run_core(session_id, allow_title_artist_only=False):
    """Heavy dry-run work, run in an RQ worker (see :func:`dry_run`).

    Fetches the target catalog, matches it against every score row, and
    persists the result (summary + ``matches`` into ``state``, target metadata
    into the ``migration_target_meta`` side table). Returns the summary dict the
    wizard renders, or ``{'error': ...}`` for a controlled abort.
    """
    session = _fetch_session_creds(session_id)
    if session is None:
        raise RuntimeError(f'migration session {session_id} not found')
    target_type, creds = session

    new_tracks = provider_probe.fetch_all_tracks(target_type, creds)

    # Safety guard: a target that returns zero tracks (transient outage, wrong
    # creds, empty/mis-scoped library) would make EVERY score row an orphan,
    # and execute would then delete the entire library. Refuse instead of
    # producing an all-orphan plan. Returned (not raised) so the UI shows the
    # reason rather than a generic failure.
    if not new_tracks:
        logger.warning(
            "provider migration dry-run: target '%s' returned 0 tracks; aborting "
            "to avoid orphaning the whole library (session %s)",
            target_type,
            session_id,
        )
        _patch_state_keys(session_id, dry_run_task_id=None)
        return {
            'error': (
                'The new provider returned 0 tracks. Aborting so your library is '
                'not deleted as orphans. Check the connection / library selection '
                'and run automatic matching again.'
            )
        }

    state = _load_state(session_id) or {}
    source_overrides = state.get('source_path_overrides') or {}
    old_rows = _load_score_rows_as_dicts()
    _apply_source_path_overrides(old_rows, source_overrides)

    import importlib

    matcher = importlib.import_module('tasks.provider_migration_matcher')
    result = matcher.match_tracks(
        old_rows,
        new_tracks,
        allow_title_artist_only=allow_title_artist_only,
    )

    state_dry_run = {
        'matches': result['matches'],
        'tier_counts': result['tier_counts'],
        'unmatched_albums': _albums_payload(result['unmatched_by_album']),
        # Full count so the wizard can warn when the rendered list is a sample.
        'unmatched_albums_total': len(result['unmatched_by_album']),
    }
    new_meta = {
        n['id']: {
            'path': n.get('path'),
            'title': n.get('title'),
            'artist': n.get('artist'),
            'album': n.get('album'),
            'album_artist': n.get('album_artist'),
            'year': n.get('year'),
        }
        for n in new_tracks
        if n.get('id')
    }
    _store_target_meta(session_id, new_meta)
    _update_state(
        session_id,
        dry_run=state_dry_run,
        manual_matches={},
        manual_unmatches=[],
        final_counts=None,
        dry_run_task_id=None,
    )

    return {
        'tier_counts': result['tier_counts'],
        'matched': len(result['matches']),
        'unmatched': len(result['unmatched']),
        'unmatched_albums_count': len(result['unmatched_by_album']),
    }


@migration_bp.route('/api/migration/match-album', methods=['POST'])
def match_album():
    """
    Manually match an album.
    ---
    tags:
      - Provider Migration
    summary: Step 4 - user picked a target album; auto-match its tracks by title (or rematch existing auto-matches).
    description: |
      With `rematch=true`, the endpoint reprocesses rows that were already
      auto-matched for this album: any auto-match for the album is discarded
      and replaced by the new target. Rows that don't match in the new target
      become explicit orphans via `manual_unmatches`.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id, old_album_key, new_album_id]
            properties:
              session_id:
                type: integer
              old_album_key:
                type: array
                items:
                  type: string
                description: "[album_artist, album]"
              new_album_id:
                type: string
              rematch:
                type: boolean
                default: false
    responses:
      200:
        description: Match result for the album.
        content:
          application/json:
            schema:
              type: object
              properties:
                matched:
                  type: integer
                unmatched:
                  type: integer
                unmatched_item_ids:
                  type: array
                  items:
                    type: string
      404:
        description: Session not found.
      500:
        description: Target provider error.
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
    except Exception:
        logger.warning("get_album_tracks failed for session %s", session_id, exc_info=True)
        return jsonify(
            {'error': 'Failed to fetch album tracks. Check the container logs for details.'}
        ), 500

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
    return jsonify(
        {
            'matched': len(newly_matched),
            'unmatched': len(still_unmatched),
            'unmatched_item_ids': still_unmatched,
        }
    )


@migration_bp.route('/api/migration/skip-album', methods=['POST'])
def skip_album():
    """
    Skip an album (mark its rows as orphans).
    ---
    tags:
      - Provider Migration
    summary: Step 4 - orphan an album so its score rows will be deleted by execute.
    description: |
      First-time skips (unmatched albums) just need a ledger note. Rematch
      skips (`rematch=true`) push every row in the album into
      `manual_unmatches` so finalize overrides the existing auto-match.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id, old_album_key]
            properties:
              session_id:
                type: integer
              old_album_key:
                type: array
                items:
                  type: string
              rematch:
                type: boolean
                default: false
    responses:
      200:
        description: Album marked as skipped.
        content:
          application/json:
            schema:
              type: object
              properties:
                ok:
                  type: boolean
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
    """
    Finalize the dry-run.
    ---
    tags:
      - Provider Migration
    summary: Compute final counts (with collision dedup) and flip status to `dry_run_ready`.
    description: |
      Runs the same one-to-one dedup logic as `execute` so the user sees any
      collisions (multiple source rows fighting for the same target track)
      before typing the confirmation phrase. Without this, execute would trip
      `UNIQUE(new_id)` on the temp rewrite table and roll back with an opaque
      Postgres error.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id]
            properties:
              session_id:
                type: integer
    responses:
      200:
        description: Final counts including collision details.
        content:
          application/json:
            schema:
              type: object
              properties:
                matched:
                  type: integer
                orphans:
                  type: integer
                collisions:
                  type: integer
                collision_details:
                  type: array
                  items:
                    type: object
                tier_counts:
                  type: object
      404:
        description: Session not found.
    """
    payload = request.get_json(silent=True) or {}
    session_id = payload.get('session_id')

    state = _load_state(session_id)
    if state is None:
        return jsonify({'error': 'session not found'}), 404

    dry = state.get('dry_run') or {}

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

    import config

    collision_details_total = collisions
    # Build human-readable collision details so the UI can tell the user
    # exactly which albums to rematch. Only the capped subset is rendered, so
    # fetch only those score rows / target metadata (by id) rather than the
    # whole catalog.
    collision_details = []
    if dropped:
        dropped_for_details = dropped[: config.MIGRATION_MAX_COLLISION_DETAILS]
        needed_old_ids = set()
        needed_new_ids = set()
        for loser_old_id, new_id, winner_old_id in dropped_for_details:
            needed_old_ids.add(loser_old_id)
            needed_old_ids.add(winner_old_id)
            needed_new_ids.add(str(new_id))
        old_by_id = _load_score_rows_by_ids(needed_old_ids)
        meta_by_id = _load_target_meta(session_id, needed_new_ids)
        for loser_old_id, new_id, winner_old_id in dropped_for_details:
            loser = old_by_id.get(loser_old_id) or {}
            winner = old_by_id.get(winner_old_id) or {}
            tgt = meta_by_id.get(str(new_id)) or {}
            collision_details.append(
                {
                    'loser_title': loser.get('title') or '',
                    'loser_artist': loser.get('album_artist') or loser.get('author') or '',
                    'loser_album': loser.get('album') or '',
                    'loser_path': loser.get('file_path') or '',
                    'winner_title': winner.get('title') or '',
                    'winner_artist': winner.get('album_artist') or winner.get('author') or '',
                    'winner_album': winner.get('album') or '',
                    'winner_path': winner.get('file_path') or '',
                    'target_title': tgt.get('title') or '',
                    'target_artist': tgt.get('artist') or '',
                    'target_album': tgt.get('album') or '',
                    'target_path': tgt.get('path') or '',
                }
            )

    final_counts = {
        'matched': matched,
        'orphans': orphans,
        'collisions': collisions,
        'collision_details': collision_details,
        'collision_details_total': collision_details_total,
        'tier_counts': dry.get('tier_counts') or {},
    }

    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "UPDATE migration_session SET "
            "  state = jsonb_set(state, '{final_counts}', %s::jsonb, true), "
            "  status = 'dry_run_ready' "
            "WHERE id = %s",
            (json.dumps(_sanitize_json_value(final_counts), ensure_ascii=False), session_id),
        )
    db.commit()
    return jsonify(final_counts)


# ---------------------------------------------------------------------------
# Routes - execute gate + status
# ---------------------------------------------------------------------------


@migration_bp.route('/api/migration/execute', methods=['POST'])
def execute():
    """
    Execute the migration.
    ---
    tags:
      - Provider Migration
    summary: Step 5 - gate on backup checkbox + confirmation phrase, then enqueue the execute job.
    description: |
      Requires the session to be in `dry_run_ready` status. The confirmation
      phrase must equal exactly:
      `I want to migrate to <target_type> and delete unmatched tracks`.
      The job rewrites every score row's `item_id` and then deletes orphans.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [session_id, backup_confirmed, confirmation_text]
            properties:
              session_id:
                type: integer
              backup_confirmed:
                type: boolean
                description: Must be true.
              confirmation_text:
                type: string
                description: Must equal the per-target confirmation phrase exactly.
    responses:
      200:
        description: Execute task enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
      400:
        description: Missing backup confirmation, wrong confirmation phrase, or session not in `dry_run_ready` state.
      404:
        description: Session not found.
    """
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
        return jsonify(
            {'error': f'Confirmation text does not match. Expected exactly: "{expected}"'}
        ), 400
    if status != 'dry_run_ready':
        return jsonify(
            {
                'error': f'Dry run must be finalized first. Session status is "{status}", '
                f'expected "dry_run_ready".'
            }
        ), 400

    # Enqueue the execute job
    from rq.job import Job  # noqa: F401  (used by enqueue internals)

    job = rq_queue_high.enqueue(
        'tasks.provider_migration_tasks.execute_provider_migration',
        session_id,
        job_timeout=3600,
    )
    # Persist the RQ task id on the session so a page refresh can resume
    # polling this job rather than losing track of it.
    try:
        _patch_state_keys(session_id, exec_task_id=job.id)
    except Exception as e:
        # Non-fatal: the execute job is already enqueued. Losing exec_task_id
        # only means the UI cannot auto-resume polling after a page refresh.
        logger.warning(
            "provider_migration execute: failed to persist exec_task_id "
            "for session %s (job %s): %s",
            session_id,
            job.id,
            e,
            exc_info=True,
        )
        try:
            db.rollback()
        except Exception:
            pass
    return jsonify({'task_id': job.id})


# Track which finished migration jobs have already triggered a Flask restart,
# so repeated status polls don't schedule the restart multiple times.
_restart_scheduled_for_tasks = set()


@migration_bp.route('/api/migration/status/<task_id>', methods=['GET'])
def job_status(task_id):
    """
    Poll the migration execute task.
    ---
    tags:
      - Provider Migration
    summary: Return RQ job status; on completion, schedule a Flask restart so config reloads.
    description: |
      When the job finishes, this endpoint reloads `config` in this Flask
      process and (once per task_id) schedules a graceful restart so any
      module-level `from config import X` bindings are rebuilt against the
      new provider settings.
    parameters:
      - name: task_id
        in: path
        required: true
        schema: { type: string }
    responses:
      200:
        description: Job status payload.
        content:
          application/json:
            schema:
              type: object
              properties:
                id:
                  type: string
                status:
                  type: string
                  enum: [queued, started, finished, failed, deferred]
                result:
                  nullable: true
                error:
                  type: string
                  nullable: true
                restart_scheduled:
                  type: boolean
      404:
        description: Job not found.
    """
    try:
        from rq.job import Job

        job = Job.fetch(task_id, connection=redis_conn)
        status = job.get_status()
        restart_scheduled = False
        # Only the EXECUTE job changes the active provider and needs the
        # config-reload + Flask restart. The dry-run / source-refresh jobs
        # share this status endpoint but must NOT restart Flask.
        is_execute_job = (getattr(job, 'func_name', '') or '').endswith(
            'execute_provider_migration'
        )
        # The execute worker reloads its own config and publishes a restart
        # request for other workers, but Flask (this process) isn't on that
        # pub/sub path. Reload here when the job finishes so subsequent
        # requests see the new provider, then schedule a Flask restart so
        # any stale module-level `from config import X` bindings across
        # blueprints are rebuilt cleanly (mirrors the setup wizard).
        if status == 'finished' and is_execute_job:
            try:
                import config as _cfg

                _cfg.refresh_config()
            except Exception as _e:
                logger.warning("post-migration config reload failed: %s", _e)
            if task_id not in _restart_scheduled_for_tasks:
                try:
                    import restart_manager

                    if restart_manager.schedule_flask_restart():
                        restart_scheduled = True
                        _restart_scheduled_for_tasks.add(task_id)
                except Exception as _e:
                    logger.warning("post-migration Flask restart scheduling failed: %s", _e)
            else:
                restart_scheduled = True
        return jsonify(
            {
                'id': job.id,
                'status': status,
                'result': job.result if job.is_finished else None,
                'error': 'Job failed. Check the container logs for details.'
                if job.is_failed
                else None,
                'restart_scheduled': restart_scheduled,
            }
        )
    except Exception:
        logger.warning("migration job status fetch failed for task %s", task_id, exc_info=True)
        return jsonify({'error': 'Job not found.'}), 404


@migration_bp.route('/api/migration/dry-run-report/<int:session_id>', methods=['GET'])
def dry_run_report(session_id):
    """
    Download the dry-run report as CSV.
    ---
    tags:
      - Provider Migration
    summary: CSV showing the planned old->new mapping for every score row (orphans have blank new-side cells).
    description: |
      Columns: old_id, old_artist, old_album, old_album_artist, old_track, old_path, new_id,
      new_artist, new_album, new_album_artist, new_track, new_path, match_source
      (`auto`/`manual`/`orphan`).
    parameters:
      - name: session_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: CSV attachment.
        content:
          text/csv:
            schema:
              type: string
      404:
        description: Session not found.
    """
    state = _load_state(session_id)
    if state is None:
        return jsonify({'error': 'session not found'}), 404

    dry_run = state.get('dry_run') or {}
    auto_matches = dry_run.get('matches') or {}
    manual_matches = state.get('manual_matches') or {}
    manual_unmatches = set(state.get('manual_unmatches') or [])
    new_meta = _load_target_meta(session_id)

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
    writer.writerow(
        [
            'old_id',
            'old_artist',
            'old_album',
            'old_album_artist',
            'old_track',
            'old_path',
            'new_id',
            'new_artist',
            'new_album',
            'new_album_artist',
            'new_track',
            'new_path',
            'match_source',
        ]
    )
    for old in old_rows:
        old_id = old.get('item_id')
        new_id = matches.get(old_id)
        meta = (new_meta.get(str(new_id)) or new_meta.get(new_id)) if new_id else None
        if new_id and manual_matches.get(old_id):
            source = 'manual'
        elif new_id:
            source = 'auto'
        else:
            source = 'orphan'
        writer.writerow(
            [
                old_id,
                old.get('author') or old.get('album_artist') or '',
                old.get('album') or '',
                old.get('album_artist') or '',
                old.get('title') or '',
                old.get('file_path') or '',
                new_id or '',
                (meta or {}).get('artist') or '',
                (meta or {}).get('album') or '',
                (meta or {}).get('album_artist') or '',
                (meta or {}).get('title') or '',
                (meta or {}).get('path') or '',
                source,
            ]
        )

    from flask import Response

    return Response(
        buf.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=migration_session_{session_id}_dry_run.csv',
        },
    )


@migration_bp.route('/api/migration/matched-albums/<int:session_id>', methods=['GET'])
def matched_albums(session_id):
    """
    List currently-matched albums.
    ---
    tags:
      - Provider Migration
    summary: Step 4 review - return albums grouped by old (album_artist, album) with their target album, used for the wizard's correction view.
    description: |
      Auto-matched rows are skipped to keep the review list focused on
      albums the user (or rematch flows) explicitly modified. New-side
      columns use the most common target album across the matched tracks
      in each group.
    parameters:
      - name: session_id
        in: path
        required: true
        schema: { type: integer }
    responses:
      200:
        description: Grouped matched-album list.
        content:
          application/json:
            schema:
              type: object
              properties:
                albums:
                  type: array
                  items:
                    type: object
      404:
        description: Session not found.
    """
    # The review list only ever shows manually re-targeted albums, so we need
    # just ``manual_matches`` (small) - not the full state blob, the auto-match
    # map, or the whole score table. Load only those rows + their target meta.
    found, manual_matches = _read_state_key(session_id, 'manual_matches')
    if not found:
        return jsonify({'error': 'session not found'}), 404
    manual_matches = manual_matches or {}
    if not manual_matches:
        return jsonify({'albums': []})

    old_rows = list(_load_score_rows_by_ids(manual_matches.keys()).values())
    new_meta = _load_target_meta(session_id, list(manual_matches.values()))
    groups = {}  # (old_artist, old_album) -> {'count', 'new_ids', 'tiers'}
    for r in old_rows:
        old_id = r['item_id']
        new_id = manual_matches.get(old_id)
        if new_id is None:
            continue
        key = (r.get('album_artist') or r.get('author') or '', r.get('album') or '')
        g = groups.setdefault(key, {'count': 0, 'new_ids': [], 'tiers': []})
        g['count'] += 1
        g['new_ids'].append(new_id)
        g['tiers'].append('manual')

    albums = []
    for (old_artist, old_album), g in groups.items():
        tally = {}  # (new_artist, new_album) -> count
        for new_id in g['new_ids']:
            meta = new_meta.get(str(new_id)) or new_meta.get(new_id) or {}
            tally_key = (
                meta.get('album_artist') or meta.get('artist') or '',
                meta.get('album') or '',
            )
            tally[tally_key] = tally.get(tally_key, 0) + 1
        if tally:
            (new_artist, new_album), _ = max(tally.items(), key=lambda kv: kv[1])
        else:
            new_artist, new_album = '', ''
        tier_tally = {}
        for t in g['tiers']:
            tier_tally[t] = tier_tally.get(t, 0) + 1
        dominant_tier = (
            max(tier_tally.items(), key=lambda kv: kv[1])[0] if tier_tally else 'unknown'
        )
        albums.append(
            {
                'old_album_artist': old_artist,
                'old_album': old_album,
                'track_count': g['count'],
                'new_album_artist': new_artist,
                'new_album': new_album,
                'tier': dominant_tier,
            }
        )

    albums.sort(
        key=lambda a: (
            (a['old_album_artist'] or '').lower(),
            (a['old_album'] or '').lower(),
        )
    )
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


def _row_to_score_dict(r):
    return {
        'item_id': r[0],
        'file_path': r[1],
        'title': r[2],
        'author': r[3],
        'album': r[4],
        'album_artist': r[5],
    }


_SCORE_COL_NAMES = ("item_id", "file_path", "title", "author", "album", "album_artist")
_SCORE_COLS = pgsql.SQL(", ").join(pgsql.Identifier(c) for c in _SCORE_COL_NAMES)


def _load_score_rows_as_dicts():
    db = get_db()
    with db.cursor() as cur:
        cur.execute(pgsql.SQL("SELECT {} FROM score").format(_SCORE_COLS))
        rows = cur.fetchall() or []
    return [_row_to_score_dict(r) for r in rows]


def _load_score_rows_by_ids(item_ids):
    """Return ``{item_id: score_dict}`` for the given ids only (PK lookup).

    Replaces full-table scans on the step-4 / finalize hot paths; an empty
    input returns ``{}`` without hitting the DB.
    """
    ids = [str(i) for i in (item_ids or [])]
    if not ids:
        return {}
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            pgsql.SQL("SELECT {} FROM score WHERE item_id = ANY(%s)").format(_SCORE_COLS),
            (ids,),
        )
        rows = cur.fetchall() or []
    return {r[0]: _row_to_score_dict(r) for r in rows}


def _sanitize_text(value):
    from sanitization import sanitize_string_for_db

    return sanitize_string_for_db(value)


def _store_target_meta(session_id, new_meta):
    """Persist target-provider track metadata into the ``migration_target_meta``
    side table (replacing any prior rows for this session).

    Kept OUT of ``migration_session.state`` so the wizard's per-click state
    writes don't drag tens of MB of catalog metadata through a JSONB
    read-modify-write on every click.
    """
    db = get_db()
    with db.cursor() as cur:
        cur.execute("DELETE FROM migration_target_meta WHERE session_id = %s", (session_id,))
        rows = [
            (
                session_id,
                _sanitize_text(new_id),
                _sanitize_text((meta or {}).get('path')),
                _sanitize_text((meta or {}).get('title')),
                _sanitize_text((meta or {}).get('artist')),
                _sanitize_text((meta or {}).get('album')),
                _sanitize_text((meta or {}).get('album_artist')),
                (meta or {}).get('year'),
            )
            for new_id, meta in (new_meta or {}).items()
        ]
        for i in range(0, len(rows), 500):
            chunk = rows[i : i + 500]
            placeholders = ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s)"] * len(chunk))
            flat = [v for row in chunk for v in row]
            cur.execute(
                "INSERT INTO migration_target_meta "
                "(session_id, new_id, path, title, artist, album, album_artist, year) "
                "VALUES " + placeholders,  # nosec B608 - %s-placeholder string only; values are bound params
                flat,
            )
    db.commit()


def _load_target_meta(session_id, new_ids=None):
    """Return ``{new_id: {path,title,artist,album,album_artist,year}}`` for the
    session. If ``new_ids`` is given, only those rows are fetched (an empty
    list short-circuits to ``{}``)."""
    if new_ids is not None:
        ids = [str(n) for n in new_ids]
        if not ids:
            return {}
    db = get_db()
    with db.cursor() as cur:
        if new_ids is None:
            cur.execute(
                "SELECT new_id, path, title, artist, album, album_artist, year "
                "FROM migration_target_meta WHERE session_id = %s",
                (session_id,),
            )
        else:
            cur.execute(
                "SELECT new_id, path, title, artist, album, album_artist, year "
                "FROM migration_target_meta WHERE session_id = %s AND new_id = ANY(%s)",
                (session_id, ids),
            )
        rows = cur.fetchall() or []
    return {
        r[0]: {
            'path': r[1],
            'title': r[2],
            'artist': r[3],
            'album': r[4],
            'album_artist': r[5],
            'year': r[6],
        }
        for r in rows
    }


def _load_rows_for_album(album_key):
    """Return all old rows in the given (album_artist, album) regardless of
    whether they were matched. Used by the step-4 re-match flow.

    The album key's artist mirrors the matcher's ``album_artist or author``
    precedence, so the SQL uses ``COALESCE(NULLIF(album_artist,''), author)``
    (NULLIF so an empty string falls through to author exactly like Python's
    ``or``) and null-safe ``IS NOT DISTINCT FROM`` so a NULL artist/album
    matches a NULL key instead of silently dropping the row.
    """
    target_artist, target_album = (
        album_key[0] if album_key else None,
        album_key[1] if album_key and len(album_key) > 1 else None,
    )
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            pgsql.SQL(
                "SELECT {} FROM score "
                "WHERE COALESCE(NULLIF(album_artist, ''), author) IS NOT DISTINCT FROM %s "
                "AND album IS NOT DISTINCT FROM %s"
            ).format(_SCORE_COLS),
            (target_artist, target_album),
        )
        rows = cur.fetchall() or []
    return [_row_to_score_dict(r) for r in rows]


def _load_unmatched_for_album(session_id, album_key):
    """Return the rows in the given (album_artist, album) that were NOT matched
    by the dry run."""
    state = _load_state(session_id) or {}
    manual_unmatches = set(state.get('manual_unmatches') or [])
    matched_ids = set((state.get('dry_run') or {}).get('matches', {}).keys()) - manual_unmatches
    matched_ids |= set((state.get('manual_matches') or {}).keys())
    return [r for r in _load_rows_for_album(album_key) if r['item_id'] not in matched_ids]


# Hard cap on the number of unmatched albums returned to the wizard. The
# value is read from ``config.MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT`` so
# operators can tune it via env var or the setup wizard's DB-backed
# overrides without touching this module. Callers that need the true
# count should use ``len(unmatched_by_album)`` separately.


def _albums_payload(unmatched_by_album):
    """Serialize ``{(album_artist, album): [rows]}`` into a JSON-safe list
    suitable for the wizard UI.

    Truncated to ``config.MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT`` entries
    to keep the persisted state and the step-4 review page bounded.
    """
    import config

    limit = config.MIGRATION_UNMATCHED_ALBUMS_PAYLOAD_LIMIT
    out = []
    for key, rows in unmatched_by_album.items():
        if len(out) >= limit:
            break
        album_artist, album = key[0], key[1] if len(key) > 1 else None
        out.append(
            {
                'album_artist': album_artist,
                'album': album,
                'track_count': len(rows),
                'sample_titles': [r.get('title') for r in rows[:5]],
            }
        )
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


def _sanitize_json_value(value):
    """Local alias for the shared JSON sanitizer in :mod:`sanitization`."""
    from sanitization import sanitize_json_for_db

    return sanitize_json_for_db(value)


def _patch_state_keys(session_id, _set_status=None, **patch):
    """Patch individual top-level keys of ``migration_session.state`` in place
    via ``jsonb_set`` instead of round-tripping the whole (possibly multi-MB)
    blob. A value of ``None`` deletes the key. Only the small patched values
    pass through the sanitizer.

    ``_set_status`` (optional) also updates the ``status`` column in the same
    transaction.
    """
    db = get_db()
    with db.cursor() as cur:
        if _set_status is not None:
            cur.execute(
                "UPDATE migration_session SET status = %s WHERE id = %s",
                (_set_status, session_id),
            )
        for k, v in patch.items():
            if v is None:
                cur.execute(
                    "UPDATE migration_session SET state = state - %s WHERE id = %s",
                    (k, session_id),
                )
            else:
                cur.execute(
                    "UPDATE migration_session SET state = jsonb_set("
                    "COALESCE(state, '{}'::jsonb), %s, %s::jsonb, true) WHERE id = %s",
                    ([k], json.dumps(_sanitize_json_value(v), ensure_ascii=False), session_id),
                )
    db.commit()


def _update_state(session_id, **patch):
    """Patch the given keys into migration_session.state and flag the session
    in_progress. Backed by :func:`_patch_state_keys` so a large ``dry_run`` /
    ``source_path_overrides`` value never forces a whole-blob rewrite of the
    other keys."""
    _patch_state_keys(session_id, _set_status='in_progress', **patch)


def _read_state_key(session_id, key):
    """Return a single top-level ``state`` key (parsed) without loading the
    whole blob. Returns ``(found, value)`` - ``found`` is False when the
    session row doesn't exist."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT state -> %s FROM migration_session WHERE id = %s",
            (key, session_id),
        )
        row = cur.fetchone()
    if row is None:
        return False, None
    return True, row[0]


def _merge_manual_matches(session_id, new_matches):
    _, manual = _read_state_key(session_id, 'manual_matches')
    manual = dict(manual or {})
    manual.update(new_matches)
    # Invalidate final_counts so the user must re-finalize
    _patch_state_keys(session_id, manual_matches=manual, final_counts=None)


def _rematch_album_rows(session_id, newly_matched, newly_unmatched):
    """Atomically replace match state for a re-targeted album.

    For each row we found in the new target: put it in manual_matches (which
    wins over dry.matches at finalize time) and make sure it's not stuck in
    manual_unmatches from a previous rematch.

    For each row we could NOT find in the new target: drop any stale
    manual_matches entry and add it to manual_unmatches so finalize treats
    it as an orphan regardless of what dry.matches said.
    """
    _, manual = _read_state_key(session_id, 'manual_matches')
    _, unmatch_list = _read_state_key(session_id, 'manual_unmatches')
    manual = dict(manual or {})
    unmatches = set(unmatch_list or [])
    for old_id, new_id in newly_matched.items():
        manual[old_id] = new_id
        unmatches.discard(old_id)
    for old_id in newly_unmatched:
        manual.pop(old_id, None)
        unmatches.add(old_id)
    _patch_state_keys(
        session_id,
        manual_matches=manual,
        manual_unmatches=sorted(unmatches),
        final_counts=None,
    )


def _mark_album_skipped(session_id, old_album_key):
    _, skipped = _read_state_key(session_id, 'skipped_albums')
    skipped = list(skipped or [])
    if old_album_key and old_album_key not in skipped:
        skipped.append(old_album_key)
    _patch_state_keys(session_id, skipped_albums=skipped, final_counts=None)


def _count_score_rows():
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM score")
        row = cur.fetchone()
    return int(row[0] or 0) if row else 0
