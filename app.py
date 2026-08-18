# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Main Flask application: root routes, task control, and blueprint wiring.

Owns the shared Flask instance (imported from `flask_app`), installs the
`before_request` auth barrier from `app_auth`, mounts Swagger, and registers
every feature blueprint (`app_*`). Sibling `app_*` modules provide the feature
routes; this module provides the app-wide plumbing they all hang off.

Main Features:
* Core routes: health, `/analysis` landing page, generic task status/cancel/
  cancel-all, last-task and active-tasks polling, `/api/config`, `/api/playlists`.
* Registers all feature blueprints and, on the Flask server only (never queue
  workers), loads similarity indexes/caches and starts the background listener.
"""

import os
from psycopg2.extras import DictCursor
from flask import jsonify, request, render_template, g
from werkzeug.exceptions import HTTPException
import logging
import threading
import time
import config
from sanitization import sanitize_for_log

from tasks.setup_manager import setup_manager

# Swagger imports
from flasgger import Swagger

# Import configuration
from config import TEMP_DIR, APP_VERSION, ENABLE_PROXY_FIX, JWT_SECRET

if ENABLE_PROXY_FIX:
    # Werkzeug import for reverse proxy support
    from werkzeug.middleware.proxy_fix import ProxyFix
    from proxy_prefix import StripDuplicatedScriptName

# --- Flask App Setup ---
# The Flask instance lives in `flask_app` so task modules can import it
# without creating a circular import back into this file.
from flask_app import app

# Import helper functions
import app_server_context
from app_helper import (
    get_db,
    close_db,
    get_task_info_from_db,
    revoke_inline_task_row,
    cancel_job_and_children_recursive,
    coerce_db_details,
    sanitize_task_details,
)
from database import init_db
from taskqueue.sql import CONTROL_TASK_TYPE
from tasks.provider_migration_tasks import MIGRATION_PLANNER_TASK_TYPE
from config import (
    TASK_STATUS_PENDING,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_REVOKED,
)
from app_auth import (
    init_app as init_auth,
    check_setup_needed,
    seed_admin_from_env,
    resolve_jwt_secret,
)

from error import error_manager
from error.error_manager import AudioMuseError
from error.error_dictionary import UNKNOWN_ERROR_CODE

# NOTE: Annoy Manager import is moved to be local where used to prevent circular imports.

# Queue rows the user never started. Both spellings come from the module that
# WRITES them, because a rename that moved only one side left these filters
# matching nothing: the handshake reappeared as a phantom dashboard task, and a
# pending restart 409-blocked the next analysis or cleaning start.
NON_USER_TASK_TYPES = (CONTROL_TASK_TYPE, MIGRATION_PLANNER_TASK_TYPE)

logger = logging.getLogger(__name__)


@app.errorhandler(AudioMuseError)
def handle_audiomuse_error(err):
    app.logger.error(
        "[%s] %s: %s", err.code, err.error_class, err.error_message, exc_info=err.cause or err
    )
    payload = {**err.to_dict(), "error": err.error_message}
    return jsonify(payload), error_manager.http_status_for_code(err.code)


@app.errorhandler(Exception)
def handle_unexpected_error(err):
    if isinstance(err, HTTPException):
        return err
    app.logger.exception("Unhandled exception during request")
    payload, status = error_manager.error_response(UNKNOWN_ERROR_CODE)
    return jsonify(payload), status


from app_logging import configure_logging

configure_logging()

if ENABLE_PROXY_FIX:
    # StripDuplicatedScriptName runs after ProxyFix (inner app) to undo a doubled
    # subpath prefix from a proxy that forwards the full path while also sending
    # X-Forwarded-Prefix, which otherwise loops redirects to /<prefix>/setup (#668).
    app.wsgi_app = ProxyFix(
        StripDuplicatedScriptName(app.wsgi_app), x_for=1, x_proto=1, x_host=1, x_prefix=1
    )

# Log the application version on startup
app.logger.info(f"Starting AudioMuse-AI Backend version {APP_VERSION}")

# --- Authentication Setup ---
# All auth logic (user accounts, password hashing, JWT, /login /auth /logout
# /api/users routes, and the setup/auth/admin barrier) lives in app_auth.
# The JWT secret is resolved after DB init (see below) so every gunicorn
# worker ends up sharing the same value.
_jwt_secret = JWT_SECRET


def _get_jwt_secret():
    return _jwt_secret


@app.context_processor
def inject_globals():
    from config import CLAP_ENABLED, LYRICS_ENABLED

    # auth_role defaults to 'admin' (set by check_auth_needed), so when
    # AUTH_ENABLED is false or the barrier has not run yet (e.g. error
    # pages), is_admin will be True and the full UI is shown.
    auth_role = getattr(g, 'auth_role', 'admin')
    current_user = getattr(g, 'auth_user', None)
    # Resolve each plugin menu link here (inside a request context) with per-item
    # isolation so a plugin whose endpoint does not build can never 500 the layout
    # that every authenticated page renders.
    plugin_menu_items = []
    try:
        from flask import url_for
        from plugin.manager import plugin_manager
        for item in plugin_manager.menu_items():
            if item.get('admin_only') and auth_role != 'admin':
                continue
            try:
                item_url = url_for(item['endpoint'])
            except Exception:
                continue
            plugin_menu_items.append({**item, 'url': item_url})
    except Exception:
        plugin_menu_items = []
    # Sidebar server picker, rendered server-side so it paints with the page
    # instead of popping in at DOMContentLoaded. Credentials are never included.
    nav_music_servers = None
    try:
        payload = app_server_context.servers_for_ui()
        nav_music_servers = {
            'multi_server_enabled': bool(payload.get('multi_server_enabled')),
            'default_id': payload.get('default_id'),
            'servers': [
                {'server_id': s['server_id'], 'name': s['name'], 'is_default': s['is_default']}
                for s in payload.get('servers', [])
            ],
        }
    except Exception:
        logger.exception("Could not build the sidebar server picker data")
        nav_music_servers = None
    return dict(
        app_version=APP_VERSION,
        clap_enabled=CLAP_ENABLED,
        lyrics_enabled=LYRICS_ENABLED,
        auth_enabled=config.AUTH_ENABLED,
        setup_saved=not check_setup_needed(),
        is_admin=(auth_role == 'admin'),
        current_user=current_user,
        plugin_menu_items=plugin_menu_items,
        nav_music_servers=nav_music_servers,
    )


# Register the auth barrier + auth routes (/login, /auth, /logout, /api/users).
init_auth(app, setup_manager, _get_jwt_secret)


@app.before_request
def log_api_request():
    if request.path.startswith('/api/') and not request.path.startswith('/static/'):
        app.logger.info('API request: %s %s', request.method, request.path)


@app.route('/api/health')
def health_check():
    """
    Liveness probe.
    ---
    tags:
      - Health
    summary: Lightweight health check.
    responses:
      200:
        description: Service is up.
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: ok
    """
    return jsonify(
        {
            'status': 'ok',
        }
    )


# --- Swagger Setup ---
app.config['SWAGGER'] = {'title': 'AudioMuse-AI API', 'uiversion': 3, 'openapi': '3.0.0'}
swagger = Swagger(app)


@app.teardown_appcontext
def teardown_db(e=None):
    close_db(e)
    try:
        from tasks.paged_ivf import end_all_requests

        end_all_requests()
    except Exception:
        pass


# Initialize the database schema when the application module is loaded.
# This is safe because it doesn't import other application modules.
# queue workers import app.py too, but they should not perform schema bootstrapping.
_is_worker = (os.environ.get('AUDIOMUSE_ROLE') or '').lower() == 'worker'
if not _is_worker:
    with app.app_context():
        init_db()
        # Keep app_config aligned with the parameters that config.py still
        # accepts. Valid rows are never rewritten; only retired keys are
        # removed. Run this before the optional empty-table bootstrap so a
        # database containing only obsolete keys can be initialized cleanly.
        setup_manager.prune_obsolete_config_values(config)
        setup_manager.bootstrap_env_config_if_empty(config)
        # Bootstrap / reconcile the first admin account:
        #   - If audiomuse_users already has an admin, purge any legacy
        #     AUDIOMUSE_USER / AUDIOMUSE_PASSWORD rows from app_config.
        #   - Else if app_config contains legacy admin values, import them into
        #     audiomuse_users and remove the legacy config.
        #   - Else if env vars contain legacy admin values, import them into
        #     audiomuse_users.
        # See app_auth.seed_admin_from_env for full precedence.
        try:
            seed_admin_from_env()
        except Exception as _seed_exc:
            app.logger.warning("seed_admin_from_env failed at startup: %s", _seed_exc)

        # Media-server settings live only in the music_servers registry: config
        # globals are projected from its default row at import (config module),
        # init_db migrates legacy app_config rows into it and deletes them, so
        # no config<->registry sync is needed here anymore.

        # One-time legacy catalogue migration, Flask startup ONLY: relabel any
        # provider-keyed (or retired-scheme) rows to the canonical signature id.
        # Pure DB work from stored embeddings; an instant no-op on every later
        # boot. A relabel renames tracks without moving a single vector, so the
        # migration repoints the existing indexes at the new ids rather than
        # rebuilding them, and similarity keeps working across it.
        _relabel = {}
        try:
            from tasks.fingerprint_canonicalize import canonicalize_fingerprinted_ids
            _relabel = canonicalize_fingerprinted_ids()
            if _relabel.get('relabelled'):
                app.logger.info(
                    "Startup migration relabelled %s catalogue ids; the similarity "
                    "indexes were repointed at them, no rebuild needed.",
                    _relabel['relabelled'],
                )
        except Exception as _migrate_exc:
            app.logger.warning(
                "Startup catalogue-id migration failed (will retry next boot): %s",
                _migrate_exc,
            )

        # One-time duplicate verification for installs merged by an early 3.0
        # build (fp_ ids, but merged before track length was considered): each
        # catalogue id mapping multiple files whose survivor has no duration yet
        # is re-checked with the duration rule; false duplicates are unmapped so
        # the next analysis re-analyzes them under their own ids. It is
        # table-derived (score.duration), so it is an instant no-op after the
        # legacy migration and after its own first pass - no stored flag that a
        # config cleanup could wipe, no second duration fetch on a legacy upgrade.
        _repair = {}
        try:
            from tasks.duplicate_repair import repair_duplicate_track_maps
            # Reuse the whole-server listing the legacy migration just did so a
            # mixed upgrade (provider ids plus leftover older-scheme rows) does not
            # list the same server twice on one boot - its only slow step.
            _repair = repair_duplicate_track_maps(
                prefetched_durations=_relabel.get('server_durations'),
            ) or {}
        except Exception as _repair_exc:
            # A thrown repair proves nothing ran to completion: mark it skipped so
            # the migration-gated steps below (same-folder split, orphan purge) do
            # not act on a catalogue whose migration state is unknown.
            _repair = {'skipped': 'error'}
            app.logger.warning(
                "Startup catalogue duplicate check failed (will retry next boot): %s",
                _repair_exc,
            )

        # Split any merge that grouped two distinct files from the SAME folder (an
        # album never holds one recording twice). The migration paths already reject
        # same-folder pairs as they resolve; this group-level pass catches the few
        # transitive leftovers. It is GATED on a scheme migration having actually run
        # this boot (older-scheme rows existed) - so once everything is on the current
        # scheme it never runs and never scans track_server_map on a steady-state boot.
        _migration_happened = bool(_relabel.get('relabelled')) or 'skipped' not in _repair
        if _migration_happened:
            try:
                from tasks.duplicate_repair import split_same_folder_merges
                split_same_folder_merges()
            except Exception as _folder_exc:
                app.logger.warning(
                    "Startup same-folder cleanup failed (will retry next boot): %s",
                    _folder_exc,
                )
            # Leave a clean catalogue after a migration: delete every row a false
            # split (or a prior run) left bound to no server, so re-analysis re-creates
            # it under its own id rather than stranding an orphan.
            try:
                from tasks.duplicate_repair import purge_orphan_catalogue_rows
                purge_orphan_catalogue_rows()
            except Exception as _purge_exc:
                app.logger.warning(
                    "Startup migration orphan purge failed (will retry next boot): %s",
                    _purge_exc,
                )

        # Finalize JWT_SECRET - must happen after DB init so the value can be
        # persisted and shared across all gunicorn workers.
        _jwt_secret = resolve_jwt_secret(setup_manager)
else:
    app.logger.info("Worker mode: skipping startup database schema bootstrap.")

import app_setup  # noqa: F401

# --- API Endpoints ---


@app.route('/analysis')
def index():
    """
    Serve the Analysis & Clustering page (legacy home).
    The application landing page is now the dashboard ('/').
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the main page.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('index.html', title='AudioMuse-AI - Home Page', active='index')


@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status_endpoint(task_id):
    """
    Get the status of a specific task.
    Retrieves status information from the database.
    ---
    tags:
      - Status
    parameters:
      - name: task_id
        in: path
        required: true
        description: The ID of the task.
        schema:
          type: string
    responses:
      200:
        description: Status information for the task.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                state:
                  type: string
                  description: Current state of the task (e.g., PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED, queued, finished, failed, canceled).
                status_message:
                  type: string
                  description: A human-readable status message.
                progress:
                  type: integer
                  description: Task progress percentage (0-100).
                running_time_seconds:
                  type: number
                  description: The total running time of the task in seconds. Updates live for running tasks.
                details:
                  type: object
                  description: Detailed information about the task. Structure varies by task type and state.
                  additionalProperties: true
                  example: {"log": ["Log message 1"], "current_album": "Album X"}
                task_type_from_db:
                  type: string
                  nullable: true
                  description: The type of the task as recorded in the database (e.g., main_analysis, album_analysis, main_clustering, clustering_batch).
      404:
        description: Task ID not found in the database.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                state:
                  type: string
                  example: UNKNOWN
                status_message:
                  type: string
                  example: Task ID not found in the database.
    """
    # One read. There is no second system holding a second opinion about this
    # task, so there is nothing to merge and no rule needed for whose answer wins.
    response = {
        'task_id': task_id,
        'state': 'UNKNOWN',
        'status_message': 'Task ID not found.',
        'progress': 0,
        'details': {},
        'task_type_from_db': None,
        'running_time_seconds': 0,
    }
    db_task_info = get_task_info_from_db(task_id)
    if not db_task_info:
        return jsonify(response), 404

    details = coerce_db_details(db_task_info.get('details')) or {}
    response['state'] = db_task_info.get('status') or 'UNKNOWN'
    response['task_type_from_db'] = db_task_info.get('task_type')
    response['running_time_seconds'] = db_task_info.get('running_time_seconds', 0)
    response['progress'] = db_task_info.get('progress', 0)
    response['details'] = details
    response['status_message'] = (
        details.get('status_message') or details.get('message') or response['state']
    )
    if response['state'] == TASK_STATUS_REVOKED:
        response['status_message'] = details.get('message') or 'Task revoked.'
        response['progress'] = 100

    response['details'] = sanitize_task_details(
        response.get('details'), response.get('state'), response.get('task_type_from_db')
    )

    # Clean up the final response to remove confusing raw time columns
    response.pop('timestamp', None)
    response.pop('start_time', None)
    response.pop('end_time', None)

    return jsonify(response)


@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task_endpoint(task_id):
    """
    Cancel a specific task and its children.
    Marks the task and its descendants as REVOKED in the database and notifies
    the queue workers to stop the process running it.
    ---
    tags:
      - Control
    parameters:
      - name: task_id
        in: path
        required: true
        description: The ID of the task.
        schema:
          type: string
    responses:
      200:
        description: Cancellation initiated for the task and its children.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                task_id:
                  type: string
                cancelled_jobs_count:
                  type: integer
      400:
        description: Task could not be cancelled (e.g., already completed or not in an active state).
      404:
        description: Task ID not found in the database.
    """
    # An in-process task has no queue job to stop, and the global cancel below is
    # destructive by design, so it gets a surgical revoke of its own row instead.
    inline_message = revoke_inline_task_row(task_id)
    if inline_message:
        return jsonify(
            {"message": inline_message, "task_id": task_id, "cancelled_jobs_count": 0}
        ), 200

    # Always perform cancel when the endpoint is invoked. No early returns.
    try:
        cancelled_count = cancel_job_and_children_recursive(
            task_id, reason=f"Cancellation requested for task {task_id} via API."
        )
    except Exception:
        logger.exception(
            "Cancellation of task %s could not be fully confirmed",
            sanitize_for_log(task_id),
        )
        return jsonify(
            {
                "error": (
                    "Cancellation could not be fully applied or confirmed; "
                    "recovery tasks may remain active."
                ),
                "task_id": task_id,
                "details": None,
            }
        ), 503
    return jsonify(
        {
            "message": f"Task {task_id} cancellation requested. {cancelled_count} cancellation actions attempted.",
            "task_id": task_id,
            "cancelled_jobs_count": cancelled_count,
        }
    ), 200


@app.route('/api/cancel_all/<task_type_prefix>', methods=['POST'])
def cancel_all_tasks_by_type_endpoint(task_type_prefix):
    """
    Cancel all active tasks of a specific type (e.g., main_analysis, main_clustering) and their children.
    ---
    tags:
      - Control
    parameters:
      - name: task_type_prefix
        in: path
        required: true
        description: The type of main tasks to cancel (e.g., "main_analysis", "main_clustering").
        schema:
          type: string
    responses:
      200:
        description: Cancellation initiated for all matching active tasks and their children.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                cancelled_main_tasks:
                  type: array
                  items:
                    type: string
      404:
        description: No active tasks of the specified type found to cancel.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute(
        "SELECT task_id, task_type FROM task_status WHERE task_type = %s "
        "AND status NOT IN %s",
        (task_type_prefix, config.TASK_STATUS_TERMINAL),
    )
    tasks_to_cancel = cur.fetchall()
    cur.close()

    # Decide 404 BEFORE any destructive call: the cancel empties the whole table
    # and revokes every row, so running it first and then reporting "nothing
    # found" would be a lie about a wipe that already happened.
    if not tasks_to_cancel:
        return jsonify(
            {"message": f"No active tasks of type '{task_type_prefix}' found to cancel."}
        ), 404

    cancelled_main_task_ids = [r['task_id'] for r in tasks_to_cancel]
    # cancel_job_and_children_recursive re-raises when the tombstone commit fails.
    # Without this the generic handler answered UNKNOWN 500 with no
    # cancelled_main_tasks, so the caller could not tell a partially applied
    # cancel from an unknown failure. The sibling /api/cancel/<task_id> answers 503.
    try:
        total_cancelled_jobs = cancel_job_and_children_recursive(
            cancelled_main_task_ids[0],
            reason=f"Bulk cancellation for task type '{task_type_prefix}' via API.",
        )
    except Exception:
        logger.exception(
            "Bulk cancellation for %s could not be fully confirmed",
            sanitize_for_log(task_type_prefix),
        )
        return jsonify(
            {
                "error": (
                    "Cancellation could not be fully applied or confirmed; "
                    "recovery tasks may remain active."
                ),
                "cancelled_main_tasks": cancelled_main_task_ids,
                "details": None,
            }
        ), 503

    return jsonify(
        {
            "message": (
                f"Cancellation initiated for {len(cancelled_main_task_ids)} main tasks "
                f"of type '{task_type_prefix}' and their children. "
                f"Total jobs affected: {total_cancelled_jobs}."
            ),
            "cancelled_main_tasks": cancelled_main_task_ids,
        }
    ), 200


@app.route('/api/last_task', methods=['GET'])
def get_last_overall_task_status_endpoint():
    """
    Get the most recent overall main task.
    ---
    tags:
      - Tasks
    summary: Status of the latest top-level main task (analysis, clustering, cleaning, etc.).
    description: |
      Returns the most recent row in `task_status` whose `parent_task_id` is
      NULL. Long log lists are truncated to the last 10 entries with a
      "... earlier entries truncated" placeholder.
    responses:
      200:
        description: Task summary or a sentinel object when no main task has ever run.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  nullable: true
                task_type:
                  type: string
                  nullable: true
                status:
                  type: string
                progress:
                  type: number
                details:
                  type: object
                running_time_seconds:
                  type: number
                  format: float
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    # Control-plane handshakes and migration planner runs are queue rows now, but
    # they are not user tasks: showing one here made every task-polling page
    # believe a restart request was "the last task".
    cur.execute(
        """
        SELECT task_id, task_type, status, progress, details, start_time, end_time
        FROM task_status
        WHERE parent_task_id IS NULL
          AND task_type NOT IN %s
        ORDER BY timestamp DESC
        LIMIT 1
    """,
        (NON_USER_TASK_TYPES,),
    )
    last_task_row = cur.fetchone()
    cur.close()

    if last_task_row:
        last_task_data = dict(last_task_row)
        if last_task_data.get('details'):
            last_task_data['details'] = coerce_db_details(last_task_data['details'])

        # Calculate running time in Python
        start_time = last_task_data.get('start_time')
        end_time = last_task_data.get('end_time')
        if start_time:
            effective_end_time = end_time if end_time is not None else time.time()
            last_task_data['running_time_seconds'] = max(0, effective_end_time - start_time)
        else:
            last_task_data['running_time_seconds'] = 0.0

        last_task_data['details'] = sanitize_task_details(
            last_task_data.get('details'),
            last_task_data.get('status'),
            last_task_data.get('task_type'),
        )

        # Clean up raw time columns before sending response
        last_task_data.pop('start_time', None)
        last_task_data.pop('end_time', None)
        last_task_data.pop('timestamp', None)

        return jsonify(last_task_data), 200

    return jsonify(
        {
            "task_id": None,
            "task_type": None,
            "status": "NO_PREVIOUS_MAIN_TASK",
            "details": {"log": ["No previous main task found."]},
        }
    ), 200


@app.route('/api/active_tasks', methods=['GET'])
def get_active_tasks_endpoint():
    """
    Get the currently active main task.
    ---
    tags:
      - Tasks
    summary: Return the in-flight top-level main task (PENDING/STARTED/PROGRESS), if any.
    description: |
      Returns `{}` when no main task is active. Strips heavyweight internal
      keys (`clustering_run_job_ids`, `checked_album_ids`, initial centroids)
      from the response payload.
    responses:
      200:
        description: Active task or empty object.
        content:
          application/json:
            schema:
              type: object
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    non_terminal_statuses = (TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS)
    # worker_control handshakes and migration planner runs are queue rows, not
    # user tasks: without this filter a slow worker restart greyed out every
    # Start button until the stale sweep failed the request row 30 minutes on.
    cur.execute(
        """
        SELECT task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, start_time, end_time
        FROM task_status
        WHERE parent_task_id IS NULL AND status IN %s
          AND task_type NOT IN %s
        ORDER BY timestamp DESC
        LIMIT 1
    """,
        (non_terminal_statuses, NON_USER_TASK_TYPES),
    )
    active_main_task_row = cur.fetchone()
    cur.close()

    if active_main_task_row:
        task_item = dict(active_main_task_row)

        # Calculate running time in Python
        start_time = task_item.get('start_time')
        if start_time:
            task_item['running_time_seconds'] = max(0, time.time() - start_time)
        else:
            task_item['running_time_seconds'] = 0.0

        if task_item.get('details'):
            details = coerce_db_details(task_item['details'])
            # Prune specific large or internal keys from details
            if isinstance(details, dict):
                details.pop('clustering_run_job_ids', None)
                details.pop('checked_album_ids', None)
                cmc = (
                    details.get('best_params', {}).get('clustering_method_config', {})
                    if isinstance(details.get('best_params'), dict)
                    else {}
                )
                if isinstance(cmc, dict) and isinstance(cmc.get('params'), dict):
                    cmc['params'].pop('initial_centroids', None)
            task_item['details'] = sanitize_task_details(
                details, task_item.get('status'), task_item.get('task_type')
            )

        # Clean up raw time columns before sending response
        task_item.pop('start_time', None)
        task_item.pop('end_time', None)
        task_item.pop('timestamp', None)

        return jsonify(task_item), 200
    return jsonify({}), 200  # Return empty object if no active main task


@app.route('/api/config', methods=['GET'])
def get_config_endpoint():
    """
    Public configuration snapshot.
    ---
    tags:
      - Config
    summary: Return the user-relevant subset of `config.*` (clustering ranges, AI provider, alchemy defaults, etc.).
    description: |
      Reads attributes from the `config` module at request time rather than
      using names imported at app.py module load, so wizard changes are
      visible without a Flask restart.
    responses:
      200:
        description: Configuration values currently in effect.
        content:
          application/json:
            schema:
              type: object
              additionalProperties: true
    """
    return jsonify(
        {
            "num_recent_albums": config.NUM_RECENT_ALBUMS,
            "max_distance": config.MAX_DISTANCE,
            "max_songs_per_cluster": config.MAX_SONGS_PER_CLUSTER,
            "max_songs_per_artist": config.MAX_SONGS_PER_ARTIST,
            "cluster_algorithm": config.CLUSTER_ALGORITHM,
            "clustering_auto_calibration": config.CLUSTERING_AUTO_CALIBRATION,
            "num_clusters_min": config.NUM_CLUSTERS_MIN,
            "num_clusters_max": config.NUM_CLUSTERS_MAX,
            "dbscan_eps_min": config.DBSCAN_EPS_MIN,
            "dbscan_eps_max": config.DBSCAN_EPS_MAX,
            "gmm_covariance_type": config.GMM_COVARIANCE_TYPE,
            "dbscan_min_samples_min": config.DBSCAN_MIN_SAMPLES_MIN,
            "dbscan_min_samples_max": config.DBSCAN_MIN_SAMPLES_MAX,
            "gmm_n_components_min": config.GMM_N_COMPONENTS_MIN,
            "gmm_n_components_max": config.GMM_N_COMPONENTS_MAX,
            "spectral_n_clusters_min": config.SPECTRAL_N_CLUSTERS_MIN,
            "spectral_n_clusters_max": config.SPECTRAL_N_CLUSTERS_MAX,
            "pca_components_min": config.PCA_COMPONENTS_MIN,
            "pca_components_max": config.PCA_COMPONENTS_MAX,
            "min_songs_per_genre_for_stratification": config.MIN_SONGS_PER_GENRE_FOR_STRATIFICATION,
            "stratified_sampling_target_percentile": config.STRATIFIED_SAMPLING_TARGET_PERCENTILE,
            "ai_model_provider": config.AI_MODEL_PROVIDER,
            "ollama_server_url": config.OLLAMA_SERVER_URL,
            "ollama_model_name": config.OLLAMA_MODEL_NAME,
            "openai_server_url": config.OPENAI_SERVER_URL,
            "openai_model_name": config.OPENAI_MODEL_NAME,
            "gemini_model_name": config.GEMINI_MODEL_NAME,
            "mistral_model_name": config.MISTRAL_MODEL_NAME,
            "top_n_moods": config.TOP_N_MOODS,
            "mood_labels": config.MOOD_LABELS,
            "clustering_runs": config.CLUSTERING_RUNS,
            "top_n_clustering_playlist": config.TOP_N_CLUSTERING_PLAYLIST,
            "enable_clustering_embeddings": config.ENABLE_CLUSTERING_EMBEDDINGS,
            "score_weight_diversity": config.SCORE_WEIGHT_DIVERSITY,
            "score_weight_silhouette": config.SCORE_WEIGHT_SILHOUETTE,
            "score_weight_davies_bouldin": config.SCORE_WEIGHT_DAVIES_BOULDIN,
            "score_weight_calinski_harabasz": config.SCORE_WEIGHT_CALINSKI_HARABASZ,
            "score_weight_purity": config.SCORE_WEIGHT_PURITY,
            "score_weight_other_feature_diversity": config.SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY,
            "score_weight_other_feature_purity": config.SCORE_WEIGHT_OTHER_FEATURE_PURITY,
            "path_distance_metric": config.PATH_DISTANCE_METRIC,
            "alchemy_default_n_results": config.ALCHEMY_DEFAULT_N_RESULTS,
            "alchemy_max_n_results": config.ALCHEMY_MAX_N_RESULTS,
            "alchemy_temperature": config.ALCHEMY_TEMPERATURE,
            "alchemy_subtract_distance_angular": config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR,
            "alchemy_subtract_distance_euclid": config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN,
        }
    )


@app.route('/api/playlists', methods=['GET'])
def get_playlists_endpoint():
    """
    All generated playlists, grouped per server.
    ---
    tags:
      - Playlists
    summary: Return the last clustering run's playlists of every server, grouped per server.
    responses:
      200:
        description: Per-server playlist groups (default server first).
        content:
          application/json:
            schema:
              type: object
              properties:
                multi_server:
                  type: boolean
                servers:
                  type: array
                  items:
                    type: object
                    properties:
                      server_id:
                        type: string
                        nullable: true
                      server_name:
                        type: string
                      is_default:
                        type: boolean
                      playlists:
                        type: object
                        additionalProperties:
                          type: array
                          items:
                            type: object
                            properties:
                              item_id:
                                type: string
                              title:
                                type: string
                              author:
                                type: string
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute(
        "SELECT playlist_name, item_id, title, author, server_id FROM playlist ORDER BY playlist_name"
    )
    rows = [dict(row) for row in cur.fetchall()]
    cur.close()
    return jsonify(app_server_context.group_playlist_rows_by_server(rows)), 200


# --- Index reload listener over Postgres LISTEN/NOTIFY ---
def listen_for_index_reloads():
    from taskqueue.listen import Listener
    from taskqueue.sql import CHANNEL_EVENT

    def _on_notify(_channel, payload):
        logger.info("Received '%s' on the index-updates channel.", payload)
        if payload != 'index-reload':
            return
        with app.app_context():
            logger.info(
                "Triggering in-memory IVF index and map reload from background listener."
            )
            try:
                from tasks.ivf_manager import load_ivf_index_for_querying

                load_ivf_index_for_querying(force_reload=True)
                from tasks.artist_gmm_manager import load_artist_index_for_querying

                load_artist_index_for_querying(force_reload=True)
                from database import load_map_projection, load_artist_projection

                load_map_projection('main_map', force_reload=True)
                load_artist_projection('artist_map', force_reload=True)
                from app_map import build_map_cache

                build_map_cache()

                logger.info("Reloading CLAP embedding cache...")
                from tasks.clap_text_search import refresh_clap_cache

                clap_success = refresh_clap_cache()

                try:
                    from config import LYRICS_ENABLED

                    if LYRICS_ENABLED:
                        logger.info("Reloading Lyrics search cache...")
                        from tasks.lyrics_manager import refresh_lyrics_cache

                        lyrics_success = refresh_lyrics_cache()
                    else:
                        lyrics_success = False
                except Exception:
                    logger.exception("Lyrics cache reload failed")
                    lyrics_success = False

                try:
                    logger.info("Reloading SemGrove merged index...")
                    from tasks.sem_grove_manager import refresh_sem_grove_cache

                    sg_success = refresh_sem_grove_cache()
                except Exception:
                    logger.exception("SemGrove cache reload failed")
                    sg_success = False

                try:
                    from tasks.hyperbolic_manager import (
                        is_hyperbolic_tree_cache_loaded,
                        load_hyperbolic_tree_cache,
                    )

                    if is_hyperbolic_tree_cache_loaded():
                        logger.info("Reloading Hyperbolic Explorer tree cache...")
                        load_hyperbolic_tree_cache()
                    else:
                        logger.info(
                            "Hyperbolic Explorer tree cache is idle; skipping reload "
                            "(lazy-loads fresh data on next /hyperbolic page open)."
                        )
                    hyper_success = True
                except Exception:
                    logger.exception("Hyperbolic Explorer cache reload failed")
                    hyper_success = False

                logger.info(
                    "In-memory reload complete: IVF OK, Artist OK, Maps OK, CLAP %s, "
                    "Lyrics %s, SemGrove %s, Hyperbolic %s",
                    'OK' if clap_success else 'X',
                    'OK' if lyrics_success else 'X',
                    'OK' if sg_success else 'X',
                    'OK' if hyper_success else 'X',
                )
            except Exception:
                logger.exception("Error reloading indexes/maps from background listener")
            finally:
                # A reload replaces every index in place, so the whole previous
                # generation plus each rebuild's scratch is garbage by now. Hand
                # it back to the kernel instead of letting RSS ratchet up once
                # per reload for the life of the process.
                try:
                    from tasks.memory_utils import release_memory_to_os

                    release_memory_to_os()
                except Exception:
                    logger.exception("Index reload: heap release to the OS failed")

    listener = Listener(
        (CHANNEL_EVENT,),
        _on_notify,
        application_name=f"audiomuse-flask-events-{os.getpid()}",
        name='index-events',
    )
    listener.start()
    logger.info("Listening for index reloads on the %s channel.", CHANNEL_EVENT)


# --- Blueprint Registration ---
# Standard Flask factory pattern: blueprint imports are inside
# this function so the eager import graph stays flat.


def _register_blueprints(flask_app):
    from app_chat import chat_bp
    from app_clustering import clustering_bp
    from app_analysis import analysis_bp
    from app_cron import cron_bp
    from app_ivf import ivf_bp
    from app_sonic_fingerprint import sonic_fingerprint_bp
    from app_path import path_bp
    from app_external import external_bp
    from app_alchemy import alchemy_bp
    from app_map import map_bp
    from app_artist_similarity import artist_similarity_bp
    from app_clap_search import clap_search_bp
    from app_lyrics import lyrics_search_bp
    from app_sem_grove import sem_grove_bp
    from app_backup import backup_bp
    from app_provider_migration import migration_bp
    from app_dashboard import dashboard_bp
    from app_users import users_bp
    from app_sync import sync_bp
    from app_music_servers import music_servers_bp
    from app_hyperbolic import hyperbolic_bp

    flask_app.register_blueprint(chat_bp, url_prefix='/chat')
    flask_app.register_blueprint(clustering_bp)
    flask_app.register_blueprint(analysis_bp)
    flask_app.register_blueprint(cron_bp)
    flask_app.register_blueprint(ivf_bp)
    flask_app.register_blueprint(sonic_fingerprint_bp)
    flask_app.register_blueprint(path_bp)
    flask_app.register_blueprint(external_bp, url_prefix='/external')
    flask_app.register_blueprint(alchemy_bp)
    flask_app.register_blueprint(map_bp)
    flask_app.register_blueprint(artist_similarity_bp)
    flask_app.register_blueprint(clap_search_bp)
    flask_app.register_blueprint(lyrics_search_bp)
    flask_app.register_blueprint(sem_grove_bp)
    flask_app.register_blueprint(backup_bp)
    flask_app.register_blueprint(migration_bp)
    flask_app.register_blueprint(dashboard_bp)
    flask_app.register_blueprint(users_bp)
    flask_app.register_blueprint(sync_bp)
    flask_app.register_blueprint(music_servers_bp)
    flask_app.register_blueprint(hyperbolic_bp)

    try:
        from plugin.blueprint import plugins_bp
        flask_app.register_blueprint(plugins_bp)
    except Exception:
        logger.exception('Failed to register plugin blueprint')


_register_blueprints(app)

# --- Plugin subsystem boot (web) ---
# Materialize enabled plugins from the DB and register their blueprints/menu on
# the Flask app. Guarded so a broken plugin can never prevent the app from booting.
# The plugin imports stay inside this function (not module scope) so app.py does not
# eagerly pull the plugin.blueprint -> manager -> api -> database chain at import time.
def _boot_plugins_web():
    try:
        from plugin.manager import boot as _plugin_boot

        _plugin_boot('web', flask_app=app)
    except Exception:
        logger.exception('Plugin subsystem web boot failed; continuing without plugins')
    try:
        from plugin.blueprint import start_catalog_auto_refresh

        start_catalog_auto_refresh()
    except Exception:
        logger.exception('Plugin catalog auto-refresh failed to start')


if not _is_worker:
    _boot_plugins_web()

# --- Startup: Load indexes and caches (Flask server only, NOT queue workers) ---
# queue workers import app.py but should NOT load indexes or start background threads.
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
except OSError:
    logger.debug(f"Could not create TEMP_DIR '{TEMP_DIR}' (may be running in test/CI environment)")

if not _is_worker:
    with app.app_context():
        # --- Initial IVF Index Load ---
        from tasks.ivf_manager import load_ivf_index_for_querying

        load_ivf_index_for_querying()
        # --- Load Artist Similarity Index ---
        from tasks.artist_gmm_manager import load_artist_index_for_querying

        try:
            load_artist_index_for_querying()
            logger.info("Artist similarity index loaded at startup.")
        except Exception as e:
            logger.warning(f"Failed to load artist similarity index at startup: {e}")
        # Also try to load precomputed map projection into memory if available
        try:
            from app_helper import load_map_projection

            load_map_projection('main_map')
            logger.info("In-memory map projection loaded at startup.")
        except Exception as e:
            logger.debug(f"No precomputed map projection to load at startup or load failed: {e}")
        # Also try to load artist component projection into memory
        try:
            from database import load_artist_projection

            load_artist_projection('artist_map')
            logger.info("In-memory artist component projection loaded at startup.")
        except Exception as e:
            logger.debug(f"No precomputed artist projection to load at startup or load failed: {e}")
        # Load CLAP embeddings cache (model will lazy-load on first use)
        try:
            from config import CLAP_ENABLED

            if CLAP_ENABLED:
                # Load CLAP embeddings cache (15MB) - model lazy-loads on first search to save 3GB RAM
                from tasks.clap_text_search import load_clap_cache_from_db, load_top_queries_from_db

                if load_clap_cache_from_db():
                    logger.info("CLAP text search cache loaded at startup (embeddings only).")
                    logger.info(
                        "CLAP model will lazy-load on first text search (~1-2s delay, saves 3GB RAM)."
                    )

                # Load top queries from database (default queries only, no computation)
                # This must run even if no CLAP embeddings exist yet (first startup)
                has_existing = load_top_queries_from_db()
                if has_existing:
                    logger.info("Loaded top queries from database (defaults).")
                else:
                    logger.info("No queries found in database (should not happen - check DB)")
        except Exception as e:
            logger.debug(f"CLAP cache not loaded at startup (may be disabled or failed): {e}")
        # Load Lyrics search cache (ivf index over per-song gte embeddings + axis-score matrix)
        try:
            from config import LYRICS_ENABLED

            if LYRICS_ENABLED:
                from tasks.lyrics_manager import load_lyrics_cache_from_db

                if load_lyrics_cache_from_db():
                    logger.info("Lyrics search cache loaded at startup (ivf index + axis matrix).")
                else:
                    logger.info("Lyrics search cache empty at startup (run analysis to populate).")
        except Exception as e:
            logger.debug(f"Lyrics cache not loaded at startup (may be disabled or failed): {e}")
        # Load SemGrove merged lyrics+audio index
        try:
            from tasks.sem_grove_manager import load_sem_grove_cache_from_db

            if load_sem_grove_cache_from_db():
                logger.info("SemGrove merged index loaded at startup.")
            else:
                logger.info(
                    "SemGrove index not found at startup (build it after analysis completes)."
                )
        except Exception as e:
            logger.debug(f"SemGrove cache not loaded at startup: {e}")

        # Every load above streams a large directory blob out of Postgres and
        # discards it once unpacked. Those frees land in the allocator's free
        # lists, not back in the kernel, so without this the pod's RSS keeps the
        # startup peak for the life of the process. One call, after all six.
        try:
            from tasks.memory_utils import release_memory_to_os

            release_memory_to_os()
        except Exception:
            logger.exception("Startup index load: heap release to the OS failed")

        def _start_map_init_background():
            try:
                from app_map import init_map_cache

                logger.info('Starting background map JSON cache build.')
                with app.app_context():
                    init_map_cache()
                logger.info('Background map JSON cache build finished.')
            except Exception:
                logger.exception('Background init_map_cache failed')

        t = threading.Thread(target=_start_map_init_background, daemon=True)
        t.start()

        # The Hyperbolic Explorer tree cache is NOT loaded here: unlike the
        # indexes above it is a fully materialized Python object tree whose RSS
        # scales with catalogue size, so it lazy-loads on the first /hyperbolic
        # page open (warmup_hyperbolic_tree_cache, called from the page and from
        # the tree API) and auto-unloads after HYPERBOLIC_TREE_WARMUP_DURATION
        # idle seconds instead of staying resident for the life of the process.

# --- Start Background Listener Thread (Flask server only) ---
if not _is_worker:
    listener_thread = threading.Thread(target=listen_for_index_reloads, daemon=True)
    listener_thread.start()

    # Start a cron manager thread that evaluates enabled cron entries once a minute.
    def _cron_manager_loop():
        try:
            import time as _time
            from app_cron import (
                run_due_cron_jobs,
                retry_due_cron_jobs,
                reap_interrupted_inline_runs,
                cron_retry_interval_seconds,
            )

            # Inline cron runs (the alchemy radio) live in THIS process and nothing
            # else writes their final status, so a restart mid-run leaves a row that
            # no later code path resolves. This process is starting, so no inline run
            # can be live: whatever is still non-terminal died with the old process.
            try:
                with app.app_context():
                    reap_interrupted_inline_runs()
            except Exception:
                app.logger.exception('cron manager startup reap failed')

            last_retry_ts = _time.time()
            while True:
                try:
                    with app.app_context():
                        run_due_cron_jobs()
                        if _time.time() - last_retry_ts >= cron_retry_interval_seconds():
                            retry_due_cron_jobs()
                            last_retry_ts = _time.time()
                except Exception:
                    app.logger.exception('cron manager failed')
                # Sleep to the next minute boundary, not a flat 60s. A flat sleep
                # AFTER variable-length work makes the tick period 60 + work_time,
                # so the second-of-minute it lands on drifts forward and eventually
                # skips a whole wall-clock minute - and a cron scheduled in a skipped
                # minute simply never ran, silently. run_due_cron_jobs claims each
                # row on its minute bucket, so an early tick cannot double-fire.
                _time.sleep(max(1.0, 60.0 - (_time.time() % 60.0)))
        except Exception:
            app.logger.exception('cron manager main loop error')

    cron_thread = threading.Thread(target=_cron_manager_loop, daemon=True)
    cron_thread.start()

    # Dashboard stats refresher: reports the analyzed catalogue only (local score
    # + track_server_map), never walks a media server. Two cadences merge into the
    # same snapshot blob - the cheap counts/per-server every 60s, and the
    # whole-library distribution charts (Genres, Moods Coverage, Tempo) hourly.
    def _dashboard_stats_refresher_loop():
        try:
            from time import sleep
            from app_dashboard import (
                refresh_dashboard_stats,
                refresh_dashboard_charts_stats,
                dashboard_refresh_interval,
                DASHBOARD_CHARTS_REFRESH_INTERVAL_SECONDS,
            )

            # Wait a minute after startup so the initial DB/index warm-up and
            # first incoming requests have time to settle before we kick off
            # the content scans.
            sleep(60)
            fast_interval = dashboard_refresh_interval()
            ticks_per_charts = max(1, DASHBOARD_CHARTS_REFRESH_INTERVAL_SECONDS // fast_interval)
            tick = 0
            while True:
                try:
                    refresh_dashboard_stats(app)
                    # Distribution charts on their own hourly cadence: at startup
                    # (tick 0) so they fill in, then once every ticks_per_charts.
                    if tick % ticks_per_charts == 0:
                        refresh_dashboard_charts_stats(app)
                except Exception:
                    app.logger.exception('dashboard stats refresh failed')
                tick += 1
                sleep(fast_interval)
        except Exception:
            app.logger.exception('dashboard stats refresher main loop error')

    dashboard_stats_thread = threading.Thread(target=_dashboard_stats_refresher_loop, daemon=True)
    dashboard_stats_thread.start()
else:
    logger.info('Running as a queue worker: skipping index loading, the event listener and the cron thread.')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
