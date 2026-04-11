import os
import psycopg2
from psycopg2.extras import DictCursor
from flask import Flask, jsonify, request, render_template, g, make_response, redirect, url_for
from argon2 import PasswordHasher
from argon2 import exceptions as argon2_exceptions
import json
import logging
import threading
import time
import datetime
import secrets
import jwt as pyjwt
import config

# RQ imports
from rq.job import Job, JobStatus
from rq.exceptions import NoSuchJobError
from tasks.setup_manager import SetupManager

# Redis client
from redis import Redis

# Swagger imports
from flasgger import Swagger, swag_from

# Import configuration
import config

if config.ENABLE_PROXY_FIX:
  # Werkzeug import for reverse proxy support
  from werkzeug.middleware.proxy_fix import ProxyFix

# --- Flask App Setup ---
app = Flask(__name__)
setup_manager = SetupManager()

# Import helper functions
from app_helper import (
    init_db, get_db, close_db,
    redis_conn, rq_queue_high, rq_queue_default,
    clean_up_previous_main_tasks,
    save_task_status,
    get_task_info_from_db,
    cancel_job_and_children_recursive,
    TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED
)

# NOTE: Annoy Manager import is moved to be local where used to prevent circular imports.

logger = logging.getLogger(__name__)

# Configure basic logging for the entire application
logging.basicConfig(
    level=logging.INFO, # Set the default logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='[%(levelname)s]-[%(asctime)s]-%(message)s', # Custom format string
    datefmt='%d-%m-%Y %H-%M-%S' # Custom date/time format
)

if config.ENABLE_PROXY_FIX:
  app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Log the application version on startup
app.logger.info(f"Starting AudioMuse-AI Backend version {config.APP_VERSION}")

# --- Authentication Setup ---
# AUTH_ENABLED is the raw env toggle. If enabled, auth is enforced everywhere.
# If auth is enabled and the user/pass env vars are missing, we provide defaults.

effective_audiomuse_user = AUDIOMUSE_USER
effective_audiomuse_password = AUDIOMUSE_PASSWORD
user_defaulted = False
new_password_generated = False

if config.AUTH_ENABLED:
    if not effective_audiomuse_user or not effective_audiomuse_user.strip():
        effective_audiomuse_user = "admin"
        user_defaulted = True
    if not effective_audiomuse_password or not effective_audiomuse_password.strip():
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        effective_audiomuse_password = ''.join(secrets.choice(alphabet) for _ in range(16))
        new_password_generated = True
    if user_defaulted or new_password_generated:
        print('\n*** AUTH_ENABLED=true: using default authentication values ***')
        if user_defaulted:
            print(f'AUDIOMUSE_USER defaulted to: {effective_audiomuse_user}')
        if new_password_generated:
            print(f'AUDIOMUSE_PASSWORD auto-generated to: {effective_audiomuse_password}')
        print('*** Persist these values in your environment if you want stable credentials ***')
        print('*** Set AUDIOMUSE_USER and AUDIOMUSE_PASSWORD in your env vars, and API_TOKEN too if you need external/token access ***\n')

# Finalize JWT_SECRET — auto-generate if not configured
_jwt_secret = config.JWT_SECRET
if not _jwt_secret and config.AUTH_ENABLED:
    _jwt_secret = secrets.token_hex(32)
    app.logger.warning(
        "JWT_SECRET is not set. A random secret has been generated. "
        "All browser sessions will be invalidated on container restart. "
        "Set JWT_SECRET in your .env for persistent sessions."
    )

# Auth is considered configured whenever the app has effective credentials,
# including runtime-generated temporary auth values.
auth_configured = bool(effective_audiomuse_user and effective_audiomuse_password)
bootstrap_auth_mode = AUTH_ENABLED and not (AUDIOMUSE_USER and AUDIOMUSE_PASSWORD)

# If auth is enabled but no explicit credentials were provided, the app is
# in bootstrap auth mode. Setup can be accessed via the temporary credentials.
def is_bootstrap_mode():
    return not config.AUTH_ENABLED or bootstrap_auth_mode


def refresh_auth_state():
    global AUDIOMUSE_USER, AUDIOMUSE_PASSWORD, effective_audiomuse_user, effective_audiomuse_password, auth_configured, bootstrap_auth_mode
    AUDIOMUSE_USER = config.AUDIOMUSE_USER
    AUDIOMUSE_PASSWORD = config.AUDIOMUSE_PASSWORD
    if AUDIOMUSE_USER and AUDIOMUSE_USER.strip():
        effective_audiomuse_user = AUDIOMUSE_USER
    if AUDIOMUSE_PASSWORD and AUDIOMUSE_PASSWORD.strip():
        effective_audiomuse_password = AUDIOMUSE_PASSWORD
    auth_configured = bool(AUDIOMUSE_USER and AUDIOMUSE_PASSWORD)
    bootstrap_auth_mode = config.AUTH_ENABLED and not auth_configured


_password_hasher = PasswordHasher()


def _is_argon2_password_hash(value):
    return isinstance(value, str) and value.startswith('$argon2')


def _verify_audiomuse_password(stored_password, provided_password):
    if not isinstance(stored_password, str) or not isinstance(provided_password, str):
        return False
    if _is_argon2_password_hash(stored_password):
        try:
            return _password_hasher.verify(stored_password, provided_password)
        except argon2_exceptions.VerifyMismatchError:
            return False
        except argon2_exceptions.VerificationError as exc:
            app.logger.error(f"Argon2 verification failed for stored password hash: {exc}", exc_info=True)
            return False
        except Exception as exc:
            app.logger.error(f"Unexpected Argon2 verification error: {exc}", exc_info=True)
            return False
    return stored_password == provided_password

@app.context_processor
def inject_globals():
    """Injects global variables into all templates."""
    return dict(
        app_version=config.APP_VERSION,
        clap_enabled=config.CLAP_ENABLED,
        mulan_enabled=config.MULAN_ENABLED,
        auth_enabled=config.AUTH_ENABLED,
        setup_saved=setup_manager.is_setup_complete(config),
    )

# --- Authentication Middleware ---
@app.before_request
def check_auth():
    """
    Enforce authentication on all routes when auth is enabled.
    Skipped entirely when AUTH_ENABLED is False (legacy mode).
    Accepts:
      - Valid JWT in HttpOnly cookie  (browser sessions)
      - Valid API_TOKEN Bearer header (M2M / scripts)
    Public routes: /login, /auth, /logout, /static/*
    """
    if not config.AUTH_ENABLED:
        return  # Auth disabled — zero impact on existing deployments

    # Always allow public routes
    public = ('/login', '/auth', '/logout')
    if request.path in public or request.path.startswith('/static/') or request.path.startswith('/api/health'):
        return

    # Always allow bootstrap setup when auth is not configured or auth is disabled.
    if is_bootstrap_mode() and (request.path == '/setup' or request.path.startswith('/api/setup')):
        return

    # Check 1: valid JWT cookie (browser users)
    token = request.cookies.get('audiomuse_jwt')
    if token:
        try:
            pyjwt.decode(token, _jwt_secret, algorithms=['HS256'])
            return  # Valid session
        except pyjwt.ExpiredSignatureError:
            pass  # Fall through to 401/redirect
        except pyjwt.InvalidTokenError:
            pass  # Fall through to 401/redirect

    # Check 2: valid Bearer token (M2M callers)
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer ') and auth_header[7:] == config.API_TOKEN:
        return  # Valid M2M token

    # Not authenticated
    if request.path.startswith('/api/'):
        return jsonify({"error": "Unauthorized"}), 401
    else:
        return redirect(url_for('login_page'))

@app.before_request
def log_api_request():
    if request.path.startswith('/api/') and not request.path.startswith('/static/'):
        app.logger.info('API request: %s %s', request.method, request.path)


@app.before_request
def require_setup_completion():
    if request.path.startswith('/static/') or request.path == '/api/health':
        return
    if setup_manager.is_setup_complete(config):
        return

    if not config.AUTH_ENABLED:
        if request.path in ('/setup',) or request.path.startswith('/api/setup'):
            return
        if setup_manager.is_valid_env_config(config):
            return
        if request.path.startswith('/api/'):
            return jsonify({"error": "Setup required"}), 403
        return redirect(url_for('setup_page'))

    if is_bootstrap_mode():
        if request.path in ('/login', '/auth', '/logout', '/setup') or request.path.startswith('/api/setup'):
            return
        if request.path.startswith('/api/'):
            return jsonify({"error": "Setup required"}), 403
        return redirect(url_for('setup_page'))

    # Auth is configured, but setup is still incomplete.
    if request.path in ('/login', '/auth', '/logout'):
        return
    if request.path.startswith('/api/setup'):
        return
    if request.path.startswith('/api/'):
        return jsonify({"error": "Setup required"}), 403
    return redirect(url_for('setup_page'))

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'ok',
    })

# --- Swagger Setup ---
app.config['SWAGGER'] = {
    'title': 'AudioMuse-AI API',
    'uiversion': 3,
    'openapi': '3.0.0'
}
swagger = Swagger(app)

@app.teardown_appcontext
def teardown_db(e=None):
    close_db(e)

# Initialize the database schema when the application module is loaded.
# This is safe because it doesn't import other application modules.
with app.app_context():
    init_db()
    setup_manager.bootstrap_env_config_if_empty(config)

import app_setup

# --- API Endpoints ---

# --- Auth Routes ---
@app.route('/login')
def login_page():
    """Serve the login page. Redirects to / if already authenticated."""
    if not config.AUTH_ENABLED:
        return redirect(url_for('index'))
    if bootstrap_auth_mode:
        return redirect(url_for('setup_page'))
    token = request.cookies.get('audiomuse_jwt')
    if token:
        try:
            pyjwt.decode(token, _jwt_secret, algorithms=['HS256'])
            return redirect(url_for('index'))
        except pyjwt.InvalidTokenError:
            pass

    auth_warning = None
    if config.AUTH_ENABLED and (not config.AUDIOMUSE_USER or not config.AUDIOMUSE_PASSWORD):
        auth_warning = (
            'AUTH_ENABLED is true by default and one or more credentials were autogenerated and visible in FLASK log. '
            'You should set the final values for AUDIOMUSE_USER and AUDIOMUSE_PASSWORD in your environment variables. '
            'Optionally set API_TOKEN if you need external/plugin access.'
        )

    return render_template('login.html', title='Login — AudioMuse-AI', auth_warning=auth_warning)

@app.route('/auth', methods=['POST'])
def auth_endpoint():
    """
    Validate credentials and issue a JWT session cookie.
    Body: { "user": "...", "password": "..." }
    On success: sets HttpOnly JWT cookie, returns 200.
    On failure: returns 401.
    The API_TOKEN is NEVER returned in the response body.
    """
    if not config.AUTH_ENABLED:
        return jsonify({"error": "Auth not configured"}), 404
    if not auth_configured:
        app.logger.warning(
            "Auth is enabled but AUDIOMUSE_USER or AUDIOMUSE_PASSWORD is missing. API_TOKEN is optional for external/plugin calls."
        )
        return jsonify({"error": "Auth not configured"}), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = request.form.to_dict()
    if not isinstance(data, dict):
        data = {}

    user = data.get('user', '')
    password = data.get('password', '')
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if user != effective_audiomuse_user or not _verify_audiomuse_password(effective_audiomuse_password, password):
        app.logger.warning(f"Failed login attempt for user: {user!r}")
        if is_ajax:
            return jsonify({"error": "Invalid credentials"}), 401
        return render_template('login.html', title='Login — AudioMuse-AI', auth_warning=None, login_error='Invalid username or password.')

    # Issue JWT — new token at every login
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        'sub': user,
        'iat': now,
        'exp': now + datetime.timedelta(hours=8),
    }
    token = pyjwt.encode(payload, _jwt_secret, algorithm='HS256')

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        resp = make_response(jsonify({"status": "ok"}), 200)
    else:
        resp = make_response(redirect(url_for('index')))
    resp.set_cookie(
        'audiomuse_jwt',
        token,
        path='/',
        httponly=True,           # JS cannot read this cookie
        samesite='Strict',       # CSRF protection
        secure=False,            # Set to True when behind HTTPS (Caddy/Traefik handle TLS)
        max_age=8 * 3600         # 8 hours, matches JWT expiry
    )
    return resp

@app.route('/logout', methods=['POST'])
def logout_endpoint():
    """Clear the JWT session cookie and redirect to /login."""
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    if is_ajax:
        resp = make_response(jsonify({"status": "logged_out"}), 200)
    else:
        resp = make_response(redirect(url_for('login_page')))
    resp.delete_cookie('audiomuse_jwt', path='/', samesite='Strict')
    return resp

@app.route('/')
def index():
    """
    Serve the main HTML page.
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
    return render_template('index.html', title = 'AudioMuse-AI - Home Page', active='index')


@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status_endpoint(task_id):
    """
    Get the status of a specific task.
    Retrieves status information from both RQ and the database.
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
        description: Task ID not found in RQ or database.
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
                  example: Task ID not found in RQ or DB.
    """
    response = {'task_id': task_id, 'state': 'UNKNOWN', 'status_message': 'Task ID not found in RQ or DB.', 'progress': 0, 'details': {}, 'task_type_from_db': None, 'running_time_seconds': 0}
    try:
        job = Job.fetch(task_id, connection=redis_conn)
        response['state'] = job.get_status() # e.g., queued, started, finished, failed
        response['status_message'] = job.meta.get('status_message', response['state'])
        response['progress'] = job.meta.get('progress', 0)
        response['details'] = job.meta.get('details', {})
        if job.is_failed:
            response['details']['error_message'] = job.exc_info if job.exc_info else "Job failed without error info."
            response['status_message'] = "FAILED"
        elif job.is_finished:
             response['status_message'] = "SUCCESS" # RQ uses 'finished' for success
             response['progress'] = 100
        elif job.is_canceled:
            response['status_message'] = "CANCELED"
            response['progress'] = 100

    except NoSuchJobError:
        # If not in RQ, it might have been cleared or never existed. Check DB.
        pass # Will fall through to DB check

    # Augment with DB data, DB is source of truth for persisted details
    db_task_info = get_task_info_from_db(task_id)
    if db_task_info:
        response['task_type_from_db'] = db_task_info.get('task_type')
        response['running_time_seconds'] = db_task_info.get('running_time_seconds', 0)
        # If RQ state is more final (e.g. failed/finished), prefer that, else use DB
        if response['state'] not in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELED]:
            response['state'] = db_task_info.get('status', response['state']) # Use DB status if RQ is still active

        response['progress'] = db_task_info.get('progress', response['progress'])
        db_details = json.loads(db_task_info.get('details')) if db_task_info.get('details') else {}
        # Merge details: RQ meta (live) can override DB details (persisted)
        response['details'] = {**db_details, **response['details']}

        # If task is marked REVOKED in DB, this is the most accurate status for cancellation
        if db_task_info.get('status') == TASK_STATUS_REVOKED:
            response['state'] = 'REVOKED'
            response['status_message'] = 'Task revoked.'
            response['progress'] = 100
    elif response['state'] == 'UNKNOWN': # Not in RQ and not in DB
        return jsonify(response), 404

    # Prune 'checked_album_ids' from details if the task is analysis-related
    if response.get('task_type_from_db') and 'analysis' in response['task_type_from_db']:
        if isinstance(response.get('details'), dict):
            response['details'].pop('checked_album_ids', None)
    
    # Truncate log entries to last 10 entries for all task types
    if isinstance(response.get('details'), dict) and 'log' in response['details']:
        log_entries = response['details']['log']
        if isinstance(log_entries, list) and len(log_entries) > 10:
            response['details']['log'] = [
                f"... ({len(log_entries) - 10} earlier log entries truncated)",
                *log_entries[-10:]
            ]
    
    # Clean up the final response to remove confusing raw time columns
    response.pop('timestamp', None)
    response.pop('start_time', None)
    response.pop('end_time', None)

    return jsonify(response)

@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task_endpoint(task_id):
    """
    Cancel a specific task and its children.
    Marks the task and its descendants as REVOKED in the database and attempts to stop/cancel them in RQ.
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
    # Always perform cancel when the endpoint is invoked. No early returns.
    cancelled_count = cancel_job_and_children_recursive(task_id, reason=f"Cancellation requested for task {task_id} via API.")
    return jsonify({"message": f"Task {task_id} cancellation requested. {cancelled_count} cancellation actions attempted.", "task_id": task_id, "cancelled_jobs_count": cancelled_count}), 200


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
    # Exclude terminal statuses
    terminal_statuses = (TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED)
    cur.execute("SELECT task_id, task_type FROM task_status WHERE task_type = %s AND status NOT IN %s", (task_type_prefix, terminal_statuses))
    tasks_to_cancel = cur.fetchall()
    cur.close()

    total_cancelled_jobs = 0
    cancelled_main_task_ids = []
    for task_row in tasks_to_cancel:
        cancelled_jobs_for_this_main_task = cancel_job_and_children_recursive(task_row['task_id'], reason=f"Bulk cancellation for task type '{task_type_prefix}' via API.")
        if cancelled_jobs_for_this_main_task > 0:
           total_cancelled_jobs += cancelled_jobs_for_this_main_task
           cancelled_main_task_ids.append(task_row['task_id'])

    if total_cancelled_jobs > 0:
        return jsonify({"message": f"Cancellation initiated for {len(cancelled_main_task_ids)} main tasks of type '{task_type_prefix}' and their children. Total jobs affected: {total_cancelled_jobs}.", "cancelled_main_tasks": cancelled_main_task_ids}), 200
    return jsonify({"message": f"No active tasks of type '{task_type_prefix}' found to cancel."}), 404

@app.route('/api/last_task', methods=['GET'])
def get_last_overall_task_status_endpoint():
    """
    Get the status of the most recent overall main task (analysis, clustering, or cleaning).
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("""
        SELECT task_id, task_type, status, progress, details, start_time, end_time
        FROM task_status 
        WHERE parent_task_id IS NULL 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    last_task_row = cur.fetchone()
    cur.close()

    if last_task_row:
        last_task_data = dict(last_task_row)
        if last_task_data.get('details'):
            try: last_task_data['details'] = json.loads(last_task_data['details'])
            except json.JSONDecodeError: pass

        # Calculate running time in Python
        start_time = last_task_data.get('start_time')
        end_time = last_task_data.get('end_time')
        if start_time:
            effective_end_time = end_time if end_time is not None else time.time()
            last_task_data['running_time_seconds'] = max(0, effective_end_time - start_time)
        else:
            last_task_data['running_time_seconds'] = 0.0
        
        # Truncate log entries to last 10 entries
        if isinstance(last_task_data.get('details'), dict) and 'log' in last_task_data['details']:
            log_entries = last_task_data['details']['log']
            if isinstance(log_entries, list) and len(log_entries) > 10:
                last_task_data['details']['log'] = [
                    f"... ({len(log_entries) - 10} earlier log entries truncated)",
                    *log_entries[-10:]
                ]
        
        # Clean up raw time columns before sending response
        last_task_data.pop('start_time', None)
        last_task_data.pop('end_time', None)
        last_task_data.pop('timestamp', None)

        return jsonify(last_task_data), 200
        
    return jsonify({"task_id": None, "task_type": None, "status": "NO_PREVIOUS_MAIN_TASK", "details": {"log": ["No previous main task found."] }}), 200

@app.route('/api/active_tasks', methods=['GET'])
def get_active_tasks_endpoint():
    """
    Get the status of the currently active main task, if any.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    non_terminal_statuses = (TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_PROGRESS)
    cur.execute("""
        SELECT task_id, parent_task_id, task_type, sub_type_identifier, status, progress, details, start_time, end_time
        FROM task_status
        WHERE parent_task_id IS NULL AND status IN %s
        ORDER BY timestamp DESC
        LIMIT 1
    """, (non_terminal_statuses,))
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
            try:
                task_item['details'] = json.loads(task_item['details'])
                # Prune specific large or internal keys from details
                if isinstance(task_item['details'], dict):
                    task_item['details'].pop('clustering_run_job_ids', None)
                    task_item['details'].pop('checked_album_ids', None)
                    if 'best_params' in task_item['details'] and \
                       isinstance(task_item['details']['best_params'], dict) and \
                       'clustering_method_config' in task_item['details']['best_params'] and \
                       isinstance(task_item['details']['best_params']['clustering_method_config'], dict) and \
                       'params' in task_item['details']['best_params']['clustering_method_config']['params'] and \
                       isinstance(task_item['details']['best_params']['clustering_method_config']['params'], dict):
                        task_item['details']['best_params']['clustering_method_config']['params'].pop('initial_centroids', None)

            except json.JSONDecodeError:
                task_item['details'] = {"raw_details": task_item['details'], "error": "Failed to parse details JSON."}

        # Clean up raw time columns before sending response
        task_item.pop('start_time', None)
        task_item.pop('end_time', None)
        task_item.pop('timestamp', None)

        return jsonify(task_item), 200
    return jsonify({}), 200 # Return empty object if no active main task

@app.route('/api/config', methods=['GET'])
def get_config_endpoint():
    """
    Get the current server configuration values.
    """
    return jsonify({
        "num_recent_albums": config.NUM_RECENT_ALBUMS, "max_distance": config.MAX_DISTANCE,
        "max_songs_per_cluster": config.MAX_SONGS_PER_CLUSTER, "max_songs_per_artist": config.MAX_SONGS_PER_ARTIST,
        "cluster_algorithm": config.CLUSTER_ALGORITHM, "num_clusters_min": config.NUM_CLUSTERS_MIN, "num_clusters_max": config.NUM_CLUSTERS_MAX,
        "dbscan_eps_min": config.DBSCAN_EPS_MIN, "dbscan_eps_max": config.DBSCAN_EPS_MAX, "gmm_covariance_type": config.GMM_COVARIANCE_TYPE,
        "dbscan_min_samples_min": config.DBSCAN_MIN_SAMPLES_MIN, "dbscan_min_samples_max": config.DBSCAN_MIN_SAMPLES_MAX,
        "gmm_n_components_min": config.GMM_N_COMPONENTS_MIN, "gmm_n_components_max": config.GMM_N_COMPONENTS_MAX,
        "spectral_n_clusters_min": config.SPECTRAL_N_CLUSTERS_MIN, "spectral_n_clusters_max": config.SPECTRAL_N_CLUSTERS_MAX,
        "pca_components_min": config.PCA_COMPONENTS_MIN, "pca_components_max": config.PCA_COMPONENTS_MAX,
        "min_songs_per_genre_for_stratification": config.MIN_SONGS_PER_GENRE_FOR_STRATIFICATION,
        "stratified_sampling_target_percentile": config.STRATIFIED_SAMPLING_TARGET_PERCENTILE,
        "ai_model_provider": config.AI_MODEL_PROVIDER,
        "ollama_server_url": config.OLLAMA_SERVER_URL, "ollama_model_name": config.OLLAMA_MODEL_NAME,
        "openai_server_url": config.OPENAI_SERVER_URL, "openai_model_name": config.OPENAI_MODEL_NAME,
        "gemini_model_name": config.GEMINI_MODEL_NAME,
        "mistral_model_name": config.MISTRAL_MODEL_NAME,
        "top_n_moods": config.TOP_N_MOODS, "mood_labels": config.MOOD_LABELS, "clustering_runs": config.CLUSTERING_RUNS,
        "top_n_playlists": config.TOP_N_PLAYLISTS,
        "enable_clustering_embeddings": config.ENABLE_CLUSTERING_EMBEDDINGS,
        "score_weight_diversity": config.SCORE_WEIGHT_DIVERSITY,
        "score_weight_silhouette": config.SCORE_WEIGHT_SILHOUETTE,
        "score_weight_davies_bouldin": config.SCORE_WEIGHT_DAVIES_BOULDIN,
        "score_weight_calinski_harabasz": config.SCORE_WEIGHT_CALINSKI_HARABASZ,
        "score_weight_purity": config.SCORE_WEIGHT_PURITY,
        "score_weight_other_feature_diversity": config.SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY,
        "score_weight_other_feature_purity": config.SCORE_WEIGHT_OTHER_FEATURE_PURITY,
        "path_distance_metric": config.PATH_DISTANCE_METRIC
      ,"alchemy_default_n_results": config.ALCHEMY_DEFAULT_N_RESULTS
      ,"alchemy_max_n_results": config.ALCHEMY_MAX_N_RESULTS
      ,"alchemy_subtract_distance": config.ALCHEMY_SUBTRACT_DISTANCE
      ,"alchemy_subtract_distance_angular": config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR
      ,"alchemy_subtract_distance_euclid": config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN
    })

@app.route('/api/playlists', methods=['GET'])
def get_playlists_endpoint():
    """
    Get all generated playlists and their tracks from the database.
    """
    from collections import defaultdict # Local import if not used elsewhere globally
    conn = get_db()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT playlist_name, item_id, title, author FROM playlist ORDER BY playlist_name")
    rows = cur.fetchall()
    cur.close()
    playlists_data = defaultdict(list)
    for row in rows:
        playlists_data[row['playlist_name']].append({"item_id": row['item_id'], "title": row['title'], "author": row['author']})
    return jsonify(dict(playlists_data)), 200


# --- Redis index reload listener (restored pre-e308673 logic, with map reload added) ---
def listen_for_index_reloads():
  """
  Runs in a background thread to listen for messages on a Redis Pub/Sub channel.
  When a 'reload' message is received, it triggers the in-memory Voyager index and map to be reloaded.
  This is the recommended pattern for inter-process communication in this architecture,
  avoiding direct HTTP calls from workers to the web server.
  """
  # Create a new Redis connection for this thread.
  # Sharing the main redis_conn object across threads is not recommended.
  thread_redis_conn = Redis.from_url(
    config.REDIS_URL,
    socket_connect_timeout=30,
    socket_timeout=60,
    socket_keepalive=True,
    health_check_interval=30,
    retry_on_timeout=True
  )
  pubsub = thread_redis_conn.pubsub()
  pubsub.subscribe('index-updates')
  logger.info("Background thread started. Listening for Voyager index reloads on Redis channel 'index-updates'.")

  for message in pubsub.listen():
    # The first message is a confirmation of subscription, so we skip it.
    if message['type'] == 'message':
      message_data = message['data'].decode('utf-8')
      logger.info(f"Received '{message_data}' message on 'index-updates' channel.")
      if message_data == 'reload':
        # We need the application context to access 'g' and the database connection.
        with app.app_context():
          logger.info("Triggering in-memory Voyager index and map reload from background listener.")
          try:
            from tasks.voyager_manager import load_voyager_index_for_querying
            load_voyager_index_for_querying(force_reload=True)
            from tasks.artist_gmm_manager import load_artist_index_for_querying
            load_artist_index_for_querying(force_reload=True)
            from app_helper import load_map_projection, load_artist_projection
            load_map_projection('main_map', force_reload=True)
            load_artist_projection('artist_map', force_reload=True)
            # Rebuild the map JSON cache used by the /api/map endpoint
            from app_map import build_map_cache
            build_map_cache()
            
            # Reload CLAP cache (with logging)
            logger.info("Reloading CLAP embedding cache...")
            from tasks.clap_text_search import refresh_clap_cache
            clap_success = refresh_clap_cache()
            
            # Reload MuLan cache (with logging)
            logger.info("Reloading MuLan embedding cache...")
            from tasks.mulan_text_search import refresh_mulan_cache
            mulan_success = refresh_mulan_cache()
            
            logger.info(f"In-memory reload complete: Voyager ✓, Artist ✓, Maps ✓, CLAP {'✓' if clap_success else '✗'}, MuLan {'✓' if mulan_success else '✗'}")
          except Exception as e:
            logger.error(f"Error reloading indexes/maps from background listener: {e}", exc_info=True)
      elif message_data == 'reload-artist':
        # Reload artist similarity index only (legacy support)
        with app.app_context():
          logger.info("Triggering in-memory artist similarity index reload from background listener.")
          try:
            from tasks.artist_gmm_manager import load_artist_index_for_querying
            load_artist_index_for_querying(force_reload=True)
            logger.info("In-memory artist similarity index reloaded successfully by background listener.")
          except Exception as e:
            logger.error(f"Error reloading artist similarity index from background listener: {e}", exc_info=True)





# --- Import and Register Blueprints ---
# This is the original, working structure.
from app_helper import get_child_tasks_from_db, get_score_data_by_ids, get_tracks_by_ids, save_track_analysis_and_embedding, track_exists, update_playlist_table

# Import tasks modules to ensure they're available to RQ workers
import tasks.clustering
import tasks.analysis


from app_chat import chat_bp
from app_clustering import clustering_bp
from app_analysis import analysis_bp
from app_cron import cron_bp, run_due_cron_jobs
from app_voyager import voyager_bp
from app_sonic_fingerprint import sonic_fingerprint_bp
from app_path import path_bp
from app_collection import collection_bp
from app_external import external_bp # --- NEW: Import the external blueprint ---
from app_alchemy import alchemy_bp
from app_map import map_bp
from app_waveform import waveform_bp
from app_artist_similarity import artist_similarity_bp
from app_clap_search import clap_search_bp
from app_mulan_search import mulan_search_bp
from app_backup import backup_bp

app.register_blueprint(chat_bp, url_prefix='/chat')
app.register_blueprint(clustering_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(cron_bp)
app.register_blueprint(voyager_bp)
app.register_blueprint(sonic_fingerprint_bp)
app.register_blueprint(path_bp)
app.register_blueprint(collection_bp)
app.register_blueprint(external_bp, url_prefix='/external') # --- NEW: Register the external blueprint ---
app.register_blueprint(alchemy_bp)
app.register_blueprint(map_bp)
app.register_blueprint(waveform_bp)
app.register_blueprint(artist_similarity_bp)
app.register_blueprint(clap_search_bp)
app.register_blueprint(mulan_search_bp)
app.register_blueprint(backup_bp)

# --- Startup: Load indexes and caches (Flask server only, NOT RQ workers) ---
# RQ workers import app.py but should NOT load indexes or start background threads.
# The env var AUDIOMUSE_ROLE is set to 'worker' by rq_worker.py / rq_worker_high_priority.py.
_is_worker = os.environ.get('AUDIOMUSE_ROLE') == 'worker'

try:
  os.makedirs(config.TEMP_DIR, exist_ok=True)
except OSError:
  logger.debug(f"Could not create TEMP_DIR '{config.TEMP_DIR}' (may be running in test/CI environment)")

if not _is_worker:
  with app.app_context():
    # --- Initial Voyager Index Load ---
    from tasks.voyager_manager import load_voyager_index_for_querying
    load_voyager_index_for_querying()
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
      from app_helper import load_artist_projection
      load_artist_projection('artist_map')
      logger.info("In-memory artist component projection loaded at startup.")
    except Exception as e:
      logger.debug(f"No precomputed artist projection to load at startup or load failed: {e}")
    # Load CLAP embeddings cache (model will lazy-load on first use)
    try:
      if config.CLAP_ENABLED:
        # Load CLAP embeddings cache (15MB) - model lazy-loads on first search to save 3GB RAM
        from tasks.clap_text_search import load_clap_cache_from_db, load_top_queries_from_db
        if load_clap_cache_from_db():
          logger.info("CLAP text search cache loaded at startup (embeddings only).")
          logger.info("CLAP model will lazy-load on first text search (~1-2s delay, saves 3GB RAM).")
        
        # Load top queries from database (default queries only, no computation)
        # This must run even if no CLAP embeddings exist yet (first startup)
        has_existing = load_top_queries_from_db()
        if has_existing:
          logger.info("Loaded top queries from database (defaults).")
        else:
          logger.info("No queries found in database (should not happen - check DB)")
    except Exception as e:
      logger.debug(f"CLAP cache not loaded at startup (may be disabled or failed): {e}")
    # Load MuLan embeddings cache (model will lazy-load on first use)
    try:
      if config.MULAN_ENABLED:
        # Load MuLan embeddings cache - models lazy-load on first search to save RAM
        from tasks.mulan_text_search import load_mulan_cache_from_db, load_top_queries_from_db as load_mulan_top_queries_from_db
        if load_mulan_cache_from_db():
          logger.info("MuLan text search cache loaded at startup (embeddings only).")
          logger.info("MuLan models will lazy-load on first text search.")
        
        # Load top queries from database
        # This must run even if no MuLan embeddings exist yet (first startup)
        has_existing = load_mulan_top_queries_from_db()
        if has_existing:
          logger.info("Loaded MuLan top queries from database (defaults).")
        else:
          logger.info("No MuLan queries found in database (defaults inserted)")
    except Exception as e:
      logger.debug(f"MuLan cache not loaded at startup (may be disabled or failed): {e}")

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

# --- Start Background Listener Thread (Flask server only) ---
if not _is_worker:
  listener_thread = threading.Thread(target=listen_for_index_reloads, daemon=True)
  listener_thread.start()

  # Start a cron manager thread that checks enabled cron entries every 60 seconds
  def _cron_manager_loop():
    try:
      from time import sleep
      while True:
        try:
          with app.app_context():
            run_due_cron_jobs()
        except Exception:
          app.logger.exception('cron manager failed')
        sleep(60)
    except Exception:
      app.logger.exception('cron manager main loop error')

  cron_thread = threading.Thread(target=_cron_manager_loop, daemon=True)
  cron_thread.start()
else:
  logger.info('Running as RQ worker — skipping index loading, Redis listener, and cron thread.')

if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0', port=8000)
