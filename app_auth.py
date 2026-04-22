"""Centralized authentication and user-management layer.

This module owns everything related to the auth/setup barrier and the
``audiomuse_users`` table:

* Role constants and password hashing.
* CRUD helpers for user accounts.
* The ``check_setup_needed`` / ``check_auth_needed`` / ``check_admin_needed``
  barrier functions used as ``before_request`` guards.
* The Flask routes for ``/login``, ``/auth``, ``/logout`` and ``/api/users``.
* One-shot helpers for the legacy env -> users-table seed and the JWT secret
  resolution used at startup.

The table creation itself still lives in ``app_helper.init_db`` (alongside
all other schema objects) so a cold start only calls one init routine.
"""

import datetime
import logging
import os
import secrets

from flask import (
    current_app,
    g,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    url_for,
)
import jwt as pyjwt
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)


# --- User model constants ---------------------------------------------------

USER_ROLE_USER = 'user'
USER_ROLE_ADMIN = 'admin'
_VALID_USER_ROLES = (USER_ROLE_USER, USER_ROLE_ADMIN)


def _normalize_role(role):
    if role is None:
        return USER_ROLE_USER
    if not isinstance(role, str):
        return None
    role = role.strip().lower()
    if role not in _VALID_USER_ROLES:
        return None
    return role


def _get_password_hasher():
    from argon2 import PasswordHasher
    return PasswordHasher()


def _get_db():
    from app_helper import get_db
    return get_db()


# --- Module-level state set by init_app -------------------------------------

# Zero-arg callable returning the current JWT secret. ``init_app`` stores the
# getter here so the route handlers below can resolve the secret lazily
# (it's assigned by ``app.py`` only after ``init_db`` completes).
_jwt_secret_getter = None


def _jwt_secret():
    if _jwt_secret_getter is None:
        return None
    return _jwt_secret_getter()


# --- User CRUD --------------------------------------------------------------

def list_additional_users(username=None):
    """Return dicts ``{id, username, role, created_at}`` for user rows.

    When ``username`` is provided, the query is scoped to that single row
    so non-admin callers never pull other accounts out of the database.
    Password hashes are never returned.
    """
    db = _get_db()
    with db.cursor(cursor_factory=DictCursor) as cur:
        if username is None:
            cur.execute(
                "SELECT id, username, role, created_at FROM audiomuse_users ORDER BY username ASC"
            )
        else:
            cur.execute(
                "SELECT id, username, role, created_at FROM audiomuse_users WHERE username = %s",
                (username,),
            )
        rows = cur.fetchall()
    out = []
    for row in rows:
        created = row['created_at']
        out.append({
            'id': row['id'],
            'username': row['username'],
            'role': row['role'] or USER_ROLE_USER,
            'created_at': created.isoformat() if created is not None else None,
        })
    return out


def count_admin_users():
    """Return the number of admin rows in ``audiomuse_users``."""
    db = _get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM audiomuse_users WHERE role = %s",
            (USER_ROLE_ADMIN,),
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def get_additional_user_by_id(user_id):
    """Return ``{id, username, role}`` for a given row id, or None."""
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return None
    db = _get_db()
    with db.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            "SELECT id, username, role FROM audiomuse_users WHERE id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        'id': row['id'],
        'username': row['username'],
        'role': row['role'] or USER_ROLE_USER,
    }


def create_additional_user(username, password, role=USER_ROLE_USER):
    """Create a new user. Returns ``(ok, error_message)``."""
    if not isinstance(username, str) or not username.strip():
        return False, "Username is required."
    if not isinstance(password, str) or not password:
        return False, "Password is required."
    normalized_role = _normalize_role(role)
    if normalized_role is None:
        return False, "Invalid role."
    username = username.strip()
    if len(username) > 128:
        return False, "Username is too long."

    hasher = _get_password_hasher()
    try:
        password_hash = hasher.hash(password)
    except Exception as exc:
        logger.error(f"Failed to hash password for new user {username!r}: {exc}", exc_info=True)
        return False, "Failed to hash password."

    db = _get_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO audiomuse_users (username, password_hash, role) VALUES (%s, %s, %s) "
            "ON CONFLICT (username) DO NOTHING RETURNING id",
            (username, password_hash, normalized_role),
        )
        row = cur.fetchone()
    db.commit()
    if row is None:
        return False, "A user with that username already exists."
    return True, None


def delete_additional_user(user_id):
    """Delete a user by id. Returns True when a row was deleted."""
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return False
    db = _get_db()
    with db.cursor() as cur:
        cur.execute("DELETE FROM audiomuse_users WHERE id = %s", (user_id,))
        deleted = cur.rowcount
    db.commit()
    return bool(deleted)


def delete_additional_user_safe(user_id):
    """Atomically delete a user, refusing to delete the last admin.

    The row lookup, last-admin check, and delete all run in a single
    transaction that locks the affected rows with ``SELECT ... FOR UPDATE``,
    so concurrent admin deletions cannot race past the guard and end up
    with zero admins.

    Returns ``(status, error)`` where ``status`` is one of:
    - ``"deleted"``: the row was deleted (error is None)
    - ``"not_found"``: no user with that id
    - ``"last_admin"``: refused because it would remove the last admin
    - ``"invalid_id"``: the id was not an integer
    - ``"error"``: a database error occurred (error carries the message)
    """
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return "invalid_id", "Invalid user id."
    db = _get_db()
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT role FROM audiomuse_users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if row is None:
                db.rollback()
                return "not_found", None
            target_role = row[0]
            if target_role == USER_ROLE_ADMIN:
                # To avoid deadlocks when two admins delete each other at the
                # same time, acquire the admin-group lock first in a stable order.
                cur.execute(
                    "SELECT id FROM audiomuse_users WHERE role = %s ORDER BY id FOR UPDATE",
                    (USER_ROLE_ADMIN,),
                )
                admin_count = len(cur.fetchall())
                if admin_count <= 1:
                    db.rollback()
                    return "last_admin", None
            cur.execute("DELETE FROM audiomuse_users WHERE id = %s", (user_id,))
            deleted = cur.rowcount
        db.commit()
        if not deleted:
            return "not_found", None
        return "deleted", None
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error(f"Failed to atomically delete user {user_id}: {exc}", exc_info=True)
        return "error", "Database error while deleting user."


def update_additional_user_password(user_id, new_password):
    """Update a user's password. Returns ``(ok, error_message)``."""
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return False, "Invalid user id."
    if not isinstance(new_password, str) or not new_password:
        return False, "Password is required."
    try:
        password_hash = _get_password_hasher().hash(new_password)
    except Exception as exc:
        logger.error(f"Failed to hash new password for user {user_id}: {exc}", exc_info=True)
        return False, "Failed to hash password."
    db = _get_db()
    with db.cursor() as cur:
        cur.execute(
            "UPDATE audiomuse_users SET password_hash = %s WHERE id = %s",
            (password_hash, user_id),
        )
        updated = cur.rowcount
    db.commit()
    if not updated:
        return False, "User not found."
    return True, None


def verify_additional_user(username, password):
    """Verify credentials. Returns the role on success, otherwise None."""
    if not isinstance(username, str) or not isinstance(password, str):
        return None
    db = _get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT password_hash, role FROM audiomuse_users WHERE username = %s",
            (username,),
        )
        row = cur.fetchone()
    if not row:
        return None
    stored, role = row[0], row[1]
    if not isinstance(stored, str) or not stored:
        return None
    try:
        import argon2
    except ImportError as exc:
        logger.error(f"argon2 is not installed: {exc}", exc_info=True)
        return None

    try:
        _get_password_hasher().verify(stored, password)
    except (argon2.exceptions.VerifyMismatchError, argon2.exceptions.VerificationError):
        return None
    except Exception as exc:
        logger.error(f"Unexpected error during password verification: {exc}", exc_info=True)
        return None
    return _normalize_role(role) or USER_ROLE_USER


def upsert_admin_user(username, password):
    """Create an admin row, or update the password and force admin role when
    the username already exists. Returns ``(ok, error_message)``.
    Used by the setup wizard for the install-time admin.
    """
    if not isinstance(username, str) or not username.strip():
        return False, "Username is required."
    if not isinstance(password, str) or not password:
        return False, "Password is required."
    username = username.strip()
    if len(username) > 128:
        return False, "Username is too long."
    try:
        password_hash = _get_password_hasher().hash(password)
    except Exception as exc:
        logger.error(f"Failed to hash password for admin {username!r}: {exc}", exc_info=True)
        return False, "Failed to hash password."
    db = _get_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO audiomuse_users (username, password_hash, role) VALUES (%s, %s, %s) "
            "ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash, role = 'admin'",
            (username, password_hash, USER_ROLE_ADMIN),
        )
    db.commit()
    return True, None


def seed_admin_from_env():
    """One-time bridge for legacy installs.

    Bootstraps the first admin row in ``audiomuse_users`` when the table is
    empty and legacy admin credentials are present. Source precedence:

    1. ``audiomuse_users`` already has an admin -> no-op and purge stale
       legacy config too.
    2. Legacy ``AUDIOMUSE_USER`` / ``AUDIOMUSE_PASSWORD`` values in
       ``app_config`` -> seed from app_config and delete those rows.
    3. Legacy ``AUDIOMUSE_USER`` / ``AUDIOMUSE_PASSWORD`` environment vars ->
       seed from env.

    Idempotent: safe to call on every startup.
    """
    # 1. Users table already has an admin - clean up legacy config rows and bail.
    try:
        if count_admin_users() > 0:
            purge_legacy_admin_config()
            return False
    except Exception as exc:
        logger.error(f"seed_admin_from_env: failed to count admins: {exc}", exc_info=True)
        return False

    # 2. Fall back to legacy rows persisted in app_config.
    user, password, source = _read_legacy_admin_from_app_config()

    # 3. Fall back to real process environment variables.
    if not (user and password):
        user = os.environ.get('AUDIOMUSE_USER', '') or ''
        password = os.environ.get('AUDIOMUSE_PASSWORD', '') or ''
        source = 'env'

    if not (isinstance(user, str) and user.strip() and isinstance(password, str) and password):
        return False

    # Support legacy argon2 hashes stored in AUDIOMUSE_PASSWORD by inserting
    # them verbatim; otherwise hash the plaintext here.
    try:
        if isinstance(password, str) and password.startswith('$argon2'):
            password_hash = password
        else:
            password_hash = _get_password_hasher().hash(password)
    except Exception as exc:
        logger.error(f"seed_admin_from_env: failed to prepare password: {exc}", exc_info=True)
        return False
    db = _get_db()
    try:
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO audiomuse_users (username, password_hash, role) VALUES (%s, %s, %s) "
                "ON CONFLICT (username) DO NOTHING",
                (user.strip(), password_hash, USER_ROLE_ADMIN),
            )
        db.commit()
        safe_source = 'app_config' if source == 'app_config' else ('env' if source == 'env' else 'unknown')
        logger.info(
            "Seeded admin into audiomuse_users from %s.",
            safe_source,
        )
        # If we seeded from app_config, drop the legacy rows so subsequent
        # deletes of this admin from /users are not undone on next boot.
        if source == 'app_config':
            purge_legacy_admin_config()
        return True
    except Exception as exc:
        db.rollback()
        logger.error(f"seed_admin_from_env: insert failed: {exc}", exc_info=True)
        return False


def _read_legacy_admin_from_app_config():
    """Return ``(user, password, 'app_config')`` when legacy admin rows are
    present in ``app_config``, otherwise ``('', '', 'app_config')``.
    """
    db = _get_db()
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT key, value FROM app_config "
                "WHERE key IN ('AUDIOMUSE_USER', 'AUDIOMUSE_PASSWORD')"
            )
            rows = cur.fetchall() or []
    except Exception as exc:
        logger.error(
            f"_read_legacy_admin_from_app_config: lookup failed: {exc}",
            exc_info=True,
        )
        return '', '', 'app_config'
    values = {row[0]: row[1] for row in rows}
    return values.get('AUDIOMUSE_USER', '') or '', values.get('AUDIOMUSE_PASSWORD', '') or '', 'app_config'


def purge_legacy_admin_config():
    """Remove any stale ``AUDIOMUSE_USER`` / ``AUDIOMUSE_PASSWORD`` rows from
    ``app_config``. Called once an admin exists in ``audiomuse_users`` so
    the users table remains the sole source of truth.
    """
    db = _get_db()
    try:
        with db.cursor() as cur:
            cur.execute(
                "DELETE FROM app_config WHERE key IN ('AUDIOMUSE_USER', 'AUDIOMUSE_PASSWORD')"
            )
            removed = cur.rowcount
        db.commit()
        if removed:
            logger.info(
                "Purged %d legacy AUDIOMUSE_USER/AUDIOMUSE_PASSWORD row(s) from app_config.",
                removed,
            )
        return removed
    except Exception as exc:
        db.rollback()
        logger.error(f"purge_legacy_admin_config failed: {exc}", exc_info=True)
        return 0


# --- Barrier helpers --------------------------------------------------------

def check_setup_needed():
    """Return True when the install still needs the setup wizard."""
    from tasks.setup_manager import SetupManager
    import config as _cfg
    sm = SetupManager()

    if not sm._is_valid_server_config(_cfg):
        return True

    auth_enabled = getattr(_cfg, 'AUTH_ENABLED', True)
    if isinstance(auth_enabled, str):
        auth_enabled = auth_enabled.strip().lower() == 'true'
    if not auth_enabled:
        return False

    try:
        return count_admin_users() <= 0
    except Exception as exc:
        logger.error(f"Failed to count admin users while checking setup status: {exc}", exc_info=True)
        return True


def check_auth_needed(jwt_secret):
    """Check if the current request requires authentication.

    Returns None when the request is authenticated or auth is disabled.
    Returns a Response (redirect or JSON 401) otherwise.
    Populates ``flask.g.auth_role`` and ``flask.g.auth_user``.
    """
    import config as _cfg

    # Default: when auth is disabled every request behaves as an admin.
    g.auth_role = 'admin'
    g.auth_user = None

    if not _cfg.AUTH_ENABLED:
        return None

    # Check valid JWT cookie
    token = request.cookies.get('audiomuse_jwt')
    if token:
        try:
            payload = pyjwt.decode(token, jwt_secret, algorithms=['HS256'])
            # Backward-compat: tokens issued before the multi-user feature
            # have no 'role' claim; treat them as admin.
            g.auth_role = payload.get('role', 'admin')
            g.auth_user = payload.get('sub')
            return None
        except pyjwt.InvalidTokenError:
            pass

    # Check valid Bearer token (M2M callers) - always admin-equivalent.
    # Use secrets.compare_digest to avoid leaking token contents via timing.
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer ') and _cfg.API_TOKEN and secrets.compare_digest(auth_header[7:], _cfg.API_TOKEN):
        g.auth_role = 'admin'
        g.auth_user = None
        return None

    # Not authenticated
    if request.path.startswith('/api/'):
        return jsonify({"error": "Unauthorized"}), 401
    return redirect(url_for('login_page'))


# URL prefixes reserved for admin users. Normal users get a 403 (API) or a
# redirect to the dashboard (page requests).
# Note: /users and /api/users are intentionally NOT here. Any authenticated
# user can reach them; the per-request handlers enforce that non-admins
# only see and modify their own account.
_ADMIN_PATH_PREFIXES = (
    '/setup', '/api/setup',
    '/cleaning', '/api/cleaning',
    '/cron', '/api/cron',
    '/backup', '/api/backup',
    '/provider-migration', '/api/migration',
    '/analysis', '/api/analysis', '/api/clustering',
)


def is_admin_path(path):
    """Return True if ``path`` should only be accessible to admin users."""
    if not path:
        return False
    for prefix in _ADMIN_PATH_PREFIXES:
        if path == prefix or path.startswith(prefix + '/'):
            return True
    return False


def check_admin_needed():
    """If the current request targets an admin-only path and the caller is
    not an admin, return an appropriate response. Otherwise return None.
    Must be called *after* ``check_auth_needed``.
    """
    import config as _cfg

    if not _cfg.AUTH_ENABLED:
        return None
    if not is_admin_path(request.path):
        return None
    role = getattr(g, 'auth_role', None)
    if role == 'admin':
        return None
    current_app.logger.warning(
        "Non-admin user denied access to admin path %s",
        request.path,
    )
    if request.path.startswith('/api/'):
        if request.path == '/api/setup':
            return jsonify({
                "error": "Error saving configuration: Non-admin user denied access to admin path. Please refresh the page and try again."
            }), 403
        return jsonify({"error": "Forbidden"}), 403
    return redirect(url_for('dashboard_bp.dashboard_page'))


def auth_setup_barrier():
    """Single before_request guard: setup -> auth -> admin."""
    if request.path.startswith('/static/') or request.path == '/api/health':
        return

    if check_setup_needed():
        if request.path in ('/setup', '/api/setup'):
            return
        if request.path.startswith('/api/'):
            current_app.logger.warning(
                "API access blocked because setup is still required: %s",
                request.path,
            )
            return jsonify({"error": "Setup required"}), 403
        return redirect(url_for('setup_page'))

    if request.path in ('/login', '/auth', '/logout'):
        return
    auth_response = check_auth_needed(_jwt_secret())
    if auth_response:
        return auth_response

    admin_response = check_admin_needed()
    if admin_response:
        return admin_response


# --- JWT secret resolution --------------------------------------------------

def resolve_jwt_secret(setup_manager):
    """Return a usable JWT secret, generating and persisting one if needed.

    Reads ``config.JWT_SECRET`` first; when empty and auth is enabled,
    refreshes config (another worker may have saved one), then generates and
    stores a new random secret. Safe to call only after ``init_db``.
    """
    import config as _cfg
    secret = _cfg.JWT_SECRET
    if secret or not _cfg.AUTH_ENABLED:
        return secret
    _cfg.refresh_config()
    secret = _cfg.JWT_SECRET
    if secret:
        return secret
    secret = secrets.token_hex(32)
    setup_manager.save_config_values({'JWT_SECRET': secret})
    _cfg.JWT_SECRET = secret
    logger.warning(
        "JWT_SECRET was not set. A random secret has been generated and saved to the database. "
        "Set JWT_SECRET in your .env for full control."
    )
    return secret


# --- Auth routes ------------------------------------------------------------
# Routes are registered directly on the Flask app inside init_app() so their
# endpoint names stay unqualified (``login_page``, ``logout_endpoint``, ...)
# and match the names used by existing templates via ``url_for``.

def login_page():
    """Serve the login page. Redirects to the dashboard when already authenticated."""
    import config as _cfg
    if not _cfg.AUTH_ENABLED:
        return redirect(url_for('dashboard_bp.dashboard_page'))
    token = request.cookies.get('audiomuse_jwt')
    if token:
        try:
            pyjwt.decode(token, _jwt_secret(), algorithms=['HS256'])
            return redirect(url_for('dashboard_bp.dashboard_page'))
        except pyjwt.InvalidTokenError:
            pass
    return render_template('login.html', title='Login - AudioMuse-AI')


def auth_endpoint():
    """Validate credentials and issue a JWT session cookie.

    Body: ``{"user": "...", "password": "..."}``
    Success: sets HttpOnly JWT cookie, returns 200.
    Failure: returns 401. The API_TOKEN is never returned in the body.
    """
    import config as _cfg
    if not _cfg.AUTH_ENABLED:
        return jsonify({"error": "Auth not configured"}), 404
    try:
        admin_count = count_admin_users()
    except Exception as exc:
        current_app.logger.error(
            'Failed to count admin users during authentication: %s',
            exc,
            exc_info=True,
        )
        return jsonify({"error": "Database error while checking admin accounts."}), 500
    if admin_count <= 0:
        current_app.logger.warning(
            "Auth is enabled but no admin account is configured. "
            "Complete the setup wizard to create one."
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

    role = verify_additional_user(user, password) if user else None
    if role is None:
        current_app.logger.warning(f"Failed login attempt for user: {user!r}")
        if is_ajax:
            return jsonify({"error": "Invalid credentials"}), 401
        return render_template(
            'login.html',
            title='Login - AudioMuse-AI',
            login_error='Invalid username or password.',
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        'sub': user,
        'role': role,
        'iat': now,
        'exp': now + datetime.timedelta(hours=8),
    }
    token = pyjwt.encode(payload, _jwt_secret(), algorithm='HS256')

    if is_ajax:
        resp = make_response(jsonify({"status": "ok"}), 200)
    else:
        resp = make_response(redirect(url_for('dashboard_bp.dashboard_page')))
    resp.set_cookie(
        'audiomuse_jwt',
        token,
        path='/',
        httponly=True,
        samesite='Strict',
        # Mark the cookie Secure when the request came in over HTTPS so
        # production deployments get the hardened flag while local HTTP
        # development still works.
        secure=request.is_secure,
        max_age=8 * 3600,
    )
    return resp


def logout_endpoint():
    """Clear the JWT session cookie and redirect to /login."""
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    if is_ajax:
        resp = make_response(jsonify({"status": "logged_out"}), 200)
    else:
        resp = make_response(redirect(url_for('login_page')))
    resp.delete_cookie('audiomuse_jwt', path='/', samesite='Strict')
    return resp


# --- /api/users -------------------------------------------------------------
# Admins can list and manage every account. Non-admins can only see and
# modify their own row; the handlers below enforce that explicitly.

def list_users_endpoint():
    """List user accounts. Admins see everyone, non-admins see only themselves."""
    import config as _cfg
    if not _cfg.AUTH_ENABLED:
        return jsonify({"error": "Auth not configured"}), 404
    role = getattr(g, 'auth_role', None)
    current_username = getattr(g, 'auth_user', None)
    try:
        if role == 'admin':
            users = list_additional_users()
        else:
            # Scope the query to the caller so non-admins never receive
            # other users' rows from the database at all.
            users = list_additional_users(username=current_username) if current_username else []
    except Exception as exc:
        current_app.logger.error(f"Failed to list users: {exc}", exc_info=True)
        return jsonify({"error": "Failed to list users"}), 500
    return jsonify({
        "users": users,
        "current_user": current_username,
        "is_admin": role == 'admin',
    })


def create_user_endpoint():
    """Create a new user (role 'user' or 'admin'). Admin-only."""
    import config as _cfg
    if not _cfg.AUTH_ENABLED:
        return jsonify({"error": "Auth not configured"}), 404
    if getattr(g, 'auth_role', None) != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    role = (data.get('role') or USER_ROLE_USER).strip().lower()
    if role not in (USER_ROLE_USER, USER_ROLE_ADMIN):
        return jsonify({"error": "Role must be 'user' or 'admin'."}), 400
    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400
    ok, err = create_additional_user(username, password, role=role)
    if not ok:
        return jsonify({"error": err or "Failed to create user."}), 400
    return jsonify({"status": "ok"}), 201


def delete_user_endpoint(user_id):
    """Delete a user by id, with safety checks:
    - admin-only
    - an admin cannot delete themselves
    - deleting the last admin is refused
    """
    import config as _cfg
    if not _cfg.AUTH_ENABLED:
        return jsonify({"error": "Auth not configured"}), 404
    if getattr(g, 'auth_role', None) != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    target = get_additional_user_by_id(user_id)
    if not target:
        return jsonify({"error": "User not found."}), 404
    current_username = getattr(g, 'auth_user', None)
    if current_username and target['username'] == current_username:
        return jsonify({"error": "You cannot delete your own account."}), 400
    status, err = delete_additional_user_safe(user_id)
    if status == "deleted":
        return jsonify({"status": "ok"})
    if status == "not_found":
        return jsonify({"error": "User not found."}), 404
    if status == "last_admin":
        return jsonify({"error": "At least one admin account must remain."}), 400
    if status == "invalid_id":
        return jsonify({"error": err or "Invalid user id."}), 400
    # status == "error"
    return jsonify({"error": err or "Could not delete user; please try again."}), 500


def update_user_password_endpoint(user_id):
    """Change a user's password.
    - admin can change anyone's password
    - a non-admin can only change their own password
    """
    import config as _cfg
    if not _cfg.AUTH_ENABLED:
        return jsonify({"error": "Auth not configured"}), 404
    target = get_additional_user_by_id(user_id)
    if not target:
        return jsonify({"error": "User not found."}), 404
    role = getattr(g, 'auth_role', None)
    current_username = getattr(g, 'auth_user', None)
    if role != 'admin' and target['username'] != current_username:
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(silent=True) or {}
    new_password = data.get('password') or ''
    if not isinstance(new_password, str) or not new_password:
        return jsonify({"error": "Password is required."}), 400
    ok, err = update_additional_user_password(user_id, new_password)
    if not ok:
        return jsonify({"error": err or "Failed to update password."}), 400
    return jsonify({"status": "ok"})


# --- Flask registration -----------------------------------------------------

def init_app(app, setup_manager, jwt_secret_getter):
    """Wire the auth barrier and auth routes onto ``app``.

    ``jwt_secret_getter`` is a zero-arg callable returning the current JWT
    secret; this indirection lets ``app.py`` resolve the secret lazily after
    ``init_db`` without a hard dependency on the startup order.
    ``setup_manager`` is accepted for API symmetry with the rest of the app;
    the module does not currently need to keep a reference.
    """
    global _jwt_secret_getter
    _jwt_secret_getter = jwt_secret_getter

    app.before_request(auth_setup_barrier)
    app.add_url_rule('/login', endpoint='login_page', view_func=login_page, methods=['GET'])
    app.add_url_rule('/auth', endpoint='auth_endpoint', view_func=auth_endpoint, methods=['POST'])
    app.add_url_rule('/logout', endpoint='logout_endpoint', view_func=logout_endpoint, methods=['POST'])
    app.add_url_rule('/api/users', endpoint='list_users_endpoint', view_func=list_users_endpoint, methods=['GET'])
    app.add_url_rule('/api/users', endpoint='create_user_endpoint', view_func=create_user_endpoint, methods=['POST'])
    app.add_url_rule('/api/users/<int:user_id>', endpoint='delete_user_endpoint', view_func=delete_user_endpoint, methods=['DELETE'])
    app.add_url_rule('/api/users/<int:user_id>/password', endpoint='update_user_password_endpoint', view_func=update_user_password_endpoint, methods=['PUT'])
