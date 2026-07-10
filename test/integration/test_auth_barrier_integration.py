# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""End-to-end auth barrier tests over a real Flask app and Postgres.

Exercises the before-request auth guard end to end: unauthenticated
requests, real logins issuing working sessions, JWT and bearer tokens,
session revocation, and admin-path role enforcement against a live database.

Main Features:
* Unauthenticated API 401 / page redirect and open health endpoint.
* Rejects alg=none, wrong-secret, expired, unknown-user, and wrong-bearer tokens.
* Enforces admin vs user roles on admin and normal paths; the database row's
  role is authoritative over the token's role claim.
* User creation, password changes, and deletions require current_password
  confirmation, and changes/deletions revoke the target user's older
  sessions (bearer callers stay exempt).
"""

import base64
import datetime
import json
import os
import shutil
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

try:
    import jwt as pyjwt
except Exception:  # pragma: no cover - PyJWT is in test/requirements.txt
    pyjwt = None


_TEST_SECRET = 'integration-test-secret-do-not-use-in-prod'
_BEARER_TOKEN = 'integration-bearer-token'

_USERS_DDL = (
    "CREATE TABLE audiomuse_users (id SERIAL PRIMARY KEY, "
    "username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, "
    "role TEXT NOT NULL DEFAULT 'user', "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "password_changed_at TIMESTAMP)"
)


@pytest.fixture(scope='session')
def pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")
    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.skip(f"AUDIOMUSE_TEST_DATABASE_URL not reachable: {e}")
        yield dsn
        return
    try:
        import pgserver
    except Exception:
        pytest.skip(
            "No test database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable "
            "DB, or `pip install pgserver` for an ephemeral local instance."
        )
    data_dir = tempfile.mkdtemp(prefix='audiomuse_pg_')
    try:
        server = pgserver.get_server(data_dir)
        try:
            yield server.get_uri()
        finally:
            server.cleanup()
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def _hs256_token(role, secret=_TEST_SECRET, expired=False, sub=None, iat_offset_seconds=0):
    if sub is None:
        sub = 'root' if role == 'admin' else 'plainuser'
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=iat_offset_seconds
    )
    exp = now - datetime.timedelta(hours=1) if expired else now + datetime.timedelta(hours=1)
    payload = {'sub': sub, 'role': role, 'iat': now, 'exp': exp}
    return pyjwt.encode(payload, secret, algorithm='HS256')


def _alg_none_token(role, sub='root'):
    def _b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b'=').decode()

    header = _b64({'alg': 'none', 'typ': 'JWT'})
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    payload = _b64({'sub': sub, 'role': role, 'iat': now, 'exp': 9999999999})
    return f"{header}.{payload}."


def _set_cookie(client, token):
    try:
        client.set_cookie('audiomuse_jwt', token)
    except TypeError:
        client.set_cookie('localhost', 'audiomuse_jwt', token)


@pytest.fixture
def barrier_client(pg_dsn, monkeypatch):
    if pyjwt is None:
        pytest.skip("PyJWT not importable")
    from flask import Flask, jsonify

    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS audiomuse_users CASCADE")
        cur.execute(_USERS_DDL)
    conn.commit()

    import config
    import app_auth
    from tasks.setup_manager import SetupManager

    monkeypatch.setattr(app_auth, '_get_db', lambda: conn)
    monkeypatch.setattr(config, 'AUTH_ENABLED', True)
    monkeypatch.setattr(config, 'API_TOKEN', _BEARER_TOKEN)
    monkeypatch.setattr(config, 'MEDIASERVER_TYPE', 'jellyfin')
    for field in config.MEDIASERVER_FIELDS_BY_TYPE['jellyfin']:
        monkeypatch.setattr(config, field, 'configured', raising=False)

    ok, err = app_auth.create_additional_user('root', 'rootpw', 'admin')
    assert ok, err
    ok, err = app_auth.create_additional_user('plainuser', 'plainpw', 'user')
    assert ok, err

    app = Flask(__name__, template_folder=os.path.join(_REPO_ROOT, 'templates'))
    app_auth.init_app(app, SetupManager(), lambda: _TEST_SECRET)

    @app.route('/api/health', methods=['GET'])
    def _health():
        return jsonify({"status": "ok"})

    @app.route('/api/protected', methods=['GET'])
    def _protected():
        return jsonify({"ok": True})

    @app.route('/api/cron', methods=['GET'])
    def _admin_only():
        return jsonify({"ok": True})

    @app.route('/somepage', methods=['GET'])
    def _page():
        return "page", 200

    client = app.test_client()
    yield client, conn
    conn.close()


@pytest.mark.integration
class TestAuthBarrierInvariants:
    def test_unauthenticated_api_is_401(self, barrier_client):
        client, _ = barrier_client
        resp = client.get('/api/protected')
        assert resp.status_code == 401
        assert resp.get_json().get('error') == 'Unauthorized'

    def test_unauthenticated_page_redirects_to_login(self, barrier_client):
        client, _ = barrier_client
        resp = client.get('/somepage')
        assert resp.status_code == 302
        assert '/login' in resp.headers.get('Location', '')

    def test_health_is_open_without_auth_and_leaks_no_secret(self, barrier_client):
        client, _ = barrier_client
        resp = client.get('/api/health')
        assert resp.status_code == 200
        body = resp.get_data(as_text=True).lower()
        for leaked in ('password', 'secret', 'token', 'jwt'):
            assert leaked not in body

    def test_real_login_issues_working_session(self, barrier_client):
        client, _ = barrier_client
        login = client.post(
            '/auth',
            data={'user': 'root', 'password': 'rootpw'},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert login.status_code == 200
        set_cookie = login.headers.get('Set-Cookie', '')
        assert 'audiomuse_jwt=' in set_cookie
        assert 'HttpOnly' in set_cookie
        token = set_cookie.split('audiomuse_jwt=', 1)[1].split(';', 1)[0]
        assert token
        ok = client.get('/api/protected')
        assert ok.status_code == 200

    def test_wrong_password_is_rejected(self, barrier_client):
        client, _ = barrier_client
        login = client.post(
            '/auth',
            data={'user': 'root', 'password': 'nope'},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert login.status_code == 401

    def test_admin_token_reaches_admin_path(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _hs256_token('admin'))
        resp = client.get('/api/cron')
        assert resp.status_code == 200

    def test_user_token_forbidden_on_admin_path(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _hs256_token('user'))
        resp = client.get('/api/cron')
        assert resp.status_code == 403

    def test_user_token_allowed_on_normal_path(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _hs256_token('user'))
        resp = client.get('/api/protected')
        assert resp.status_code == 200

    def test_alg_none_token_rejected(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _alg_none_token('admin'))
        resp = client.get('/api/protected')
        assert resp.status_code == 401

    def test_wrong_secret_token_rejected(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _hs256_token('admin', secret='attacker-secret-attacker-secret-32'))
        resp = client.get('/api/protected')
        assert resp.status_code == 401

    def test_expired_token_rejected(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _hs256_token('admin', expired=True))
        resp = client.get('/api/protected')
        assert resp.status_code == 401

    def test_valid_bearer_token_is_admin(self, barrier_client):
        client, _ = barrier_client
        resp = client.get('/api/cron', headers={'Authorization': f'Bearer {_BEARER_TOKEN}'})
        assert resp.status_code == 200

    def test_wrong_bearer_token_rejected(self, barrier_client):
        client, _ = barrier_client
        resp = client.get('/api/protected', headers={'Authorization': 'Bearer not-the-token'})
        assert resp.status_code == 401

    def test_token_for_unknown_user_rejected(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _hs256_token('admin', sub='ghost'))
        resp = client.get('/api/protected')
        assert resp.status_code == 401

    def test_token_role_claim_cannot_escalate_beyond_db_row(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _hs256_token('admin', sub='plainuser'))
        resp = client.get('/api/cron')
        assert resp.status_code == 403


def _row_id(conn, username):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM audiomuse_users WHERE username = %s", (username,))
        return cur.fetchone()[0]


def _age_password_stamp(conn, username, seconds=60):
    aged = datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0, tzinfo=None
    ) - datetime.timedelta(seconds=seconds)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE audiomuse_users SET password_changed_at = %s WHERE username = %s",
            (aged, username),
        )
    conn.commit()


@pytest.mark.integration
class TestSessionRevocation:
    def test_create_user_needs_admin_password(self, barrier_client):
        client, conn = barrier_client
        _set_cookie(client, _hs256_token('admin'))
        resp = client.post('/api/users', json={'username': 'newbie', 'password': 'pw'})
        assert resp.status_code == 400
        assert 'required' in resp.get_json()['error'].lower()
        resp = client.post(
            '/api/users',
            json={'username': 'newbie', 'password': 'pw', 'current_password': 'wrong'},
        )
        assert resp.status_code == 400
        assert 'incorrect' in resp.get_json()['error'].lower()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM audiomuse_users WHERE username = %s", ('newbie',))
            assert cur.fetchone()[0] == 0
        resp = client.post(
            '/api/users',
            json={'username': 'newbie', 'password': 'pw', 'current_password': 'rootpw'},
        )
        assert resp.status_code == 201
        login = client.post(
            '/auth',
            data={'user': 'newbie', 'password': 'pw'},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert login.status_code == 200

    def test_password_change_needs_current_password(self, barrier_client):
        client, conn = barrier_client
        uid = _row_id(conn, 'plainuser')
        _set_cookie(client, _hs256_token('user'))
        resp = client.put(f'/api/users/{uid}/password', json={'password': 'new-pw'})
        assert resp.status_code == 400
        assert 'required' in resp.get_json()['error'].lower()
        resp = client.put(
            f'/api/users/{uid}/password',
            json={'password': 'new-pw', 'current_password': 'wrong'},
        )
        assert resp.status_code == 400
        assert 'incorrect' in resp.get_json()['error'].lower()

    def test_admin_reset_of_other_user_needs_admin_password(self, barrier_client):
        client, conn = barrier_client
        uid = _row_id(conn, 'plainuser')
        _age_password_stamp(conn, 'plainuser')
        victim_token = _hs256_token('user', iat_offset_seconds=-5)
        _set_cookie(client, victim_token)
        assert client.get('/api/protected').status_code == 200
        _set_cookie(client, _hs256_token('admin'))
        resp = client.put(f'/api/users/{uid}/password', json={'password': 'new-pw'})
        assert resp.status_code == 400
        resp = client.put(
            f'/api/users/{uid}/password',
            json={'password': 'new-pw', 'current_password': 'rootpw'},
        )
        assert resp.status_code == 200
        _set_cookie(client, victim_token)
        assert client.get('/api/protected').status_code == 401
        login = client.post(
            '/auth',
            data={'user': 'plainuser', 'password': 'new-pw'},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert login.status_code == 200

    def test_password_change_revokes_old_sessions_and_reissues_cookie(self, barrier_client):
        client, conn = barrier_client
        uid = _row_id(conn, 'plainuser')
        _age_password_stamp(conn, 'plainuser')
        old_token = _hs256_token('user', iat_offset_seconds=-5)
        _set_cookie(client, old_token)
        assert client.get('/api/protected').status_code == 200
        resp = client.put(
            f'/api/users/{uid}/password',
            json={'password': 'brand-new-pw', 'current_password': 'plainpw'},
        )
        assert resp.status_code == 200
        set_cookie = resp.headers.get('Set-Cookie', '')
        assert 'audiomuse_jwt=' in set_cookie
        fresh_token = set_cookie.split('audiomuse_jwt=', 1)[1].split(';', 1)[0]
        _set_cookie(client, old_token)
        assert client.get('/api/protected').status_code == 401
        _set_cookie(client, fresh_token)
        assert client.get('/api/protected').status_code == 200

    def test_delete_needs_admin_password_and_revokes_session(self, barrier_client):
        client, conn = barrier_client
        import app_auth

        ok, err = app_auth.create_additional_user('doomed', 'doomedpw', 'user')
        assert ok, err
        uid = _row_id(conn, 'doomed')
        victim_token = _hs256_token('user', sub='doomed')
        _set_cookie(client, victim_token)
        assert client.get('/api/protected').status_code == 200
        _set_cookie(client, _hs256_token('admin'))
        resp = client.delete(f'/api/users/{uid}')
        assert resp.status_code == 400
        resp = client.delete(f'/api/users/{uid}', json={'current_password': 'wrong'})
        assert resp.status_code == 400
        resp = client.delete(f'/api/users/{uid}', json={'current_password': 'rootpw'})
        assert resp.status_code == 200
        _set_cookie(client, victim_token)
        assert client.get('/api/protected').status_code == 401

    def test_bearer_caller_is_exempt_from_current_password(self, barrier_client):
        client, conn = barrier_client
        import app_auth

        ok, err = app_auth.create_additional_user('m2mtarget', 'm2mpw', 'user')
        assert ok, err
        uid = _row_id(conn, 'm2mtarget')
        headers = {'Authorization': f'Bearer {_BEARER_TOKEN}'}
        resp = client.post(
            '/api/users', json={'username': 'm2mspawn', 'password': 'pw'}, headers=headers
        )
        assert resp.status_code == 201
        resp = client.put(f'/api/users/{uid}/password', json={'password': 'np'}, headers=headers)
        assert resp.status_code == 200
        resp = client.delete(f'/api/users/{uid}', headers=headers)
        assert resp.status_code == 200

    def test_recreated_username_does_not_resurrect_revoked_session(self, barrier_client):
        client, conn = barrier_client
        import app_auth

        ok, err = app_auth.create_additional_user('phoenix', 'first-pw', 'user')
        assert ok, err
        uid = _row_id(conn, 'phoenix')
        _age_password_stamp(conn, 'phoenix')
        old_token = _hs256_token('user', sub='phoenix', iat_offset_seconds=-5)
        _set_cookie(client, old_token)
        assert client.get('/api/protected').status_code == 200
        _set_cookie(client, _hs256_token('admin'))
        resp = client.delete(f'/api/users/{uid}', json={'current_password': 'rootpw'})
        assert resp.status_code == 200
        _set_cookie(client, old_token)
        assert client.get('/api/protected').status_code == 401
        ok, err = app_auth.create_additional_user('phoenix', 'second-pw', 'user')
        assert ok, err
        _set_cookie(client, old_token)
        assert client.get('/api/protected').status_code == 401

    def test_login_page_renders_for_revoked_session(self, barrier_client):
        client, _ = barrier_client
        _set_cookie(client, _hs256_token('user', sub='ghost'))
        resp = client.get('/login')
        assert resp.status_code == 200
