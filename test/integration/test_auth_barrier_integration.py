"""Real-Postgres integration test that automates the manual auth pen test.

Mounts the production ``app_auth`` request barrier (the ``before_request``
guard that protects every route) on a Flask app backed by a live
``audiomuse_users`` table, then drives it through Flask's test client to prove
the security invariants a black-box pen test checks by hand:

  * an unauthenticated API call is refused with 401, a page request redirects
    to /login, and /api/health stays open without auth;
  * a real argon2 login through /auth issues a usable HttpOnly session cookie;
  * a tampered token (alg=none, wrong secret, or expired) is always rejected;
  * a non-admin session is forbidden from an admin-only path while an admin
    session and a valid Bearer API token are allowed.

These exercise the real barrier, the real PyJWT round-trip and the real argon2
verify against Postgres, so a regression that opens a hole - a skipped auth
check, an accepted unsigned token, a broken admin gate - fails the build. A
mocked client could not prove the signature pinning or the argon2 round-trip.

Redis is not required: the barrier only reads config and the users table.

Database selection mirrors test_auth_users_integration.py:
  * AUDIOMUSE_TEST_DATABASE_URL - a throwaway DB the test fully owns, or
  * an ephemeral instance via the optional ``pgserver`` package, or
  * the module is skipped.

Run locally:
    pip install pgserver
    pytest test/integration/test_auth_barrier_integration.py -m integration -s -v --tb=short
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
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
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


def _hs256_token(role, secret=_TEST_SECRET, expired=False, sub='tester'):
    now = datetime.datetime.now(datetime.timezone.utc)
    exp = now - datetime.timedelta(hours=1) if expired else now + datetime.timedelta(hours=1)
    payload = {'sub': sub, 'role': role, 'iat': now, 'exp': exp}
    return pyjwt.encode(payload, secret, algorithm='HS256')


def _alg_none_token(role, sub='tester'):
    def _b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b'=').decode()
    header = _b64({'alg': 'none', 'typ': 'JWT'})
    payload = _b64({'sub': sub, 'role': role, 'exp': 9999999999})
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
