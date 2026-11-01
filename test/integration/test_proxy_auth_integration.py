# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Reverse-proxy (forward-auth) login over a real Flask app and Postgres.

Drives the full before-request barrier registered by ``app_auth.init_app``
with proxy authentication enabled, next to a real users table, so the proxy
path, the local login fallback and the bearer token are checked together.

Main Features:
* Proxy admin reaches admin paths; proxy user is limited to normal paths;
  a user in neither group gets 403.
* Identity headers from an untrusted peer or with a wrong/missing shared
  secret are ignored (401), and ProxyFix cannot spoof the trusted peer.
* Cross-site unsafe requests of proxy users are rejected (403).
* Local login and the API bearer token keep working with proxy auth on.
* The request hook binds the proxy user for the Navidrome ExtAuth header and
  releases it after the request; playlists created for that user are private
  unless the request asks to share them, service-account playlists stay public.
"""

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


_TEST_SECRET = 'integration-test-secret-do-not-use-in-prod'
_BEARER_TOKEN = 'integration-bearer-token'
_PROXY_IP = '10.20.30.40'
_PROXY_SECRET = 'proxy-shared-secret-' + 'x' * 40

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
        psycopg2.connect(dsn).close()
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


@pytest.fixture
def proxy_client(pg_dsn, monkeypatch):
    from flask import Flask, g, jsonify

    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS audiomuse_users CASCADE")
        cur.execute(_USERS_DDL)
    conn.commit()

    import config
    import app_auth
    from tasks.mediaserver import context as ms_context
    from tasks.mediaserver import navidrome
    from tasks.setup_manager import SetupManager

    monkeypatch.setattr(app_auth, '_get_db', lambda: conn)
    monkeypatch.setattr(config, 'AUTH_ENABLED', True)
    monkeypatch.setattr(config, 'API_TOKEN', _BEARER_TOKEN)
    monkeypatch.setattr(config, 'MEDIASERVER_TYPE', 'navidrome')
    for field in config.MEDIASERVER_FIELDS_BY_TYPE['navidrome']:
        monkeypatch.setattr(config, field, 'configured', raising=False)

    monkeypatch.setattr(config, 'AUTH_PROXY_ENABLED', True)
    monkeypatch.setattr(config, 'AUTH_PROXY_TRUSTED_SOURCES', f'{_PROXY_IP}/32')
    monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', _PROXY_SECRET)
    monkeypatch.setattr(config, 'AUTH_PROXY_SECRET_HEADER', 'X-Proxy-Secret')
    monkeypatch.setattr(config, 'AUTH_PROXY_USER_HEADER', 'X-Authentik-Username')
    monkeypatch.setattr(config, 'AUTH_PROXY_GROUPS_HEADER', 'X-Authentik-Groups')
    monkeypatch.setattr(config, 'AUTH_PROXY_ADMIN_GROUP', 'audiomuse-admins')
    monkeypatch.setattr(config, 'AUTH_PROXY_USER_GROUP', 'audiomuse-users')
    monkeypatch.setattr(config, 'AUTH_PROXY_LOGOUT_URL', '')
    monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_ENABLED', True)
    monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_USER_HEADER', 'X-Authentik-Username')

    ok, err = app_auth.create_additional_user('root', 'rootpw', 'admin')
    assert ok, err

    app = Flask(__name__, template_folder=os.path.join(_REPO_ROOT, 'templates'))
    app_auth.init_app(app, SetupManager(), lambda: _TEST_SECRET)

    @app.route('/api/protected', methods=['GET', 'POST'])
    def _protected():
        return jsonify({
            "user": g.auth_user,
            "role": g.auth_role,
            "method": g.auth_method,
        })

    @app.route('/api/cron', methods=['GET'])
    def _admin_only():
        return jsonify({"ok": True})

    @app.route('/api/navidrome_headers', methods=['GET'])
    def _navidrome_headers():
        return jsonify({"headers": navidrome._extauth_headers()})

    @app.route('/api/create_playlist', methods=['POST'])
    def _create_playlist():
        return jsonify({"public": navidrome._playlist_public_param()})

    client = app.test_client()
    yield client, ms_context
    conn.close()


def _proxy_get(client, path, user, groups, peer=_PROXY_IP, secret=_PROXY_SECRET, **headers):
    sent = {'X-Authentik-Username': user, 'X-Authentik-Groups': groups}
    if secret is not None:
        sent['X-Proxy-Secret'] = secret
    sent.update(headers)
    return client.get(path, headers=sent, environ_base={'REMOTE_ADDR': peer})


@pytest.mark.integration
class TestProxyRoles:
    def test_admin_group_reaches_admin_path(self, proxy_client):
        client, _ = proxy_client
        resp = _proxy_get(client, '/api/cron', 'chris', 'family|audiomuse-admins')
        assert resp.status_code == 200

    def test_identity_comes_from_the_proxy(self, proxy_client):
        client, _ = proxy_client
        resp = _proxy_get(client, '/api/protected', 'chris', 'audiomuse-admins')
        assert resp.status_code == 200
        assert resp.get_json() == {"user": "chris", "role": "admin", "method": "proxy"}

    def test_user_group_is_limited_to_normal_paths(self, proxy_client):
        client, _ = proxy_client
        assert _proxy_get(client, '/api/protected', 'anna', 'audiomuse-users').status_code == 200
        assert _proxy_get(client, '/api/cron', 'anna', 'audiomuse-users').status_code == 403

    def test_user_in_neither_group_is_forbidden(self, proxy_client):
        client, _ = proxy_client
        resp = _proxy_get(client, '/api/protected', 'guest', 'other')
        assert resp.status_code == 403

    def test_proxy_user_needs_no_local_account(self, proxy_client):
        client, _ = proxy_client
        resp = _proxy_get(client, '/api/protected', 'nobody-local', 'audiomuse-users')
        assert resp.status_code == 200


@pytest.mark.integration
class TestProxyTrust:
    def test_untrusted_peer_is_ignored(self, proxy_client):
        client, _ = proxy_client
        resp = _proxy_get(client, '/api/cron', 'chris', 'audiomuse-admins', peer='10.20.30.41')
        assert resp.status_code == 401

    def test_missing_secret_is_ignored(self, proxy_client):
        client, _ = proxy_client
        resp = _proxy_get(client, '/api/cron', 'chris', 'audiomuse-admins', secret=None)
        assert resp.status_code == 401

    def test_wrong_secret_is_ignored(self, proxy_client):
        client, _ = proxy_client
        resp = _proxy_get(client, '/api/cron', 'chris', 'audiomuse-admins', secret='wrong')
        assert resp.status_code == 401

    def test_forwarded_for_cannot_spoof_the_peer(self, proxy_client):
        client, _ = proxy_client
        resp = _proxy_get(
            client, '/api/cron', 'chris', 'audiomuse-admins',
            peer='192.0.2.10', **{'X-Forwarded-For': _PROXY_IP},
        )
        assert resp.status_code == 401


@pytest.mark.integration
class TestProxyCsrf:
    def _post(self, client, **headers):
        sent = {
            'X-Authentik-Username': 'chris',
            'X-Authentik-Groups': 'audiomuse-admins',
            'X-Proxy-Secret': _PROXY_SECRET,
        }
        sent.update(headers)
        return client.post('/api/protected', json={}, headers=sent, environ_base={'REMOTE_ADDR': _PROXY_IP})

    def test_cross_site_post_is_rejected(self, proxy_client):
        client, _ = proxy_client
        resp = self._post(client, Origin='https://evil.example', **{'Sec-Fetch-Site': 'cross-site'})
        assert resp.status_code == 403

    def test_same_origin_post_is_allowed(self, proxy_client):
        client, _ = proxy_client
        resp = self._post(client, Origin='http://localhost', **{'Sec-Fetch-Site': 'same-origin'})
        assert resp.status_code == 200


@pytest.mark.integration
class TestFallbacks:
    def test_local_login_still_works(self, proxy_client):
        client, _ = proxy_client
        login = client.post(
            '/auth',
            data={'user': 'root', 'password': 'rootpw'},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert login.status_code == 200
        resp = client.get('/api/protected')
        assert resp.status_code == 200
        assert resp.get_json()["method"] == "session"

    def test_bearer_token_still_works(self, proxy_client):
        client, _ = proxy_client
        resp = client.get('/api/cron', headers={'Authorization': f'Bearer {_BEARER_TOKEN}'})
        assert resp.status_code == 200


@pytest.mark.integration
class TestNavidromePassthrough:
    def test_proxy_request_binds_the_navidrome_user(self, proxy_client):
        client, ms_context = proxy_client
        resp = _proxy_get(client, '/api/navidrome_headers', 'chris', 'audiomuse-users')
        assert resp.status_code == 200
        assert resp.get_json() == {"headers": {"X-Authentik-Username": "chris"}}
        assert ms_context.acting_user() is None

    def test_local_session_does_not_bind_a_navidrome_user(self, proxy_client):
        client, _ = proxy_client
        client.post(
            '/auth',
            data={'user': 'root', 'password': 'rootpw'},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        resp = client.get('/api/navidrome_headers')
        assert resp.status_code == 200
        assert resp.get_json() == {"headers": {}}


@pytest.mark.integration
class TestPlaylistVisibility:
    def _post(self, client, query=''):
        return client.post(
            '/api/create_playlist' + query,
            json={'playlist_name': 'Mix', 'track_ids': ['1']},
            headers={
                'X-Authentik-Username': 'chris',
                'X-Authentik-Groups': 'audiomuse-users',
                'X-Proxy-Secret': _PROXY_SECRET,
                'Origin': 'http://localhost',
                'Sec-Fetch-Site': 'same-origin',
            },
            environ_base={'REMOTE_ADDR': _PROXY_IP},
        )

    def test_proxy_user_playlist_is_private_by_default(self, proxy_client):
        client, _ = proxy_client
        resp = self._post(client)
        assert resp.status_code == 200
        assert resp.get_json() == {"public": "false"}

    def test_proxy_user_can_share_the_playlist(self, proxy_client):
        client, _ = proxy_client
        resp = self._post(client, '?playlist_public=true')
        assert resp.status_code == 200
        assert resp.get_json() == {"public": "true"}

    def test_local_session_playlist_stays_public(self, proxy_client):
        client, _ = proxy_client
        client.post(
            '/auth',
            data={'user': 'root', 'password': 'rootpw'},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        resp = client.post(
            '/api/create_playlist',
            json={'playlist_name': 'Mix', 'track_ids': ['1']},
            headers={'Origin': 'http://localhost'},
        )
        assert resp.status_code == 200
        assert resp.get_json() == {"public": "true"}
