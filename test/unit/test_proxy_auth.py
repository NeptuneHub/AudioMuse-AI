# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for reverse-proxy (forward-auth) authentication.

Main Features:
* Header identity honoured only with the mandatory shared secret (and from
  trusted peers when configured); ProxyFix-rewritten remote_addr cannot spoof
  trust; without a secret proxy auth stays off.
* Group -> role mapping (admin group, user group, open user role) and 403 deny.
* CSRF guard on unsafe methods for proxy-authenticated requests.
* Playlist owner: only inside ``playlist_owner_context`` with
  AUTH_PROXY_PLAYLIST_OWNER=user does Navidrome run as the proxy user (header
  set, service credentials dropped); explicit creds (user or API key), local
  sessions and every call outside the playlist write keep the service account.
* A local account with the proxy user's name is a separate account; logout
  sends only proxy sessions to the IdP sign-out.
* Playlist visibility: service-account playlists stay public, proxy-user
  playlists are private unless the request asks to share them; a rejected
  ExtAuth call is reported as an auth error.
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Blueprint, Flask, g

import app_auth
import app_proxy_auth
import config
from tasks.mediaserver import context as ms_context
from tasks.mediaserver import navidrome


@pytest.fixture
def app():
    app = Flask(__name__)
    app.add_url_rule('/login', 'login_page', lambda: 'login')
    dash = Blueprint('dashboard_bp', __name__)
    dash.add_url_rule('/dashboard', 'dashboard_page', lambda: 'dash')
    app.register_blueprint(dash)
    return app


SECRET = 'proxy-shared-secret-' + 'x' * 40


@pytest.fixture
def proxy_cfg(monkeypatch):
    monkeypatch.setattr(config, 'AUTH_ENABLED', True)
    monkeypatch.setattr(config, 'API_TOKEN', '')
    monkeypatch.setattr(config, 'AUTH_PROXY_ENABLED', True)
    monkeypatch.setattr(config, 'AUTH_PROXY_TRUSTED_SOURCES', '10.0.0.5, 192.168.10.0/24')
    monkeypatch.setattr(config, 'AUTH_PROXY_USER_HEADER', 'Remote-User')
    monkeypatch.setattr(config, 'AUTH_PROXY_GROUPS_HEADER', 'Remote-Groups')
    monkeypatch.setattr(config, 'AUTH_PROXY_ADMIN_GROUP', 'audiomuse-admins')
    monkeypatch.setattr(config, 'AUTH_PROXY_USER_GROUP', 'audiomuse-users')
    monkeypatch.setattr(config, 'AUTH_PROXY_LOGOUT_URL', '')
    monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', SECRET)
    monkeypatch.setattr(config, 'AUTH_PROXY_SECRET_HEADER', 'X-Proxy-Secret')
    monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'service')
    return config


def _status(resp):
    """json_error returns (response, status); plain responses carry status_code."""
    if isinstance(resp, tuple):
        return resp[1]
    return resp.status_code


def _req(app, path='/api/x', method='GET', peer='10.0.0.5', headers=None, environ=None, secret=SECRET):
    """A request from the proxy; identity requests carry the shared secret
    unless ``secret`` is None."""
    env = {'REMOTE_ADDR': peer}
    env.update(environ or {})
    sent = dict(headers or {})
    if secret is not None and 'Remote-User' in sent:
        sent.setdefault('X-Proxy-Secret', secret)
    return app.test_request_context(path, method=method, headers=sent, environ_base=env)


class TestIdentity:
    def test_admin_group_gives_admin(self, app, proxy_cfg):
        with _req(app, headers={'Remote-User': 'chris', 'Remote-Groups': 'family|audiomuse-admins'}):
            assert app_auth.check_auth_needed('s') is None
            assert (g.auth_user, g.auth_role, g.auth_method) == ('chris', 'admin', 'proxy')

    def test_user_group_gives_user_comma_separated(self, app, proxy_cfg):
        with _req(app, headers={'Remote-User': 'anna', 'Remote-Groups': 'x, audiomuse-users'}):
            assert app_auth.check_auth_needed('s') is None
            assert g.auth_role == 'user'

    def test_not_in_any_group_is_forbidden(self, app, proxy_cfg):
        with _req(app, headers={'Remote-User': 'guest', 'Remote-Groups': 'other'}):
            assert _status(app_auth.check_auth_needed('s')) == 403

    def test_empty_user_group_allows_everyone_as_user(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_USER_GROUP', '')
        with _req(app, headers={'Remote-User': 'guest'}):
            assert app_auth.check_auth_needed('s') is None
            assert g.auth_role == 'user'

    def test_untrusted_peer_header_ignored(self, app, proxy_cfg):
        with _req(app, peer='10.0.0.99', headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins'}):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_cidr_trusted(self, app, proxy_cfg):
        with _req(app, peer='192.168.10.42', headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-users'}):
            assert app_auth.check_auth_needed('s') is None

    def test_empty_trusted_sources_still_needs_the_secret(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_TRUSTED_SOURCES', '')
        with _req(app, peer='172.18.0.77', headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins'}):
            assert app_auth.check_auth_needed('s') is None
        with _req(app, peer='172.18.0.77', secret=None,
                  headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins'}):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_proxyfix_cannot_spoof_trust(self, app, proxy_cfg):
        """ProxyFix moved the attacker-controlled X-Forwarded-For into
        REMOTE_ADDR; the real socket peer is in werkzeug.proxy_fix.orig."""
        env = {'werkzeug.proxy_fix.orig': {'REMOTE_ADDR': '172.17.0.9'}}
        with _req(app, peer='10.0.0.5', environ=env,
                  headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins'}):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_disabled_ignores_header(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_ENABLED', False)
        with _req(app, headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins'}):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_bearer_still_works_without_header(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'API_TOKEN', 'tok')
        with _req(app, peer='10.0.0.99', headers={'Authorization': 'Bearer tok'}):
            assert app_auth.check_auth_needed('s') is None
            assert g.auth_method == 'bearer'

    def test_invalid_username_rejected(self, app, proxy_cfg):
        with _req(app, headers={'Remote-User': 'a' * 500, 'Remote-Groups': 'audiomuse-users'}):
            assert _status(app_auth.check_auth_needed('s')) == 403


class TestSharedSecret:
    ADMIN = {'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins'}

    def test_secret_and_ip_both_required(self, app, proxy_cfg):
        with _req(app, headers=self.ADMIN):
            assert app_auth.check_auth_needed('s') is None
        with _req(app, headers=self.ADMIN, secret=None):
            assert _status(app_auth.check_auth_needed('s')) == 401
        with _req(app, peer='10.9.9.9', headers=self.ADMIN):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_wrong_secret_rejected(self, app, proxy_cfg):
        with _req(app, headers=self.ADMIN, secret=SECRET + 'x'):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_ip_only_mode_no_longer_exists(self, app, proxy_cfg, monkeypatch):
        """A bypass route on the same proxy shares its IP; only the secret,
        added on the forward-auth route alone, tells them apart."""
        monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', '')
        with _req(app, headers=self.ADMIN, secret=None):
            assert _status(app_auth.check_auth_needed('s')) == 401
        with _req(app, headers=self.ADMIN, secret=''):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_custom_secret_header(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_SECRET_HEADER', 'X-AudioMuse-Proxy')
        with _req(app, headers={**self.ADMIN, 'X-AudioMuse-Proxy': SECRET}, secret=None):
            assert app_auth.check_auth_needed('s') is None

    def test_startup_warns_without_secret(self, proxy_cfg, monkeypatch, caplog):
        monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', '')
        with caplog.at_level('ERROR', logger='app_proxy_auth'):
            app_proxy_auth.warn_on_misconfiguration()
        assert 'AUTH_PROXY_SHARED_SECRET is empty' in caplog.text

    def test_untrusted_warnings_are_capped(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(app_proxy_auth, '_warned_untrusted', set())
        for i in range(app_proxy_auth._MAX_WARNED_PEERS + 10):
            with _req(app, peer=f'10.1.{i // 250}.{i % 250}', headers=self.ADMIN, secret=None):
                app_auth.check_auth_needed('s')
        assert len(app_proxy_auth._warned_untrusted) == app_proxy_auth._MAX_WARNED_PEERS


class TestEnvOnlySettings:
    PROXY_KEYS = {
        'AUTH_PROXY_ENABLED', 'AUTH_PROXY_SHARED_SECRET', 'AUTH_PROXY_SECRET_HEADER',
        'AUTH_PROXY_TRUSTED_SOURCES', 'AUTH_PROXY_USER_HEADER', 'AUTH_PROXY_GROUPS_HEADER',
        'AUTH_PROXY_ADMIN_GROUP', 'AUTH_PROXY_USER_GROUP', 'AUTH_PROXY_LOGOUT_URL',
        'AUTH_PROXY_PLAYLIST_OWNER', 'NAVIDROME_EXTAUTH_USER_HEADER',
    }

    def test_never_mirrored_into_app_config(self):
        assert self.PROXY_KEYS <= config.SETUP_BOOTSTRAP_EXCLUDED_KEYS

    def test_never_persisted_by_the_setup_manager(self):
        from tasks.setup_manager import SetupManager

        persisted = SetupManager()._get_env_config_values(config)
        assert not (self.PROXY_KEYS & set(persisted))

    def test_hidden_from_the_wizard_and_secret_masked(self):
        import app_setup

        assert self.PROXY_KEYS <= app_setup.HIDDEN_ADVANCED_FIELDS
        assert 'AUTH_PROXY_SHARED_SECRET' in app_setup.SECRET_FIELDS

    def test_playlist_owner_defaults_to_service(self):
        assert config.AUTH_PROXY_PLAYLIST_OWNER in ('service', 'user')


class TestCsrf:
    def test_cross_site_post_rejected(self, app, proxy_cfg):
        hdrs = {'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins',
                'Origin': 'https://evil.example'}
        with _req(app, method='POST', headers=hdrs):
            assert _status(app_auth.check_auth_needed('s')) == 403

    def test_sec_fetch_site_cross_site_rejected(self, app, proxy_cfg):
        hdrs = {'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins',
                'Sec-Fetch-Site': 'cross-site'}
        with _req(app, method='POST', headers=hdrs):
            assert _status(app_auth.check_auth_needed('s')) == 403

    def test_same_origin_post_allowed(self, app, proxy_cfg):
        hdrs = {'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins',
                'Origin': 'http://localhost', 'Sec-Fetch-Site': 'same-origin'}
        with _req(app, method='POST', headers=hdrs):
            assert app_auth.check_auth_needed('s') is None

    def test_get_not_checked(self, app, proxy_cfg):
        hdrs = {'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins',
                'Origin': 'https://evil.example'}
        with _req(app, headers=hdrs):
            assert app_auth.check_auth_needed('s') is None


class TestPasswordConfirmation:
    def test_proxy_admin_exempt(self, app):
        with app.test_request_context('/'):
            g.auth_method, g.auth_role, g.auth_user = 'proxy', 'admin', 'chris'
            assert app_auth._confirm_password_error({}) is None

    def test_proxy_user_not_exempt(self, app):
        with app.test_request_context('/'):
            g.auth_method, g.auth_role, g.auth_user = 'proxy', 'user', 'anna'
            assert app_auth._confirm_password_error({}) == "Current password is required."


class TestLogout:
    IDP = 'https://auth.example/outpost.goauthentik.io/sign_out'

    def test_proxy_session_goes_to_idp_logout(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_LOGOUT_URL', self.IDP)
        with _req(app, path='/logout', method='POST',
                  headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-users'}):
            resp = app_auth.logout_endpoint()
            assert resp.status_code == 302
            assert resp.location == self.IDP

    def test_local_session_goes_to_login_page(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_LOGOUT_URL', self.IDP)
        with _req(app, path='/logout', method='POST', peer='192.0.2.50'):
            resp = app_auth.logout_endpoint()
            assert resp.status_code == 302
            assert resp.location.endswith('/login')

    def test_spoofed_header_without_secret_is_a_local_logout(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_LOGOUT_URL', self.IDP)
        with _req(app, path='/logout', method='POST', secret=None,
                  headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-users'}):
            resp = app_auth.logout_endpoint()
            assert resp.location.endswith('/login')

    def test_ajax_proxy_logout_returns_redirect_target(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_LOGOUT_URL', self.IDP)
        with _req(app, path='/logout', method='POST',
                  headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-users',
                           'X-Requested-With': 'XMLHttpRequest'}):
            resp = app_auth.logout_endpoint()
            assert resp.get_json() == {"status": "logged_out", "redirect": self.IDP}


class TestSameNameLocalAccount:
    LOCAL_ALICE = {'id': 7, 'username': 'alice', 'role': 'user'}

    @pytest.fixture
    def users(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(app_auth, 'list_additional_users', lambda username=None: [self.LOCAL_ALICE])
        monkeypatch.setattr(app_auth, 'get_additional_user_by_id', lambda user_id: dict(self.LOCAL_ALICE))

        def _update(user_id, password):
            calls['updated'] = user_id
            return True, None

        def _delete(user_id):
            calls['deleted'] = user_id
            return 'deleted', None

        monkeypatch.setattr(app_auth, 'update_additional_user_password', _update)
        monkeypatch.setattr(app_auth, 'delete_additional_user_safe', _delete)
        monkeypatch.setattr(app_auth, '_jwt_secret', lambda: 'jwt-secret-for-tests-' + 'x' * 20)
        return calls

    def test_proxy_user_never_sees_the_local_namesake(self, app, users):
        with app.test_request_context('/api/users'):
            g.auth_method, g.auth_role, g.auth_user = 'proxy', 'user', 'alice'
            body = app_auth.list_users_endpoint().get_json()
        assert body['users'] == []
        assert body['current_user'] == 'alice'

    def test_local_user_still_sees_own_row(self, app, users):
        with app.test_request_context('/api/users'):
            g.auth_method, g.auth_role, g.auth_user = 'session', 'user', 'alice'
            body = app_auth.list_users_endpoint().get_json()
        assert body['users'] == [self.LOCAL_ALICE]

    def test_proxy_user_cannot_change_local_namesake_password(self, app, users):
        with app.test_request_context('/api/users/7/password', method='PUT',
                                      json={'password': 'new', 'current_password': 'old'}):
            g.auth_method, g.auth_role, g.auth_user = 'proxy', 'user', 'alice'
            resp = app_auth.update_user_password_endpoint(7)
        assert _status(resp) == 403
        assert 'updated' not in users

    def test_proxy_admin_change_issues_no_local_session(self, app, users):
        with app.test_request_context('/api/users/7/password', method='PUT', json={'password': 'new'}):
            g.auth_method, g.auth_role, g.auth_user = 'proxy', 'admin', 'alice'
            resp = app_auth.update_user_password_endpoint(7)
        assert resp.status_code == 200
        assert users['updated'] == 7
        assert 'audiomuse_jwt' not in resp.headers.get('Set-Cookie', '')

    def test_proxy_admin_may_delete_local_namesake(self, app, users):
        with app.test_request_context('/api/users/7', method='DELETE', json={}):
            g.auth_method, g.auth_role, g.auth_user = 'proxy', 'admin', 'alice'
            resp = app_auth.delete_user_endpoint(7)
        assert resp.status_code == 200
        assert users['deleted'] == 7


class TestNavidromePassthrough:
    @pytest.fixture
    def nd(self, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'user')
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_USER_HEADER', 'Remote-User')
        monkeypatch.setattr(config, 'NAVIDROME_URL', 'http://nd:4533')
        monkeypatch.setattr(config, 'NAVIDROME_USER', 'svc')
        monkeypatch.setattr(config, 'NAVIDROME_PASSWORD', 'svcpw')
        monkeypatch.setattr(config, 'NAVIDROME_API_KEY', '')
        resp = MagicMock()
        resp.json.return_value = {'subsonic-response': {'status': 'ok'}}
        with patch.object(navidrome.requests, 'request', return_value=resp) as req:
            yield req

    def test_acting_user_sends_header_without_service_creds(self, nd):
        token = ms_context.set_acting_user('chris')
        try:
            navidrome._navidrome_request('getPlaylists')
        finally:
            ms_context.reset_acting_user(token)
        kwargs = nd.call_args.kwargs
        assert kwargs['headers'] == {'Remote-User': 'chris'}
        assert 'u' not in kwargs['params']
        assert 'p' not in kwargs['params']

    def test_no_acting_user_uses_service_account(self, nd):
        navidrome._navidrome_request('getPlaylists')
        kwargs = nd.call_args.kwargs
        assert kwargs['headers'] is None
        assert kwargs['params']['u'] == 'svc'

    def test_explicit_creds_win(self, nd):
        token = ms_context.set_acting_user('chris')
        try:
            navidrome._navidrome_request('getPlaylists', user_creds={'user': 'bob', 'password': 'x'})
        finally:
            ms_context.reset_acting_user(token)
        kwargs = nd.call_args.kwargs
        assert kwargs['headers'] is None
        assert kwargs['params']['u'] == 'bob'

    def test_owner_service_uses_service_account(self, nd, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'service')
        token = ms_context.set_acting_user('chris')
        try:
            navidrome._navidrome_request('getPlaylists')
        finally:
            ms_context.reset_acting_user(token)
        assert nd.call_args.kwargs['params']['u'] == 'svc'

    def test_explicit_api_key_wins(self, nd):
        token = ms_context.set_acting_user('chris')
        try:
            navidrome._navidrome_request('getPlaylists', user_creds={'api_key': 'other-key'})
        finally:
            ms_context.reset_acting_user(token)
        assert nd.call_args.kwargs['headers'] is None

    def test_service_api_key_is_not_explicit(self, nd, monkeypatch):
        monkeypatch.setattr(config, 'NAVIDROME_API_KEY', 'svc-key')
        assert navidrome._is_explicit_user({'api_key': 'svc-key'}) is False
        assert navidrome._is_explicit_user({'api_key': 'other'}) is True


class TestPlaylistOwnerContext:
    def _owner_inside(self, app, monkeypatch, owner, method, user='chris'):
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', owner)
        with app.test_request_context('/api/create_playlist', method='POST', json={'playlist_name': 'x'}):
            g.auth_method, g.auth_user = method, user
            assert ms_context.acting_user() is None
            with app_proxy_auth.playlist_owner_context():
                inside = ms_context.acting_user()
            assert ms_context.acting_user() is None
        return inside

    def test_proxy_user_with_owner_user(self, app, monkeypatch):
        assert self._owner_inside(app, monkeypatch, 'user', 'proxy') == 'chris'

    def test_proxy_user_with_owner_service(self, app, monkeypatch):
        assert self._owner_inside(app, monkeypatch, 'service', 'proxy') is None

    def test_local_session_never_acts_as_user(self, app, monkeypatch):
        assert self._owner_inside(app, monkeypatch, 'user', 'session') is None

    def test_bearer_never_acts_as_user(self, app, monkeypatch):
        assert self._owner_inside(app, monkeypatch, 'user', 'bearer', user=None) is None

    def test_outside_a_request_does_nothing(self, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'user')
        with app_proxy_auth.playlist_owner_context():
            assert ms_context.acting_user() is None

    def test_calls_outside_the_playlist_write_keep_the_service_account(self, app, monkeypatch):
        """A proxy request is not bound as a whole any more: connection tests,
        library reads and plugin helpers outside the write use the service
        account even with AUTH_PROXY_PLAYLIST_OWNER=user."""
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'user')
        monkeypatch.setattr(config, 'NAVIDROME_URL', 'http://nd:4533')
        monkeypatch.setattr(config, 'NAVIDROME_USER', 'svc')
        monkeypatch.setattr(config, 'NAVIDROME_PASSWORD', 'svcpw')
        monkeypatch.setattr(config, 'NAVIDROME_API_KEY', '')
        resp = MagicMock()
        resp.json.return_value = {'subsonic-response': {'status': 'ok'}}
        with app.test_request_context('/api/servers/test', method='POST'):
            g.auth_method, g.auth_user = 'proxy', 'chris'
            with patch.object(navidrome.requests, 'request', return_value=resp) as req:
                navidrome._navidrome_request('ping')
        assert req.call_args.kwargs['headers'] is None
        assert req.call_args.kwargs['params']['u'] == 'svc'

    def test_no_request_hooks_bind_the_user(self):
        assert not hasattr(app_proxy_auth, 'bind_acting_user')


class TestBoundServerAndVisibility:
    @pytest.fixture
    def nd(self, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'user')
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_USER_HEADER', 'Remote-User')
        monkeypatch.setattr(config, 'NAVIDROME_URL', 'http://nd:4533')
        monkeypatch.setattr(config, 'NAVIDROME_USER', 'svc')
        monkeypatch.setattr(config, 'NAVIDROME_PASSWORD', 'svcpw')
        monkeypatch.setattr(config, 'NAVIDROME_API_KEY', '')
        resp = MagicMock()
        resp.json.return_value = {'subsonic-response': {'status': 'ok'}}
        with patch.object(navidrome.requests, 'request', return_value=resp) as req:
            yield req

    @staticmethod
    def _bound_server():
        return {
            'server_id': 'nd2', 'server_type': 'navidrome',
            'creds': {'url': 'http://nd2:4533', 'user': 'svc2', 'password': 'pw2'},
        }

    def test_bound_server_service_creds_do_not_disable_passthrough(self, nd):
        """A selected server binds its service creds; they must not count as an
        explicit user choice, or the call silently runs as the service account."""
        token = ms_context.set_acting_user('chris')
        try:
            with ms_context.use_server(self._bound_server()):
                creds = ms_context.active_creds(None)
                navidrome._navidrome_request('getPlaylists', user_creds=creds)
        finally:
            ms_context.reset_acting_user(token)
        kwargs = nd.call_args.kwargs
        assert kwargs['headers'] == {'Remote-User': 'chris'}
        assert 'u' not in kwargs['params']

    def test_other_explicit_user_still_wins_on_bound_server(self, nd):
        token = ms_context.set_acting_user('chris')
        try:
            with ms_context.use_server(self._bound_server()):
                navidrome._navidrome_request('getPlaylists', user_creds={'user': 'bob', 'password': 'x'})
        finally:
            ms_context.reset_acting_user(token)
        assert nd.call_args.kwargs['headers'] is None
        assert nd.call_args.kwargs['params']['u'] == 'bob'

    def test_service_account_playlist_stays_public(self, nd):
        assert navidrome._playlist_public_param() == "true"

    def test_proxy_user_playlist_is_private_by_default(self, nd):
        token = ms_context.set_acting_user('chris')
        try:
            assert navidrome._playlist_public_param() == "false"
        finally:
            ms_context.reset_acting_user(token)

    def test_proxy_user_can_share_playlist(self, nd):
        user_token = ms_context.set_acting_user('chris')
        public_token = ms_context.set_playlist_public(True)
        try:
            assert navidrome._playlist_public_param() == "true"
        finally:
            ms_context.reset_playlist_public(public_token)
            ms_context.reset_acting_user(user_token)

    def test_created_playlist_sent_private_for_proxy_user(self, nd):
        nd.return_value.json.return_value = {
            'subsonic-response': {'status': 'ok', 'playlist': {'id': 'pl1'}}
        }
        token = ms_context.set_acting_user('chris')
        try:
            navidrome._create_playlist_batched('Mix', ['a', 'b'])
        finally:
            ms_context.reset_acting_user(token)
        public_values = [
            c.kwargs['params'].get('public') for c in nd.call_args_list if 'public' in c.kwargs['params']
        ]
        assert public_values and set(public_values) == {"false"}

    def test_rejected_extauth_is_reported_as_auth_error(self, nd):
        nd.return_value.json.return_value = {
            'subsonic-response': {'status': 'failed', 'error': {'code': 10, 'message': "missing parameter: 'u'"}}
        }
        request_token = ms_context.begin_passthrough_request()
        token = ms_context.set_acting_user('chris')
        try:
            data, error = navidrome._navidrome_request_ex('getPlaylists')
        finally:
            ms_context.reset_acting_user(token)
            ms_context.end_passthrough_request(request_token)
        assert data is None
        assert error['kind'] == 'auth'


class TestVisibilityRequest:
    def _bind(self, app, monkeypatch, **request_kwargs):
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'user')
        with app.test_request_context('/api/create_playlist', method='POST', **request_kwargs):
            g.auth_method, g.auth_user = 'proxy', 'chris'
            with app_proxy_auth.playlist_owner_context():
                return ms_context.playlist_public()

    def test_private_by_default(self, app, monkeypatch):
        assert self._bind(app, monkeypatch, json={'playlist_name': 'x'}) is False

    def test_query_flag_shares(self, app, monkeypatch):
        assert self._bind(app, monkeypatch, query_string={'playlist_public': 'true'}, json={'playlist_name': 'x'}) is True

    def test_json_flag_shares(self, app, monkeypatch):
        assert self._bind(app, monkeypatch, json={'playlist_name': 'x', 'public': True}) is True

    def test_truthy_strings_do_not_share(self, app, monkeypatch):
        assert self._bind(app, monkeypatch, json={'playlist_name': 'x', 'public': 'yes'}) is False


class TestPassthroughDenied:
    def test_rejected_header_marks_the_request(self, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'user')
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_USER_HEADER', 'Remote-User')
        monkeypatch.setattr(config, 'NAVIDROME_URL', 'http://nd:4533')
        resp = MagicMock()
        resp.json.return_value = {
            'subsonic-response': {'status': 'failed', 'error': {'code': 10, 'message': "missing parameter: 'u'"}}
        }
        request_token = ms_context.begin_passthrough_request()
        user_token = ms_context.set_acting_user('chris')
        try:
            with patch.object(navidrome.requests, 'request', return_value=resp):
                navidrome._navidrome_request('createPlaylist')
            assert ms_context.passthrough_rejected() is True
        finally:
            ms_context.reset_acting_user(user_token)
            ms_context.end_passthrough_request(request_token)
        assert ms_context.passthrough_rejected() is False

    def test_service_account_failure_does_not_mark(self, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_PLAYLIST_OWNER', 'user')
        monkeypatch.setattr(config, 'NAVIDROME_URL', 'http://nd:4533')
        monkeypatch.setattr(config, 'NAVIDROME_USER', 'svc')
        monkeypatch.setattr(config, 'NAVIDROME_PASSWORD', 'svcpw')
        monkeypatch.setattr(config, 'NAVIDROME_API_KEY', '')
        resp = MagicMock()
        resp.json.return_value = {
            'subsonic-response': {'status': 'failed', 'error': {'code': 10, 'message': 'x'}}
        }
        token = ms_context.begin_passthrough_request()
        try:
            with patch.object(navidrome.requests, 'request', return_value=resp):
                navidrome._navidrome_request('createPlaylist')
            assert ms_context.passthrough_rejected() is False
        finally:
            ms_context.end_passthrough_request(token)

    def test_create_for_server_raises_a_dedicated_error(self, monkeypatch):
        import app_server_context
        from tasks import mediaserver
        from tasks.mediaserver import registry

        bound = MagicMock()

        def _refused(*args, **kwargs):
            ms_context.mark_passthrough_rejected()

        bound.create_instant_playlist.side_effect = _refused
        monkeypatch.setattr(registry, 'translate_ids', lambda ids, server_id: {str(i): i for i in ids})
        monkeypatch.setattr(mediaserver, 'for_server', lambda server_id: bound)
        token = ms_context.begin_passthrough_request()
        try:
            with pytest.raises(app_server_context.PassthroughDeniedError):
                app_server_context.create_instant_playlist_for_server('Mix', ['1'], None)
        finally:
            ms_context.end_passthrough_request(token)

    def test_error_code_is_registered_as_a_media_server_error(self):
        from error import error_dictionary as ed
        from error.error_manager import http_status_for_code

        assert ed.ERR_MEDIASERVER_PASSTHROUGH_DENIED == 1108
        assert 'ExtAuth.TrustedSources' in ed.get_default_message(1108)
        assert http_status_for_code(1108) == 502
