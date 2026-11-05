# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for reverse-proxy (forward-auth) authentication.

Main Features:
* Header identity honoured only from trusted peers; ProxyFix-rewritten
  remote_addr cannot spoof trust.
* Group -> role mapping (admin group, user group, open user role) and 403 deny.
* CSRF guard on unsafe methods for proxy-authenticated requests.
* Navidrome ExtAuth passthrough: header set for the acting user, service
  credentials dropped, explicit creds and worker jobs unaffected, also when a
  selected server binds its own service credentials.
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
    monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', '')
    monkeypatch.setattr(config, 'AUTH_PROXY_SECRET_HEADER', 'X-Proxy-Secret')
    return config


def _status(resp):
    """json_error returns (response, status); plain responses carry status_code."""
    if isinstance(resp, tuple):
        return resp[1]
    return resp.status_code


def _req(app, path='/api/x', method='GET', peer='10.0.0.5', headers=None, environ=None):
    env = {'REMOTE_ADDR': peer}
    env.update(environ or {})
    return app.test_request_context(path, method=method, headers=headers or {}, environ_base=env)


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

    def test_empty_trusted_sources_fails_closed(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_TRUSTED_SOURCES', '')
        with _req(app, headers={'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins'}):
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
    SECRET = 's3cr3t-' + 'x' * 40
    ADMIN = {'Remote-User': 'chris', 'Remote-Groups': 'audiomuse-admins'}

    def test_secret_and_ip_both_required(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', self.SECRET)
        with _req(app, headers={**self.ADMIN, 'X-Proxy-Secret': self.SECRET}):
            assert app_auth.check_auth_needed('s') is None
        with _req(app, headers=self.ADMIN):
            assert _status(app_auth.check_auth_needed('s')) == 401
        with _req(app, peer='10.9.9.9', headers={**self.ADMIN, 'X-Proxy-Secret': self.SECRET}):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_wrong_secret_rejected(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', self.SECRET)
        with _req(app, headers={**self.ADMIN, 'X-Proxy-Secret': self.SECRET + 'x'}):
            assert _status(app_auth.check_auth_needed('s')) == 401

    def test_secret_only_mode(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_TRUSTED_SOURCES', '')
        monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', self.SECRET)
        with _req(app, peer='172.18.0.77', headers={**self.ADMIN, 'X-Proxy-Secret': self.SECRET}):
            assert app_auth.check_auth_needed('s') is None
            assert g.auth_role == 'admin'

    def test_custom_secret_header(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_SHARED_SECRET', self.SECRET)
        monkeypatch.setattr(config, 'AUTH_PROXY_SECRET_HEADER', 'X-AudioMuse-Proxy')
        with _req(app, headers={**self.ADMIN, 'X-AudioMuse-Proxy': self.SECRET}):
            assert app_auth.check_auth_needed('s') is None


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
    def test_redirects_to_idp_logout(self, app, proxy_cfg, monkeypatch):
        monkeypatch.setattr(config, 'AUTH_PROXY_LOGOUT_URL', 'https://auth.example/outpost.goauthentik.io/sign_out')
        with app.test_request_context('/logout', method='POST'):
            resp = app_auth.logout_endpoint()
            assert resp.status_code == 302
            assert resp.location.startswith('https://auth.example/')


class TestNavidromePassthrough:
    @pytest.fixture
    def nd(self, monkeypatch):
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_ENABLED', True)
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

    def test_disabled_flag_uses_service_account(self, nd, monkeypatch):
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_ENABLED', False)
        token = ms_context.set_acting_user('chris')
        try:
            navidrome._navidrome_request('getPlaylists')
        finally:
            ms_context.reset_acting_user(token)
        assert nd.call_args.kwargs['params']['u'] == 'svc'

    def test_bind_hook_only_for_proxy_requests(self, app, monkeypatch):
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_ENABLED', True)
        with app.test_request_context('/'):
            g.auth_method, g.auth_user = 'session', 'local'
            app_proxy_auth.bind_acting_user()
            assert ms_context.acting_user() is None
            g.auth_method, g.auth_user = 'proxy', 'chris'
            app_proxy_auth.bind_acting_user()
            assert ms_context.acting_user() == 'chris'
            app_proxy_auth.reset_acting_user()
            assert ms_context.acting_user() is None


class TestBoundServerAndVisibility:
    @pytest.fixture
    def nd(self, monkeypatch):
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_ENABLED', True)
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
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_ENABLED', True)
        with app.test_request_context('/api/create_playlist', method='POST', **request_kwargs):
            g.auth_method, g.auth_user = 'proxy', 'chris'
            app_proxy_auth.bind_acting_user()
            try:
                return ms_context.playlist_public()
            finally:
                app_proxy_auth.reset_acting_user()

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
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_ENABLED', True)
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
        monkeypatch.setattr(config, 'NAVIDROME_EXTAUTH_ENABLED', True)
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
