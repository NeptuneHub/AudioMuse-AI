# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the app_auth before-request guard and user hashing.

Drives the JWT-cookie and bearer-token checks, admin-path enforcement, and
the create/verify user helpers with mocked persistence.

Main Features:
* JWT cookie sessions validated against the user row: unknown user,
  missing claims, stale iat after a password change, and DB-role authority.
* Bearer compare_digest checks and admin-path 403 or login redirect.
* current_password confirmation gates on password change and delete
  endpoints, with the bearer exemption and self-change cookie re-issue.
* Argon2 create/verify roundtrip with empty, invalid-role, and duplicate guards.
"""

import datetime
from unittest.mock import MagicMock

import jwt as pyjwt
import pytest
from flask import Flask, Blueprint, g

import app_auth


def _token(secret, sub='alice', role='user', iat_offset_seconds=0, **extra):
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=iat_offset_seconds
    )
    payload = {'sub': sub, 'role': role, 'iat': now, 'exp': now + datetime.timedelta(hours=1)}
    payload.update(extra)
    for key, value in list(payload.items()):
        if value is None:
            del payload[key]
    return pyjwt.encode(payload, secret, algorithm='HS256')


def _session_row(username='alice', role='user', password_changed_at=None):
    return {
        'username': username,
        'role': role,
        'password_changed_at': password_changed_at,
    }


@pytest.fixture
def app():
    app = Flask(__name__)
    app.add_url_rule('/login', 'login_page', lambda: 'login')
    dash = Blueprint('dashboard_bp', __name__)
    dash.add_url_rule('/dashboard', 'dashboard_page', lambda: 'dash')
    app.register_blueprint(dash)
    return app


def _fake_db(fetchone=None):
    cur = MagicMock()
    cur.__enter__ = lambda self: self
    cur.__exit__ = lambda self, *a: None
    cur.fetchone.return_value = fetchone
    db = MagicMock()
    db.cursor.return_value = cur
    return db, cur


class TestCheckAuthNeededJwt:
    def test_auth_disabled_passes_as_admin(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', False)
        with app.test_request_context('/api/foo'):
            result = app_auth.check_auth_needed('secret')
            assert result is None
            assert g.auth_role == 'admin'

    def test_valid_token_sets_role_and_user(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        monkeypatch.setattr(app_auth, 'get_session_user', lambda u: _session_row(username=u))
        secret = 'unit-secret'
        token = _token(secret)
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is None
            assert g.auth_role == 'user'
            assert g.auth_user == 'alice'
            assert g.auth_method == 'session'

    def test_db_role_wins_over_token_role_claim(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        monkeypatch.setattr(
            app_auth, 'get_session_user', lambda u: _session_row(username=u, role='user')
        )
        secret = 'unit-secret'
        token = _token(secret, role='admin')
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is None
            assert g.auth_role == 'user'

    def test_token_without_sub_rejected(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        monkeypatch.setattr(app_auth, 'get_session_user', lambda u: _session_row(username=u))
        secret = 'unit-secret'
        token = _token(secret, sub=None)
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is not None
            assert result[1] == 401

    def test_token_without_iat_rejected(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        monkeypatch.setattr(app_auth, 'get_session_user', lambda u: _session_row(username=u))
        secret = 'unit-secret'
        token = pyjwt.encode({'sub': 'alice', 'role': 'user'}, secret, algorithm='HS256')
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is not None
            assert result[1] == 401

    def test_token_for_unknown_user_rejected(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        monkeypatch.setattr(app_auth, 'get_session_user', lambda u: None)
        secret = 'unit-secret'
        token = _token(secret)
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is not None
            assert result[1] == 401

    def test_token_issued_before_password_change_rejected(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        changed_at = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=60)
        ).replace(microsecond=0, tzinfo=None)
        monkeypatch.setattr(
            app_auth,
            'get_session_user',
            lambda u: _session_row(username=u, password_changed_at=changed_at),
        )
        secret = 'unit-secret'
        stale = _token(secret, iat_offset_seconds=-120)
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={stale}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is not None
            assert result[1] == 401
        fresh = _token(secret)
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={fresh}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is None
            assert g.auth_user == 'alice'

    def test_db_error_fails_closed(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')

        def boom(_):
            raise RuntimeError("db down")

        monkeypatch.setattr(app_auth, 'get_session_user', boom)
        secret = 'unit-secret'
        token = _token(secret)
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is not None
            assert result[1] == 401

    def test_absurd_iat_fails_closed_not_500(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        changed_at = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=60)
        ).replace(microsecond=0, tzinfo=None)
        monkeypatch.setattr(
            app_auth,
            'get_session_user',
            lambda u: _session_row(username=u, password_changed_at=changed_at),
        )
        secret = 'unit-secret'
        now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        token = pyjwt.encode(
            {'sub': 'alice', 'role': 'user', 'iat': -(10 ** 25), 'exp': now + 3600},
            secret,
            algorithm='HS256',
        )
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is not None
            assert result[1] == 401

    def test_iat_equal_to_changed_at_is_accepted_one_second_earlier_rejected(
        self, app, monkeypatch
    ):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        changed_at = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=60)
        ).replace(microsecond=0, tzinfo=None)
        monkeypatch.setattr(
            app_auth,
            'get_session_user',
            lambda u: _session_row(username=u, password_changed_at=changed_at),
        )
        secret = 'unit-secret'
        epoch = int(changed_at.replace(tzinfo=datetime.timezone.utc).timestamp())
        same_second = pyjwt.encode(
            {'sub': 'alice', 'role': 'user', 'iat': epoch, 'exp': epoch + 3600},
            secret,
            algorithm='HS256',
        )
        with app.test_request_context(
            '/api/foo', headers={'Cookie': f'audiomuse_jwt={same_second}'}
        ):
            result = app_auth.check_auth_needed(secret)
            assert result is None
        one_before = pyjwt.encode(
            {'sub': 'alice', 'role': 'user', 'iat': epoch - 1, 'exp': epoch + 3600},
            secret,
            algorithm='HS256',
        )
        with app.test_request_context(
            '/api/foo', headers={'Cookie': f'audiomuse_jwt={one_before}'}
        ):
            result = app_auth.check_auth_needed(secret)
            assert result is not None
            assert result[1] == 401

    def test_empty_secret_rejects_present_cookie(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        token = pyjwt.encode({'sub': 'x', 'role': 'admin'}, 'whatever', algorithm='HS256')
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed('')
            assert result is not None
            assert result[1] == 401

    def test_tampered_token_is_unauthorized(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        secret = 'unit-secret'
        token = pyjwt.encode({'sub': 'alice', 'role': 'user'}, secret, algorithm='HS256') + 'tamper'
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is not None
            assert result[1] == 401


class TestConfirmPasswordErrorUnit:
    def test_verifies_the_acting_user(self, app, monkeypatch):
        calls = []

        def fake_verify(username, password):
            calls.append((username, password))
            return 'admin'

        monkeypatch.setattr(app_auth, 'verify_additional_user', fake_verify)
        with app.test_request_context('/api/users/5/password'):
            g.auth_user, g.auth_method = 'root', 'session'
            assert app_auth._confirm_password_error({'current_password': 'pw'}) is None
        assert calls == [('root', 'pw')]

    def test_wrong_password_reports_incorrect(self, app, monkeypatch):
        monkeypatch.setattr(app_auth, 'verify_additional_user', lambda u, p: None)
        with app.test_request_context('/api/users/5/password'):
            g.auth_user, g.auth_method = 'root', 'session'
            err = app_auth._confirm_password_error({'current_password': 'bad'})
        assert 'incorrect' in err.lower()

    @pytest.mark.parametrize('data', [None, {}, {'current_password': ''}, {'current_password': 5}])
    def test_missing_password_is_required_without_verify_call(self, app, monkeypatch, data):
        def fake_verify(username, password):
            raise AssertionError("must not verify")

        monkeypatch.setattr(app_auth, 'verify_additional_user', fake_verify)
        with app.test_request_context('/api/users/5/password'):
            g.auth_user, g.auth_method = 'root', 'session'
            err = app_auth._confirm_password_error(data)
        assert 'required' in err.lower()

    def test_session_without_username_is_required(self, app, monkeypatch):
        monkeypatch.setattr(app_auth, 'verify_additional_user', lambda u, p: 'admin')
        with app.test_request_context('/api/users/5/password'):
            g.auth_user, g.auth_method = None, 'session'
            err = app_auth._confirm_password_error({'current_password': 'pw'})
        assert 'required' in err.lower()

    def test_bearer_is_exempt_without_verify_call(self, app, monkeypatch):
        def fake_verify(username, password):
            raise AssertionError("must not verify")

        monkeypatch.setattr(app_auth, 'verify_additional_user', fake_verify)
        with app.test_request_context('/api/users/5'):
            g.auth_user, g.auth_method = None, 'bearer'
            assert app_auth._confirm_password_error({}) is None


class TestCheckAuthNeededBearer:
    def test_bearer_uses_compare_digest(self, app, monkeypatch):
        import config
        import secrets as _secrets

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', 'tok-123')
        calls = []
        real = _secrets.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(app_auth.secrets, 'compare_digest', spy)
        with app.test_request_context('/api/foo', headers={'Authorization': 'Bearer tok-123'}):
            result = app_auth.check_auth_needed('s')
            assert result is None
            assert g.auth_role == 'admin'
            assert g.auth_method == 'bearer'
        assert (b'tok-123', b'tok-123') in calls

    def test_bearer_wrong_token_rejected(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', 'tok-123')
        with app.test_request_context('/api/foo', headers={'Authorization': 'Bearer nope'}):
            result = app_auth.check_auth_needed('s')
            assert result is not None
            assert result[1] == 401

    def test_bearer_ignored_when_api_token_unset(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', '')
        with app.test_request_context('/api/foo', headers={'Authorization': 'Bearer anything'}):
            result = app_auth.check_auth_needed('s')
            assert result is not None
            assert result[1] == 401


class TestAdminPathEnforcement:
    def test_non_admin_gets_403_on_admin_api(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        with app.test_request_context('/api/analysis/start'):
            g.auth_role = 'user'
            result = app_auth.check_admin_needed()
        assert result is not None
        assert result[1] == 403

    def test_admin_passes_admin_api(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        with app.test_request_context('/api/analysis/start'):
            g.auth_role = 'admin'
            result = app_auth.check_admin_needed()
        assert result is None

    def test_non_admin_page_redirects(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        with app.test_request_context('/analysis'):
            g.auth_role = 'user'
            result = app_auth.check_admin_needed()
        assert result is not None
        assert result.status_code == 302

    def test_users_path_not_admin_gated(self, app, monkeypatch):
        import config

        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        with app.test_request_context('/api/users'):
            g.auth_role = 'user'
            result = app_auth.check_admin_needed()
        assert result is None

    @pytest.mark.parametrize(
        'path,expected',
        [
            ('/setup', True),
            ('/api/setup', True),
            ('/api/migration/session/start', True),
            ('/api/analysis', True),
            ('/api/clustering', True),
            ('/api/cron', True),
            ('/api/backup', True),
            ('/api/cancel/abc-123', True),
            ('/api/cancel_all/main_analysis', True),
            ('/api/rebuild_map_cache', True),
            ('/api/clap/cache/refresh', True),
            ('/api/lyrics/cache/refresh', True),
            ('/api/sem_grove/cache/refresh', True),
            ('/api/users', False),
            ('/api/anchors', False),
            ('/api/anchors/5', False),
            ('/api/clap/search', False),
            ('/api/clap/warmup', False),
            ('/chat/api/create_playlist', False),
            ('/dashboard', False),
            ('/login', False),
            ('/', False),
        ],
    )
    def test_is_admin_path_matrix(self, path, expected):
        assert app_auth.is_admin_path(path) is expected


class TestSetupBarrierAllowsSetupApiSubtree:
    @pytest.mark.parametrize(
        'path,allowed',
        [
            ('/api/setup', True),
            ('/api/setup/plex/pin', True),
            ('/api/setup/plex/pin/12345', True),
            ('/api/setup/providers/libraries', True),
            ('/api/setup/lyrics-api/analyze', True),
            # The wizard configures its media servers through the registry API
            # and polls the alignment sweep that adding one enqueues.
            ('/api/servers', True),
            ('/api/servers/test', True),
            ('/api/servers/libraries', True),
            ('/api/servers/align', True),
            ('/api/servers/abc123/default', True),
            ('/api/status/task-1', True),
            ('/api/cancel/task-1', True),
            ('/api/analysis/start', False),
            ('/api/cancel_all/analysis', False),
            ('/api/playlists', False),
        ],
    )
    def test_setup_api_subtree_reachable_during_first_run(self, app, monkeypatch, path, allowed):
        monkeypatch.setattr(app_auth, 'check_setup_needed', lambda: True)
        with app.test_request_context(path):
            result = app_auth.auth_setup_barrier()
        if allowed:
            assert result is None
        else:
            assert result is not None
            assert result[1] == 403

    def test_setup_needed_flag_is_exposed_to_handlers(self, app, monkeypatch):
        from flask import g

        monkeypatch.setattr(app_auth, 'check_setup_needed', lambda: True)
        with app.test_request_context('/api/servers'):
            assert app_auth.auth_setup_barrier() is None
            assert g.setup_needed is True

    def test_no_setup_flag_once_setup_is_complete(self, app, monkeypatch):
        from flask import g

        monkeypatch.setattr(app_auth, 'check_setup_needed', lambda: False)
        monkeypatch.setattr(app_auth, 'check_auth_needed', lambda secret: None)
        monkeypatch.setattr(app_auth, 'check_admin_needed', lambda: None)
        with app.test_request_context('/api/servers'):
            assert app_auth.auth_setup_barrier() is None
            assert getattr(g, 'setup_needed', False) is False


class TestPasswordHashingUnit:
    def test_create_user_stores_argon2_hash(self, monkeypatch):
        db, cur = _fake_db(fetchone=(1,))
        monkeypatch.setattr(app_auth, '_get_db', lambda: db)
        ok, err = app_auth.create_additional_user('alice', 'pw-secret', 'user')
        assert ok is True
        assert err is None
        params = cur.execute.call_args[0][1]
        assert params[0] == 'alice'
        assert params[1].startswith('$argon2')
        assert params[2] == 'user'
        assert params[1] != 'pw-secret'

    def test_create_user_rejects_empty_username(self):
        ok, err = app_auth.create_additional_user('', 'pw', 'user')
        assert ok is False
        assert 'Username' in err

    def test_create_user_rejects_empty_password(self):
        ok, err = app_auth.create_additional_user('bob', '', 'user')
        assert ok is False
        assert 'Password' in err

    def test_create_user_rejects_invalid_role(self):
        ok, err = app_auth.create_additional_user('bob', 'pw', 'superuser')
        assert ok is False
        assert 'role' in err.lower()

    def test_duplicate_username_returns_error(self, monkeypatch):
        db, cur = _fake_db(fetchone=None)
        monkeypatch.setattr(app_auth, '_get_db', lambda: db)
        ok, err = app_auth.create_additional_user('alice', 'pw', 'user')
        assert ok is False
        assert 'exists' in err.lower()

    def test_verify_accepts_correct_password(self, monkeypatch):
        from argon2 import PasswordHasher

        stored = PasswordHasher().hash('correct-horse')
        db, cur = _fake_db(fetchone=(stored, 'admin'))
        monkeypatch.setattr(app_auth, '_get_db', lambda: db)
        assert app_auth.verify_additional_user('alice', 'correct-horse') == 'admin'

    def test_verify_rejects_wrong_password(self, monkeypatch):
        from argon2 import PasswordHasher

        stored = PasswordHasher().hash('correct-horse')
        db, cur = _fake_db(fetchone=(stored, 'admin'))
        monkeypatch.setattr(app_auth, '_get_db', lambda: db)
        assert app_auth.verify_additional_user('alice', 'wrong') is None

    def test_verify_unknown_user_returns_none(self, monkeypatch):
        db, cur = _fake_db(fetchone=None)
        monkeypatch.setattr(app_auth, '_get_db', lambda: db)
        assert app_auth.verify_additional_user('ghost', 'pw') is None

    def test_update_password_stamps_changed_at(self, monkeypatch):
        db, cur = _fake_db()
        cur.rowcount = 1
        monkeypatch.setattr(app_auth, '_get_db', lambda: db)
        ok, err = app_auth.update_additional_user_password(7, 'new-pw')
        assert ok is True
        assert err is None
        sql, params = cur.execute.call_args[0]
        assert 'password_changed_at' in sql
        assert params[0].startswith('$argon2')
        assert isinstance(params[1], datetime.datetime)
        assert params[1].microsecond == 0
        assert params[2] == 7


def _install_endpoint_mocks(monkeypatch, target_role='user', verify_result='user',
                            update_result=(True, None), delete_result=("deleted", None)):
    import config

    verify_calls = []
    monkeypatch.setattr(config, 'AUTH_ENABLED', True)
    monkeypatch.setattr(
        app_auth,
        'get_additional_user_by_id',
        lambda user_id: {'id': int(user_id), 'username': 'bob', 'role': target_role},
    )

    def fake_verify(username, password):
        verify_calls.append((username, password))
        return verify_result if password == 'right-pw' else None

    monkeypatch.setattr(app_auth, 'verify_additional_user', fake_verify)
    monkeypatch.setattr(
        app_auth, 'update_additional_user_password', lambda user_id, pw: update_result
    )
    monkeypatch.setattr(
        app_auth, 'delete_additional_user_safe', lambda user_id: delete_result
    )
    monkeypatch.setattr(
        app_auth, 'create_additional_user', lambda username, pw, role=None: (True, None)
    )
    monkeypatch.setattr(app_auth, '_jwt_secret', lambda: 'unit-secret')
    return verify_calls


def _status(result):
    if isinstance(result, tuple):
        return result[1]
    return result.status_code


class TestPasswordChangeConfirmation:
    def test_self_change_requires_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        with app.test_request_context(
            '/api/users/5/password', method='PUT', json={'password': 'new'}
        ):
            g.auth_role, g.auth_user, g.auth_method = 'user', 'bob', 'session'
            result = app_auth.update_user_password_endpoint(5)
        assert _status(result) == 400
        assert 'required' in result[0].get_json()['error'].lower()

    def test_self_change_rejects_wrong_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        with app.test_request_context(
            '/api/users/5/password',
            method='PUT',
            json={'password': 'new', 'current_password': 'wrong'},
        ):
            g.auth_role, g.auth_user, g.auth_method = 'user', 'bob', 'session'
            result = app_auth.update_user_password_endpoint(5)
        assert _status(result) == 400
        assert 'incorrect' in result[0].get_json()['error'].lower()

    def test_self_change_succeeds_and_reissues_cookie(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        with app.test_request_context(
            '/api/users/5/password',
            method='PUT',
            json={'password': 'new', 'current_password': 'right-pw'},
        ):
            g.auth_role, g.auth_user, g.auth_method = 'user', 'bob', 'session'
            result = app_auth.update_user_password_endpoint(5)
        assert _status(result) == 200
        set_cookie = result.headers.get('Set-Cookie', '')
        assert 'audiomuse_jwt=' in set_cookie

    def test_admin_changing_other_user_requires_own_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context(
            '/api/users/5/password', method='PUT', json={'password': 'new'}
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.update_user_password_endpoint(5)
        assert _status(result) == 400

    def test_admin_changing_other_user_with_password_no_cookie_reissue(self, app, monkeypatch):
        verify_calls = _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context(
            '/api/users/5/password',
            method='PUT',
            json={'password': 'new', 'current_password': 'right-pw'},
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.update_user_password_endpoint(5)
        assert _status(result) == 200
        assert 'audiomuse_jwt=' not in (result.headers.get('Set-Cookie') or '')
        assert verify_calls == [('root', 'right-pw')]

    def test_non_admin_cannot_change_other_user(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        with app.test_request_context(
            '/api/users/5/password',
            method='PUT',
            json={'password': 'new', 'current_password': 'right-pw'},
        ):
            g.auth_role, g.auth_user, g.auth_method = 'user', 'mallory', 'session'
            result = app_auth.update_user_password_endpoint(5)
        assert _status(result) == 403

    def test_non_admin_gets_403_not_404_for_unknown_id(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        monkeypatch.setattr(app_auth, 'get_additional_user_by_id', lambda user_id: None)
        with app.test_request_context(
            '/api/users/999/password',
            method='PUT',
            json={'password': 'new', 'current_password': 'right-pw'},
        ):
            g.auth_role, g.auth_user, g.auth_method = 'user', 'mallory', 'session'
            result = app_auth.update_user_password_endpoint(999)
        assert _status(result) == 403

    def test_admin_gets_404_for_unknown_id(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        monkeypatch.setattr(app_auth, 'get_additional_user_by_id', lambda user_id: None)
        with app.test_request_context(
            '/api/users/999/password',
            method='PUT',
            json={'password': 'new', 'current_password': 'right-pw'},
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.update_user_password_endpoint(999)
        assert _status(result) == 404

    def test_bearer_caller_exempt_from_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        with app.test_request_context(
            '/api/users/5/password', method='PUT', json={'password': 'new'}
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', None, 'bearer'
            result = app_auth.update_user_password_endpoint(5)
        assert _status(result) == 200


class TestDeleteConfirmation:
    def test_delete_requires_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context('/api/users/5', method='DELETE'):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.delete_user_endpoint(5)
        assert _status(result) == 400
        assert 'required' in result[0].get_json()['error'].lower()

    def test_delete_rejects_wrong_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context(
            '/api/users/5', method='DELETE', json={'current_password': 'wrong'}
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.delete_user_endpoint(5)
        assert _status(result) == 400
        assert 'incorrect' in result[0].get_json()['error'].lower()

    def test_delete_succeeds_with_current_password(self, app, monkeypatch):
        verify_calls = _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context(
            '/api/users/5', method='DELETE', json={'current_password': 'right-pw'}
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.delete_user_endpoint(5)
        assert _status(result) == 200
        assert verify_calls == [('root', 'right-pw')]

    def test_bearer_delete_exempt_from_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        with app.test_request_context('/api/users/5', method='DELETE'):
            g.auth_role, g.auth_user, g.auth_method = 'admin', None, 'bearer'
            result = app_auth.delete_user_endpoint(5)
        assert _status(result) == 200

    def test_self_delete_still_blocked(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context(
            '/api/users/5', method='DELETE', json={'current_password': 'right-pw'}
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'bob', 'session'
            result = app_auth.delete_user_endpoint(5)
        assert _status(result) == 400
        assert 'own account' in result[0].get_json()['error'].lower()


class TestCreateUserConfirmation:
    def test_create_requires_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context(
            '/api/users', method='POST', json={'username': 'new', 'password': 'pw'}
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.create_user_endpoint()
        assert _status(result) == 400
        assert 'required' in result[0].get_json()['error'].lower()

    def test_create_rejects_wrong_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context(
            '/api/users',
            method='POST',
            json={'username': 'new', 'password': 'pw', 'current_password': 'wrong'},
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.create_user_endpoint()
        assert _status(result) == 400
        assert 'incorrect' in result[0].get_json()['error'].lower()

    def test_create_succeeds_with_current_password(self, app, monkeypatch):
        verify_calls = _install_endpoint_mocks(monkeypatch, verify_result='admin')
        with app.test_request_context(
            '/api/users',
            method='POST',
            json={'username': 'new', 'password': 'pw', 'current_password': 'right-pw'},
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', 'root', 'session'
            result = app_auth.create_user_endpoint()
        assert _status(result) == 201
        assert verify_calls == [('root', 'right-pw')]

    def test_bearer_create_exempt_from_current_password(self, app, monkeypatch):
        _install_endpoint_mocks(monkeypatch)
        with app.test_request_context(
            '/api/users', method='POST', json={'username': 'new', 'password': 'pw'}
        ):
            g.auth_role, g.auth_user, g.auth_method = 'admin', None, 'bearer'
            result = app_auth.create_user_endpoint()
        assert _status(result) == 201
