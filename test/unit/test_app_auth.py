"""Security-contract unit tests for app_auth (first coverage for this module).

Covers the JWT cookie path (incl. the empty-secret fail-closed behaviour), the
Bearer token constant-time path, admin-only path enforcement, and the argon2
password hashing of the user CRUD. The DB is mocked; the auth/admin checks run
inside a real request context so g/request/url_for behave as in production.
"""
from unittest.mock import MagicMock

import jwt as pyjwt
import pytest
from flask import Flask, Blueprint, g

import app_auth


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
        secret = 'unit-secret'
        token = pyjwt.encode({'sub': 'alice', 'role': 'user'}, secret, algorithm='HS256')
        with app.test_request_context('/api/foo', headers={'Cookie': f'audiomuse_jwt={token}'}):
            result = app_auth.check_auth_needed(secret)
            assert result is None
            assert g.auth_role == 'user'
            assert g.auth_user == 'alice'

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
        assert ('tok-123', 'tok-123') in calls

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

    @pytest.mark.parametrize('path,expected', [
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
    ])
    def test_is_admin_path_matrix(self, path, expected):
        assert app_auth.is_admin_path(path) is expected


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
