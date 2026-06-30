import pytest
from flask import Flask, Blueprint

import app_auth


@pytest.fixture
def app():
    app = Flask(__name__)
    app.add_url_rule('/login', 'login_page', lambda: 'login')
    dash = Blueprint('dashboard_bp', __name__)
    dash.add_url_rule('/dashboard', 'dashboard_page', lambda: 'dash')
    app.register_blueprint(dash)
    return app


class TestOriginalRequestIsHttps:
    def test_direct_https_request_is_true(self, app):
        with app.test_request_context(
            '/auth', base_url='https://example.test'
        ):
            assert app_auth._original_request_is_https() is True

    def test_plain_http_no_header_is_false(self, app):
        with app.test_request_context(
            '/auth', base_url='http://example.test'
        ):
            assert app_auth._original_request_is_https() is False

    def test_forwarded_proto_https_is_true(self, app):
        with app.test_request_context(
            '/auth',
            base_url='http://example.test',
            headers={'X-Forwarded-Proto': 'https'},
        ):
            assert app_auth._original_request_is_https() is True

    def test_forwarded_proto_http_is_false(self, app):
        with app.test_request_context(
            '/auth',
            base_url='http://example.test',
            headers={'X-Forwarded-Proto': 'http'},
        ):
            assert app_auth._original_request_is_https() is False

    def test_forwarded_proto_list_first_value_https_is_true(self, app):
        with app.test_request_context(
            '/auth',
            base_url='http://example.test',
            headers={'X-Forwarded-Proto': 'https, http'},
        ):
            assert app_auth._original_request_is_https() is True

    def test_forwarded_proto_uppercase_is_true(self, app):
        with app.test_request_context(
            '/auth',
            base_url='http://example.test',
            headers={'X-Forwarded-Proto': 'HTTPS'},
        ):
            assert app_auth._original_request_is_https() is True


class TestBearerTokenCompareDigest:
    def test_unicode_token_does_not_raise(self, app, monkeypatch):
        import config
        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', 'tok-123')
        unicode_token = 'tok-é中'
        with app.test_request_context(
            '/api/foo',
            headers={'Authorization': f'Bearer {unicode_token}'},
        ):
            result = app_auth.check_auth_needed('s')
        assert result is not None
        assert result[1] == 401

    def test_unicode_token_matches_when_correct(self, app, monkeypatch):
        import config
        from flask import g
        unicode_token = 'tok-é中'
        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', unicode_token)
        with app.test_request_context(
            '/api/foo',
            headers={'Authorization': f'Bearer {unicode_token}'},
        ):
            result = app_auth.check_auth_needed('s')
            assert result is None
            assert g.auth_role == 'admin'

    def test_correct_token_authenticates(self, app, monkeypatch):
        import config
        from flask import g
        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', 'tok-123')
        with app.test_request_context(
            '/api/foo', headers={'Authorization': 'Bearer tok-123'}
        ):
            result = app_auth.check_auth_needed('s')
            assert result is None
            assert g.auth_role == 'admin'

    def test_wrong_token_rejected(self, app, monkeypatch):
        import config
        monkeypatch.setattr(config, 'AUTH_ENABLED', True)
        monkeypatch.setattr(config, 'API_TOKEN', 'tok-123')
        with app.test_request_context(
            '/api/foo', headers={'Authorization': 'Bearer wrong'}
        ):
            result = app_auth.check_auth_needed('s')
        assert result is not None
        assert result[1] == 401
