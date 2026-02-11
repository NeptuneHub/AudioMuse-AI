"""Unit tests for app_setup.py Flask blueprint

Tests cover the setup wizard and provider management:
- Provider config validation (PROVIDER_SCHEMAS)
- Setup status detection (env auto-detect, DB flag)
- Provider CRUD operations
- Settings management (get/set/apply)
- API endpoint responses
- Multi-provider mode
"""
import json
import sys
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call

flask = pytest.importorskip('flask', reason='Flask not installed')
from flask import Flask

# Pre-register mock for tasks.mediaserver to avoid pydub/audioop import chain
if 'tasks.mediaserver' not in sys.modules:
    _mock_mediaserver = MagicMock()
    _mock_mediaserver.get_available_provider_types = Mock(return_value={})
    _mock_mediaserver.get_provider_info = Mock(return_value=None)
    _mock_mediaserver.test_provider_connection = Mock(return_value=(True, 'OK'))
    _mock_mediaserver.get_sample_tracks_from_provider = Mock(return_value=[])
    _mock_mediaserver.get_libraries_for_provider = Mock(return_value=[])
    _mock_mediaserver.PROVIDER_TYPES = {
        'jellyfin': {'name': 'Jellyfin', 'description': 'Jellyfin Server',
                     'supports_user_auth': True, 'supports_play_history': True},
        'navidrome': {'name': 'Navidrome', 'description': 'Navidrome Server',
                      'supports_user_auth': True, 'supports_play_history': True},
        'lyrion': {'name': 'Lyrion', 'description': 'Lyrion Music Server',
                   'supports_user_auth': False, 'supports_play_history': True},
        'emby': {'name': 'Emby', 'description': 'Emby Server',
                 'supports_user_auth': True, 'supports_play_history': True},
        'localfiles': {'name': 'Local Files', 'description': 'Local file system',
                       'supports_user_auth': False, 'supports_play_history': False},
    }
    sys.modules['tasks.mediaserver'] = _mock_mediaserver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a Flask app with the setup blueprint registered."""
    with patch('app_setup.get_db') as _mock_get_db, \
         patch('app_setup.detect_music_path_prefix') as _mock_detect, \
         patch('app_setup.encrypt_provider_config', side_effect=lambda x: x), \
         patch('app_setup.decrypt_provider_config', side_effect=lambda x: x):
        from app_setup import setup_bp
        flask_app = Flask(__name__)
        flask_app.register_blueprint(setup_bp)
        flask_app.config['TESTING'] = True
        yield flask_app


@pytest.fixture
def client(app):
    """Create a Flask test client."""
    return app.test_client()


def _make_mock_cursor(rows=None, fetchone_val=None, rowcount=1):
    """Helper to create a mock DB cursor with context-manager support."""
    mock_cur = MagicMock()
    if rows is not None:
        mock_cur.fetchall.return_value = rows
    if fetchone_val is not None:
        mock_cur.fetchone.return_value = fetchone_val
    mock_cur.rowcount = rowcount
    return mock_cur


def _make_mock_db(cursor):
    """Helper to create a mock DB connection that yields the given cursor."""
    mock_db = MagicMock()
    mock_db.cursor.return_value.__enter__ = Mock(return_value=cursor)
    mock_db.cursor.return_value.__exit__ = Mock(return_value=False)
    return mock_db


# ---------------------------------------------------------------------------
# Provider Config Validation (PROVIDER_SCHEMAS)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProviderConfigValidation:
    """Test validate_provider_config() for all provider types."""

    def _validate(self, provider_type, config_data):
        from app_setup import validate_provider_config
        return validate_provider_config(provider_type, config_data)

    def test_unknown_provider_type_invalid(self):
        valid, errors = self._validate('unknown_type', {})
        assert not valid
        assert 'Unknown provider type' in errors[0]

    def test_jellyfin_valid(self):
        valid, errors = self._validate('jellyfin', {
            'url': 'http://localhost:8096',
            'user_id': 'user123',
            'token': 'abc123'
        })
        assert valid
        assert len(errors) == 0

    def test_jellyfin_missing_required_fields(self):
        valid, errors = self._validate('jellyfin', {'url': 'http://localhost'})
        assert not valid
        assert any('user_id' in e for e in errors)
        assert any('token' in e for e in errors)

    def test_jellyfin_invalid_url_scheme(self):
        valid, errors = self._validate('jellyfin', {
            'url': 'ftp://localhost:8096',
            'user_id': 'user123',
            'token': 'abc123'
        })
        assert not valid
        assert any('http://' in e for e in errors)

    def test_navidrome_valid(self):
        valid, errors = self._validate('navidrome', {
            'url': 'https://navidrome.local',
            'user': 'admin',
            'password': 'pass123'
        })
        assert valid

    def test_navidrome_missing_password(self):
        valid, errors = self._validate('navidrome', {
            'url': 'https://navidrome.local',
            'user': 'admin'
        })
        assert not valid
        assert any('password' in e for e in errors)

    def test_lyrion_valid(self):
        valid, errors = self._validate('lyrion', {
            'url': 'http://lyrion.local:9000'
        })
        assert valid

    def test_lyrion_missing_url(self):
        valid, errors = self._validate('lyrion', {})
        assert not valid
        assert any('url' in e for e in errors)

    def test_emby_valid(self):
        valid, errors = self._validate('emby', {
            'url': 'http://emby.local:8096',
            'user_id': 'uid',
            'token': 'tok'
        })
        assert valid

    def test_localfiles_valid(self):
        with patch('os.path.isabs', return_value=True):
            valid, errors = self._validate('localfiles', {
                'music_directory': '/music/library'
            })
        assert valid

    def test_localfiles_relative_path_invalid(self):
        valid, errors = self._validate('localfiles', {
            'music_directory': 'relative/path'
        })
        assert not valid
        assert any('absolute' in e for e in errors)

    def test_localfiles_missing_music_directory(self):
        valid, errors = self._validate('localfiles', {})
        assert not valid
        assert any('music_directory' in e for e in errors)

    def test_all_provider_types_in_schema(self):
        """All known provider types have validation schemas."""
        from app_setup import PROVIDER_SCHEMAS
        expected = {'jellyfin', 'navidrome', 'lyrion', 'emby', 'localfiles'}
        assert set(PROVIDER_SCHEMAS.keys()) == expected


# ---------------------------------------------------------------------------
# Setup Status Detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSetupStatus:
    """Test is_setup_completed() with env-var auto-detection."""

    @patch('app_setup.get_setting', return_value=True)
    def test_completed_from_db_flag(self, mock_get):
        from app_setup import is_setup_completed
        assert is_setup_completed() is True

    @patch('app_setup.get_setting', return_value=None)
    @patch('app_setup.create_default_provider_from_env')
    @patch('app_setup.set_setting')
    def test_auto_detect_jellyfin_env(self, mock_set, mock_create, mock_get):
        """Jellyfin env vars auto-complete setup."""
        import config
        with patch.object(config, 'MEDIASERVER_TYPE', 'jellyfin'), \
             patch.object(config, 'JELLYFIN_URL', 'http://jf:8096'), \
             patch.object(config, 'JELLYFIN_TOKEN', 'tok123'), \
             patch.object(config, 'JELLYFIN_USER_ID', 'user1'):
            from app_setup import is_setup_completed
            result = is_setup_completed()
            assert result is True
            mock_create.assert_called_once()

    @patch('app_setup.get_setting', return_value=None)
    def test_localfiles_requires_wizard(self, mock_get):
        """localfiles provider type requires the wizard."""
        import config
        with patch.object(config, 'MEDIASERVER_TYPE', 'localfiles'):
            from app_setup import is_setup_completed
            result = is_setup_completed()
            assert result is False

    @patch('app_setup.get_setting', return_value=None)
    def test_placeholder_values_not_detected(self, mock_get):
        """Placeholder values like 'your_...' are not auto-detected."""
        import config
        with patch.object(config, 'MEDIASERVER_TYPE', 'jellyfin'), \
             patch.object(config, 'JELLYFIN_URL', 'http://your_jellyfin_url'), \
             patch.object(config, 'JELLYFIN_TOKEN', 'your_token'), \
             patch.object(config, 'JELLYFIN_USER_ID', 'your_user_id'):
            from app_setup import is_setup_completed
            result = is_setup_completed()
            assert result is False


# ---------------------------------------------------------------------------
# Settings Management
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSettingsManagement:
    """Test get/set settings and apply_settings_to_config."""

    @patch('app_setup.get_setting')
    def test_apply_int_setting(self, mock_get_setting):
        """Integer settings are type-cast correctly."""
        mock_get_setting.return_value = '10'
        import config
        original = config.MAX_SONGS_PER_ARTIST_PLAYLIST
        try:
            from app_setup import apply_settings_to_config
            # Only the max_songs_per_artist_playlist key should match
            mock_get_setting.side_effect = lambda key: '10' if key == 'max_songs_per_artist_playlist' else None
            apply_settings_to_config()
            assert config.MAX_SONGS_PER_ARTIST_PLAYLIST == 10
        finally:
            config.MAX_SONGS_PER_ARTIST_PLAYLIST = original

    @patch('app_setup.get_setting')
    def test_apply_bool_setting(self, mock_get_setting):
        """Boolean settings are type-cast correctly."""
        import config
        original = config.PLAYLIST_ENERGY_ARC
        try:
            mock_get_setting.side_effect = lambda key: 'true' if key == 'playlist_energy_arc' else None
            from app_setup import apply_settings_to_config
            apply_settings_to_config()
            assert config.PLAYLIST_ENERGY_ARC is True
        finally:
            config.PLAYLIST_ENERGY_ARC = original


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSetupEndpoints:
    """Test setup API endpoints."""

    def test_setup_page_renders(self, client):
        """GET /setup returns 200."""
        with patch('app_setup.render_template', return_value='<html>setup</html>'):
            resp = client.get('/setup')
            assert resp.status_code == 200
            assert 'text/html' in resp.content_type

    def test_settings_page_renders(self, client):
        """GET /settings returns 200."""
        with patch('app_setup.render_template', return_value='<html>settings</html>'):
            resp = client.get('/settings')
            assert resp.status_code == 200
            assert 'text/html' in resp.content_type

    @patch('app_setup.get_providers', return_value=[])
    @patch('app_setup.is_setup_completed', return_value=False)
    @patch('app_setup.is_multi_provider_enabled', return_value=False)
    @patch('app_setup.create_default_provider_from_env')
    def test_status_endpoint(self, mock_create, mock_multi, mock_setup, mock_providers, client):
        """GET /api/setup/status returns status JSON."""
        resp = client.get('/api/setup/status')
        assert resp.status_code == 200
        assert resp.content_type.startswith('application/json')
        data = resp.get_json()
        assert 'setup_completed' in data
        assert 'provider_count' in data

    @patch('app_setup.get_available_provider_types')
    @patch('app_setup.get_provider_info')
    def test_provider_types_endpoint(self, mock_info, mock_types, client):
        """GET /api/setup/providers/types returns provider type list."""
        mock_types.return_value = {
            'jellyfin': {'name': 'Jellyfin', 'description': 'Jellyfin Server',
                         'supports_user_auth': True, 'supports_play_history': True}
        }
        mock_info.return_value = {'config_fields': [{'name': 'url'}]}
        resp = client.get('/api/setup/providers/types')
        assert resp.status_code == 200
        assert resp.content_type.startswith('application/json')
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]['type'] == 'jellyfin'

    @patch('app_setup.get_providers', return_value=[])
    def test_list_providers_empty(self, mock_providers, client):
        """GET /api/setup/providers returns empty list."""
        resp = client.get('/api/setup/providers')
        assert resp.status_code == 200
        assert resp.content_type.startswith('application/json')
        assert resp.get_json() == []


# ---------------------------------------------------------------------------
# Provider CRUD
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProviderCRUD:
    """Test provider create/update/delete endpoints."""

    def test_create_provider_missing_data(self, client):
        """POST /api/setup/providers with no body returns 400."""
        resp = client.post('/api/setup/providers',
                          data='', content_type='application/json')
        assert resp.status_code in (400, 415)

    def test_create_provider_missing_type(self, client):
        """POST /api/setup/providers without provider_type returns 400."""
        resp = client.post('/api/setup/providers', json={'name': 'Test'})
        assert resp.status_code == 400
        assert 'provider_type' in resp.get_json()['error']

    def test_create_provider_missing_name(self, client):
        """POST /api/setup/providers without name returns 400."""
        resp = client.post('/api/setup/providers', json={'provider_type': 'jellyfin'})
        assert resp.status_code == 400
        assert 'name' in resp.get_json()['error']

    @patch('app_setup.PROVIDER_TYPES', {'jellyfin': {'name': 'Jellyfin'}})
    @patch('app_setup.validate_provider_config', return_value=(True, []))
    @patch('app_setup.get_providers', return_value=[])
    @patch('app_setup.add_provider', return_value=1)
    def test_create_provider_success(self, mock_add, mock_get, mock_validate, client):
        """Successful provider creation returns 201."""
        resp = client.post('/api/setup/providers', json={
            'provider_type': 'jellyfin',
            'name': 'My Jellyfin',
            'config': {'url': 'http://jf:8096', 'user_id': 'u', 'token': 't'}
        })
        assert resp.status_code == 201
        assert resp.get_json()['id'] == 1

    @patch('app_setup.PROVIDER_TYPES', {'jellyfin': {'name': 'Jellyfin'}})
    @patch('app_setup.validate_provider_config', return_value=(True, []))
    @patch('app_setup.get_providers', return_value=[
        {'id': 1, 'provider_type': 'jellyfin', 'name': 'Old', 'config': {}}
    ])
    @patch('app_setup.update_provider', return_value=True)
    def test_create_provider_upserts_existing(self, mock_update, mock_get, mock_validate, client):
        """Creating a provider of existing type upserts instead."""
        resp = client.post('/api/setup/providers', json={
            'provider_type': 'jellyfin',
            'name': 'Updated Jellyfin',
            'config': {'url': 'http://jf:8096', 'user_id': 'u', 'token': 't'}
        })
        assert resp.status_code == 200
        assert resp.get_json().get('was_update') is True

    @patch('app_setup.validate_provider_config', return_value=(False, ['Missing url']))
    @patch('app_setup.PROVIDER_TYPES', {'jellyfin': {'name': 'Jellyfin'}})
    def test_create_provider_validation_failure(self, mock_validate, client):
        """Provider creation with invalid config returns 400."""
        resp = client.post('/api/setup/providers', json={
            'provider_type': 'jellyfin',
            'name': 'Bad Config',
            'config': {}
        })
        assert resp.status_code == 400
        assert 'Validation failed' in resp.get_json()['error']

    @patch('app_setup.delete_provider', return_value=True)
    def test_delete_provider_success(self, mock_delete, client):
        """DELETE /api/setup/providers/<id> returns success."""
        resp = client.delete('/api/setup/providers/1')
        assert resp.status_code == 200

    @patch('app_setup.delete_provider', return_value=False)
    def test_delete_provider_not_found(self, mock_delete, client):
        """DELETE nonexistent provider returns 404."""
        resp = client.delete('/api/setup/providers/999')
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Complete Setup & Multi-Provider
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSetupCompletion:
    """Test setup completion and multi-provider mode."""

    @patch('app_setup.set_setting')
    def test_complete_setup_marks_flag(self, mock_set, client):
        """POST /api/setup/complete marks setup as completed."""
        resp = client.post('/api/setup/complete')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['setup_completed'] is True
        # Verify set_setting was called with setup_completed=True
        calls = [c for c in mock_set.call_args_list if c[0][0] == 'setup_completed']
        assert len(calls) >= 1

    @patch('app_setup.set_setting')
    def test_enable_multi_provider(self, mock_set, client):
        """POST /api/setup/multi-provider enables multi-provider mode."""
        resp = client.post('/api/setup/multi-provider', json={'enabled': True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['multi_provider_enabled'] is True

    @patch('app_setup.set_setting')
    def test_disable_multi_provider(self, mock_set, client):
        """POST /api/setup/multi-provider disables multi-provider mode."""
        resp = client.post('/api/setup/multi-provider', json={'enabled': False})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['multi_provider_enabled'] is False

    def test_multi_provider_no_data(self, client):
        """POST /api/setup/multi-provider with no data returns 400."""
        resp = client.post('/api/setup/multi-provider',
                          data='', content_type='application/json')
        assert resp.status_code in (400, 415)


# ---------------------------------------------------------------------------
# Settings Endpoints
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSettingsEndpoints:
    """Test settings API endpoints."""

    @patch('app_setup.get_all_settings', return_value={'general': {'key1': {'value': 'val1'}}})
    def test_get_settings(self, mock_all, client):
        """GET /api/setup/settings returns grouped settings."""
        resp = client.get('/api/setup/settings')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'general' in data

    @patch('app_setup.set_setting')
    @patch('app_setup.apply_settings_to_config')
    def test_update_settings(self, mock_apply, mock_set, client):
        """PUT /api/setup/settings updates settings and applies them."""
        resp = client.put('/api/setup/settings', json={'ai_provider': 'GEMINI'})
        assert resp.status_code == 200
        mock_set.assert_called_once_with('ai_provider', 'GEMINI')
        mock_apply.assert_called_once()

    def test_update_settings_no_data(self, client):
        """PUT /api/setup/settings with no data returns 400."""
        resp = client.put('/api/setup/settings',
                         data='', content_type='application/json')
        assert resp.status_code in (400, 415)


# ---------------------------------------------------------------------------
# Provider Update Endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProviderUpdateEndpoint:
    """Test PUT /api/setup/providers/<id> endpoint."""

    @patch('app_setup.get_provider_by_id', return_value=None)
    def test_update_nonexistent_provider_returns_404(self, mock_get, client):
        """PUT on nonexistent provider returns 404."""
        resp = client.put('/api/setup/providers/999', json={'name': 'New Name'})
        assert resp.status_code == 404
        assert resp.content_type.startswith('application/json')
        assert 'error' in resp.get_json()

    @patch('app_setup.get_provider_by_id', return_value={
        'id': 1, 'provider_type': 'jellyfin', 'name': 'Jelly',
        'config': {'url': 'http://jf:8096', 'user_id': 'u', 'token': 't'},
        'enabled': True, 'priority': 0,
    })
    def test_update_provider_no_data_returns_400(self, mock_get, client):
        """PUT with empty body returns 400."""
        resp = client.put('/api/setup/providers/1',
                         data='', content_type='application/json')
        assert resp.status_code in (400, 415)

    @patch('app_setup.update_provider', return_value=True)
    @patch('app_setup.validate_provider_config', return_value=(True, []))
    @patch('app_setup.get_provider_by_id', return_value={
        'id': 1, 'provider_type': 'jellyfin', 'name': 'Jelly',
        'config': {'url': 'http://jf:8096', 'user_id': 'u', 'token': 't'},
        'enabled': True, 'priority': 0,
    })
    def test_update_provider_success(self, mock_get, mock_validate, mock_update, client):
        """Successful provider update returns 200."""
        resp = client.put('/api/setup/providers/1', json={
            'name': 'Updated Jellyfin',
            'config': {'url': 'http://jf:8096', 'user_id': 'u2', 'token': 't2'}
        })
        assert resp.status_code == 200
        assert resp.content_type.startswith('application/json')
        assert 'message' in resp.get_json()


# ---------------------------------------------------------------------------
# Provider Test Endpoints
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProviderTestEndpoints:
    """Test provider connection test endpoints."""

    @patch('app_setup.get_provider_by_id', return_value=None)
    def test_test_saved_provider_not_found(self, mock_get, client):
        """POST /api/setup/providers/<id>/test returns 404 for missing provider."""
        resp = client.post('/api/setup/providers/999/test')
        assert resp.status_code == 404
        assert 'error' in resp.get_json()

    @patch('app_setup.test_provider_connection', return_value=(True, 'Connection OK'))
    @patch('app_setup.get_provider_by_id', return_value={
        'id': 1, 'provider_type': 'jellyfin', 'name': 'Jelly',
        'config': {'url': 'http://jf:8096', 'user_id': 'u', 'token': 't'},
        'enabled': True, 'priority': 0,
    })
    def test_test_saved_provider_success(self, mock_get, mock_test, client):
        """POST /api/setup/providers/<id>/test returns success result."""
        resp = client.post('/api/setup/providers/1/test')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert 'message' in data
        assert data['provider_type'] == 'jellyfin'

    def test_test_unsaved_provider_no_data(self, client):
        """POST /api/setup/providers/test with no data returns 400."""
        resp = client.post('/api/setup/providers/test',
                          data='', content_type='application/json')
        assert resp.status_code in (400, 415)

    def test_test_unsaved_provider_missing_type(self, client):
        """POST /api/setup/providers/test without provider_type returns 400."""
        resp = client.post('/api/setup/providers/test', json={'config': {}})
        assert resp.status_code == 400
        assert 'error' in resp.get_json()

    @patch('app_setup.test_provider_connection', return_value=(False, 'Connection refused'))
    def test_test_unsaved_provider_failure(self, mock_test, client):
        """POST /api/setup/providers/test with failing connection."""
        resp = client.post('/api/setup/providers/test', json={
            'provider_type': 'jellyfin',
            'config': {'url': 'http://bad:8096'},
            'detect_prefix': False
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is False
        assert 'message' in data


# ---------------------------------------------------------------------------
# Browse Directories Endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBrowseDirectoriesEndpoint:
    """Test GET /api/setup/browse-directories endpoint."""

    def test_path_traversal_rejected(self, client):
        """Path traversal with '..' is rejected."""
        resp = client.get('/api/setup/browse-directories?path=/music/../etc')
        assert resp.status_code == 400
        assert 'error' in resp.get_json()
