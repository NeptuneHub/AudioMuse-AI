from unittest.mock import patch

import pytest
from flask import Flask

from app_smart_sessions import smart_sessions_bp


@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(smart_sessions_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_page_renders_with_active_nav(client):
    with patch('app_smart_sessions.render_template', return_value='<html>ok</html>') as mock_render:
        response = client.get('/smart_sessions')

    assert response.status_code == 200
    assert mock_render.call_args.kwargs['active'] == 'smart_sessions'


def test_capabilities_endpoint_returns_builder_data(client):
    with patch('app_smart_sessions.get_smart_session_capabilities', return_value={'clap_enabled': True}):
        response = client.get('/api/smart_sessions/capabilities')

    assert response.status_code == 200
    assert response.get_json() == {'clap_enabled': True}


def test_preview_rejects_empty_request(client):
    response = client.post('/api/smart_sessions/preview', json={})

    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_preview_returns_builder_response(client):
    expected = {'session_id': None, 'playlist_name': 'Smart Session', 'tracks': [], 'warnings': []}
    with patch('app_smart_sessions.build_smart_session_preview', return_value=expected) as mock_build:
        response = client.post('/api/smart_sessions/preview', json={'prompt': 'ambient focus'})

    assert response.status_code == 200
    assert response.get_json() == expected
    mock_build.assert_called_once_with({'prompt': 'ambient focus'})


def test_export_rejects_missing_playlist_name(client):
    response = client.post('/api/smart_sessions/export', json={'track_ids': ['a']})

    assert response.status_code == 400
    assert 'playlist_name' in response.get_json()['error']


def test_export_uses_builder_helper(client):
    result = {'message': "Playlist 'Smart Session' created successfully!", 'playlist_id': 'pl-1'}
    with patch('app_smart_sessions.export_smart_session_playlist', return_value=result) as mock_export:
        response = client.post('/api/smart_sessions/export', json={
            'playlist_name': 'Smart Session',
            'track_ids': ['a', 'b'],
        })

    assert response.status_code == 201
    assert response.get_json() == result
    mock_export.assert_called_once_with({'playlist_name': 'Smart Session', 'track_ids': ['a', 'b']})