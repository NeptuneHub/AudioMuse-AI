import pytest
from unittest.mock import patch
from flask import Flask

from app_alchemy import alchemy_bp


@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(alchemy_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestCreateAnchorValidation:
    @patch('app_helper.save_alchemy_anchor')
    def test_whitespace_only_name_returns_400(self, mock_save, client):
        response = client.post('/api/anchors', json={'name': '   ', 'centroid': [0.1, 0.2]})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Anchor name is required'}
        mock_save.assert_not_called()

    @patch('app_helper.save_alchemy_anchor')
    def test_non_list_centroid_returns_400(self, mock_save, client):
        response = client.post('/api/anchors', json={'name': 'My Anchor', 'centroid': 'not-a-list'})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Anchor centroid is required and must be a list'}
        mock_save.assert_not_called()

    @patch('app_helper.save_alchemy_anchor')
    def test_empty_list_centroid_returns_400(self, mock_save, client):
        response = client.post('/api/anchors', json={'name': 'My Anchor', 'centroid': []})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Anchor centroid is required and must be a list'}
        mock_save.assert_not_called()


class TestRenameAnchorValidation:
    @patch('app_helper.update_alchemy_anchor_name')
    def test_whitespace_only_name_returns_400(self, mock_update, client):
        response = client.put('/api/anchors/7', json={'name': '   '})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Anchor name is required'}
        mock_update.assert_not_called()
