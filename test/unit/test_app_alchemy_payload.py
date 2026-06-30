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


class TestAlchemyApiPayloadValidation:
    def test_items_without_any_add_op_returns_400(self, client):
        response = client.post('/api/alchemy', json={'items': [{'id': 'song-1', 'op': 'SUBTRACT'}]})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Invalid request'}

    def test_add_item_missing_id_returns_400(self, client):
        response = client.post('/api/alchemy', json={'items': [{'op': 'ADD', 'type': 'song'}]})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Invalid request'}

    @patch('app_alchemy.song_alchemy')
    def test_add_items_without_id_are_filtered_before_dispatch(self, mock_alchemy, client):
        mock_alchemy.side_effect = ValueError('At least one item must be in the ADD set')
        response = client.post(
            '/api/alchemy', json={'items': [{'op': 'ADD'}, {'op': 'ADD', 'id': ''}]}
        )
        assert response.status_code == 400
        assert mock_alchemy.call_args.kwargs['add_items'] == []
        assert mock_alchemy.call_args.kwargs['subtract_items'] == []
