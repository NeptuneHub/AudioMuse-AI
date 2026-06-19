import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

import config
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


class TestListRadios:
    @patch('database.get_alchemy_radios')
    def test_returns_radio_list(self, mock_get, client):
        mock_get.return_value = [
            {'id': 1, 'anchor_id': 5, 'name': 'Chill', 'temperature': 1.0, 'n_results': 100, 'enabled': True},
            {'id': 2, 'anchor_id': 7, 'name': 'Rock', 'temperature': 0.5, 'n_results': 50, 'enabled': False},
        ]
        response = client.get('/api/radios')
        assert response.status_code == 200
        payload = response.get_json()
        assert payload == {'radios': [
            {'id': 1, 'anchor_id': 5, 'name': 'Chill', 'temperature': 1.0, 'n_results': 100, 'enabled': True},
            {'id': 2, 'anchor_id': 7, 'name': 'Rock', 'temperature': 0.5, 'n_results': 50, 'enabled': False},
        ]}


class TestCreateRadioValidation:
    @patch('database.create_alchemy_radio')
    def test_missing_anchor_id_returns_400(self, mock_create, client):
        response = client.post('/api/radios', json={'temperature': 1.0, 'n_results': 100})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Radio anchor is required'}
        mock_create.assert_not_called()

    @patch('database.create_alchemy_radio')
    def test_missing_temperature_returns_400(self, mock_create, client):
        response = client.post('/api/radios', json={'anchor_id': 5, 'n_results': 100})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Radio temperature is required'}
        mock_create.assert_not_called()

    @patch('database.create_alchemy_radio')
    def test_missing_n_results_returns_400(self, mock_create, client):
        response = client.post('/api/radios', json={'anchor_id': 5, 'temperature': 1.0})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Radio number of results is required'}
        mock_create.assert_not_called()

    @patch('database.create_alchemy_radio')
    def test_non_numeric_temperature_returns_400(self, mock_create, client):
        response = client.post('/api/radios', json={'anchor_id': 5, 'temperature': 'hot', 'n_results': 100})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Radio temperature must be a number'}
        mock_create.assert_not_called()

    @patch('database.create_alchemy_radio')
    def test_negative_temperature_returns_400(self, mock_create, client):
        response = client.post('/api/radios', json={'anchor_id': 5, 'temperature': -1, 'n_results': 100})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Radio temperature must be 0 or greater'}
        mock_create.assert_not_called()

    @patch('database.create_alchemy_radio')
    def test_non_finite_temperature_returns_400(self, mock_create, client):
        for bad_value in ('NaN', 'Infinity', 'inf'):
            response = client.post('/api/radios',
                                   data='{"anchor_id": 5, "temperature": ' + ('"' + bad_value + '"' if bad_value == 'inf' else bad_value) + ', "n_results": 100}',
                                   content_type='application/json')
            assert response.status_code == 400
            assert response.get_json() == {'error': 'Radio temperature must be a finite number'}
        mock_create.assert_not_called()

    @patch('database.create_alchemy_radio')
    def test_n_results_out_of_range_returns_400(self, mock_create, client):
        for bad_value in (0, config.ALCHEMY_MAX_N_RESULTS + 1):
            response = client.post('/api/radios', json={'anchor_id': 5, 'temperature': 1.0, 'n_results': bad_value})
            assert response.status_code == 400
        mock_create.assert_not_called()

    @patch('database.create_alchemy_radio')
    def test_valid_payload_creates_radio(self, mock_create, client):
        mock_create.return_value = {'id': 1, 'anchor_id': 5, 'temperature': 1.0, 'n_results': 100, 'enabled': True}
        response = client.post('/api/radios', json={'anchor_id': 5, 'temperature': 1.0, 'n_results': 100})
        assert response.status_code == 200
        assert response.get_json() == {'radio': {'id': 1, 'anchor_id': 5, 'temperature': 1.0, 'n_results': 100, 'enabled': True}}
        mock_create.assert_called_once_with(5, 1.0, 100, True)

    @patch('database.create_alchemy_radio')
    def test_duplicate_anchor_returns_400(self, mock_create, client):
        mock_create.return_value = None
        response = client.post('/api/radios', json={'anchor_id': 5, 'temperature': 1.0, 'n_results': 100})
        assert response.status_code == 400
        assert 'error' in response.get_json()


class TestUpdateRadio:
    @patch('database.update_alchemy_radio')
    def test_missing_settings_returns_400(self, mock_update, client):
        response = client.put('/api/radios/3', json={'enabled': False})
        assert response.status_code == 400
        mock_update.assert_not_called()

    @patch('database.update_alchemy_radio')
    def test_unknown_radio_returns_404(self, mock_update, client):
        mock_update.return_value = None
        response = client.put('/api/radios/3', json={'temperature': 0.5, 'n_results': 20, 'enabled': False})
        assert response.status_code == 404

    @patch('database.update_alchemy_radio')
    def test_valid_update_saves_disabled_state(self, mock_update, client):
        mock_update.return_value = {'id': 3, 'anchor_id': 5, 'temperature': 0.5, 'n_results': 20, 'enabled': False}
        response = client.put('/api/radios/3', json={'temperature': 0.5, 'n_results': 20, 'enabled': False})
        assert response.status_code == 200
        mock_update.assert_called_once_with(3, 0.5, 20, False)


class TestDeleteRadio:
    @patch('database.delete_alchemy_radio')
    def test_unknown_radio_returns_404(self, mock_delete, client):
        mock_delete.return_value = False
        response = client.delete('/api/radios/9')
        assert response.status_code == 404

    @patch('database.delete_alchemy_radio')
    def test_delete_returns_ok(self, mock_delete, client):
        mock_delete.return_value = True
        response = client.delete('/api/radios/9')
        assert response.status_code == 200
        assert response.get_json() == {'deleted': True}
        mock_delete.assert_called_once_with(9)


class TestRunRadioPlaylistsEndpoint:
    @patch('tasks.radio_manager.run_radio_playlists')
    def test_returns_summary(self, mock_run, client):
        mock_run.return_value = {'message': 'Created 2 radio playlist(s) from 2 enabled radio(s).',
                                 'radios_enabled': 2, 'playlists_created': 2, 'failed': []}
        response = client.post('/api/radios/run')
        assert response.status_code == 200
        assert response.get_json()['playlists_created'] == 2
        mock_run.assert_called_once_with()

    @patch('tasks.radio_manager.run_radio_playlists')
    def test_failure_returns_generic_error(self, mock_run, client):
        mock_run.side_effect = RuntimeError('internal detail that must stay in logs')
        response = client.post('/api/radios/run')
        assert response.status_code == 500
        payload = response.get_json()
        assert 'internal detail' not in payload['error']


class TestRunRadioPlaylists:
    def _radio(self, radio_id, anchor_id, name, temperature=1.0, n_results=100, enabled=True):
        return {'id': radio_id, 'anchor_id': anchor_id, 'name': name,
                'temperature': temperature, 'n_results': n_results, 'enabled': enabled}

    @patch('database.get_alchemy_radios')
    @patch('tasks.radio_manager.create_or_replace_playlist')
    @patch('tasks.radio_manager.song_alchemy')
    def test_creates_playlists_for_enabled_radios_only(self, mock_alchemy, mock_upsert,
                                                       mock_get_radios):
        from tasks.radio_manager import run_radio_playlists

        mock_get_radios.return_value = [
            self._radio(1, 10, 'Chill', temperature=0.5, n_results=30),
            self._radio(2, 11, 'Rock', enabled=False),
        ]
        mock_alchemy.return_value = {'results': [{'item_id': 'a'}, {'item_id': 'b'}]}

        summary = run_radio_playlists()

        mock_alchemy.assert_called_once_with(
            add_items=[{'type': 'anchor', 'id': 10}], n_results=30, temperature=0.5)
        mock_upsert.assert_called_once_with('Chill', ['a', 'b'])
        assert summary['playlists_created'] == 1
        assert summary['radios_enabled'] == 1

    @patch('database.get_alchemy_radios')
    @patch('tasks.radio_manager.create_or_replace_playlist')
    @patch('tasks.radio_manager.song_alchemy')
    def test_upserts_after_generation(self, mock_alchemy, mock_upsert,
                                       mock_get_radios):
        from tasks.radio_manager import run_radio_playlists

        mock_get_radios.return_value = [self._radio(1, 10, 'Chill')]
        mock_alchemy.return_value = {'results': [{'item_id': 'a'}]}
        order_tracker = MagicMock()
        order_tracker.attach_mock(mock_alchemy, 'alchemy')
        order_tracker.attach_mock(mock_upsert, 'upsert')

        run_radio_playlists()

        names = [c[0] for c in order_tracker.mock_calls]
        assert names.index('alchemy') < names.index('upsert')

    @patch('database.get_alchemy_radios')
    @patch('tasks.radio_manager.create_or_replace_playlist')
    @patch('tasks.radio_manager.song_alchemy')
    def test_one_failing_radio_does_not_block_others(self, mock_alchemy, mock_upsert,
                                                     mock_get_radios):
        from tasks.radio_manager import run_radio_playlists

        mock_get_radios.return_value = [
            self._radio(1, 10, 'Broken'),
            self._radio(2, 11, 'Chill'),
        ]
        mock_alchemy.side_effect = [RuntimeError('boom'), {'results': [{'item_id': 'x'}]}]

        summary = run_radio_playlists()

        mock_upsert.assert_called_once_with('Chill', ['x'])
        assert summary['playlists_created'] == 1
        assert summary['failed'] == ['Broken']

    @patch('database.get_alchemy_radios')
    @patch('tasks.radio_manager.create_or_replace_playlist')
    @patch('tasks.radio_manager.song_alchemy')
    def test_radio_with_no_results_creates_no_playlist(self, mock_alchemy, mock_upsert,
                                                       mock_get_radios):
        from tasks.radio_manager import run_radio_playlists

        mock_get_radios.return_value = [self._radio(1, 10, 'Empty')]
        mock_alchemy.return_value = {'results': []}

        summary = run_radio_playlists()

        mock_upsert.assert_not_called()
        assert summary['playlists_created'] == 0
        assert summary['failed'] == ['Empty']


class TestDeletePlaylistsBySuffix:
    @patch('tasks.mediaserver.config')
    @patch('tasks.mediaserver.jellyfin.get_all_playlists')
    @patch('tasks.mediaserver.jellyfin.delete_playlist')
    def test_only_deletes_radio_suffix_playlists(self, mock_delete, mock_get, mock_config):
        from tasks.mediaserver import delete_playlists_by_suffix

        mock_config.MEDIASERVER_TYPE = 'jellyfin'
        mock_get.return_value = [
            {'Id': '1', 'Name': 'Chill_radio'},
            {'Id': '2', 'Name': 'Rock_automatic'},
            {'Id': '3', 'Name': 'My Favorites'},
            {'Id': '4', 'Name': 'Jazz_radio'},
        ]
        mock_delete.return_value = True

        delete_playlists_by_suffix('_radio')

        assert mock_delete.call_count == 2
        deleted_ids = [c[0][0] for c in mock_delete.call_args_list]
        assert deleted_ids == ['1', '4']
