# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Recording capabilities: authenticated, source-scoped and free of warmup effects.

Main Features:
* Verifies runtime capability fields and selected-server index coverage.
* Exercises ordinary-user authentication and the existing setup barrier.
* Checks count caching, invalidation, unknown states and sanitized failures.
* Rejects model loading, inference and recording warmup during status checks.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import jwt
import numpy as np
import pytest
from flask import Flask

import app_auth
import app_recording_search
import config
from tasks import neural_fingerprint as nf
from tasks import neural_fingerprint_index as nfi
from tasks import recording_search_manager as manager
from tasks.index_availability import AvailabilityCache
from tasks.mediaserver import registry

URL = '/api/recording_search/status'
SECRET = 'recording-status-unit-secret-at-least-32-bytes'


@pytest.fixture
def environment(monkeypatch, tmp_path):
    files = [tmp_path / 'encoder.onnx', tmp_path / 'codebook.npz']
    for path in files:
        path.touch()
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', True)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', str(files[0]))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(files[1]))
    monkeypatch.setattr(config, 'AUTH_ENABLED', False)
    monkeypatch.setattr(config, 'API_TOKEN', 'unit-api-token')
    monkeypatch.setattr(app_auth, 'check_setup_needed', lambda: False)
    monkeypatch.setattr(app_auth, '_jwt_secret', lambda: SECRET)
    monkeypatch.setattr(registry, 'get_default_server_id', lambda: 'primary')
    monkeypatch.setattr(registry, 'get_server', lambda value: (
        {'server_id': value} if value in ('primary', 'secondary') else None
    ))
    monkeypatch.setattr(registry, 'get_server_by_name', lambda value: (
        {'server_id': 'secondary'} if value == 'Other library' else None
    ))
    monkeypatch.setattr(registry, 'has_secondary_servers', lambda: True)
    pack = SimpleNamespace(build_id='build-1', ids=np.array(['fp_a', 'fp_b', 'legacy']), live_tracks=3)
    state = {'pack': pack, 'building': False, 'error': None}
    monkeypatch.setattr(nfi, '_STATE', state)
    monkeypatch.setattr(nfi, '_CANONICAL', {})
    cache = AvailabilityCache()
    monkeypatch.setattr(cache, '_arm_idle_drop', lambda: None)
    monkeypatch.setattr(nfi, '_STATUS_COUNTS', cache)

    # Exercise the real shared availability-mask builder, with only persistence
    # mocked. The default server also owns legacy IDs; the secondary owns none.
    import database
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    selected = {'id': None}

    def execute(sql, args):
        assert sql.startswith('SELECT ')
        selected['id'] = args[0]

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = lambda: (selected['id'] == 'primary', None)
    cursor.fetchall.side_effect = lambda: [('fp_a',)] if selected['id'] == 'primary' else []
    monkeypatch.setattr(database, 'get_db', lambda: connection)

    def forbidden(*args, **kwargs):
        pytest.fail('Status must not load models/indexes, infer, or renew recording warmup')

    for module, names in (
        (nf, ('warm_session', 'fingerprint_audio', 'codebook')),
        (nfi, ('ensure_loaded', 'start_background_load', 'reload_from_db', 'identify')),
        (manager, ('warmup_recording_models', '_arm_idle_unload', 'run_recording_search')),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(manager._TIMER, 'arm', forbidden)
    return SimpleNamespace(state=state, files=files, cursor=cursor, cache=cache, pack=pack)


@pytest.fixture
def client(environment):
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.before_request(app_auth.auth_setup_barrier)
    app.register_blueprint(app_recording_search.recording_search_bp)
    return app.test_client()


def test_runtime_contract_and_read_only_repeated_checks(client, environment, monkeypatch):
    values = {
        'APP_VERSION': '3.6.1-rc.2', 'RECORDING_SEARCH_RECORD_SECONDS': 17,
        'RECORDING_SEARCH_MAX_CLIP_SECONDS': 42, 'RECORDING_SEARCH_MAX_UPLOAD_MB': 7,
        'RECORDING_SEARCH_DEFAULT_N_RESULTS': 12,
    }
    for key, value in values.items():
        monkeypatch.setattr(config, key, value)
    response = client.get(URL)
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.json == {
        'api_version': 1, 'app_version': '3.6.1-rc.2', 'server_id': 'primary',
        'enabled': True, 'model_available': True, 'ready': True,
        'index': {'state': 'ready', 'indexed_tracks': 2},
        'recording': {'recommended_seconds': 17, 'max_clip_seconds': 42,
                      'max_upload_bytes': 7 * 1024 * 1024, 'default_n_results': 12},
    }
    first_reads = environment.cursor.execute.call_count
    assert first_reads == 2
    for _ in range(3):
        assert client.get(URL).json == response.json
    assert environment.cursor.execute.call_count == first_reads
    assert environment.state == {'pack': environment.pack, 'building': False, 'error': None}
    assert all(getattr(config, key) == value for key, value in values.items())


@pytest.mark.parametrize('query,server,count', [
    ('', 'primary', 2), ('?server_id=primary', 'primary', 2),
    ('?server_id=secondary', 'secondary', 0), ('?server=Other%20library', 'secondary', 0),
    ('?server=secondary&server_id=primary', 'secondary', 0),
])
def test_resolved_source_and_real_availability_rules(client, query, server, count):
    body = client.get(URL + query).json
    assert body['server_id'] == server
    assert body['index'] == {'state': 'ready' if count else 'empty', 'indexed_tracks': count}
    assert body['ready'] is bool(count)


@pytest.mark.parametrize('enabled,missing,ready', [(False, None, False), (True, 0, False), (True, 1, False)])
def test_enabled_and_model_files_are_independent(client, environment, monkeypatch, enabled, missing, ready):
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', enabled)
    if missing is not None:
        environment.files[missing].unlink()
    body = client.get(URL).json
    assert body['enabled'] is enabled
    assert body['model_available'] is (missing is None)
    assert body['index']['state'] == 'ready'
    assert body['ready'] is ready
    assert nf.is_available() is (enabled and missing is None)


@pytest.mark.parametrize('loaded,building,error,expected', [
    (False, False, None, 'not_loaded'), (False, True, None, 'loading'),
    (True, True, None, 'loading'), (False, False, '/private/model: secret', 'error'),
    (True, False, '/private/model: secret', 'error'),
    (True, True, 'previous failure', 'loading'),
])
def test_unprepared_states_have_unknown_counts(client, environment, loaded, building, error, expected):
    environment.state.update(pack=environment.pack if loaded else None, building=building, error=error)
    response = client.get(URL)
    assert response.status_code == 200
    assert response.json['index'] == {'state': expected, 'indexed_tracks': None}
    assert response.json['ready'] is False
    assert 'secret' not in response.text
    environment.cursor.execute.assert_not_called()


def test_loaded_empty_index(client, environment):
    environment.state['pack'] = SimpleNamespace(build_id='empty', ids=np.array([]), live_tracks=0)
    assert client.get(URL).json['index'] == {'state': 'empty', 'indexed_tracks': 0}
    environment.cursor.execute.assert_not_called()


def test_count_invalidation_and_new_build(client, environment):
    assert client.get(URL).json['index']['indexed_tracks'] == 2
    environment.cursor.fetchall.side_effect = lambda: []
    assert client.get(URL).json['index']['indexed_tracks'] == 2
    nfi.invalidate_availability_cache('primary')
    assert client.get(URL).json['index']['indexed_tracks'] == 1
    environment.state['pack'] = SimpleNamespace(build_id='build-2', ids=np.array(['fp_a']), live_tracks=1)
    assert client.get(URL).json['index'] == {'state': 'empty', 'indexed_tracks': 0}


def test_count_cache_expires(client, environment, monkeypatch):
    assert client.get(URL).json['index']['indexed_tracks'] == 2
    environment.cursor.fetchall.side_effect = lambda: []
    monkeypatch.setattr(environment.cache, '_ttl', 0)
    assert client.get(URL).json['index']['indexed_tracks'] == 1


def test_no_selected_source_never_reports_union(client, monkeypatch):
    monkeypatch.setattr(registry, 'get_default_server_id', lambda: None)
    response = client.get(URL)
    assert response.status_code == 500
    assert response.json == {'error': 'Could not determine recording search status.'}
    assert response.headers['Cache-Control'] == 'no-store'


def test_lookup_failure_is_sanitized_not_empty_or_union(client, environment):
    environment.cursor.execute.side_effect = RuntimeError('secret database password')
    response = client.get(URL)
    assert response.status_code == 500
    assert response.json == {'error': 'Could not determine recording search status.'}
    assert response.headers['Cache-Control'] == 'no-store'


def test_invalid_selection(client):
    response = client.get(URL + '?server_id=private-unknown-name')
    assert response.status_code == 400
    assert response.json == {'error': 'Invalid server selection.'}
    assert response.headers['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('auth,expected', [('user', 200), ('bearer', 200), ('none', 401), ('bad', 401)])
def test_real_auth_barrier_allows_ordinary_users(client, monkeypatch, auth, expected):
    monkeypatch.setattr(config, 'AUTH_ENABLED', True)
    monkeypatch.setattr(app_auth, 'get_session_user', lambda name: {
        'username': name, 'role': 'user', 'password_changed_at': None,
    })
    headers = {}
    if auth == 'user':
        now = datetime.datetime.now(datetime.timezone.utc)
        token = jwt.encode({'sub': 'alice', 'role': 'user', 'iat': now,
                            'exp': now + datetime.timedelta(hours=1)}, SECRET, algorithm='HS256')
        client.set_cookie('audiomuse_jwt', token)
    elif auth in ('bearer', 'bad'):
        headers['Authorization'] = 'Bearer ' + ('unit-api-token' if auth == 'bearer' else 'wrong')
    response = client.get(URL, headers=headers)
    assert response.status_code == expected
    assert response.headers['Cache-Control'] == 'no-store'


def test_initial_setup_policy_is_unchanged(client, monkeypatch):
    monkeypatch.setattr(app_auth, 'check_setup_needed', lambda: True)
    response = client.get(URL)
    assert response.status_code == 403
    assert response.json == {'error': 'Setup required'}
    assert response.headers['Cache-Control'] == 'no-store'
