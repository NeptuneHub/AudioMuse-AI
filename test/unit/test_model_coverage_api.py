# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""All-model coverage and separate version API contracts.

Main Features:
* Checks all four models, effective enablement and global/local percentages.
* Exercises source masks and paged directory decoding without model warmup.
* Verifies auth, setup, version isolation, unknown counts and cache invalidation.
"""

import ast
import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import jwt
import numpy as np
import pytest
from flask import Flask, jsonify

import app_auth
import app_models
import config
from tasks import neural_fingerprint as nf
from tasks import neural_fingerprint_index as nfi
from tasks import paged_ivf
from tasks import recording_search_manager as manager
from tasks.index_availability import AvailabilityCache
from tasks.mediaserver import registry

URL = '/api/models'
SECRET = 'model-coverage-unit-secret-at-least-32-bytes'


@pytest.fixture
def environment(monkeypatch, tmp_path):
    files = [tmp_path / 'encoder.onnx', tmp_path / 'codebook.npz']
    for path in files:
        path.touch()
    for key in ('NEURAL_FINGERPRINT_ENABLED', 'CLAP_ENABLED', 'LYRICS_ENABLED'):
        monkeypatch.setattr(config, key, True)
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
    for module, attr in ((nfi, '_STATUS_COUNTS'), (paged_ivf, '_SCOPED_COUNTS')):
        cache = AvailabilityCache()
        monkeypatch.setattr(cache, '_arm_idle_drop', lambda: None)
        monkeypatch.setattr(module, attr, cache)

    import database
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    selected = {'sql': '', 'params': None}
    totals = {'global': 4, 'primary': 2, 'secondary': 1}
    mapped = {'primary': [('fp_a',)], 'secondary': [('fp_b',)]}

    def execute(sql, params=None):
        assert sql.startswith('SELECT ')
        selected.update(sql=sql, params=params)

    def fetchone():
        sql, params = selected['sql'], selected['params']
        if 'lyrics_embedding' in sql:
            return (2,)
        if 'COUNT(*)' in sql:
            if params:
                assert 'EXISTS' in sql
                assert 'left(s.item_id, 3)' in sql
                assert params[1] is (params[0] == 'primary')
            return (totals[params[0] if params else 'global'],)
        return (params[0] == 'primary', None)

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = fetchone
    cursor.fetchall.side_effect = lambda: mapped[selected['params'][0]]
    monkeypatch.setattr(database, 'get_db', lambda: connection)
    index_ids = {config.INDEX_NAME: ['fp_a', 'fp_b', 'legacy'], 'clap_index': ['fp_b', 'legacy'], 'lyrics_index': ['fp_a']}
    monkeypatch.setattr(paged_ivf, 'paged_ivf_item_count', lambda db, name: len(index_ids[name]))
    from tasks import index_build_helpers

    def directory(db, table, name):
        assert table == 'ivf_dir'
        ids = index_ids[name.removesuffix('__ivf_dir')]
        return paged_ivf.pack_directory(np.zeros((1, 2), dtype=np.float32),
                                       np.zeros(len(ids), dtype=np.uint32), ids, 2, 'angular')

    loader = MagicMock(side_effect=directory)
    monkeypatch.setattr(index_build_helpers, 'load_segmented_blob', loader)

    def forbidden(*args, **kwargs):
        pytest.fail('Metadata checks must not load search indexes/models, infer, or renew warmup')

    for module, names in (
        (nf, ('warm_session', 'fingerprint_audio', 'codebook')),
        (nfi, ('ensure_loaded', 'start_background_load', 'reload_from_db', 'identify')),
        (manager, ('warmup_recording_models', '_arm_idle_unload', 'run_recording_search')),
        (paged_ivf, ('load_paged_ivf_index', 'load_index_auto', '_setup_disk_cell_file')),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(manager._TIMER, 'arm', forbidden)
    return SimpleNamespace(state=state, files=files, cursor=cursor, pack=pack,
                           loader=loader, totals=totals, mapped=mapped, db=connection)


@pytest.fixture
def client(environment):
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.before_request(app_auth.auth_setup_barrier)
    app.register_blueprint(app_models.models_bp)
    # Execute the actual app.py route in an isolated Flask app; importing the
    # whole server would also start unrelated process/bootstrap infrastructure.
    source = Path(__file__).resolve().parents[2] / 'app.py'
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'version_api')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'),
         {'app': app, 'config': config, 'jsonify': jsonify})
    return app.test_client()


def test_global_only_contract_uses_whole_catalogue_even_for_lyrics(client, environment):
    response = client.get(URL)
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store'
    assert set(response.json) == {'models'}
    models = response.json['models']
    assert set(models) == {'musicnn', 'clap', 'lyrics', 'neural-fingerprint'}
    for model, count in (('musicnn', 3), ('clap', 2), ('lyrics', 1), ('neural-fingerprint', 3)):
        assert models[model] == {'enabled': True, 'global': {'count': count, 'total': 4, 'percentage': count * 25.0}}
    environment.loader.assert_not_called()


@pytest.mark.parametrize('query,server,counts', [
    ('?server_id=primary', 'primary', [2, 1, 1, 2]),
    ('?server_id=secondary', 'secondary', [1, 1, 0, 1]),
    ('?server=Other%20library', 'secondary', [1, 1, 0, 1]),
    ('?server=secondary&server_id=primary', 'secondary', [1, 1, 0, 1]),
])
def test_explicit_scope_keeps_global_and_adds_local(client, query, server, counts):
    response = client.get(URL + query)
    assert response.status_code == 200
    body = response.json
    assert body['server_id'] == server
    total = 2 if server == 'primary' else 1
    for name, count in zip(('musicnn', 'clap', 'lyrics', 'neural-fingerprint'), counts):
        assert body['models'][name]['global']['total'] == 4
        assert body['models'][name]['local'] == {'count': count, 'total': total, 'percentage': count * 100.0 / total}


def test_empty_server_is_not_the_union(client, environment):
    environment.totals['secondary'] = 0
    environment.mapped['secondary'] = []
    response = client.get(URL + '?server_id=secondary')
    assert response.status_code == 200
    for model in response.json['models'].values():
        assert model['local'] == {'count': 0, 'total': 0, 'percentage': 0.0}


def test_runtime_enablement_and_version_are_separate(client, monkeypatch):
    monkeypatch.setattr(config, 'APP_VERSION', '3.6.1-rc.2')
    for key in ('CLAP_ENABLED', 'LYRICS_ENABLED', 'NEURAL_FINGERPRINT_ENABLED'):
        monkeypatch.setattr(config, key, False)
    body = client.get(URL).json
    assert body['models']['musicnn']['enabled'] is True
    assert all(not body['models'][key]['enabled'] for key in ('clap', 'lyrics', 'neural-fingerprint'))
    assert 'recording' not in body
    assert 'api_version' not in body
    assert 'app_version' not in body
    response = client.get('/api/version')
    assert response.json == {'app_version': '3.6.1-rc.2'}
    assert response.headers['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('enabled,missing', [(False, None), (True, 0), (True, 1), (True, None)])
def test_model_files_availability_split_is_preserved(environment, monkeypatch, enabled, missing):
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', enabled)
    if missing is not None:
        environment.files[missing].unlink()
    assert nf.model_files_available() is (missing is None)
    assert nf.is_available() is (enabled and missing is None)


@pytest.mark.parametrize('loaded,building,error,expected', [
    (False, False, None, 'not_loaded'), (False, True, None, 'loading'),
    (True, True, None, 'loading'), (False, False, 'secret', 'error'),
    (True, False, 'secret', 'error'),
])
def test_neural_unknown_coverage_stays_null(client, environment, loaded, building, error, expected):
    environment.state.update(pack=environment.pack if loaded else None, building=building, error=error)
    response = client.get(URL + '?server_id=primary')
    assert response.status_code == 200
    assert response.json['models']['neural-fingerprint']['local'] == {'count': None, 'total': 2, 'percentage': None}
    assert nfi.get_scoped_status('primary')['state'] == expected
    assert 'secret' not in response.text
    if not loaded:
        assert response.json['models']['neural-fingerprint']['global']['count'] is None


def test_unknown_global_index_is_not_zero(client, monkeypatch):
    monkeypatch.setattr(paged_ivf, 'paged_ivf_item_count', lambda db, name: None)
    assert client.get(URL).json['models']['clap']['global'] == {'count': None, 'total': 4, 'percentage': None}


def test_percentage_rounding_zero_total_and_stale_index():
    from tasks.model_coverage import _coverage
    assert _coverage(1, 3)['percentage'] == 33.33
    assert _coverage(5, 4) == {'count': 5, 'total': 4, 'percentage': 100.0}
    assert _coverage(0, 0)['percentage'] == 0.0
    assert _coverage(None, 0)['percentage'] is None


def test_scoped_counts_reuse_cache_and_invalidate(client, environment):
    query = URL + '?server_id=primary'
    assert client.get(query).json['models']['musicnn']['local']['count'] == 2
    assert environment.loader.call_count == 3
    environment.mapped['primary'] = []
    assert client.get(query).json['models']['musicnn']['local']['count'] == 2
    assert environment.loader.call_count == 3
    paged_ivf.invalidate_availability_cache('primary')
    nfi.invalidate_availability_cache('primary')
    body = client.get(query).json
    assert body['models']['musicnn']['local']['count'] == 1
    assert body['models']['neural-fingerprint']['local']['count'] == 1
    assert environment.loader.call_count == 6


def test_missing_paged_directory_is_zero(client, environment):
    environment.loader.side_effect = None
    environment.loader.return_value = None
    assert client.get(URL + '?server_id=secondary').json['models']['clap']['local']['count'] == 0


def test_lookup_failure_is_sanitized(client, environment):
    environment.cursor.execute.side_effect = RuntimeError('secret database password')
    response = client.get(URL)
    assert response.status_code == 500
    assert response.json == {'error': 'Could not determine model coverage.'}
    assert response.headers['Cache-Control'] == 'no-store'
    environment.db.rollback.assert_called_once()


@pytest.mark.parametrize('query', ['?server_id=unknown', '?server=', '?server_id='])
def test_invalid_scope(client, query):
    response = client.get(URL + query)
    assert response.status_code == 400
    assert response.json == {'error': 'Invalid server selection.'}
    assert response.headers['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('route', [URL, '/api/version'])
@pytest.mark.parametrize('auth,expected', [('user', 200), ('bearer', 200), ('none', 401), ('bad', 401)])
def test_auth_barrier_allows_ordinary_users(client, monkeypatch, route, auth, expected):
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
    response = client.get(route, headers=headers)
    assert response.status_code == expected
    assert response.headers['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('route', [URL, '/api/version'])
def test_initial_setup_policy(client, monkeypatch, route):
    monkeypatch.setattr(app_auth, 'check_setup_needed', lambda: True)
    response = client.get(route)
    assert response.status_code == 403
    assert response.json == {'error': 'Setup required'}
    assert response.headers['Cache-Control'] == 'no-store'


def test_index_replacement_invalidates_scoped_counts(client, environment):
    url = URL + '?server_id=primary'
    client.get(url)
    assert environment.loader.call_count == 3
    paged_ivf.invalidate_global_cell_cache(config.INDEX_NAME)
    client.get(url)
    assert environment.loader.call_count == 6


def test_scoped_count_ttl(client, environment, monkeypatch):
    url = URL + '?server_id=primary'
    client.get(url)
    monkeypatch.setattr(paged_ivf._SCOPED_COUNTS, '_ttl', 0)
    client.get(url)
    assert environment.loader.call_count == 6


def test_directory_failure_is_not_reported_as_empty(client, environment):
    environment.loader.side_effect = ValueError('secret corrupt directory path')
    response = client.get(URL + '?server_id=secondary')
    assert response.status_code == 500
    assert response.json == {'error': 'Could not determine model coverage.'}


def test_generated_openapi_references_resolve(client):
    from flasgger import Swagger
    client.application.config['SWAGGER'] = {'title': 'Models', 'openapi': '3.0.0'}
    Swagger(client.application)
    response = client.get('/apispec_1.json')
    assert response.status_code == 200
    spec = response.json
    assert '/api/models' in spec['paths']
    assert '/api/version' in spec['paths']

    def check_refs(value):
        if isinstance(value, dict):
            if '$ref' in value:
                ref = value['$ref']
                assert ref.startswith('#/')
                target = spec
                for key in ref[2:].split('/'):
                    target = target[key]
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    check_refs(spec)
