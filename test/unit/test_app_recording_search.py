# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Recording search blueprint: request validation and error mapping.

Main Features:
* A request without a clip or with a bad count answers 400
* A clip over RECORDING_SEARCH_MAX_UPLOAD_MB answers 413
* The manager's ValueError answers 400 with its message, RuntimeError 503,
  any other exception a generic 500 that carries no internal detail
* A successful search hands the clip to the manager as a stream (never read
  into memory by the blueprint), passes the count through, scopes the results
  to the requested count and reports the count
* The count defaults to the config default
* The by_track endpoint answers 400 without a song, with a bad count or an
  unknown song, passes the canonical id through, echoes the song id as it was
  sent (never the canonical one), scopes the results and maps the manager
  errors like the clip search
* The warmup endpoint relays the manager status
* The page hands the template the built-in HTTPS port, or zero when the
  listener is disabled, and the reason when it could not start
"""

import importlib.util
import io
import os
import sys

import pytest


def _load_bp_module():
    mod_name = 'app_recording_search'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'app_recording_search.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bp_mod():
    return _load_bp_module()


@pytest.fixture
def client(bp_mod):
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(bp_mod.recording_search_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture(autouse=True)
def neutral_server_scope(monkeypatch):
    import app_helper
    import app_server_context

    monkeypatch.setattr(app_server_context, 'resolve_request_server_id', lambda data=None: None)
    monkeypatch.setattr(
        app_server_context,
        'scope_results',
        lambda rows, requested_n=None, id_key='item_id', translate=True: (
            rows[:requested_n] if requested_n else rows
        ),
    )
    monkeypatch.setattr(app_helper, 'attach_song_features', lambda rows, id_key='item_id': rows)


def _post(client, data):
    return client.post('/api/recording_search/search', data=data, content_type='multipart/form-data')


def _clip(name='a.wav'):
    return (io.BytesIO(b'abc'), name)


def _patch_manager(monkeypatch, func):
    import tasks.recording_search_manager as rsm

    monkeypatch.setattr(rsm, 'run_recording_search', func)


def _render_page(bp_mod, monkeypatch, status):
    import tls_listener
    import tasks.recording_search_manager as rsm

    captured = {}
    monkeypatch.setattr(tls_listener, 'https_status', lambda: status)
    monkeypatch.setattr(rsm, 'get_index_status', lambda: {'neural': 'ready'})
    monkeypatch.setattr(bp_mod, 'render_template', lambda name, **context: captured.update(context) or 'page')
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(bp_mod.recording_search_bp)
    assert app.test_client().get('/recording_search').status_code == 200
    return captured


def test_the_page_carries_the_built_in_https_port_or_zero_and_the_start_error(bp_mod, monkeypatch):
    running = _render_page(bp_mod, monkeypatch, {'enabled': True, 'running': True, 'port': 8443, 'error': None})
    assert running['https_port'] == 8443
    assert running['https_error'] == ''
    assert running['index_status'] == {'neural': 'ready'}
    disabled = _render_page(bp_mod, monkeypatch, {'enabled': False, 'running': False, 'port': 0, 'error': 'disabled by FLASK_HTTPS_PORT'})
    assert disabled['https_port'] == 0
    assert disabled['https_error'] == 'disabled by FLASK_HTTPS_PORT'
    failed = _render_page(bp_mod, monkeypatch, {'enabled': True, 'running': False, 'port': 8443, 'error': 'address already in use'})
    assert failed['https_port'] == 8443
    assert failed['https_error'] == 'address already in use'


def test_missing_clip_answers_400(client):
    response = _post(client, {'n_results': '5'})
    assert response.status_code == 400
    assert 'clip' in response.get_json()['error']


def test_bad_count_answers_400(client):
    response = _post(client, {'clip': _clip(), 'n_results': 'ten'})
    assert response.status_code == 400


def test_oversized_clip_answers_413(client, monkeypatch):
    import config

    monkeypatch.setattr(config, 'RECORDING_SEARCH_MAX_UPLOAD_MB', 0)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 413


def test_manager_value_error_answers_400_with_its_message(client, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError('The clip is silent.')

    _patch_manager(monkeypatch, boom)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'The clip is silent.'


def test_manager_runtime_error_answers_503(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('No neural fingerprints are stored yet. Run analysis first.')

    _patch_manager(monkeypatch, boom)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 503
    assert 'Run analysis first' in response.get_json()['error']


def test_unexpected_error_answers_generic_500_without_detail(client, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyError('secret-internal-detail')

    _patch_manager(monkeypatch, boom)
    response = _post(client, {'clip': _clip()})
    assert response.status_code == 500
    assert 'secret' not in response.get_json()['error']


def test_success_passes_the_clip_through_scopes_results_and_reports_count(client, monkeypatch):
    seen = {}

    def fake(clip, filename, n_results):
        seen.update(file_bytes=clip.read(), filename=filename, n_results=n_results)
        return {
            'clip_seconds': 20.0,
            'clip_level_db': -14.0,
            'results': [
                {'item_id': 'a', 'title': 'A', 'author': 'x', 'score': 0.9},
                {'item_id': 'b', 'title': 'B', 'author': 'y', 'score': 0.8},
            ],
            'count': 2,
        }

    _patch_manager(monkeypatch, fake)
    response = _post(client, {'clip': _clip('rec.webm'), 'n_results': '1'})
    body = response.get_json()
    assert response.status_code == 200
    assert seen == {'file_bytes': b'abc', 'filename': 'rec.webm', 'n_results': 1}
    assert body['count'] == 1
    assert [row['item_id'] for row in body['results']] == ['a']


def test_count_defaults_to_the_config_default(client, monkeypatch):
    import config

    seen = {}

    def fake(clip, filename, n_results):
        seen.update(n_results=n_results)
        return {'clip_seconds': 1.0, 'clip_level_db': -14.0, 'results': [], 'count': 0}

    _patch_manager(monkeypatch, fake)
    response = _post(client, {'clip': _clip('rec.webm')})
    assert response.status_code == 200
    assert seen == {'n_results': config.RECORDING_SEARCH_DEFAULT_N_RESULTS}


def _post_track(client, body):
    return client.post('/api/recording_search/by_track', json=body)


def test_by_track_answers_400_without_a_song_or_with_an_unknown_one(client, monkeypatch):
    import app_server_context

    assert _post_track(client, {}).status_code == 400
    assert _post_track(client, {'item_id': 'x', 'n_results': 'ten'}).status_code == 400

    def unknown(raw_id, data=None):
        raise ValueError('unknown song')

    monkeypatch.setattr(app_server_context, 'resolve_input_item_id', unknown)
    response = _post_track(client, {'item_id': 'x'})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'unknown song'


def test_by_track_passes_the_canonical_id_and_scopes_the_results(client, monkeypatch):
    import app_server_context
    import tasks.recording_search_manager as rsm

    seen = {}
    monkeypatch.setattr(app_server_context, 'resolve_input_item_id', lambda raw_id, data=None: f'fp_{raw_id}')

    def fake(item_id, n_results):
        seen.update(item_id=item_id, n_results=n_results)
        return {
            'item_id': item_id,
            'results': [
                {'item_id': 'a', 'title': 'A', 'author': 'x', 'score': 0.9},
                {'item_id': 'b', 'title': 'B', 'author': 'y', 'score': 0.8},
            ],
            'count': 2,
        }

    monkeypatch.setattr(rsm, 'search_by_track', fake)
    response = _post_track(client, {'item_id': '42', 'n_results': 1})
    body = response.get_json()
    assert response.status_code == 200
    assert seen == {'item_id': 'fp_42', 'n_results': 1}
    assert body['item_id'] == '42'
    assert body['count'] == 1
    assert [row['item_id'] for row in body['results']] == ['a']


def test_by_track_maps_the_manager_errors_like_the_clip_search(client, monkeypatch):
    import tasks.recording_search_manager as rsm

    def rejected(item_id, n_results):
        raise ValueError('This song has no neural fingerprint yet.')

    monkeypatch.setattr(rsm, 'search_by_track', rejected)
    response = _post_track(client, {'item_id': 'x'})
    assert response.status_code == 400
    assert 'no neural fingerprint' in response.get_json()['error']

    def down(item_id, n_results):
        raise RuntimeError('No neural fingerprint index is built yet.')

    monkeypatch.setattr(rsm, 'search_by_track', down)
    assert _post_track(client, {'item_id': 'x'}).status_code == 503

    def boom(item_id, n_results):
        raise KeyError('secret')

    monkeypatch.setattr(rsm, 'search_by_track', boom)
    response = _post_track(client, {'item_id': 'x'})
    assert response.status_code == 500
    assert 'secret' not in response.get_json()['error']


def test_warmup_relays_the_manager_status(client, monkeypatch):
    import tasks.recording_search_manager as rsm

    monkeypatch.setattr(
        rsm,
        'warmup_recording_models',
        lambda: {'loaded': True, 'models': {'neural': True}, 'expiry_seconds': 300},
    )
    response = client.post('/api/recording_search/warmup')
    assert response.status_code == 200
    assert response.get_json()['models']['neural'] is True
