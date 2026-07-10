# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint routes for the provider-migration wizard.

Drives the migration endpoints with a test client and fake DB, asserting the
session state machine, source-path override handling and SSRF/confirmation gates.

Main Features:
* Session start creates a row and rejects unknown target types
* Source-path override refresh stores overrides and warns on non-absolute paths
* Dry-run gate returns 409 on bad source paths unless overridden or bypassed
* Execute gate requires backup confirmation and dry-run-ready state; probe URLs SSRF-validated
"""

import os
import sys
import importlib.util
import pytest
from unittest.mock import MagicMock, patch


def _load_bp_module():
    mod_name = 'app_provider_migration'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'app_provider_migration.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bp_mod():
    return _load_bp_module()


@pytest.fixture
def app(bp_mod):
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(bp_mod.migration_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def fake_db(bp_mod):
    cur = MagicMock()
    cur.__enter__ = lambda self: self
    cur.__exit__ = lambda self, *a: None
    cur._fetchone_queue = []
    cur.fetchone.side_effect = lambda: cur._fetchone_queue.pop(0) if cur._fetchone_queue else None

    db = MagicMock()
    db.cursor.return_value = cur
    db.commit = MagicMock()

    bp_mod.get_db = MagicMock(return_value=db)
    return db, cur


class TestMigrationPageRoute:
    def test_renders_with_layout(self, bp_mod, client):
        with patch.object(bp_mod, 'render_template', return_value='<html>ok</html>') as mock_rt:
            resp = client.get('/provider-migration')
        assert resp.status_code == 200
        assert mock_rt.called
        kwargs = mock_rt.call_args[1]
        assert kwargs.get('active') == 'provider_migration'


class TestSessionStart:
    def test_creates_session_row(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.append((123,))
        import config

        config.MEDIASERVER_TYPE = 'jellyfin'

        resp = client.post(
            '/api/migration/session/start',
            json={
                'target_type': 'navidrome',
                'target_creds': {'url': 'http://127.0.0.1', 'user': 'u', 'password': 'p'},
            },
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['session_id'] == 123
        sqls = [c[0][0] for c in cur.execute.call_args_list]
        assert any('INSERT INTO migration_session' in s for s in sqls)

    def test_rejects_unknown_target_type(self, bp_mod, client, fake_db):
        resp = client.post(
            '/api/migration/session/start',
            json={
                'target_type': 'bogus',
                'target_creds': {},
            },
        )
        assert resp.status_code == 400


class TestProbeTest:
    def test_calls_provider_probe_and_returns_shape(self, bp_mod, client):
        fake = {
            'ok': True,
            'error': None,
            'sample_count': 5,
            'path_format': 'absolute',
            'warnings': [],
        }
        with patch.object(bp_mod, 'provider_probe', MagicMock()) as p:
            p.test_connection.return_value = fake
            resp = client.post(
                '/api/migration/probe/test',
                json={
                    'type': 'navidrome',
                    'creds': {'url': 'http://127.0.0.1', 'user': 'u', 'password': 'p'},
                },
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['ok'] is True
        assert data['path_format'] == 'absolute'


class TestApplySourcePathOverrides:
    def test_noop_when_overrides_empty(self, bp_mod):
        rows = [{'item_id': 'a', 'file_path': '/old/a.mp3'}]
        bp_mod._apply_source_path_overrides(rows, {})
        assert rows[0]['file_path'] == '/old/a.mp3'

    def test_patches_matching_ids_only(self, bp_mod):
        rows = [
            {'item_id': 'a', 'file_path': ''},
            {'item_id': 'b', 'file_path': ''},
            {'item_id': 'c', 'file_path': '/unchanged/c.mp3'},
        ]
        bp_mod._apply_source_path_overrides(
            rows,
            {
                'a': '/music/a.mp3',
                'b': '/music/b.mp3',
            },
        )
        assert rows[0]['file_path'] == '/music/a.mp3'
        assert rows[1]['file_path'] == '/music/b.mp3'
        assert rows[2]['file_path'] == '/unchanged/c.mp3'

    def test_skips_empty_override_values(self, bp_mod):
        rows = [{'item_id': 'a', 'file_path': '/old/a.mp3'}]
        bp_mod._apply_source_path_overrides(rows, {'a': None})
        assert rows[0]['file_path'] == '/old/a.mp3'


class TestSourcePathsRefreshRoute:
    def test_stores_overrides_in_session_state(self, bp_mod):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        config.NAVIDROME_URL = 'http://nav'
        config.NAVIDROME_USER = 'u'
        config.NAVIDROME_PASSWORD = 'p'

        fake_tracks = [
            {'id': 't1', 'path': '/music/rock/a.mp3'},
            {'id': 't2', 'path': '/music/rock/b.mp3'},
            {'id': 't3', 'path': None},
        ]
        with (
            patch.object(bp_mod, 'provider_probe', MagicMock()) as p,
            patch.object(bp_mod, '_detect_path_format') as mock_detect,
            patch.object(bp_mod, '_update_state') as mock_update,
        ):
            p.fetch_all_tracks.return_value = fake_tracks
            mock_detect.return_value = 'absolute'
            data = bp_mod.run_source_refresh_core(7)

        assert data['ok'] is True
        assert data['source_type'] == 'navidrome'
        assert data['path_format'] == 'absolute'
        assert data['overrides_count'] == 2

        mock_update.assert_called_once()
        call_kwargs = mock_update.call_args.kwargs
        assert call_kwargs['source_path_overrides'] == {
            't1': '/music/rock/a.mp3',
            't2': '/music/rock/b.mp3',
        }

    def test_returns_warning_when_still_not_absolute(self, bp_mod):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        config.NAVIDROME_URL = 'http://nav'
        config.NAVIDROME_USER = 'u'
        config.NAVIDROME_PASSWORD = 'p'

        with (
            patch.object(bp_mod, 'provider_probe', MagicMock()) as p,
            patch.object(bp_mod, '_detect_path_format') as mock_detect,
            patch.object(bp_mod, '_update_state'),
        ):
            p.fetch_all_tracks.return_value = [{'id': 't1', 'path': 'relative/path.mp3'}]
            mock_detect.return_value = 'relative'
            data = bp_mod.run_source_refresh_core(1)

        assert data['path_format'] == 'relative'
        assert data['warnings']
        assert 'report real path' in data['warnings'][0].lower()

    def test_enqueues_job_for_supported_provider(self, bp_mod, client):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        config.NAVIDROME_URL = 'http://nav'
        config.NAVIDROME_USER = 'u'
        config.NAVIDROME_PASSWORD = 'p'
        fake_queue = MagicMock()
        fake_job = MagicMock()
        fake_job.id = 'src-job-1'
        fake_queue.enqueue.return_value = fake_job
        with (
            patch.object(bp_mod, '_patch_state_keys'),
            patch.object(bp_mod, 'rq_queue_high', fake_queue),
        ):
            resp = client.post('/api/migration/source-paths/refresh', json={'session_id': 5})
        assert resp.status_code == 200
        assert resp.get_json().get('task_id') == 'src-job-1'
        assert fake_queue.enqueue.called

    def test_rejects_unsupported_current_provider(self, bp_mod, client):
        import config

        config.MEDIASERVER_TYPE = 'spotify'
        resp = client.post('/api/migration/source-paths/refresh', json={'session_id': 1})
        assert resp.status_code == 400

    def test_requires_session_id(self, bp_mod, client):
        resp = client.post('/api/migration/source-paths/refresh', json={})
        assert resp.status_code == 400


class TestDryRunSourcePathGate:
    def test_returns_409_when_source_paths_bad_and_no_overrides(self, bp_mod, client):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        config.NAVIDROME_URL = 'http://nav'
        config.NAVIDROME_USER = 'u'
        config.NAVIDROME_PASSWORD = 'p'

        with (
            patch.object(
                bp_mod, '_fetch_session_creds', return_value=('jellyfin', {'url': 'http://jf'})
            ),
            patch.object(bp_mod, '_load_state', return_value={}),
            patch.object(bp_mod, '_detect_source_path_format', return_value='none'),
        ):
            resp = client.post('/api/migration/dry-run', json={'session_id': 1})

        assert resp.status_code == 409
        data = resp.get_json()
        assert data['needs_source_refresh'] is True
        assert data['current_source_type'] == 'navidrome'
        assert data['path_format'] == 'none'

    def test_bypass_flag_skips_gate(self, bp_mod, client):
        fake_queue = MagicMock()
        fake_job = MagicMock()
        fake_job.id = 'dry-job-1'
        fake_queue.enqueue.return_value = fake_job
        with (
            patch.object(
                bp_mod, '_fetch_session_creds', return_value=('jellyfin', {'url': 'http://jf'})
            ),
            patch.object(bp_mod, '_load_state', return_value={}),
            patch.object(bp_mod, '_detect_source_path_format', return_value='none'),
            patch.object(bp_mod, '_patch_state_keys'),
            patch.object(bp_mod, 'rq_queue_high', fake_queue),
        ):
            resp = client.post(
                '/api/migration/dry-run', json={'session_id': 1, 'bypass_source_check': True}
            )

        assert resp.status_code == 200
        assert resp.get_json().get('task_id') == 'dry-job-1'
        assert fake_queue.enqueue.called

    def test_overrides_present_skip_gate_and_apply_to_rows(self, bp_mod):
        old_rows = [
            {
                'item_id': 'a',
                'file_path': '',
                'title': 't',
                'author': 'x',
                'album': 'y',
                'album_artist': 'x',
            },
        ]
        overrides = {'a': '/music/real.mp3'}
        fake_matcher = MagicMock()
        fake_matcher.match_tracks.return_value = {
            'matches': {},
            'match_tiers': {},
            'tier_counts': {},
            'unmatched': [],
            'unmatched_by_album': {},
        }
        with (
            patch.object(bp_mod, '_fetch_session_creds', return_value=('jellyfin', {})),
            patch.object(bp_mod, '_load_state', return_value={'source_path_overrides': overrides}),
            patch.object(bp_mod, '_load_score_rows_as_dicts', return_value=old_rows),
            patch.object(bp_mod, 'provider_probe', MagicMock()) as p,
            patch.object(bp_mod, '_store_target_meta'),
            patch.object(bp_mod, '_albums_payload', return_value=[]),
            patch.object(bp_mod, '_update_state'),
            patch('importlib.import_module', return_value=fake_matcher),
        ):
            p.fetch_all_tracks.return_value = [{'id': 'n1', 'path': '/x', 'title': 't'}]
            result = bp_mod.run_dry_run_core(1, allow_title_artist_only=False)

        assert result.get('matched') == 0
        called_old_rows = fake_matcher.match_tracks.call_args[0][0]
        assert called_old_rows[0]['file_path'] == '/music/real.mp3'

    def test_dry_run_zero_tracks_guard_aborts(self, bp_mod):
        with (
            patch.object(bp_mod, '_fetch_session_creds', return_value=('jellyfin', {})),
            patch.object(bp_mod, 'provider_probe', MagicMock()) as p,
            patch.object(bp_mod, '_patch_state_keys'),
            patch.object(bp_mod, '_load_score_rows_as_dicts') as mock_load,
            patch.object(bp_mod, '_store_target_meta') as mock_store,
        ):
            p.fetch_all_tracks.return_value = []
            result = bp_mod.run_dry_run_core(1)

        assert 'error' in result and '0 tracks' in result['error']
        mock_load.assert_not_called()
        mock_store.assert_not_called()


class TestExecuteGate:
    def _base_payload(self, target='navidrome'):
        return {
            'session_id': 1,
            'backup_confirmed': True,
            'confirmation_text': f'I want to migrate to {target} and delete unmatched tracks',
        }

    def test_rejects_missing_backup_confirmation(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.append(('navidrome', 'dry_run_ready'))
        p = self._base_payload()
        p['backup_confirmed'] = False
        resp = client.post('/api/migration/execute', json=p)
        assert resp.status_code == 400
        assert 'backup' in resp.get_json().get('error', '').lower()

    def test_rejects_wrong_confirmation_text(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.append(('navidrome', 'dry_run_ready'))
        p = self._base_payload()
        p['confirmation_text'] = 'LGTM ship it'
        resp = client.post('/api/migration/execute', json=p)
        assert resp.status_code == 400
        assert 'confirm' in resp.get_json().get('error', '').lower()

    def test_rejects_session_not_in_dry_run_ready(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.append(('navidrome', 'in_progress'))
        resp = client.post('/api/migration/execute', json=self._base_payload())
        assert resp.status_code == 400
        err = resp.get_json().get('error', '').lower()
        assert 'dry' in err or 'status' in err

    def test_happy_path_enqueues_job(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.append(('navidrome', 'dry_run_ready'))
        fake_queue = MagicMock()
        fake_job = MagicMock()
        fake_job.id = 'job-xyz'
        fake_queue.enqueue.return_value = fake_job
        bp_mod.rq_queue_high = fake_queue

        resp = client.post('/api/migration/execute', json=self._base_payload())

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['task_id'] == 'job-xyz'
        assert fake_queue.enqueue.called


class TestProbeUrlValidation:
    ACCEPTED = [
        'http://127.0.0.1',
        'http://127.0.0.1:8096',
        'http://192.168.1.50:8096',
        'http://10.0.0.5/rest',
        'http://172.16.3.4',
        'https://8.8.8.8',
        'http://1.2.3.4:8096',
    ]

    REJECTED = [
        'http://169.254.169.254/latest/meta-data',
        'http://169.254.10.20',
        'http://0.0.0.0',
        'http://224.0.0.1',
        'http://',
        'not-a-url',
        'file:///etc/passwd',
        'gopher://10.0.0.1:6379/',
        'ftp://1.2.3.4/',
        'redis://1.2.3.4:6379',
    ]

    @pytest.mark.parametrize('url', ACCEPTED)
    def test_accepts_safe_urls(self, bp_mod, url):
        ok, reason = bp_mod._validate_probe_url({'url': url})
        assert ok is True, f'{url!r} should be accepted (reason={reason!r})'
        assert reason is None

    @pytest.mark.parametrize('url', REJECTED)
    def test_rejects_unsafe_urls(self, bp_mod, url):
        ok, reason = bp_mod._validate_probe_url({'url': url})
        assert ok is False, f'{url!r} should be rejected'
        assert isinstance(reason, str) and reason

    @pytest.mark.parametrize('creds', [{}, {'url': ''}, {'url': None}])
    def test_missing_url_is_allowed(self, bp_mod, creds):
        ok, reason = bp_mod._validate_probe_url(creds)
        assert ok is True
        assert reason is None

    def test_probe_endpoint_rejects_metadata_url(self, bp_mod, client):
        with patch.object(bp_mod, 'provider_probe', MagicMock()) as p:
            resp = client.post(
                '/api/migration/probe/test',
                json={
                    'type': 'navidrome',
                    'creds': {'url': 'http://169.254.169.254/'},
                },
            )
        assert resp.status_code == 200
        assert resp.get_json()['ok'] is False
        assert not p.test_connection.called

    def test_session_start_rejects_disallowed_scheme(self, client):
        resp = client.post(
            '/api/migration/session/start',
            json={
                'target_type': 'navidrome',
                'target_creds': {'url': 'file:///etc/passwd'},
            },
        )
        assert resp.status_code == 400
        assert 'not allowed' in resp.get_json().get('error', '').lower()
