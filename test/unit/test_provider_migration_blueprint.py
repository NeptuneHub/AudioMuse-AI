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
from contextlib import contextmanager, nullcontext
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


@pytest.fixture(autouse=True)
def stub_the_start_lock(bp_mod, monkeypatch):
    monkeypatch.setattr(bp_mod, 'main_task_start_lock', nullcontext)


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
    def _start(self, client):
        return client.post(
            '/api/migration/session/start',
            json={
                'target_type': 'navidrome',
                'target_creds': {'url': 'http://127.0.0.1', 'user': 'u', 'password': 'p'},
            },
        )

    def test_creates_session_row(self, bp_mod, client, fake_db):
        db, cur = fake_db
        # First fetchone answers the advisory-lock probe: True means nothing holds
        # the migration lock, so it is safe to prune the older sessions.
        cur._fetchone_queue.append((True,))
        cur._fetchone_queue.append((False,))  # no committed restart handshake
        cur._fetchone_queue.append((123,))
        import config

        config.MEDIASERVER_TYPE = 'jellyfin'

        with patch.object(bp_mod, 'get_active_main_task', return_value=None):
            resp = self._start(client)

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['session_id'] == 123
        sqls = [c[0][0] for c in cur.execute.call_args_list]
        assert any('INSERT INTO migration_session' in s for s in sqls)
        assert any('DELETE FROM migration_session' in s for s in sqls)

    def test_completed_tombstones_are_pruned_by_rq_lifecycle_not_fixed_count(
        self, bp_mod, client, fake_db
    ):
        db, cur = fake_db
        cur._fetchone_queue.append((True,))
        cur._fetchone_queue.append((False,))
        cur._fetchone_queue.append((123,))
        import config

        config.MEDIASERVER_TYPE = 'jellyfin'

        with patch.object(bp_mod, 'get_active_main_task', return_value=None):
            self._start(client)

        delete = next(
            c[0][0] for c in cur.execute.call_args_list
            if 'DELETE FROM migration_session' in c[0][0]
        )
        assert "status <> 'completed'" in delete
        assert 'ORDER BY id DESC LIMIT' not in delete
        assert not hasattr(bp_mod, '_COMPLETED_SESSIONS_KEPT')

    def test_start_is_refused_while_a_migration_is_queued_or_running(
        self, bp_mod, client, fake_db
    ):
        db, cur = fake_db
        import config

        config.MEDIASERVER_TYPE = 'jellyfin'

        with patch.object(
            bp_mod, 'get_active_main_task', return_value={'task_id': 'mig-1'}
        ):
            resp = self._start(client)

        assert resp.status_code == 409
        sqls = [c[0][0] for c in cur.execute.call_args_list]
        assert not any('DELETE FROM migration_session' in s for s in sqls)
        assert not any('INSERT INTO migration_session' in s for s in sqls)

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


class TestOverridesAreRekeyedOntoCatalogueIds:
    """The probe hands back the PROVIDER's ids; the rows these paths patch are
    catalogue rows keyed by the canonical fp_ content id. Stored as probed, the map
    was looked up with ids it could never contain, so "Refresh source paths"
    reported N overrides and changed nothing on every canonicalized install - and
    the matcher fell back to weaker metadata tiers with the stale path.
    """

    def test_provider_ids_are_translated_to_canonical_ids(self, bp_mod, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry, 'canonical_input_ids',
            lambda ids, server_id=None, conn=None: {
                'prov-1': 'fp_3aaa', 'prov-2': 'fp_3bbb',
            },
        )

        out = bp_mod._overrides_by_catalogue_id(
            {'prov-1': '/music/a.flac', 'prov-2': '/music/b.flac'}
        )

        assert out == {'fp_3aaa': '/music/a.flac', 'fp_3bbb': '/music/b.flac'}

        rows = [{'item_id': 'fp_3aaa', 'file_path': '/stale/a.flac'}]
        bp_mod._apply_source_path_overrides(rows, out)
        assert rows[0]['file_path'] == '/music/a.flac', (
            "the refreshed path must actually reach the row the matcher reads"
        )

    def test_a_pre_canonicalization_install_is_unaffected(self, bp_mod, monkeypatch):
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry, 'canonical_input_ids',
            lambda ids, server_id=None, conn=None: {i: i for i in ids},
        )

        out = bp_mod._overrides_by_catalogue_id({'prov-1': '/music/a.flac'})

        assert out == {'prov-1': '/music/a.flac'}

    def test_an_empty_probe_needs_no_registry_call(self, bp_mod):
        assert bp_mod._overrides_by_catalogue_id({}) == {}

    def test_duplicate_files_of_one_song_collapse_deterministically(
        self, bp_mod, monkeypatch
    ):
        """Two files of one song share a canonical id. Left to dict order the
        provider's listing order decides which path survives, so the refreshed path
        could flip between runs for no reason the user can see."""
        from tasks.mediaserver import registry

        monkeypatch.setattr(
            registry, 'canonical_input_ids',
            lambda ids, server_id=None, conn=None: {i: 'fp_3same' for i in ids},
        )

        forward = bp_mod._overrides_by_catalogue_id({
            'prov-a': '/music/a.flac',
            'prov-b': '/music/b.flac',
        })
        reversed_order = bp_mod._overrides_by_catalogue_id({
            'prov-b': '/music/b.flac',
            'prov-a': '/music/a.flac',
        })

        assert forward == reversed_order == {'fp_3same': '/music/a.flac'}, (
            "the lowest provider id wins, whatever order the provider listed them in"
        )


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
            patch.object(bp_mod, '_fetch_session_creds', return_value=('navidrome', {})),
            patch.object(bp_mod, 'provider_probe', MagicMock()) as p,
            patch.object(bp_mod, '_detect_path_format') as mock_detect,
            patch.object(bp_mod, '_patch_state_keys') as mock_patch,
        ):
            p.fetch_all_tracks.return_value = fake_tracks
            mock_detect.return_value = 'absolute'
            data = bp_mod.run_source_refresh_core(7)

        assert data['ok'] is True
        assert data['source_type'] == 'navidrome'
        assert data['path_format'] == 'absolute'
        assert data['overrides_count'] == 2

        mock_patch.assert_called_once()
        call_kwargs = mock_patch.call_args.kwargs
        assert call_kwargs['source_path_overrides'] == {
            't1': '/music/rock/a.mp3',
            't2': '/music/rock/b.mp3',
        }

    def test_persisting_overrides_does_not_touch_the_session_status(self, bp_mod):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        config.NAVIDROME_URL = 'http://nav'
        config.NAVIDROME_USER = 'u'
        config.NAVIDROME_PASSWORD = 'p'

        with (
            patch.object(bp_mod, '_fetch_session_creds', return_value=('navidrome', {})),
            patch.object(bp_mod, 'provider_probe', MagicMock()) as p,
            patch.object(bp_mod, '_detect_path_format') as mock_detect,
            patch.object(bp_mod, '_patch_state_keys') as mock_patch,
        ):
            p.fetch_all_tracks.return_value = [{'id': 't1', 'path': '/music/a.mp3'}]
            mock_detect.return_value = 'absolute'
            bp_mod.run_source_refresh_core(7)

        assert '_set_status' not in mock_patch.call_args.kwargs, (
            "the refresh only fills in overrides; flagging in_progress here demoted a "
            "finalized dry_run_ready session and Execute then refused the migration"
        )

    def test_no_status_update_reaches_the_database_on_refresh(self, bp_mod, fake_db):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        config.NAVIDROME_URL = 'http://nav'
        config.NAVIDROME_USER = 'u'
        config.NAVIDROME_PASSWORD = 'p'
        db, cur = fake_db

        with (
            patch.object(bp_mod, '_fetch_session_creds', return_value=('navidrome', {})),
            patch.object(bp_mod, 'provider_probe', MagicMock()) as p,
            patch.object(bp_mod, '_detect_path_format') as mock_detect,
        ):
            p.fetch_all_tracks.return_value = [{'id': 't1', 'path': '/music/a.mp3'}]
            mock_detect.return_value = 'absolute'
            bp_mod.run_source_refresh_core(7)

        statements = [" ".join(c[0][0].split()) for c in cur.execute.call_args_list]
        assert statements, "the refresh must still persist its overrides"
        assert not [s for s in statements if 'SET status' in s], (
            f"a refresh must never rewrite migration_session.status, got: {statements}"
        )
        assert [s for s in statements if 'jsonb_set' in s]

    def test_returns_warning_when_still_not_absolute(self, bp_mod):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        config.NAVIDROME_URL = 'http://nav'
        config.NAVIDROME_USER = 'u'
        config.NAVIDROME_PASSWORD = 'p'

        with (
            patch.object(bp_mod, '_fetch_session_creds', return_value=('navidrome', {})),
            patch.object(bp_mod, 'provider_probe', MagicMock()) as p,
            patch.object(bp_mod, '_detect_path_format') as mock_detect,
            patch.object(bp_mod, '_patch_state_keys'),
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
        with (
            patch.object(
                bp_mod, '_claim_and_enqueue_planner', return_value=('src-job-1', False)
            ) as claim,
        ):
            resp = client.post('/api/migration/source-paths/refresh', json={'session_id': 5})
        assert resp.status_code == 200
        assert resp.get_json().get('task_id') == 'src-job-1'
        assert claim.called

    def test_route_claims_without_demoting_a_finalized_dry_run(self, bp_mod, client):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        config.NAVIDROME_URL = 'http://nav'
        config.NAVIDROME_USER = 'u'
        config.NAVIDROME_PASSWORD = 'p'
        with (
            patch.object(
                bp_mod, '_claim_and_enqueue_planner', return_value=('src-job-1', False)
            ) as claim,
        ):
            resp = client.post('/api/migration/source-paths/refresh', json={'session_id': 5})
        assert resp.status_code == 200
        assert claim.call_args.kwargs['claim_status'] is None, (
            "the refresh claim must leave status alone; 'in_progress' threw away a "
            "finalized dry_run_ready session and Execute then refused the migration"
        )

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
        with (
            patch.object(
                bp_mod, '_fetch_session_creds', return_value=('jellyfin', {'url': 'http://jf'})
            ),
            patch.object(bp_mod, '_load_state', return_value={}),
            patch.object(bp_mod, '_detect_source_path_format', return_value='none'),
            patch.object(
                bp_mod, '_claim_and_enqueue_planner', return_value=('dry-job-1', False)
            ) as claim,
        ):
            resp = client.post(
                '/api/migration/dry-run', json={'session_id': 1, 'bypass_source_check': True}
            )

        assert resp.status_code == 200
        assert resp.get_json().get('task_id') == 'dry-job-1'
        assert claim.called

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
            'confirmation_text': f'I want to migrate to {target} and unbind unmatched tracks',
        }

    def test_rejects_missing_backup_confirmation(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.append((True,))
        cur._fetchone_queue.append(('navidrome', 'dry_run_ready', True))
        p = self._base_payload()
        p['backup_confirmed'] = False
        resp = client.post('/api/migration/execute', json=p)
        assert resp.status_code == 400
        assert 'backup' in resp.get_json().get('error', '').lower()

    def test_rejects_wrong_confirmation_text(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.extend([(0,), (0,)])
        cur._fetchone_queue.append((True,))
        cur._fetchone_queue.append((False,))
        cur._fetchone_queue.append(('navidrome', 'dry_run_ready', True))
        p = self._base_payload()
        p['confirmation_text'] = 'LGTM ship it'
        resp = client.post('/api/migration/execute', json=p)
        assert resp.status_code == 400
        assert 'confirm' in resp.get_json().get('error', '').lower()

    def test_rejects_session_not_in_dry_run_ready(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.extend([(0,), (0,)])
        cur._fetchone_queue.append((True,))
        cur._fetchone_queue.append((False,))
        cur._fetchone_queue.append(('navidrome', 'in_progress', True))
        resp = client.post('/api/migration/execute', json=self._base_payload())
        assert resp.status_code == 400
        err = resp.get_json().get('error', '').lower()
        assert 'dry' in err or 'status' in err

    def test_happy_path_enqueues_job(self, bp_mod, client, fake_db):
        db, cur = fake_db
        cur._fetchone_queue.extend([(0,), (0,)])
        cur._fetchone_queue.append((True,))
        cur._fetchone_queue.append((False,))
        cur._fetchone_queue.append(('navidrome', 'dry_run_ready', True))
        fake_queue = MagicMock()
        fake_job = MagicMock()
        fake_job.id = 'job-xyz'
        fake_queue.enqueue.return_value = fake_job
        bp_mod.rq_queue_high = fake_queue

        with (
            patch.object(bp_mod, 'get_active_main_task', return_value=None),
            patch.object(bp_mod, 'save_task_status'),
            patch.object(bp_mod, 'prune_task_status_history'),
            patch.object(bp_mod, '_patch_state_keys'),
        ):
            resp = client.post('/api/migration/execute', json=self._base_payload())

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['task_id'] == fake_queue.enqueue.call_args.kwargs['job_id']
        assert fake_queue.enqueue.called


class TestPlannerReservationProtocol:
    @staticmethod
    def _claim_db(session_row=('in_progress', True, None, None)):
        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        answers = iter([(0,), (0,), (True,), session_row, (False,), (1,)])
        cur.fetchone.side_effect = lambda: next(answers)
        db = MagicMock()
        db.cursor.return_value = cur
        return db, cur

    def test_claim_and_enqueue_share_one_locked_transaction(self, bp_mod, monkeypatch):
        db, cur = self._claim_db()
        events = []
        held = {'main': False}
        original_execute = cur.execute.side_effect

        @contextmanager
        def start_lock():
            held['main'] = True
            try:
                yield
            finally:
                held['main'] = False

        def execute(sql, params=None):
            events.append(('sql', sql))
            if original_execute:
                return original_execute(sql, params)

        cur.execute.side_effect = execute
        queue = MagicMock()

        def enqueue(*args, **kwargs):
            events.append(('enqueue', kwargs['job_id']))
            assert held['main'] is True
            assert db.commit.call_count == 0
            return MagicMock(id=kwargs['job_id'])

        queue.enqueue.side_effect = enqueue
        monkeypatch.setattr(bp_mod, 'get_db', lambda: db)
        monkeypatch.setattr(bp_mod, 'rq_queue_high', queue)
        monkeypatch.setattr(bp_mod, 'get_active_main_task', lambda **_kw: None)
        monkeypatch.setattr(bp_mod, 'main_task_start_lock', start_lock)

        job_id, reused = bp_mod._claim_and_enqueue_planner(
            9, 'dry_run_task_id', 'tasks.fake', (9,)
        )

        sql = [event[1] for event in events if event[0] == 'sql']
        assert 'pg_try_advisory_xact_lock' in sql[2]
        assert 'FOR UPDATE' in sql[3]
        assert 'jsonb_set' in sql[-1]
        assert events[-1] == ('enqueue', job_id)
        assert reused is False
        db.commit.assert_called_once()
        assert held['main'] is False

    def test_cancel_that_wins_the_lock_invalidates_a_waiting_planner(
        self, bp_mod, monkeypatch
    ):
        db = MagicMock()
        locked = MagicMock()
        monkeypatch.setattr(bp_mod, 'get_db', lambda: db)
        monkeypatch.setattr(bp_mod, '_global_cancel_epoch', MagicMock(side_effect=[4, 5]))
        monkeypatch.setattr(bp_mod, '_claim_and_enqueue_planner_locked', locked)

        with pytest.raises(bp_mod._PlanningClaimError, match='global cancellation'):
            bp_mod._claim_and_enqueue_planner(
                9, 'dry_run_task_id', 'tasks.fake', (9,)
            )

        locked.assert_not_called()

    def test_definitive_missing_enqueue_rolls_back_reservation(self, bp_mod, monkeypatch):
        db, _cur = self._claim_db()
        queue = MagicMock()
        queue.enqueue.side_effect = RuntimeError('lost reply')
        monkeypatch.setattr(bp_mod, 'get_db', lambda: db)
        monkeypatch.setattr(bp_mod, 'rq_queue_high', queue)
        monkeypatch.setattr(bp_mod, 'get_active_main_task', lambda **_kw: None)

        with (
            patch('app_helper.resolve_enqueue_outcome', return_value=('missing', None)),
            pytest.raises(bp_mod._PlanningEnqueueRejected),
        ):
            bp_mod._claim_and_enqueue_planner(
                9, 'dry_run_task_id', 'tasks.fake', (9,)
            )

        db.rollback.assert_called_once()
        db.commit.assert_not_called()

    def test_unknown_enqueue_blocks_then_authoritative_missing_reconciles(
        self, bp_mod, monkeypatch
    ):
        db, _cur = self._claim_db()
        queue = MagicMock()
        queue.enqueue.side_effect = RuntimeError('redis unavailable')
        monkeypatch.setattr(bp_mod, 'get_db', lambda: db)
        monkeypatch.setattr(bp_mod, 'rq_queue_high', queue)
        monkeypatch.setattr(bp_mod, 'get_active_main_task', lambda **_kw: None)

        with (
            patch('app_helper.resolve_enqueue_outcome', return_value=('unknown', None)),
            pytest.raises(bp_mod._PlanningEnqueueUncertain) as exc,
        ):
            bp_mod._claim_and_enqueue_planner(
                9, 'dry_run_task_id', 'tasks.fake', (9,)
            )
        assert exc.value.job_id
        db.commit.assert_called_once()

        blocked = MagicMock()
        blocked.fetchall.return_value = [(9, {'dry_run_task_id': exc.value.job_id})]
        with patch.object(bp_mod, '_rq_jobs_by_id', side_effect=RuntimeError('still down')):
            assert bp_mod._migration_job_in_flight(blocked) is True
        assert not any(
            'state = state -' in call.args[0] for call in blocked.execute.call_args_list
        )

        reconciled = MagicMock()
        reconciled.fetchall.return_value = [(9, {'dry_run_task_id': exc.value.job_id})]
        with patch.object(
            bp_mod, '_rq_jobs_by_id', return_value={exc.value.job_id: None}
        ):
            assert bp_mod._migration_job_in_flight(reconciled) is False
        assert any(
            'state = state -' in call.args[0] for call in reconciled.execute.call_args_list
        )

    @pytest.mark.parametrize(
        ('session_row', 'status_code'),
        [
            (None, 404),
            (('completed', True, None, None), 409),
            (('in_progress', False, None, None), 409),
        ],
    )
    def test_completed_deleted_and_noncurrent_sessions_cannot_claim(
        self, bp_mod, monkeypatch, session_row, status_code
    ):
        db, cur = self._claim_db(session_row)
        queue = MagicMock()
        monkeypatch.setattr(bp_mod, 'get_db', lambda: db)
        monkeypatch.setattr(bp_mod, 'rq_queue_high', queue)
        monkeypatch.setattr(bp_mod, 'get_active_main_task', lambda **_kw: None)

        with pytest.raises(bp_mod._PlanningClaimError) as exc:
            bp_mod._claim_and_enqueue_planner(
                9, 'dry_run_task_id', 'tasks.fake', (9,)
            )
        assert exc.value.status_code == status_code
        queue.enqueue.assert_not_called()

    def test_fast_worker_waits_on_same_lock_before_reading_claim(self, bp_mod, monkeypatch):
        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        cur.fetchone.return_value = ('in_progress', True, 'job-9')
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(bp_mod, 'get_db', lambda: db)

        bp_mod._validate_planner_worker_claim(9, 'dry_run_task_id', 'job-9')

        sql = [call.args[0] for call in cur.execute.call_args_list]
        assert 'pg_advisory_xact_lock' in sql[0]
        assert "state->>%s" in sql[1]
        db.commit.assert_called_once()


class TestMigrationMetadataCompletionFence:
    def test_completed_session_rejects_late_target_metadata(self, bp_mod, monkeypatch):
        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        cur.fetchone.return_value = ('completed', True)
        db = MagicMock()
        db.cursor.return_value = cur
        monkeypatch.setattr(bp_mod, 'get_db', lambda: db)

        with pytest.raises(RuntimeError, match='late target metadata'):
            bp_mod._store_target_meta(3, {'new': {'title': 'late'}})

        sql = [call.args[0] for call in cur.execute.call_args_list]
        assert 'FOR UPDATE' in sql[0]
        assert not any(statement.startswith('DELETE FROM migration_target_meta') for statement in sql)
        db.commit.assert_not_called()

    def test_completion_pruning_requires_ack_and_terminal_rq_job(self, bp_mod):
        cur = MagicMock()
        cur.fetchall.return_value = [
            (1, 'alive', True),
            (2, 'dead', True),
            (3, 'unacked', False),
            (4, None, False),
            (5, 'missing', True),
        ]
        jobs = {'alive': 'alive', 'dead': 'dead', 'unacked': 'dead', 'missing': None}
        with (
            patch.object(bp_mod, '_rq_jobs_by_id', return_value=jobs),
            patch.object(bp_mod, '_rq_job_is_alive', side_effect=lambda job: job == 'alive'),
        ):
            assert bp_mod._completed_sessions_safe_to_prune(cur) == [2, 5]


class TestExecuteEnqueueResolution:
    @staticmethod
    def _prime_execute(cur):
        cur._fetchone_queue.extend(
            [(0,), (0,), (True,), (False,), ('navidrome', 'dry_run_ready', True)]
        )

    @pytest.mark.parametrize(
        ('outcome', 'expected_status'),
        [('accepted', 200), ('missing', 500), ('unknown', 503)],
    )
    def test_execute_enqueue_exception_is_resolved_under_start_lock(
        self, bp_mod, client, fake_db, monkeypatch, outcome, expected_status
    ):
        db, cur = fake_db
        self._prime_execute(cur)
        held = {'value': False}

        @contextmanager
        def start_lock():
            held['value'] = True
            try:
                yield
            finally:
                held['value'] = False

        queue = MagicMock()

        def enqueue(*_args, **_kwargs):
            assert held['value'] is True
            raise RuntimeError('enqueue reply lost')

        queue.enqueue.side_effect = enqueue
        monkeypatch.setattr(bp_mod, 'rq_queue_high', queue)
        monkeypatch.setattr(bp_mod, 'main_task_start_lock', start_lock)
        saves = MagicMock()

        with (
            patch.object(bp_mod, 'get_active_main_task', return_value=None),
            patch.object(bp_mod, 'save_task_status', saves),
            patch.object(bp_mod, 'prune_task_status_history'),
            patch.object(bp_mod, '_patch_state_keys') as patch_state,
            patch('app_helper.resolve_enqueue_outcome', return_value=(outcome, None)),
        ):
            resp = client.post(
                '/api/migration/execute',
                json={
                    'session_id': 1,
                    'backup_confirmed': True,
                    'confirmation_text': (
                        'I want to migrate to navidrome and unbind unmatched tracks'
                    ),
                },
            )

        assert resp.status_code == expected_status
        assert held['value'] is False
        first_claim = patch_state.call_args_list[0].kwargs['exec_task_id']
        if outcome == 'missing':
            assert patch_state.call_args_list[-1].kwargs['exec_task_id'] is None
            assert any(call.args[2] == 'FAILURE' for call in saves.call_args_list)
        else:
            assert all(
                call.kwargs.get('exec_task_id', first_claim) is not None
                for call in patch_state.call_args_list
            )
            assert not any(call.args[2] == 'FAILURE' for call in saves.call_args_list)

    def test_cancel_that_wins_the_lock_prevents_execute_enqueue(
        self, bp_mod, client, fake_db, monkeypatch
    ):
        db, _cur = fake_db
        inner = MagicMock()
        monkeypatch.setattr(bp_mod, '_global_cancel_epoch', MagicMock(side_effect=[8, 9]))
        monkeypatch.setattr(bp_mod, '_execute_locked', inner)

        resp = client.post(
            '/api/migration/execute',
            json={
                'session_id': 1,
                'backup_confirmed': True,
                'confirmation_text': (
                    'I want to migrate to navidrome and unbind unmatched tracks'
                ),
            },
        )

        assert resp.status_code == 409
        assert 'global cancellation' in resp.get_json()['error']
        inner.assert_not_called()


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


class TestPlannerClaimStatus:
    """A source-path refresh must not demote a finalized dry run.

    ``status = COALESCE(%s, status)`` with a NULL parameter leaves the session
    exactly as it was.  Passing 'in_progress' unconditionally threw away a
    dry_run_ready session, and Execute then refused the migration outright, so the
    user had to redo the entire dry run just to correct one source path.
    """

    @staticmethod
    def _claim(bp_mod, fake_db, state_key, claim_status):
        db, cur = fake_db
        cur._fetchone_queue.append((True,))                 # advisory xact lock
        cur._fetchone_queue.append(('dry_run_ready', True, None, None))
        cur._fetchone_queue.append((False,))                # no pending handshake
        cur._fetchone_queue.append((7,))                    # UPDATE ... RETURNING id
        with (
            patch.object(bp_mod, 'get_active_main_task', return_value=None),
            patch.object(bp_mod, '_rq_jobs_by_id', return_value={}),
            patch.object(bp_mod, 'rq_queue_high') as queue,
        ):
            queue.enqueue.return_value = MagicMock()
            bp_mod._claim_and_enqueue_planner_locked(
                db, 1, state_key, 'tasks.x', (1,), claim_status=claim_status,
            )
        for call in cur.execute.call_args_list:
            sql = call[0][0]
            if sql.strip().upper().startswith('UPDATE MIGRATION_SESSION'):
                return " ".join(sql.split()), call[0][1]
        raise AssertionError('no UPDATE migration_session was issued')

    def test_source_refresh_leaves_the_session_status_untouched(self, bp_mod, fake_db):
        sql, params = self._claim(
            bp_mod, fake_db, 'source_refresh_task_id', None,
        )
        assert 'status = COALESCE(%s, status)' in sql
        assert params[2] is None

    def test_dry_run_still_demotes_the_session_to_in_progress(self, bp_mod, fake_db):
        sql, params = self._claim(
            bp_mod, fake_db, 'dry_run_task_id', 'in_progress',
        )
        assert 'status = COALESCE(%s, status)' in sql
        assert params[2] == 'in_progress'
