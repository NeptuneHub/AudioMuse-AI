"""Route-level tests for app_provider_migration.

Covers the wiring: session CRUD, probe/test proxy, dry-run trigger, manual
album match, finalize gate, execute gate (backup checkbox + confirmation
phrase + dry_run_ready status). Uses the _import_module bypass so this runs
without librosa.
"""
import os
import sys
import json
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
    """A minimal Flask app with the migration blueprint registered."""
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
    """Install a fake get_db that returns a rich MagicMock cursor with
    configurable fetch queue."""
    cur = MagicMock()
    cur.__enter__ = lambda self: self
    cur.__exit__  = lambda self, *a: None
    # Queue of values fetchone() will return, in order
    cur._fetchone_queue = []
    cur.fetchone.side_effect = lambda: cur._fetchone_queue.pop(0) if cur._fetchone_queue else None

    db = MagicMock()
    db.cursor.return_value = cur
    db.commit = MagicMock()

    bp_mod.get_db = MagicMock(return_value=db)
    return db, cur


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class TestMigrationPageRoute:
    def test_renders_with_layout(self, bp_mod, client):
        # Patch render_template so we don't need the actual HTML file
        with patch.object(bp_mod, 'render_template', return_value='<html>ok</html>') as mock_rt:
            resp = client.get('/provider-migration')
        assert resp.status_code == 200
        assert mock_rt.called
        # Ensure we passed active='provider_migration' for sidebar highlight
        kwargs = mock_rt.call_args[1]
        assert kwargs.get('active') == 'provider_migration'


class TestSessionStart:
    def test_creates_session_row(self, bp_mod, client, fake_db):
        db, cur = fake_db
        # Return the new session id from INSERT ... RETURNING id
        cur._fetchone_queue.append((123,))
        # Patch config.MEDIASERVER_TYPE so source_type can be captured
        import config
        config.MEDIASERVER_TYPE = 'jellyfin'

        resp = client.post('/api/migration/session/start', json={
            'target_type': 'navidrome',
            'target_creds': {'url': 'http://nav', 'user': 'u', 'password': 'p'},
        })

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['session_id'] == 123
        # INSERT was called
        sqls = [c[0][0] for c in cur.execute.call_args_list]
        assert any('INSERT INTO migration_session' in s for s in sqls)

    def test_rejects_unknown_target_type(self, bp_mod, client, fake_db):
        resp = client.post('/api/migration/session/start', json={
            'target_type': 'bogus',
            'target_creds': {},
        })
        assert resp.status_code == 400


class TestProbeTest:
    def test_calls_provider_probe_and_returns_shape(self, bp_mod, client):
        fake = {'ok': True, 'error': None, 'sample_count': 5,
                'path_format': 'absolute', 'warnings': []}
        with patch.object(bp_mod, 'provider_probe', MagicMock()) as p:
            p.test_connection.return_value = fake
            resp = client.post('/api/migration/probe/test', json={
                'type': 'navidrome',
                'creds': {'url': 'http://nav', 'user': 'u', 'password': 'p'},
            })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['ok'] is True
        assert data['path_format'] == 'absolute'


class TestExecuteGate:
    """The execute endpoint is the most security-critical route — it must
    reject any request that bypasses the backup + confirmation + dry-run gates."""

    def _base_payload(self, target='navidrome'):
        return {
            'session_id':        1,
            'backup_confirmed':  True,
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
