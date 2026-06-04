"""Real-Postgres integration tests for the read endpoints.

Drives the actual external_bp / waveform_bp blueprints against a live database
seeded with a real ``score`` / ``embedding`` layout. Beyond happy-path lookups,
it proves the SQL-injection regression at the database level: an item id full of
SQL metacharacters returns 404 and leaves the ``score`` table intact, which a
mocked cursor cannot demonstrate.

Database selection mirrors test_provider_migration_integration.py:
  * AUDIOMUSE_TEST_DATABASE_URL — a throwaway DB the test fully owns, or
  * an ephemeral instance via the optional ``pgserver`` package, or
  * the module is skipped.

Run locally:
    pip install pgserver
    pytest test/test_app_endpoints_integration.py -m integration -s -v --tb=short
"""
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None


_SCORE_DDL = (
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
    "album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, "
    "mood_vector TEXT, energy REAL, other_features TEXT, year INTEGER, "
    "rating INTEGER, file_path TEXT)"
)
_EMBEDDING_DDL = (
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, "
    "FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)"
)

_INJECTION_ID = "x'; DROP TABLE score; --"


@pytest.fixture(scope='session')
def pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")
    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.skip(f"AUDIOMUSE_TEST_DATABASE_URL not reachable: {e}")
        yield dsn
        return
    try:
        import pgserver
    except Exception:
        pytest.skip(
            "No test database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable "
            "DB, or `pip install pgserver` for an ephemeral local instance."
        )
    data_dir = tempfile.mkdtemp(prefix='audiomuse_pg_')
    server = pgserver.get_server(data_dir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture
def endpoints_db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS embedding")
        cur.execute("DROP TABLE IF EXISTS score CASCADE")
        cur.execute(_SCORE_DDL)
        cur.execute(_EMBEDDING_DDL)
        cur.execute(
            "INSERT INTO score (item_id, title, author) VALUES (%s, %s, %s)",
            ('track-1', 'Hello', 'Adele'),
        )
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        cur.execute(
            "INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)",
            ('track-1', psycopg2.Binary(vec)),
        )
    yield conn
    conn.close()


def _import_app_external():
    if 'app_external' in sys.modules:
        return sys.modules['app_external']
    fake_vm = types.ModuleType('tasks.voyager_manager')
    fake_vm.search_tracks_unified = MagicMock(return_value=[])
    stubs = {'tasks.voyager_manager': fake_vm}
    if 'tasks' not in sys.modules:
        stubs['tasks'] = types.ModuleType('tasks')
    with patch.dict(sys.modules, stubs):
        import app_external
    return app_external


def _external_client(ext):
    app = Flask(__name__)
    app.register_blueprint(ext.external_bp)
    app.config['TESTING'] = True
    return app.test_client()


def _table_exists(conn, name):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f'public.{name}',))
        return cur.fetchone()[0] is not None


def _score_count(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM score")
        return cur.fetchone()[0]


@pytest.mark.integration
class TestScoreEndpointRealDb:
    def test_seeded_id_returns_row(self, endpoints_db, monkeypatch):
        import app_helper
        ext = _import_app_external()
        monkeypatch.setattr(app_helper, 'get_db', lambda: endpoints_db)
        resp = _external_client(ext).get('/get_score', query_string={'id': 'track-1'})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['item_id'] == 'track-1'
        assert body['title'] == 'Hello'
        assert body['author'] == 'Adele'

    def test_missing_id_returns_404(self, endpoints_db, monkeypatch):
        import app_helper
        ext = _import_app_external()
        monkeypatch.setattr(app_helper, 'get_db', lambda: endpoints_db)
        resp = _external_client(ext).get('/get_score', query_string={'id': 'does-not-exist'})
        assert resp.status_code == 404

    def test_injection_id_is_safe(self, endpoints_db, monkeypatch):
        import app_helper
        ext = _import_app_external()
        monkeypatch.setattr(app_helper, 'get_db', lambda: endpoints_db)
        resp = _external_client(ext).get('/get_score', query_string={'id': _INJECTION_ID})
        assert resp.status_code == 404
        assert _table_exists(endpoints_db, 'score')
        assert _score_count(endpoints_db) == 1


@pytest.mark.integration
class TestEmbeddingEndpointRealDb:
    def test_seeded_id_returns_embedding(self, endpoints_db, monkeypatch):
        import app_helper
        ext = _import_app_external()
        monkeypatch.setattr(app_helper, 'get_db', lambda: endpoints_db)
        resp = _external_client(ext).get('/get_embedding', query_string={'id': 'track-1'})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['item_id'] == 'track-1'
        assert isinstance(body['embedding'], list)
        assert len(body['embedding']) == 3

    def test_missing_id_returns_404(self, endpoints_db, monkeypatch):
        import app_helper
        ext = _import_app_external()
        monkeypatch.setattr(app_helper, 'get_db', lambda: endpoints_db)
        resp = _external_client(ext).get('/get_embedding', query_string={'id': 'does-not-exist'})
        assert resp.status_code == 404


@pytest.mark.integration
class TestWaveformLookupRealDb:
    def _client(self):
        import app_waveform
        app = Flask(__name__)
        app.register_blueprint(app_waveform.waveform_bp)
        app.config['TESTING'] = True
        return app_waveform, app.test_client()

    def test_missing_track_returns_404(self, endpoints_db, monkeypatch):
        wf, client = self._client()
        monkeypatch.setattr(wf, 'get_db', lambda: endpoints_db)
        resp = client.get('/api/waveform', query_string={'item_id': 'does-not-exist'})
        assert resp.status_code == 404

    def test_injection_item_id_is_safe(self, endpoints_db, monkeypatch):
        wf, client = self._client()
        monkeypatch.setattr(wf, 'get_db', lambda: endpoints_db)
        resp = client.get('/api/waveform', query_string={'item_id': _INJECTION_ID})
        assert resp.status_code == 404
        assert _table_exists(endpoints_db, 'score')
        assert _score_count(endpoints_db) == 1
