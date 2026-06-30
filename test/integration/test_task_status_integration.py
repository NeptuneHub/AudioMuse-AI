import copy
import os
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

pytestmark = pytest.mark.integration

_TASK_STATUS_DDL = (
    "CREATE TABLE task_status ("
    "id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, "
    "task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, "
    "progress INTEGER DEFAULT 0, details TEXT, "
    "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "start_time DOUBLE PRECISION, end_time DOUBLE PRECISION)"
)

_TASK_STATUS_JSONB_DDL = _TASK_STATUS_DDL.replace("details TEXT,", "details JSONB,")

_SAMPLE_DETAILS = {
    "log": ["Analyzing album", "Done"],
    "current_album": "Album X",
    "status_message": "running",
    "nested": {"a": 1, "b": [2, 3]},
}


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


def _make_db(pg_dsn, ddl):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        cur.execute(ddl)
    return conn


@pytest.fixture
def text_details_db(pg_dsn):
    conn = _make_db(pg_dsn, _TASK_STATUS_DDL)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
    conn.close()


@pytest.fixture
def jsonb_details_db(pg_dsn):
    conn = _make_db(pg_dsn, _TASK_STATUS_JSONB_DDL)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
    conn.close()


class TestTaskStatusDetailsRoundTrip:
    def test_text_details_round_trip_is_json_string(self, text_details_db, monkeypatch):
        import database
        import app_helper
        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        database.save_task_status(
            'task-text', 'main_analysis', status='PROGRESS', progress=42,
            details=copy.deepcopy(_SAMPLE_DETAILS),
        )
        row = database.get_task_info_from_db('task-text')
        assert row is not None
        assert isinstance(row['details'], str)

        surfaced = app_helper.coerce_db_details(row['details'])
        assert surfaced == _SAMPLE_DETAILS
        assert surfaced['nested'] == {"a": 1, "b": [2, 3]}
        assert row['task_type'] == 'main_analysis'
        assert row['progress'] == 42
        assert row['running_time_seconds'] >= 0

    def test_jsonb_details_round_trip_returns_dict_no_reparse(self, jsonb_details_db, monkeypatch):
        import database
        import app_helper
        monkeypatch.setattr(database, 'get_db', lambda: jsonb_details_db)

        database.save_task_status(
            'task-jsonb', 'main_clustering', status='SUCCESS', progress=100,
            details=copy.deepcopy(_SAMPLE_DETAILS),
        )
        row = database.get_task_info_from_db('task-jsonb')
        assert row is not None
        assert isinstance(row['details'], dict)

        surfaced = app_helper.coerce_db_details(row['details'])
        assert surfaced is row['details']
        assert surfaced == _SAMPLE_DETAILS
        assert row['running_time_seconds'] >= 0

    def test_both_paths_surface_identical_content(self, text_details_db, jsonb_details_db, monkeypatch):
        import database
        import app_helper

        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)
        database.save_task_status('t-text', 'main_analysis', status='PROGRESS',
                                  details=copy.deepcopy(_SAMPLE_DETAILS))
        text_surfaced = app_helper.coerce_db_details(
            database.get_task_info_from_db('t-text')['details'])

        monkeypatch.setattr(database, 'get_db', lambda: jsonb_details_db)
        database.save_task_status('t-jsonb', 'main_clustering', status='SUCCESS',
                                  details=copy.deepcopy(_SAMPLE_DETAILS))
        jsonb_surfaced = app_helper.coerce_db_details(
            database.get_task_info_from_db('t-jsonb')['details'])

        assert text_surfaced == jsonb_surfaced == _SAMPLE_DETAILS
        assert isinstance(text_surfaced['log'], list)
        assert isinstance(jsonb_surfaced['log'], list)

    def test_null_details_surfaces_empty_dict(self, text_details_db, monkeypatch):
        import database
        import app_helper
        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        database.save_task_status('task-null', 'main_analysis', status='PENDING', details=None)
        row = database.get_task_info_from_db('task-null')
        assert row is not None
        assert row['details'] is None
        assert app_helper.coerce_db_details(row['details']) == {}
