"""Real-Postgres integration test for the task-status details round-trip.

The ``/api/status/<task_id>`` endpoint persists ``details`` via
``database.save_task_status`` and reads it back via
``database.get_task_info_from_db``, then normalizes it with
``app_helper.coerce_db_details`` (no double-parse). The DB-dependent fact a mock
cannot show is that the *real* driver hands ``details`` back as a JSON **string**
from a TEXT column but as an already-decoded **dict** from a JSONB column. This
test drives the real save -> real getter -> real coercion chain against a live
Postgres for both column types, plus the NULL case.

app.py itself is not imported end-to-end on purpose: importing it runs
``init_db`` which needs the unaccent/pg_trgm contrib extensions that the
ephemeral pgserver build lacks (the CI Postgres has them, local pgserver does
not). So we drive the real ``database`` functions directly -- every SQL
statement, transaction and returned type below is real -- and feed the real
returned value through the real shared helper the endpoint uses.

Database selection mirrors test_app_endpoints_integration.py:
  * AUDIOMUSE_TEST_DATABASE_URL -- a throwaway DB the test fully owns, or
  * an ephemeral instance via the optional ``pgserver`` package, or
  * the module is skipped.

Run locally:
    pip install pgserver
    pytest test/integration/test_task_status_integration.py -m integration -s -v --tb=short
"""
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

# Real task_status schema from database.init_db: details is TEXT, plus the
# start_time/end_time DOUBLE PRECISION columns the migration adds.
_TASK_STATUS_DDL = (
    "CREATE TABLE task_status ("
    "id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, "
    "task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, "
    "progress INTEGER DEFAULT 0, details TEXT, "
    "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "start_time DOUBLE PRECISION, end_time DOUBLE PRECISION)"
)

# Same table but details is JSONB: psycopg2 decodes JSONB to a dict on fetch,
# which is the real source of the endpoint's isinstance() branch.
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
        """Real save -> TEXT column -> real getter returns a JSON string -> coerced."""
        import database
        import app_helper
        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        database.save_task_status(
            'task-text', 'main_analysis', status='PROGRESS', progress=42,
            details=copy.deepcopy(_SAMPLE_DETAILS),
        )
        row = database.get_task_info_from_db('task-text')
        assert row is not None
        # The real driver hands a TEXT column back as a Python str, not a dict.
        assert isinstance(row['details'], str)

        surfaced = app_helper.coerce_db_details(row['details'])
        assert surfaced == _SAMPLE_DETAILS
        assert surfaced['nested'] == {"a": 1, "b": [2, 3]}
        assert row['task_type'] == 'main_analysis'
        assert row['progress'] == 42
        # start_time set, end_time NULL (PROGRESS is non-terminal) -> still running.
        assert row['running_time_seconds'] >= 0

    def test_jsonb_details_round_trip_returns_dict_no_reparse(self, jsonb_details_db, monkeypatch):
        """Real save -> JSONB column -> real getter returns a dict -> not re-parsed."""
        import database
        import app_helper
        monkeypatch.setattr(database, 'get_db', lambda: jsonb_details_db)

        database.save_task_status(
            'task-jsonb', 'main_clustering', status='SUCCESS', progress=100,
            details=copy.deepcopy(_SAMPLE_DETAILS),
        )
        row = database.get_task_info_from_db('task-jsonb')
        assert row is not None
        # psycopg2 auto-decodes JSONB to a dict; coerce must hand it back as-is.
        assert isinstance(row['details'], dict)

        surfaced = app_helper.coerce_db_details(row['details'])
        assert surfaced is row['details']
        assert surfaced == _SAMPLE_DETAILS
        # SUCCESS is terminal: end_time recorded, so a real duration is computed.
        assert row['running_time_seconds'] >= 0

    def test_both_paths_surface_identical_content(self, text_details_db, jsonb_details_db, monkeypatch):
        """The str path and the dict path collapse to the same dict; neither errors."""
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
        """Real save with details=None -> NULL column -> getter None -> {} (no error)."""
        import database
        import app_helper
        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        database.save_task_status('task-null', 'main_analysis', status='PENDING', details=None)
        row = database.get_task_info_from_db('task-null')
        assert row is not None
        assert row['details'] is None
        assert app_helper.coerce_db_details(row['details']) == {}
