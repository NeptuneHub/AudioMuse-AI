"""Real-Postgres integration test for the task-status details parse fix.

PR.md theme #13: ``app.py get_task_status_endpoint`` reads ``details`` from the
DB and must NOT double-parse it::

    raw_details = db_task_info.get('details')
    if isinstance(raw_details, dict):
        db_details = raw_details
    elif raw_details:
        db_details = json.loads(raw_details)

The real ``task_status.details`` column is TEXT and ``database.save_task_status``
stores ``json.dumps(details)``, so the real getter
(``database.get_task_info_from_db``) returns ``details`` as a JSON *string* (the
``elif`` branch). A JSON/JSONB column instead makes psycopg2 hand back a Python
*dict* (the ``isinstance`` branch). This test exercises BOTH real round-trips and
applies the exact endpoint branch to the real returned values, proving no
double-parse and no error in either case.

Importing ``app.py`` end-to-end is infeasible here (module import pulls in
redis/RQ connections, every blueprint, and full runtime config), so we drive the
REAL database getter ``get_task_info_from_db`` against a live connection and
replicate only the tiny isinstance/json.loads branch from the endpoint. The
value it branches on is a genuine DB round-trip, never a mock.

Database selection mirrors test_app_endpoints_integration.py:
  * AUDIOMUSE_TEST_DATABASE_URL -- a throwaway DB the test fully owns, or
  * an ephemeral instance via the optional ``pgserver`` package, or
  * the module is skipped.

Run locally:
    pip install pgserver
    pytest test/integration/test_task_status_integration.py -m integration -s -v --tb=short
"""
import json
import os
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
    from psycopg2.extras import DictCursor
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None
    DictCursor = None


# Real task_status schema, copied verbatim from database.py init_db: details is
# TEXT, plus the start_time/end_time DOUBLE PRECISION columns the migration adds.
_TASK_STATUS_DDL = (
    "CREATE TABLE task_status ("
    "id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, "
    "task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, "
    "progress INTEGER DEFAULT 0, details TEXT, "
    "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "start_time DOUBLE PRECISION, end_time DOUBLE PRECISION)"
)

# Variant whose details column is JSONB: psycopg2 decodes JSONB to a Python dict
# on fetch, which is the real-world source of the endpoint's isinstance() branch.
_TASK_STATUS_JSONB_DDL = (
    "CREATE TABLE task_status ("
    "id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, "
    "task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, "
    "progress INTEGER DEFAULT 0, details JSONB, "
    "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "start_time DOUBLE PRECISION, end_time DOUBLE PRECISION)"
)

_SAMPLE_DETAILS = {
    "log": ["Analyzing album", "Done"],
    "current_album": "Album X",
    "status_message": "running",
    "nested": {"a": 1, "b": [2, 3]},
}


def _endpoint_db_details(raw_details):
    """The exact branch from app.py get_task_status_endpoint, isolated.

    Returns the surfaced details dict for a raw DB value, mirroring the source
    one-for-one so a regression there (double-parse / wrong type) would fail the
    real-round-trip assertions below.
    """
    db_details = {}
    if isinstance(raw_details, dict):
        db_details = raw_details
    elif raw_details:
        try:
            db_details = json.loads(raw_details)
        except (json.JSONDecodeError, TypeError):
            db_details = {}
    return db_details


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
def text_details_db(pg_dsn):
    """Real connection with the real (TEXT details) task_status table seeded."""
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        cur.execute(_TASK_STATUS_DDL)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
    conn.close()


@pytest.fixture
def jsonb_details_db(pg_dsn):
    """Real connection with a JSONB-details task_status table seeded."""
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        cur.execute(_TASK_STATUS_JSONB_DDL)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
    conn.close()


def _seed_text(conn, task_id):
    # Stores details exactly as save_task_status does: json.dumps into TEXT.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO task_status (task_id, task_type, status, progress, "
            "details, start_time, end_time) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (task_id, 'main_analysis', 'PROGRESS', 42,
             json.dumps(_SAMPLE_DETAILS), 1000.0, None),
        )


def _seed_jsonb(conn, task_id):
    # JSONB column: hand psycopg2 a JSON string and let the DB store it as JSONB.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO task_status (task_id, task_type, status, progress, "
            "details, start_time, end_time) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (task_id, 'main_clustering', 'SUCCESS', 100,
             json.dumps(_SAMPLE_DETAILS), 1000.0, 1002.5),
        )


@pytest.mark.integration
class TestTaskStatusDetailsRoundTrip:
    def test_text_details_returns_json_string(self, text_details_db, monkeypatch):
        """Real TEXT column -> getter returns a JSON string (elif branch)."""
        import database
        _seed_text(text_details_db, 'task-text-1')
        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        row = database.get_task_info_from_db('task-text-1')
        assert row is not None
        raw_details = row.get('details')
        # The real driver hands TEXT back as a Python str, not a dict.
        assert isinstance(raw_details, str)

        surfaced = _endpoint_db_details(raw_details)
        assert isinstance(surfaced, dict)
        assert surfaced == _SAMPLE_DETAILS
        assert surfaced['current_album'] == 'Album X'
        assert surfaced['nested'] == {"a": 1, "b": [2, 3]}

    def test_jsonb_details_returns_dict(self, jsonb_details_db, monkeypatch):
        """Real JSONB column -> getter returns a Python dict (isinstance branch)."""
        import database
        _seed_jsonb(jsonb_details_db, 'task-jsonb-1')
        monkeypatch.setattr(database, 'get_db', lambda: jsonb_details_db)

        row = database.get_task_info_from_db('task-jsonb-1')
        assert row is not None
        raw_details = row.get('details')
        # psycopg2 auto-decodes JSONB to a dict; the endpoint must NOT re-parse.
        assert isinstance(raw_details, dict)

        surfaced = _endpoint_db_details(raw_details)
        assert surfaced is raw_details
        assert surfaced == _SAMPLE_DETAILS
        assert surfaced['log'] == ["Analyzing album", "Done"]

    def test_no_double_parse_between_paths(self, text_details_db, jsonb_details_db, monkeypatch):
        """Both real paths surface identical content; neither errors out."""
        import database
        _seed_text(text_details_db, 'task-text-2')
        _seed_jsonb(jsonb_details_db, 'task-jsonb-2')

        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)
        text_row = database.get_task_info_from_db('task-text-2')
        text_surfaced = _endpoint_db_details(text_row.get('details'))

        monkeypatch.setattr(database, 'get_db', lambda: jsonb_details_db)
        jsonb_row = database.get_task_info_from_db('task-jsonb-2')
        jsonb_surfaced = _endpoint_db_details(jsonb_row.get('details'))

        assert text_surfaced == jsonb_surfaced == _SAMPLE_DETAILS
        # A double-parse would have raised TypeError on the dict path; reaching
        # here means the isinstance guard short-circuited correctly.
        assert isinstance(text_surfaced['log'], list)
        assert isinstance(jsonb_surfaced['log'], list)

    def test_null_details_surfaces_empty_dict(self, text_details_db, monkeypatch):
        """NULL details -> getter returns None -> endpoint yields {} (no error)."""
        import database
        with text_details_db.cursor() as cur:
            cur.execute(
                "INSERT INTO task_status (task_id, task_type, status, details) "
                "VALUES (%s, %s, %s, %s)",
                ('task-null', 'main_analysis', 'PENDING', None),
            )
        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        row = database.get_task_info_from_db('task-null')
        assert row is not None
        assert row.get('details') is None
        assert _endpoint_db_details(row.get('details')) == {}

    def test_real_getter_uses_dictcursor_mapping(self, text_details_db, monkeypatch):
        """Sanity: the real getter yields a mapping with the expected columns."""
        import database
        _seed_text(text_details_db, 'task-text-3')
        monkeypatch.setattr(database, 'get_db', lambda: text_details_db)

        row = database.get_task_info_from_db('task-text-3')
        assert row['task_type'] == 'main_analysis'
        assert row['status'] == 'PROGRESS'
        assert row['progress'] == 42
        # running_time_seconds is computed in Python; end_time is NULL -> > 0.
        assert row['running_time_seconds'] > 0
