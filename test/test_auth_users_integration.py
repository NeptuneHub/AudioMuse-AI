"""Real-Postgres integration tests for the audiomuse_users layer.

Drives app_auth's user CRUD against a live ``audiomuse_users`` table so the
argon2 hash round-trip, the ON CONFLICT duplicate guard, and the
SELECT ... FOR UPDATE last-admin gate all run as production code paths. A
mocked cursor cannot prove the hash verifies against what was actually stored,
nor that the last-admin deletion is refused atomically.

Database selection mirrors test_provider_migration_integration.py:
  * AUDIOMUSE_TEST_DATABASE_URL — a throwaway DB the test fully owns, or
  * an ephemeral instance via the optional ``pgserver`` package, or
  * the module is skipped.

Run locally:
    pip install pgserver
    pytest test/test_auth_users_integration.py -m integration -s -v --tb=short
"""
import os
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None


_USERS_DDL = (
    "CREATE TABLE audiomuse_users (id SERIAL PRIMARY KEY, "
    "username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, "
    "role TEXT NOT NULL DEFAULT 'user', "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)


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
def users_db(pg_dsn, monkeypatch):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS audiomuse_users CASCADE")
        cur.execute(_USERS_DDL)
    conn.commit()
    import app_auth
    monkeypatch.setattr(app_auth, '_get_db', lambda: conn)
    yield conn, app_auth
    conn.close()


def _user_id(conn, username):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM audiomuse_users WHERE username = %s", (username,))
        return cur.fetchone()[0]


@pytest.mark.integration
class TestCreateAndVerifyUserRealDb:
    def test_create_stores_argon2_and_verify_roundtrip(self, users_db):
        conn, app_auth = users_db
        ok, err = app_auth.create_additional_user('alice', 'pw-1', 'admin')
        assert ok is True
        assert err is None
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash, role FROM audiomuse_users WHERE username = %s",
                ('alice',),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0].startswith('$argon2')
        assert row[1] == 'admin'
        assert app_auth.verify_additional_user('alice', 'pw-1') == 'admin'
        assert app_auth.verify_additional_user('alice', 'wrong') is None

    def test_duplicate_username_rejected(self, users_db):
        conn, app_auth = users_db
        ok1, _ = app_auth.create_additional_user('bob', 'pw', 'user')
        ok2, err2 = app_auth.create_additional_user('bob', 'pw2', 'user')
        assert ok1 is True
        assert ok2 is False
        assert 'exists' in err2.lower()

    def test_count_admin_users_reflects_inserts(self, users_db):
        conn, app_auth = users_db
        assert app_auth.count_admin_users() == 0
        app_auth.create_additional_user('a1', 'pw', 'admin')
        app_auth.create_additional_user('u1', 'pw', 'user')
        assert app_auth.count_admin_users() == 1
        app_auth.create_additional_user('a2', 'pw', 'admin')
        assert app_auth.count_admin_users() == 2


@pytest.mark.integration
class TestDeleteLastAdminGateRealDb:
    def test_refuses_to_delete_last_admin(self, users_db):
        conn, app_auth = users_db
        app_auth.create_additional_user('only-admin', 'pw', 'admin')
        uid = _user_id(conn, 'only-admin')
        status, err = app_auth.delete_additional_user_safe(uid)
        assert status == 'last_admin'
        assert app_auth.count_admin_users() == 1

    def test_deletes_non_last_admin(self, users_db):
        conn, app_auth = users_db
        app_auth.create_additional_user('admin-1', 'pw', 'admin')
        app_auth.create_additional_user('admin-2', 'pw', 'admin')
        uid = _user_id(conn, 'admin-2')
        status, err = app_auth.delete_additional_user_safe(uid)
        assert status == 'deleted'
        assert app_auth.count_admin_users() == 1
