import db_migrations
from db_migrations import run_schema_migrations, get_schema_version, BASELINE_VERSION


class FakeCursor:
    """Minimal stand-in for a psycopg2 cursor backed by an in-memory set of
    applied schema versions. Only the statements db_migrations issues are
    recognised."""

    def __init__(self):
        self.applied = set()
        self._result = None

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT version FROM schema_version"):
            self._result = [(v,) for v in sorted(self.applied)]
        elif normalized.startswith("INSERT INTO schema_version"):
            self.applied.add(params[0])
            self._result = None
        elif normalized.startswith("SELECT COALESCE(MAX(version)"):
            self._result = [(max(self.applied) if self.applied else -1,)]
        else:
            # CREATE TABLE IF NOT EXISTS / migration DDL: no result.
            self._result = None

    def fetchall(self):
        return self._result or []

    def fetchone(self):
        return (self._result or [(None,)])[0]


def test_empty_migrations_records_baseline():
    cur = FakeCursor()
    applied = run_schema_migrations(cur, migrations=[])
    assert applied == 0
    assert BASELINE_VERSION in cur.applied
    assert get_schema_version(cur) == BASELINE_VERSION


def test_pending_migrations_apply_in_order():
    calls = []
    migrations = [
        (2, "second", lambda c: calls.append(2)),
        (1, "first", lambda c: calls.append(1)),
    ]
    cur = FakeCursor()
    applied = run_schema_migrations(cur, migrations=migrations)
    assert applied == 2
    assert calls == [1, 2]
    assert get_schema_version(cur) == 2


def test_already_applied_migrations_are_skipped():
    calls = []
    migrations = [(1, "first", lambda c: calls.append(1))]
    cur = FakeCursor()
    run_schema_migrations(cur, migrations=migrations)
    assert calls == [1]
    # Second run: nothing new, fn not invoked again.
    applied = run_schema_migrations(cur, migrations=migrations)
    assert applied == 0
    assert calls == [1]


def test_get_schema_version_empty():
    cur = FakeCursor()
    assert get_schema_version(cur) == -1


def test_default_registry_is_a_list():
    assert isinstance(db_migrations.MIGRATIONS, list)
