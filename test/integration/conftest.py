# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared Postgres fixtures for the integration suite.

Offers one disposable database and one copy of the task_status DDL, so a schema
change lands in a single place. The last time task_status gained columns
(start_time, end_time) a second hand-written copy is exactly the kind of drift
that would have gone unnoticed.

KNOWN LIMITATION, do not read the fixture list as a claim of the opposite: 16
modules under test/integration still define their own module-level `pg_dsn`,
which SHADOWS the one here, so each of them starts its own pgserver. Only
modules that do not define it share this instance. Until the rest are
converted, a full `pytest test/unit
test/integration` in one process starts many servers at once and has been seen
to take WSL down with E_UNEXPECTED; running the two suites separately is
reliable.

The DDL is exposed as fixtures rather than importable constants because a test
module cannot `import conftest`: pytest loads it under its own name, so the
plain import fails at collection.

Main Features:
* Session-scoped `shared_pg_dsn` backed by AUDIOMUSE_TEST_DATABASE_URL or an ephemeral
  pgserver instance, for modules that do not shadow it
* `task_status_ddl` / `task_status_jsonb_ddl` expose the one canonical schema
"""

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

TASK_STATUS_DDL = (
    "CREATE TABLE task_status ("
    "id SERIAL PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, parent_task_id TEXT, "
    "task_type TEXT NOT NULL, sub_type_identifier TEXT, status TEXT, "
    "progress INTEGER DEFAULT 0, details TEXT, "
    "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "start_time DOUBLE PRECISION, end_time DOUBLE PRECISION)"
)

TASK_STATUS_JSONB_DDL = TASK_STATUS_DDL.replace("details TEXT,", "details JSONB,")


@pytest.fixture(scope='session')
def task_status_ddl():
    return TASK_STATUS_DDL


@pytest.fixture(scope='session')
def task_status_jsonb_ddl():
    return TASK_STATUS_JSONB_DDL


@pytest.fixture(scope='session')
def shared_pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")
    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.fail(f"AUDIOMUSE_TEST_DATABASE_URL is set but not reachable, refusing to skip: {e}")
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
