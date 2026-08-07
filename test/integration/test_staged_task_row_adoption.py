# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""A staged placeholder row is still adoptable by the enqueue that follows it.

A start path that must revoke, insert and enqueue atomically writes its task row
BEFORE it has a func: the row records the intent inside the caller's transaction,
and the enqueue on the same connection fills in the func and commits both
together. That only works because ``_INSERT_JOB``'s ON CONFLICT clause updates a
row whose ``func IS NULL`` and whose status is still live - a private predicate
the staging INSERT cannot see. When the two drifted apart the staged row was
committed with no func, nothing claimed it, and the alignment sat visibly
"queued" until the 30-minute stale sweep failed it.

The contract now has one owner, ``database.stage_pending_task_row``, and this is
the test that fails if a staged row stops being adoptable: narrowing the guard in
taskqueue/sql.py, or writing a status outside ``TASK_STATUS_LIVE`` in the helper,
breaks it. It runs against a real Postgres because ON CONFLICT ... WHERE and the
partial unique index are both evaluated by the server, not by the driver.

Main Features:
* The staged row carries no func, so nothing can claim it yet
* The very next insert_job adopts it, keeps its task_id and gives it a func
* A worker then claims the adopted row with the func the enqueue wrote
* A row that already went terminal is NOT adopted, which is why the status matters
* A second live sweep is refused by the index without poisoning the transaction
"""

import os
import sys
import uuid

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

pytestmark = pytest.mark.integration

import config  # noqa: E402
import database  # noqa: E402
from taskqueue import sql  # noqa: E402

SWEEP_FUNC = 'tasks.multiserver_sync.sweep_all_secondary_servers'


@pytest.fixture
def queue_db(shared_pg_dsn, task_status_ddl):
    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        cur.execute(task_status_ddl)
    conn.autocommit = False
    with conn.cursor() as cur:
        sql.ensure_schema(cur)
    conn.commit()
    yield conn
    conn.close()


def _stage(conn, task_id, task_type=None):
    with conn.cursor() as cur:
        return database.stage_pending_task_row(
            cur,
            task_id,
            task_type or sql.SWEEP_TASK_TYPE,
            {'message': 'Server alignment queued for all servers.'},
        )


def _adopt(conn, task_id, task_type=None):
    with conn.cursor() as cur:
        return sql.insert_job(
            cur,
            task_id,
            task_type or sql.SWEEP_TASK_TYPE,
            SWEEP_FUNC,
            kwargs={'task_id': task_id},
            queue=sql.QUEUE_HIGH,
        )


def _row(conn, task_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, func, payload, queue_name, details FROM task_status "
            "WHERE task_id = %s",
            (task_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        'status': row[0],
        'func': row[1],
        'payload': row[2],
        'queue_name': row[3],
        'details': row[4],
    }


class TestAStagedRowIsAdopted:

    def test_the_staged_row_carries_no_func_so_nothing_can_run_it_yet(self, queue_db):
        task_id = str(uuid.uuid4())
        assert _stage(queue_db, task_id) is True

        row = _row(queue_db, task_id)
        assert row['func'] is None
        assert row['status'] in config.TASK_STATUS_LIVE

    def test_the_staged_row_is_not_claimable_before_the_enqueue(self, queue_db):
        task_id = str(uuid.uuid4())
        _stage(queue_db, task_id)
        queue_db.commit()

        with queue_db.cursor() as cur:
            assert sql.claim(cur, sql.QUEUE_HIGH, 0.0, worker_id='w1') is None

    def test_the_next_insert_job_adopts_it_instead_of_being_refused(self, queue_db):
        task_id = str(uuid.uuid4())
        _stage(queue_db, task_id)

        assert _adopt(queue_db, task_id) is True

    def test_the_adopted_row_keeps_its_task_id_and_gains_the_func(self, queue_db):
        task_id = str(uuid.uuid4())
        _stage(queue_db, task_id)
        _adopt(queue_db, task_id)
        queue_db.commit()

        row = _row(queue_db, task_id)
        assert row['func'] == SWEEP_FUNC
        assert row['status'] == config.TASK_STATUS_NEW
        assert row['queue_name'] == sql.QUEUE_HIGH

    def test_the_staged_details_survive_the_adoption(self, queue_db):
        task_id = str(uuid.uuid4())
        _stage(queue_db, task_id)
        _adopt(queue_db, task_id)
        queue_db.commit()

        assert 'Server alignment queued for all servers.' in _row(queue_db, task_id)['details']

    def test_a_worker_then_claims_the_adopted_row(self, queue_db):
        task_id = str(uuid.uuid4())
        _stage(queue_db, task_id)
        _adopt(queue_db, task_id)
        queue_db.commit()

        with queue_db.cursor() as cur:
            claimed = sql.claim(cur, sql.QUEUE_HIGH, 0.0, worker_id='w1')
        queue_db.commit()

        assert claimed['task_id'] == task_id
        assert claimed['func'] == SWEEP_FUNC
        assert claimed['kwargs'] == {'task_id': task_id}


class TestTheGuardStillHasTeeth:

    def test_a_row_that_already_went_terminal_is_not_adopted(self, queue_db):
        task_id = str(uuid.uuid4())
        _stage(queue_db, task_id)
        with queue_db.cursor() as cur:
            cur.execute(
                "UPDATE task_status SET status = %s WHERE task_id = %s",
                (config.TASK_STATUS_REVOKED, task_id),
            )

        assert _adopt(queue_db, task_id) is False
        assert _row(queue_db, task_id)['func'] is None

    def test_a_row_that_already_has_a_func_is_not_re_adopted(self, queue_db):
        task_id = str(uuid.uuid4())
        _stage(queue_db, task_id)
        _adopt(queue_db, task_id)

        assert _adopt(queue_db, task_id) is False


class TestASecondLiveSweepIsRefusedNotRaised:

    def test_the_second_stage_reports_failure(self, queue_db):
        _stage(queue_db, str(uuid.uuid4()))

        assert _stage(queue_db, str(uuid.uuid4())) is False

    def test_the_transaction_is_still_usable_afterwards(self, queue_db):
        first = str(uuid.uuid4())
        _stage(queue_db, first)
        _stage(queue_db, str(uuid.uuid4()))

        assert _adopt(queue_db, first) is True
        queue_db.commit()
        assert _row(queue_db, first)['func'] == SWEEP_FUNC
