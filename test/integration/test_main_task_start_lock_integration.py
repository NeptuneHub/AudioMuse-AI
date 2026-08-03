# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Serializing the check-cleanup-claim sequence every main-task start runs.

Starting a main task is four separate database operations: ask whether one is
already running, archive the previous run, write the PENDING row that claims the
slot, then enqueue. Without a lock across the first three, two starters (a double
click, or a cron tick landing on a manual start) could both read "nothing
running" before either had written its row, and both launch - or one archival
could revoke the row the other had just created.

The lock is deliberately SESSION scoped, not transaction scoped:
clean_up_previous_main_tasks commits in the middle of the sequence, and a
transaction lock would be dropped by that commit, reopening the gap it exists to
close. That property is the one worth pinning, because it is invisible in code
review and a `pg_advisory_xact_lock` typo would still pass every other test.

Mutual exclusion cannot be shown against a mocked cursor, so these run against a
real Postgres and probe from a SECOND connection.

Main Features:
* A second connection cannot take the lock while the context manager holds it
* The lock is released when the block exits, including on an exception
* A commit inside the block does not drop the lock
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = pytest.mark.integration


def _try_lock_from_other_connection(verifier, key):
    with verifier.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (key,))
        got = bool(cur.fetchone()[0])
        if got:
            cur.execute("SELECT pg_advisory_unlock(%s)", (key,))
    return got


class TestMainTaskStartLock:
    def test_a_second_starter_cannot_take_the_lock_while_it_is_held(
        self, task_status_db, monkeypatch
    ):
        import database

        conn, verifier = task_status_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        key = database.MAIN_TASK_START_LOCK_KEY

        with database.main_task_start_lock():
            assert _try_lock_from_other_connection(verifier, key) is False

        assert _try_lock_from_other_connection(verifier, key) is True

    def test_a_commit_inside_the_block_does_not_drop_the_lock(
        self, task_status_db, monkeypatch
    ):
        import database

        conn, verifier = task_status_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        key = database.MAIN_TASK_START_LOCK_KEY

        with database.main_task_start_lock():
            conn.commit()
            assert _try_lock_from_other_connection(verifier, key) is False

        assert _try_lock_from_other_connection(verifier, key) is True

    def test_the_lock_is_released_when_the_block_raises(self, task_status_db, monkeypatch):
        import database

        conn, verifier = task_status_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        key = database.MAIN_TASK_START_LOCK_KEY

        with pytest.raises(RuntimeError):
            with database.main_task_start_lock():
                raise RuntimeError("start failed")

        assert _try_lock_from_other_connection(verifier, key) is True

    def test_the_real_start_sequence_holds_the_lock_across_its_own_commit(
        self, task_status_db, monkeypatch
    ):
        import database

        conn, verifier = task_status_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        key = database.MAIN_TASK_START_LOCK_KEY

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO task_status (task_id, task_type, status, progress) "
                "VALUES ('old-run', 'main_analysis', 'SUCCESS', 100)"
            )
        conn.commit()

        with database.main_task_start_lock():
            # This is the call that commits mid-sequence.
            database.clean_up_previous_main_tasks()
            assert _try_lock_from_other_connection(verifier, key) is False

        assert _try_lock_from_other_connection(verifier, key) is True
