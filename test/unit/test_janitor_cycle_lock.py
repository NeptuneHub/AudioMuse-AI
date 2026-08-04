# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The janitor/Cancel cycle lock: database.janitor_cycle_lock.

These live outside test_app_cancel.py on purpose: that module has an autouse
fixture that stubs database.janitor_cycle_lock out, so every assertion here would
have silently passed against the stub instead of the real helper.

Cancel must serialize with the RQ janitor - otherwise a janitor holding a stale
retries snapshot requeues a job after Cancel has emptied the queues. But Cancel is
also the one operation that must never hang, and the janitor can itself sit behind
a multi-minute migration transaction, so the wait carries a deadline.

Main Features:
* The bounded blocking mode gives up on its deadline instead of hanging forever
* A retry that wins the lock returns immediately and still unlocks on exit
* Janitors keep their non-blocking election and skip a contended pass
* A lock we never acquired is never unlocked, so the real owner keeps it
"""

from unittest.mock import MagicMock


class _ScriptedCursor:
    def __init__(self, grants):
        self._grants = list(grants)
        self.sqls = []

    def execute(self, sql, params=None):
        self.sqls.append(" ".join(sql.split()))

    def fetchone(self):
        return (self._grants.pop(0) if self._grants else False,)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _db(grants):
    cur = _ScriptedCursor(grants)
    db = MagicMock()
    db.cursor.return_value = cur
    return db, cur


def test_timed_cycle_lock_gives_up_instead_of_blocking_forever(monkeypatch):
    import database

    db, cur = _db([])
    monkeypatch.setattr(database.time, 'sleep', lambda s: None)

    with database.janitor_cycle_lock(db, blocking=True, timeout_seconds=0.05) as owns:
        assert owns is False

    # An unbounded pg_advisory_lock() is what hung the gunicorn thread serving
    # Cancel behind a janitor that was itself waiting on a long migration.
    assert cur.sqls
    assert all('SELECT pg_advisory_lock' not in s for s in cur.sqls)
    assert any('pg_try_advisory_lock' in s for s in cur.sqls)
    assert all('pg_advisory_unlock' not in s for s in cur.sqls)


def test_timed_cycle_lock_returns_as_soon_as_a_retry_wins(monkeypatch):
    import database

    db, cur = _db([False, False, True])
    monkeypatch.setattr(database.time, 'sleep', lambda s: None)

    with database.janitor_cycle_lock(db, blocking=True, timeout_seconds=30) as owns:
        assert owns is True

    assert len([s for s in cur.sqls if 'pg_try_advisory_lock' in s]) == 3
    assert any('pg_advisory_unlock' in s for s in cur.sqls)


def test_untimed_blocking_mode_still_uses_the_plain_waiting_lock():
    import database

    db, cur = _db([])

    with database.janitor_cycle_lock(db, blocking=True) as owns:
        assert owns is True

    assert any('SELECT pg_advisory_lock' in s for s in cur.sqls)
    assert all('pg_try_advisory_lock' not in s for s in cur.sqls)


def test_janitors_keep_non_blocking_election_and_skip_a_contended_pass():
    import database

    db, cur = _db([False])

    with database.janitor_cycle_lock(db) as owns:
        assert owns is False

    assert any('pg_try_advisory_lock' in s for s in cur.sqls)
    # Unlocking a lock we never took would release the real owner's.
    assert all('pg_advisory_unlock' not in s for s in cur.sqls)
