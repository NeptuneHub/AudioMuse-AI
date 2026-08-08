# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The advisory locks the queue depends on, and the properties reviews miss.

Two of these are invisible in a diff. The start lock is deliberately SESSION
scoped because clean_up_previous_main_tasks commits in the middle of the sequence
it guards, so a pg_advisory_xact_lock typo would still pass every other test
while quietly reopening the double-start gap. And the same key is spelled in two
modules on purpose - the queue must serialize against a Flask process still
holding the legacy lock during a rolling upgrade - which only works while the two
numbers agree.

Main Features:
* The queue and the database spell the one start-lock key identically
* The database-side start lock is session scoped, the queue-side one is per txn
* A maintenance process that loses the election does no work at all
* The maintenance lock is released even when a step of the cycle raises
"""

from unittest.mock import MagicMock

import pytest

import database
from taskqueue import maintenance
from taskqueue import sql


class TestTheStartLockIsOneKeyOnBothSidesOfTheBoundary:
    def test_the_queue_and_the_database_spell_the_same_key(self):
        assert sql.START_LOCK_KEY == database.MAIN_TASK_START_LOCK_KEY, (
            'Cancel, the legacy Flask start path and the queue enqueue all order '
            'against this one number; two spellings that drift stop excluding '
            'each other and nothing raises'
        )


class TestTheDatabaseStartLockSurvivesACommit:
    def test_it_is_session_scoped_not_transaction_scoped(self):
        assert 'pg_advisory_lock' in database._ADVISORY_LOCK_SQL
        assert 'xact' not in database._ADVISORY_LOCK_SQL, (
            'clean_up_previous_main_tasks commits inside the guarded sequence, and '
            'a transaction lock would be dropped by that commit'
        )

    def test_it_is_released_explicitly_rather_than_left_to_the_connection(self):
        assert 'pg_advisory_unlock' in database._ADVISORY_UNLOCK_SQL


class TestTheQueueStartLockIsScopedToItsOwnTransaction:
    def test_take_start_lock_uses_a_transaction_lock(self):
        recorded = []
        cur = MagicMock()
        cur.execute.side_effect = lambda statement, params=None: recorded.append(statement)

        sql.take_start_lock(cur)

        assert 'pg_advisory_xact_lock' in recorded[0], (
            'the queue takes this inside the enqueue transaction, where the commit '
            'that ends the transaction is exactly when the lock should go'
        )


class _Cycle:
    def __init__(self, monkeypatch, elected=True, failing_step=None):
        self.conn = MagicMock()
        self.released = []
        self.ran = []
        monkeypatch.setattr(maintenance.sql, 'try_maintenance_lock', lambda _cur: elected)
        monkeypatch.setattr(
            maintenance.sql, 'release_maintenance_lock',
            lambda _cur: self.released.append(True),
        )
        for step in (
            'reclaim_orphans', 'fail_stale_inline_rows', 'clear_terminal_shared_payloads',
        ):
            monkeypatch.setattr(
                maintenance, step, self._make_step(step, failing_step)
            )
        monkeypatch.setattr(
            maintenance, 'recover_migration_handshakes',
            self._make_step('recover_migration_handshakes', failing_step, takes_conn=False),
        )

    def _make_step(self, name, failing_step, takes_conn=True):
        def step(*_args):
            self.ran.append(name)
            if name == failing_step:
                raise RuntimeError(f'{name} blew up')
            return 0
        return step


class TestOnlyTheElectedMaintenanceProcessRunsTheCycle:
    def test_a_lost_election_does_no_work(self, monkeypatch):
        cycle = _Cycle(monkeypatch, elected=False)

        assert maintenance.run_cycle(cycle.conn) is False
        assert cycle.ran == [], (
            'every worker container runs this loop; without the election they would '
            'all reclaim and prune the same rows at once'
        )

    def test_a_lost_election_releases_nothing_it_never_took(self, monkeypatch):
        cycle = _Cycle(monkeypatch, elected=False)

        maintenance.run_cycle(cycle.conn)

        assert cycle.released == []

    def test_the_winner_runs_the_whole_cycle_and_releases(self, monkeypatch):
        cycle = _Cycle(monkeypatch)

        assert maintenance.run_cycle(cycle.conn) is True
        assert cycle.ran == [
            'reclaim_orphans', 'fail_stale_inline_rows',
            'recover_migration_handshakes', 'clear_terminal_shared_payloads',
        ]
        assert cycle.released == [True]

    def test_a_cycle_that_is_not_due_for_retention_only_reclaims(self, monkeypatch):
        cycle = _Cycle(monkeypatch)

        assert maintenance.run_cycle(cycle.conn, with_retention=False) is True
        assert cycle.ran == ['reclaim_orphans'], (
            'reclaim runs every few seconds so a dead worker is noticed in seconds; '
            'the full-table retention passes must not run at that cadence'
        )
        assert cycle.released == [True]


class TestAFailedStepStillReleasesTheMaintenanceLock:
    @pytest.mark.parametrize('failing_step', [
        'reclaim_orphans', 'fail_stale_inline_rows',
        'recover_migration_handshakes', 'clear_terminal_shared_payloads',
    ])
    def test_the_lock_never_leaks_to_the_next_cycle(self, monkeypatch, failing_step):
        cycle = _Cycle(monkeypatch, failing_step=failing_step)

        with pytest.raises(RuntimeError):
            maintenance.run_cycle(cycle.conn)

        assert cycle.released == [True], (
            'the lock is session scoped on a long-lived connection, so leaking it '
            'once means this process wins every future election and never works'
        )
