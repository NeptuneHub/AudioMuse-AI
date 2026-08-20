# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The queue's SQL speaks the status vocabulary config defines, and only that.

Every statement in ``taskqueue/sql.py`` is interpolated from the five
``config.TASK_STATUS_*`` constants, so renaming one there moves the claim, the
reclaim, the reap and the indexes together. Hardcoding a spelling in even one of
them is silent: the index names and the Python comparisons follow config, the
statement does not, and the queue simply stops claiming or reaping rows.

The generated text is pinned byte for byte as well as by spelling, because the
whole point of the interpolation is that it produces exactly the SQL the queue
shipped with - a stray space inside an IN list is a different statement to every
test that greps for one.

Main Features:
* No statement carries a status spelling config does not define
* The pre-queue spellings survive only in the one-time migration
* Renaming a status in config moves every statement, index DDL included
* The shipped predicates keep their exact text
* One parameterised statement drops the stale indexes of every family
"""

import importlib
import re

import pytest
from unittest.mock import patch

import config
from taskqueue import sql

LEGACY_SPELLINGS = ('PENDING', 'STARTED', 'PROGRESS', 'FAILURE')

RENAMED = {
    'TASK_STATUS_NEW': 'QUEUED',
    'TASK_STATUS_RUNNING': 'BUSY',
    'TASK_STATUS_SUCCESS': 'DONE',
    'TASK_STATUS_FAIL': 'BROKEN',
    'TASK_STATUS_REVOKED': 'DROPPED',
}

QUOTED_CAPS = re.compile(r"'([A-Z][A-Z]+)'")


def _sql_strings(module):
    found = []
    for name, value in vars(module).items():
        if name.startswith('__'):
            continue
        if isinstance(value, str):
            found.append((name, value))
        elif isinstance(value, tuple):
            for index, item in enumerate(value):
                if isinstance(item, str):
                    found.append((f"{name}[{index}]", item))
    return found


class _RecordingCursor:
    def __init__(self):
        self.calls = []
        self.rows = []

    def execute(self, statement, params=None):
        self.calls.append((statement, params))

    def fetchone(self):
        return None

    def fetchall(self):
        return list(self.rows)


@pytest.fixture
def renamed(monkeypatch):
    for name, spelling in RENAMED.items():
        monkeypatch.setattr(config, name, spelling)
    monkeypatch.setattr(
        config, 'TASK_STATUS_LIVE', (RENAMED['TASK_STATUS_NEW'], RENAMED['TASK_STATUS_RUNNING'])
    )
    monkeypatch.setattr(
        config,
        'TASK_STATUS_TERMINAL',
        (
            RENAMED['TASK_STATUS_SUCCESS'],
            RENAMED['TASK_STATUS_FAIL'],
            RENAMED['TASK_STATUS_REVOKED'],
        ),
    )
    importlib.reload(sql)
    yield sql
    monkeypatch.undo()
    importlib.reload(sql)


class TestEveryStatementSpeaksTheConfigVocabulary:
    def test_no_statement_carries_a_status_config_does_not_define(self):
        allowed = set(config.TASK_STATUS_LIVE) | set(config.TASK_STATUS_TERMINAL)
        allowed.update(LEGACY_SPELLINGS)

        for name, statement in _sql_strings(sql):
            for spelling in QUOTED_CAPS.findall(statement):
                assert spelling in allowed, (
                    f"{name} quotes {spelling!r}, which config does not define"
                )

    def test_the_pre_queue_spellings_survive_only_in_the_one_time_migration(self):
        for name, statement in _sql_strings(sql):
            for legacy in LEGACY_SPELLINGS:
                if f"'{legacy}'" in statement:
                    assert name.startswith('_MIGRATE_STATUSES'), (
                        f"{name} still compares against the pre-queue spelling {legacy!r}"
                    )

    def test_the_migration_rewrites_every_legacy_spelling(self):
        migration = '\n'.join(sql._MIGRATE_STATUSES)

        for legacy in LEGACY_SPELLINGS:
            assert f"'{legacy}'" in migration or f"''{legacy}''" in migration


class TestRenamingAStatusInConfigMovesEveryStatement:
    def test_the_claim_moves_the_renamed_new_to_the_renamed_running(self, renamed):
        assert "SET status='BUSY'" in renamed._CLAIM
        assert "WHERE status='QUEUED'" in renamed._CLAIM

    def test_every_live_predicate_follows_the_rename(self, renamed):
        for statement in (
            renamed._INSERT_JOB,
            renamed._RETIRE_SURPLUS_LIVE_ROOTS,
            renamed._REQUEUE_OR_FAIL,
            renamed._PUT_SHARED,
            renamed._LIVE_CHILDREN,
            renamed.TERMINAL_AND_NOT_A_LIVE_PARENTS_CHILD,
        ):
            assert "IN ('QUEUED','BUSY')" in statement

    def test_every_terminal_predicate_follows_the_rename(self, renamed):
        for statement in (
            renamed._REAP_CHILDREN,
            renamed.TERMINAL_AND_NOT_A_LIVE_PARENTS_CHILD,
            renamed._CLEAR_TASK_STATUS,
        ):
            assert "IN ('DONE','BROKEN','DROPPED')" in statement

    def test_the_index_ddl_follows_the_rename_too(self, renamed):
        for statement in (
            renamed._ONE_LIVE_MAIN_INDEX,
            renamed._ONE_LIVE_SWEEP_INDEX,
            renamed._LIVE_INDEX,
        ):
            assert "IN ('QUEUED', 'BUSY')" in statement
        assert "WHERE status = 'QUEUED'" in renamed._CLAIM_INDEX

    def test_the_reclaim_fails_a_row_under_the_renamed_spelling(self, renamed):
        assert "THEN 'QUEUED' ELSE 'BROKEN' END" in renamed._REQUEUE_OR_FAIL
        assert "SET status='DROPPED'" in renamed._RETIRE_SURPLUS_LIVE_ROOTS

    def test_not_one_statement_keeps_the_spelling_config_no_longer_uses(self, renamed):
        for name, statement in _sql_strings(renamed):
            for spelling in ('NEW', 'RUNNING', 'SUCCESS', 'FAIL', 'REVOKED'):
                assert f"'{spelling}'" not in statement, (
                    f"{name} still hardcodes {spelling!r} after config renamed it"
                )

    def test_the_migration_still_targets_the_legacy_spellings(self, renamed):
        migration = '\n'.join(renamed._MIGRATE_STATUSES)

        assert "SET status='QUEUED' WHERE status='PENDING'" in migration
        assert "IN ('STARTED','PROGRESS')" in migration
        assert "WHERE status='FAILURE'" in migration


class TestTheShippedPredicatesKeepTheirExactText:
    def test_the_live_list_is_the_compact_two_status_form(self):
        assert sql._LIVE_IN_LIST == "'NEW','RUNNING'"

    def test_the_terminal_list_is_the_compact_three_status_form(self):
        assert sql._TERMINAL_IN_LIST == "'SUCCESS','FAIL','REVOKED'"

    def test_the_index_ddl_keeps_the_spaced_list_it_was_created_with(self):
        assert sql._LIVE_STATUS_SQL == "'NEW', 'RUNNING'"

    def test_the_claim_still_moves_new_to_running(self):
        assert "SET status='RUNNING'" in sql._CLAIM
        assert "WHERE status='NEW' AND queue_name = %s" in sql._CLAIM

    def test_the_main_retire_keeps_only_the_newest_live_main_root(self):
        cur = _RecordingCursor()
        # ORDER BY id DESC means the first row is the newest; the retire keeps
        # it and revokes every older live main root.
        cur.rows = [('fingerprint-live',), ('analysis-live',)]

        with (
            patch('taskqueue.sql.try_hold', return_value=True),
            patch('taskqueue.sql.release'),
        ):
            assert sql._retire_surplus_main_live_roots(cur) is True

        update = next(
            statement for statement, _ in cur.calls
            if statement.startswith('UPDATE task_status')
        )
        assert "task_id = ANY(%s)" in update
        assert cur.calls[-1][1] == (config.TASK_STATUS_REVOKED, ['analysis-live'])

    def test_the_main_retire_never_revokes_a_row_a_worker_is_still_running(self):
        cur = _RecordingCursor()
        cur.rows = [('analysis-live',), ('fingerprint-live',)]

        def _try_hold(c, task_id):
            return task_id != 'fingerprint-live'

        with patch('taskqueue.sql.try_hold', side_effect=_try_hold):
            assert sql._retire_surplus_main_live_roots(cur) is True

        assert cur.calls[-1][1] == (config.TASK_STATUS_REVOKED, ['analysis-live'])

    def test_the_main_retire_defers_when_two_rows_are_still_running(self):
        cur = _RecordingCursor()
        cur.rows = [('analysis-live',), ('fingerprint-live',)]

        with patch('taskqueue.sql.try_hold', return_value=False):
            assert sql._retire_surplus_main_live_roots(cur) is False

        assert all(
            not statement.startswith('UPDATE task_status')
            for statement, _ in cur.calls
        )

    def test_the_terminal_write_still_requires_a_running_row(self):
        assert "AND status = 'RUNNING'" in sql._FINISH_TASK

    def test_the_uncharged_requeue_still_moves_running_back_to_new(self):
        assert "SET status='NEW'" in sql._REQUEUE_UNCHARGED
        assert "AND status='RUNNING'" in sql._REQUEUE_UNCHARGED

    def test_the_reap_still_deletes_only_terminal_children(self):
        assert "status IN ('SUCCESS','FAIL','REVOKED')" in sql._REAP_CHILDREN

    def test_the_wipe_still_spares_a_live_parents_child(self):
        assert "status IN ('SUCCESS','FAIL','REVOKED')" in sql._CLEAR_TASK_STATUS
        assert "live.status IN ('NEW','RUNNING')" in sql._CLEAR_TASK_STATUS

    def test_the_orphan_scan_still_looks_at_running_rows(self):
        assert "WHERE t.status='RUNNING'" in sql._RUNNING_TASKS

    def test_the_worker_snapshot_still_joins_on_running_rows(self):
        assert "ON t.status = 'RUNNING'" in sql._WORKER_SNAPSHOT


class TestOneParameterisedStatementDropsEveryIndexFamily:
    def test_the_like_pattern_is_a_parameter_not_a_literal(self):
        assert 'AND indexname LIKE %s' in sql._DROP_STALE_INDEXES
        assert 'idx_task_status_one_live' not in sql._DROP_STALE_INDEXES

    def test_the_percent_escape_for_the_format_call_survives(self):
        assert "format('DROP INDEX IF EXISTS %%I', stale)" in sql._DROP_STALE_INDEXES

    def test_each_prefix_is_the_prefix_of_the_index_it_protects(self):
        assert sql.MAIN_INDEX_NAME.startswith(sql.MAIN_INDEX_PREFIX + '_')
        assert sql.SWEEP_INDEX_NAME.startswith(sql.SWEEP_INDEX_PREFIX + '_')
        assert sql.LIVE_INDEX_NAME.startswith(sql.LIVE_INDEX_PREFIX + '_')

    def test_every_family_runs_the_same_statement_with_its_own_prefix(self):
        cur = _RecordingCursor()

        sql.ensure_schema(cur)

        drops = [
            params for statement, params in cur.calls
            if statement == sql._DROP_STALE_INDEXES
        ]
        assert drops == [
            (sql.MAIN_INDEX_PREFIX + '%', sql.MAIN_INDEX_NAME),
            (sql.SWEEP_INDEX_PREFIX + '%', sql.SWEEP_INDEX_NAME),
            (sql.LIVE_INDEX_PREFIX + '%', sql.LIVE_INDEX_NAME),
        ]
