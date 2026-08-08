# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""There is exactly ONE way to write a placeholder task row.

A placeholder row records a task's intent inside the caller's transaction BEFORE
it has a func, and the ``taskqueue.enqueue`` that follows on the same connection
adopts it, writes the func and commits both together. The adoption only happens
while the row still matches ``_INSERT_JOB``'s ON CONFLICT predicate - func still
NULL, status still live - and that predicate lives in taskqueue/sql.py where the
staging INSERT cannot see it. There used to be two hand-written copies of that
INSERT, one in app_music_servers and one in tasks/multiserver_sync; they agreed
by luck, and the next change to the predicate had to be found in two files
nothing connects. ``database.stage_pending_task_row`` is now the single owner and
``insert_pending_sweep_row`` is a name over it, so this file fails if a third
copy is hand-written anywhere or if the two staging entry points drift apart.

Main Features:
* No module outside database.py writes an ON CONFLICT (task_id) DO NOTHING insert
  into task_status, which is the shape of an adoptable placeholder
* The sweep staging path and the generic one issue the identical statement, with
  the identical status and column tuple, and neither writes a func
* Both open a savepoint and release it, so a second live row is refused by the
  partial unique index without poisoning the caller's transaction
* Both report the refusal as False rather than raising, and roll back only as far
  as their own savepoint
* True means THIS call created the row. ON CONFLICT (task_id) DO NOTHING skips an
  id that is already there without raising, so a helper that returned True
  whenever nothing raised told ``_claim_replacement_sweep`` it owned a slot
  another run already held; every staging test here used a fresh uuid, so nothing
  saw it
"""

import inspect
import json
import os
import re

import psycopg2
import pytest

import config

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)

_SKIPPED_DIRS = {
    '__pycache__', 'node_modules', 'site-packages', 'venv', 'test',
    'build', 'dist', 'native-build', 'pginstall', 'model', 'models',
}


def _is_skipped(name):
    return name.startswith('.') or name in _SKIPPED_DIRS

_PLACEHOLDER_INSERT = re.compile(
    r'INSERT INTO task_status.{0,800}?ON CONFLICT \( ?task_id ?\) DO (NOTHING|UPDATE)',
    re.IGNORECASE,
)


def _flattened(source):
    return re.sub(r'\s+', ' ', source.replace('"', '').replace("'", ''))


def _source_files():
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [d for d in dirnames if not _is_skipped(d)]
        for name in filenames:
            if name.endswith('.py'):
                yield os.path.join(dirpath, name)


def _modules_with_a_placeholder_insert():
    owners = set()
    for path in _source_files():
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            flat = _flattened(handle.read())
        for conflict_action in _PLACEHOLDER_INSERT.findall(flat):
            if conflict_action.upper() == 'NOTHING':
                owners.add(os.path.relpath(path, _REPO_ROOT).replace('\\', '/'))
    return owners


class _RecordingCursor:
    def __init__(self, raise_on_insert=None):
        self.calls = []
        self.raise_on_insert = raise_on_insert
        self.rowcount = -1
        self.task_ids = set()

    def execute(self, statement, params=None):
        self.calls.append((' '.join(statement.split()), params))
        if not statement.lstrip().upper().startswith('INSERT'):
            self.rowcount = -1
            return
        if self.raise_on_insert is not None:
            raise self.raise_on_insert
        task_id = params[0]
        self.rowcount = 0 if task_id in self.task_ids else 1
        self.task_ids.add(task_id)

    @property
    def statements(self):
        return [statement for statement, _params in self.calls]

    @property
    def insert(self):
        return next(
            (call for call in self.calls if call[0].upper().startswith('INSERT')), None
        )


class TestOnlyOneModuleWritesAPlaceholderRow:

    def test_database_py_is_the_only_owner_of_a_do_nothing_task_status_insert(self):
        assert _modules_with_a_placeholder_insert() == {'database.py'}, (
            'a placeholder task row must be written only by '
            'database.stage_pending_task_row: a second hand-written copy has to '
            'agree with the adoption predicate in taskqueue/sql.py that it cannot '
            'see, and the last time there were two they drifted'
        )

    def test_the_sweep_staging_helper_holds_no_insert_of_its_own(self):
        from tasks import multiserver_sync

        body = inspect.getsource(multiserver_sync.insert_pending_sweep_row)
        assert 'INSERT INTO' not in body.upper()
        assert 'SAVEPOINT' not in body.upper()


class TestBothStagingPathsWriteTheSameRow:

    def _staged_by_the_sweep_path(self):
        from tasks import multiserver_sync

        cur = _RecordingCursor()
        staged = multiserver_sync.insert_pending_sweep_row(
            cur, 'task-1', 'Aligning the migrated server.'
        )
        return staged, cur

    def _staged_by_the_generic_path(self):
        import database

        cur = _RecordingCursor()
        staged = database.stage_pending_task_row(
            cur,
            'task-1',
            'server_sweep',
            {
                'message': 'Aligning the migrated server.',
                'status_message': 'Aligning the migrated server.',
            },
        )
        return staged, cur

    def test_both_paths_issue_the_identical_statement(self):
        _sweep_staged, sweep_cur = self._staged_by_the_sweep_path()
        _generic_staged, generic_cur = self._staged_by_the_generic_path()

        assert sweep_cur.insert[0] == generic_cur.insert[0]
        assert sweep_cur.statements == generic_cur.statements

    def test_both_paths_write_the_same_row_but_for_their_own_start_time(self):
        _sweep_staged, sweep_cur = self._staged_by_the_sweep_path()
        _generic_staged, generic_cur = self._staged_by_the_generic_path()

        assert sweep_cur.insert[1][:-1] == generic_cur.insert[1][:-1]

    def test_the_staged_row_is_new_and_carries_no_func(self):
        _staged, cur = self._staged_by_the_sweep_path()

        statement, params = cur.insert
        assert config.TASK_STATUS_NEW in params
        assert 'func' not in statement.lower()
        assert 'ON CONFLICT (task_id) DO NOTHING' in statement

    def test_the_sweep_details_still_carry_message_and_status_message(self):
        _staged, cur = self._staged_by_the_sweep_path()

        details = json.loads(cur.insert[1][3])
        assert details == {
            'message': 'Aligning the migrated server.',
            'status_message': 'Aligning the migrated server.',
        }

    def test_both_paths_report_success(self):
        assert self._staged_by_the_sweep_path()[0] is True
        assert self._staged_by_the_generic_path()[0] is True


class TestTheSavepointStillProtectsTheCallersTransaction:

    def _refused(self):
        from tasks import multiserver_sync

        cur = _RecordingCursor(
            raise_on_insert=psycopg2.errors.UniqueViolation('one live sweep only')
        )
        staged = multiserver_sync.insert_pending_sweep_row(cur, 'task-2', 'Aligning.')
        return staged, cur

    def test_a_second_live_sweep_is_refused_rather_than_raised(self):
        staged, _cur = self._refused()

        assert staged is False

    def test_the_refusal_rolls_back_only_as_far_as_the_helpers_own_savepoint(self):
        _staged, cur = self._refused()

        savepoints = [s for s in cur.statements if 'SAVEPOINT' in s.upper()]
        assert savepoints[0].startswith('SAVEPOINT ')
        assert savepoints[-1] == 'ROLLBACK TO SAVEPOINT ' + savepoints[0].split()[1]
        assert not any(s.upper() == 'ROLLBACK' for s in cur.statements), (
            'the migration stages this row deep inside its own transaction, so a '
            'refused sweep must never roll that transaction back'
        )

    def test_a_successful_stage_releases_the_savepoint_it_opened(self):
        from tasks import multiserver_sync

        cur = _RecordingCursor()
        multiserver_sync.insert_pending_sweep_row(cur, 'task-3', 'Aligning.')

        savepoints = [s for s in cur.statements if 'SAVEPOINT' in s.upper()]
        assert savepoints[0].startswith('SAVEPOINT ')
        assert savepoints[-1] == 'RELEASE SAVEPOINT ' + savepoints[0].split()[1]


class TestStagingAnIdThatAlreadyExistsClaimsNothing:

    def _staged_twice(self, task_id='task-5'):
        import database

        cur = _RecordingCursor()
        details = {'message': 'Aligning.', 'status_message': 'Aligning.'}
        first = database.stage_pending_task_row(cur, task_id, 'server_sweep', details)
        second = database.stage_pending_task_row(cur, task_id, 'server_sweep', details)
        return first, second, cur

    def test_the_first_stage_of_an_id_reports_that_it_created_the_row(self):
        first, _second, _cur = self._staged_twice()

        assert first is True

    def test_the_second_stage_of_the_same_id_reports_that_it_created_nothing(self):
        _first, second, _cur = self._staged_twice()

        assert second is False, (
            'the INSERT is ON CONFLICT (task_id) DO NOTHING, so a second stage of a '
            'live id writes no row and raises nothing; reporting success there tells '
            'the sweep claim it owns a slot another run already holds, and it '
            'enqueues straight over it'
        )

    def test_the_second_stage_still_writes_its_insert_and_releases_its_savepoint(self):
        _first, _second, cur = self._staged_twice()

        inserts = [s for s in cur.statements if s.upper().startswith('INSERT')]
        assert len(inserts) == 2
        assert cur.statements[-1].startswith('RELEASE SAVEPOINT '), (
            'a skipped insert is not an error, so the helper must close the '
            'savepoint it opened rather than leaving it open on the caller'
        )
        assert not any(s.upper().startswith('ROLLBACK') for s in cur.statements)

    def test_the_verdict_is_read_from_the_insert_and_not_after_the_release(self):
        import database

        cur = _RecordingCursor()
        created = database.stage_pending_task_row(
            cur, 'task-6', 'server_sweep', {'message': 'Aligning.'}
        )

        assert created is True
        assert cur.rowcount == -1, (
            'RELEASE SAVEPOINT is a utility statement and leaves rowcount at -1, so '
            'the row count of the INSERT has to be taken before it runs'
        )

    def test_the_sweep_name_forwards_the_same_verdict_it_was_given(self):
        from tasks import multiserver_sync

        cur = _RecordingCursor()
        first = multiserver_sync.insert_pending_sweep_row(cur, 'task-7', 'Aligning.')
        second = multiserver_sync.insert_pending_sweep_row(cur, 'task-7', 'Aligning.')

        assert (first, second) == (True, False)


class TestARefusedStatusWriteDoesNotCommitTheCallersWork:

    def _save(self, rowcount, monkeypatch):
        import database

        class _Cursor:
            def __init__(self):
                self.rowcount = rowcount

            def execute(self, *_args, **_kwargs):
                return None

            def fetchone(self):
                return (None,)

            def close(self):
                return None

        class _Conn:
            def __init__(self):
                self.commits = 0
                self.rollbacks = 0

            def cursor(self, *_args, **_kwargs):
                return _Cursor()

            def commit(self):
                self.commits += 1

            def rollback(self):
                self.rollbacks += 1

        conn = _Conn()
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        monkeypatch.setattr(database, '_maybe_record_task_history', lambda *a, **k: None)
        monkeypatch.setattr(database, '_collapse_finished_task', lambda *a, **k: 0)
        written = database.save_task_status('task-4', 'main_clustering')
        return written, conn

    def test_a_write_that_landed_commits(self, monkeypatch):
        written, conn = self._save(1, monkeypatch)

        assert written is True
        assert (conn.commits, conn.rollbacks) == (1, 0)

    def test_a_write_the_row_refused_rolls_back_instead_of_committing(self, monkeypatch):
        written, conn = self._save(0, monkeypatch)

        assert written is False
        assert (conn.commits, conn.rollbacks) == (0, 1), (
            'the clustering parent reaps a finished batch on this connection and '
            'lets the status write publish it, so committing a write that did not '
            'land destroys the reaped result with nothing left recording it'
        )


if __name__ == '__main__':
    pytest.main([__file__])
