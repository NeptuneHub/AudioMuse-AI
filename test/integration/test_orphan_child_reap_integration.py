# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Starting a run clears the child rows every previous run left behind.

clean_up_previous_main_tasks only ever archived parents it could still archive
(PENDING/STARTED/PROGRESS/SUCCESS), so a run that ended FAILURE kept one child
row per album forever: no later run selected that parent, and nothing else in
the codebase deletes children except the global cancel's full-table wipe.

The reclaim is deliberately time-free. Deciding a task is dead because it has
been quiet for N minutes is a deadline, and a big library on old hardware can be
legitimately quiet for a very long time. The signal used instead is the one the
archival loop already trusts: a NEW main task is starting, therefore every
earlier run is over. The only rows spared are those of a live self-managed task
(server_sweep, alchemy_radio), which the archival loop skips by type and which
may legitimately run alongside.

Assertions read through a SECOND connection so an uncommitted DELETE cannot pass:
the code under test shares one connection, where its own uncommitted changes
would be visible to any read issued on it.

Main Features:
* Children of a FAILURE or REVOKED parent, and children whose parent row is gone,
  are reclaimed regardless of the child's own status
* Reclaiming does not depend on how recently anything was written
* Children of a live self-managed parent are never touched
* A live ordinary parent is still archived by the pre-existing loop, so its
  children go with it exactly as before
* The FAILURE parent row itself survives with its error details intact
"""

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = pytest.mark.integration


@pytest.fixture
def task_db(task_status_db):
    return task_status_db


@pytest.fixture(autouse=True)
def no_live_rq_jobs(request, monkeypatch):
    if 'real_rq_liveness' in request.keywords:
        return
    import database

    monkeypatch.setattr(database, '_parents_with_live_jobs', lambda parent_ids: set())


def _insert(db, task_id, status, parent=None, task_type='main_analysis', details=None):
    conn, _ = db
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO task_status "
            "(task_id, parent_task_id, task_type, status, progress, details) "
            "VALUES (%s, %s, %s, %s, 0, %s)",
            (task_id, parent, task_type, status, details),
        )
    conn.commit()


def _committed_ids(db):
    _, verifier = db
    with verifier.cursor() as cur:
        cur.execute("SELECT task_id FROM task_status ORDER BY task_id")
        return [r[0] for r in cur.fetchall()]


def _committed_row(db, task_id):
    _, verifier = db
    with verifier.cursor() as cur:
        cur.execute("SELECT status, details FROM task_status WHERE task_id = %s", (task_id,))
        return cur.fetchone()


def _run_cleanup(monkeypatch, db):
    import database

    conn, _ = db
    monkeypatch.setattr(database, 'get_db', lambda: conn)
    database.clean_up_previous_main_tasks()


class TestReclaimsStrandedChildren:
    def test_children_of_a_failed_parent_are_reclaimed(self, task_db, monkeypatch):
        _insert(task_db, 'parent-failed', 'FAILURE')
        _insert(task_db, 'child-a', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis')
        _insert(task_db, 'child-b', 'STARTED', parent='parent-failed',
                task_type='album_analysis')

        _run_cleanup(monkeypatch, task_db)

        assert _committed_ids(task_db) == ['parent-failed']

    def test_children_stuck_non_terminal_by_a_killed_worker_are_reclaimed(self, task_db, monkeypatch):
        _insert(task_db, 'parent-failed', 'FAILURE')
        _insert(task_db, 'child-stuck', 'PROGRESS', parent='parent-failed',
                task_type='album_analysis')

        _run_cleanup(monkeypatch, task_db)

        assert 'child-stuck' not in _committed_ids(task_db)

    def test_children_whose_parent_row_is_gone_are_reclaimed(self, task_db, monkeypatch):
        _insert(task_db, 'orphan', 'SUCCESS', parent='parent-that-never-existed',
                task_type='clustering_batch')

        _run_cleanup(monkeypatch, task_db)

        assert _committed_ids(task_db) == []

    def test_children_of_a_revoked_parent_are_reclaimed(self, task_db, monkeypatch):
        _insert(task_db, 'parent-revoked', 'REVOKED')
        _insert(task_db, 'child-c', 'SUCCESS', parent='parent-revoked',
                task_type='album_analysis')

        _run_cleanup(monkeypatch, task_db)

        assert _committed_ids(task_db) == ['parent-revoked']

    def test_reclaiming_does_not_wait_for_any_quiet_period(self, task_db, monkeypatch):
        _insert(task_db, 'parent-failed', 'FAILURE')
        _insert(task_db, 'child-just-written', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis')

        _run_cleanup(monkeypatch, task_db)

        assert 'child-just-written' not in _committed_ids(task_db)

    def test_a_failed_parent_keeps_its_row_and_error_details(self, task_db, monkeypatch):
        details = json.dumps({'error': 'boom', 'status_message': 'it broke'})
        _insert(task_db, 'parent-failed', 'FAILURE', details=details)
        _insert(task_db, 'child-a', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis')

        _run_cleanup(monkeypatch, task_db)

        status, stored = _committed_row(task_db, 'parent-failed')
        assert status == 'FAILURE'
        assert json.loads(stored)['error'] == 'boom'


class TestLeavesLiveWorkAlone:
    @pytest.mark.parametrize('live_status', ['PENDING', 'STARTED', 'PROGRESS'])
    def test_a_live_self_managed_parents_children_are_never_touched(
        self, task_db, monkeypatch, live_status
    ):
        _insert(task_db, 'sweep', live_status, task_type='server_sweep')
        _insert(task_db, 'sweep-child', 'SUCCESS', parent='sweep',
                task_type='server_sweep')

        _run_cleanup(monkeypatch, task_db)

        assert 'sweep-child' in _committed_ids(task_db)

    def test_a_finished_self_managed_parents_children_are_reclaimed(self, task_db, monkeypatch):
        _insert(task_db, 'sweep', 'SUCCESS', task_type='server_sweep')
        _insert(task_db, 'sweep-child', 'SUCCESS', parent='sweep',
                task_type='server_sweep')

        _run_cleanup(monkeypatch, task_db)

        assert 'sweep-child' not in _committed_ids(task_db)

    @pytest.mark.parametrize('live_status', ['PENDING', 'STARTED', 'PROGRESS'])
    def test_starting_a_run_still_archives_a_live_previous_parent_and_drops_its_children(
        self, task_db, monkeypatch, live_status
    ):
        _insert(task_db, 'parent-live', live_status)
        _insert(task_db, 'child-live', 'SUCCESS', parent='parent-live',
                task_type='album_analysis')

        _run_cleanup(monkeypatch, task_db)

        assert _committed_row(task_db, 'parent-live')[0] == 'REVOKED'
        assert 'child-live' not in _committed_ids(task_db)


class TestLiveParentProtection:
    @pytest.mark.parametrize('live_status', ['PENDING', 'STARTED', 'PROGRESS'])
    def test_the_prune_alone_never_touches_a_live_parents_children(
        self, task_db, monkeypatch, live_status
    ):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'parent-live', live_status)
        _insert(task_db, 'child-live', 'SUCCESS', parent='parent-live',
                task_type='album_analysis')
        _insert(task_db, 'parent-done', 'FAILURE')
        _insert(task_db, 'child-done', 'SUCCESS', parent='parent-done',
                task_type='album_analysis')

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert 'child-live' in surviving
        assert 'child-done' not in surviving

    def test_a_cron_tick_firing_beside_a_running_analysis_takes_nothing(
        self, task_db, monkeypatch
    ):
        # app_cron calls the prune on ANY minute that fired a row, including an
        # alchemy_radio or plugin tick that never consults get_active_main_task.
        # Nothing but the parent's own status stands between that call and a
        # nightly analysis that is halfway through its albums.
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'nightly-analysis', 'PROGRESS', task_type='main_analysis')
        for i in range(3):
            _insert(task_db, f'album-{i}', 'PROGRESS', parent='nightly-analysis',
                    task_type='album_analysis')
        _insert(task_db, 'radio-tick', 'SUCCESS', task_type='alchemy_radio')

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert 'nightly-analysis' in surviving
        for i in range(3):
            assert f'album-{i}' in surviving

    def test_liveness_not_task_type_is_what_protects(self, task_db, monkeypatch):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'parent-live', 'PROGRESS', task_type='main_clustering')
        _insert(task_db, 'child-live', 'STARTED', parent='parent-live',
                task_type='clustering_batch')

        database.prune_task_status_history()

        assert 'child-live' in _committed_ids(task_db)


class TestSupersededParentRows:
    def test_terminal_roots_are_capped_per_task_type(self, task_db, monkeypatch):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        cap = database.TASK_STATUS_MAX_ROOTS_PER_TYPE
        for i in range(cap + 5):
            _insert(task_db, f'radio-{i:03d}', 'SUCCESS', task_type='alchemy_radio')

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert len(surviving) == cap
        assert surviving[-1] == f'radio-{cap + 4:03d}'
        assert 'radio-000' not in surviving

    def test_a_recent_tombstone_survives_so_a_delayed_retry_still_finds_it(
        self, task_db, monkeypatch
    ):
        # _run_already_finished and the clustering entry guard read these rows to
        # refuse a requeued job for a run that already finished. Deleting the
        # newest one would let a delayed RQ retry resurrect completed work.
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'run-old', 'SUCCESS', task_type='main_analysis')
        _insert(task_db, 'run-new', 'REVOKED', task_type='main_analysis')

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert 'run-new' in surviving
        assert 'run-old' in surviving

    def test_each_task_type_is_capped_independently(self, task_db, monkeypatch):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        cap = database.TASK_STATUS_MAX_ROOTS_PER_TYPE
        for i in range(cap + 3):
            _insert(task_db, f'sweep-{i:03d}', 'SUCCESS', task_type='server_sweep')
        _insert(task_db, 'clean-only', 'SUCCESS', task_type='cleaning')

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert 'clean-only' in surviving
        assert len([s for s in surviving if s.startswith('sweep-')]) == cap

    def test_a_live_parent_is_never_dropped_even_with_a_newer_row(
        self, task_db, monkeypatch
    ):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'run-live', 'PROGRESS', task_type='main_analysis')
        _insert(task_db, 'run-newer', 'SUCCESS', task_type='main_analysis')

        database.prune_task_status_history()

        assert 'run-live' in _committed_ids(task_db)


class TestDurability:
    def test_the_reap_is_committed_not_left_in_an_open_transaction(self, task_db, monkeypatch):
        _insert(task_db, 'parent-failed', 'FAILURE')
        _insert(task_db, 'child-a', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis')

        _run_cleanup(monkeypatch, task_db)

        conn, _ = task_db
        conn.rollback()

        assert 'child-a' not in _committed_ids(task_db)


class TestRqLivenessGuard:
    def test_a_parent_whose_rq_job_is_alive_keeps_its_children(self, task_db, monkeypatch):
        # The janitor stamps FAILURE on a live root when Redis restarts and the job
        # looks missing, while the worker keeps analysing. Only RQ can tell that
        # apart from a genuinely dead run without inventing a timeout.
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        monkeypatch.setattr(
            database, '_parents_with_live_jobs', lambda parent_ids: {'parent-failed'}
        )

        _insert(task_db, 'parent-failed', 'FAILURE')
        _insert(task_db, 'child-a', 'PROGRESS', parent='parent-failed',
                task_type='album_analysis')

        database.prune_task_status_history()

        assert 'child-a' in _committed_ids(task_db)

    @pytest.mark.real_rq_liveness
    def test_an_unreachable_rq_keeps_every_candidate(self, monkeypatch):
        import sys

        import database

        monkeypatch.setitem(sys.modules, 'rq_job_state', None)

        assert database._parents_with_live_jobs(['a', 'b']) == {'a', 'b'}

    @pytest.mark.real_rq_liveness
    def test_no_candidates_never_touches_redis(self, monkeypatch):
        import sys

        import database

        monkeypatch.setitem(sys.modules, 'rq_job_state', None)

        assert database._parents_with_live_jobs([]) == set()
