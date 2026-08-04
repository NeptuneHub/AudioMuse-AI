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
        monkeypatch.setattr(database, '_parents_with_live_jobs', lambda _ids: set())

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
    def test_only_the_newest_terminal_root_per_task_type_survives(
        self, task_db, monkeypatch
    ):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        monkeypatch.setattr(database, '_parents_with_live_jobs', lambda _ids: set())

        for i in range(25):
            _insert(task_db, f'radio-{i:03d}', 'SUCCESS', task_type='alchemy_radio')

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert surviving == ['radio-024']

    def test_the_newest_tombstone_survives_so_a_delayed_retry_still_finds_it(
        self, task_db, monkeypatch
    ):
        # _run_already_finished and the clustering entry guard read this row to
        # refuse a requeued job for a run that already finished. Deleting the
        # newest one would let a delayed RQ retry resurrect completed work.
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        monkeypatch.setattr(database, '_parents_with_live_jobs', lambda _ids: set())

        _insert(task_db, 'run-old', 'SUCCESS', task_type='main_analysis')
        _insert(task_db, 'run-new', 'REVOKED', task_type='main_analysis')

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert 'run-new' in surviving
        assert 'run-old' not in surviving

    def test_every_task_type_keeps_its_own_recap(self, task_db, monkeypatch):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        monkeypatch.setattr(database, '_parents_with_live_jobs', lambda _ids: set())

        for i in range(5):
            _insert(task_db, f'sweep-{i:03d}', 'SUCCESS', task_type='server_sweep')
        _insert(task_db, 'clean-only', 'SUCCESS', task_type='cleaning')

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert 'clean-only' in surviving
        assert [s for s in surviving if s.startswith('sweep-')] == ['sweep-004']

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

    def test_a_superseded_root_is_kept_while_its_rq_job_is_still_live(
        self, task_db, monkeypatch
    ):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        for i in range(3):
            _insert(task_db, f'guarded-{i:03d}', 'FAILURE', task_type='main_analysis')
        # The janitor stamps FAILURE on a live root when Redis blips, so a
        # terminal status is not proof the worker stopped.
        monkeypatch.setattr(
            database, '_parents_with_live_jobs', lambda _ids: {'guarded-000'}
        )

        database.prune_task_status_history()

        surviving = _committed_ids(task_db)
        assert 'guarded-000' in surviving
        assert 'guarded-001' not in surviving
        assert 'guarded-002' in surviving


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


class TestEndOfTaskCollapse:
    """A finished root keeps its one-line recap and nothing else.

    The rule the owner requires is: START of a task wipes the previous run, END of
    a task leaves ONE line and drops the rest. This is the END half, and it fires
    the moment the root writes SUCCESS/FAILURE/REVOKED - no timeout, no cap.
    """

    def test_a_finished_root_drops_its_children_immediately(self, task_db, monkeypatch):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'parent-run', 'PROGRESS', task_type='main_analysis')
        _insert(task_db, 'album-1', 'SUCCESS', parent='parent-run',
                task_type='album_analysis')
        _insert(task_db, 'album-2', 'SUCCESS', parent='parent-run',
                task_type='album_analysis')

        database.save_task_status(
            'parent-run', 'main_analysis', database.TASK_STATUS_SUCCESS,
            progress=100, details={'message': 'done'},
        )

        surviving = _committed_ids(task_db)
        assert surviving == ['parent-run']

    def test_finishing_also_drops_the_previous_run_of_the_same_type(
        self, task_db, monkeypatch
    ):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'old-run', 'SUCCESS', task_type='main_analysis')
        _insert(task_db, 'old-album', 'SUCCESS', parent='old-run',
                task_type='album_analysis')
        _insert(task_db, 'new-run', 'PROGRESS', task_type='main_analysis')
        _insert(task_db, 'other-type', 'SUCCESS', task_type='cleaning')

        database.save_task_status(
            'new-run', 'main_analysis', database.TASK_STATUS_SUCCESS,
            progress=100, details={'message': 'done'},
        )

        surviving = _committed_ids(task_db)
        assert 'new-run' in surviving
        assert 'other-type' in surviving
        assert 'old-run' not in surviving
        assert 'old-album' not in surviving

    def test_history_is_written_before_the_children_are_deleted(
        self, task_db, monkeypatch
    ):
        # _build_task_note sums each child's tracks_analyzed to produce the
        # "Songs analyzed: N" note. It runs from record_task_history inside the very
        # save_task_status call that deletes those children, so reordering the two
        # silently degrades every analysis note. Nothing else pins this.
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        seen = {}
        real_record = database.record_task_history

        def spy(task_id, task_type, status, duration_seconds=None, note=None,
                details=None):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM task_status WHERE parent_task_id = %s",
                    (task_id,),
                )
                seen['children_at_history_time'] = cur.fetchone()[0]
            return real_record(task_id, task_type, status, duration_seconds, note,
                               details)

        monkeypatch.setattr(database, 'record_task_history', spy)

        _insert(task_db, 'hist-run', 'PROGRESS', task_type='main_analysis')
        _insert(task_db, 'hist-album', 'SUCCESS', parent='hist-run',
                task_type='album_analysis',
                details=json.dumps({'tracks_analyzed': 7}))

        database.save_task_status(
            'hist-run', 'main_analysis', database.TASK_STATUS_SUCCESS,
            progress=100, details={'message': 'done'},
        )

        assert seen['children_at_history_time'] == 1
        assert _committed_ids(task_db) == ['hist-run']

    def test_a_child_write_costs_no_extra_queries_and_touches_no_rows(
        self, task_db, monkeypatch
    ):
        # An analysis writes thousands of album_analysis rows. Without the
        # parent_task_id early-out every one of them would run the superseded-root
        # SELECT plus a DELETE, against the table the run is still writing.
        import database

        conn, _ = task_db
        cursors = {'n': 0}

        class _CountingConn:
            def __getattr__(self, name):
                return getattr(conn, name)

            def cursor(self, *a, **kw):
                cursors['n'] += 1
                return conn.cursor(*a, **kw)

        monkeypatch.setattr(database, 'get_db', lambda: _CountingConn())

        _insert(task_db, 'live-parent', 'PROGRESS', task_type='main_analysis')
        _insert(task_db, 'sibling', 'PROGRESS', parent='live-parent',
                task_type='album_analysis')

        database.save_task_status(
            'batch-1', 'album_analysis', database.TASK_STATUS_SUCCESS,
            parent_task_id='live-parent', progress=100,
        )

        assert cursors['n'] == 1

        surviving = _committed_ids(task_db)
        assert 'live-parent' in surviving
        assert 'sibling' in surviving

    def test_a_straggler_child_cannot_resurrect_a_row_after_the_recap(
        self, task_db, monkeypatch
    ):
        # The parent can fail while album jobs are still in flight. The UPSERT's
        # REVOKED guard sits on the UPDATE arm only, so without an INSERT guard the
        # straggler's next report re-created the row the recap had just dropped.
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'dying-parent', 'PROGRESS', task_type='main_analysis')
        _insert(task_db, 'inflight', 'PROGRESS', parent='dying-parent',
                task_type='album_analysis')

        database.save_task_status(
            'dying-parent', 'main_analysis', database.TASK_STATUS_FAILURE,
            progress=100, details={'message': 'boom'},
        )
        assert _committed_ids(task_db) == ['dying-parent']

        database.save_task_status(
            'inflight', 'album_analysis', database.TASK_STATUS_SUCCESS,
            parent_task_id='dying-parent', progress=100,
        )

        assert _committed_ids(task_db) == ['dying-parent']

    def test_a_child_of_a_live_parent_is_still_inserted_normally(
        self, task_db, monkeypatch
    ):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'live-root', 'PROGRESS', task_type='main_analysis')

        database.save_task_status(
            'fresh-album', 'album_analysis', database.TASK_STATUS_PROGRESS,
            parent_task_id='live-root', progress=10,
        )

        assert 'fresh-album' in _committed_ids(task_db)

    def test_an_existing_child_can_still_write_its_terminal_status(
        self, task_db, monkeypatch
    ):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'terminal-parent', 'SUCCESS', task_type='main_clustering')
        _insert(task_db, 'straggler', 'PROGRESS', parent='terminal-parent',
                task_type='clustering_batch')

        assert database.save_task_status(
            'straggler', 'clustering_batch', database.TASK_STATUS_SUCCESS,
            parent_task_id='terminal-parent', progress=100,
        ) is True

        with conn.cursor() as cur:
            cur.execute("SELECT status FROM task_status WHERE task_id = 'straggler'")
            assert cur.fetchone()[0] == 'SUCCESS'

    def test_save_task_status_reports_whether_it_actually_wrote(
        self, task_db, monkeypatch
    ):
        import database

        conn, _ = task_db
        monkeypatch.setattr(database, 'get_db', lambda: conn)

        _insert(task_db, 'gone-parent', 'REVOKED', task_type='main_analysis')

        assert database.save_task_status(
            'never-existed', 'album_analysis', database.TASK_STATUS_PROGRESS,
            parent_task_id='gone-parent', progress=10,
        ) is False
        assert 'never-existed' not in _committed_ids(task_db)

        assert database.save_task_status(
            'a-root', 'main_analysis', database.TASK_STATUS_PROGRESS, progress=1,
        ) is True
