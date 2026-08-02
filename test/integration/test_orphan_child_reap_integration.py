# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Reclaiming child task_status rows stranded by a run that never archived.

clean_up_previous_main_tasks only ever archived parents it could still archive
(PENDING/STARTED/PROGRESS/SUCCESS), so a run that ended FAILURE kept one child
row per album forever: no later run selected that parent, and nothing else in
the codebase deletes children except the global cancel's full-table wipe. The
reap is keyed on the PARENT being non-live, never on the child's own status,
because a killed worker leaves children stuck at STARTED and a
terminal-children-only rule would never reclaim those.

The liveness guard is the load-bearing part. The RQ janitor writes FAILURE to a
top-level row with raw SQL while its worker may still be running, and
get_active_main_task does not match FAILURE, so a second run can legitimately
start next to a live one. A run is therefore treated as dead only when NOTHING
in the family - parent row or any child - has been written inside the grace
window, which covers a long final phase that writes only the parent row.

Assertions read through a SECOND connection so an uncommitted DELETE cannot pass:
the code under test shares one connection, where its own uncommitted changes
would be visible to any read issued on it.

Main Features:
* Children of a FAILURE or REVOKED parent, and children whose parent row is gone,
  are reclaimed regardless of the child's own status
* A parent row touched inside the grace window protects its children even when
  every child is stale, so a long index-rebuild phase is not mistaken for death
* Children of a live self-managed parent, which the archival loop skips by type,
  are never touched
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


def _insert(db, task_id, status, parent=None, task_type='main_analysis',
            age_minutes=0, details=None):
    conn, _ = db
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO task_status "
            "(task_id, parent_task_id, task_type, status, progress, details, timestamp) "
            "VALUES (%s, %s, %s, %s, 0, %s, NOW() - make_interval(mins => %s))",
            (task_id, parent, task_type, status, details, age_minutes),
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
        _insert(task_db, 'parent-failed', 'FAILURE', age_minutes=120)
        _insert(task_db, 'child-a', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis', age_minutes=120)
        _insert(task_db, 'child-b', 'STARTED', parent='parent-failed',
                task_type='album_analysis', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        assert _committed_ids(task_db) == ['parent-failed']

    def test_children_stuck_non_terminal_by_a_killed_worker_are_reclaimed(self, task_db, monkeypatch):
        _insert(task_db, 'parent-failed', 'FAILURE', age_minutes=120)
        _insert(task_db, 'child-stuck', 'PROGRESS', parent='parent-failed',
                task_type='album_analysis', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        assert 'child-stuck' not in _committed_ids(task_db)

    def test_children_whose_parent_row_is_gone_are_reclaimed(self, task_db, monkeypatch):
        _insert(task_db, 'orphan', 'SUCCESS', parent='parent-that-never-existed',
                task_type='clustering_batch', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        assert _committed_ids(task_db) == []

    def test_children_of_a_revoked_parent_are_reclaimed(self, task_db, monkeypatch):
        _insert(task_db, 'parent-revoked', 'REVOKED', age_minutes=120)
        _insert(task_db, 'child-c', 'SUCCESS', parent='parent-revoked',
                task_type='album_analysis', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        assert _committed_ids(task_db) == ['parent-revoked']

    def test_a_failed_parent_keeps_its_row_and_error_details(self, task_db, monkeypatch):
        details = json.dumps({'error': 'boom', 'status_message': 'it broke'})
        _insert(task_db, 'parent-failed', 'FAILURE', age_minutes=120, details=details)
        _insert(task_db, 'child-a', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        status, stored = _committed_row(task_db, 'parent-failed')
        assert status == 'FAILURE'
        assert json.loads(stored)['error'] == 'boom'


class TestLeavesLiveWorkAlone:
    def test_a_recently_written_parent_protects_its_stale_children(self, task_db, monkeypatch):
        _insert(task_db, 'parent-failed', 'FAILURE', age_minutes=0)
        _insert(task_db, 'child-stale', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        assert 'child-stale' in _committed_ids(task_db)

    def test_a_janitor_failure_on_a_still_writing_run_destroys_nothing(self, task_db, monkeypatch):
        _insert(task_db, 'parent-failed', 'FAILURE', age_minutes=120)
        _insert(task_db, 'child-recent', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis', age_minutes=0)
        _insert(task_db, 'child-older', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        surviving = _committed_ids(task_db)
        assert 'child-recent' in surviving
        assert 'child-older' in surviving

    @pytest.mark.parametrize('live_status', ['PENDING', 'STARTED', 'PROGRESS'])
    def test_a_live_self_managed_parents_children_are_never_touched(
        self, task_db, monkeypatch, live_status
    ):
        _insert(task_db, 'sweep', live_status, task_type='server_sweep', age_minutes=120)
        _insert(task_db, 'sweep-child', 'SUCCESS', parent='sweep',
                task_type='server_sweep', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        assert 'sweep-child' in _committed_ids(task_db)

    @pytest.mark.parametrize('live_status', ['PENDING', 'STARTED', 'PROGRESS'])
    def test_starting_a_run_still_archives_a_live_previous_parent_and_drops_its_children(
        self, task_db, monkeypatch, live_status
    ):
        _insert(task_db, 'parent-live', live_status, age_minutes=120)
        _insert(task_db, 'child-live', 'SUCCESS', parent='parent-live',
                task_type='album_analysis', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        assert _committed_row(task_db, 'parent-live')[0] == 'REVOKED'
        assert 'child-live' not in _committed_ids(task_db)


class TestDurability:
    def test_the_reap_is_committed_not_left_in_an_open_transaction(self, task_db, monkeypatch):
        _insert(task_db, 'parent-failed', 'FAILURE', age_minutes=120)
        _insert(task_db, 'child-a', 'SUCCESS', parent='parent-failed',
                task_type='album_analysis', age_minutes=120)

        _run_cleanup(monkeypatch, task_db)

        conn, _ = task_db
        conn.rollback()

        assert 'child-a' not in _committed_ids(task_db)
