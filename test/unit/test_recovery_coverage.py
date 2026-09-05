# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The recovery table is complete, and the drain logic behind it has one owner.

Three of the four tasks with a long opaque phase were missing a heartbeat, not
because anyone decided they did not need one but because nothing anywhere said
what a task was supposed to have. That is what these tests are for: a task type
that declares no stance on a blocking scenario FAILS, and "not applicable" is a
first-class answer that has to carry a reason. Silence is not an answer.

The second half pins the thing that actually drifted. Analysis and clustering
each carried their own copy of the give-up rule and fixed the same bug in
opposite directions - one spared queued children into a forever-hang, the other
burnt them. Both now drive ChildDrainSupervisor, so a test on the supervisor is a
test on both parents.

Main Features:
* Every MAIN_TASK_TYPES entry and every child type has a stance on all SCENARIOS
* A "not applicable" stance must say why, so a gap cannot hide as a decision
* Every task that runs a long opaque phase declares a heartbeat, and the modules
  that need one actually import row_heartbeat
* The supervisor ends held children, or every live child when none is held
* The give-up counter bounds a run, and both parents pass a real bound
* No heartbeat is unbounded, and its budget runs on the clock rather than on a
  count of successful beats, so a database outage cannot extend it
"""

import ast
import pathlib

import pytest

import task_types
from tasks import recovery
from taskqueue import sql


REPO = pathlib.Path(__file__).resolve().parents[2]

CHILD_TASK_TYPES = task_types.CHILD_TASK_TYPES

EVERY_TASK_TYPE = task_types.NAMES + task_types.PREFIXES


class TestTheRecoveryTableIsComplete:
    @pytest.mark.parametrize('task_type', sql.MAIN_TASK_TYPES)
    def test_every_main_task_type_is_listed(self, task_type):
        assert task_type in recovery.RECOVERY, (
            f'{task_type} holds the one-live-main index, so a run of it that '
            'blocks locks out every other main task; it cannot be absent from '
            'the table that says who unblocks it'
        )

    @pytest.mark.parametrize('task_type', CHILD_TASK_TYPES)
    def test_every_child_task_type_is_listed(self, task_type):
        assert task_type in recovery.RECOVERY

    @pytest.mark.parametrize('task_type', EVERY_TASK_TYPE)
    def test_every_task_type_the_registry_knows_is_listed(self, task_type):
        assert task_type in recovery.RECOVERY, (
            f'{task_type} is a real task type the queue can hold and it declares '
            'no stance on anything. This is how plugin tasks and the migration '
            'planner went unwatched: the completeness check only ever looked at '
            'a hand-written list that neither of them was on'
        )

    @pytest.mark.parametrize('task_type', sorted(recovery.RECOVERY))
    def test_every_task_answers_every_scenario(self, task_type):
        stances = recovery.RECOVERY[task_type]
        missing = [s for s in recovery.SCENARIOS if s not in stances]
        assert not missing, (
            f'{task_type} says nothing about {missing}. A scenario a task does '
            'not face must say so out loud, because a scenario nobody wrote down '
            'is how three tasks went without a heartbeat'
        )

    @pytest.mark.parametrize('task_type', sorted(recovery.RECOVERY))
    def test_no_stance_is_left_blank(self, task_type):
        for scenario, stance in recovery.RECOVERY[task_type].items():
            if stance.applicable:
                assert stance.mechanism, f'{task_type}/{scenario} names no mechanism'
            else:
                assert stance.reason, (
                    f'{task_type}/{scenario} claims not to apply and does not say '
                    'why; that is indistinguishable from having been forgotten'
                )

    def test_the_table_lists_nothing_that_is_not_a_real_task_type(self):
        assert set(recovery.RECOVERY) <= set(EVERY_TASK_TYPE)


def _imports_row_heartbeat(relative_path):
    tree = ast.parse((REPO / relative_path).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == 'row_heartbeat' for alias in node.names):
                return True
    return False


class TestEveryLongOpaquePhaseHasAHeartbeat:
    @pytest.mark.parametrize('relative_path', [
        'tasks/analysis/index.py',
        'tasks/analysis/main.py',
        'tasks/cleaning.py',
        'tasks/clustering.py',
        'tasks/sonic_fingerprint_manager.py',
        'tasks/provider_migration_tasks.py',
        'plugin/manager.py',
    ])
    def test_the_module_imports_the_heartbeat(self, relative_path):
        assert _imports_row_heartbeat(relative_path), (
            f'{relative_path} runs a phase that is one call writing no row while '
            'it runs, which the wedged-main nudge and the stall valve both read '
            'as a wedge; without row_heartbeat a healthy run gets cancelled'
        )

    def test_cleaning_hands_its_task_id_to_the_index_build(self):
        source = (REPO / 'tasks/cleaning.py').read_text(encoding='utf-8')
        assert 'task_id=current_task_id' in source, (
            'cleaning is a MAIN_TASK_TYPE and rebuilds the indexes inline; '
            'without its task id _run_all_index_builds cannot heartbeat and the '
            'nudge cancels a healthy clean at QUEUE_WEDGED_MAIN_TASK_MINUTES'
        )

    @pytest.mark.parametrize('task_type', sorted(recovery.RECOVERY))
    def test_a_heartbeat_claim_is_backed_by_a_real_import(self, task_type):
        stance = recovery.RECOVERY[task_type][recovery.MAIN_ROW_SILENT]
        claims = stance.applicable and 'row_heartbeat' in (stance.mechanism or '')
        if not claims:
            return
        modules = {
            'main_analysis': ('tasks/analysis/index.py', 'tasks/analysis/main.py'),
            'main_clustering': ('tasks/clustering.py',),
            'cleaning': ('tasks/analysis/index.py', 'tasks/cleaning.py'),
            'sonic_fingerprint': ('tasks/sonic_fingerprint_manager.py',),
            'provider_migration': ('tasks/provider_migration_tasks.py',),
            'server_sweep': ('tasks/multiserver_sync.py',),
            'plugin.': ('plugin/manager.py',),
        }
        assert task_type in modules, (
            f'{task_type} claims a row_heartbeat and names no module that could '
            'hold one; a claim nobody can check is how a phase went unguarded'
        )
        for relative_path in modules[task_type]:
            assert _imports_row_heartbeat(relative_path), relative_path


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance_minutes(self, minutes):
        self.now += minutes * 60.0


def _supervisor(clock, max_give_ups=3, timeout_minutes=60):
    ended = []
    supervisor = recovery.ChildDrainSupervisor(
        'parent-1', lambda task_id, message: ended.append(task_id) or True,
        timeout_minutes, max_give_ups, clock, label='child',
    )
    return supervisor, ended


class TestTheOneDrainSupervisorBothParentsUse:
    def test_a_signature_that_keeps_moving_never_expires(self):
        clock = _Clock()
        supervisor, _ended = _supervisor(clock)

        for step in range(10):
            clock.advance_minutes(59)
            supervisor.moved((step,))
            assert not supervisor.expired()

    def test_it_ends_only_the_children_a_worker_holds(self):
        clock = _Clock()
        supervisor, ended = _supervisor(clock)

        supervisor.give_up(
            [('a', 'RUNNING'), ('b', 'NEW'), ('c', 'NEW')], ['a', 'b', 'c'],
        )

        assert ended == ['a']
        assert supervisor.last_spared == 2

    def test_it_ends_every_child_when_no_worker_holds_one(self):
        clock = _Clock()
        supervisor, ended = _supervisor(clock)

        supervisor.give_up([('a', 'NEW'), ('b', 'NEW')], ['a', 'b'])

        assert sorted(ended) == ['a', 'b'], (
            'nothing is running, so no worker is coming for these; a parent that '
            'spares them waits on a queue forever, which is the clustering hang'
        )
        assert supervisor.last_spared == 0

    def test_the_give_up_count_is_what_bounds_a_run(self):
        clock = _Clock()
        supervisor, _ended = _supervisor(clock, max_give_ups=2)

        assert not supervisor.exhausted()
        supervisor.give_up([('a', 'RUNNING')], ['a'])
        assert not supervisor.exhausted()
        supervisor.give_up([('b', 'RUNNING')], ['b'])
        assert supervisor.exhausted(), (
            'a worker that wedges on every child it claims otherwise makes the '
            'run last one stall window per child, forever'
        )

    def test_zero_means_unbounded_not_immediately_exhausted(self):
        clock = _Clock()
        supervisor, _ended = _supervisor(clock, max_give_ups=0)

        supervisor.give_up([('a', 'RUNNING')], ['a'])

        assert not supervisor.exhausted()

    def test_a_child_that_cannot_be_ended_never_stops_the_others(self):
        clock = _Clock()
        ended = []

        def end_child(task_id, _message):
            if task_id == 'b':
                raise RuntimeError('cancel failed')
            ended.append(task_id)
            return True

        supervisor = recovery.ChildDrainSupervisor(
            'parent-1', end_child, 60, 3, clock, label='child',
        )
        count, _minutes = supervisor.give_up(
            [('a', 'RUNNING'), ('b', 'RUNNING'), ('c', 'RUNNING')], ['a', 'b', 'c'],
        )

        assert sorted(ended) == ['a', 'c']
        assert count == 2


class TestNoHeartbeatIsUnbounded:
    @pytest.mark.parametrize('relative_path,call', [
        ('tasks/analysis/index.py', 'row_heartbeat('),
        ('tasks/clustering.py', 'row_heartbeat('),
    ])
    def test_the_call_passes_a_stop_after(self, relative_path, call):
        source = (REPO / relative_path).read_text(encoding='utf-8')
        start = source.index(call)
        window = source[start:start + 400]

        assert 'stop_after_minutes' in window, (
            f'{relative_path} props its row up forever. On a MAIN task that is '
            'strictly worse than having no heartbeat at all: the wedged-main nudge '
            'reads the row and nothing else, so a build that never returns would '
            'hold the one-live-main index against every other run forever'
        )

    def test_the_budget_ends_the_beating(self):
        clock = {'now': 0.0}

        def now():
            return clock['now']

        assert not recovery._budget_spent(0.0, now, 600.0)
        clock['now'] = 599.0
        assert not recovery._budget_spent(0.0, now, 600.0)
        clock['now'] = 600.0
        assert recovery._budget_spent(0.0, now, 600.0), (
            'without this a phase that never returns is propped up forever and the '
            'wedged-main nudge can never fire on it'
        )

    def test_no_budget_means_never_spent(self):
        assert not recovery._budget_spent(0.0, lambda: 1e9, None)

    def test_the_budget_runs_on_the_clock_not_on_successful_beats(self):
        clock = {'now': 0.0}

        def now():
            return clock['now']

        clock['now'] = 600.0

        assert recovery._budget_spent(0.0, now, 600.0), (
            'it must not matter how many beats actually landed: counting only '
            'SUCCESSFUL beats meant a database outage froze the budget, so a '
            'flapping database let a phase beat past its limit'
        )


class TestTheSlowStepBudget:
    def test_a_disabled_stall_valve_gives_an_unbounded_heartbeat(self):
        assert recovery.slow_step_budget_minutes(0) is None

    def test_the_budget_is_several_windows_not_one(self):
        budget = recovery.slow_step_budget_minutes(60)

        assert budget > 60, (
            'the whole point is that one step may legitimately outlive the window '
            'the parent measures the WHOLE run with'
        )
        assert budget == 60 * recovery._MAX_SLOW_STEP_WINDOWS


class TestTheTwoRetryEnginesOutsideTheQueueAreNamed:
    @pytest.mark.parametrize('engine', ['cron_retry', 'provider_migration_restart_handshake'])
    def test_each_says_what_it_retries_and_why_the_queue_does_not(self, engine):
        reason = recovery.OUTSIDE_THE_QUEUE[engine]

        assert 'retr' in reason.lower() or 'wait' in reason.lower()
        assert len(reason) > 80, (
            "these two used to read as gaps in the queue's retry; a one-word entry "
            'would read that way again'
        )

    def test_nothing_else_claims_to_be_a_retry_engine(self):
        assert set(recovery.OUTSIDE_THE_QUEUE) == {
            'cron_retry', 'provider_migration_restart_handshake',
        }


class TestObserveAnswersMovedOrGaveUp:
    def test_a_changed_signature_is_moved_and_nothing_is_given_up(self):
        clock = _Clock()
        supervisor, ended = _supervisor(clock)

        assert supervisor.observe([('a', 'RUNNING', '10', '', 'album')]) == (True, None)
        assert ended == []

    def test_an_unchanged_signature_inside_the_window_is_waiting(self):
        clock = _Clock()
        supervisor, ended = _supervisor(clock, timeout_minutes=60)
        marks = [('a', 'RUNNING', '10', '', 'album')]
        supervisor.observe(marks)
        clock.advance_minutes(30)

        assert supervisor.observe(marks) == (False, None)
        assert ended == []

    def test_an_expired_window_with_live_children_gives_up_and_says_so(self):
        clock = _Clock()
        supervisor, ended = _supervisor(clock, timeout_minutes=60)
        marks = [('a', 'RUNNING', '10', '', 'album')]
        supervisor.observe(marks)
        clock.advance_minutes(61)

        moved, gave_up = supervisor.observe(marks)

        assert moved is False
        assert gave_up == (1, 61.0), (
            'the caller reports the give-up from this tuple; it used to have to infer '
            'movement from the valve clock instead, a 0.6-second probe that any '
            'restart() inside observe would silently break'
        )
        assert ended == ['a']

    def test_an_expired_window_with_nothing_live_is_still_waiting(self):
        clock = _Clock()
        supervisor, ended = _supervisor(clock, timeout_minutes=60)
        supervisor.observe([])
        clock.advance_minutes(61)

        assert supervisor.observe([]) == (False, None)
        assert ended == []


class TestChildMarksReadOnTheCallersConnection:
    def test_the_connection_handed_in_is_the_one_the_queue_reads_on(self, monkeypatch):
        import taskqueue

        seen = []
        monkeypatch.setattr(
            taskqueue, 'live_children',
            lambda parent_task_id, conn=None: seen.append(conn) or [],
        )
        sentinel = object()

        assert recovery.child_marks('parent-1', conn=sentinel) == ()
        assert seen == [sentinel], (
            'without a connection the queue resolves get_db(), the SAME Flask '
            'connection a parent has its reap open on, and commits it under the '
            'parent; the rollback the parent promises then undoes nothing'
        )
