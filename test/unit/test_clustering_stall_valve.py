# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The clustering drain loop: its launch transaction and its last-resort stall valve.

Drives the real _cluster_one_server drain loop against a scripted queue and a
fake clock, so a whole simulated run of days passes in milliseconds. The scripted
queue is transactional: a batch enqueued on the parent's connection only becomes
live when that connection commits, and disappears when it rolls back, so the
launch coupling is exercised rather than described. The loop is only allowed to
end by itself: the fake clock aborts the test if the parent is still sleeping
after thousands of passes, which is exactly the forever-hang the valve exists to
bound.

Main Features:
* A batch is enqueued on the very connection the launch counter is written on,
  so the enqueue cannot commit itself a pass ahead of the counter
* A launch write that fails rolls the enqueue back with the counter, so the next
  pass launches that same batch exactly once instead of skipping or doubling it
* A parent row that vanished before the launch ends its transaction instead of
  returning with the connection idle in one
* A batch that keeps delivering results is never cancelled, however long the
  whole run takes in wall-clock terms
* A batch that delivers no result but is still advancing its own iterations is
  never cancelled either, and neither is one a fresh worker restarted from zero
  after the worker holding it died
* A batch that delivers nothing at all is cancelled once nothing anywhere in the
  run has changed for CLUSTERING_STALL_TIMEOUT_MINUTES, and the parent returns
* Giving up on the live batches also stops new launches, so the valve never
  abandons what is running and then queues more of it
* Batches that crash count toward CLUSTERING_EARLY_STOP_BATCHES like any other
  batch that brought back nothing better, so a run whose workers keep dying ends
  on its own after that many and never has to wait out the stall valve
* The parent's own status says how long it waited and how many batches it
  dropped, so the give-up is visible rather than silent
"""

from contextlib import nullcontext
from unittest.mock import MagicMock

import config


_UNSET = object()


class _Clock:
    def __init__(self, step_minutes):
        self.now = 1_000_000.0
        self.step_seconds = step_minutes * 60.0
        self.sleeps = 0

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, _seconds):
        self.sleeps += 1
        if self.sleeps > 5000:
            raise AssertionError(
                'the clustering drain loop never returned; the parent is hung on a '
                'child that will never finish'
            )
        self.now += self.step_seconds

    @property
    def elapsed_minutes(self):
        return (self.now - 1_000_000.0) / 60.0


class _ScriptedQueue:
    def __init__(
        self, batch_ids, finish_every_reaps=None, progress_script=None,
        finish_status=None,
    ):
        self.live_ids = list(batch_ids)
        self.pending = []
        self.enqueued = []
        self.terminal = []
        self.cancelled = []
        self.reaps = 0
        self.finish_every_reaps = finish_every_reaps
        self.progress_script = list(progress_script or ())
        self.finish_status = finish_status or config.TASK_STATUS_SUCCESS
        self.marks = {}

    def live_children(self, _parent_task_id, conn=None):
        return [
            {
                'task_id': tid,
                'sub_type_identifier': tid,
                'progress': self.marks.get(tid, 0),
            }
            for tid in self.live_ids
        ]

    def _finish_oldest(self):
        if not self.live_ids:
            return
        self.terminal.append({
            'task_id': self.live_ids.pop(0),
            'status': self.finish_status,
            'sub_type_identifier': 'batch',
            'details': {'iterations_completed_in_batch': config.ITERATIONS_PER_BATCH_JOB},
        })

    def reap_finished_children(self, _parent_task_id, conn=None):
        self.reaps += 1
        if self.progress_script:
            step = self.progress_script.pop(0)
            if step is None:
                self._finish_oldest()
            else:
                for tid in self.live_ids:
                    self.marks[tid] = step
        if (
            self.finish_every_reaps
            and self.live_ids
            and self.reaps % self.finish_every_reaps == 0
        ):
            self._finish_oldest()
        drained, self.terminal = self.terminal, []
        return drained

    def enqueue(self, _func, **kwargs):
        task_id = kwargs.get('task_id')
        conn = kwargs.get('conn')
        self.enqueued.append((task_id, conn))
        if conn is None:
            self.live_ids.append(task_id)
        else:
            self.pending.append(task_id)

    def commit(self):
        self.live_ids.extend(self.pending)
        self.pending = []

    def rollback(self):
        self.pending = []

    def request_cancel(self, task_id):
        self.cancelled.append(task_id)
        if task_id in self.live_ids:
            self.live_ids.remove(task_id)
            self.terminal.append({
                'task_id': task_id,
                'status': config.TASK_STATUS_REVOKED,
                'sub_type_identifier': 'batch',
                'details': {},
            })


class _Run:
    def __init__(self, queue, db, state):
        self.queue = queue
        self.db = db
        self.state = state
        self.messages = []
        self.events = []
        self.revoked = []
        self.status = None
        self.reason = None

    @property
    def enqueued_ids(self):
        return [task_id for task_id, _conn in self.queue.enqueued]

    @property
    def launch_trail(self):
        return [
            event for event in self.events
            if event[0] != 'commit'
            and (event[0] != 'write' or event[1].startswith('Started batch'))
        ]

    def counter_after_the_failed_write(self):
        for index, event in enumerate(self.events):
            if event[0] == 'write' and event[3] is False:
                following = [e for e in self.events[index + 1:] if e[0] == 'write']
                return following[0][2] if following else None
        raise AssertionError('no status write failed, so there was nothing to roll back')


def _lightweight_rows(count=40):
    return [
        {'item_id': f'song{i}', 'mood_vector': 'rock:0.9,pop:0.1'}
        for i in range(count)
    ]


def _drive(
    monkeypatch, queue, clock, batches_launched, total_batches=None,
    report_script=None, parent_row=_UNSET, max_concurrent=None,
):
    from tasks import clustering

    if total_batches is None:
        total_batches = batches_launched
    if parent_row is _UNSET:
        parent_row = {'status': config.TASK_STATUS_PROGRESS}

    db = MagicMock()
    db.cursor.return_value.fetchall.return_value = _lightweight_rows()

    state = {
        'runs_completed': 0, 'best_score': -1.0, 'best_result': None,
        'elite_solutions': [], 'last_subset_ids': [], 'failed_batches': 0,
        'stale_batches': 0, 'batches_launched': batches_launched,
        'job_prefix': 'main_s0',
    }
    run = _Run(queue, db, state)

    def commit():
        run.events.append(('commit',))
        queue.commit()

    def rollback():
        run.events.append(('rollback',))
        queue.rollback()

    db.commit.side_effect = commit
    db.rollback.side_effect = rollback

    def enqueue(func, **kwargs):
        run.events.append(('enqueue', kwargs.get('task_id')))
        queue.enqueue(func, **kwargs)

    def report(message, _local_pct, task_state=config.TASK_STATUS_PROGRESS):
        accepted = True if report_script is None else report_script(message)
        run.messages.append(message)
        run.events.append(('write', message, state['batches_launched'], accepted))
        if accepted:
            db.commit()
        return accepted

    def save_status(task_id, task_type, status, **kwargs):
        run.revoked.append((task_id, status, kwargs.get('details', {})))
        return True

    monkeypatch.setattr(clustering, 'time', clock)
    monkeypatch.setattr(clustering, 'get_db', lambda: db)
    monkeypatch.setattr(clustering, 'save_task_status', save_status)
    monkeypatch.setattr(clustering, 'main_task_start_lock', nullcontext)
    monkeypatch.setattr(
        clustering, 'get_task_info_from_db', lambda _task_id: parent_row,
    )
    monkeypatch.setattr(clustering, 'get_child_tasks_from_db', lambda _task_id: [])
    monkeypatch.setattr(
        clustering, '_get_stratified_song_subset',
        lambda _genre_map, _target: [{'item_id': 'song0'}],
    )
    if max_concurrent is not None:
        monkeypatch.setattr(clustering, 'MAX_CONCURRENT_BATCH_JOBS', max_concurrent)
    monkeypatch.setattr(
        clustering.taskqueue, 'put_shared_payload', lambda *a, **k: 'shared-token'
    )
    monkeypatch.setattr(
        clustering.taskqueue, 'clear_shared_payload', lambda *a, **k: None
    )
    monkeypatch.setattr(
        clustering.taskqueue, 'live_children', queue.live_children
    )
    monkeypatch.setattr(
        clustering.taskqueue, 'reap_finished_children', queue.reap_finished_children
    )
    monkeypatch.setattr(clustering.taskqueue, 'request_cancel', queue.request_cancel)
    monkeypatch.setattr(clustering.taskqueue, 'enqueue', enqueue)

    run.status, run.reason = clustering._cluster_one_server(
        None, state, report, None, 'main',
        'kmeans', 2, 5, 0.1, 0.5, 2, 5, 0, 8,
        total_batches * config.ITERATIONS_PER_BATCH_JOB,
        20, 2, 5, 2, 5, 5, 50,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        'NONE', '', '', '', '', '', '', '', '', '',
        3, 8, False, False,
    )
    return run


def _batch_ids(count):
    return [f'main_s0_batch_{i}' for i in range(count)]


class TestClusteringLaunchTransaction:
    def test_the_enqueue_runs_on_the_connection_that_writes_the_launch_counter(
        self, monkeypatch
    ):
        clock = _Clock(step_minutes=1)
        queue = _ScriptedQueue([], finish_every_reaps=1)

        run = _drive(
            monkeypatch, queue, clock, batches_launched=0, total_batches=1
        )

        assert queue.enqueued == [('main_s0_batch_0', run.db)], (
            'the enqueue must be handed the parent connection so the status write '
            'that carries batches_launched is what commits it; committing itself '
            'lets the counter reach the row a pass late and a resumed parent then '
            're-enqueues a batch that already ran'
        )
        assert run.state['batches_launched'] == 1

    def test_a_failed_launch_write_rolls_the_enqueue_back_and_relaunches_that_batch(
        self, monkeypatch
    ):
        clock = _Clock(step_minutes=1)
        queue = _ScriptedQueue([], finish_every_reaps=1)
        refused = []

        def report_script(message):
            if message.startswith('Started batch') and not refused:
                refused.append(message)
                return False
            return True

        run = _drive(
            monkeypatch, queue, clock, batches_launched=0, total_batches=1,
            report_script=report_script,
        )

        assert run.launch_trail == [
            ('enqueue', 'main_s0_batch_0'),
            ('write', 'Started batch 1/1.', 1, False),
            ('rollback',),
            ('enqueue', 'main_s0_batch_0'),
            ('write', 'Started batch 1/1.', 1, True),
        ], (
            'the write is what commits the enqueue, so a write that never lands must '
            'take the enqueue down with it and let the next pass queue the same batch '
            'index again'
        )
        assert run.counter_after_the_failed_write() == 0, (
            'every later write copies the in-memory counters onto the row, so a '
            'batches_launched left at the rolled-back value would tell a resumed '
            'parent to skip a batch that was never queued'
        )
        assert queue.live_ids == [], (
            'the rolled-back enqueue left no queue row behind'
        )
        assert run.state['batches_launched'] == 1, (
            'exactly one launch is recorded for the one batch that actually ran'
        )

    def test_a_parent_row_that_vanished_before_the_launch_ends_its_transaction(
        self, monkeypatch
    ):
        clock = _Clock(step_minutes=1)
        queue = _ScriptedQueue([])

        run = _drive(
            monkeypatch, queue, clock, batches_launched=0, total_batches=1,
            parent_row=None,
        )

        assert run.status == 'revoked'
        assert queue.enqueued == []
        assert run.db.rollback.called, (
            'the launch transaction was already open when the parent turned out to be '
            'gone, so it has to be closed before returning rather than left idle in '
            'transaction on a pooled connection'
        )


class TestClusteringWorkerDeath:
    def test_crashed_batches_end_the_run_at_the_early_stop_not_at_the_valve(
        self, monkeypatch
    ):
        clock = _Clock(step_minutes=1)
        queue = _ScriptedQueue(
            [], finish_every_reaps=1, finish_status=config.TASK_STATUS_FAIL,
        )

        run = _drive(
            monkeypatch, queue, clock, batches_launched=0, total_batches=200,
            max_concurrent=1,
        )

        assert len(run.enqueued_ids) == config.CLUSTERING_EARLY_STOP_BATCHES, (
            'a crashed batch counts toward the early stop exactly like one that came '
            'back no better, so the run stops after that many and finishes with the '
            'best result it holds rather than grinding through the whole plan'
        )
        assert [m for m in run.messages if m.startswith('Early stop')]
        assert queue.cancelled == [], (
            'the run ended itself through the early stop, so no batch was left live '
            'for the stall valve to give up on'
        )
        assert clock.elapsed_minutes < config.CLUSTERING_STALL_TIMEOUT_MINUTES, (
            'the crashes ended the run in minutes; waiting out the valve first would '
            'be an hour of a frozen progress bar for a run already known to be over'
        )


class TestClusteringStallValve:
    def test_a_progressing_child_is_never_given_up_on(self, monkeypatch):
        clock = _Clock(step_minutes=config.CLUSTERING_STALL_TIMEOUT_MINUTES / 6.0)
        queue = _ScriptedQueue(_batch_ids(20), finish_every_reaps=3)

        run = _drive(monkeypatch, queue, clock, batches_launched=20)

        assert run.state['runs_completed'] == 20 * config.ITERATIONS_PER_BATCH_JOB
        assert queue.cancelled == [], (
            'every gap between two results stayed under the valve, so no batch may be '
            'cancelled no matter how long the whole run took'
        )
        assert run.revoked == []
        assert not [m for m in run.messages if 'Gave up' in m]
        assert clock.elapsed_minutes > config.CLUSTERING_STALL_TIMEOUT_MINUTES, (
            'the valve must be a no-progress bound, not a total-runtime bound: this '
            'run outlived it many times over and was never touched'
        )

    def test_a_batch_that_only_advances_its_own_iterations_is_never_given_up_on(
        self, monkeypatch
    ):
        clock = _Clock(step_minutes=30)
        queue = _ScriptedQueue(
            _batch_ids(1), progress_script=list(range(1, 21)) + [None]
        )

        run = _drive(monkeypatch, queue, clock, batches_launched=1)

        assert queue.cancelled == [], (
            'nothing the parent can count moved for ten hours - no batch finished, '
            'failed or launched - and the single batch was still working through its '
            'iterations the whole time, so the run must be left alone'
        )
        assert run.revoked == []
        assert clock.elapsed_minutes > config.CLUSTERING_STALL_TIMEOUT_MINUTES
        assert run.state['runs_completed'] == config.ITERATIONS_PER_BATCH_JOB

    def test_a_batch_a_fresh_worker_restarted_from_zero_holds_the_valve_open(
        self, monkeypatch
    ):
        clock = _Clock(step_minutes=30)
        queue = _ScriptedQueue(
            _batch_ids(1), progress_script=[1, 2, 3, 0, 1, 2, 3, None]
        )

        run = _drive(monkeypatch, queue, clock, batches_launched=1)

        assert queue.cancelled == [], (
            'the drop back to zero is a worker dying and its replacement claiming the '
            'same batch, which is the run carrying on rather than wedging: the valve '
            'must read it as life and not cancel the batch out from under the new worker'
        )
        assert clock.elapsed_minutes > config.CLUSTERING_STALL_TIMEOUT_MINUTES
        assert run.state['runs_completed'] == config.ITERATIONS_PER_BATCH_JOB

    def test_a_completely_stalled_child_is_eventually_given_up_on(self, monkeypatch):
        clock = _Clock(step_minutes=30)
        queue = _ScriptedQueue(_batch_ids(1))

        run = _drive(monkeypatch, queue, clock, batches_launched=1)

        assert queue.cancelled == ['main_s0_batch_0']
        assert run.revoked == [(
            'main_s0_batch_0', config.TASK_STATUS_REVOKED, run.revoked[0][2]
        )]
        assert run.state['failed_batches'] == 1
        assert run.status == 'failed', (
            'the parent returns with what it has instead of waiting forever on a '
            'child whose worker is alive and holding the advisory lock'
        )
        assert clock.elapsed_minutes >= config.CLUSTERING_STALL_TIMEOUT_MINUTES

    def test_giving_up_on_a_stalled_batch_is_recorded_on_the_parent_status(
        self, monkeypatch
    ):
        clock = _Clock(step_minutes=30)
        queue = _ScriptedQueue(_batch_ids(1))

        run = _drive(monkeypatch, queue, clock, batches_launched=1)

        gave_up = [m for m in run.messages if 'Gave up on' in m]
        assert len(gave_up) == 1
        assert 'No progress of any kind' in gave_up[0]
        assert str(config.CLUSTERING_STALL_TIMEOUT_MINUTES) in gave_up[0]
        assert 'gave up on this batch' in run.revoked[0][2]['message']

    def test_giving_up_on_the_live_batches_also_stops_launching_the_remaining_ones(
        self, monkeypatch
    ):
        clock = _Clock(step_minutes=30)
        queue = _ScriptedQueue([])

        run = _drive(
            monkeypatch, queue, clock, batches_launched=0, total_batches=5,
            max_concurrent=1,
        )

        assert queue.cancelled == ['main_s0_batch_0']
        assert run.enqueued_ids == ['main_s0_batch_0'], (
            'the valve fires because the run is wedged, so the four batches never '
            'launched must stay unlaunched: queueing more work onto the same wedged '
            'worker pool is how the give-up would restart the hang it just ended'
        )
        assert run.state['batches_launched'] == 1
        assert run.status == 'failed'
