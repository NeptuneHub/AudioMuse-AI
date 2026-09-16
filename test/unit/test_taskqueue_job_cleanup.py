# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""A job's memory dies with the job: fork-per-job on POSIX, unload-after-job on Windows.

Where the platform can fork, run_job runs the task in a forked child that
reports its outcome over a pipe and exits, so the parent worker never loads the
analysis models at all. Where it cannot (Windows), the job runs in the worker
process itself and _unload_job_models drops any resident CLAP/lyrics sessions
and trims the heap after every job, success or failure.

Main Features:
* The inline path unloads models after a job and skips the cleanup entirely
  when the ML stack was never imported
* A lyrics- or CLAP-only job is unloaded straight off sys.modules, without
  importing tasks.analysis.song (librosa/onnxruntime) just to clean up
* run_job routes through the forked path when forking is on and never runs the
  parent-side model cleanup there
* The parent decodes the child's pickled report, and a child that dies without
  reporting fails the job with its signal or exit code
* A child the worker itself killed on a cancel, reclaim or wedged nudge is logged
  as that stop, never as an out-of-memory crash, and is still requeued when its
  row is RUNNING and has restarts left
* A connectivity failure reported by the child requeues the row uncharged
  instead of failing it
* A real fork round-trip returns the task result and survives a SIGKILLed child
"""

import logging
import os
import pickle
import signal
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

import config
import taskqueue.worker as worker_mod
from error.error_dictionary import (
    ERR_ANALYSIS_FAILED,
    ERR_JOB_PROCESS_DIED,
    ERR_OUT_OF_MEMORY,
    ERR_PROCESS_CRASHED,
)
from taskqueue import process as process_mod
from taskqueue import retry
from taskqueue.worker import Worker, _encode_outcome


def _worker():
    instance = Worker.__new__(Worker)
    instance.identity = 'audiomuse-worker-default-hostA-11'
    instance.queue = 'default'
    instance.max_jobs = 0
    instance._held_task_id = None
    instance._held_parent_id = None
    instance._held_attempts = None
    instance._conn = MagicMock()
    instance._conn.closed = 0
    instance._jobs_done = 0
    instance._shared_cache = {}
    instance._abandoned = []
    instance._uncharged = {}
    instance._wake = threading.Event()
    instance._claim_txn = threading.Lock()
    instance._fork_jobs = False
    return instance


def _job(task_id):
    return {
        'task_id': task_id,
        'task_type': 'main_analysis',
        'parent_task_id': None,
        'func': 'tasks.analysis.run_analysis_task',
        'args': (),
        'kwargs': {},
        'attempts': 0,
        'max_attempts': 3,
    }


_ML_MODULES = ('tasks.clap_analyzer', 'tasks.analysis.song', 'lyrics')


class TestUnloadJobModels:
    def test_is_a_noop_when_the_ml_stack_was_never_imported(self, monkeypatch):
        instance = _worker()
        for name in _ML_MODULES:
            monkeypatch.delitem(sys.modules, name, raising=False)

        imported = []
        import builtins

        real_import = builtins.__import__

        def _spy(name, *args, **kwargs):
            imported.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _spy)
        instance._unload_job_models()
        assert 'tasks.analysis.song' not in imported
        assert 'tasks.memory_utils' not in imported

    def test_unloads_optional_models_and_trims_when_the_stack_is_resident(self, monkeypatch):
        instance = _worker()
        song = MagicMock()
        memory_utils = MagicMock()
        monkeypatch.setitem(sys.modules, 'tasks.clap_analyzer', MagicMock())
        monkeypatch.setitem(sys.modules, 'tasks.analysis.song', song)
        monkeypatch.setitem(sys.modules, 'lyrics', MagicMock())
        monkeypatch.setitem(sys.modules, 'tasks.memory_utils', memory_utils)
        try:
            instance._unload_job_models()
        finally:
            for name in _ML_MODULES + ('tasks.memory_utils',):
                sys.modules.pop(name, None)
        song.cleanup_optional_models.assert_called_once_with(context='worker job end')
        memory_utils.release_memory_to_os.assert_called_once()

    def test_a_lyrics_only_job_unloads_without_importing_the_analysis_stack(
        self, monkeypatch
    ):
        instance = _worker()
        monkeypatch.delitem(sys.modules, 'tasks.analysis.song', raising=False)
        lyrics = MagicMock()
        lyrics.is_lyrics_loaded.return_value = True
        clap = MagicMock()
        clap.is_clap_model_loaded.return_value = True
        memory_utils = MagicMock()
        monkeypatch.setitem(sys.modules, 'lyrics', lyrics)
        monkeypatch.setitem(sys.modules, 'tasks.clap_analyzer', clap)
        monkeypatch.setitem(sys.modules, 'tasks.memory_utils', memory_utils)

        imported = []
        import builtins

        real_import = builtins.__import__

        def _spy(name, *args, **kwargs):
            imported.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _spy)
        instance._unload_job_models()
        assert 'tasks.analysis.song' not in imported
        lyrics.unload_lyrics_models.assert_called_once()
        clap.unload_clap_model.assert_called_once()
        memory_utils.release_memory_to_os.assert_called_once()

    def test_an_unloaded_model_is_left_alone_when_its_flag_reports_not_loaded(
        self, monkeypatch
    ):
        instance = _worker()
        monkeypatch.delitem(sys.modules, 'tasks.analysis.song', raising=False)
        monkeypatch.delitem(sys.modules, 'tasks.clap_analyzer', raising=False)
        lyrics = MagicMock()
        lyrics.is_lyrics_loaded.return_value = False
        memory_utils = MagicMock()
        monkeypatch.setitem(sys.modules, 'lyrics', lyrics)
        monkeypatch.setitem(sys.modules, 'tasks.memory_utils', memory_utils)
        instance._unload_job_models()
        lyrics.unload_lyrics_models.assert_not_called()
        memory_utils.release_memory_to_os.assert_called_once()

    def test_run_job_cleans_up_after_a_successful_inline_job(self, monkeypatch):
        import taskqueue

        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, '_unload_job_models', MagicMock())
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)
        monkeypatch.setattr(
            taskqueue, 'resolve_func', lambda _dotted: lambda *a, **k: {'done': True}
        )
        instance.run_job(_job('task-ok'))
        instance._unload_job_models.assert_called_once()

    def test_run_job_cleans_up_even_when_the_inline_job_raises(self, monkeypatch):
        import taskqueue

        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, '_unload_job_models', MagicMock())
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)

        def _raising(*_a, **_k):
            raise ValueError('boom')

        monkeypatch.setattr(taskqueue, 'resolve_func', lambda _dotted: _raising)
        instance.run_job(_job('task-boom'))
        instance._unload_job_models.assert_called_once()


class TestRunJobRouting:
    def test_run_job_uses_the_forked_path_and_skips_the_parent_side_cleanup(
        self, monkeypatch
    ):
        instance = _worker()
        instance._fork_jobs = True
        finalized = []
        monkeypatch.setattr(instance, '_unload_job_models', MagicMock())
        monkeypatch.setattr(
            instance, '_run_in_child',
            MagicMock(return_value=(config.TASK_STATUS_SUCCESS, None, {'done': True})),
        )
        monkeypatch.setattr(
            instance, 'finalize', lambda *a, **k: finalized.append((a, k))
        )
        instance.run_job(_job('task-forked'))
        instance._run_in_child.assert_called_once()
        instance._unload_job_models.assert_not_called()
        assert finalized == [
            ((_job('task-forked'), config.TASK_STATUS_SUCCESS, None),
             {'result': {'done': True}}),
        ]

    def test_a_connectivity_report_from_the_child_requeues_instead_of_failing(
        self, monkeypatch
    ):
        instance = _worker()
        instance._fork_jobs = True
        monkeypatch.setattr(
            instance, '_run_in_child',
            MagicMock(return_value=(None, 'connection lost', None)),
        )
        monkeypatch.setattr(instance, 'finalize', MagicMock())
        instance.run_job(_job('task-lost'))
        instance.finalize.assert_not_called()
        assert instance._abandoned == ['task-lost']


class TestChildOutcome:
    def test_a_pickled_report_is_the_job_outcome(self):
        instance = _worker()
        payload = pickle.dumps((config.TASK_STATUS_SUCCESS, None, {'albums': 3}))
        outcome = instance._child_outcome('task-1', 0, payload)
        assert outcome == (config.TASK_STATUS_SUCCESS, None, {'albums': 3})

    @pytest.mark.skipif(
        not hasattr(signal, 'SIGKILL'), reason='POSIX wait statuses only'
    )
    def test_a_child_killed_by_a_signal_fails_with_the_signal_number(self):
        instance = _worker()
        outcome = instance._child_outcome('task-2', signal.SIGKILL, b'')
        assert outcome[0] == retry.FAIL_RETRYABLE, (
            'a child the kernel killed for memory is retried like any other failure: '
            'memory pressure is transient, and the queue bounds the retries'
        )
        assert 'signal 9' in outcome[1]
        assert 'out-of-memory' in outcome[1]
        assert outcome[2]['error_code'] == ERR_JOB_PROCESS_DIED, (
            'with no before/after oom_kill reading the queue cannot prove the kernel '
            'killed it for memory, so it must not claim an out-of-memory error'
        )

    @pytest.mark.skipif(
        not hasattr(signal, 'SIGKILL'), reason='POSIX wait statuses only'
    )
    def test_without_a_kernel_log_a_sigkill_is_out_of_memory_only_when_the_cgroup_counted_one(
        self, monkeypatch
    ):
        instance = _worker()
        monkeypatch.setattr(worker_mod, '_kernel_oom_victims', lambda since: None)
        monkeypatch.setattr(worker_mod, '_oom_kill_count', lambda: 4)

        confirmed = instance._child_outcome(
            'task-oom', signal.SIGKILL, b'', {'kills': 3, 'since_usec': 1, 'pid': 77}
        )
        not_counted = instance._child_outcome(
            'task-kill', signal.SIGKILL, b'', {'kills': 4, 'since_usec': 1, 'pid': 78}
        )

        assert confirmed[2]['error_code'] == ERR_OUT_OF_MEMORY
        assert 'recorded an out-of-memory kill' in confirmed[1]
        assert 'without naming the victim' in confirmed[1], (
            'the cgroup counter covers the whole container; the summary must not claim '
            'it proved this job was the victim'
        )
        assert not_counted[2]['error_code'] == ERR_JOB_PROCESS_DIED
        assert 'recorded no out-of-memory kill' in not_counted[1]
        assert 'systemd-oomd' in not_counted[1]

    @pytest.mark.skipif(
        not hasattr(signal, 'SIGKILL'), reason='POSIX wait statuses only'
    )
    def test_the_kernel_log_pins_the_oom_kill_on_the_victim_process_not_the_container(
        self, monkeypatch
    ):
        instance = _worker()
        monkeypatch.setattr(worker_mod, '_oom_kill_count', lambda: 99)
        monkeypatch.setattr(worker_mod, '_kernel_oom_victims', lambda since: {4242})

        victim = instance._child_outcome(
            'task-victim', signal.SIGKILL, b'', {'kills': 1, 'since_usec': 5, 'pid': 4242}
        )
        bystander = instance._child_outcome(
            'task-bystander', signal.SIGKILL, b'', {'kills': 1, 'since_usec': 5, 'pid': 5151}
        )

        assert victim[2]['error_code'] == ERR_OUT_OF_MEMORY
        assert 'pid 4242' in victim[1]
        assert bystander[2]['error_code'] == ERR_JOB_PROCESS_DIED, (
            'another process was the out-of-memory victim; a container-wide counter that '
            'rose during this job must not pin that kill on an unrelated SIGKILL'
        )
        assert 'different process (pid 4242)' in bystander[1]

    @pytest.mark.skipif(
        not hasattr(signal, 'SIGKILL'), reason='POSIX wait statuses only'
    )
    def test_a_kernel_log_that_names_no_victim_defers_to_the_cgroup_counter(self, monkeypatch):
        instance = _worker()
        monkeypatch.setattr(worker_mod, '_kernel_oom_victims', lambda since: set())
        monkeypatch.setattr(worker_mod, '_oom_kill_count', lambda: 99)

        lost_record = instance._child_outcome(
            'task-rotated', signal.SIGKILL, b'', {'kills': 1, 'since_usec': 5, 'pid': 6}
        )
        evicted = instance._child_outcome(
            'task-evicted', signal.SIGKILL, b'', {'kills': 99, 'since_usec': 5, 'pid': 7}
        )

        assert lost_record[2]['error_code'] == ERR_OUT_OF_MEMORY, (
            'an empty victim set only means no record was found: the kill record can '
            'rotate out of the ring buffer or carry a timestamp that lags the job start, '
            'and the cgroup counter still saw the out-of-memory kill'
        )
        assert evicted[2]['error_code'] == ERR_JOB_PROCESS_DIED
        assert 'recorded no out-of-memory kill' in evicted[1]

    def test_an_empty_kernel_log_without_a_readable_counter_is_not_out_of_memory(
        self, monkeypatch
    ):
        instance = _worker()
        monkeypatch.setattr(worker_mod, '_kernel_oom_victims', lambda since: set())
        monkeypatch.setattr(worker_mod, '_oom_kill_count', lambda: None)

        outcome = instance._child_outcome(
            'task-evicted', signal.SIGKILL, b'', {'kills': None, 'since_usec': 5, 'pid': 6}
        )

        assert outcome[2]['error_code'] == ERR_JOB_PROCESS_DIED
        assert 'recorded no out-of-memory kill' in outcome[1]

    def test_every_sigkill_summary_ends_by_pointing_at_the_container_logs(self, monkeypatch):
        instance = _worker()
        monkeypatch.setattr(worker_mod, '_oom_kill_count', lambda: None)
        readings = (None, set(), {6}, {8})

        for victims in readings:
            monkeypatch.setattr(worker_mod, '_kernel_oom_victims', lambda since, v=victims: v)
            outcome = instance._child_outcome(
                'task-any', signal.SIGKILL, b'', {'kills': None, 'since_usec': 5, 'pid': 6}
            )
            assert outcome[1].endswith('Check the container logs for details.'), victims

    @pytest.mark.skipif(
        not hasattr(signal, 'SIGKILL'), reason='POSIX wait statuses only'
    )
    def test_a_child_the_worker_killed_itself_is_never_called_out_of_memory(
        self, monkeypatch
    ):
        instance = _worker()
        monkeypatch.setattr(worker_mod, '_oom_kill_count', lambda: 9)
        monkeypatch.setattr(worker_mod, '_kernel_oom_victims', lambda since: {7})

        outcome = instance._child_outcome(
            'task-self', signal.SIGKILL, b'', {'kills': 1, 'since_usec': 1, 'pid': 7},
            killed_by_worker=True,
        )

        assert outcome[2]['error_code'] == ERR_JOB_PROCESS_DIED, (
            'the worker SIGKILLs its child when the report pipe cannot be read; an OOM '
            'kill elsewhere in the container while it ran must not be pinned on this job'
        )
        assert 'could not be read' in outcome[1]

    @pytest.mark.skipif(
        not hasattr(signal, 'SIGSEGV'), reason='POSIX wait statuses only'
    )
    def test_a_native_crash_in_the_model_runtime_is_not_called_out_of_memory(self):
        instance = _worker()

        segfault = instance._child_outcome('task-segv', signal.SIGSEGV, b'')
        illegal = instance._child_outcome('task-ill', signal.SIGILL, b'')

        assert segfault[2]['error_code'] == ERR_PROCESS_CRASHED, (
            'a SIGSEGV inside ONNX inference (#767) used to be reported as the system '
            'running out of memory, sending users after RAM that was never the problem'
        )
        assert 'SIGSEGV' in segfault[1] and 'not an out-of-memory' in segfault[1]
        assert illegal[2]['error_code'] == ERR_PROCESS_CRASHED
        assert 'illegal CPU instruction' in illegal[1]

    def test_the_oom_kill_counter_reads_the_cgroup_v2_events_file(self, tmp_path, monkeypatch):
        events = tmp_path / 'memory.events'
        events.write_text('low 0\nhigh 0\nmax 12\noom 2\noom_kill 2\n', encoding='ascii')
        monkeypatch.setattr(
            worker_mod, '_own_cgroup_counter_files',
            lambda: (str(tmp_path / 'missing'), str(events)),
        )

        assert worker_mod._oom_kill_count() == 2

    def test_no_readable_oom_kill_counter_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            worker_mod, '_own_cgroup_counter_files', lambda: (str(tmp_path / 'missing'),)
        )

        assert worker_mod._oom_kill_count() is None

    def test_the_counter_is_read_from_the_workers_own_cgroup_first(self, tmp_path, monkeypatch):
        import builtins

        membership = tmp_path / 'cgroup'
        membership.write_text('0::/system.slice/audiomuse.service\n', encoding='ascii')
        real_open = builtins.open

        def _open(path, *args, **kwargs):
            if path == '/proc/self/cgroup':
                return real_open(membership, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, 'open', _open)

        files = worker_mod._own_cgroup_counter_files()

        assert files[0] == '/sys/fs/cgroup/system.slice/audiomuse.service/memory.events', (
            'a native Linux install runs inside a systemd unit; the root cgroup has no '
            'memory.events, so reading only the root never found a counter there'
        )
        assert '/sys/fs/cgroup/memory.events' in files

    def test_the_kernel_log_is_ignored_outside_the_initial_pid_namespace(self, monkeypatch):
        class _Stat:
            st_ino = 4026532000

        monkeypatch.setattr(worker_mod.os, 'stat', lambda path: _Stat())

        assert worker_mod._kernel_oom_victims(1) is None, (
            'inside a pid namespace the kernel log names host pids, which never match '
            "the job's namespace pid"
        )

    def test_the_kernel_log_parser_keeps_only_oom_victims_logged_after_the_job_started(
        self, monkeypatch
    ):
        class _Stat:
            st_ino = worker_mod._INITIAL_PID_NAMESPACE_INODE

        records = [
            b'3,100,500,-;Out of memory: Killed process 11 (python3) total-vm:1kB',
            b'3,101,2000,-;oom-kill:constraint=CONSTRAINT_MEMCG,task=python3,pid=4242,uid=1000',
            b'3,102,2100,-;Memory cgroup out of memory: Killed process 4242 (python3)',
            b'6,103,2200,-;usb 1-1: new device',
        ]

        def _read(fd, size):
            if records:
                return records.pop(0)
            raise BlockingIOError

        monkeypatch.setattr(worker_mod.os, 'stat', lambda path: _Stat())
        monkeypatch.setattr(worker_mod.os, 'open', lambda path, flags: 99)
        monkeypatch.setattr(worker_mod.os, 'read', _read)
        monkeypatch.setattr(worker_mod.os, 'close', lambda fd: None)

        assert worker_mod._kernel_oom_victims(1000) == {4242}, (
            'pid 11 was killed before this job started; a reused pid must not match it'
        )

    def test_a_child_that_exits_without_reporting_fails_with_its_exit_code(self):
        instance = _worker()
        outcome = instance._child_outcome('task-3', 1 << 8, b'')
        assert outcome[0] == retry.FAIL_RETRYABLE
        assert 'exited with code 1' in outcome[1]
        assert outcome[2]['error_code'] == ERR_JOB_PROCESS_DIED

    def test_a_garbage_report_falls_back_to_the_death_summary(self):
        instance = _worker()
        outcome = instance._child_outcome('task-4', 3 << 8, b'not-a-pickle')
        assert outcome[0] == retry.FAIL_RETRYABLE
        assert 'exited with code 3' in outcome[1]

    def test_a_malformed_report_falls_back_to_the_death_summary(self):
        instance = _worker()
        outcome = instance._child_outcome('task-5', 0, pickle.dumps(['wrong', 'shape']))
        assert outcome[0] == retry.FAIL_RETRYABLE

    @pytest.mark.skipif(
        not hasattr(signal, 'SIGTERM'), reason='POSIX wait statuses only'
    )
    def test_a_child_this_worker_killed_is_a_stop_not_an_out_of_memory_crash(
        self, monkeypatch, caplog
    ):
        instance = _worker()
        monkeypatch.setattr(process_mod, '_STOPPING', ['task task-6 was cancelled'])
        with caplog.at_level(logging.INFO, logger=worker_mod.logger.name):
            outcome = instance._child_outcome('task-6', signal.SIGTERM, b'')
        assert outcome[0] == retry.FAIL_RETRYABLE, (
            'the wedged-task nudge sends this same cancel while the row is still RUNNING; '
            'a revoked outcome would end that task for good instead of requeueing it'
        )
        assert 'was cancelled' in outcome[1]
        assert 'out of memory' not in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_a_stop_does_not_hide_a_report_the_child_managed_to_send(self, monkeypatch):
        instance = _worker()
        monkeypatch.setattr(process_mod, '_STOPPING', ['recycling'])
        payload = pickle.dumps((config.TASK_STATUS_SUCCESS, None, {'albums': 1}))
        assert instance._child_outcome('task-7', 0, payload) == (
            config.TASK_STATUS_SUCCESS, None, {'albums': 1}
        )

    def _stopped_job_run(self, monkeypatch, job):
        instance = _worker()
        instance._fork_jobs = True
        monkeypatch.setattr(process_mod, '_STOPPING', ['task %s was cancelled' % job['task_id']])
        monkeypatch.setattr(instance, 'hydrate_shared', lambda kwargs: kwargs)
        monkeypatch.setattr(
            instance, '_run_in_child',
            lambda job, kwargs: instance._child_outcome(job['task_id'], 15, b''),
        )
        monkeypatch.setattr(instance, 'finalize', MagicMock())
        monkeypatch.setattr(instance, '_requeue_for_retry', MagicMock())
        return instance

    def test_a_nudged_child_with_restarts_left_is_requeued_not_revoked(self, monkeypatch, caplog):
        job = _job('task-8')
        instance = self._stopped_job_run(monkeypatch, job)
        with caplog.at_level(logging.INFO, logger=worker_mod.logger.name):
            instance.run_job(job)
        instance._requeue_for_retry.assert_called_once()
        instance.finalize.assert_not_called()
        assert 'out of memory' not in caplog.text

    def test_a_stopped_child_with_no_restart_left_finishes_without_an_error_verdict(
        self, monkeypatch, caplog
    ):
        job = dict(_job('task-9'), max_attempts=0)
        instance = self._stopped_job_run(monkeypatch, job)
        with caplog.at_level(logging.INFO, logger=worker_mod.logger.name):
            instance.run_job(job)
        instance._requeue_for_retry.assert_not_called()
        instance.finalize.assert_called_once()
        assert instance.finalize.call_args[0][1] == config.TASK_STATUS_FAIL, (
            'the terminal write is fenced by the row status: a cancelled row is left '
            'alone, a nudged RUNNING row with no restart left ends FAIL as before'
        )
        assert 'failed on attempt' not in caplog.text

    def test_not_retrying_a_row_a_cancel_already_ended_is_not_an_error(self, monkeypatch, caplog):
        instance = _worker()
        monkeypatch.setattr(process_mod, '_STOPPING', ['task task-10 was cancelled'])
        monkeypatch.setattr(worker_mod.sql, 'current_row', lambda cur, task_id: {
            'status': config.TASK_STATUS_REVOKED, 'worker_id': instance.identity,
        })
        requeue = MagicMock()
        monkeypatch.setattr(worker_mod.sql, 'requeue_or_fail', requeue)
        with caplog.at_level(logging.INFO, logger=worker_mod.logger.name):
            instance._requeue_for_retry(_job('task-10'), 'stopped')
        requeue.assert_not_called()
        assert 'Not retrying task-10' in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestEncodeOutcome:
    def test_a_dict_result_survives_the_round_trip(self):
        payload = _encode_outcome((config.TASK_STATUS_SUCCESS, None, {'n': 1}))
        assert pickle.loads(payload) == (config.TASK_STATUS_SUCCESS, None, {'n': 1})

    def test_a_non_dict_result_is_not_shipped_back(self):
        payload = _encode_outcome((config.TASK_STATUS_SUCCESS, None, object()))
        assert pickle.loads(payload) == (config.TASK_STATUS_SUCCESS, None, None)

    def test_an_unpicklable_result_is_dropped_but_the_status_survives(self):
        payload = _encode_outcome(
            (config.TASK_STATUS_FAIL, 'boom', {'callback': lambda: None})
        )
        assert pickle.loads(payload) == (config.TASK_STATUS_FAIL, 'boom', None)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork is POSIX-only')
class TestRealFork:
    def test_the_child_runs_the_job_and_the_parent_reads_its_report(self, monkeypatch):
        import taskqueue

        instance = _worker()
        instance._fork_jobs = True
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(
            taskqueue, 'resolve_func', lambda _dotted: lambda *a, **k: {'answer': 42}
        )
        outcome = instance._run_in_child(_job('fork-ok'), {})
        assert outcome == (config.TASK_STATUS_SUCCESS, None, {'answer': 42})

    def test_a_task_exception_in_the_child_reaches_the_parent_as_a_failure(
        self, monkeypatch
    ):
        import taskqueue

        instance = _worker()
        instance._fork_jobs = True
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)

        def _raising(*_a, **_k):
            raise ValueError('exploded in the child')

        monkeypatch.setattr(taskqueue, 'resolve_func', lambda _dotted: _raising)
        outcome = instance._run_in_child(_job('fork-boom'), {})
        assert outcome[0] == retry.FAIL_RETRYABLE
        assert 'exploded in the child' in outcome[1]
        assert outcome[2]['error_code'] == ERR_ANALYSIS_FAILED, (
            'the structured record rides back through the pickled report, classified '
            "against the task function's own domain code"
        )
        assert 'exploded in the child' in outcome[2]['error_message']

    def test_a_sigkilled_child_fails_the_job_but_not_the_worker(self, monkeypatch):
        import taskqueue

        instance = _worker()
        instance._fork_jobs = True
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)

        def _suicide(*_a, **_k):
            os.kill(os.getpid(), signal.SIGKILL)

        monkeypatch.setattr(taskqueue, 'resolve_func', lambda _dotted: _suicide)
        outcome = instance._run_in_child(_job('fork-killed'), {})
        assert outcome[0] == retry.FAIL_RETRYABLE
        assert 'signal 9' in outcome[1]


class TestParentDeathBinding:
    def test_linux_uses_pdeathsig(self, monkeypatch):
        from taskqueue.worker import _bind_to_parent_death

        monkeypatch.setattr('taskqueue.worker.sys.platform', 'linux')
        with patch('taskqueue.worker._bind_linux_pdeathsig') as linux_bind, \
             patch('taskqueue.worker._watch_parent_death') as watch:
            _bind_to_parent_death(1234)
        linux_bind.assert_called_once_with(1234)
        watch.assert_not_called()

    def test_non_linux_uses_the_getppid_watchdog(self, monkeypatch):
        from taskqueue.worker import _bind_to_parent_death

        monkeypatch.setattr('taskqueue.worker.sys.platform', 'darwin')
        with patch('taskqueue.worker._bind_linux_pdeathsig') as linux_bind, \
             patch('taskqueue.worker._watch_parent_death') as watch:
            _bind_to_parent_death(1234)
        linux_bind.assert_not_called()
        watch.assert_called_once_with(1234)

    def test_watchdog_starts_a_daemon_that_exits_when_the_parent_dies(self, monkeypatch):
        from taskqueue.worker import _watch_parent_death

        spawned = {}

        class _FakeThread:
            def __init__(self, target, name, daemon):
                spawned['target'] = target
                spawned['name'] = name
                spawned['daemon'] = daemon
                self.daemon = daemon

            def start(self):
                spawned['started'] = True

        monkeypatch.setattr(threading, 'Thread', _FakeThread)
        exit_calls = []
        monkeypatch.setattr('taskqueue.worker.os._exit', exit_calls.append)
        monkeypatch.setattr('taskqueue.worker.os.getppid', lambda: 999)

        def _stop_sleeping(*_a, **_k):
            raise RuntimeError('watchdog loop stopped for test')

        monkeypatch.setattr('taskqueue.worker.time.sleep', _stop_sleeping)
        _watch_parent_death(111)
        assert spawned['daemon'] is True
        assert spawned['name'] == 'worker-parent-watchdog'
        with pytest.raises(RuntimeError, match='watchdog loop stopped for test'):
            spawned['target']()
        assert exit_calls == [1]


class TestParentSideConfigHydration:
    def test_config_is_hydrated_in_the_parent_before_forking(self, monkeypatch):
        instance = _worker()
        instance._fork_jobs = True
        calls = []
        monkeypatch.setattr(instance, 'hydrate_config', lambda: calls.append('hydrate'))

        def _no_pipe(*_a, **_k):
            raise OSError('no pipe')

        monkeypatch.setattr('taskqueue.worker.os.pipe', _no_pipe)
        instance._run_in_child(_job('task-x'), {})
        assert calls == ['hydrate']


class TestAStopHoldsTheJobPath:
    def test_run_job_cannot_write_the_row_while_a_stop_holds_the_claim_lock(self, monkeypatch):
        instance = _worker()
        instance._fork_jobs = True
        monkeypatch.setattr(process_mod, '_STOPPING', ['task t-1 was cancelled'])
        monkeypatch.setattr(instance, 'hydrate_shared', lambda kwargs: kwargs)
        monkeypatch.setattr(instance, '_run_in_child',
                            lambda job, kwargs: instance._child_outcome(job['task_id'], 15, b''))
        monkeypatch.setattr(instance, 'finalize', MagicMock())
        monkeypatch.setattr(instance, '_requeue_for_retry', MagicMock())
        instance._claim_txn.acquire()
        runner = threading.Thread(target=instance.run_job, args=(_job('t-1'),), daemon=True)
        runner.start()
        runner.join(1.0)
        assert runner.is_alive(), 'the stop keeps the lock until the process exits'
        instance.finalize.assert_not_called()
        instance._requeue_for_retry.assert_not_called()
        instance._claim_txn.release()
        runner.join(5.0)
