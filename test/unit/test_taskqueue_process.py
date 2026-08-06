# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Killing a worker's process tree, and sweeping what a previous kill leaked.

Both bugs this covers were platform-shaped and neither could be caught by the
rest of the suite, because the cancel path reaches ``stop_hard`` from the
Listener THREAD and the liveness probe only misbehaves on win32. The module had
no test at all, and the one test that touches it elsewhere monkeypatches
``stop_hard`` away entirely.

Main Features:
* The POSIX kill order: children individually, grace, survivors, group last
* No signal disposition is ever set, so the path is identical on any thread
* The win32 liveness probe uses psutil, never os.kill, which terminates there
* The sweep removes only folders whose owning pid is provably gone
"""

import os
import signal
import sys
import threading

import pytest

from taskqueue import process as process_mod

psutil = pytest.importorskip('psutil')


class _Recorder:
    def __init__(self):
        self.signals = []
        self.groups = []

    def kill(self, pid, sig):
        self.signals.append((pid, sig))

    def killpg(self, pgid, sig):
        self.groups.append((pgid, sig))


@pytest.fixture
def recorder(monkeypatch):
    calls = _Recorder()
    monkeypatch.setattr(process_mod.os, 'getpgid', lambda _pid: 4242)
    monkeypatch.setattr(process_mod.os, 'kill', calls.kill)
    monkeypatch.setattr(process_mod.os, 'killpg', calls.killpg)
    monkeypatch.setattr(process_mod.time, 'sleep', lambda _seconds: None)
    return calls


def _children(monkeypatch, *rounds):
    remaining = list(rounds)

    def live():
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    monkeypatch.setattr(process_mod, '_live_children', live)


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX kill path')
class TestThePosixKillOrder:
    def test_children_are_signalled_individually_before_the_group(self, monkeypatch, recorder):
        _children(monkeypatch, [11, 12], [])

        process_mod._kill_tree_posix(0)

        assert recorder.signals[:2] == [(11, signal.SIGTERM), (12, signal.SIGTERM)]
        assert recorder.groups == [(4242, signal.SIGKILL)]

    def test_the_group_sweep_is_sigkill_and_comes_last(self, monkeypatch, recorder):
        _children(monkeypatch, [11], [])

        process_mod._kill_tree_posix(0)

        assert recorder.groups == [(4242, signal.SIGKILL)]

    def test_a_survivor_of_the_grace_period_is_sigkilled_individually(
        self, monkeypatch, recorder
    ):
        _children(monkeypatch, [11], [11])

        process_mod._kill_tree_posix(0)

        assert (11, signal.SIGTERM) in recorder.signals
        assert (11, signal.SIGKILL) in recorder.signals

    def test_the_group_is_never_sent_sigterm_because_this_process_is_in_it(
        self, monkeypatch, recorder
    ):
        _children(monkeypatch, [11], [])

        process_mod._kill_tree_posix(0)

        assert signal.SIGTERM not in [sig for _pgid, sig in recorder.groups]


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX kill path')
class TestTheKillPathIsThreadSafe:
    def test_the_posix_kill_tree_never_sets_a_signal_disposition(
        self, monkeypatch, recorder
    ):
        def refuse(*_args, **_kwargs):
            raise AssertionError("signal.signal is illegal off the main thread")

        monkeypatch.setattr(process_mod.signal, 'signal', refuse)
        _children(monkeypatch, [11], [])

        process_mod._kill_tree_posix(0)

        assert recorder.groups == [(4242, signal.SIGKILL)]

    def test_a_cancel_arriving_on_a_listener_thread_still_reaps_every_child(
        self, monkeypatch, recorder
    ):
        _children(monkeypatch, [11, 12], [])
        failures = []

        def run():
            try:
                process_mod._kill_tree_posix(0)
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=run, name='listen-default')
        thread.start()
        thread.join(timeout=5)

        assert failures == []
        assert (11, signal.SIGTERM) in recorder.signals
        assert (12, signal.SIGTERM) in recorder.signals
        assert recorder.groups == [(4242, signal.SIGKILL)]


class TestTheOwnerPid:
    def test_a_joblib_folder_names_its_pid_in_the_first_numeric_field(self):
        assert process_mod._owner_pid('joblib_memmapping_folder_9124_abc_1') == 9124

    def test_a_loky_folder_names_its_pid_in_the_second_hyphen_field_not_the_last(self):
        assert process_mod._owner_pid('loky-9124-t3mpsuffix') == 9124

    def test_a_loky_folder_with_an_all_digit_suffix_still_reads_the_pid(self):
        assert process_mod._owner_pid('loky-9124-88888888') == 9124

    def test_an_unreadable_name_yields_no_pid(self):
        assert process_mod._owner_pid('loky-notapid-xyz') is None
        assert process_mod._owner_pid('joblib_memmapping_folder_none') is None


class TestTheLivenessProbe:
    def test_win32_uses_psutil_and_never_os_kill(self, monkeypatch):
        monkeypatch.setattr(process_mod.sys, 'platform', 'win32')

        def refuse(*_args, **_kwargs):
            raise AssertionError("os.kill terminates the target on win32")

        monkeypatch.setattr(process_mod.os, 'kill', refuse)
        monkeypatch.setattr(psutil, 'pid_exists', lambda _pid: False)

        assert process_mod._pid_is_alive(9124) is False

    def test_win32_reports_a_live_pid_as_live(self, monkeypatch):
        monkeypatch.setattr(process_mod.sys, 'platform', 'win32')
        monkeypatch.setattr(psutil, 'pid_exists', lambda _pid: True)

        assert process_mod._pid_is_alive(9124) is True

    def test_this_process_is_reported_alive(self):
        assert process_mod._pid_is_alive(os.getpid()) is True


class TestTheStaleSweep:
    def test_a_folder_owned_by_a_live_process_is_left_alone(self, tmp_path, monkeypatch):
        folder = tmp_path / 'loky-9124-abc'
        folder.mkdir()
        monkeypatch.setattr(process_mod, '_pid_is_alive', lambda _pid: True)

        assert process_mod.sweep_stale_temp_dirs(str(tmp_path)) == 0
        assert folder.exists()

    def test_a_folder_owned_by_a_dead_process_is_removed(self, tmp_path, monkeypatch):
        folder = tmp_path / 'joblib_memmapping_folder_9124_abc'
        folder.mkdir()
        monkeypatch.setattr(process_mod, '_pid_is_alive', lambda _pid: False)

        assert process_mod.sweep_stale_temp_dirs(str(tmp_path)) == 1
        assert not folder.exists()

    def test_a_folder_whose_name_hides_its_owner_is_left_alone(self, tmp_path, monkeypatch):
        folder = tmp_path / 'loky-notapid-abc'
        folder.mkdir()
        monkeypatch.setattr(process_mod, '_pid_is_alive', lambda _pid: False)

        assert process_mod.sweep_stale_temp_dirs(str(tmp_path)) == 0
        assert folder.exists()

    def test_an_unrelated_folder_is_never_touched(self, tmp_path, monkeypatch):
        folder = tmp_path / 'analysis_cache'
        folder.mkdir()
        monkeypatch.setattr(process_mod, '_pid_is_alive', lambda _pid: False)

        assert process_mod.sweep_stale_temp_dirs(str(tmp_path)) == 0
        assert folder.exists()
