# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Frozen-child argv dispatch shared by the three standalone launchers.

Guards issue #827: inside a PyInstaller bundle the worker processes that
multiprocessing and joblib/loky spawn re-run the app binary, so the launcher
must recognise their argv and run the requested payload instead of starting a
second copy of the app.

Main Features:
* Every spawn form loky and multiprocessing emit is recognised and dispatched
* The dispatched child sees the argv tail its parent passed, minus the switch
* Non-spawn argv, unlisted modules and unfrozen processes are left alone
* The forms are also taken from the installed multiprocessing/loky themselves,
  so an upstream change to how a child is spawned fails here instead of in a bundle
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
NATIVE_BUILD = os.path.join(REPO_ROOT, 'native-build')

EXE = '/Applications/AudioMuse-AI.app/Contents/MacOS/AudioMuse-AI'

LOKY_WORKER_ARGV = [
    EXE,
    '-m', 'joblib.externals.loky.backend.popen_loky_posix',
    '--process-name', 'LokyProcess-3',
    '--pipe', '11',
]
LOKY_TRACKER_ARGV = [
    EXE,
    '-c',
    'from joblib.externals.loky.backend.resource_tracker import main; main(9, 0)',
]
MP_TRACKER_ARGV = [
    EXE,
    '-c',
    'from multiprocessing.resource_tracker import main;main(7)',
]
MP_FORKSERVER_ARGV = [
    EXE,
    '-c',
    "from multiprocessing.forkserver import main; "
    "main(4, 5, ['numpy'], **{'sys_path': ['/a']})",
]


def _load():
    for entry in (REPO_ROOT, NATIVE_BUILD):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    path = os.path.join(NATIVE_BUILD, 'native_common', 'frozen_children.py')
    spec = importlib.util.spec_from_file_location('frozen_children_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def frozen_children(monkeypatch):
    mod = _load()
    monkeypatch.setattr(sys, 'argv', [EXE], raising=False)
    return mod


@pytest.fixture
def ran_module(frozen_children, monkeypatch):
    calls = []
    monkeypatch.setattr(
        frozen_children.runpy, 'run_module',
        lambda name, **kwargs: calls.append((name, kwargs, list(sys.argv))),
    )
    return calls


@pytest.fixture
def ran_main(frozen_children, monkeypatch):
    calls = []

    class _Module:
        @staticmethod
        def main(*args, **kwargs):
            calls.append((args, kwargs, list(sys.argv)))

    monkeypatch.setattr(
        frozen_children.importlib, 'import_module',
        lambda name: calls.append(name) or _Module,
    )
    return calls


def test_an_unfrozen_process_never_dispatches_even_on_spawn_argv(frozen_children):
    assert frozen_children.run_frozen_child(LOKY_WORKER_ARGV, frozen=False) is False


def test_a_plain_user_launch_is_not_treated_as_a_spawned_child(frozen_children):
    assert frozen_children.run_frozen_child([EXE], frozen=True) is False


def test_a_role_launch_is_not_treated_as_a_spawned_child(frozen_children):
    argv = [EXE, '--role=worker-default']
    assert frozen_children.run_frozen_child(argv, frozen=True) is False


def test_a_loky_posix_worker_argv_runs_the_worker_module(ran_module, frozen_children):
    assert frozen_children.run_frozen_child(LOKY_WORKER_ARGV, frozen=True) is True
    assert len(ran_module) == 1
    name, kwargs, _ = ran_module[0]
    assert name == 'joblib.externals.loky.backend.popen_loky_posix'
    assert kwargs == {'run_name': '__main__', 'alter_sys': False}, (
        "alter_sys=True swaps sys.modules['__main__'], so loky's "
        "_fixup_main_from_path stops matching the parent and re-runs the bundle "
        "entry script from a path that does not exist on disk"
    )


def test_the_loky_worker_module_sees_its_own_flags_without_the_m_switch(
    ran_module, frozen_children
):
    frozen_children.run_frozen_child(LOKY_WORKER_ARGV, frozen=True)
    _, _, argv_seen = ran_module[0]
    assert argv_seen == [EXE, '--process-name', 'LokyProcess-3', '--pipe', '11']


def test_a_loky_resource_tracker_argv_runs_its_main_with_the_parent_arguments(
    ran_main, frozen_children
):
    assert frozen_children.run_frozen_child(LOKY_TRACKER_ARGV, frozen=True) is True
    assert ran_main[0] == 'joblib.externals.loky.backend.resource_tracker'
    args, kwargs, _ = ran_main[1]
    assert args == (9, 0)
    assert kwargs == {}


def test_a_multiprocessing_resource_tracker_argv_runs_its_main(ran_main, frozen_children):
    assert frozen_children.run_frozen_child(MP_TRACKER_ARGV, frozen=True) is True
    assert ran_main[0] == 'multiprocessing.resource_tracker'
    assert ran_main[1][0] == (7,)


def test_a_forkserver_argv_keeps_its_positional_and_keyword_arguments(
    ran_main, frozen_children
):
    assert frozen_children.run_frozen_child(MP_FORKSERVER_ARGV, frozen=True) is True
    assert ran_main[0] == 'multiprocessing.forkserver'
    args, kwargs, _ = ran_main[1]
    assert args == (4, 5, ['numpy'])
    assert kwargs == {'sys_path': ['/a']}


def test_a_module_outside_the_allowlist_is_left_for_the_launcher(
    ran_module, frozen_children
):
    argv = [EXE, '-m', 'http.server', '--pipe', '11']
    assert frozen_children.run_frozen_child(argv, frozen=True) is False
    assert ran_module == []


def test_code_outside_the_allowlist_is_left_for_the_launcher(ran_main, frozen_children):
    argv = [EXE, '-c', 'from shutil import main; main(1)']
    assert frozen_children.run_frozen_child(argv, frozen=True) is False
    assert ran_main == []


def test_code_that_is_not_a_bare_main_call_is_left_for_the_launcher(
    ran_main, frozen_children
):
    argv = [
        EXE, '-c',
        'from multiprocessing.resource_tracker import main; import os; os.system("id")',
    ]
    assert frozen_children.run_frozen_child(argv, frozen=True) is False
    assert ran_main == []


def test_interpreter_flags_before_the_code_switch_do_not_hide_the_child(
    ran_main, frozen_children
):
    argv = [EXE, '-E', '-s', '-c', MP_TRACKER_ARGV[2]]
    assert frozen_children.run_frozen_child(argv, frozen=True) is True
    assert ran_main[0] == 'multiprocessing.resource_tracker'


def test_a_multiprocessing_fork_argv_runs_spawn_main_with_its_keywords(
    frozen_children, monkeypatch
):
    import multiprocessing.spawn as spawn

    calls = []
    monkeypatch.setattr(spawn, 'spawn_main', lambda **kwargs: calls.append(kwargs))
    argv = [EXE, '--multiprocessing-fork', 'tracker_fd=8', 'pipe_handle=12']
    assert frozen_children.run_frozen_child(argv, frozen=True) is True
    assert calls == [{'tracker_fd': 8, 'pipe_handle': 12}]


def test_a_multiprocessing_fork_argv_maps_the_none_literal_to_none(
    frozen_children, monkeypatch
):
    import multiprocessing.spawn as spawn

    calls = []
    monkeypatch.setattr(spawn, 'spawn_main', lambda **kwargs: calls.append(kwargs))
    argv = [EXE, '--multiprocessing-fork', 'tracker_fd=None', 'pipe_handle=12']
    assert frozen_children.run_frozen_child(argv, frozen=True) is True
    assert calls == [{'tracker_fd': None, 'pipe_handle': 12}]


def test_a_bare_handle_fork_argv_is_ignored_off_windows(frozen_children, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    argv = [EXE, '--multiprocessing-fork', '736']
    assert frozen_children.run_frozen_child(argv, frozen=True) is False


def test_a_bare_handle_fork_argv_runs_the_loky_win32_child_on_windows(
    frozen_children, monkeypatch
):
    calls = []
    fake_module = type('M', (), {'main': staticmethod(lambda **kw: calls.append(kw))})
    monkeypatch.setitem(
        sys.modules, 'joblib.externals.loky.backend.popen_loky_win32', fake_module
    )
    monkeypatch.setattr(sys, 'platform', 'win32')
    argv = [EXE, '--multiprocessing-fork', '736']
    assert frozen_children.run_frozen_child(argv, frozen=True) is True
    assert calls == [{'pipe_handle': 736}]


def test_the_argv_multiprocessing_itself_builds_when_frozen_is_dispatched(
    frozen_children, monkeypatch
):
    import multiprocessing.spawn as spawn

    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    argv = [EXE] + spawn.get_command_line(tracker_fd=8, pipe_handle=12)[1:]
    calls = []
    monkeypatch.setattr(spawn, 'spawn_main', lambda **kwargs: calls.append(kwargs))

    assert frozen_children.run_frozen_child(argv, frozen=True) is True
    assert calls == [{'tracker_fd': 8, 'pipe_handle': 12}]


def test_the_module_path_loky_spawns_its_resource_tracker_from_is_dispatched(
    ran_main, frozen_children
):
    from joblib.externals.loky.backend import resource_tracker

    argv = [EXE, '-c', f'from {resource_tracker.main.__module__} import main; main(9, 0)']

    assert frozen_children.run_frozen_child(argv, frozen=True) is True
    assert ran_main[0] == resource_tracker.main.__module__


def test_the_module_path_multiprocessing_spawns_its_resource_tracker_from_is_dispatched(
    ran_main, frozen_children
):
    from multiprocessing import resource_tracker

    argv = [EXE, '-c', f'from {resource_tracker.main.__module__} import main;main(7)']

    assert frozen_children.run_frozen_child(argv, frozen=True) is True
    assert ran_main[0] == resource_tracker.main.__module__


def test_the_module_path_loky_spawns_its_workers_from_is_dispatched(
    ran_module, frozen_children
):
    from joblib.externals.loky.backend import popen_loky_posix

    argv = [EXE, '-m', popen_loky_posix.__name__, '--process-name', 'L-1', '--pipe', '11']

    assert frozen_children.run_frozen_child(argv, frozen=True) is True
    assert ran_module[0][0] == popen_loky_posix.__name__
