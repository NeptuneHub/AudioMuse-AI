# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Native standalone supervisor thread and child-process lifecycle.

Covers the process supervisor that boots and health-checks Flask and worker
children, focusing on its threading guards and start/stop state machine.

Main Features:
* join_workers returns promptly from the boot thread and skips the main thread on Windows
* The health loop clears a preset stop flag and spawns a live watcher thread
* start_child refuses to spawn while stopping but is allowed while starting
* start_in_background owns the boot thread and invokes start_all
"""

import importlib.util
import os
import sys
import threading
import time

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
NATIVE_BUILD = os.path.join(REPO_ROOT, 'native-build')

SKIPS_MAIN_THREAD = {'windows'}


def _ensure_path(entry):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _load_supervisor(platform_name):
    _ensure_path(REPO_ROOT)
    _ensure_path(NATIVE_BUILD)
    mod_name = 'native_supervisor_under_test_' + platform_name
    path = os.path.join(NATIVE_BUILD, platform_name, 'supervisor.py')
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        pytest.skip(f"{platform_name} supervisor does not import on this platform: {exc!r}")
    return mod


def _bare_supervisor(mod):
    sup = mod.ProcessSupervisor.__new__(mod.ProcessSupervisor)
    sup._lock = threading.RLock()
    sup._children = {}
    sup._desired = set()
    sup._state = 'stopped'
    sup._stop_requested = threading.Event()
    sup._health_stop = threading.Event()
    sup._health_thread = None
    sup._boot_thread = None
    sup._database_url = 'postgresql://unused'
    sup._redis_url = 'redis://unused'
    sup._db_conn = 'postgresql://unused'

    class _Log:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    sup._log = _Log()
    return sup


PLATFORMS = ['linux', 'macos', 'windows']


@pytest.fixture(params=PLATFORMS)
def supervisor_case(request):
    platform_name = request.param
    mod = _load_supervisor(platform_name)
    return platform_name, mod


class TestJoinWorkersSkips:
    def test_returns_promptly_when_boot_thread_is_current(self, supervisor_case):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        sup._boot_thread = threading.current_thread()
        sup._health_thread = None
        start = time.time()
        sup._join_workers()
        assert time.time() - start < 1.0

    def test_alive_non_current_thread_would_be_joined(self, supervisor_case):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        release = threading.Event()
        joined = threading.Event()

        def _sentinel_body():
            release.wait(2)

        sentinel = threading.Thread(target=_sentinel_body, name='sentinel', daemon=True)
        sentinel.start()
        sup._boot_thread = sentinel
        sup._health_thread = None

        def _runner():
            sup._join_workers()
            joined.set()

        runner = threading.Thread(target=_runner, name='join-runner', daemon=True)
        runner.start()
        assert not joined.wait(0.5)
        release.set()
        assert joined.wait(2.0)
        sentinel.join(2)

    def test_skips_main_thread_only_on_windows(self, supervisor_case):
        platform_name, mod = supervisor_case
        sup = _bare_supervisor(mod)
        sup._boot_thread = threading.main_thread()
        sup._health_thread = None

        done = threading.Event()

        def _runner():
            sup._join_workers()
            done.set()

        runner = threading.Thread(target=_runner, name='main-skip-runner', daemon=True)
        runner.start()

        if platform_name in SKIPS_MAIN_THREAD:
            assert done.wait(1.0)
        else:
            assert not done.wait(0.5)
        runner.join(1)


class TestStartHealthLoopClearsStop:
    def test_clears_preset_stop_and_spawns_live_thread(self, supervisor_case, monkeypatch):
        _, mod = supervisor_case
        sup = _bare_supervisor(mod)
        sup._state = 'running'

        body_ran = threading.Event()

        def _record_and_stop(*_a, **_k):
            body_ran.set()
            sup._health_stop.set()

        for name in ('_ensure_postgres_healthy', '_ensure_redis_healthy'):
            if hasattr(sup, name):
                monkeypatch.setattr(sup, name, _record_and_stop)
        if hasattr(mod, 'urllib'):
            monkeypatch.setattr(mod.urllib.request, 'urlopen', _record_and_stop)

        sup._health_stop.set()
        assert sup._health_stop.is_set()

        real_event = sup._health_stop

        def non_blocking_wait(timeout=None):
            return real_event.is_set()

        monkeypatch.setattr(real_event, 'wait', non_blocking_wait)

        try:
            sup._start_health_loop()
            assert sup._health_thread is not None
            assert sup._health_thread.is_alive() or body_ran.is_set()
            assert body_ran.wait(2.0), "spawned health loop body never executed"
        finally:
            real_event.set()
            if sup._health_thread is not None:
                sup._health_thread.join(2)


class TestSpawnRefusedWhileStopping:
    def test_start_child_refuses_when_stopping(self, supervisor_case, monkeypatch):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)

        popen_calls = []

        def _fake_popen(*a, **k):
            popen_calls.append((a, k))
            raise AssertionError("Popen must not be called once stopping")

        monkeypatch.setattr(mod.subprocess, 'Popen', _fake_popen)

        sup._state = 'stopping'
        sup._stop_requested.set()

        result = sup.start_child('flask')
        assert result is False
        assert popen_calls == []
        assert 'flask' not in sup._children
        assert 'flask' not in sup._desired

    def test_start_child_allowed_while_starting(self, supervisor_case, monkeypatch):
        _, mod = supervisor_case
        sup = _bare_supervisor(mod)

        class _FakePopen:
            def __init__(self, *a, **k):
                self.pid = 4321
                self.stdout = None

            def poll(self):
                return None

        monkeypatch.setattr(mod.subprocess, 'Popen', _FakePopen)
        for name in ('_terminate_named',):
            if hasattr(sup, name):
                monkeypatch.setattr(sup, name, lambda *a, **k: None)
        if hasattr(mod, 'threading'):
            monkeypatch.setattr(
                mod.threading,
                'Thread',
                lambda *a, **k: type('T', (), {'start': lambda self: None, 'daemon': True})(),
            )
        if hasattr(mod, 'db_backend'):
            monkeypatch.setattr(
                mod.db_backend, 'ensure_embedded_running', lambda *a, **k: 'postgresql://x'
            )
        if hasattr(mod, 'env_builder'):
            monkeypatch.setattr(mod.env_builder, 'build_child_env', lambda *a, **k: {})
        if hasattr(sup, '_ensure_redis_running'):
            monkeypatch.setattr(sup, '_ensure_redis_running', lambda *a, **k: 'redis://x')

        sup._state = 'starting'
        result = sup.start_child('flask')
        assert result is True
        assert 'flask' in sup._desired


class TestStartInBackground:
    def test_owns_boot_thread_and_invokes_start_all(self, supervisor_case, monkeypatch):
        platform_name, mod = supervisor_case
        if not hasattr(mod.ProcessSupervisor, 'start_in_background'):
            pytest.skip(f"{platform_name} supervisor has no start_in_background")

        sup = _bare_supervisor(mod)
        started = threading.Event()
        boot_thread_seen = {}

        def _fake_start_all():
            boot_thread_seen['thread'] = threading.current_thread()
            started.set()

        monkeypatch.setattr(sup, 'start_all', _fake_start_all)
        monkeypatch.setattr(sup, 'is_running', lambda: False)

        thread = sup.start_in_background()
        assert thread is sup._boot_thread
        assert started.wait(2.0)
        assert boot_thread_seen['thread'] is sup._boot_thread
        assert boot_thread_seen['thread'] is not threading.current_thread()
        thread.join(2)
