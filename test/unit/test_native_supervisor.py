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
* A Windows console close stops everything instead of resurrecting children
* Startup reaps role children a dead supervisor left behind, and nothing else
* Linux and macOS reap restarted role children too, a closed terminal stops the
  Linux stack in order, and supervisor tracebacks land in a crash file
* join_workers returns promptly from the boot thread and skips the main thread on Windows
* The health loop clears a preset stop flag and spawns a live watcher thread
* All three platforms share one health loop, and none keeps a private copy
* A stop requested during the database probe restarts no child
* start_child refuses to spawn while stopping but is allowed while starting
* start_in_background owns the boot thread and invokes start_all
* The database probe holds one autocommit session and reuses it across ticks
* A dropped session is replaced once before the database is declared unhealthy
* An unreachable database returns False and is logged with its traceback
* A programming error in the probe surfaces instead of faking an unhealthy database
"""

import importlib.util
import os
import pathlib
import subprocess
import sys
import threading
import time
import types
import urllib.request
from unittest.mock import MagicMock

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
    sup._db_conn = 'postgresql://unused'

    class _Log:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    sup._log = _Log()
    return sup


PLATFORMS = ['linux', 'macos', 'windows']


class _ProbeError(Exception):
    pass


class _ProbeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def execute(self, sql):
        if self._conn.probe_fails:
            raise _ProbeError('server closed the connection unexpectedly')
        self._conn.executed.append(sql)

    def fetchone(self):
        return (1,)


class _ProbeConnection:
    def __init__(self, probe_fails=False):
        self.closed = 0
        self.autocommit = None
        self.executed = []
        self.probe_fails = probe_fails

    def cursor(self):
        if self.closed:
            raise _ProbeError('connection already closed')
        return _ProbeCursor(self)

    def set_session(self, autocommit=False):
        self.autocommit = autocommit

    def close(self):
        self.closed = 1


def _fake_psycopg2(connect):
    return types.SimpleNamespace(connect=connect, Error=_ProbeError)


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

        if hasattr(sup, '_ensure_postgres_healthy'):
            monkeypatch.setattr(sup, '_ensure_postgres_healthy', _record_and_stop)
        monkeypatch.setattr(urllib.request, 'urlopen', _record_and_stop)

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

        monkeypatch.setattr(subprocess, 'Popen', _fake_popen)

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

        monkeypatch.setattr(subprocess, 'Popen', _FakePopen)
        for name in ('_terminate_named',):
            if hasattr(sup, name):
                monkeypatch.setattr(sup, name, lambda *a, **k: True)
        monkeypatch.setattr(
            threading,
            'Thread',
            lambda *a, **k: type('T', (), {'start': lambda self: None, 'daemon': True})(),
        )
        if hasattr(mod, 'db_backend'):
            monkeypatch.setattr(
                mod.db_backend, 'ensure_embedded_running', lambda *a, **k: 'postgresql://x'
            )
        if hasattr(mod, 'env_builder'):
            monkeypatch.setattr(mod.env_builder, 'build_child_env', lambda *a, **k: {})

        sup._state = 'starting'
        result = sup.start_child('flask')
        assert result is True
        assert 'flask' in sup._desired

    def test_start_is_idempotent_for_an_existing_live_child(self, supervisor_case, monkeypatch):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        sup._state = 'running'

        class _Live:
            def poll(self):
                return None

        live = _Live()
        sup._children['queue-worker-default'] = live
        popen = MagicMock(side_effect=AssertionError('must not spawn a duplicate child'))
        monkeypatch.setattr(subprocess, 'Popen', popen)

        assert sup.start_child('queue-worker-default') is True
        assert sup._children['queue-worker-default'] is live
        assert 'queue-worker-default' in sup._desired
        popen.assert_not_called()


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


@pytest.mark.parametrize('platform_name', PLATFORMS)
def test_control_reply_waits_for_all_services_and_aggregates_failure(
    platform_name, monkeypatch
):
    mod = _load_supervisor(platform_name)
    sup = _bare_supervisor(mod)
    completed = []

    def restart(service):
        completed.append(service)
        return service != 'queue-worker-high'

    monkeypatch.setattr(sup, 'restart_child', restart)

    result = sup.dispatch_control(
        'restart', ['queue-worker-default', 'queue-worker-high', 'queue-maintenance']
    )

    assert completed == ['queue-worker-default', 'queue-worker-high', 'queue-maintenance']
    assert result is False


@pytest.mark.parametrize('platform_name', PLATFORMS)
def test_control_service_exception_is_a_negative_result_and_later_services_run(
    platform_name, monkeypatch
):
    mod = _load_supervisor(platform_name)
    sup = _bare_supervisor(mod)
    completed = []

    def stop(service):
        completed.append(service)
        if service == 'queue-worker-default':
            raise RuntimeError('stop failed')
        return True

    monkeypatch.setattr(sup, 'stop_child', stop)

    assert sup.dispatch_control('stop', ['queue-worker-default', 'queue-maintenance']) is False
    assert completed == ['queue-worker-default', 'queue-maintenance']


@pytest.mark.parametrize('platform_name', PLATFORMS)
def test_unstoppable_child_returns_false_and_remains_tracked(platform_name, monkeypatch):
    mod = _load_supervisor(platform_name)
    sup = _bare_supervisor(mod)

    class _StubbornProcess:
        pid = 4321
        stdout = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise TimeoutError('still alive')

        def send_signal(self, _signal):
            return None

        def kill(self):
            return None

    proc = _StubbornProcess()
    sup._children['queue-worker-default'] = proc
    sup._desired.add('queue-worker-default')
    if platform_name != 'windows':
        monkeypatch.setattr(os, 'getpgid', lambda _pid: 4321, raising=False)
        monkeypatch.setattr(os, 'killpg', lambda *_args: None, raising=False)

    assert sup.stop_child('queue-worker-default') is False
    assert sup._children['queue-worker-default'] is proc
    assert 'queue-worker-default' in sup._desired


@pytest.mark.parametrize('platform_name', ['linux', 'macos'])
def test_postgres_health_uses_lossless_socket_kwargs(platform_name, monkeypatch):
    mod = _load_supervisor(platform_name)
    sup = _bare_supervisor(mod)
    socket_dir = '/tmp/Audio&Muse+#socket'
    sup._database_url = f'postgresql://postgres:@/postgres?host={socket_dir}'
    seen = {}
    conn = _ProbeConnection()

    def connect(*args, **kwargs):
        seen['args'] = args
        seen['kwargs'] = kwargs
        return conn

    monkeypatch.setitem(sys.modules, 'psycopg2', _fake_psycopg2(connect))
    restart = MagicMock(side_effect=AssertionError('healthy PostgreSQL must not be restarted'))
    if hasattr(mod, 'db_backend'):
        monkeypatch.setattr(mod.db_backend, 'ensure_embedded_running', restart)
    else:
        monkeypatch.setattr(mod.database, 'ensure_embedded_running', restart)

    sup._ensure_postgres_healthy()

    assert seen['args'] == ()
    assert seen['kwargs']['host'] == socket_dir
    assert seen['kwargs']['port'] == '5432'
    assert seen['kwargs']['user'] == 'postgres'
    assert seen['kwargs']['dbname'] == 'postgres'
    assert conn.executed == ['SELECT 1']
    assert conn.autocommit is True
    assert conn.closed == 0
    assert sup._probe_connection is conn
    restart.assert_not_called()


class TestTheHealthLoopHasOneImplementationForEveryPlatform:
    def test_each_supervisor_inherits_the_shared_loop(self, supervisor_case):
        from native_common.supervisor_health import HealthLoopMixin

        _platform, mod = supervisor_case

        assert issubclass(mod.ProcessSupervisor, HealthLoopMixin)

    def test_no_platform_keeps_its_own_copy(self, supervisor_case):
        from native_common import supervisor_health

        _platform, mod = supervisor_case

        assert (mod.ProcessSupervisor._health_loop
                is supervisor_health.HealthLoopMixin._health_loop)
        assert (mod.ProcessSupervisor._start_health_loop
                is supervisor_health.HealthLoopMixin._start_health_loop)

    def test_a_stop_during_the_database_probe_restarts_nothing(
        self, supervisor_case, monkeypatch
    ):
        from native_common import supervisor_health

        _platform, mod = supervisor_case
        monkeypatch.setattr(supervisor_health, 'HEALTH_INTERVAL_SECONDS', 0.01)
        sup = _bare_supervisor(mod)
        sup._state = 'running'
        dead = MagicMock()
        dead.poll.return_value = 1
        sup._children = {'flask': dead}
        sup._desired = {'flask'}
        started = []
        monkeypatch.setattr(sup, 'start_child', lambda name: started.append(name))
        monkeypatch.setattr(
            sup, '_ensure_postgres_healthy', lambda: sup._health_stop.set()
        )

        sup._health_loop()

        assert started == []

    def test_a_dead_child_is_restarted_while_the_supervisor_runs(
        self, supervisor_case, monkeypatch
    ):
        from native_common import supervisor_health

        _platform, mod = supervisor_case
        monkeypatch.setattr(supervisor_health, 'HEALTH_INTERVAL_SECONDS', 0.01)
        sup = _bare_supervisor(mod)
        sup._state = 'running'
        dead = MagicMock()
        dead.poll.return_value = 1
        sup._children = {'flask': dead}
        sup._desired = {'flask'}
        started = []

        def _restart(name):
            started.append(name)
            sup._health_stop.set()

        monkeypatch.setattr(sup, 'start_child', _restart)
        monkeypatch.setattr(sup, '_ensure_postgres_healthy', lambda: None)

        sup._health_loop()

        assert started == ['flask']


class TestTheDatabaseProbeHoldsOneSession:
    def test_a_second_tick_reuses_the_session_instead_of_reconnecting(
        self, supervisor_case, monkeypatch
    ):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        opened = []

        def connect(**_kwargs):
            conn = _ProbeConnection()
            opened.append(conn)
            return conn

        monkeypatch.setitem(sys.modules, 'psycopg2', _fake_psycopg2(connect))

        assert sup._probe_postgres(host='/tmp/sock', port='5432') is True
        assert sup._probe_postgres(host='/tmp/sock', port='5432') is True

        assert len(opened) == 1
        assert opened[0].executed == ['SELECT 1', 'SELECT 1']
        assert opened[0].autocommit is True
        assert opened[0].closed == 0

    def test_a_dropped_session_is_replaced_once_before_calling_the_database_unhealthy(
        self, supervisor_case, monkeypatch
    ):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        stale = _ProbeConnection(probe_fails=True)
        fresh = _ProbeConnection()
        sup._probe_connection = stale
        monkeypatch.setitem(sys.modules, 'psycopg2', _fake_psycopg2(lambda **_kwargs: fresh))

        assert sup._probe_postgres(host='/tmp/sock') is True

        assert stale.closed == 1
        assert fresh.executed == ['SELECT 1']
        assert sup._probe_connection is fresh

    def test_a_closed_session_is_never_reused(self, supervisor_case, monkeypatch):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        gone = _ProbeConnection()
        gone.close()
        fresh = _ProbeConnection()
        sup._probe_connection = gone
        monkeypatch.setitem(sys.modules, 'psycopg2', _fake_psycopg2(lambda **_kwargs: fresh))

        assert sup._probe_postgres(host='/tmp/sock') is True

        assert gone.executed == []
        assert fresh.executed == ['SELECT 1']

    def test_an_unreachable_database_returns_false_and_logs_the_traceback(
        self, supervisor_case, monkeypatch
    ):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        logged = []

        def _refused(**_kwargs):
            raise _ProbeError('could not connect to server')

        monkeypatch.setitem(sys.modules, 'psycopg2', _fake_psycopg2(_refused))
        sup._log = types.SimpleNamespace(
            exception=lambda msg, *a, **k: logged.append(msg),
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            info=lambda *a, **k: None,
        )

        assert sup._probe_postgres(host='/tmp/sock') is False

        assert len(logged) == 1
        assert getattr(sup, '_probe_connection', None) is None

    def test_a_probe_that_never_runs_is_false_rather_than_healthy(
        self, supervisor_case, monkeypatch
    ):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        dead = _ProbeConnection(probe_fails=True)
        monkeypatch.setitem(sys.modules, 'psycopg2', _fake_psycopg2(lambda **_kwargs: dead))

        assert sup._probe_postgres(host='/tmp/sock') is False

        assert dead.closed == 1
        assert getattr(sup, '_probe_connection', None) is None

    def test_a_misused_connect_call_surfaces_instead_of_faking_an_unhealthy_database(
        self, supervisor_case, monkeypatch
    ):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)

        logged = []

        def _misused(**_kwargs):
            raise TypeError("connect() got an unexpected keyword argument 'keepalives'")

        monkeypatch.setitem(sys.modules, 'psycopg2', _fake_psycopg2(_misused))
        sup._log = types.SimpleNamespace(
            exception=lambda msg, *a, **k: logged.append(msg),
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            info=lambda *a, **k: None,
        )

        with pytest.raises(TypeError):
            sup._probe_postgres(host='/tmp/sock')

        assert len(logged) == 1

    def test_a_session_call_that_does_not_exist_surfaces_instead_of_faking_unhealthy(
        self, supervisor_case, monkeypatch
    ):
        _platform, mod = supervisor_case
        sup = _bare_supervisor(mod)
        without_set_session = types.SimpleNamespace(closed=0, close=lambda: None)
        monkeypatch.setitem(
            sys.modules, 'psycopg2', _fake_psycopg2(lambda **_kwargs: without_set_session)
        )

        with pytest.raises(AttributeError):
            sup._probe_postgres(host='/tmp/sock')

        assert getattr(sup, '_probe_connection', None) is None

    def test_the_health_loop_closes_the_held_session_when_it_exits(
        self, supervisor_case, monkeypatch
    ):
        from native_common import supervisor_health

        _platform, mod = supervisor_case
        monkeypatch.setattr(supervisor_health, 'HEALTH_INTERVAL_SECONDS', 0.01)
        sup = _bare_supervisor(mod)
        sup._state = 'running'
        held = _ProbeConnection()
        sup._probe_connection = held
        monkeypatch.setattr(sup, '_ensure_postgres_healthy', lambda: sup._health_stop.set())

        sup._health_loop()

        assert held.closed == 1
        assert sup._probe_connection is None


class _FakeProc:
    def __init__(self, pid, name, cmdline, exe=None):
        self.info = {'pid': pid, 'name': name, 'cmdline': cmdline}
        self._exe = exe
        self.exe_calls = 0
        self.terminated = False

    def exe(self):
        self.exe_calls += 1
        if self._exe is None:
            raise RuntimeError('access denied')
        return self._exe

    def terminate(self):
        self.terminated = True


def _fake_psutil(procs):
    mod = types.ModuleType('psutil')
    mod.NoSuchProcess = type('NoSuchProcess', (Exception,), {})
    mod.AccessDenied = type('AccessDenied', (Exception,), {})
    mod.process_iter = lambda attrs=None: list(procs)
    return mod


class TestATornDownWindowsSupervisorLeavesNoOrphanBehind:
    def _windows(self, monkeypatch):
        mod = _load_supervisor('windows')
        sup = _bare_supervisor(mod)
        monkeypatch.setattr(mod.paths, 'pgdata_dir', lambda: 'C:\\Data\\pgdata')
        return mod, sup

    def test_startup_reaps_role_children_left_by_an_earlier_supervisor(self, monkeypatch):
        mod, sup = self._windows(monkeypatch)
        exe = sys.executable
        mine = MagicMock(pid=77)
        mine.poll.return_value = None
        sup._children['queue-worker-high'] = mine
        procs = [
            _FakeProc(os.getpid(), 'me', [exe, '--role=flask']),
            _FakeProc(77, 'child', [exe, '--role=worker-high']),
            _FakeProc(501, 'orphan', [exe, '--role=worker-default']),
            _FakeProc(502, 'orphan', [exe, '--role=worker-high']),
            _FakeProc(503, 'other', ['/opt/other/python', '--role=worker-high'], exe='/opt/other/python'),
            _FakeProc(504, 'postgres', ['C:/pg/bin/postgres.exe', '-D', 'C:/Data/pgdata']),
            _FakeProc(505, 'shell', [exe, '-c', 'print(1)']),
            _FakeProc(506, 'orphan', ['AudioMuse-AI.exe', '--role=maintenance'], exe=exe),
            _FakeProc(507, 'postgres', ['C:/pg/bin/postgres.exe', '-D', 'C:/Other/pgdata']),
            _FakeProc(508, 'postgres', ['C:/pg/bin/postgres.exe', '-D', 'C:/Data/pgdata_backup']),
            _FakeProc(509, 'other install', ['AudioMuse-AI.exe', '--role=flask'],
                      exe='C:/Program Files/AudioMuse-AI/AudioMuse-AI.exe'),
        ]
        monkeypatch.setitem(sys.modules, 'psutil', _fake_psutil(procs))

        sup._reap_orphans()

        assert [p.info['pid'] for p in procs if p.terminated] == [501, 502, 504, 506], (
            'the workers a dead supervisor restarted kept draining the queue in '
            'silence; our own live child and unrelated processes must be spared, '
            'a child spawned from a relative launch path is still ours, the '
            'postgres pgserver started names our data dir with forward slashes, '
            'a sibling data dir is not ours, and another install launched by the '
            'same file name is not ours either'
        )
        assert procs[6].exe_calls == 0 and procs[2].exe_calls == 0, (
            'the executable image is resolved only for a role child whose argv[0] '
            'did not already settle it; never for every process on the machine'
        )

    def test_two_installs_launched_by_the_same_file_name_never_reap_each_other(self, monkeypatch):
        mod, _sup = self._windows(monkeypatch)
        monkeypatch.setattr(sys, 'frozen', True, raising=False)
        monkeypatch.setattr(sys, 'argv', ['AudioMuse-AI.exe', 'start'])

        own = mod.own_executables()

        assert os.path.normcase('AudioMuse-AI.exe') not in own
        assert os.path.normcase(os.path.abspath('AudioMuse-AI.exe')) not in own, (
            'a bare or CWD-relative launch name is shared by every install of the '
            'same file name; only the executable image identifies ours'
        )
        assert os.path.normcase(sys.executable) in own
        roles = ['flask']
        assert mod.stale_role_child(['AudioMuse-AI.exe', '--role=flask'], own, roles) is None
        assert mod.stale_role_child(
            ['AudioMuse-AI.exe', '--role=flask'], own, roles,
            exe_of=lambda: 'C:/Program Files/Other/AudioMuse-AI.exe',
        ) is None
        assert mod.stale_role_child(
            ['AudioMuse-AI.exe', '--role=flask'], own, roles, exe_of=lambda: sys.executable,
        ) == 'flask'
        assert mod.stale_role_child([sys.executable, '--role=nope'], own, roles) is None
        assert mod.references_pgdata(['postgres', '-D', 'C:\\Data\\pgdata\\'], 'C:/Data/pgdata')
        assert not mod.references_pgdata(['postgres', '-D', 'C:/Data/pgdata_backup'], 'C:/Data/pgdata')
        assert mod.references_pgdata(['pg_ctl', '-DC:/Data/pgdata', 'start'], 'C:\\Data\\pgdata')

    def test_a_child_killed_with_the_console_is_never_restarted(self, monkeypatch):
        mod, sup = self._windows(monkeypatch)
        popen = MagicMock(stdout=iter(()), returncode=0xC000013A)
        sup._children['queue-worker-high'] = popen
        sup._desired.add('queue-worker-high')
        sup._state = 'running'
        sup.start_child = MagicMock()
        started = time.monotonic()

        sup._pump('queue-worker-high', popen)

        assert not sup.start_child.called, (
            'STATUS_CONTROL_C_EXIT says the child died with the console; a restart '
            'would outlive the supervisor as an orphan'
        )
        assert time.monotonic() - started < 0.5, 'the exit code decides; no grace wait'

    def test_the_log_pump_never_restarts_a_child_while_a_stop_is_requested(self, monkeypatch):
        mod, sup = self._windows(monkeypatch)
        popen = MagicMock(stdout=iter(()))
        sup._children['queue-worker-high'] = popen
        sup._desired.add('queue-worker-high')
        sup._state = 'running'
        sup.start_child = MagicMock()

        sup._stop_requested.set()
        sup._pump('queue-worker-high', popen)
        assert not sup.start_child.called, (
            'a console close kills every child at once; resurrecting one here '
            'left it running with no supervisor after the tray died'
        )

        sup._stop_requested.clear()
        sup._pump('queue-worker-high', popen)
        sup.start_child.assert_called_once_with('queue-worker-high')

    def test_the_log_pump_waits_for_a_stop_that_arrives_just_after_the_child_died(self, monkeypatch):
        mod, sup = self._windows(monkeypatch)
        popen = MagicMock(stdout=iter(()))
        sup._children['queue-worker-high'] = popen
        sup._desired.add('queue-worker-high')
        sup._state = 'running'
        sup.start_child = MagicMock()
        threading.Timer(0.2, sup._stop_requested.set).start()

        sup._pump('queue-worker-high', popen)

        assert not sup.start_child.called, (
            'a console close kills every attached child before it reaches the '
            'supervisor handler; a restart decided in that gap outlived the '
            'supervisor as an orphan (the 19:15 crash)'
        )

    def test_a_console_close_kills_the_children_and_stops_postgres_before_the_orderly_stop(self, monkeypatch):
        mod, sup = self._windows(monkeypatch)
        sup._log = MagicMock(handlers=[])
        order = []
        child = MagicMock()
        child.poll.return_value = None
        child.kill.side_effect = lambda: order.append('kill')
        sup._children['flask'] = child
        sup._desired.add('flask')
        monkeypatch.setattr(mod.db_backend, 'stop_embedded', lambda: order.append('stop_embedded'))
        sup._clear_pidfile = lambda: order.append('pidfile')
        sup.stop_all = MagicMock(side_effect=lambda: order.append('stop_all'))

        assert sup._console_event(2) is True

        assert order == ['kill', 'stop_embedded', 'pidfile', 'stop_all'] and child.wait.called, (
            'Windows gives the handler five seconds and the console CTRL_BREAK '
            'would travel through is already gone: the children must die outright '
            'and postgres must stop before the orderly stop and its thread joins, '
            'or the supervisor is killed mid-way with postgres left behind'
        )
        assert not sup._desired

    def test_a_console_close_stops_everything_and_flushes_the_log(self, monkeypatch):
        mod, sup = self._windows(monkeypatch)
        handler = MagicMock()
        sup._log = MagicMock(handlers=[handler])
        sup.stop_all = MagicMock()
        monkeypatch.setattr(mod.db_backend, 'stop_embedded', lambda: None)
        sup._clear_pidfile = lambda: None

        assert sup._console_event(0) is False
        assert not sup.stop_all.called and not sup._stop_requested.is_set()

        assert sup._console_event(2) is True
        assert sup._stop_requested.is_set()
        sup.stop_all.assert_called_once_with()
        assert sup._log.warning.call_args.args[1] == 'console closed'
        assert handler.flush.call_count == 3, (
            'the newest-first log flushes on a timer; a dying process must flush '
            'by hand (after the reason, after the fast teardown, at the end) or '
            'the reason it stopped is lost'
        )

    def test_installing_the_console_handler_is_a_no_op_off_windows(self, monkeypatch):
        mod, sup = self._windows(monkeypatch)
        monkeypatch.setattr(sys, 'platform', 'linux')

        assert sup.install_console_handler() is False


class TestEveryPlatformReapsTheOrphansADeadSupervisorLeft:
    @pytest.mark.parametrize('platform_name', ['linux', 'macos'])
    def test_posix_startup_reaps_restarted_role_children_too(self, platform_name, monkeypatch):
        mod = _load_supervisor(platform_name)
        sup = _bare_supervisor(mod)
        sup._clear_pidfile = lambda: None
        monkeypatch.setattr(sup.paths, 'pgdata_dir', lambda: '/data/pgdata')
        exe = sys.executable
        mine = MagicMock(pid=77)
        mine.poll.return_value = None
        sup._children['queue-worker-high'] = mine
        procs = [
            _FakeProc(os.getpid(), 'me', [exe, '--role=flask']),
            _FakeProc(77, 'child', [exe, '--role=worker-high']),
            _FakeProc(601, 'orphan', [exe, '--role=worker-default']),
            _FakeProc(602, 'other', ['/opt/other/python', '--role=worker-high'], exe='/opt/other/python'),
            _FakeProc(603, 'postgres', ['postgres', '-D', '/data/pgdata']),
            _FakeProc(604, 'orphan', ['/usr/bin/audiomuse-ai', '--role=maintenance'], exe=exe),
            _FakeProc(605, 'postgres', ['postgres', '-D', '/data/pgdata_old']),
        ]
        fake = _fake_psutil(procs)
        fake.wait_procs = lambda procs, timeout=None: (list(procs), [])
        monkeypatch.setitem(sys.modules, 'psutil', fake)

        sup._reap_stale_infra()

        assert [p.info['pid'] for p in procs if p.terminated] == [601, 603, 604], (
            'the pid file only knows the children of the last boot; a child the '
            'health loop restarted later outlived a dead supervisor unseen, the '
            'deb launches through a /usr/bin symlink the argv never shows, and a '
            'sibling data directory is not ours'
        )

    def test_a_closed_terminal_stops_the_linux_stack_like_ctrl_c(self):
        source = (pathlib.Path(NATIVE_BUILD) / 'linux' / 'launcher.py').read_text(encoding='utf-8')
        handler = source.split('def _run_supervisor', 1)[1].split('def _cmd_stop', 1)[0]

        assert 'signal.signal(signal.SIGHUP, _handle_signal)' in handler, (
            'children run in their own sessions, so a SIGHUP that kills only the '
            'supervisor leaves the whole stack running with nobody supervising it'
        )


class TestSupervisorFatalErrorsAreWrittenToAFile:
    def test_unhandled_exceptions_land_in_the_crash_log(self, tmp_path, monkeypatch):
        import faulthandler
        import importlib.util as util

        spec = util.spec_from_file_location(
            'crash_log_under_test', os.path.join(NATIVE_BUILD, 'native_common', 'crash_log.py')
        )
        crash_log = util.module_from_spec(spec)
        spec.loader.exec_module(crash_log)
        monkeypatch.setattr(sys, 'excepthook', sys.excepthook)
        monkeypatch.setattr(threading, 'excepthook', threading.excepthook)
        monkeypatch.setattr(sys, '__excepthook__', lambda *a: None)

        path = crash_log.capture_fatal_errors(str(tmp_path))
        try:
            assert faulthandler.is_enabled()
            try:
                raise RuntimeError('the supervisor blew up')
            except RuntimeError:
                sys.excepthook(*sys.exc_info())
        finally:
            faulthandler.disable()
            crash_log._crash_log.close()

        text = pathlib.Path(path).read_text(encoding='utf-8')
        assert path.endswith('supervisor-crash.log')
        assert 'unhandled exception in the supervisor' in text
        assert 'RuntimeError: the supervisor blew up' in text, (
            'a hidden console swallowed every supervisor traceback; the file is '
            'the only place a crash can be read back from'
        )
