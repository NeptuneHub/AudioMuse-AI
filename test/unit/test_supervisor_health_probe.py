# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The held PostgreSQL probe connection of the native supervisor health loop.

The loop used to open, query and close a connection every five seconds on every
native platform, which is roughly seventeen thousand backend forks a day on an
idle standalone install. These tests pin the replacement: one session held
across ticks, a real round trip on every tick so a stale session can never pass
for health, exactly one reconnect before the database is declared dead, and a
clean close when the loop exits.

Main Features:
* One connection is opened and reused for every probe
* The held session is autocommit so it never idles inside a transaction
* A probe that errors drops the session, reconnects once and stays healthy
* A dead database is unhealthy after that single reconnect and gets restarted
* A closed or stale session is replaced before it can report health
* Leaving the health loop closes the session, and closing twice is harmless
* A bug in the probe is logged and raised, never reported as an unreachable server
"""

import importlib.util
import os
import sys
import threading
import types
from unittest.mock import MagicMock

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
NATIVE_BUILD = os.path.join(REPO_ROOT, 'native-build')

for _entry in (REPO_ROOT, NATIVE_BUILD):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from native_common.supervisor_health import HealthLoopMixin  # noqa: E402

SOCKET_DIR = '/tmp/Audio&Muse+#socket'
SOCKET_URL = f'postgresql://postgres:@/postgres?host={SOCKET_DIR}'


class _SilentLog:
    def __getattr__(self, _name):
        return lambda *a, **k: None


class _RecordingLog(_SilentLog):
    def __init__(self):
        self.exceptions = []

    def exception(self, msg, *_a, **_k):
        self.exceptions.append(msg)


class _ProbeError(Exception):
    pass


class _StubCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql):
        self._conn.executed.append(sql)
        if self._conn.probe_error is not None:
            raise self._conn.probe_error

    def fetchone(self):
        return (1,)


class _StubConnection:
    def __init__(self, probe_error=None):
        self.probe_error = probe_error
        self.executed = []
        self.closed = 0
        self.autocommit = False
        self.close_calls = 0

    def cursor(self):
        return _StubCursor(self)

    def set_session(self, autocommit=False):
        self.autocommit = autocommit

    def close(self):
        self.close_calls += 1
        self.closed = 1


class _Psycopg2Stub:
    def __init__(self, connections=None, connect_error=None):
        self.connections = list(connections or [])
        self.connect_error = connect_error
        self.calls = []
        self.opened = []

    def connect(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.connect_error is not None:
            raise self.connect_error
        conn = self.connections.pop(0) if self.connections else _StubConnection()
        self.opened.append(conn)
        return conn


def _install(monkeypatch, stub):
    monkeypatch.setitem(
        sys.modules,
        'psycopg2',
        types.SimpleNamespace(connect=stub.connect, Error=_ProbeError),
    )
    return stub


class _MixinHost(HealthLoopMixin):
    def __init__(self):
        self._lock = threading.RLock()
        self._log = _SilentLog()
        self._state = 'running'
        self._children = {}
        self._desired = set()
        self._health_stop = threading.Event()
        self._health_thread = None


class TestTheProbeConnectionIsHeld:
    def test_one_connection_is_opened_and_reused_for_every_probe(self, monkeypatch):
        stub = _install(monkeypatch, _Psycopg2Stub())
        host = _MixinHost()

        assert host._probe_postgres(host='127.0.0.1', port='5432') is True
        assert host._probe_postgres(host='127.0.0.1', port='5432') is True

        assert len(stub.calls) == 1
        conn = stub.opened[0]
        assert conn.executed == ['SELECT 1', 'SELECT 1']
        assert conn.close_calls == 0

    def test_the_held_session_is_autocommit_so_it_never_idles_in_a_transaction(self, monkeypatch):
        stub = _install(monkeypatch, _Psycopg2Stub())
        host = _MixinHost()

        assert host._probe_postgres(host='127.0.0.1', port='5432') is True

        assert stub.opened[0].autocommit is True

    def test_the_probe_bounds_its_connect_and_asks_for_keepalives(self, monkeypatch):
        stub = _install(monkeypatch, _Psycopg2Stub())
        host = _MixinHost()

        host._probe_postgres(host='127.0.0.1', port='5432')

        _args, kwargs = stub.calls[0]
        assert kwargs['connect_timeout'] > 0
        assert kwargs['keepalives'] == 1
        assert kwargs['keepalives_idle'] > 0
        assert kwargs['keepalives_interval'] > 0
        assert kwargs['keepalives_count'] > 0


class TestAFailedProbeReconnects:
    def test_a_held_session_that_errors_is_dropped_and_the_tick_stays_healthy(self, monkeypatch):
        held = _StubConnection()
        fresh = _StubConnection()
        stub = _install(monkeypatch, _Psycopg2Stub(connections=[held, fresh]))
        host = _MixinHost()
        assert host._probe_postgres(host='127.0.0.1', port='5432') is True

        held.probe_error = _ProbeError('server closed the connection')

        assert host._probe_postgres(host='127.0.0.1', port='5432') is True
        assert len(stub.calls) == 2
        assert held.executed == ['SELECT 1', 'SELECT 1']
        assert held.close_calls == 1
        assert fresh.executed == ['SELECT 1']
        assert host._held_probe_conn() is fresh

    def test_a_reconnect_that_also_fails_reports_unhealthy_and_forgets_the_session(
        self, monkeypatch
    ):
        held = _StubConnection()
        also_broken = _StubConnection(probe_error=_ProbeError('server closed the connection'))
        stub = _install(monkeypatch, _Psycopg2Stub(connections=[held, also_broken]))
        host = _MixinHost()
        assert host._probe_postgres(host='127.0.0.1', port='5432') is True

        held.probe_error = _ProbeError('server closed the connection')

        assert host._probe_postgres(host='127.0.0.1', port='5432') is False
        assert len(stub.calls) == 2
        assert held.close_calls == 1
        assert also_broken.close_calls == 1
        assert host._held_probe_conn() is None

    @pytest.mark.parametrize(
        'connect_error',
        [OSError('connection refused'), _ProbeError('could not connect to server')],
        ids=['os_error', 'psycopg2_error'],
    )
    def test_an_unreachable_database_is_unhealthy_after_a_single_connect_attempt(
        self, monkeypatch, connect_error
    ):
        stub = _install(monkeypatch, _Psycopg2Stub(connect_error=connect_error))
        host = _MixinHost()
        host._log = _RecordingLog()

        assert host._probe_postgres(host='127.0.0.1', port='5432') is False

        assert len(stub.calls) == 1
        assert len(host._log.exceptions) == 1
        assert host._held_probe_conn() is None

    def test_a_stale_held_session_never_reports_health_without_a_fresh_round_trip(
        self, monkeypatch
    ):
        held = _StubConnection()
        stub = _install(monkeypatch, _Psycopg2Stub(connections=[held]))
        host = _MixinHost()
        assert host._probe_postgres(host='127.0.0.1', port='5432') is True

        held.probe_error = _ProbeError('server closed the connection')
        stub.connect_error = OSError('connection refused')

        assert host._probe_postgres(host='127.0.0.1', port='5432') is False
        assert held.executed == ['SELECT 1', 'SELECT 1']
        assert len(stub.calls) == 2
        assert held.close_calls == 1
        assert host._held_probe_conn() is None

    def test_a_session_closed_underneath_us_is_replaced_before_the_next_probe(self, monkeypatch):
        first = _StubConnection()
        second = _StubConnection()
        stub = _install(monkeypatch, _Psycopg2Stub(connections=[first, second]))
        host = _MixinHost()
        assert host._probe_postgres(host='127.0.0.1', port='5432') is True

        first.closed = 1

        assert host._probe_postgres(host='127.0.0.1', port='5432') is True
        assert len(stub.calls) == 2
        assert second.executed == ['SELECT 1']


class TestABugInTheProbeIsNotAnUnreachableDatabase:
    def test_a_bug_in_the_round_trip_is_raised_instead_of_reported_unhealthy(self, monkeypatch):
        held = _StubConnection()
        stub = _install(monkeypatch, _Psycopg2Stub(connections=[held]))
        host = _MixinHost()
        host._log = _RecordingLog()
        assert host._probe_postgres(host='127.0.0.1', port='5432') is True

        held.probe_error = AttributeError("'_StubCursor' object has no attribute 'fetchall'")

        with pytest.raises(AttributeError):
            host._probe_postgres(host='127.0.0.1', port='5432')
        assert len(stub.calls) == 1
        assert held.close_calls == 0
        assert len(host._log.exceptions) == 1

    def test_a_bug_while_opening_the_probe_is_logged_and_raised_not_reported_unhealthy(
        self, monkeypatch
    ):
        misuse = AttributeError("'NoneType' object has no attribute 'copy'")
        stub = _install(monkeypatch, _Psycopg2Stub(connect_error=misuse))
        host = _MixinHost()
        host._log = _RecordingLog()

        with pytest.raises(AttributeError):
            host._probe_postgres(host='127.0.0.1', port='5432')

        assert len(stub.calls) == 1
        assert len(host._log.exceptions) == 1
        assert host._held_probe_conn() is None


class TestTheSessionIsReleasedOnShutdown:
    def test_leaving_the_health_loop_closes_the_probe_connection(self, monkeypatch):
        _install(monkeypatch, _Psycopg2Stub())
        host = _MixinHost()
        held = _StubConnection()
        host._probe_connection = held
        host._health_stop.set()

        host._health_loop()

        assert held.close_calls == 1
        assert host._held_probe_conn() is None

    def test_closing_twice_closes_once_and_forgets_the_session(self, monkeypatch):
        _install(monkeypatch, _Psycopg2Stub())
        host = _MixinHost()
        held = _StubConnection()
        host._probe_connection = held

        host._close_probe_conn()
        host._close_probe_conn()

        assert held.close_calls == 1
        assert host._held_probe_conn() is None

    def test_a_close_failure_is_swallowed_so_shutdown_continues(self, monkeypatch):
        _install(monkeypatch, _Psycopg2Stub())
        host = _MixinHost()
        held = MagicMock()
        held.close.side_effect = RuntimeError('socket already gone')
        host._probe_connection = held

        host._close_probe_conn()

        assert host._held_probe_conn() is None


def _load_supervisor(platform_name):
    mod_name = 'native_supervisor_probe_' + platform_name
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = os.path.join(NATIVE_BUILD, platform_name, 'supervisor.py')
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
    sup._log = _SilentLog()
    sup._database_url = SOCKET_URL
    return sup


def _restart_hook(mod, monkeypatch, restart):
    backend = getattr(mod, 'db_backend', None) or mod.database
    monkeypatch.setattr(backend, 'ensure_embedded_running', restart)


@pytest.mark.parametrize('platform_name', ['linux', 'macos'])
def test_the_platform_probe_passes_lossless_socket_parameters_and_holds_the_session(
    platform_name, monkeypatch
):
    mod = _load_supervisor(platform_name)
    sup = _bare_supervisor(mod)
    stub = _install(monkeypatch, _Psycopg2Stub())
    restart = MagicMock(side_effect=AssertionError('healthy PostgreSQL must not be restarted'))
    _restart_hook(mod, monkeypatch, restart)

    sup._ensure_postgres_healthy()
    sup._ensure_postgres_healthy()

    assert len(stub.calls) == 1
    _args, kwargs = stub.calls[0]
    assert _args == ()
    assert kwargs['host'] == SOCKET_DIR
    assert kwargs['port'] == '5432'
    assert kwargs['user'] == 'postgres'
    assert kwargs['dbname'] == 'postgres'
    conn = stub.opened[0]
    assert conn.executed == ['SELECT 1', 'SELECT 1']
    assert conn.close_calls == 0
    restart.assert_not_called()


@pytest.mark.parametrize('platform_name', ['linux', 'macos'])
def test_a_database_that_stops_answering_is_restarted_on_the_next_tick(
    platform_name, monkeypatch
):
    mod = _load_supervisor(platform_name)
    sup = _bare_supervisor(mod)
    held = _StubConnection()
    stub = _install(monkeypatch, _Psycopg2Stub(connections=[held]))
    restart = MagicMock(return_value=SOCKET_URL)
    _restart_hook(mod, monkeypatch, restart)

    sup._ensure_postgres_healthy()
    restart.assert_not_called()

    held.probe_error = _ProbeError('server closed the connection unexpectedly')
    stub.connect_error = OSError('connection refused')
    sup._ensure_postgres_healthy()

    restart.assert_called_once()
    assert held.close_calls == 1
    assert sup._held_probe_conn() is None
