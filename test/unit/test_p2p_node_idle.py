# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""When the p2p-node service stays idle, and the rows it stores (p2p.node, p2p.store).

The same supervisor program starts in every container, so the node decides
for itself whether to run. These tests drive node.main with every external
step replaced, read the module source for the import rules, and run the
store's onion key and peer list statements against recording cursors.

Main Features:
* outside the Flask container the node idles before any database work
* with P2P sharing off it idles after reloading the settings, writing nothing
* when p2p.available names a reason (the node libraries or the Tor program
  are missing) it records 7001, writes Unavailable with that reason and idles
  before the lock
* with everything in place it takes the lock and hands that session to the
  runtime; a runtime failure records the Error status and exits after a pause
* the status row it writes itself carries state, reason and an empty peer id
* the schema wait and the lock retry loop until they succeed
* p2p.runtime is the only lazy import, and importing p2p.node loads neither
  the runtime nor stem, trio or coincurve
* the onion key row is "ED25519-V3:" plus a 64-byte base64 key; anything else
  is refused, never replaced; a new key is inserted once with ON CONFLICT DO
  NOTHING and the stored one wins
* the saved peer list reads back as a dict; a broken row reads as no list
"""

import ast
import base64
import json
import logging
import os
import subprocess
import sys

import pytest

import config
from p2p import available, ipc, node, store as p2p_store
from p2p.errors import P2PError

ONION_KEY = base64.b64encode(bytes(range(64))).decode('ascii')
OTHER_KEY = base64.b64encode(bytes(range(64, 128))).decode('ascii')
REASON = 'the Tor program (tor) is not installed'


class Idle(Exception):
    pass


class Calls:
    def __init__(self):
        self.log = []


@pytest.fixture
def calls(monkeypatch):
    record = Calls()

    def idle(reason):
        record.log.append(('idle', reason))
        raise Idle(reason)

    monkeypatch.setattr(node, 'idle_forever', idle)
    monkeypatch.setattr(node, 'configure_logging', lambda: None)
    monkeypatch.setattr(node, 'wait_for_schema', lambda: record.log.append(('schema',)))
    monkeypatch.setattr(config, 'refresh_config', lambda: record.log.append(('refresh',)))
    monkeypatch.setattr(node, 'write_status', lambda state, reason: record.log.append(('status', state)))
    monkeypatch.setattr(node, 'acquire_lock', lambda: record.log.append(('lock',)) or 'lock-conn')
    monkeypatch.setattr(node, 'run_node', lambda conn: record.log.append(('run', conn)))
    monkeypatch.setattr(node, '_close', lambda conn: record.log.append(('close', conn)))
    monkeypatch.setattr(node.available, 'unavailable_reason', lambda: None)
    monkeypatch.setattr(node.time, 'sleep', lambda seconds: record.log.append(('sleep', seconds)))
    monkeypatch.setenv('SERVICE_TYPE', 'flask')
    monkeypatch.setattr(config, 'P2P_ENABLED', True)
    return record


def _steps(record):
    return [entry[0] for entry in record.log]


class TestIdleRules:
    @pytest.mark.parametrize('service_type', ['worker', '', 'WORKER'])
    def test_outside_the_flask_container_it_idles_before_any_database_work(self, calls, monkeypatch,
                                                                             service_type):
        monkeypatch.setenv('SERVICE_TYPE', service_type)

        with pytest.raises(Idle):
            node.main()

        assert _steps(calls) == ['idle']

    def test_with_sharing_off_it_idles_after_the_settings_reload_and_writes_nothing(self, calls, monkeypatch):
        monkeypatch.setattr(config, 'P2P_ENABLED', False)

        with pytest.raises(Idle):
            node.main()

        assert _steps(calls) == ['schema', 'refresh', 'idle']

    def test_the_setting_is_read_after_the_reload(self, calls, monkeypatch):
        monkeypatch.setattr(config, 'P2P_ENABLED', True)
        monkeypatch.setattr(config, 'refresh_config',
                            lambda: calls.log.append(('refresh',)) or setattr(config, 'P2P_ENABLED', False))

        with pytest.raises(Idle):
            node.main()

        assert _steps(calls) == ['schema', 'refresh', 'idle']

    @pytest.mark.parametrize('reason', [
        'this build lacks the Python libraries stem, trio',
        REASON,
    ])
    def test_an_unavailable_build_reports_its_reason_and_idles_before_the_lock(self, calls, monkeypatch, caplog,
                                                                              reason):
        monkeypatch.setattr(node.available, 'unavailable_reason', lambda: reason)
        monkeypatch.setattr(node, 'write_status',
                            lambda state, why: calls.log.append(('status', state, why)))

        with caplog.at_level(logging.ERROR, logger='p2p.node'):
            with pytest.raises(Idle) as idled:
                node.main()

        assert calls.log == [
            ('schema',), ('refresh',), ('status', ipc.STATE_UNAVAILABLE, reason), ('idle', reason),
        ]
        assert str(idled.value) == reason
        assert '7001' in caplog.text and reason in caplog.text

    def test_availability_is_checked_only_with_sharing_on(self, calls, monkeypatch):
        asked = []
        monkeypatch.setattr(config, 'P2P_ENABLED', False)
        monkeypatch.setattr(node.available, 'unavailable_reason', lambda: asked.append(True) or REASON)

        with pytest.raises(Idle):
            node.main()

        assert asked == []

    def test_the_node_asks_the_shared_availability_module(self):
        assert node.available is available
        assert not hasattr(node, 'missing_modules')

    def test_with_everything_in_place_it_runs_on_the_lock_session(self, calls):
        node.main()

        assert calls.log == [('schema',), ('refresh',), ('lock',), ('run', 'lock-conn'), ('close', 'lock-conn')]

    def test_a_runtime_failure_records_the_error_and_exits_after_a_pause(self, calls, monkeypatch, caplog):
        def broken(conn):
            raise RuntimeError('port in use')

        monkeypatch.setattr(node, 'run_node', broken)

        with caplog.at_level(logging.ERROR, logger='p2p.node'):
            with pytest.raises(SystemExit) as exited:
                node.main()

        assert exited.value.code == 1
        assert ('status', ipc.STATE_ERROR) in calls.log
        assert ('sleep', node.FAILURE_PAUSE_SECONDS) in calls.log
        assert '7005' in caplog.text

    def test_a_known_p2p_failure_names_its_reason_even_inside_exception_groups(self):
        class Group(Exception):
            def __init__(self, exceptions):
                super().__init__('group')
                self.exceptions = exceptions

        nested = Group([RuntimeError('noise'), Group([P2PError('Tor could not connect to the Tor network in time')])])

        assert node.failure_reason(nested) == 'Tor could not connect to the Tor network in time'
        assert node.failure_reason(RuntimeError('boom')) == node.GENERIC_FAILURE


class _Conn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class TestRetryLoops:
    def test_the_schema_wait_retries_until_ready(self, monkeypatch):
        answers = [False, False, True]
        conns = []
        sleeps = []

        def connect(read_only=False):
            conn = _Conn()
            conns.append(conn)
            return conn

        monkeypatch.setattr(node.p2p_store, 'connect', connect)
        monkeypatch.setattr(node.p2p_store, 'schema_ready', lambda conn: answers.pop(0))
        monkeypatch.setattr(node.time, 'sleep', sleeps.append)

        node.wait_for_schema()

        assert sleeps == [node.SCHEMA_RETRY_SECONDS] * 2
        assert all(conn.closed for conn in conns)

    def test_a_database_error_is_logged_and_retried(self, monkeypatch, caplog):
        attempts = []

        def connect(read_only=False):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError('database starting')
            return _Conn()

        monkeypatch.setattr(node.p2p_store, 'connect', connect)
        monkeypatch.setattr(node.p2p_store, 'schema_ready', lambda conn: True)
        monkeypatch.setattr(node.time, 'sleep', lambda seconds: None)

        with caplog.at_level(logging.ERROR, logger='p2p.node'):
            node.wait_for_schema()

        assert len(attempts) == 2
        assert 'could not reach the database' in caplog.text

    def test_the_lock_is_retried_every_30_seconds_and_kept_on_success(self, monkeypatch):
        results = [False, False, True]
        conns = []
        sleeps = []

        def connect(read_only=False):
            conn = _Conn()
            conns.append(conn)
            return conn

        monkeypatch.setattr(node.p2p_store, 'connect', connect)
        monkeypatch.setattr(node.p2p_store, 'try_lock', lambda conn: results.pop(0))
        monkeypatch.setattr(node.time, 'sleep', sleeps.append)

        held = node.acquire_lock()

        assert held is conns[-1] and not held.closed
        assert all(conn.closed for conn in conns[:-1])
        assert sleeps == [node.LOCK_RETRY_SECONDS] * 2


class TestStatusRow:
    def test_an_idle_status_names_its_reason_and_closes_its_connection(self, monkeypatch):
        conn = _Conn()
        written = []
        monkeypatch.setattr(node.p2p_store, 'connect', lambda read_only=False: conn)
        monkeypatch.setattr(node.p2p_store, 'write_status_on', lambda target, status: written.append((target, status)))

        node.write_status(ipc.STATE_UNAVAILABLE, REASON)

        assert written == [(conn, {'state': ipc.STATE_UNAVAILABLE, 'reason': REASON, 'peer_id': ''})]
        assert conn.closed


class TestImportRules:
    def _tree(self):
        with open(node.__file__, 'r', encoding='utf-8') as handle:
            return ast.parse(handle.read())

    def test_the_runtime_is_the_only_lazy_import(self):
        lazy = []
        for function in ast.walk(self._tree()):
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(function):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        lazy.append((child.module if isinstance(child, ast.ImportFrom) else None,
                                     [alias.name for alias in child.names]))

        assert lazy == [('p2p', ['runtime'])]

    def test_importing_the_node_loads_neither_the_runtime_nor_its_libraries(self):
        code = (
            "import sys\n"
            "import p2p.node\n"
            "print(*[name in sys.modules for name in ('p2p.runtime', 'p2p.tor', 'stem', 'trio', 'coincurve')])\n"
        )
        repo = os.path.dirname(os.path.dirname(os.path.abspath(node.__file__)))
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=repo,
                                timeout=120, env=dict(os.environ, SERVICE_TYPE='worker'))

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == 'False False False False False'


class _Cursor:
    def __init__(self, stored):
        self.stored = stored
        self.executed = []
        self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if sql.startswith('INSERT') and self.stored is None:
            self.stored = params[1]

    def fetchone(self):
        return None if self.stored is None else (self.stored,)


class _StoreConn:
    def __init__(self, stored=None):
        self.cur = _Cursor(stored)

    def cursor(self):
        return self.cur


class TestOnionKey:
    def test_the_row_text_is_the_key_type_and_the_base64_key(self):
        text = p2p_store.onion_key_text(ONION_KEY)

        assert text == 'ED25519-V3:' + ONION_KEY
        assert p2p_store.parse_onion_key(text) == ONION_KEY

    @pytest.mark.parametrize('text', [
        None, '', 42, ONION_KEY, 'RSA1024:' + ONION_KEY, 'ed25519-v3:' + ONION_KEY, 'ED25519-V3:',
        'ED25519-V3:not base64!', 'ED25519-V3:' + base64.b64encode(bytes(63)).decode('ascii'),
        'ED25519-V3:' + base64.b64encode(bytes(65)).decode('ascii'), 'ED25519-V3: ' + ONION_KEY,
    ])
    def test_anything_else_is_refused(self, text):
        with pytest.raises(ValueError):
            p2p_store.parse_onion_key(text)

    def test_the_key_type_and_size_are_tor_ed25519_v3(self):
        assert (p2p_store.ONION_KEY_TYPE, p2p_store.ONION_KEY_BYTES) == ('ED25519-V3', 64)
        assert ipc.ONION_KEY_KEY == 'p2p_onion_key'

    def test_a_new_key_is_inserted_once_and_never_replaced(self):
        conn = _StoreConn()

        saved = p2p_store.Store(conn).save_onion_key(ONION_KEY)

        assert saved == ONION_KEY
        insert_sql, insert_params = conn.cur.executed[0]
        assert 'ON CONFLICT (key) DO NOTHING' in insert_sql
        assert insert_params == (ipc.ONION_KEY_KEY, 'ED25519-V3:' + ONION_KEY)
        assert not any('UPDATE' in sql for sql, _params in conn.cur.executed)

    def test_the_stored_key_wins_over_a_second_new_key(self):
        conn = _StoreConn(stored=p2p_store.onion_key_text(ONION_KEY))

        assert p2p_store.Store(conn).save_onion_key(OTHER_KEY) == ONION_KEY

    def test_the_stored_key_is_read_and_checked(self, monkeypatch):
        rows = {ipc.ONION_KEY_KEY: p2p_store.onion_key_text(ONION_KEY)}
        monkeypatch.setattr(p2p_store.database, 'get_app_config_value', lambda key, conn=None: rows.get(key))
        store = p2p_store.Store(_StoreConn())

        assert store.onion_key() == ONION_KEY
        rows[ipc.ONION_KEY_KEY] = 'ED25519-V3:broken'
        with pytest.raises(ValueError):
            store.onion_key()
        rows.pop(ipc.ONION_KEY_KEY)
        assert store.onion_key() is None


class TestPeerCache:
    def _store(self, monkeypatch, rows):
        monkeypatch.setattr(p2p_store.database, 'get_app_config_value', lambda key, conn=None: rows.get(key))
        monkeypatch.setattr(p2p_store.database, 'set_app_config_value',
                            lambda key, value, conn=None: rows.__setitem__(key, value))
        return p2p_store.Store(_StoreConn())

    def test_the_saved_list_reads_back(self, monkeypatch):
        rows = {}
        store = self._store(monkeypatch, rows)
        data = {'v': 1, 'peers': {'a' * 56: {'heard': 100, 'ok': None}}}

        store.save_peer_cache(data)

        assert json.loads(rows[ipc.PEERS_KEY]) == data
        assert store.peer_cache() == data

    @pytest.mark.parametrize('raw', [None, '', '[1, 2]', '"text"'])
    def test_a_missing_or_foreign_row_reads_as_no_list(self, monkeypatch, raw):
        store = self._store(monkeypatch, {ipc.PEERS_KEY: raw})

        assert store.peer_cache() == {}

    def test_a_broken_row_is_logged_and_reads_as_no_list(self, monkeypatch, caplog):
        store = self._store(monkeypatch, {ipc.PEERS_KEY: '{not json'})

        with caplog.at_level(logging.ERROR, logger='p2p.store'):
            assert store.peer_cache() == {}

        assert ipc.PEERS_KEY in caplog.text
