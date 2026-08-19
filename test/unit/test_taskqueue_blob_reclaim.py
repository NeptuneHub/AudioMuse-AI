# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Autovacuum counts ROWS, so a table of a few huge blobs never qualifies.

Its threshold is 50 + 20% of the LIVE row count. ivf_dir carries eight rows,
150-odd MB and the hyperbolic tree, so it needs ~50 dead rows - about seven full
index rebuilds - before autovacuum touches it, and sits at several times its
useful size until then. This sweep keeps that plateau low.

It must never collide with anything. Plain VACUUM only, which takes SHARE UPDATE
EXCLUSIVE and so blocks no reader and no writer. It stands down while a task is
live, and while any other session holds an old snapshot - which is exactly what a
backup's pg_dump looks like, and during which VACUUM could reclaim nothing
anyway. A lock_timeout means it can never queue behind another lock. It runs in
the WEB process, which a restore stops before psql replaces the database, and as
a daemon thread a setup-wizard restart kills it mid-VACUUM harmlessly.

It sweeps hourly, not every few minutes: dead rows only appear when a rebuild or
analysis replaces a blob, and a swept table drops to zero dead rows and leaves
the set until it dirties again. The session caps are re-applied every pass rather
than once, because the loop rebuilds its connection after any error and a fresh
session would otherwise run uncapped.

Main Features:
* Plain VACUUM only - never FULL, so readers and writers are never blocked
* Stands down for a live task, and for a backup's long-lived snapshot
* lock_timeout keeps it from queueing; statement_timeout is generous enough to
  finish the biggest table on slow storage rather than starving it every pass
* One table's failure never blocks the rest, and none escapes the cycle
* The worker cycle never reclaims; the web process schedules it, on its own
  connection, where a restore has already stopped Flask
"""

import psycopg2
import pytest

import config
from taskqueue import maintenance, sql

_ROWS = [
    ('public.ivf_dir', 6, '153 MB'),
    ('public.artist_metadata_data', 1, '31 MB'),
]


class _Cursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        flat = ' '.join(statement.split())
        self._conn.statements.append(flat)
        self._conn.autocommit_at_execute.append(self._conn.autocommit)
        if flat.startswith('VACUUM'):
            target = flat.split(' ', 1)[1]
            if target in self._conn.busy:
                raise psycopg2.errors.LockNotAvailable('lock timeout')
            if target in self._conn.failing:
                raise RuntimeError('permission denied: ' + target)

    def fetchone(self):
        return ('x',) if self._conn.sentinel else None

    def fetchall(self):
        return list(self._conn.rows)

    def close(self):
        pass


class _Conn:
    def __init__(self, rows=()):
        self.autocommit = False
        self.rows = list(rows)
        self.statements = []
        self.autocommit_at_execute = []
        self.busy = set()
        self.failing = set()
        self.sentinel = False

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass


def _quiet(monkeypatch, conn, live=False, snapshot=False):
    monkeypatch.setattr(sql, 'begin_reclaim_session', lambda cur: None)
    monkeypatch.setattr(sql, 'any_live_task', lambda cur: live)
    monkeypatch.setattr(sql, 'snapshot_holder_blocking_reclaim', lambda cur: snapshot)
    monkeypatch.setattr(
        sql, 'blob_tables_autovacuum_cannot_reach',
        lambda cur, min_bytes=None: list(conn.rows),
    )


def _vacuums(conn):
    return [s for s in conn.statements if s.startswith('VACUUM')]


class TestNeverBlocksAnyone:
    def test_it_only_ever_runs_a_plain_vacuum_never_full(self, monkeypatch):
        conn = _Conn(_ROWS)
        _quiet(monkeypatch, conn)
        maintenance.reclaim_blob_space(conn)
        assert _vacuums(conn) == ['VACUUM public.ivf_dir', 'VACUUM public.artist_metadata_data']
        assert not any('FULL' in s for s in conn.statements)

    def test_a_busy_table_is_skipped_rather_than_queued_behind(self, monkeypatch):
        conn = _Conn(_ROWS)
        conn.busy = {'public.ivf_dir'}
        _quiet(monkeypatch, conn)
        assert maintenance.reclaim_blob_space(conn) == ['public.artist_metadata_data']

    def test_the_session_caps_how_long_it_waits_for_a_lock(self):
        import inspect

        src = inspect.getsource(sql.begin_reclaim_session)
        assert 'lock_timeout' in src
        assert 'statement_timeout' in src

    def test_the_lock_timeout_is_short_enough_to_never_stall_a_restore(self):
        assert config.BLOB_RECLAIM_LOCK_TIMEOUT.endswith('s')
        assert int(config.BLOB_RECLAIM_LOCK_TIMEOUT.rstrip('s')) <= 5

    def test_the_statement_timeout_is_generous_enough_to_finish_a_big_table(self):
        assert config.BLOB_RECLAIM_STATEMENT_TIMEOUT == '10min'

    def test_the_session_caps_are_reapplied_every_pass_because_reconnects_reset_them(
        self, monkeypatch
    ):
        conn = _Conn(_ROWS)
        applied = []
        monkeypatch.setattr(sql, 'begin_reclaim_session', lambda cur: applied.append(cur))
        monkeypatch.setattr(sql, 'any_live_task', lambda cur: False)
        monkeypatch.setattr(sql, 'snapshot_holder_blocking_reclaim', lambda cur: False)
        monkeypatch.setattr(
            sql, 'blob_tables_autovacuum_cannot_reach',
            lambda cur, min_bytes=None: [],
        )
        maintenance.reclaim_blob_space(conn)
        maintenance.reclaim_blob_space(conn)
        assert len(applied) == 2


class TestStandsDownForOtherWork:
    def test_a_live_task_defers_the_whole_sweep(self, monkeypatch):
        conn = _Conn(_ROWS)
        _quiet(monkeypatch, conn, live=True)
        assert maintenance.reclaim_blob_space(conn) == []
        assert _vacuums(conn) == []

    def test_a_backups_long_snapshot_defers_the_whole_sweep(self, monkeypatch):
        conn = _Conn(_ROWS)
        _quiet(monkeypatch, conn, snapshot=True)
        assert maintenance.reclaim_blob_space(conn) == []
        assert _vacuums(conn) == []

    def test_the_snapshot_probe_ignores_this_sessions_own_backend(self):
        assert 'pid <> pg_backend_pid()' in ' '.join(sql._OLD_SNAPSHOT_HOLDER.split())

    def test_the_snapshot_probe_keys_on_the_xmin_horizon_not_transaction_state(self):
        flat = ' '.join(sql._OLD_SNAPSHOT_HOLDER.split())
        assert 'backend_xmin IS NOT NULL' in flat
        assert "state IN ('active', 'idle in transaction')" not in flat


class TestSweepMechanics:
    def test_every_vacuum_runs_with_autocommit_on(self, monkeypatch):
        conn = _Conn(_ROWS)
        _quiet(monkeypatch, conn)
        maintenance.reclaim_blob_space(conn)
        during = [
            on
            for stmt, on in zip(conn.statements, conn.autocommit_at_execute)
            if stmt.startswith('VACUUM')
        ]
        assert during == [True, True]

    def test_autocommit_is_restored_afterwards(self, monkeypatch):
        conn = _Conn(_ROWS)
        _quiet(monkeypatch, conn)
        maintenance.reclaim_blob_space(conn)
        assert conn.autocommit is False

    def test_autocommit_is_restored_when_a_vacuum_raises(self, monkeypatch):
        conn = _Conn(_ROWS)
        conn.failing = {name for name, _d, _t in _ROWS}
        _quiet(monkeypatch, conn)
        assert maintenance.reclaim_blob_space(conn) == []
        assert conn.autocommit is False

    def test_one_failing_table_never_blocks_the_others(self, monkeypatch):
        conn = _Conn(_ROWS)
        conn.failing = {'public.ivf_dir'}
        _quiet(monkeypatch, conn)
        assert maintenance.reclaim_blob_space(conn) == ['public.artist_metadata_data']

    def test_nothing_dead_runs_no_vacuum(self, monkeypatch):
        conn = _Conn([])
        _quiet(monkeypatch, conn)
        assert maintenance.reclaim_blob_space(conn) == []
        assert _vacuums(conn) == []


class TestAlwaysLogs:
    def test_a_pass_with_nothing_to_do_still_logs(self, monkeypatch, caplog):
        conn = _Conn([])
        _quiet(monkeypatch, conn)
        with caplog.at_level('INFO', logger='taskqueue.maintenance'):
            maintenance.reclaim_blob_space(conn)
        assert any('found no tables' in record.message for record in caplog.records)

    def test_a_pass_that_cleaned_tables_logs_a_summary(self, monkeypatch, caplog):
        conn = _Conn(_ROWS)
        _quiet(monkeypatch, conn)
        with caplog.at_level('INFO', logger='taskqueue.maintenance'):
            maintenance.reclaim_blob_space(conn)
        assert any('vacuumed 2 table(s)' in record.message for record in caplog.records)


class TestDetectionQuery:
    def test_threshold_mirrors_postgres_own_autovacuum_settings(self):
        query = sql._BLOB_TABLES_AUTOVACUUM_CANNOT_REACH
        assert 'autovacuum_vacuum_threshold' in query
        assert 'autovacuum_vacuum_scale_factor' in query

    def test_only_tables_that_already_have_dead_rows_are_swept(self):
        assert 'n_dead_tup > 0' in ' '.join(
            sql._BLOB_TABLES_AUTOVACUUM_CANNOT_REACH.split()
        )

    def test_a_big_table_is_swept_however_many_live_rows_it_carries(self):
        assert 'n_live_tup <' not in ' '.join(
            sql._BLOB_TABLES_AUTOVACUUM_CANNOT_REACH.split()
        )

    def test_target_is_schema_qualified_so_search_path_cannot_redirect_it(self):
        flat = ' '.join(sql._BLOB_TABLES_AUTOVACUUM_CANNOT_REACH.split())
        assert "quote_ident(s.schemaname) || '.' || quote_ident(s.relname)" in flat

    def test_trivially_small_tables_stay_with_autovacuum(self):
        assert config.BLOB_RECLAIM_MIN_BYTES >= 1024 * 1024

    def test_the_sweep_does_the_small_blob_tables_first(self):
        flat = ' '.join(sql._BLOB_TABLES_AUTOVACUUM_CANNOT_REACH.split())
        assert 'ORDER BY pg_total_relation_size(s.relid) ASC' in flat

    def test_bytea_tables_are_swept_however_small(self):
        flat = ' '.join(sql._BLOB_TABLES_AUTOVACUUM_CANNOT_REACH.split())
        assert "'bytea'::regtype" in flat


class _FakeApp:
    def __init__(self):
        import logging
        self.logger = logging.getLogger('test_blob_reclaim_app')


class _BlobConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class TestWhereItRuns:
    def test_the_worker_retention_cycle_never_reclaims(self, monkeypatch):
        calls = []
        monkeypatch.setattr(maintenance, 'reclaim_blob_space', calls.append)
        monkeypatch.setattr(sql, 'try_maintenance_lock', lambda cur: True)
        monkeypatch.setattr(sql, 'release_maintenance_lock', lambda cur: None)
        monkeypatch.setattr(maintenance, 'reclaim_orphans', lambda conn: None)
        monkeypatch.setattr(maintenance, 'fail_stale_inline_rows', lambda conn: None)
        monkeypatch.setattr(maintenance, 'recover_migration_handshakes', lambda: None)
        monkeypatch.setattr(maintenance, 'clear_terminal_shared_payloads', lambda conn: None)
        maintenance.run_cycle(_Conn(), with_retention=True)
        assert calls == []

    def test_the_reclaim_loop_uses_its_own_connection_never_the_app_context_one(self):
        conn = _BlobConn()
        sweeps = []

        def _reclaim(c):
            sweeps.append(c)
            raise SystemExit

        with pytest.raises(SystemExit):
            maintenance._blob_reclaim_loop(
                _FakeApp(), connect_raw=lambda **kw: conn, sleep=lambda s: None,
                reclaim=_reclaim,
            )
        assert sweeps == [conn]

    def test_the_reclaim_loop_sweeps_on_the_config_interval(self):
        conn = _BlobConn()
        sweeps = []
        sleeps = []

        def _reclaim(c):
            sweeps.append(c)

        def _sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) > 1:
                raise SystemExit

        with pytest.raises(SystemExit):
            maintenance._blob_reclaim_loop(
                _FakeApp(), connect_raw=lambda **kw: conn, sleep=_sleep,
                reclaim=_reclaim,
            )
        assert sleeps == [
            config.BLOB_RECLAIM_STARTUP_DELAY_SECONDS,
            config.BLOB_RECLAIM_INTERVAL_SECONDS,
        ]
        assert sweeps == [conn]

    def test_an_exception_in_a_pass_reconnects_for_the_next_one(self):
        first, second = _BlobConn(), _BlobConn()
        conns = [first, second]
        sweeps = []
        sleeps = []

        def _connect(**kw):
            return conns.pop(0)

        def _reclaim(c):
            sweeps.append(c)
            if len(sweeps) == 1:
                raise RuntimeError('boom')
            raise SystemExit

        def _sleep(seconds):
            sleeps.append(seconds)

        with pytest.raises(SystemExit):
            maintenance._blob_reclaim_loop(
                _FakeApp(), connect_raw=_connect, sleep=_sleep, reclaim=_reclaim,
            )
        assert sweeps == [first, second]
        assert first.closed is True
        assert second.closed is False

    def test_the_web_process_starts_it_as_a_daemon_thread(self, monkeypatch):
        captured = {}

        class _FakeThread:
            def __init__(self, *args, **kwargs):
                captured['kwargs'] = kwargs
                self.started = False

            def start(self):
                self.started = True

        monkeypatch.setattr(maintenance.threading, 'Thread', _FakeThread)
        thread = maintenance.start_blob_reclaim_thread(_FakeApp())
        assert thread.started is True
        assert captured['kwargs'].get('daemon') is True
