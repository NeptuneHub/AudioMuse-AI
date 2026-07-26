# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Guards how a database connection lost mid-run is handled.

A worker job holds a single Flask app context for its whole run, so the
connection cached in ``g`` outlives any database restart or idle disconnect that
happens during it; handing that dead handle back turned every later write into
"connection already closed" until the job ended. Silently swapping in a fresh
connection instead is worse: a helper called with ``conn=None`` would commit its
half on the new handle while the caller's uncommitted statements stayed lost. So
the dead handle is dropped AND the in-flight unit of work is failed, leaving the
next call free to open a clean connection.

Main Features:
* An open cached connection is still reused (no reconnect per call)
* A server-side drop (closed == 2) fails loudly once, then the context recovers
* The drop raises ``ConnectionLostError``, an ``OperationalError`` subclass, so
  every task-level ``except OperationalError`` carve-out catches it
* ``save_task_status`` absorbs the one-shot drop and lands the row on a fresh
  connection, so a terminal FAILURE write is never lost to the poisoned context
* A locally closed connection is a deliberate boundary and is just replaced
* Any other truthy ``closed`` (a test double, say) is replaced, never raised on
* ``close_db`` keeps closing and clearing the cached connection
"""

from unittest.mock import MagicMock

import pytest
from flask import Flask


class FakeConnection:
    def __init__(self):
        self.closed = 0
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        self.closed = 1

    def cursor(self, *args, **kwargs):
        return MagicMock()

    def commit(self):
        pass

    def rollback(self):
        pass


def _patch_connect(monkeypatch):
    import database

    created = []

    def fake_connect(*args, **kwargs):
        conn = FakeConnection()
        created.append(conn)
        return conn

    monkeypatch.setattr(database.psycopg2, 'connect', fake_connect)
    return created


def test_open_cached_connection_is_reused(monkeypatch):
    import database

    created = _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        first = database.get_db()
        assert database.get_db() is first
    assert len(created) == 1


def test_the_drop_error_is_caught_by_every_operational_error_carveout():
    import database
    import psycopg2

    assert issubclass(database.ConnectionLostError, psycopg2.OperationalError)
    assert not issubclass(database.ConnectionLostError, psycopg2.InterfaceError)


def test_a_server_drop_fails_the_unit_of_work_once_then_the_context_recovers(monkeypatch):
    import database
    import psycopg2

    created = _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        first = database.get_db()
        first.closed = 2
        with pytest.raises(psycopg2.OperationalError):
            database.get_db()
        assert len(created) == 1
        second = database.get_db()
        assert second is not first
        assert second is created[-1]
        assert database.get_db() is second
    assert len(created) == 2


def test_connection_closed_locally_is_replaced(monkeypatch):
    import database

    _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        first = database.get_db()
        first.close()
        assert database.get_db() is not first


def test_only_the_server_drop_code_fails_the_unit_of_work(monkeypatch):
    import database

    created = _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        first = database.get_db()
        first.closed = MagicMock()
        second = database.get_db()
        assert second is not first
    assert len(created) == 2


def test_save_task_status_lands_on_a_fresh_connection_after_a_server_side_drop(monkeypatch):
    import database

    created = _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        first = database.get_db()
        first.closed = 2
        database.save_task_status('task-1', 'test_task')
    assert len(created) == 2


def test_save_task_status_still_fails_when_the_database_is_really_down(monkeypatch):
    import database
    import psycopg2

    def refuse(*args, **kwargs):
        raise psycopg2.OperationalError('connection refused')

    monkeypatch.setattr(database.psycopg2, 'connect', refuse)
    with Flask(__name__).app_context():
        with pytest.raises(psycopg2.OperationalError):
            database.save_task_status('task-1', 'test_task')


def test_close_db_closes_and_clears_the_cached_connection(monkeypatch):
    import database
    from flask import g

    _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        conn = database.get_db()
        database.close_db()
        assert conn.close_calls == 1
        assert 'db' not in g
