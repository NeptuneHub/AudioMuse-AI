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


def test_connection_dropped_by_the_server_fails_the_unit_of_work_that_held_it(monkeypatch):
    import database
    import psycopg2

    created = _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        first = database.get_db()
        first.closed = 2
        with pytest.raises(psycopg2.InterfaceError):
            database.get_db()
    assert len(created) == 1


def test_the_app_context_recovers_on_the_call_after_the_drop_was_reported(monkeypatch):
    import database
    import psycopg2

    created = _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        first = database.get_db()
        first.closed = 2
        with pytest.raises(psycopg2.InterfaceError):
            database.get_db()
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


def test_close_db_closes_and_clears_the_cached_connection(monkeypatch):
    import database
    from flask import g

    _patch_connect(monkeypatch)
    with Flask(__name__).app_context():
        conn = database.get_db()
        database.close_db()
        assert conn.close_calls == 1
        assert 'db' not in g
