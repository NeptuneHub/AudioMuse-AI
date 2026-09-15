# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""AI chat reads the main database through a read-only session; the retired
ai_user role is never created and is disabled once on legacy installs.

Main Features:
* a fresh install never gains an ai_user role, neither from the chat
  connection nor from the legacy check
* the chat connection logs in as the main database user through the standard
  helper: tagged audiomuse-ai-chat, serial plans, the statement timeout
* the chat connection refuses writes and still reads, in transaction and
  autocommit mode alike (the read-only default is a server-side session option)
* a legacy ai_user that still accepts the shipped default password is set
  NOLOGIN once; the second run is a no-op
* a legacy ai_user that is already NOLOGIN is left alone
* the check never disables the database user it runs as, proven on a throwaway
  superuser that carries the default password: a broken guard could only ever
  disable that throwaway, never the real login of the shared server
* a legacy ai_user with a custom password keeps its login on a server that
  checks passwords (skipped on trust authentication, where any password logs
  in and the role is disabled by design)
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:
    psycopg2 = None

pytestmark = pytest.mark.integration

_ROLE = 'ai_user'
_GUARD_ROLE = 'ai_chat_guard_probe'
_TABLE = 'ai_chat_read_only_probe'


def _default_password():
    import database

    return database._LEGACY_AI_CHAT_DEFAULT_PASSWORD


def _password_is_checked(dsn):
    try:
        psycopg2.connect(dsn, user=_ROLE, password='wrong-on-purpose').close()
    except psycopg2.OperationalError:
        return True
    return False


def _role_can_login(admin, name):
    with admin.cursor() as cur:
        cur.execute("SELECT rolcanlogin FROM pg_roles WHERE rolname = %s", (name,))
        row = cur.fetchone()
    return None if row is None else row[0]


def _main_user(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT current_user")
        return cur.fetchone()[0]


@pytest.fixture
def chat_db(shared_pg_dsn, monkeypatch):
    import config

    admin = psycopg2.connect(shared_pg_dsn)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("DROP ROLE IF EXISTS ai_user")
        cur.execute("DROP ROLE IF EXISTS ai_chat_guard_probe")
        cur.execute("DROP TABLE IF EXISTS ai_chat_read_only_probe")
        cur.execute("CREATE TABLE ai_chat_read_only_probe (x INTEGER)")
        cur.execute("INSERT INTO ai_chat_read_only_probe VALUES (1)")
    monkeypatch.setattr(config, 'DATABASE_URL', shared_pg_dsn)
    yield admin
    with admin.cursor() as cur:
        cur.execute("DROP ROLE IF EXISTS ai_user")
        cur.execute("DROP ROLE IF EXISTS ai_chat_guard_probe")
        cur.execute("DROP TABLE IF EXISTS ai_chat_read_only_probe")
    admin.close()


def test_a_fresh_install_never_creates_the_ai_chat_role(chat_db):
    import database
    from tasks.mcp_helper import get_db_connection

    get_db_connection().close()
    assert database.disable_legacy_ai_chat_role() is False
    assert _role_can_login(chat_db, _ROLE) is None


def test_the_chat_connection_is_the_main_user_through_the_standard_helper(chat_db):
    from tasks.mcp_helper import get_db_connection

    conn = get_db_connection()
    try:
        assert _main_user(conn) == _main_user(chat_db)
        with conn.cursor() as cur:
            cur.execute("SELECT application_name FROM pg_stat_activity WHERE pid = pg_backend_pid()")
            assert cur.fetchone()[0] == 'audiomuse-ai-chat'
            cur.execute("SHOW max_parallel_workers_per_gather")
            assert cur.fetchone()[0] == '0'
            cur.execute("SHOW statement_timeout")
            assert cur.fetchone()[0] == '10min'
    finally:
        conn.close()


def test_the_chat_connection_refuses_writes_and_still_reads(chat_db):
    from tasks.mcp_helper import get_db_connection

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
                cur.execute("INSERT INTO ai_chat_read_only_probe VALUES (2)")
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT x FROM ai_chat_read_only_probe")
            assert cur.fetchall() == [(1,)]
    finally:
        conn.close()
    with chat_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM ai_chat_read_only_probe")
        assert cur.fetchone()[0] == 1


def test_the_chat_connection_refuses_writes_in_autocommit_mode_too(chat_db):
    from tasks.mcp_helper import get_db_connection

    conn = get_db_connection()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SHOW default_transaction_read_only")
            assert cur.fetchone()[0] == 'on'
            with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
                cur.execute("INSERT INTO ai_chat_read_only_probe VALUES (3)")
            cur.execute("SELECT x FROM ai_chat_read_only_probe")
            assert cur.fetchall() == [(1,)]
    finally:
        conn.close()
    with chat_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM ai_chat_read_only_probe")
        assert cur.fetchone()[0] == 1


def test_a_legacy_role_with_the_default_password_is_set_nologin_once(chat_db):
    import database

    with chat_db.cursor() as cur:
        cur.execute("CREATE ROLE ai_user LOGIN PASSWORD %s", (_default_password(),))
    assert _role_can_login(chat_db, _ROLE) is True
    assert database.disable_legacy_ai_chat_role() is True
    assert _role_can_login(chat_db, _ROLE) is False
    assert database.disable_legacy_ai_chat_role() is False
    assert _role_can_login(chat_db, _ROLE) is False


def test_a_legacy_role_already_nologin_is_left_alone(chat_db):
    import database

    with chat_db.cursor() as cur:
        cur.execute("CREATE ROLE ai_user NOLOGIN")
    assert database.disable_legacy_ai_chat_role() is False
    assert _role_can_login(chat_db, _ROLE) is False


def test_the_check_never_disables_the_database_user_it_runs_as(chat_db, monkeypatch, shared_pg_dsn):
    import config
    import database
    from psycopg2.extensions import make_dsn

    with chat_db.cursor() as cur:
        try:
            cur.execute(
                "CREATE ROLE ai_chat_guard_probe LOGIN SUPERUSER PASSWORD %s", (_default_password(),)
            )
        except psycopg2.errors.InsufficientPrivilege:
            pytest.skip("the test database user cannot create a throwaway superuser")
    monkeypatch.setattr(
        config, 'DATABASE_URL', make_dsn(shared_pg_dsn, user=_GUARD_ROLE, password=_default_password())
    )
    monkeypatch.setattr(database, '_LEGACY_AI_CHAT_ROLE', _GUARD_ROLE)
    assert database.disable_legacy_ai_chat_role() is False
    assert _role_can_login(chat_db, _GUARD_ROLE) is True


def test_a_legacy_role_with_a_custom_password_is_left_alone(chat_db, shared_pg_dsn):
    import database

    with chat_db.cursor() as cur:
        cur.execute("CREATE ROLE ai_user LOGIN PASSWORD %s", ('not-the-shipped-default',))
    if not _password_is_checked(shared_pg_dsn):
        pytest.skip("trust authentication: this server never checks a password")
    assert database.disable_legacy_ai_chat_role() is False
    assert _role_can_login(chat_db, _ROLE) is True
