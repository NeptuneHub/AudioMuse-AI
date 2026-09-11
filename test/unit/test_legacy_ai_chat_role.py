# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The retired AI chat database role: never created, disabled once on legacy installs.

Main Features:
* No role, a role that already cannot log in, or a role that is the main
  database user itself: no login probe and no ALTER
* A role that still accepts the shipped default password is set NOLOGIN
* A role the probe cannot log into is left alone and a warning carries the
  reason from the server plus the manual command
* An ALTER refused for lack of privilege becomes one actionable warning, not a
  traceback that repeats on every boot
* The probe only ever uses the retired name and password over the main URL
* Flask startup runs the check exactly once, after the config bootstrap, inside
  the non-worker block
* config.py and the setup wizard no longer expose the two retired parameters
"""

import logging
import os
from unittest.mock import MagicMock, patch

import psycopg2
from psycopg2 import sql

import app_setup
import config
import database

_RETIRED_NAME = 'ai_user'
_RETIRED_PASSWORD = 'ChangeThisSecurePassword123!'
_REJECTED = psycopg2.OperationalError('FATAL:  password authentication failed for user "ai_user"')


def _admin(row):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = row
    return conn, cur


def _run(row, **probe):
    conn, cur = _admin(row)
    with patch.object(database, 'connect_raw', return_value=conn):
        with patch.object(database.psycopg2, 'connect', **probe) as connect:
            result = database.disable_legacy_ai_chat_role()
    return result, cur, connect, conn


def test_no_legacy_role_means_no_probe_and_no_alter():
    result, cur, connect, conn = _run(None, return_value=MagicMock())
    assert result is False
    connect.assert_not_called()
    assert cur.execute.call_count == 1
    conn.close.assert_called_once()


def test_a_legacy_role_that_cannot_log_in_is_not_probed():
    result, cur, connect, _conn = _run((False, False), return_value=MagicMock())
    assert result is False
    connect.assert_not_called()
    assert cur.execute.call_count == 1


def test_the_main_database_user_is_never_probed_or_altered():
    result, cur, connect, _conn = _run((True, True), return_value=MagicMock())
    assert result is False
    connect.assert_not_called()
    assert cur.execute.call_count == 1


def test_a_legacy_role_with_the_default_password_is_altered_to_nologin():
    probe_conn = MagicMock()
    result, cur, connect, conn = _run((True, False), return_value=probe_conn)
    assert result is True
    connect.assert_called_once()
    assert cur.execute.call_count == 2
    executed = cur.execute.call_args_list[1].args[0]
    assert executed == sql.SQL("ALTER ROLE {} NOLOGIN").format(sql.Identifier(_RETIRED_NAME))
    assert conn.autocommit is True
    probe_conn.close.assert_called_once()
    conn.close.assert_called_once()


def test_a_legacy_role_the_probe_cannot_log_into_is_left_alone_with_the_reason_in_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger='database'):
        result, cur, connect, conn = _run((True, False), side_effect=_REJECTED)
    assert result is False
    connect.assert_called_once()
    assert cur.execute.call_count == 1
    conn.close.assert_called_once()
    assert 'left untouched' in caplog.text
    assert 'password authentication failed' in caplog.text
    assert 'ALTER ROLE ai_user NOLOGIN' in caplog.text


def test_an_alter_refused_for_lack_of_privilege_is_one_actionable_warning(caplog):
    conn, cur = _admin((True, False))
    cur.execute.side_effect = [None, psycopg2.errors.InsufficientPrivilege('permission denied')]
    with caplog.at_level(logging.WARNING, logger='database'):
        with patch.object(database, 'connect_raw', return_value=conn):
            with patch.object(database.psycopg2, 'connect', return_value=MagicMock()):
                assert database.disable_legacy_ai_chat_role() is False
    assert 'run as a superuser: ALTER ROLE ai_user NOLOGIN' in caplog.text
    conn.close.assert_called_once()


def test_the_probe_uses_only_the_retired_name_and_password_over_the_main_url():
    _result, _cur, connect, _conn = _run((True, False), return_value=MagicMock())
    connect.assert_called_once_with(
        config.DATABASE_URL,
        user=_RETIRED_NAME,
        password=_RETIRED_PASSWORD,
        connect_timeout=10,
    )
    assert database._LEGACY_AI_CHAT_ROLE == _RETIRED_NAME
    assert database._LEGACY_AI_CHAT_DEFAULT_PASSWORD == _RETIRED_PASSWORD


def test_a_failing_admin_connection_is_still_closed():
    conn, cur = _admin((True, False))
    cur.execute.side_effect = psycopg2.OperationalError('server closed the connection')
    with patch.object(database, 'connect_raw', return_value=conn):
        with patch.object(database.psycopg2, 'connect') as connect:
            try:
                database.disable_legacy_ai_chat_role()
            except psycopg2.OperationalError:
                pass
            else:
                raise AssertionError('the admin failure must surface to the caller')
    connect.assert_not_called()
    conn.close.assert_called_once()


def test_flask_startup_runs_the_legacy_check_exactly_once_after_the_config_bootstrap():
    app_path = os.path.join(os.path.dirname(os.path.abspath(config.__file__)), 'app.py')
    with open(app_path, encoding='utf-8') as handle:
        source = handle.read()
    assert 'coerce_db_details, disable_legacy_ai_chat_role' in source
    assert source.count('disable_legacy_ai_chat_role()') == 1
    worker_gate = source.index('if not _is_worker:')
    bootstrap = source.index('setup_manager.persist_missing_config_values(config)', worker_gate)
    call = source.index('if disable_legacy_ai_chat_role():', bootstrap)
    assert worker_gate < bootstrap < call


def test_config_and_the_wizard_no_longer_know_the_ai_chat_db_user():
    for name in ('AI_CHAT_DB_USER_NAME', 'AI_CHAT_DB_USER_PASSWORD'):
        assert not hasattr(config, name)
        assert name not in app_setup.SECRET_FIELDS
        assert name not in app_setup.HIDDEN_ADVANCED_FIELDS


def test_config_source_no_longer_reads_the_retired_parameters():
    config_path = os.path.join(os.path.dirname(os.path.abspath(config.__file__)), 'config.py')
    with open(config_path, encoding='utf-8') as handle:
        source = handle.read()
    assert 'AI_CHAT_DB_USER' not in source
    assert _RETIRED_PASSWORD not in source
