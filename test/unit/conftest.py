# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared pytest fixtures and helpers for the unit-test suite.

Provides file-path module loaders, fake DB rows and connections, an
autouse fixture that snapshots and restores mutated config attributes,
and a terminal-summary hook that prints the import-architecture report.

Main Features:
* Forces spawn over fork on Windows and loads modules by file path.
* Restores config values touched by tests and reports import architecture.
"""

import sys as _sys

if _sys.platform == 'win32':
    import multiprocessing as _mp

    _o = _mp.get_context
    _mp.get_context = lambda m=None: _o('spawn') if m == 'fork' else _o(m)

import os
import sys
import importlib.util
import pytest
from unittest.mock import Mock, MagicMock


def _import_module(mod_name: str, relative_path: str):
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.normpath(os.path.join(repo_root, relative_path))

    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


@pytest.fixture(scope='session')
def mcp_server_mod():
    return _import_module('tasks.mcp_helper', 'tasks/mcp_helper.py')


def make_dict_row(mapping: dict):
    class FakeRow(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name) from None

    return FakeRow(mapping)


def make_mock_connection(cursor):
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.close = Mock()
    return conn


_CONFIG_ATTRS_TO_RESTORE = (
    'ENERGY_MIN',
    'ENERGY_MAX',
    'MAX_SONGS_PER_ARTIST_PLAYLIST',
    'PLAYLIST_ENERGY_ARC',
    'CLAP_ENABLED',
    'AI_REQUEST_TIMEOUT_SECONDS',
    'AUTH_ENABLED',
    'API_TOKEN',
    'JWT_SECRET',
    'OLLAMA_SERVER_URL',
    'OPENAI_SERVER_URL',
    'OPENAI_API_KEY',
)


@pytest.fixture(autouse=True)
def config_restore():
    import config as cfg

    saved = {}
    for attr in _CONFIG_ATTRS_TO_RESTORE:
        if hasattr(cfg, attr):
            saved[attr] = getattr(cfg, attr)
    yield
    for attr, val in saved.items():
        setattr(cfg, attr, val)


# The unit suite is hermetic: no test in this directory may open a real
# database connection. This mask was previously an ACCIDENT - a module-level
# patcher in test_memory_cleanup.py started at collection time and was never
# stopped, so dozens of endpoint tests passed only because every stray
# get_db() silently hit a MagicMock, and the integration suite broke whenever
# it shared a process with the units. MODULE scope on purpose: it unwinds when
# each unit module finishes, so nothing is left patched by the time an
# integration test runs in the same process. A test that needs a specific
# connection shape still wins, because monkeypatch/patch apply on top of this.
@pytest.fixture(autouse=True, scope='module')
def _no_real_database_connections():
    from unittest.mock import patch as _patch

    conn = MagicMock()
    conn.cursor.return_value.rowcount = 1
    conn.cursor.return_value.__enter__.return_value = conn.cursor.return_value
    patcher = _patch('psycopg2.connect', return_value=conn)
    patcher.start()
    yield
    patcher.stop()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    mod = next(
        (
            m
            for name, m in sys.modules.items()
            if name == "test_import_architecture" or name.endswith(".test_import_architecture")
        ),
        None,
    )
    if mod is None:
        return
    ran = any(
        "test_import_architecture" in getattr(rep, "nodeid", "")
        for key in ("passed", "failed", "error")
        for rep in terminalreporter.stats.get(key, [])
    )
    if not ran:
        return
    try:
        lines = mod.architecture_report()
    except Exception as exc:
        terminalreporter.write_line(f"[architecture] report unavailable: {exc}")
        return
    terminalreporter.section("Import-architecture report", "=")
    for line in lines:
        terminalreporter.write_line(line)
