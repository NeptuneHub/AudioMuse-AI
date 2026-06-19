"""Shared fixtures and helpers for AudioMuse-AI test suite.

Centralises duplicated helpers across test files:
- importlib bypass loader (avoids tasks/__init__.py -> pydub -> audioop chain)
- Session-scoped module fixtures for mcp_server, ai_mcp_client
- FakeRow / mock-connection helpers
- Autouse config restoration fixture
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


# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

def _import_module(mod_name: str, relative_path: str):
    """Load a module directly by file path, bypassing package __init__.py.

    Args:
        mod_name: Dotted module name to register in sys.modules
                  (e.g. 'tasks.mcp_helper').
        relative_path: Path relative to the repo root
                       (e.g. 'tasks/mcp_helper.py').
    """
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


# ---------------------------------------------------------------------------
# Session-scoped module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def mcp_server_mod():
    """Load tasks.mcp_helper directly (session-scoped).

    Fixture name kept as ``mcp_server_mod`` for historical reasons; the
    underlying module is now ``tasks.mcp_helper``.
    """
    return _import_module('tasks.mcp_helper', 'tasks/mcp_helper.py')


# ---------------------------------------------------------------------------
# DB mock helpers
# ---------------------------------------------------------------------------

def make_dict_row(mapping: dict):
    """Create an object that supports both dict-key and attribute access,
    mimicking psycopg2 DictRow."""
    class FakeRow(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
    return FakeRow(mapping)


def make_mock_connection(cursor):
    """Wrap a mock cursor in a mock connection with close()."""
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.close = Mock()
    return conn


# ---------------------------------------------------------------------------
# Config restoration (autouse)
# ---------------------------------------------------------------------------

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
    """Save and restore mutated config attributes after each test."""
    import config as cfg
    saved = {}
    for attr in _CONFIG_ATTRS_TO_RESTORE:
        if hasattr(cfg, attr):
            saved[attr] = getattr(cfg, attr)
    yield
    for attr, val in saved.items():
        setattr(cfg, attr, val)


# ---------------------------------------------------------------------------
# Import-architecture report (terminal summary)
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the import-architecture report (layer table, max-chain confirmation,
    and the recap of chains at the ceiling) whenever the architecture gate ran.

    Uses the terminal reporter so the report shows on every run -- pass or fail --
    without needing ``-s``. A PR that deepens the eager import graph will see the
    new chains listed here (and the depth test will fail with the same recap).
    """
    # The gate may be imported bare ("test_import_architecture") or
    # package-qualified ("test.unit.test_import_architecture") depending on the
    # presence of __init__.py files, so match by suffix.
    mod = next(
        (m for name, m in list(sys.modules.items())
         if name == "test_import_architecture" or name.endswith(".test_import_architecture")),
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
    except Exception as exc:  # never let the report break the run
        terminalreporter.write_line(f"[architecture] report unavailable: {exc}")
        return
    terminalreporter.section("Import-architecture report", "=")
    for line in lines:
        terminalreporter.write_line(line)
