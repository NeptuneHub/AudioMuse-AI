"""Shared fixtures and helpers for AudioMuse-AI test suite.

Centralises duplicated helpers across test files:
- importlib bypass loader (avoids tasks/__init__.py -> pydub -> audioop chain)
- Session-scoped module fixtures for mcp_server, ai_mcp_client, mediaserver_localfiles
- FakeRow / mock-connection helpers
- Autouse config restoration fixture
"""
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
                  (e.g. 'tasks.mcp_server').
        relative_path: Path relative to the repo root
                       (e.g. 'tasks/mcp_server.py').
    """
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
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
    """Load tasks.mcp_server directly (session-scoped)."""
    return _import_module('tasks.mcp_server', 'tasks/mcp_server.py')


@pytest.fixture(scope='session')
def ai_mcp_client_mod():
    """Load ai_mcp_client directly (session-scoped)."""
    return _import_module('ai_mcp_client', 'ai_mcp_client.py')


@pytest.fixture(scope='session')
def localfiles_mod():
    """Load tasks.mediaserver_localfiles directly (session-scoped)."""
    return _import_module(
        'tasks.mediaserver_localfiles',
        'tasks/mediaserver_localfiles.py',
    )


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
