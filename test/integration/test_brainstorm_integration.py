# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Grounded brainstorm retrieval test against a real Postgres database.

Seeds the score table and runs the brainstorm recipe filter channel
through tasks.ai.tool_impl to confirm grounding happens inside the tool
against the real library.

Main Features:
* Filter channel surfaces only rows matching the recipe filters.
"""

import importlib.util
import json
import os
import sys
import tempfile
import types
from unittest.mock import Mock, patch

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None


_SCORE_DDL = (
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
    "album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, "
    "mood_vector TEXT, energy REAL, other_features TEXT, year INTEGER, "
    "rating INTEGER, file_path TEXT)"
)

_SEED_ROWS = [
    ('r1', 'Rock One', 'Band A', 'rock:0.82,pop:0.20', 1995),
    ('r2', 'Rock Two', 'Band B', 'rock:0.55,indie:0.30', 1992),
    ('r3', 'Pop One', 'Band C', 'pop:0.90', 1995),
    ('r4', 'Rock Late', 'Band D', 'rock:0.70', 2010),
    ('r5', 'Rock Faint', 'Band E', 'rock:0.10,pop:0.60', 1994),
]


def _import_module(mod_name, relative_path):
    mod_path = os.path.normpath(os.path.join(_REPO_ROOT, relative_path))
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


@pytest.fixture(scope='session')
def pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")
    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.skip(f"AUDIOMUSE_TEST_DATABASE_URL not reachable: {e}")
        yield dsn
        return
    try:
        import pgserver
    except Exception:
        pytest.skip(
            "No test database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable "
            "DB, or `pip install pgserver` for an ephemeral local instance."
        )
    data_dir = tempfile.mkdtemp(prefix='audiomuse_pg_')
    server = pgserver.get_server(data_dir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture
def brainstorm_db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS score CASCADE")
        cur.execute(_SCORE_DDL)
        for item_id, title, author, mood_vector, year in _SEED_ROWS:
            cur.execute(
                "INSERT INTO score (item_id, title, author, mood_vector, year) "
                "VALUES (%s, %s, %s, %s, %s)",
                (item_id, title, author, mood_vector, year),
            )
    conn.close()
    yield pg_dsn


def _fake_ai_module(recipe_obj):
    mod = types.ModuleType('tasks.ai.api')
    mod.generate_text = Mock(return_value=json.dumps(recipe_obj))
    return mod


@pytest.mark.integration
class TestBrainstormGroundedRetrievalRealDb:
    def test_filter_channel_surfaces_only_matching_rows(self, brainstorm_db, monkeypatch):
        _import_module('tasks.mcp_helper', 'tasks/mcp_helper.py')
        _import_module('tasks.ai.prompts', 'tasks/ai/prompts.py')
        tool_impl = _import_module('tasks.ai.tool_impl', 'tasks/ai/tool_impl.py')
        import config as cfg

        recipe = {
            "filters": {"genres": ["rock"], "year_min": 1990, "year_max": 1999},
            "sound_descriptions": [],
            "seed_artists": [],
            "lyric_themes": [],
        }

        monkeypatch.setattr(cfg, 'AI_BRAINSTORM_POOL_FLOOR', 1)
        monkeypatch.setattr(tool_impl, 'get_db_connection', lambda: psycopg2.connect(brainstorm_db))

        with patch.dict(sys.modules, {'tasks.ai.api': _fake_ai_module(recipe)}):
            result = tool_impl._ai_brainstorm_sync(
                "best rock of the 90s", {"provider": "OLLAMA"}, 50
            )

        ids = {s["item_id"] for s in result["songs"]}
        assert ids == {"r1", "r2"}
        for song in result["songs"]:
            assert song["item_id"] and song["title"] and song["artist"]
