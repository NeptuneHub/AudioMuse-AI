# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One spelling per queue-written task type, across every module that filters on it.

``taskqueue/sql.py`` owns ``CONTROL_TASK_TYPE`` and ``SWEEP_TASK_TYPE`` because it
is the module whose statements WRITE those rows. Every re-spelling elsewhere is a
filter that a rename silently switches off: nothing fails to import, nothing
raises, the predicate simply stops matching. That has already shipped twice - the
restart handshake reappeared as a phantom dashboard task, and a pending handshake
409-blocked the next analysis and cleaning start.

The Flask modules import the constants, so those sites cannot drift at all.
``database.py`` cannot import them at module level: that adds ``app ->
tasks.duplicate_repair -> tasks.mediaserver.registry -> database -> taskqueue.sql
-> config``, a sixth link on a chain test_import_architecture caps at five. Its
two module-level tuples therefore stay literal and are pinned by value here
instead, which fails on exactly the rename the import would have absorbed.

The modules are read as source rather than imported: importing ``app`` runs the
whole Flask bootstrap, ``init_db`` included, which no unit test does.

Main Features:
* app, app_helper and app_music_servers import the spellings they filter on
* None of the three re-spells one as a bare literal
* database's self-managed and non-blocking lists still hold the right spellings
* The collapse exemption compares against the constant, not a literal
"""

import ast
import re
from pathlib import Path

import pytest

import database
from taskqueue import sql

REPO_ROOT = Path(__file__).resolve().parents[2]

RESPELLABLE = (sql.CONTROL_TASK_TYPE, sql.SWEEP_TASK_TYPE)

FILTERING_MODULES = ('app.py', 'app_helper.py', 'app_music_servers.py')

EXPECTED_IMPORTS = (
    ('app.py', 'taskqueue.sql', 'CONTROL_TASK_TYPE'),
    ('app.py', 'tasks.provider_migration_tasks', 'MIGRATION_PLANNER_TASK_TYPE'),
    ('app_helper.py', 'taskqueue.sql', 'CONTROL_TASK_TYPE'),
    ('app_music_servers.py', 'taskqueue.sql', 'SWEEP_TASK_TYPE'),
)


def _tree(filename):
    return ast.parse((REPO_ROOT / filename).read_text(encoding='utf-8'))


def _imported_names(filename):
    found = set()
    for node in ast.walk(_tree(filename)):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                found.add((node.module, alias.name))
    return found


def _string_literals(filename):
    return [
        node.value
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


class TestTheFilteringModulesImportTheSpelling:

    @pytest.mark.parametrize('filename, module, name', EXPECTED_IMPORTS)
    def test_the_constant_is_imported_from_the_module_that_writes_the_rows(
        self, filename, module, name
    ):
        assert (module, name) in _imported_names(filename)

    @pytest.mark.parametrize('filename', FILTERING_MODULES)
    @pytest.mark.parametrize('spelling', RESPELLABLE)
    def test_none_of_them_respells_it_as_a_bare_literal(self, filename, spelling):
        assert spelling not in _string_literals(filename), (
            f"{filename} spells '{spelling}' by hand; import it from taskqueue.sql "
            f"so a rename there moves this site too"
        )


class TestDatabaseKeepsTheSameVocabulary:

    def test_the_self_managed_list_holds_the_queues_control_task_type(self):
        assert sql.CONTROL_TASK_TYPE in database.SELF_MANAGED_TASK_TYPES

    def test_the_self_managed_list_holds_the_queues_sweep_task_type(self):
        assert sql.SWEEP_TASK_TYPE in database.SELF_MANAGED_TASK_TYPES

    def test_the_non_blocking_list_holds_the_queues_control_task_type(self):
        assert sql.CONTROL_TASK_TYPE in database.NON_BLOCKING_TASK_TYPES

    def test_the_sweep_still_blocks_a_batch_start(self):
        assert sql.SWEEP_TASK_TYPE not in database.NON_BLOCKING_TASK_TYPES

    def test_the_collapse_exemption_reads_the_constant_not_a_literal(self):
        source = (REPO_ROOT / 'database.py').read_text(encoding='utf-8')
        body = source.split('def _collapse_finished_task', 1)[1].split('\ndef ', 1)[0]
        assert 'CONTROL_TASK_TYPE' in body
        assert re.search(r"==\s*'worker_control'", body) is None
