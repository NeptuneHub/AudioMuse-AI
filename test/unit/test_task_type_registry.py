# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every task-type filter derives from task_types, and the derivation is exact.

Nine tuples used to spell out overlapping subsets of the same task types by
hand, and they drifted: plugin tasks and the migration planner reached the
recovery table through none of them, and the nudge gained server_sweep only
after a stuck sweep locked out cleaning and migration with nothing watching it.
These tests fail when a derivation stops matching what the queue actually ran
before it, which is the only way a re-derivation can be proved safe.

config.QUEUE_BLOCKING_TASK_TYPES cannot import the registry: config.py is a
foundation leaf that test_import_architecture forbids from importing any project
module. It is pinned by VALUE here instead, which fails on exactly the edit an
import would have absorbed.

The one-live-main index NAME is a crc32 of the joined MAIN_TASK_TYPES order, so
reordering that tuple silently renames the index, drops the old one and rebuilds
it on every install. The order is pinned as a literal for that reason.

Main Features:
* The historical MAIN order and the index name it checksums are both unchanged
* Every derived tuple equals the literal the queue shipped with
* config's copy is pinned by value because it may not import the registry
* server_sweep is in the nudge set, which nothing asserted before
* The registry stays free of project imports so it can hang below database
"""

import ast
import pathlib

import pytest

import config
import database
import task_types
from taskqueue import sql

REPO = pathlib.Path(__file__).resolve().parents[2]

HISTORICAL_MAIN_TASK_TYPES = (
    'main_analysis', 'main_clustering', 'cleaning', 'provider_migration',
    'sonic_fingerprint',
)

HISTORICAL_MAIN_INDEX_NAME = 'idx_task_status_one_live_main_5d694b56'


class TestTheOneLiveMainIndexDoesNotMove:
    def test_the_main_order_is_the_one_the_index_name_checksums(self):
        assert task_types.MAIN_TASK_TYPES == HISTORICAL_MAIN_TASK_TYPES

    def test_the_index_name_is_unchanged_by_the_derivation(self):
        assert sql.MAIN_INDEX_NAME == HISTORICAL_MAIN_INDEX_NAME, (
            'the index name is a crc32 of the joined MAIN_TASK_TYPES order, so a '
            'different name here means every existing install drops and rebuilds '
            'its admission index on the next boot for no reason'
        )


class TestEveryDerivedTupleMatchesWhatShipped:
    @pytest.mark.parametrize('derived,literal', [
        (sql.MAIN_TASK_TYPES, HISTORICAL_MAIN_TASK_TYPES),
        (sql.NUDGE_TASK_TYPES, HISTORICAL_MAIN_TASK_TYPES + ('server_sweep',)),
        (database.SELF_MANAGED_TASK_TYPES,
         ('server_sweep', 'alchemy_radio', 'worker_control',
          'provider_migration_planner')),
        (database.SELF_MANAGED_TASK_TYPE_PREFIXES, ('plugin.',)),
        (database.INLINE_FLASK_TASK_TYPES, ('alchemy_radio',)),
        (task_types.NON_WORKER_TASK_TYPES, ('alchemy_radio', 'worker_control')),
    ])
    def test_the_derived_tuple_is_identical(self, derived, literal):
        assert derived == literal

    def test_the_non_blocking_set_is_identical_though_its_order_is_not_load_bearing(self):
        assert set(database.NON_BLOCKING_TASK_TYPES) == {
            'worker_control', 'alchemy_radio', 'provider_migration_planner',
        }, (
            'this tuple only ever reaches SQL as a NOT IN list and a set issubset '
            'check, so its ORDER is free, but its membership decides which rows '
            'refuse a batch start'
        )


class TestConfigsCopyIsPinnedByValue:
    def test_config_still_agrees_with_the_registry(self):
        assert tuple(config.QUEUE_BLOCKING_TASK_TYPES) == \
            task_types.QUEUE_BLOCKING_TASK_TYPES, (
            'config.py is a foundation leaf and may not import task_types, so '
            'this equality is the only thing keeping the queue guard and the '
            'admission index describing the same set'
        )


class TestTheNudgeWatchesWhatBlocksAStart:
    def test_the_sweep_is_watched(self):
        assert 'server_sweep' in sql.NUDGE_TASK_TYPES, (
            'a live sweep refuses a cleaning start and a provider-migration '
            'execute, and reclaim needs the worker to DIE, so a wedged sweep '
            'that nothing nudges locks the catalogue out until a restart'
        )

    def test_every_main_index_holder_is_watched(self):
        unwatched = [
            entry.name for entry in task_types.ALL
            if entry.holds_main_index and not entry.watched_by_nudge
        ]
        assert not unwatched, (
            f'{unwatched} hold the one-live-main index, so a wedged run of one '
            'locks out every other main task, and only the nudge can end a task '
            'whose worker is alive but silent'
        )


class TestTheRegistryStaysALeaf:
    def test_it_imports_no_project_module(self):
        tree = ast.parse((REPO / 'task_types.py').read_text(encoding='utf-8'))
        imported = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not imported, (
            'database.py already sits at the bottom of a five-module eager import '
            'chain, which is the ceiling test_import_architecture pins; the '
            'registry can only hang below it while it imports nothing at all'
        )


class TestThePluginPrefixIsSpelledOnce:
    def test_the_blocking_prefixes_are_the_plugin_namespace(self):
        import task_types

        assert task_types.BLOCKING_TASK_TYPE_PREFIXES == ('plugin.',)

    def test_the_queue_guard_derives_its_like_patterns_from_the_registry(self):
        import database
        import task_types

        assert database._BLOCKING_TASK_TYPE_PATTERNS == [
            prefix + '%' for prefix in task_types.BLOCKING_TASK_TYPE_PREFIXES
        ], (
            "get_queue_blocking_task used to OR in a hand-written 'plugin.%'; a "
            'renamed namespace would have left plugin tasks invisible to the guard'
        )

    def test_matches_is_the_one_prefix_and_name_test(self):
        import task_types

        assert task_types.matches('plugin.demo.daily', prefixes=task_types.PREFIXES)
        assert task_types.matches('cleaning', names=task_types.NAMES)
        assert not task_types.matches('plugin', prefixes=task_types.PREFIXES)
        assert not task_types.matches('main_analysis', prefixes=task_types.PREFIXES)
