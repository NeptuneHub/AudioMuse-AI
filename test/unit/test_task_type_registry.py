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
* The cron tables derive from the registry, so the task type a blocked schedule
  fails under and the one its retry looks for cannot describe different rows
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
)

HISTORICAL_MAIN_INDEX_NAME = 'idx_task_status_one_live_main_88c6fe'


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
        (sql.NUDGE_TASK_TYPES, HISTORICAL_MAIN_TASK_TYPES + ('server_sweep', 'naming_preview')),
        (database.SELF_MANAGED_TASK_TYPES,
         ('sonic_fingerprint', 'album_of_the_week', 'server_sweep',
          'alchemy_radio', 'worker_control', 'provider_migration_planner',
          'naming_preview')),
        (database.SELF_MANAGED_TASK_TYPE_PREFIXES, ('plugin.',)),
        (database.INLINE_FLASK_TASK_TYPES,
         ('sonic_fingerprint', 'album_of_the_week', 'alchemy_radio')),
        (task_types.NON_WORKER_TASK_TYPES,
         ('sonic_fingerprint', 'album_of_the_week', 'alchemy_radio', 'worker_control')),
        (task_types.SIDE_JOB_TASK_TYPES, ('naming_preview',)),
        (task_types.BATCH_GATE_TASK_TYPES,
         HISTORICAL_MAIN_TASK_TYPES + ('naming_preview',)),
    ])
    def test_the_derived_tuple_is_identical(self, derived, literal):
        assert derived == literal

    def test_the_non_blocking_set_is_identical_though_its_order_is_not_load_bearing(self):
        assert set(database.NON_BLOCKING_TASK_TYPES) == {
            'worker_control', 'alchemy_radio', 'provider_migration_planner',
            'sonic_fingerprint', 'album_of_the_week',
        }, (
            'this tuple only ever reaches SQL as a NOT IN list and a set issubset '
            'check, so its ORDER is free, but its membership decides which rows '
            'refuse a batch start'
        )


class TestTheCronTablesAreDerivedFromTheRegistry:
    def test_the_queue_type_map_is_the_one_both_readers_shipped_with(self):
        assert task_types.CRON_TASK_TYPE_TO_QUEUE_TYPE == {
            'analysis': 'main_analysis',
            'clustering': 'main_clustering',
            'sonic_fingerprint': 'sonic_fingerprint',
            'album_of_the_week': 'album_of_the_week',
        }, (
            'app_cron and database.cron_retry_task_already_done each held a copy '
            'of this literal; a row failed under one task type while its retry '
            'looked for a SUCCESS under another is exactly what that allowed'
        )

    def test_the_retry_tuple_is_the_same_set_as_the_map(self):
        assert task_types.CRON_RETRY_TASK_TYPES == (
            'analysis', 'clustering', 'sonic_fingerprint', 'album_of_the_week',
        )

    def test_the_playlist_tasks_run_inline_because_only_flask_holds_the_index(self):
        assert not hasattr(task_types, 'CRON_QUEUED_TASKS'), (
            'every scheduled playlist builder runs inline in Flask; a queued '
            'dispatch table would only be dead code that could route one to a '
            'worker that never loads the index'
        )
        assert task_types.CRON_INLINE_TASKS == {
            'sonic_fingerprint':
                'tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task',
            'album_of_the_week':
                'tasks.album_creation_manager.run_album_of_the_week_task',
        }
        for cron_type in task_types.CRON_INLINE_TASKS:
            assert cron_type in task_types.NON_WORKER_TASK_TYPES, (
                f'{cron_type} queries the in-memory index, which a worker never '
                'loads, so it must never be enqueued'
            )
            assert cron_type in database.NON_BLOCKING_TASK_TYPES, (
                f'{cron_type} is an online run like the radio: it must never hold '
                'the one-live-main slot nor refuse, or wait for, a batch start'
            )
            assert cron_type not in task_types.MAIN_TASK_TYPES
            assert cron_type not in task_types.BATCH_GATE_TASK_TYPES
            assert cron_type not in task_types.NUDGE_TASK_TYPES

    def test_every_inline_playlist_task_resolves_through_the_queue_allow_list(self):
        import taskqueue

        for cron_type, dotted in task_types.CRON_INLINE_TASKS.items():
            assert dotted in taskqueue.ALLOWED_FUNCS, (
                f'app_cron resolves {dotted} through taskqueue.resolve_func, which '
                'refuses a path outside the allow list, so every run would fail'
            )
            assert dotted in taskqueue.TASK_FUNC_ERROR_CODES
            assert task_types.CRON_TASK_TYPE_TO_QUEUE_TYPE[cron_type] == cron_type

    def test_an_inline_cron_row_declares_no_queue_type(self):
        assert 'alchemy_radio' not in task_types.CRON_TASK_TYPE_TO_QUEUE_TYPE, (
            'the radio runs inline in Flask under its own name and never reaches '
            'the queue, so it has no second name to map to'
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

    def test_no_inline_type_holds_the_main_index(self):
        assert not [
            entry.name for entry in task_types.ALL
            if entry.role == task_types.ROLE_INLINE
            and (entry.holds_main_index or entry.blocks_starts or entry.watched_by_nudge)
        ]

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


class TestAChildDeclaresItsOwnRestartBudget:
    @pytest.mark.parametrize('child', task_types.CHILD_TASK_TYPES)
    def test_every_child_gets_one_restart(self, child):
        assert task_types.restarts_for(child) == 1, (
            'a lost or failed album, batch or rebuild is re-run once; the next run '
            're-queues whatever it missed, so three restarts only made the parent '
            'wait out three backoffs on a failure that would not heal'
        )

    def test_the_planner_opts_out_and_its_enqueue_agrees(self):
        assert task_types.restarts_for('provider_migration_planner') == 0
        source = (REPO / 'app_provider_migration.py').read_text(encoding='utf-8')
        assert 'max_attempts=0' in source

    @pytest.mark.parametrize(
        'task_type', task_types.MAIN_TASK_TYPES + ('server_sweep', 'plugin.demo.daily')
    )
    def test_a_main_task_keeps_the_queue_default(self, task_type):
        assert task_types.restarts_for(task_type) is None

    def test_enqueue_reads_the_budget_from_the_registry(self, monkeypatch):
        from unittest.mock import MagicMock

        import taskqueue

        seen = {}
        monkeypatch.setattr(
            sql, 'insert_job', lambda cur, **kwargs: seen.update(kwargs) or True
        )
        monkeypatch.setattr(sql, 'notify_job', lambda cur, queue: None)

        taskqueue.enqueue(
            'tasks.analysis.analyze_album_task', task_id='kid-1',
            task_type='album_analysis', parent_task_id='root-1', conn=MagicMock(),
        )

        assert seen['max_attempts'] == 1

    def test_an_explicit_budget_still_wins(self, monkeypatch):
        from unittest.mock import MagicMock

        import taskqueue

        seen = {}
        monkeypatch.setattr(
            sql, 'insert_job', lambda cur, **kwargs: seen.update(kwargs) or True
        )
        monkeypatch.setattr(sql, 'notify_job', lambda cur, queue: None)

        taskqueue.enqueue(
            'tasks.analysis.analyze_album_task', task_id='kid-1',
            task_type='album_analysis', parent_task_id='root-1', max_attempts=5,
            conn=MagicMock(),
        )

        assert seen['max_attempts'] == 5
