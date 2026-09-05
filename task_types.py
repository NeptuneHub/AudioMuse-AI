# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every task type the queue can hold, and the properties every filter reads.

Nine separate tuples used to spell out overlapping subsets of the same task
types - the one-live-main index, the wedged-task nudge, the queue guard, the
archive exemption, the non-blocking starts, the inline Flask rows - and each was
edited by hand. They drifted: plugin tasks and the migration planner reached the
recovery table through none of them, and NUDGE_TASK_TYPES gained server_sweep
only after a stuck sweep locked out cleaning and migration with nothing watching
it. A task type that is absent from a list nobody cross-checks is invisible.

This module is the one declaration. Every list above is DERIVED from it, so a
new task type cannot be added to one filter and forgotten in another.

It imports nothing, deliberately. config.py is a foundation leaf that may not
import a project module at all, and database.py already sits at the bottom of a
five-module eager chain, which is the ceiling test_import_architecture pins. A
leaf with no imports of its own can hang below database without lengthening that
chain, which is the same reason queue_names.py exists.

config.QUEUE_BLOCKING_TASK_TYPES therefore stays a literal in config.py, because
the wizard's two exclusion lists reference it by name; it is pinned against this
module by test instead of imported.

Main Features:
* ALL is the single ordered declaration of every queue-written task type
* MAIN_TASK_TYPES keeps its historical order, which the one-live-main index name
  is a checksum of, so deriving it never forces an index rebuild
* Nudge, archive-exemption, non-blocking and inline lists all derive from flags
* PREFIXES carries the plugin namespace, which is matched by prefix not by name
"""

ROLE_MAIN = 'main'
ROLE_CHILD = 'child'
ROLE_PLANNER = 'planner'
ROLE_INLINE = 'inline'
ROLE_CONTROL = 'control'


class TaskType:
    def __init__(self, name, role, queue=None, holds_main_index=False,
                 watched_by_nudge=False, blocks_starts=False,
                 self_managed=False, is_prefix=False):
        self.name = name
        self.role = role
        self.queue = queue
        self.holds_main_index = holds_main_index
        self.watched_by_nudge = watched_by_nudge
        self.blocks_starts = blocks_starts
        self.self_managed = self_managed
        self.is_prefix = is_prefix


ALL = (
    TaskType('main_analysis', ROLE_MAIN, queue='high',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('main_clustering', ROLE_MAIN, queue='high',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('cleaning', ROLE_MAIN, queue='high',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('provider_migration', ROLE_MAIN, queue='high',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('sonic_fingerprint', ROLE_MAIN, queue='default',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('server_sweep', ROLE_MAIN, queue='high',
             watched_by_nudge=True, blocks_starts=True, self_managed=True),
    TaskType('alchemy_radio', ROLE_INLINE, self_managed=True),
    TaskType('worker_control', ROLE_CONTROL, self_managed=True),
    TaskType('provider_migration_planner', ROLE_PLANNER, queue='high',
             self_managed=True),
    TaskType('album_analysis', ROLE_CHILD, queue='default'),
    TaskType('clustering_batch', ROLE_CHILD, queue='default'),
    TaskType('index_rebuild', ROLE_CHILD, queue='default'),
    TaskType('plugin.', ROLE_MAIN, blocks_starts=True, self_managed=True,
             watched_by_nudge=True, is_prefix=True),
)

NAMES = tuple(entry.name for entry in ALL if not entry.is_prefix)

PREFIXES = tuple(entry.name for entry in ALL if entry.is_prefix)

MAIN_TASK_TYPES = tuple(entry.name for entry in ALL if entry.holds_main_index)

NUDGE_TASK_TYPES = tuple(
    entry.name for entry in ALL if entry.watched_by_nudge and not entry.is_prefix
)

NUDGE_TASK_TYPE_PREFIXES = tuple(
    entry.name for entry in ALL if entry.watched_by_nudge and entry.is_prefix
)

CHILD_TASK_TYPES = tuple(entry.name for entry in ALL if entry.role == ROLE_CHILD)

SELF_MANAGED_TASK_TYPES = tuple(
    entry.name for entry in ALL if entry.self_managed and not entry.is_prefix
)

SELF_MANAGED_TASK_TYPE_PREFIXES = PREFIXES

NON_BLOCKING_TASK_TYPES = tuple(
    entry.name for entry in ALL
    if entry.self_managed and not entry.is_prefix and not entry.blocks_starts
)

INLINE_FLASK_TASK_TYPES = tuple(
    entry.name for entry in ALL if entry.role == ROLE_INLINE
)

NON_WORKER_TASK_TYPES = tuple(
    entry.name for entry in ALL if entry.role in (ROLE_INLINE, ROLE_CONTROL)
)

QUEUE_BLOCKING_TASK_TYPES = MAIN_TASK_TYPES

BLOCKING_TASK_TYPE_PREFIXES = tuple(
    entry.name for entry in ALL if entry.blocks_starts and entry.is_prefix
)


def matches(task_type, names=(), prefixes=()):
    if task_type in names:
        return True
    return any(task_type.startswith(prefix) for prefix in prefixes)

