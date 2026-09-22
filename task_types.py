# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every task type the queue can hold, and the properties every filter reads.

Nine separate tuples used to spell out overlapping subsets of the same task
types - the one-live-main index, the nudge, the queue guard, the archive
exemption, the non-blocking starts, the inline Flask rows - and they drifted,
because a type absent from a list nobody cross-checks is invisible. This module
is the one declaration and every list above is DERIVED from it. It imports nothing deliberately, so it can hang below database.py
without lengthening the eager chain test_import_architecture pins.
config.QUEUE_BLOCKING_TASK_TYPES stays a literal there, pinned against this
module by test.

Main Features:
* ALL is the single ordered declaration of every queue-written task type
* MAIN_TASK_TYPES keeps its historical order, which the one-live-main index name
  is a checksum of, so deriving it never forces an index rebuild
* PREFIXES carries the plugin namespace, matched by prefix, not by name
* restarts is the restart budget a type declares and taskqueue.enqueue reads
  when its caller passes none. A child (album, clustering batch, index rebuild)
  gets ONE, since the next run re-queues what it missed; the migration planner
  gets none, because a re-run dry run holds its session claimed for another
  attempt. None means QUEUE_MAX_ATTEMPTS
* cron is the name a Scheduled Tasks row carries for a schedule that starts
  this type, where the names differ (the analysis row queues main_analysis).
  CRON_TASK_TYPE_TO_QUEUE_TYPE and CRON_RETRY_TASK_TYPES derive from it, so a
  blocked start and the retry that looks for its SUCCESS cannot name different
  types. alchemy_radio declares none: it runs inline under its own name
* dotted is the function a cron row runs INLINE in Flask for the playlist
  builders; CRON_INLINE_TASKS derives from it. Like the radio they are online,
  self-managed types that never hold nor wait for the one-live-main slot
* side_job marks a short read-only batch a page starts, such as the wizard's
  naming preview. SIDE_JOB_TASK_TYPES derives every rule: it shows as the
  running task so the dashboard can stop it, but never becomes the last recap,
  stays out of the cancel history, records no task history, and refuses and is
  refused by every other batch start
* BATCH_GATE_TASK_TYPES is every named type batch admission refuses to start
  beside: the main types plus the side jobs. server_sweep blocks starts only
  through the active-task check
"""

ROLE_MAIN = 'main'
ROLE_CHILD = 'child'
ROLE_PLANNER = 'planner'
ROLE_INLINE = 'inline'
ROLE_CONTROL = 'control'

NAMING_PREVIEW_TASK_TYPE = 'naming_preview'


class TaskType:
    def __init__(self, name, role, queue=None, holds_main_index=False,
                 watched_by_nudge=False, blocks_starts=False,
                 self_managed=False, is_prefix=False, restarts=None,
                 side_job=False, cron=None, dotted=None):
        self.name = name
        self.role = role
        self.queue = queue
        self.holds_main_index = holds_main_index
        self.watched_by_nudge = watched_by_nudge
        self.blocks_starts = blocks_starts
        self.self_managed = self_managed
        self.is_prefix = is_prefix
        self.restarts = restarts
        self.side_job = side_job
        self.cron = cron
        self.dotted = dotted


ALL = (
    TaskType('main_analysis', ROLE_MAIN, queue='high', cron='analysis',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('main_clustering', ROLE_MAIN, queue='high', cron='clustering',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('cleaning', ROLE_MAIN, queue='high',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('provider_migration', ROLE_MAIN, queue='high',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('sonic_fingerprint', ROLE_INLINE,
             cron='sonic_fingerprint',
             dotted='tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task',
             self_managed=True),
    TaskType('album_of_the_week', ROLE_INLINE,
             cron='album_of_the_week',
             dotted='tasks.album_creation_manager.run_album_of_the_week_task',
             self_managed=True),
    TaskType('server_sweep', ROLE_MAIN, queue='high',
             watched_by_nudge=True, blocks_starts=True, self_managed=True),
    TaskType('alchemy_radio', ROLE_INLINE, self_managed=True),
    TaskType('worker_control', ROLE_CONTROL, self_managed=True),
    TaskType('provider_migration_planner', ROLE_PLANNER, queue='high',
             self_managed=True, restarts=0),
    TaskType(NAMING_PREVIEW_TASK_TYPE, ROLE_PLANNER, queue='default',
             watched_by_nudge=True, blocks_starts=True, self_managed=True,
             restarts=0, side_job=True),
    TaskType('album_analysis', ROLE_CHILD, queue='default', restarts=1),
    TaskType('clustering_batch', ROLE_CHILD, queue='default', restarts=1),
    TaskType('index_rebuild', ROLE_CHILD, queue='default', restarts=1),
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

SIDE_JOB_TASK_TYPES = tuple(
    entry.name for entry in ALL if entry.side_job and not entry.is_prefix
)

BATCH_GATE_TASK_TYPES = QUEUE_BLOCKING_TASK_TYPES + SIDE_JOB_TASK_TYPES

CRON_TASK_TYPE_TO_QUEUE_TYPE = {
    entry.cron: entry.name for entry in ALL if entry.cron
}

CRON_RETRY_TASK_TYPES = tuple(CRON_TASK_TYPE_TO_QUEUE_TYPE)

CRON_INLINE_TASKS = {
    entry.cron: entry.dotted for entry in ALL
    if entry.dotted and entry.role == ROLE_INLINE
}


def matches(task_type, names=(), prefixes=()):
    if task_type in names:
        return True
    return any(task_type.startswith(prefix) for prefix in prefixes)


def restarts_for(task_type):
    for entry in ALL:
        if entry.is_prefix:
            if task_type.startswith(entry.name):
                return entry.restarts
        elif entry.name == task_type:
            return entry.restarts
    return None

