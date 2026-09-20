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
* restarts is the restart budget a type declares and taskqueue.enqueue reads
  when its caller passes none. A child (album, clustering batch, index rebuild)
  gets ONE: the next run re-queues whatever it missed, so three restarts only
  made its parent wait out three backoffs on a failure that would not heal. The
  migration planner gets none, because a silently re-run dry run holds its
  session claimed for a whole extra attempt. None means QUEUE_MAX_ATTEMPTS
* cron is the name a Scheduled Tasks row carries for a schedule that ENQUEUES
  this type, where the two names differ (the analysis row queues main_analysis).
  CRON_TASK_TYPE_TO_QUEUE_TYPE and CRON_RETRY_TASK_TYPES derive from it, so the
  row a blocked cron start fails and the SUCCESS its retry looks for can no
  longer describe different task types. A row that never reaches the queue
  declares none: alchemy_radio runs inline in Flask under its own name
* dotted is the function a cron row enqueues verbatim, for the scheduled
  playlist builders that take nothing but a server scope. CRON_QUEUED_TASKS
  derives from it and IS app_cron's dispatch for them, so a third one is a line
  here rather than a fourth copy of the same eleven-line enqueue call
* side_job marks a short read-only batch the user starts from a page, such as the
  setup wizard naming preview. SIDE_JOB_TASK_TYPES derives every rule it needs in
  one place: it shows as the running task so the dashboard can stop it, but it
  never becomes the last task recap and is left out of the cancel history, its
  start never erases the last task recap, its finish neither collapses the table nor
  records task history, and it refuses and is refused by every other batch start
  through get_queue_blocking_task, because only one batch runs at a time
* BATCH_GATE_TASK_TYPES is every named type that batch admission refuses to
  start beside: the main types plus the side jobs. server_sweep blocks starts
  only through the active-task check, never through this gate
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
    TaskType('sonic_fingerprint', ROLE_MAIN, queue='default',
             cron='sonic_fingerprint',
             dotted='tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
    TaskType('album_of_the_week', ROLE_MAIN, queue='default',
             cron='album_of_the_week',
             dotted='tasks.album_creation_manager.run_album_of_the_week_task',
             holds_main_index=True, watched_by_nudge=True, blocks_starts=True),
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

CRON_QUEUED_TASKS = {entry.cron: entry.dotted for entry in ALL if entry.dotted}


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

