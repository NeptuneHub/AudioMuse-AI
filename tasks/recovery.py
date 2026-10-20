# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every way a batch task can block forever, and who unblocks it.

There are exactly five (SCENARIOS). Four are generic and live in
taskqueue.maintenance; the fifth, a child whose worker is alive but whose work
never comes back, can only be judged by the parent waiting on it. Each fan-out
parent used to carry its own copy of that judgement and they DRIFTED, and three
of the four long-opaque-phase tasks had no heartbeat at all. So it lives here
once, each task supplies only what is genuinely its own, and RECOVERY records
the stance of EVERY task type on EVERY scenario - including "not applicable",
with the reason. A missing stance is a test failure, not a silence.

Main Features:
* SCENARIOS names the five ways a run blocks: the main task's worker dies, its
  worker lives but its row goes silent, a child's worker dies, a child's worker
  lives but never returns, and the parent gives up forever
* ChildDrainSupervisor is the ONE implementation of the sliding no-progress
  window, of WHICH children a give-up ends and of how many give-ups a run gets.
  Both fan-out parents drive it through observe with the same child_marks record
  and differ only in end_child; observe answers (moved, gave_up), so no caller
  infers movement from the clock
* stalled_victims is the victim rule: the children a worker is HOLDING, or every
  live child when nothing runs at all, because then the queue is the wedge and
  sparing it hangs the parent
* row_heartbeat answers the sixth thing, which is not a way to block but a way
  to be WRONGLY unblocked: a phase that is one long opaque call writes no row
  and is indistinguishable from a wedge. It bumps from its own connection under
  a lock_timeout, so a task sitting on that row cannot block its own beat,
  and it stops after stop_after_minutes, which EVERY caller passes - an
  unbounded heartbeat on a main task is worse than none, holding the
  one-live-main index open forever. It bounds ONE step: a caller whose label
  names the step restarts it on every change, so a loop of slow steps cannot
  spend a budget sized for one. _MAX_SLOW_STEP_WINDOWS is 2, not the 6 it
  started at: being wrong is cheap, the task resumes from its progress
* OUTSIDE_THE_QUEUE names the two retry engines deliberately not the queue's:
  the cron retry of a BLOCKED START and the migration restart HANDSHAKE
* database is imported INSIDE the heartbeat thread: at the top it put
  recovery -> database -> config over the eager-import ceiling
"""

import logging
import threading
import time
from contextlib import contextmanager

import config
from config import TASK_STATUS_RUNNING

logger = logging.getLogger(__name__)


MAIN_WORKER_DIED = 'main_worker_died'
MAIN_ROW_SILENT = 'main_row_silent'
CHILD_WORKER_DIED = 'child_worker_died'
CHILD_NEVER_RETURNS = 'child_never_returns'
GIVE_UP_RUNS_FOREVER = 'give_up_runs_forever'

SCENARIOS = (
    MAIN_WORKER_DIED,
    MAIN_ROW_SILENT,
    CHILD_WORKER_DIED,
    CHILD_NEVER_RETURNS,
    GIVE_UP_RUNS_FOREVER,
)

_MAX_SLOW_STEP_WINDOWS = 2


_NO_SIGNATURE = object()


class StallValve:
    def __init__(self, timeout_minutes, clock):
        self._timeout_seconds = max(0.0, float(timeout_minutes or 0)) * 60.0
        self._clock = clock
        self._signature = _NO_SIGNATURE
        self._since = clock()

    def moved(self, signature):
        if signature == self._signature:
            return False
        self._signature = signature
        self._since = self._clock()
        return True

    def stalled_minutes(self):
        return (self._clock() - self._since) / 60.0

    def expired(self):
        if self._timeout_seconds <= 0:
            return False
        return (self._clock() - self._since) >= self._timeout_seconds

    def restart(self):
        self._since = self._clock()


def stalled_victims(marks, live_ids):
    held = [task_id for task_id, status in marks if status == TASK_STATUS_RUNNING]
    return held or sorted(live_ids)


def child_marks(parent_task_id, prefix=None, conn=None):
    import taskqueue

    return tuple(sorted(
        (
            str(child.get('task_id')),
            str(child.get('status') or ''),
            str(child.get('progress')),
            str(child.get('beat_at') or ''),
            str(child.get('task_type') or ''),
        )
        for child in taskqueue.live_children(parent_task_id, conn=conn)
        if prefix is None or str(child.get('task_id') or '').startswith(prefix)
    ))


class ChildDrainSupervisor:
    def __init__(self, parent_task_id, end_child, timeout_minutes, max_give_ups,
                 clock, label='child'):
        self._parent_task_id = parent_task_id
        self._end_child = end_child
        self._timeout_minutes = timeout_minutes
        self._max_give_ups = max_give_ups
        self._label = label
        self._valve = StallValve(timeout_minutes, clock)
        self.give_ups = 0
        self.last_spared = 0

    def moved(self, signature):
        return self._valve.moved(signature)

    def expired(self):
        return self._valve.expired()

    def restart(self):
        self._valve.restart()

    def stalled_minutes(self):
        return self._valve.stalled_minutes()

    def exhausted(self):
        return self._max_give_ups > 0 and self.give_ups >= self._max_give_ups

    def observe(self, marks, pending_ids=(), extras=()):
        if self.moved((tuple(marks), tuple(extras))):
            return True, None
        if not self.expired():
            return False, None
        live_ids = {mark[0] for mark in marks} | set(pending_ids)
        if not live_ids:
            return False, None
        return False, self.give_up(
            [(mark[0], mark[1]) for mark in marks], sorted(live_ids)
        )

    def give_up(self, marks, live_ids):
        stalled_minutes = self._valve.stalled_minutes()
        self._valve.restart()
        victims = stalled_victims(marks, live_ids)
        live_count = len(live_ids)
        self.last_spared = live_count - len(victims)
        message = (
            f'The parent gave up on this {self._label}: nothing anywhere in the '
            f'run changed for {stalled_minutes:.0f} minutes, so it stopped waiting '
            'rather than hang the whole run on it.'
        )
        ended = 0
        for task_id in victims:
            try:
                if self._end_child(task_id, message):
                    ended += 1
            except Exception:
                logger.exception("Could not end the %s %s", self._label, task_id)
        self.give_ups += 1
        logger.warning(
            "%s %s made no progress of any kind for %.0f minutes (limit: %s "
            "minutes); ended %d of %d unfinished %s(ren) and left %d that a worker "
            "may still pick up. Give-up %d of %s.",
            self._label.capitalize(), self._parent_task_id, stalled_minutes,
            self._timeout_minutes, ended, live_count, self._label,
            self.last_spared, self.give_ups, self._max_give_ups or 'unbounded',
        )
        return ended, stalled_minutes


_JOIN_TIMEOUT_SECONDS = 10.0
_HEARTBEAT_LOCK_TIMEOUT = "SET LOCAL lock_timeout = '5s'"
_HEARTBEAT_SQL = (
    "UPDATE task_status SET timestamp = NOW() WHERE task_id = %s AND status = %s"
)


def _default_heartbeat_minutes():
    return float(config.QUEUE_WEDGED_MAIN_TASK_MINUTES or 0)


def _heartbeat_interval_seconds(every_minutes):
    minutes = float(
        _default_heartbeat_minutes() if every_minutes is None else every_minutes
    )
    if minutes <= 0:
        return 0.0
    return max(30.0, (minutes * 60.0) / 4.0)


def _budget_spent(started, clock, budget):
    if budget is None:
        return False
    return (clock() - started) >= budget


def _beat_once(conn, task_id):
    with conn.cursor() as cur:
        cur.execute(_HEARTBEAT_LOCK_TIMEOUT)
        cur.execute(_HEARTBEAT_SQL, (str(task_id), TASK_STATUS_RUNNING))
        touched = cur.rowcount
    conn.commit()
    return touched != 0


@contextmanager
def row_heartbeat(task_id, label=None, every_minutes=None, stop_after_minutes=None,
                  clock=time.monotonic):
    interval = _heartbeat_interval_seconds(every_minutes)
    if not task_id or interval <= 0:
        yield
        return
    budget = (
        None if stop_after_minutes is None else max(0.0, float(stop_after_minutes)) * 60.0
    )
    stop = threading.Event()

    def _loop():
        from database import connect_raw

        conn = None
        step = _describe(label)
        started = clock()
        while not stop.wait(interval):
            running = _describe(label)
            if running != step:
                step = running
                started = clock()
            held = clock() - started
            if _budget_spent(started, clock, budget):
                logger.warning(
                    "Task %s has been inside '%s' for %.0f minutes, past the %.0f "
                    "minutes a single step is allowed to hold its row open; the "
                    "heartbeat stops here so the stall valve can see it.",
                    task_id, step, held / 60.0, budget / 60.0,
                )
                break
            try:
                if conn is None:
                    conn = connect_raw(application_name='audiomuse-task-heartbeat')
                if not _beat_once(conn, task_id):
                    logger.info(
                        "Task %s is no longer RUNNING, so its heartbeat stopped; "
                        "whoever owns the row now owns it.", task_id,
                    )
                    break
            except Exception:
                logger.exception(
                    "Could not refresh the row of long-running task %s; the "
                    "wedged-main nudge may cancel it while it is still working",
                    task_id,
                )
                conn = _close_quietly(conn)
                continue
            logger.warning(
                "Task %s has been inside '%s' for %.0f minutes without finishing "
                "it; refreshing its row so it is not mistaken for a wedge.",
                task_id, step, held / 60.0,
            )
        _close_quietly(conn)

    thread = threading.Thread(target=_loop, name=f"heartbeat-{task_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=_JOIN_TIMEOUT_SECONDS)


def _describe(label):
    if callable(label):
        try:
            return label()
        except Exception:
            logger.debug("Heartbeat label callback failed", exc_info=True)
            return 'one long step'
    return label or 'one long step'


def _close_quietly(conn):
    if conn is not None:
        try:
            conn.close()
        except Exception:
            logger.debug("Heartbeat connection close failed", exc_info=True)
    return None


def slow_step_budget_minutes(stall_timeout_minutes):
    if not stall_timeout_minutes or stall_timeout_minutes <= 0:
        return None
    return stall_timeout_minutes * _MAX_SLOW_STEP_WINDOWS


class Stance:
    def __init__(self, mechanism, reason=None, applicable=True):
        self.mechanism = mechanism
        self.reason = reason
        self.applicable = applicable


def handled(mechanism):
    return Stance(mechanism)


def not_applicable(reason):
    return Stance(None, reason=reason, applicable=False)


_RECLAIM = 'taskqueue.maintenance.reclaim_orphans requeues or fails the row'
_NUDGE = (
    'taskqueue.maintenance.nudge_wedged_main_tasks cancels, then terminates the '
    'worker backends at twice the limit'
)
_INLINE_NEVER_ON_A_WORKER = (
    'it never runs on a worker: it is an inline Flask run '
    '(INLINE_FLASK_TASK_TYPES) and its row carries no func, which is '
    'exactly what reclaim filters on'
)
_INLINE_SILENT = (
    'taskqueue.maintenance.fail_stale_inline_rows fails it once its row '
    'has sat untouched for QUEUE_INLINE_STALE_SECONDS, and '
    'app_cron.reap_interrupted_inline_runs fails whatever a restart left '
    'behind at boot'
)
_INLINE_PLAYLIST_HEARTBEAT = (
    '; it is self-managed and holds no one-live-main slot, so a silent row blocks '
    'no other task. The shared scheduled-playlist scaffold in tasks.task_run '
    'wraps {builder} in row_heartbeat on the inline row, beating well inside '
    'QUEUE_INLINE_STALE_SECONDS, because it writes no row between the start and '
    'the end of a server'
)
_NO_CHILDREN = 'this task fans out to nothing, so it never waits on a child'
_NEVER_GIVES_UP = 'this task never gives up on a child, because it has none'


OUTSIDE_THE_QUEUE = {
    'cron_retry': (
        'database.record_cron_retry / app_cron.retry_due_cron_jobs retry a cron '
        'START that the queue guard refused because another catalogue task was '
        'live. Nothing failed: the run never began. It is re-attempted every '
        'CRON_RETRY_INTERVAL_MINUTES up to CRON_RETRY_MAX_MINUTES and then recorded '
        'as a visible skip, never started after that. It never re-runs an inline '
        'online row that executed. A failure retry would be the wrong tool: there '
        'is no row, no attempt and no worker to charge'
    ),
    'provider_migration_restart_handshake': (
        'tasks.provider_migration_tasks._await_worker_restart waits up to '
        '_RESTART_HANDSHAKE_MAX_SECONDS for every worker to acknowledge the restart '
        'a committed migration published, rotating the request id when one is '
        'refused. The catalogue swap is already durable when it runs, so it must '
        'NEVER end as a failure retry of the migration: re-running the migration is '
        'what the handshake guard exists to refuse. recover_migration_handshakes '
        'resumes it from the session row if the worker died mid-wait'
    ),
}

RECOVERY = {
    'main_analysis': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; row_heartbeat covers the nine index builds and the opening '
            'album listing plus work-map scan, once per server on the union path'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM + ' for each album_analysis child'),
        CHILD_NEVER_RETURNS: handled(
            'ChildDrainSupervisor.observe on ANALYSIS_STALL_TIMEOUT_MINUTES fails '
            'the albums a worker holds, or every live album when none is running, '
            'through taskqueue.end_child, the one terminal row a task may write; '
            'the ids it has launched but not yet read back are pending_ids, so a '
            'give-up can never miss an album the cached read had not shown yet. '
            'It ends a victim as FAILURE, not REVOKED, on purpose: the ordinary '
            'reap then counts it into the album failure tally the run reports'
        ),
        GIVE_UP_RUNS_FOREVER: handled(
            'ANALYSIS_MAX_STALL_GIVE_UPS stops dispatching and finishes the run'
        ),
    },
    'main_clustering': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; the parent writes a row every drain pass, and row_heartbeat '
            'covers the two phases where it does not: calibration, which runs the '
            'same opaque fit the batch heartbeats, and the tail, which is one '
            'uncapped LLM naming call plus one media-server write PER PLAYLIST'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM + ' for each clustering_batch child'),
        CHILD_NEVER_RETURNS: handled(
            'ChildDrainSupervisor.observe on CLUSTERING_STALL_TIMEOUT_MINUTES '
            'revokes the batches a worker holds, or every live batch when none is '
            'running, through taskqueue.end_child. It ends a victim as REVOKED, '
            'not FAILURE, on purpose: a '
            'revoked batch is absorbed as failed AND stale, so it feeds '
            'CLUSTERING_MAX_FAILED_BATCHES and CLUSTERING_EARLY_STOP_BATCHES both'
        ),
        GIVE_UP_RUNS_FOREVER: handled(
            'CLUSTERING_MAX_STALL_GIVE_UPS stops launching and finishes the run; '
            'a given-up batch also counts as failed, so CLUSTERING_EARLY_STOP_'
            'BATCHES / CLUSTERING_MAX_FAILED_BATCHES can end it sooner'
        ),
    },
    'cleaning': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; row_heartbeat covers its final index rebuild and the one '
            'whole-catalogue fetch it makes per server'
        ),
        CHILD_WORKER_DIED: not_applicable(_NO_CHILDREN),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'provider_migration': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; row_heartbeat covers the migration transaction, which '
            'rewrites every catalogue id in one statement sequence and writes no '
            'row until it commits. The whole-catalogue fetch is not this task: it '
            'belongs to the dry run, which is a provider_migration_planner row'
        ),
        CHILD_WORKER_DIED: not_applicable(
            'its only child is the restart handshake, which recover_migration_'
            'handshakes re-reserves from the session row rather than waiting on'
        ),
        CHILD_NEVER_RETURNS: not_applicable(
            'the migration never blocks on a child: it hands the restart handshake '
            'over and returns, and the session row is what resumes it'
        ),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'sonic_fingerprint': {
        MAIN_WORKER_DIED: not_applicable(_INLINE_NEVER_ON_A_WORKER),
        MAIN_ROW_SILENT: handled(
            _INLINE_SILENT
            + _INLINE_PLAYLIST_HEARTBEAT.format(builder='generate_sonic_fingerprint')
        ),
        CHILD_WORKER_DIED: not_applicable(_NO_CHILDREN),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'album_of_the_week': {
        MAIN_WORKER_DIED: not_applicable(_INLINE_NEVER_ON_A_WORKER),
        MAIN_ROW_SILENT: handled(
            _INLINE_SILENT
            + _INLINE_PLAYLIST_HEARTBEAT.format(builder='create_album_of_the_week')
        ),
        CHILD_WORKER_DIED: not_applicable(_NO_CHILDREN),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'album_analysis': {
        MAIN_WORKER_DIED: not_applicable('this IS a child, not a main task'),
        MAIN_ROW_SILENT: not_applicable(
            'the wedged-main nudge skips children on purpose: ending a child from '
            'there would take its whole run down with it'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM),
        CHILD_NEVER_RETURNS: handled(
            'its parent gives up on it; it writes a row per TRACK, so a slow album '
            'keeps the window open on its own'
        ),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'clustering_batch': {
        MAIN_WORKER_DIED: not_applicable('this IS a child, not a main task'),
        MAIN_ROW_SILENT: not_applicable(
            'the wedged-main nudge skips children on purpose: ending a child from '
            'there would take its whole run down with it'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM),
        CHILD_NEVER_RETURNS: handled(
            'its parent gives up on it; row_heartbeat covers ONE iteration, which '
            'is a single opaque fit that writes no row until it returns'
        ),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'index_rebuild': {
        MAIN_WORKER_DIED: not_applicable('this IS a child, not a main task'),
        MAIN_ROW_SILENT: not_applicable(
            'the wedged-main nudge skips children on purpose'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM),
        CHILD_NEVER_RETURNS: not_applicable(
            'nothing waits on it: the analysis enqueues it and moves on, so a wedged '
            'rebuild delays the next rebuild and blocks no run. row_heartbeat still '
            'keeps its own row honest'
        ),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'plugin.': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '. It is matched by PREFIX (NUDGE_TASK_TYPE_PREFIXES) '
            'because the namespace is open and there is no fixed list to put in '
            'an IN clause. It needs watching for the same reason the sweep did: '
            'get_queue_blocking_task ORs in plugin.%, so a live plugin task '
            'refuses every cron start and every manual batch start, and reclaim '
            'cannot help because reclaim needs the worker to DIE. row_heartbeat '
            'covers the plugin function itself, one call per server that writes '
            'no row until it returns'
        ),
        CHILD_WORKER_DIED: not_applicable(
            'plugin.manager runs the plugin function inline, once per server in '
            'scope, and enqueues nothing'
        ),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'provider_migration_planner': {
        MAIN_WORKER_DIED: handled(
            'taskqueue.maintenance.reclaim_orphans FAILS it on the first worker '
            'loss rather than requeueing, because it is enqueued with '
            'max_attempts=0 on purpose: a dry run that silently re-ran would '
            'hold the migration session claimed for a whole extra attempt'
        ),
        MAIN_ROW_SILENT: not_applicable(
            'it holds no admission index and is in NON_BLOCKING_TASK_TYPES, so '
            'it refuses no start, and session_discard CANCELS a live planner '
            'rather than blocking on it. A silent planner therefore delays '
            'nothing the user cannot clear from the wizard'
        ),
        CHILD_WORKER_DIED: not_applicable(_NO_CHILDREN),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'naming_preview': {
        MAIN_WORKER_DIED: handled(
            'taskqueue.maintenance.reclaim_orphans FAILS it on the first worker '
            'loss because its type declares restarts=0, and a deliberate restart '
            'that requeues it is caught by run_naming_preview_task itself: a row '
            'that already started fails at once instead of repeating every AI call '
            'with an unsaved prompt'
        ),
        MAIN_ROW_SILENT: handled(
            'it blocks every batch start through get_queue_blocking_task, so '
            'nudge_wedged_main_tasks watches it (watched_by_nudge) and restarts its '
            'worker after QUEUE_WEDGED_MAIN_TASK_MINUTES unchanged. The preview '
            'rewrites its row after every playlist, and the wizard Stop button runs '
            'the global cancel'
        ),
        CHILD_WORKER_DIED: not_applicable(_NO_CHILDREN),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'alchemy_radio': {
        MAIN_WORKER_DIED: not_applicable(_INLINE_NEVER_ON_A_WORKER),
        MAIN_ROW_SILENT: handled(_INLINE_SILENT),
        CHILD_WORKER_DIED: not_applicable(_NO_CHILDREN),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'worker_control': {
        MAIN_WORKER_DIED: not_applicable(
            'the control listener owns this row, not a queue worker; it carries '
            'no func, so reclaim never considers it'
        ),
        MAIN_ROW_SILENT: handled(
            'its timestamp is written once when the action is published and is '
            'never refreshed, so QUEUE_CONTROL_ACTION_WINDOW_SECONDS expires it '
            'on its own and fail_stale_inline_rows then fails the row. A '
            'heartbeat here would be the unbounded reclaim stand-down this must '
            'never have'
        ),
        CHILD_WORKER_DIED: not_applicable(
            'an acknowledgement row is INSERTed already terminal, so no worker '
            'ever claims one and there is no worker to lose'
        ),
        CHILD_NEVER_RETURNS: handled(
            'the publisher polls for acknowledgements on a widening cadence '
            'bounded by QUEUE_CONTROL_TIMEOUT_SECONDS, then stops waiting'
        ),
        GIVE_UP_RUNS_FOREVER: not_applicable(
            'giving up is the caller returning; the request row is left RUNNING '
            'on purpose so late acknowledgements still land on it'
        ),
    },
    'server_sweep': {
        MAIN_WORKER_DIED: handled(
            _RECLAIM + '; it holds the sweep index rather than the one-live-main '
            'one, but a live sweep still refuses a cleaning start and a '
            'provider-migration execute, so it is not free to sit there'
        ),
        MAIN_ROW_SILENT: handled(
            _NUDGE + ', which watches this type too (NUDGE_TASK_TYPES): the old '
            'stance that a stuck sweep "blocks only other sweeps" was wrong, and '
            'nothing else watches it because reclaim needs the worker to DIE. '
            'row_heartbeat covers its one whole-catalogue fetch per server'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
}
