# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every way a batch task can block forever, and who unblocks it.

There are exactly five of them (SCENARIOS). Four are generic and live in
taskqueue.maintenance; the fifth, a child whose worker is alive but whose work
never comes back, can only be judged by the parent that is waiting on it. Before
this module each fan-out parent carried its own copy of that judgement and they
DRIFTED: clustering and analysis fixed the same bug in opposite directions, and
three of the four long-opaque-phase tasks were simply never given a heartbeat at
all because nothing listed what a task was supposed to have.

So the judgement lives here once (ChildDrainSupervisor), each task supplies only
the part that is genuinely its own (how to END a child), and RECOVERY records the
stance of EVERY task type on EVERY scenario - including "not applicable", with
the reason. A missing stance is a test failure, not a silence.

Main Features:
* SCENARIOS names the five ways a run blocks: the main task's worker dies, its
  worker lives but its row goes silent, a child's worker dies, a child's worker
  lives but never returns, and the parent giving up over and over forever.
* ChildDrainSupervisor is the ONE implementation of the sliding no-progress
  window, of WHICH children a give-up ends, and of how many give-ups a run gets.
  Both fan-out parents drive it; they differ only in end_child.
* stalled_victims is the victim rule: the children a worker is HOLDING, or every
  live child when nothing is running at all, because then the queue itself is the
  wedge and sparing it hangs the parent forever.
* row_heartbeat answers the sixth thing, which is not a way to block but a way to
  be WRONGLY unblocked: a phase that is one long opaque call writes no row while
  it runs and is indistinguishable from a wedge. It bumps the row from its own
  connection under a lock_timeout, so a task whose own transaction is sitting on
  that row cannot make its heartbeat block instead, and it stops after
  stop_after_minutes so a phase that really never returns still gets caught
  instead of being propped up forever. EVERY caller passes that bound: an
  unbounded heartbeat on a main task is strictly worse than no heartbeat, because
  it holds the one-live-main index open against every other run forever. The
  budget is measured against the clock, not by adding up intervals, so a database
  outage that skips beats cannot quietly extend it either.
* RECOVERY maps every task type to its stance on every scenario, so a gap is
  visible in one table instead of being spread over six modules.

This module is imported eagerly by every task that can block, so it deliberately
imports database INSIDE the heartbeat thread rather than at the top: at the top it
put app -> ... -> recovery -> database -> config over the eager-import ceiling
test_import_architecture pins.
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

_MAX_SLOW_STEP_WINDOWS = 6


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

    def give_up(self, marks, live_ids):
        stalled_minutes = self._valve.stalled_minutes()
        self._valve.restart()
        victims = stalled_victims(marks, live_ids)
        self.last_spared = len(list(live_ids)) - len(victims)
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
            self._timeout_minutes, ended, len(list(live_ids)), self._label,
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

    started = clock()

    def _loop():
        from database import connect_raw

        conn = None
        while not stop.wait(interval):
            held = clock() - started
            if _budget_spent(started, clock, budget):
                logger.warning(
                    "Task %s has been inside '%s' for %.0f minutes, past the %.0f "
                    "minutes a single step is allowed to hold its row open; the "
                    "heartbeat stops here so the stall valve can see it.",
                    task_id, _describe(label), held / 60.0, budget / 60.0,
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
                task_id, _describe(label), held / 60.0,
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
_NO_CHILDREN = 'this task fans out to nothing, so it never waits on a child'
_NEVER_GIVES_UP = 'this task never gives up on a child, because it has none'


RECOVERY = {
    'main_analysis': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; row_heartbeat covers the nine index builds'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM + ' for each album_analysis child'),
        CHILD_NEVER_RETURNS: handled(
            'ChildDrainSupervisor on ANALYSIS_STALL_TIMEOUT_MINUTES fails the '
            'albums a worker holds, or every live album when none is running'
        ),
        GIVE_UP_RUNS_FOREVER: handled(
            'ANALYSIS_MAX_STALL_GIVE_UPS stops dispatching and finishes the run'
        ),
    },
    'main_clustering': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; the parent writes a row every drain pass'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM + ' for each clustering_batch child'),
        CHILD_NEVER_RETURNS: handled(
            'ChildDrainSupervisor on CLUSTERING_STALL_TIMEOUT_MINUTES revokes the '
            'batches a worker holds, or every live batch when none is running'
        ),
        GIVE_UP_RUNS_FOREVER: handled(
            'the first give-up sets stop_launching, and CLUSTERING_EARLY_STOP_'
            'BATCHES / CLUSTERING_MAX_FAILED_BATCHES end the run before that'
        ),
    },
    'cleaning': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; row_heartbeat covers its final index rebuild'
        ),
        CHILD_WORKER_DIED: not_applicable(_NO_CHILDREN),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
    'provider_migration': {
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; row_heartbeat covers the one whole-catalogue fetch per '
            'server, which is a single call that writes no row while it runs'
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
        MAIN_WORKER_DIED: handled(_RECLAIM),
        MAIN_ROW_SILENT: handled(
            _NUDGE + '; row_heartbeat covers generate_sonic_fingerprint, which '
            'writes no row between the start and the end of a server'
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
    'server_sweep': {
        MAIN_WORKER_DIED: not_applicable(
            'not a main task type: it holds the sweep index, not the one-live-main '
            'index, so a stuck sweep blocks only other sweeps'
        ),
        MAIN_ROW_SILENT: not_applicable(
            'the nudge covers MAIN_TASK_TYPES only; a sweep that goes quiet is '
            'reclaimed when its worker dies and blocks no main task meanwhile'
        ),
        CHILD_WORKER_DIED: handled(_RECLAIM),
        CHILD_NEVER_RETURNS: not_applicable(_NO_CHILDREN),
        GIVE_UP_RUNS_FOREVER: not_applicable(_NEVER_GIVES_UP),
    },
}
