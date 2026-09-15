# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Waiting for the workers to answer, without asking 120 times to hear it once.

The publisher polls because the answer is a row rather than a message, and the
only thing the cadence may buy is fewer round trips: the verdict, the deadline
and the late-acknowledgement warning all have to come out exactly as they did
under the flat quarter-second loop. These tests pin both halves - the cadence
widens, and every observable of the wait stays where it was. The widening is
capped in seconds as well as in fraction, because a tenth of the restore's
action-window budget is fifteen seconds of dead time on the stop request and
fifteen more on the start request.

The wait's verdict and the row it writes are two different things, and the tests
below separate them deliberately. The caller learns about a failure the instant
one listener reports it, but the request row is also the marker
``taskqueue.maintenance`` stands its reclaim down for, so a refusal may not take
that marker away from the listeners that are still stopping their services. Only
a fleet-wide SUCCESS does, the finish never rewrites the marker's timestamp, and
an action that stops no worker never becomes a marker in the first place.

The queue-side status vocabulary is pinned here too, for the request row and the
control-in-flight guard specifically: those two statements are the two ends of
one handshake, written by ``taskqueue.control`` and read by
``taskqueue.maintenance``, so a spelling hardcoded in either module is a
handshake that silently stops matching the moment config renames a status.

Main Features:
* The acknowledgement wait widens its cadence instead of polling flat
* The widening is capped in seconds, so a late ack on a long budget is seen at once
* It still ends on the deadline, having slept exactly the budget and no more
* A verdict that is already available costs no sleep at all
* A refusal is written to the request row and still leaves it standing as a marker
* Only a fleet-wide success retracts the marker, and never its publish timestamp
* Reclaim stands down for exactly the actions whose listener requeues the tasks
* Renaming a status in config moves both modules' statements together
* The shipped statements keep the exact text they were reviewed with
"""

import importlib
import json
from unittest.mock import MagicMock

import pytest

import config
from taskqueue import control
from taskqueue import maintenance
from taskqueue import sql

RENAMED = {
    'TASK_STATUS_NEW': 'QUEUED',
    'TASK_STATUS_RUNNING': 'BUSY',
    'TASK_STATUS_SUCCESS': 'DONE',
    'TASK_STATUS_FAIL': 'BROKEN',
    'TASK_STATUS_REVOKED': 'DROPPED',
}

TIMEOUT = 30.0

WINDOW = config.QUEUE_CONTROL_ACTION_WINDOW_SECONDS

RESTORE_BUDGET = WINDOW

TASK_ID = 'main_analysis-1'

REQUEST_ID = 'control-1'

NO_REQUEST_ROW = object()


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class _AckConn:
    def __init__(self, rows=None, rows_after=1):
        self._rows = rows or []
        self._rows_after = rows_after
        self.counts = 0

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        self.counts += 1

    def fetchall(self):
        return list(self._rows) if self.counts >= self._rows_after else []

    def commit(self):
        pass


class _TimedAckConn:
    def __init__(self, clock, rows_at, rows):
        self._clock = clock
        self._rows_at = rows_at
        self._rows = rows
        self.counts = 0

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        self.counts += 1

    def fetchall(self):
        return list(self._rows) if self._clock.now >= self._rows_at else []

    def commit(self):
        pass


class _RequestCursor:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, statement, params=None):
        self._conn.calls.append((' '.join(statement.split()), params))

    def fetchone(self):
        return (self._conn.listeners,)

    def fetchall(self):
        return list(self._conn.acks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _RequestConn:
    def __init__(self, acks=(), listeners=2):
        self.calls = []
        self.acks = list(acks)
        self.listeners = listeners

    def cursor(self):
        return _RequestCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def written(self, prefix):
        return [params for statement, params in self.calls if statement.startswith(prefix)]

    def verdicts(self):
        return [params[0] for params in self.written('UPDATE task_status')]

    def inserted(self):
        return self.written('INSERT INTO task_status')[0]


class _MarkerCursor:
    def __init__(self, action, status, age):
        self._action = action
        self._status = status
        self._age = age
        self.statement = None
        self.params = None
        self.answer = None

    def execute(self, statement, params=None):
        statement = ' '.join(statement.split())
        if 'make_interval' not in statement:
            return
        self.statement = statement
        self.params = params
        self.answer = (1,) if self._in_flight() else None

    def _in_flight(self):
        if self._action is NO_REQUEST_ROW:
            return False
        if self._age >= self.params[-1]:
            return False
        return self._status_matches() and self._action_matches()

    def _status_matches(self):
        if f"status <> '{config.TASK_STATUS_SUCCESS}'" in self.statement:
            return self._status != config.TASK_STATUS_SUCCESS
        if f"status = '{config.TASK_STATUS_RUNNING}'" in self.statement:
            return self._status == config.TASK_STATUS_RUNNING
        raise AssertionError(
            f"the guard no longer compares the status in a shape this test can "
            f"evaluate; teach it the new one: {self.statement}"
        )

    def _action_matches(self):
        if 'sub_type_identifier' not in self.statement:
            return True
        if self._action is None:
            return 'sub_type_identifier IS NULL' in self.statement
        return self._action in self.params[1]

    def fetchone(self):
        return self.answer

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _MarkerConn:
    def __init__(self, action, status=None, age=0.0):
        self.cur = _MarkerCursor(
            action, status if status is not None else config.TASK_STATUS_RUNNING, age
        )

    def cursor(self):
        return self.cur

    def commit(self):
        pass

    def rollback(self):
        pass


def _requeues(monkeypatch, action):
    import restart_manager

    listener = control.ControlListener()
    called = []
    monkeypatch.setattr(listener, 'connect', lambda: MagicMock())
    monkeypatch.setattr(listener, '_already_acknowledged', lambda *_args: None)
    monkeypatch.setattr(restart_manager, 'stop_supervisor_workers', lambda: True)
    monkeypatch.setattr(restart_manager, 'start_supervisor_workers', lambda: True)
    monkeypatch.setattr(restart_manager, 'restart_supervisor_workers', lambda: True)
    monkeypatch.setattr(listener, '_dispatch_plugin_sync', lambda: True)
    monkeypatch.setattr(listener, '_record_ack', lambda conn, *_args: conn)
    monkeypatch.setattr(
        listener, '_requeue_tasks_of_stopped_workers', lambda _conn: called.append(action)
    )
    listener.on_notify(
        sql.CHANNEL_CONTROL, json.dumps({'action': action, 'request_id': REQUEST_ID})
    )
    return bool(called)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(control, 'time', fake)
    return fake


@pytest.fixture
def charged(monkeypatch):
    attempts = []

    def requeue_or_fail(_cur, task_id, _now, _details):
        attempts.append(task_id)
        return config.TASK_STATUS_NEW

    monkeypatch.setattr(
        maintenance.sql, 'running_tasks',
        lambda _cur, grace_seconds=None: [
            {'task_id': TASK_ID, 'attempts': 0, 'max_attempts': config.QUEUE_MAX_ATTEMPTS}
        ],
    )
    monkeypatch.setattr(maintenance.sql, 'try_hold', lambda _cur, _task_id: True)
    monkeypatch.setattr(maintenance.sql, 'release', lambda _cur, _task_id: None)
    monkeypatch.setattr(maintenance.sql, 'notify_job', lambda _cur, _queue: None)
    monkeypatch.setattr(maintenance.sql, 'requeue_or_fail', requeue_or_fail)
    return attempts


@pytest.fixture
def renamed(monkeypatch):
    for name, spelling in RENAMED.items():
        monkeypatch.setattr(config, name, spelling)
    monkeypatch.setattr(
        config, 'TASK_STATUS_LIVE', (RENAMED['TASK_STATUS_NEW'], RENAMED['TASK_STATUS_RUNNING'])
    )
    monkeypatch.setattr(
        config,
        'TASK_STATUS_TERMINAL',
        (
            RENAMED['TASK_STATUS_SUCCESS'],
            RENAMED['TASK_STATUS_FAIL'],
            RENAMED['TASK_STATUS_REVOKED'],
        ),
    )
    importlib.reload(control)
    importlib.reload(maintenance)
    yield control, maintenance
    monkeypatch.undo()
    importlib.reload(control)
    importlib.reload(maintenance)


class TestTheWaitWidensItsCadence:
    def test_a_silent_fleet_costs_a_fraction_of_the_flat_quarter_second_loop(self, clock):
        conn = _AckConn(rows=[])

        control._await_acks(conn, REQUEST_ID, 2, TIMEOUT)

        flat = TIMEOUT / config.QUEUE_CONTROL_POLL_INTERVAL_SECONDS
        assert conn.counts < flat / 4

    def test_the_first_wait_is_still_the_configured_interval(self, clock):
        control._await_acks(_AckConn(rows=[]), REQUEST_ID, 2, TIMEOUT)

        assert clock.sleeps[0] == config.QUEUE_CONTROL_POLL_INTERVAL_SECONDS

    def test_no_single_wait_ever_exceeds_a_tenth_of_the_budget(self, clock):
        control._await_acks(_AckConn(rows=[]), REQUEST_ID, 2, TIMEOUT)

        assert max(clock.sleeps) <= TIMEOUT * control.POLL_INTERVAL_CEILING_FRACTION

    def test_the_ceiling_follows_the_budget_it_is_waiting_on(self):
        assert control.poll_interval_ceiling(TIMEOUT) == pytest.approx(3.0)
        assert control.poll_interval_ceiling(5.0) == pytest.approx(0.5)

    def test_a_budget_shorter_than_the_interval_never_polls_slower_than_configured(self):
        interval = config.QUEUE_CONTROL_POLL_INTERVAL_SECONDS

        assert control.poll_interval_ceiling(interval) == interval


class TestALateAcknowledgementIsNoticedPromptlyOnALongBudget:
    def test_the_restore_budget_never_buys_a_fifteen_second_blind_tail(self, clock):
        control._await_acks(_AckConn(rows=[]), REQUEST_ID, 2, RESTORE_BUDGET)

        assert max(clock.sleeps) <= control.POLL_INTERVAL_CEILING_SECONDS

    def test_the_seconds_cap_binds_before_the_fraction_does_on_that_budget(self):
        assert (
            RESTORE_BUDGET * control.POLL_INTERVAL_CEILING_FRACTION
            > control.POLL_INTERVAL_CEILING_SECONDS
        ), 'the fraction alone would be the ceiling again and the cap would be dead code'
        assert control.poll_interval_ceiling(RESTORE_BUDGET) == pytest.approx(
            control.POLL_INTERVAL_CEILING_SECONDS
        )

    def test_an_ack_that_lands_at_seventeen_seconds_is_not_seen_at_thirty(self, clock):
        landed = 17.0
        conn = _TimedAckConn(clock, landed, [(config.TASK_STATUS_SUCCESS, 2)])

        assert control._await_acks(conn, REQUEST_ID, 2, RESTORE_BUDGET) == (True, True)
        assert clock.now <= landed + control.POLL_INTERVAL_CEILING_SECONDS, (
            'under the fraction alone the poll after t=15.75s was t=30.75s, so every '
            'restore paid that tail twice: once on the stop and once on the start'
        )

    def test_the_cap_still_costs_a_small_fraction_of_the_flat_loop(self, clock):
        conn = _AckConn(rows=[])

        control._await_acks(conn, REQUEST_ID, 2, RESTORE_BUDGET)

        flat = RESTORE_BUDGET / config.QUEUE_CONTROL_POLL_INTERVAL_SECONDS
        assert conn.counts < flat / 8


class TestTheDeadlineIsExactlyWhereItWas:
    def test_the_wait_sleeps_the_budget_and_not_one_second_more(self, clock):
        control._await_acks(_AckConn(rows=[]), REQUEST_ID, 2, TIMEOUT)

        assert sum(clock.sleeps) == pytest.approx(TIMEOUT)
        assert clock.now == pytest.approx(TIMEOUT)

    def test_an_unanswered_request_is_left_open_for_the_late_acknowledgements(self, clock):
        assert control._await_acks(_AckConn(rows=[]), REQUEST_ID, 2, TIMEOUT) == (False, False)

    def test_the_advisory_budget_is_honoured_just_as_exactly(self, clock):
        timeout = config.QUEUE_CONTROL_ADVISORY_TIMEOUT_SECONDS

        control._await_acks(_AckConn(rows=[]), REQUEST_ID, 2, timeout)

        assert sum(clock.sleeps) == pytest.approx(timeout)

    def test_the_restore_budget_is_honoured_just_as_exactly_under_the_cap(self, clock):
        control._await_acks(_AckConn(rows=[]), REQUEST_ID, 2, RESTORE_BUDGET)

        assert sum(clock.sleeps) == pytest.approx(RESTORE_BUDGET)


class TestAVerdictIsReturnedTheMomentItExists:
    def test_a_fleet_that_already_answered_costs_no_sleep_at_all(self, clock):
        conn = _AckConn(rows=[(config.TASK_STATUS_SUCCESS, 2)])

        assert control._await_acks(conn, REQUEST_ID, 2, TIMEOUT) == (True, True)
        assert clock.sleeps == []

    def test_an_acknowledgement_that_arrives_mid_wait_ends_it_early(self, clock):
        conn = _AckConn(rows_after=3, rows=[(config.TASK_STATUS_SUCCESS, 2)])

        assert control._await_acks(conn, REQUEST_ID, 2, TIMEOUT) == (True, True)
        assert clock.now < TIMEOUT

    def test_a_listener_that_reported_a_failure_ends_the_wait_as_a_failure(self, clock):
        conn = _AckConn(
            rows=[(config.TASK_STATUS_SUCCESS, 1), (config.TASK_STATUS_FAIL, 1)]
        )

        assert control._await_acks(conn, REQUEST_ID, 2, TIMEOUT) == (False, True)
        assert clock.sleeps == []


class TestThePublisherWritesTheVerdictItPromised:
    def test_a_fleet_wide_success_finishes_the_request_row(self, clock):
        conn = _RequestConn(acks=[(config.TASK_STATUS_SUCCESS, 2)])

        assert control.publish_control_request(
            control.ACTION_RESTART, request_id=REQUEST_ID, conn=conn
        ) is True
        assert conn.verdicts() == [config.TASK_STATUS_SUCCESS]

    def test_a_refusal_is_recorded_at_once_so_the_migration_can_rotate_its_request(
        self, clock
    ):
        conn = _RequestConn(acks=[(config.TASK_STATUS_FAIL, 1)])

        assert control.publish_control_request(
            control.ACTION_RESTART, request_id=REQUEST_ID, conn=conn
        ) is False
        assert conn.verdicts() == [config.TASK_STATUS_FAIL], (
            'the caller and the durable request row must both say FAIL: '
            'get_control_request_result is what rotates a failed restart handshake'
        )
        assert clock.sleeps == []

    def test_a_request_nobody_answered_is_left_running_for_the_late_acks(self, clock):
        conn = _RequestConn(acks=[])

        assert control.publish_control_request(
            control.ACTION_RESTART, request_id=REQUEST_ID, timeout_seconds=TIMEOUT, conn=conn
        ) is False
        assert conn.verdicts() == []

    def test_nothing_is_published_when_no_listener_is_connected(self, clock):
        conn = _RequestConn(acks=[(config.TASK_STATUS_SUCCESS, 2)], listeners=0)

        assert control.publish_control_request(
            control.ACTION_RESTART, request_id=REQUEST_ID, conn=conn
        ) is False
        assert conn.written('INSERT INTO task_status') == []
        assert conn.verdicts() == []


class TestAFailureNeverShortensTheProtectionOfTheListenersStillWorking:
    def test_the_verdict_never_rewrites_the_timestamp_the_window_is_measured_from(self):
        assert 'timestamp' not in ' '.join(control._FINISH_REQUEST.split()), (
            'the stand-down expires one action window after the PUBLISH; a finish '
            'that touched the timestamp would push that instant out by the whole '
            'ack wait'
        )

    def test_a_refused_request_row_still_stands_the_reclaim_down(self, charged):
        conn = _MarkerConn(
            control.ACTION_RESTART, status=config.TASK_STATUS_FAIL, age=35.0
        )

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == []
        assert charged == [], (
            'pod A can answer FAIL after it has already killed its workers while '
            'pod B is still legitimately stopping its three services'
        )

    def test_it_stands_down_for_the_rest_of_the_window_and_not_one_second_longer(
        self, charged
    ):
        conn = _MarkerConn(
            control.ACTION_RESTART, status=config.TASK_STATUS_FAIL, age=WINDOW + 1.0
        )

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == [
            (TASK_ID, config.TASK_STATUS_NEW)
        ]
        assert charged == [TASK_ID]

    def test_a_fleet_wide_success_ends_the_stand_down_immediately(self, charged):
        conn = _MarkerConn(
            control.ACTION_RESTART, status=config.TASK_STATUS_SUCCESS, age=1.0
        )

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == [
            (TASK_ID, config.TASK_STATUS_NEW)
        ]
        assert charged == [TASK_ID], (
            'SUCCESS is written only once every live listener has acknowledged, so '
            'nobody is still stopping anything and a lock nobody holds is a real loss'
        )


class TestAnActionThatStopsNoWorkerSuspendsNothing:
    def test_a_plugin_sync_in_flight_does_not_hold_the_reclaim_off(self, charged):
        conn = _MarkerConn(control.ACTION_PLUGIN_SYNC)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == [
            (TASK_ID, config.TASK_STATUS_NEW)
        ]
        assert charged == [TASK_ID], (
            'a pre-sync that outran the advisory budget left its row RUNNING, and a '
            'worker that really died in the next window was never reclaimed'
        )

    def test_starting_the_workers_stops_none_of_them_either(self, charged):
        conn = _MarkerConn(control.ACTION_START)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == [
            (TASK_ID, config.TASK_STATUS_NEW)
        ]
        assert charged == [TASK_ID]

    @pytest.mark.parametrize('action', control.WORKER_STOPPING_ACTIONS)
    def test_an_action_that_stops_workers_still_holds_it_off(self, charged, action):
        conn = _MarkerConn(action)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == []
        assert charged == []

    def test_a_request_row_that_names_no_action_is_read_as_one_that_stops_workers(
        self, charged
    ):
        conn = _MarkerConn(None)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == []
        assert charged == [], (
            'the expensive mistake is charging an attempt for a loss that never '
            'happened, so an unreadable marker is read the protective way'
        )

    def test_no_control_row_at_all_reclaims_immediately(self, charged):
        conn = _MarkerConn(NO_REQUEST_ROW)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == [
            (TASK_ID, config.TASK_STATUS_NEW)
        ]
        assert charged == [TASK_ID]


class TestTheStandDownCoversExactlyWhatTheListenerTakesResponsibilityFor:
    def test_the_actions_it_covers_are_the_ones_whose_listener_requeues_the_tasks(
        self, monkeypatch
    ):
        requeueing = tuple(
            action for action in control.VALID_ACTIONS if _requeues(monkeypatch, action)
        )

        assert requeueing == control.WORKER_STOPPING_ACTIONS, (
            'standing reclaim down means deferring to the uncharged requeue, so an '
            'action that never performs one has nothing to defer to'
        )

    @pytest.mark.parametrize('action', control.VALID_ACTIONS)
    def test_the_request_row_records_the_action_the_guard_filters_on(self, clock, action):
        conn = _RequestConn(acks=[(config.TASK_STATUS_SUCCESS, 2)])

        control.publish_control_request(action, request_id=REQUEST_ID, conn=conn)

        assert 'sub_type_identifier' in control._INSERT_REQUEST
        assert conn.inserted()[2] == action

    def test_the_guard_binds_the_control_task_type_and_the_stopping_actions(self, charged):
        conn = _MarkerConn(control.ACTION_RESTART)

        maintenance.reclaim_orphans(conn, grace_seconds=0)

        assert conn.cur.params[0] == sql.CONTROL_TASK_TYPE
        assert conn.cur.params[1] == list(control.WORKER_STOPPING_ACTIONS)

    def test_the_window_stays_the_last_parameter_the_guard_binds(self, charged):
        conn = _MarkerConn(control.ACTION_RESTART)

        maintenance.reclaim_orphans(conn, grace_seconds=0)

        assert conn.cur.params[-1] == config.QUEUE_CONTROL_ACTION_WINDOW_SECONDS

    def test_the_guard_still_ignores_the_acknowledgement_rows(self, charged):
        conn = _MarkerConn(control.ACTION_RESTART)

        maintenance.reclaim_orphans(conn, grace_seconds=0)

        assert 'AND parent_task_id IS NULL' in conn.cur.statement, (
            'every listener writes an ack row of the same task_type; without this '
            'condition one ack would keep the stand-down alive by itself'
        )


class TestBothEndsOfTheHandshakeSpeakConfigsVocabulary:
    def test_the_request_row_and_the_guard_that_reads_it_follow_a_rename(self, renamed):
        renamed_control, renamed_maintenance = renamed

        assert "%s, 'BUSY', 0, %s, NOW(), %s)" in renamed_control._INSERT_REQUEST
        assert "AND status <> 'DONE'" in renamed_maintenance._CONTROL_ACTION_IN_FLIGHT

    def test_the_previous_handshake_is_spared_under_the_renamed_spelling(self, renamed):
        assert "AND r.status = 'BUSY'" in renamed[0]._CLEAR_PREVIOUS_CONTROL_ROWS

    def test_the_stale_inline_sweep_follows_the_rename(self, renamed):
        statement = renamed[1]._FAIL_STALE_INLINE_ROWS

        assert "SET status = 'BROKEN'" in statement
        assert "AND status IN ('QUEUED','BUSY')" in statement

    def test_the_shared_payload_wipe_follows_the_rename(self, renamed):
        assert "AND status IN ('DONE','BROKEN','DROPPED')" in renamed[1]._CLEAR_TERMINAL_SHARED

    def test_not_one_of_the_four_statements_keeps_the_old_spelling(self, renamed):
        renamed_control, renamed_maintenance = renamed

        for statement in (
            renamed_control._INSERT_REQUEST,
            renamed_control._CLEAR_PREVIOUS_CONTROL_ROWS,
            renamed_maintenance._FAIL_STALE_INLINE_ROWS,
            renamed_maintenance._CLEAR_TERMINAL_SHARED,
            renamed_maintenance._CONTROL_ACTION_IN_FLIGHT,
        ):
            for spelling in ('NEW', 'RUNNING', 'SUCCESS', 'FAIL', 'REVOKED'):
                assert f"'{spelling}'" not in statement


class TestTheShippedStatementsKeepTheirExactText:
    def test_the_request_row_is_still_inserted_running(self):
        assert (
            "VALUES (%s, NULL, %s, %s, 'RUNNING', 0, %s, NOW(), %s)"
            in control._INSERT_REQUEST
        )

    def test_the_clear_still_spares_a_running_handshake(self):
        assert "AND r.status = 'RUNNING'" in control._CLEAR_PREVIOUS_CONTROL_ROWS

    def test_the_inline_sweep_still_fails_live_rows_only(self):
        assert "SET status = 'FAIL', progress = 100" in maintenance._FAIL_STALE_INLINE_ROWS
        assert "AND status IN ('NEW','RUNNING')" in maintenance._FAIL_STALE_INLINE_ROWS

    def test_the_payload_wipe_still_targets_terminal_rows_only(self):
        assert "AND status IN ('SUCCESS','FAIL','REVOKED')" in maintenance._CLEAR_TERMINAL_SHARED

    def test_the_guard_still_stands_down_for_an_unfinished_request_row(self):
        assert "AND status <> 'SUCCESS'" in maintenance._CONTROL_ACTION_IN_FLIGHT
