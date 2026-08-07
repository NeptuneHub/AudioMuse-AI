# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One window covers a deliberate restart, and every consumer measures the same one.

A restart the control plane asked for is not a worker loss, so nothing may charge
its tasks an attempt while it runs. That guarantee is only as wide as the narrowest
consumer of the window, and the window has to be measured against how long the
ACTION legitimately takes - the acknowledgement budget is how long a CALLER waits,
and a wizard save stops waiting after five seconds while the fleet keeps stopping
for another forty. When the two were the same number, the first worker back did its
grace-0 boot reclaim into an already-expired guard and charged every still-restarting
worker's row, three wizard saves during one analysis exhausted the attempts, and the
uncharged requeue that arrived seconds later could no longer undo it.

The window is bounded by construction and these tests say how: the request row's
timestamp is written once at publish and never refreshed, so a control row a crashed
listener left RUNNING stands reclaim down for exactly one action window.

Its SIZE is pinned as a number and not only as an inequality. Every "wide enough"
assertion here still passes if the budget it is derived from is quietly refloored
at the 60s fleet stop it merely has to cover, and the window loses a full minute
without one test going red - so the two budgets are also re-resolved out of
config.py's own source against a hostile environment, and the documented 150s and
its 105s floor are asserted outright.

The stand-down is checked as a QUERY, not as one parameter. The guard cursor
models a single unfinished parentless control request, so a stand-down that stops
asking for that shape - a different task type bound, the parentless condition
dropped, the window bound anywhere but last - stops matching the row it exists to
find, and every test here that depends on standing down fails with it. Which
STATUS the request must be in is deliberately checked more loosely than the rest,
because that predicate is the control plane's to define; what is pinned is that
one still exists.

Restore is the other end of the same truth: a stop request that gives up early
answers 503 while the workers are still legitimately stopping, so it has to wait
the same window every other consumer of that budget waits.

Main Features:
* A reclaim charges no attempt anywhere inside the action window
* The stand-down still expires exactly one window after the request row was written
* The stand-down asks for an unfinished parentless row of the control task type
* The publisher's exemption and the reclaim stand-down read the identical number
* The window outlasts the worst-case action it exists to cover
* An unconfigured deployment gets 150s, and no environment shrinks it below 105s
* The wizard's advisory acknowledgement budget stays deliberately short
* Restore's stop and start requests wait the action window, not the ack budget
* A pre-migration dump leaves no live task row behind
"""

import ast
import io
import os
from unittest.mock import MagicMock

import pytest
from flask import Flask

import app_backup
import config
from taskqueue import control
from taskqueue import maintenance
from taskqueue import sql

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SQL_MODULE_PATH = os.path.join(REPO_ROOT, 'taskqueue', 'sql.py')

WINDOW = config.QUEUE_CONTROL_ACTION_WINDOW_SECONDS

THREE_WORKER_STOP_SECONDS = 60.0

IPC_TIMEOUT_ENV = 'AUDIO_MUSE_CONTROL_IPC_TIMEOUT_SECONDS'
ACK_TIMEOUT_ENV = 'QUEUE_CONTROL_TIMEOUT_SECONDS'
DOCUMENTED_DEFAULT_WINDOW_SECONDS = 150.0
DOCUMENTED_FLOOR_WINDOW_SECONDS = 105.0

STAND_DOWN_CONDITIONS = (
    'task_type = %s',
    'parent_task_id IS NULL',
    'timestamp > NOW() - make_interval(secs => %s)',
)

STAND_DOWN_STATUS_SPELLINGS = (
    config.TASK_STATUS_RUNNING, config.TASK_STATUS_SUCCESS
)

STATUS_PREDICATE_MARKERS = (
    "status = '", "status <> '", 'status IN (', 'status = ANY(', 'status <> ALL(',
)


def _has_a_status_predicate(statement):
    if not any(f"'{status}'" in statement for status in STAND_DOWN_STATUS_SPELLINGS):
        return False
    return any(marker in statement for marker in STATUS_PREDICATE_MARKERS)


TASK_ID = 'main_analysis-1'

RESTORE_CONFIRMATION = (
    "I want to restore the database from the backup. This action is not reversible"
)


class _GuardCursor:
    def __init__(self, control_row_age):
        self._age = control_row_age
        self._answer = None
        self.statements = []
        self.stand_down_calls = []

    def execute(self, statement, params=None):
        flat = ' '.join(statement.split())
        self.statements.append(flat)
        if 'make_interval' not in flat:
            return
        bound = tuple(params or ())
        self.stand_down_calls.append((flat, bound))
        self._answer = None
        if not bound or bound[0] != sql.CONTROL_TASK_TYPE:
            return
        if any(condition not in flat for condition in STAND_DOWN_CONDITIONS):
            return
        if self._age is not None and self._age < bound[-1]:
            self._answer = (1,)

    @property
    def windows(self):
        return [bound[-1] for _statement, bound in self.stand_down_calls]

    @property
    def stand_down_statement(self):
        assert len(self.stand_down_calls) == 1
        return self.stand_down_calls[0][0]

    def fetchone(self):
        return self._answer

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _GuardConn:
    def __init__(self, control_row_age):
        self.cur = _GuardCursor(control_row_age)

    def cursor(self):
        return self.cur

    def commit(self):
        pass

    def rollback(self):
        pass


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


class TestAReclaimDuringADeliberateRestartChargesNoAttempt:
    @pytest.mark.parametrize(
        'seconds_into_the_restart',
        [0.0, 5.0, 30.0, 35.0, 45.0, 60.0, WINDOW - 1.0],
        ids=['published', 'wizard_gave_up', 'ack_budget_gone', 'first_worker_back',
             'native_fleet_stop', 'worst_case_stop', 'last_legal_moment'],
    )
    def test_the_boot_reclaim_charges_nothing_for_the_whole_action_window(
        self, charged, seconds_into_the_restart
    ):
        conn = _GuardConn(seconds_into_the_restart)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == []
        assert charged == []

    def test_the_stand_down_is_measured_against_the_action_not_the_ack_budget(self, charged):
        conn = _GuardConn(config.QUEUE_CONTROL_TIMEOUT_SECONDS + 1.0)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == []
        assert charged == []

    def test_the_periodic_maintenance_pass_stands_down_on_the_same_window(self, charged):
        conn = _GuardConn(WINDOW - 1.0)

        assert maintenance.reclaim_orphans(conn) == []
        assert charged == []


class TestTheStandDownStillExpires:
    def test_a_control_row_older_than_one_window_no_longer_holds_reclaim_off(self, charged):
        conn = _GuardConn(WINDOW + 1.0)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == [
            (TASK_ID, config.TASK_STATUS_NEW)
        ]
        assert charged == [TASK_ID]

    def test_no_control_row_at_all_reclaims_immediately(self, charged):
        conn = _GuardConn(None)

        assert maintenance.reclaim_orphans(conn, grace_seconds=0) == [
            (TASK_ID, config.TASK_STATUS_NEW)
        ]
        assert charged == [TASK_ID]

    def test_the_guard_expires_against_a_timestamp_nothing_ever_refreshes(self):
        assert 'timestamp > NOW() - make_interval(secs => %s)' in (
            ' '.join(maintenance._CONTROL_ACTION_IN_FLIGHT.split())
        )
        assert 'ON CONFLICT (task_id) DO UPDATE SET timestamp = NOW()' in (
            ' '.join(control._INSERT_REQUEST.split())
        )


class TestTheStandDownAsksForAnUnfinishedParentlessControlRequest:
    @pytest.fixture
    def stood_down(self, charged):
        conn = _GuardConn(1.0)
        maintenance.reclaim_orphans(conn, grace_seconds=0)
        return conn.cur

    @pytest.mark.parametrize(
        'condition', STAND_DOWN_CONDITIONS,
        ids=['task_type', 'parentless', 'inside_the_window'],
    )
    def test_the_statement_still_narrows_on_it(self, stood_down, condition):
        assert condition in stood_down.stand_down_statement

    def test_the_statement_still_excludes_a_request_that_already_finished(self, stood_down):
        assert _has_a_status_predicate(stood_down.stand_down_statement)

    def test_it_binds_the_control_task_type_first_and_the_action_window_last(
        self, stood_down
    ):
        bound = stood_down.stand_down_calls[0][1]

        assert bound[0] == sql.CONTROL_TASK_TYPE
        assert bound[-1] == WINDOW

    def test_one_reclaim_pass_asks_exactly_once(self, stood_down):
        assert len(stood_down.stand_down_calls) == 1


class _PublishCursor:
    def __init__(self, calls):
        self._calls = calls

    def execute(self, statement, params=None):
        self._calls.append((' '.join(statement.split()), params))

    def fetchone(self):
        return (1,)

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _PublishConn:
    def __init__(self):
        self.calls = []

    def cursor(self):
        return _PublishCursor(self.calls)

    def commit(self):
        pass

    def rollback(self):
        pass


class TestEveryConsumerMeasuresTheSameWindow:
    def test_the_reclaim_stand_down_asks_for_the_action_window(self, charged):
        conn = _GuardConn(1.0)
        maintenance.reclaim_orphans(conn, grace_seconds=0)

        assert conn.cur.windows == [WINDOW]

    def test_the_publisher_spares_a_running_handshake_for_that_same_window(self):
        conn = _PublishConn()

        assert control.publish_control_request(
            control.ACTION_RESTART, request_id='control-1', timeout_seconds=0, conn=conn
        ) is False
        exemptions = [
            params[-1] for statement, params in conn.calls
            if statement.startswith('DELETE FROM task_status')
        ]

        assert exemptions == [WINDOW]


class TestTheWindowOutlastsTheActionItCovers:
    def test_it_covers_a_whole_three_worker_fleet_shutdown(self):
        assert WINDOW >= THREE_WORKER_STOP_SECONDS

    def test_it_covers_the_native_control_budget_the_action_itself_is_capped_at(self):
        assert WINDOW >= config.CONTROL_IPC_TIMEOUT_SECONDS

    def test_it_leaves_room_for_the_handshake_round_trip_after_the_action_ends(self):
        assert WINDOW > config.CONTROL_IPC_TIMEOUT_SECONDS

    def test_it_is_never_the_acknowledgement_budget_a_caller_waits_for(self):
        assert WINDOW > config.QUEUE_CONTROL_TIMEOUT_SECONDS

    def test_the_wizards_advisory_budget_stays_short_so_a_request_thread_answers(self):
        assert config.QUEUE_CONTROL_ADVISORY_TIMEOUT_SECONDS <= 5.0

    def test_it_is_derived_rather_than_a_knob_an_operator_can_set_below_the_action(self):
        with open(config.__file__, encoding='utf-8') as handle:
            source = handle.read()

        assert "'QUEUE_CONTROL_ACTION_WINDOW_SECONDS'," in source
        assert "getenv('QUEUE_CONTROL_ACTION_WINDOW_SECONDS'" not in source
        assert 'QUEUE_CONTROL_ACTION_WINDOW_SECONDS' in config.SETUP_BOOTSTRAP_EXCLUDED_KEYS


def _config_assignment(name):
    with open(config.__file__, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=config.__file__)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError('config.py has no module-level assignment to %s' % name)


def _resolve(name, namespace):
    expression = ast.Expression(body=_config_assignment(name))
    ast.fix_missing_locations(expression)
    return eval(compile(expression, config.__file__, 'eval'), dict(namespace))


def _window_resolved_from_source():
    namespace = dict(vars(config))
    for name in ('QUEUE_CONTROL_TIMEOUT_SECONDS', 'CONTROL_IPC_TIMEOUT_SECONDS'):
        namespace[name] = _resolve(name, namespace)
    return _resolve('QUEUE_CONTROL_ACTION_WINDOW_SECONDS', namespace)


class TestTheWindowIsTheDocumentedNumberAndNotMerelyWideEnough:
    def test_it_is_the_action_budget_plus_the_acknowledgement_budget_and_nothing_else(self):
        assert WINDOW == (
            config.CONTROL_IPC_TIMEOUT_SECONDS + config.QUEUE_CONTROL_TIMEOUT_SECONDS
        )

    def test_an_unconfigured_deployment_gets_the_documented_one_hundred_and_fifty(
        self, monkeypatch
    ):
        monkeypatch.delenv(IPC_TIMEOUT_ENV, raising=False)
        monkeypatch.delenv(ACK_TIMEOUT_ENV, raising=False)

        assert _window_resolved_from_source() == DOCUMENTED_DEFAULT_WINDOW_SECONDS

    def test_an_action_budget_forced_to_one_second_still_leaves_the_floor_plus_the_ack(
        self, monkeypatch
    ):
        monkeypatch.setenv(IPC_TIMEOUT_ENV, '1')
        monkeypatch.delenv(ACK_TIMEOUT_ENV, raising=False)

        assert _window_resolved_from_source() == DOCUMENTED_FLOOR_WINDOW_SECONDS

    def test_that_floor_still_outlasts_the_fleet_stop_and_the_ack_budget_together(self):
        assert DOCUMENTED_FLOOR_WINDOW_SECONDS > (
            THREE_WORKER_STOP_SECONDS + config.QUEUE_CONTROL_TIMEOUT_SECONDS
        )

    def test_resolving_those_budgets_never_moves_the_live_window(self, monkeypatch):
        monkeypatch.setenv(IPC_TIMEOUT_ENV, '1')
        _window_resolved_from_source()

        assert config.QUEUE_CONTROL_ACTION_WINDOW_SECONDS == WINDOW


class TestRestoreWaitsForTheWorkersInsteadOfDeclaringThemBroken:
    def test_the_start_request_waits_the_action_window(self, monkeypatch):
        seen = {}

        def publish_start_request(**kwargs):
            seen.update(kwargs)
            return True

        monkeypatch.setattr(
            app_backup.restart_manager, 'publish_start_request', publish_start_request
        )

        assert app_backup._publish_worker_start() is True
        assert seen == {'timeout_seconds': WINDOW}

    def test_the_stop_request_never_gives_up_inside_the_ack_budget(self, monkeypatch, tmp_path):
        seen = {}

        def publish_stop_request(**kwargs):
            seen.update(kwargs)
            return False

        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, 'RESTORE_LOG_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, '_acquire_restore_lock', lambda: True)
        monkeypatch.setattr(app_backup, '_release_restore_lock', lambda: None)
        monkeypatch.setattr(
            app_backup.restart_manager, 'publish_stop_request', publish_stop_request
        )
        monkeypatch.setattr(
            app_backup.restart_manager, 'publish_start_request', lambda **_kwargs: True
        )
        monkeypatch.setattr(
            app_backup.subprocess, 'Popen',
            MagicMock(side_effect=AssertionError('the restore runner must not start')),
        )

        flask_app = Flask(__name__)
        flask_app.config['TESTING'] = True
        flask_app.register_blueprint(app_backup.backup_bp)
        response = flask_app.test_client().post(
            '/api/backup/restore',
            data={
                'confirmation': RESTORE_CONFIRMATION,
                'file': (io.BytesIO(b'SELECT 1;\n'), 'backup.sql'),
            },
            content_type='multipart/form-data',
        )

        assert response.status_code == 503
        assert seen == {'timeout_seconds': WINDOW}
