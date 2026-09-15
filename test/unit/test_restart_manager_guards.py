# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The local half of a restart: the IPC budget, the self-restart gate, the poll.

The native control server answers only AFTER it has synchronously stopped and
started every worker child, so the socket timeout is a correctness constant and
not a convenience: undershooting it makes a correct restart report failure, and
the tasks that restart killed are then reclaimed with an attempt charge. The
delayed Flask self-restart is equally one-sided - it belongs to the Flask process
and to nothing else - and the control-result poll runs every few seconds for up
to fifteen minutes, so it may not open a connection per column it reads.

The budget's floor is pinned WITHOUT reloading config. Importing config runs
``_apply_db_overrides``, which on anything that is not an explicit worker calls
``ensure_table()`` and bootstraps schema into whatever DATABASE_URL happens to
resolve to, so re-executing that module from a unit test is a landmine that only
goes off on the machine where the role env var is set. The floor is therefore
resolved by lifting the single assignment expression out of config.py with ``ast``
and evaluating that expression alone against a patched environment: the same
contract, none of the module body.

Both NUMBERS are pinned exactly, not merely bounded from below by the fleet stop
they have to cover. A floor of 60 and a default of 60 satisfy every "at least a
three-worker shutdown" assertion in this file while quietly halving the action
window derived from them, and config says in as many words that this floor is not
negotiable downwards because it was already raised once after a real incident.

``_supervisorctl_already_satisfied`` is the container-side other half: supervisord
exits non-zero when a start finds the program already running or a stop finds it
already down, and both of those ARE the requested end state. It is per-action and
per-line on purpose - "already started" answers a start and not a stop, and one
genuinely failed service in a multi-service command still fails the command.

Main Features:
* The control IPC budget covers three sequential worker shutdowns, floor included
* The floor is exactly the 75s an incident raised it to and the default exactly 120s
* An override under the floor is raised back to it; one over the floor is honoured
* Resolving an override never re-executes config, so the live constant never moves
* The socket really receives that budget before it connects
* Only a Flask process arms the delayed self-restart timer
* DISABLE_FLASK_RESTART suppresses the restart without raising
* An already-running start and an already-stopped stop are successes, not failures
* Neither one answers the OTHER action, and a mixed result is still a failure
* A legacy JSONB details column still identifies the recorded action
* A details payload that is not an object falls back to a match instead of raising
* One control-result poll opens exactly one database connection
* The setup-wizard parameter catalog cannot drift: every bootstrap-excluded key
  stays hidden or basic, and every static/setup.js section key exists in config,
  reaches the advanced list (neither hidden nor basic) and is listed in exactly
  one section
"""

import ast
import re
import socket
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app_setup
import config
import restart_manager

_IPC_TIMEOUT_ENV = 'AUDIO_MUSE_CONTROL_IPC_TIMEOUT_SECONDS'
_IPC_TIMEOUT_NAME = 'CONTROL_IPC_TIMEOUT_SECONDS'
_THREE_WORKER_SHUTDOWN_SECONDS = 60
_DOCUMENTED_IPC_FLOOR_SECONDS = 75.0
_DOCUMENTED_IPC_DEFAULT_SECONDS = 120.0


def _config_assignment(name):
    with open(config.__file__, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=config.__file__)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
            return node.value
    raise AssertionError('config.py has no module-level assignment to %s' % name)


def _resolve_config_value(name):
    expression = ast.Expression(body=_config_assignment(name))
    ast.fix_missing_locations(expression)
    return eval(compile(expression, config.__file__, 'eval'), dict(vars(config)))


@pytest.fixture
def mock_timer(monkeypatch):
    timer_cls = MagicMock()
    monkeypatch.setattr(restart_manager.threading, 'Timer', timer_cls)
    return timer_cls


class _Cursor:
    def __init__(self, row):
        self._row = row
        self.statements = []

    def execute(self, statement, params=None):
        self.statements.append((' '.join(statement.split()), params))

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _conn(row):
    cursor = _Cursor(row)
    connection = MagicMock()
    connection.cursor.side_effect = lambda *_a, **_k: cursor
    return connection


class TestTheNativeControlBudgetOutlastsAWholeWorkerFleetShutdown:
    def test_the_budget_covers_three_sequential_worker_shutdowns_of_twenty_seconds(self):
        assert config.CONTROL_IPC_TIMEOUT_SECONDS >= _THREE_WORKER_SHUTDOWN_SECONDS
        assert restart_manager.CONTROL_IPC_TIMEOUT_SECONDS >= _THREE_WORKER_SHUTDOWN_SECONDS

    def test_an_environment_override_under_the_floor_is_raised_back_to_it(
        self, monkeypatch
    ):
        monkeypatch.setenv(_IPC_TIMEOUT_ENV, '1')

        assert _resolve_config_value(_IPC_TIMEOUT_NAME) >= _THREE_WORKER_SHUTDOWN_SECONDS

    def test_that_floor_is_exactly_the_seventy_five_seconds_an_incident_raised_it_to(
        self, monkeypatch
    ):
        monkeypatch.setenv(_IPC_TIMEOUT_ENV, '1')

        assert _resolve_config_value(_IPC_TIMEOUT_NAME) == _DOCUMENTED_IPC_FLOOR_SECONDS

    def test_the_pinned_floor_is_above_the_fleet_stop_the_live_budget_must_cover(self):
        assert _DOCUMENTED_IPC_FLOOR_SECONDS > _THREE_WORKER_SHUTDOWN_SECONDS
        assert config.CONTROL_IPC_TIMEOUT_SECONDS >= _DOCUMENTED_IPC_FLOOR_SECONDS

    def test_an_unset_environment_leaves_the_documented_one_hundred_and_twenty_seconds(
        self, monkeypatch
    ):
        monkeypatch.delenv(_IPC_TIMEOUT_ENV, raising=False)

        assert _resolve_config_value(_IPC_TIMEOUT_NAME) == _DOCUMENTED_IPC_DEFAULT_SECONDS

    def test_an_environment_override_over_the_floor_is_honoured_in_full(
        self, monkeypatch
    ):
        monkeypatch.setenv(_IPC_TIMEOUT_ENV, '900')

        assert _resolve_config_value(_IPC_TIMEOUT_NAME) == 900.0

    def test_resolving_the_override_never_re_executes_the_config_module(
        self, monkeypatch
    ):
        live = config.CONTROL_IPC_TIMEOUT_SECONDS
        monkeypatch.setenv(_IPC_TIMEOUT_ENV, '900')

        assert _resolve_config_value(_IPC_TIMEOUT_NAME) == 900.0
        assert config.CONTROL_IPC_TIMEOUT_SECONDS == live
        assert restart_manager.CONTROL_IPC_TIMEOUT_SECONDS == live

    def test_the_budget_is_never_aliased_to_the_short_advisory_ack_deadline(self):
        assert (
            restart_manager.CONTROL_IPC_TIMEOUT_SECONDS
            > restart_manager.CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS
        )

    def test_the_socket_receives_that_budget_before_it_connects(self, monkeypatch):
        seen = {}

        class _Socket:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def settimeout(self, timeout):
                seen['timeout'] = timeout

            def connect(self, address):
                seen['address'] = address

            def sendall(self, payload):
                seen['payload'] = payload

            def recv(self, _size):
                return b'ok'

        monkeypatch.setattr(
            restart_manager,
            '_control_endpoint',
            lambda: (socket.AF_INET, ('127.0.0.1', 12345), 'test-control'),
        )
        monkeypatch.setattr(restart_manager.socket, 'socket', lambda *_a: _Socket())

        assert restart_manager._send_control(['restart', 'queue-worker-default']) is True
        assert seen['timeout'] >= _THREE_WORKER_SHUTDOWN_SECONDS


class TestAnActionAlreadyInItsRequestedEndStateIsNotASupervisorctlFailure:
    def test_a_start_that_finds_the_program_already_running_is_a_success(self):
        assert restart_manager._supervisorctl_already_satisfied(
            'start', 'queue-worker-high: ERROR (already started)', ''
        ) is True

    def test_a_stop_that_finds_the_program_already_down_is_a_success(self):
        assert restart_manager._supervisorctl_already_satisfied(
            'stop', 'queue-worker-high: ERROR (not running)', ''
        ) is True

    def test_a_stop_whose_programs_all_report_stopped_is_a_success(self):
        assert restart_manager._supervisorctl_already_satisfied(
            'stop', 'queue-worker-high: stopped\nqueue-maintenance: ERROR (not running)', ''
        ) is True

    def test_an_already_running_program_never_satisfies_a_stop(self):
        assert restart_manager._supervisorctl_already_satisfied(
            'stop', 'queue-worker-high: ERROR (already started)', ''
        ) is False

    def test_a_program_that_is_not_running_never_satisfies_a_start(self):
        assert restart_manager._supervisorctl_already_satisfied(
            'start', 'queue-worker-high: ERROR (not running)', ''
        ) is False

    def test_one_still_failing_service_fails_the_whole_multi_service_command(self):
        assert restart_manager._supervisorctl_already_satisfied(
            'stop', 'queue-worker-high: stopped\nqueue-maintenance: ERROR (spawn error)', ''
        ) is False

    def test_no_output_at_all_is_never_read_as_an_end_state_already_reached(self):
        assert restart_manager._supervisorctl_already_satisfied('stop', '', '') is False
        assert restart_manager._supervisorctl_already_satisfied('start', '', '') is False

    def test_a_blank_line_in_the_output_does_not_defeat_the_check(self):
        assert restart_manager._supervisorctl_already_satisfied(
            'start', 'queue-worker-high: ERROR (already started)\n\n', ''
        ) is True

    def test_the_reason_is_read_from_stderr_as_well_as_stdout(self):
        assert restart_manager._supervisorctl_already_satisfied(
            'start', '', 'queue-worker-high: ERROR (already started)'
        ) is True

    @pytest.mark.parametrize('action', ['restart', 'status', ''])
    def test_no_other_action_is_ever_assumed_satisfied(self, action):
        assert restart_manager._supervisorctl_already_satisfied(
            action,
            'queue-worker-high: ERROR (already started)',
            'queue-maintenance: ERROR (not running)',
        ) is False


class TestTheDetailedSupervisorctlCallActsOnThatVerdict:
    @staticmethod
    def _run(monkeypatch, stdout):
        monkeypatch.setattr(restart_manager, '_use_control_ipc', lambda: False)
        completed = MagicMock(returncode=1, stdout=stdout, stderr='')
        monkeypatch.setattr(
            restart_manager.subprocess, 'run', lambda *_a, **_k: completed
        )
        return restart_manager.run_supervisorctl_detail(['stop', 'queue-worker-high'])

    def test_a_non_zero_exit_for_a_program_already_down_is_reported_as_success(
        self, monkeypatch
    ):
        ok, detail = self._run(monkeypatch, 'queue-worker-high: ERROR (not running)\n')

        assert ok is True
        assert 'not running' in detail

    def test_a_non_zero_exit_that_really_failed_is_still_a_failure_with_its_reason(
        self, monkeypatch
    ):
        ok, detail = self._run(monkeypatch, 'queue-worker-high: ERROR (spawn error)\n')

        assert ok is False
        assert detail.startswith('exit 1')
        assert 'spawn error' in detail


class TestOnlyTheFlaskProcessArmsTheDelayedSelfRestart:
    def test_a_worker_process_never_arms_the_timer(self, monkeypatch, mock_timer):
        monkeypatch.setenv('SERVICE_TYPE', 'worker')
        monkeypatch.setattr(config, 'DISABLE_FLASK_RESTART', False)

        assert restart_manager.schedule_flask_restart() is False
        mock_timer.assert_not_called()

    def test_an_unset_service_type_never_arms_the_timer(self, monkeypatch, mock_timer):
        monkeypatch.delenv('SERVICE_TYPE', raising=False)
        monkeypatch.setattr(config, 'DISABLE_FLASK_RESTART', False)

        assert restart_manager.schedule_flask_restart() is False
        mock_timer.assert_not_called()

    def test_a_flask_process_arms_one_daemon_timer_with_the_requested_delay(
        self, monkeypatch, mock_timer
    ):
        monkeypatch.setenv('SERVICE_TYPE', 'FLASK')
        monkeypatch.setattr(config, 'DISABLE_FLASK_RESTART', False)

        assert restart_manager.schedule_flask_restart() is True
        mock_timer.assert_called_once_with(2.5, restart_manager._restart_flask_program)
        mock_timer.return_value.start.assert_called_once_with()


class TestTheDisableFlaskRestartEscapeHatchSuppressesTheRestart:
    def test_an_otherwise_eligible_flask_process_does_not_arm_the_timer(
        self, monkeypatch, mock_timer
    ):
        monkeypatch.setenv('SERVICE_TYPE', 'flask')
        monkeypatch.setattr(config, 'DISABLE_FLASK_RESTART', True)

        assert restart_manager.schedule_flask_restart() is False
        mock_timer.assert_not_called()

    def test_the_flag_is_read_live_from_config_so_a_reload_can_still_change_it(
        self, monkeypatch, mock_timer
    ):
        monkeypatch.setenv('SERVICE_TYPE', 'flask')
        monkeypatch.setattr(config, 'DISABLE_FLASK_RESTART', True)
        assert restart_manager.schedule_flask_restart() is False

        monkeypatch.setattr(config, 'DISABLE_FLASK_RESTART', False)
        assert restart_manager.schedule_flask_restart() is True


class TestPerContainerPlumbingNeverBecomesADatabaseWideGlobal:
    @pytest.mark.parametrize('name', [
        'AUDIO_MUSE_LISTENER_ID',
        'SUPERVISORCTL_CMD',
        'SUPERVISOR_CONF',
        'DISABLE_FLASK_RESTART',
    ])
    def test_it_is_neither_bootstrapped_into_app_config_nor_overridden_from_it(self, name):
        assert name in config.SETUP_BOOTSTRAP_EXCLUDED_KEYS, (
            '%s is per-container plumbing: one database-wide value collapses two worker '
            'containers on the same host onto a single ack row while pg_stat_activity '
            'still counts two listeners, so no handshake ever ends' % name
        )

    @pytest.mark.parametrize('name', [
        'AUDIO_MUSE_LISTENER_ID',
        'SUPERVISORCTL_CMD',
        'SUPERVISOR_CONF',
        'DISABLE_FLASK_RESTART',
    ])
    def test_the_setup_wizard_neither_renders_nor_accepts_it(self, name):
        assert name in app_setup.HIDDEN_ADVANCED_FIELDS
        assert app_setup.should_show_advanced(name) is False


def _setup_wizard_section_names():
    setup_js = Path(__file__).resolve().parents[2] / 'static' / 'setup.js'
    text = setup_js.read_text(encoding='utf-8')
    start = text.index('var ADVANCED_SECTIONS')
    end = text.index('var ADVANCED_OTHER_TITLE', start)
    body = text[start:end]
    names = []
    for items_block in re.findall(r'items:\s*\[(.*?)\]', body, re.S):
        names.extend(re.findall(r"'([A-Z][A-Z0-9_]{2,})'", items_block))
    return names


class TestSetupWizardParameterCatalogDoesNotDrift:
    def test_every_bootstrap_excluded_key_is_hidden_or_a_basic_field(self):
        excluded = set(config.SETUP_BOOTSTRAP_EXCLUDED_KEYS)
        assert excluded, 'the bootstrap-excluded set must never be empty'
        still_visible = sorted(
            name for name in excluded
            if name not in app_setup.BASIC_FIELDS
            and app_setup.should_show_advanced(name)
        )
        assert still_visible == [], (
            'these SETUP_BOOTSTRAP_EXCLUDED_KEYS are still rendered and accepted by '
            'the wizard, so an operator can save a value that is never applied and '
            'then pruned on the next boot: %s' % still_visible
        )

    def test_every_js_section_key_is_a_real_visible_config_field(self):
        names = _setup_wizard_section_names()
        assert names, 'static/setup.js ADVANCED_SECTIONS must not be empty'

        missing = sorted(name for name in names if not hasattr(config, name))
        assert missing == [], (
            'static/setup.js ADVANCED_SECTIONS references config keys that do not '
            'exist: %s' % missing
        )

        hidden = sorted(
            name for name in names if not app_setup.should_show_advanced(name)
        )
        assert hidden == [], (
            'static/setup.js ADVANCED_SECTIONS lists keys the wizard hides '
            'server-side, so those rows silently render nothing: %s' % hidden
        )

        basic = sorted(name for name in names if name in app_setup.BASIC_FIELDS)
        assert basic == [], (
            'static/setup.js ADVANCED_SECTIONS claims basic fields. /api/setup '
            'routes those to basic_fields, so the advanced section never receives '
            'them and renders nothing, exactly like a hidden key: %s' % basic
        )

    @pytest.mark.parametrize('name', [
        'SIMILARITY_DEFAULT_N_RESULTS',
        'ARTIST_SIMILARITY_DEFAULT_N_RESULTS',
        'CLAP_SEARCH_DEFAULT_LIMIT',
        'LYRICS_AXES_DEFAULT_LIMIT',
        'LYRICS_TEXT_DEFAULT_LIMIT',
        'SEM_GROVE_DEFAULT_LIMIT',
    ])
    def test_a_wizard_hidden_api_default_is_also_bootstrap_excluded(self, name):
        assert hasattr(config, name)
        assert app_setup.should_show_advanced(name) is False
        assert name in config.SETUP_BOOTSTRAP_EXCLUDED_KEYS, (
            '%s is hidden from the wizard but still mirrored into app_config, so the '
            'first boot freezes it in the database where no operator can reach it: '
            'the wizard will not render it and the DB row outranks the environment '
            'variable that is now its only remaining input' % name
        )

    def test_js_sections_list_each_field_only_once(self):
        names = _setup_wizard_section_names()
        assert names, 'static/setup.js ADVANCED_SECTIONS must not be empty'
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert duplicates == [], (
            'static/setup.js ADVANCED_SECTIONS lists the same field in more than one '
            'section: %s' % duplicates
        )


class TestTheRecordedActionIsDecodedFromEitherDetailsShape:
    def test_a_legacy_jsonb_dict_still_rejects_another_actions_request(self):
        conn = _conn(({'action': 'stop'},))

        assert restart_manager._action_matches(conn, 'control-1', 'restart') is False

    def test_a_text_json_details_column_rejects_another_actions_request(self):
        conn = _conn(('{"action": "stop"}',))

        assert restart_manager._action_matches(conn, 'control-2', 'restart') is False

    @pytest.mark.parametrize('row', [
        ({'action': 'restart'},),
        ('{"action": "restart"}',),
        ({},),
        (None,),
        None,
    ])
    def test_a_matching_or_unrecorded_action_is_accepted(self, row):
        conn = _conn(row)

        assert restart_manager._action_matches(conn, 'control-3', 'restart') is True

    @pytest.mark.parametrize('row', [
        ('["restart"]',),
        ('"restart"',),
        ('42',),
        ('true',),
        ([{'action': 'stop'}],),
        ('not json at all',),
    ])
    def test_a_details_payload_that_is_not_an_object_assumes_a_match_rather_than_raising(
        self, row
    ):
        conn = _conn(row)

        assert restart_manager._action_matches(conn, 'control-7', 'restart') is True

    def test_an_unreadable_row_assumes_a_match_rather_than_dropping_the_verdict(self):
        conn = MagicMock()
        conn.cursor.side_effect = RuntimeError('connection went away')

        assert restart_manager._action_matches(conn, 'control-4', 'restart') is True


class TestOneControlResultPollOpensOneConnection:
    def test_the_action_check_and_the_status_read_share_a_single_connection(
        self, monkeypatch
    ):
        import database

        conn = _conn(({'action': 'restart'},))
        opened = []

        def connect_raw(*_a, **_k):
            opened.append(conn)
            return conn

        monkeypatch.setattr(database, 'connect_raw', connect_raw)
        monkeypatch.setattr(
            'taskqueue.control.get_control_request_result',
            lambda request_id, conn=None: conn is not None,
        )

        assert restart_manager.get_control_request_result('restart', 'control-5') is True
        assert len(opened) == 1
        conn.close.assert_called_once_with()

    def test_a_mismatched_action_closes_the_connection_and_answers_false(
        self, monkeypatch
    ):
        import database

        conn = _conn(({'action': 'stop'},))
        monkeypatch.setattr(database, 'connect_raw', lambda *_a, **_k: conn)

        assert restart_manager.get_control_request_result('restart', 'control-6') is False
        conn.close.assert_called_once_with()
