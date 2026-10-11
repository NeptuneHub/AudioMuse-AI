# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Backend guards for the setup wizard's P2P Sharing section.

The section edits five settings through the normal wizard save and reads the
P2P node's state as one word. The view functions are called directly inside a
Flask request context with the database, the setup store and the restart
manager replaced by fakes.

Main Features:
* The five settings are hidden from the generic advanced list yet accepted on
  save, the two internals stay in the advanced list, and the env-only Tor
  program and directory relays are never wizard fields, never saved and never
  mirrored into app_config
* The removed network settings (listen port, public addresses, UPnP, bootstrap
  peers, the main network server address) and the reset identity route and
  button are gone for good
* Save validation: whole numbers, parallel 1 to 64, KiB/s 0 or more, and no
  enabling while p2p.available names a reason (the node libraries or the Tor
  program are missing), with that reason in the message; valid values are
  normalised before they are stored
* The availability check is a spec lookup plus a Tor program lookup that never
  imports the node libraries; a failing check reads as unavailable and is
  logged
* GET /api/setup hands out p2p_fields with every value plus available
* The status route answers 200 with only state and label in every state,
  including a stale row, an unknown word and a failing database, and never an
  fp_ id, an onion address or the exception text
"""

import os
import textwrap
from unittest.mock import MagicMock

import pytest
import yaml

import app_setup
from p2p import available as p2p_available
from p2p import ipc as p2p_ipc
from tasks.mediaserver import registry as mediaserver_registry

CONTRACT_FIELDS = [
    'P2P_ENABLED',
    'P2P_MAX_PARALLEL_DOWNLOADS',
    'P2P_MAX_PARALLEL_UPLOADS',
    'P2P_MAX_DOWNLOAD_KIBPS',
    'P2P_MAX_UPLOAD_KIBPS',
]
INTERNAL_FIELDS = [
    'P2P_MAX_PEERS',
    'P2P_LOOKUP_TIMEOUT_SECONDS',
]
REMOVED_FIELDS = [
    'P2P_LISTEN_PORT',
    'P2P_ANNOUNCE_ADDRS',
    'P2P_UPNP_ENABLED',
    'P2P_BOOTSTRAP_PEERS',
]
ENV_ONLY_FIELDS = [
    'TOR_BINARY',
    'P2P_DIRECTORY_RELAYS',
]
RESET_ROUTE = '/api/setup/p2p/reset-identity'
ONION_ID = 'statusrowonion' + 'q' * 42
MISSING_REASON = 'the Tor program (tor) is not installed'
REPO_ROOT = os.path.dirname(os.path.abspath(app_setup.__file__))


def _source():
    with open(app_setup.__file__, encoding='utf-8') as handle:
        return handle.read()


def _repo_file(*parts):
    with open(os.path.join(REPO_ROOT, *parts), encoding='utf-8') as handle:
        return handle.read()


@pytest.fixture
def p2p_present(monkeypatch):
    monkeypatch.setattr(app_setup.p2p_available, 'unavailable_reason', lambda: None)


@pytest.fixture
def p2p_missing(monkeypatch):
    monkeypatch.setattr(app_setup.p2p_available, 'unavailable_reason', lambda: MISSING_REASON)


class TestFieldCatalog:
    def test_the_section_fields_are_the_contract_list(self):
        assert app_setup.P2P_CONFIG_FIELDS == CONTRACT_FIELDS

    def test_every_section_field_exists_in_config(self):
        for field in CONTRACT_FIELDS + INTERNAL_FIELDS:
            assert hasattr(app_setup.config, field), field

    def test_the_section_fields_are_hidden_from_the_advanced_list(self):
        for field in CONTRACT_FIELDS:
            assert field in app_setup.HIDDEN_ADVANCED_FIELDS, field
            assert not app_setup.should_show_advanced(field), field

    def test_the_internals_stay_in_the_advanced_list(self):
        for field in INTERNAL_FIELDS:
            assert app_setup.should_show_advanced(field), field
            assert field not in app_setup.P2P_CONFIG_FIELDS

    def test_the_section_fields_are_accepted_on_save(self, monkeypatch):
        monkeypatch.setattr(app_setup.setup_manager, 'get_all_fields', lambda _config: [])
        assert set(CONTRACT_FIELDS) <= app_setup._get_allowed_setup_keys()
        assert 'allowed_keys.update(P2P_CONFIG_FIELDS)' in _source()

    @pytest.mark.parametrize('field', ENV_ONLY_FIELDS)
    def test_the_tor_program_and_the_relays_are_never_wizard_fields(self, monkeypatch, field):
        monkeypatch.setattr(app_setup.setup_manager, 'get_all_fields',
                            lambda _config: [{'name': field, 'value': 'x'}])
        assert hasattr(app_setup.config, field)
        assert field in app_setup.HIDDEN_ADVANCED_FIELDS
        assert not app_setup.should_show_advanced(field)
        assert field not in app_setup.P2P_CONFIG_FIELDS
        assert field not in app_setup._get_allowed_setup_keys()

    @pytest.mark.parametrize('field', ENV_ONLY_FIELDS)
    def test_the_tor_program_and_the_relays_are_never_mirrored_into_app_config(self, field):
        assert field in app_setup.config.SETUP_BOOTSTRAP_EXCLUDED_KEYS
        assert field not in app_setup.setup_manager._get_env_config_values(app_setup.config)


class TestRemovedNetworkSettings:
    def test_the_main_network_server_address_is_gone(self):
        assert not hasattr(app_setup.config, 'P2P_MAIN_SERVER')
        assert 'P2P_MAIN_SERVER' not in app_setup.HIDDEN_ADVANCED_FIELDS
        assert 'P2P_MAIN_SERVER' not in app_setup.config.SETUP_BOOTSTRAP_EXCLUDED_KEYS
        assert 'P2P_MAIN_SERVER' not in app_setup.P2P_CONFIG_FIELDS

    def test_the_network_settings_and_the_identity_reset_are_gone(self):
        template = _repo_file('templates', 'setup.html')
        script = _repo_file('static', 'setup.js')
        rules = {rule.rule for rule in app_setup.app.url_map.iter_rules()}
        for field in REMOVED_FIELDS:
            assert field not in app_setup.P2P_CONFIG_FIELDS, field
            assert field not in template and field not in script, field
        assert RESET_ROUTE not in rules
        assert not hasattr(app_setup, 'setup_p2p_reset_identity')
        for marker in (RESET_ROUTE, 'p2p-reset-identity', 'Reset P2P identity', 'P2P_RESET_CONFIRM'):
            assert marker not in template and marker not in script, marker


class TestNumbers:
    @pytest.mark.parametrize('key', ['P2P_MAX_PARALLEL_DOWNLOADS', 'P2P_MAX_PARALLEL_UPLOADS'])
    @pytest.mark.parametrize('value', ['0', '65', '-1', 'abc', '4.5', '', True, 4.0, None])
    def test_parallel_settings_outside_one_to_sixty_four_are_refused(self, key, value):
        problem = app_setup._validate_p2p_values({key: value})
        assert problem and '1 to 64' in problem

    @pytest.mark.parametrize('value,expected', [('1', 1), (' 8 ', 8), ('64', 64), (16, 16)])
    def test_parallel_settings_are_stored_as_integers(self, value, expected):
        values = {'P2P_MAX_PARALLEL_UPLOADS': value}
        assert app_setup._validate_p2p_values(values) is None
        assert values['P2P_MAX_PARALLEL_UPLOADS'] == expected

    @pytest.mark.parametrize('key', ['P2P_MAX_DOWNLOAD_KIBPS', 'P2P_MAX_UPLOAD_KIBPS'])
    @pytest.mark.parametrize('value', ['-1', 'fast', '1.5', ''])
    def test_negative_or_non_integer_caps_are_refused(self, key, value):
        problem = app_setup._validate_p2p_values({key: value})
        assert problem and 'KiB/s' in problem and '0 means unlimited' in problem

    @pytest.mark.parametrize('value,expected', [('0', 0), ('256', 256), ('1000000', 1000000)])
    def test_caps_accept_zero_and_any_positive_number(self, value, expected):
        values = {'P2P_MAX_DOWNLOAD_KIBPS': value}
        assert app_setup._validate_p2p_values(values) is None
        assert values['P2P_MAX_DOWNLOAD_KIBPS'] == expected

    @pytest.mark.parametrize('value', ['0', '0.5', '121', '-3', 'soon', 'nan', 'inf', ''])
    def test_a_lookup_timeout_the_node_would_drop_is_refused(self, value):
        problem = app_setup._validate_p2p_values({'P2P_LOOKUP_TIMEOUT_SECONDS': value})
        assert problem and 'from 1 to 120' in problem

    @pytest.mark.parametrize('value, expected', [('1', 1.0), ('5', 5.0), ('2.5', 2.5), ('120', 120.0)])
    def test_a_lookup_timeout_inside_the_node_limits_is_stored_as_seconds(self, value, expected):
        values = {'P2P_LOOKUP_TIMEOUT_SECONDS': value}
        assert app_setup._validate_p2p_values(values) is None
        assert values['P2P_LOOKUP_TIMEOUT_SECONDS'] == expected

    def test_the_wizard_and_the_node_share_the_lookup_limits(self):
        assert (p2p_ipc.MIN_REQUEST_TTL_SECONDS, p2p_ipc.MAX_REQUEST_TTL_SECONDS) == (1.0, 120.0)


class TestSwitches:
    def test_a_switch_that_is_not_a_boolean_is_refused(self):
        problem = app_setup._validate_p2p_values({'P2P_ENABLED': 'maybe'})
        assert problem and 'true or false' in problem

    def test_enabling_is_refused_when_p2p_cannot_run(self, p2p_missing):
        problem = app_setup._validate_p2p_values({'P2P_ENABLED': True})
        assert problem == app_setup.P2P_UNAVAILABLE_MESSAGE % MISSING_REASON
        assert MISSING_REASON in problem
        assert 'Tor' in problem and 'noavx2' in problem and '%s' not in problem

    def test_disabling_is_allowed_when_p2p_cannot_run(self, p2p_missing):
        assert app_setup._validate_p2p_values({'P2P_ENABLED': False}) is None

    def test_enabling_is_allowed_when_p2p_can_run(self, p2p_present):
        assert app_setup._validate_p2p_values({'P2P_ENABLED': True}) is None

    def test_untouched_fields_pass(self, p2p_missing):
        assert app_setup._validate_p2p_values({'OTHER': 'x'}) is None

    def test_missing_libraries_are_named_in_the_message(self, monkeypatch):
        monkeypatch.setattr(p2p_available, 'missing_modules', lambda: ['stem', 'trio_websocket'])

        problem = app_setup._validate_p2p_values({'P2P_ENABLED': True})

        assert problem == app_setup.P2P_UNAVAILABLE_MESSAGE % (
            'this build lacks the Python libraries stem, trio_websocket')

    def test_a_missing_tor_program_is_named_in_the_message(self, monkeypatch):
        monkeypatch.setattr(p2p_available, 'missing_modules', lambda: [])
        monkeypatch.setattr(app_setup.config, 'TOR_BINARY', 'no-such-tor-program-for-tests')

        problem = app_setup._validate_p2p_values({'P2P_ENABLED': True})

        assert 'the Tor program (no-such-tor-program-for-tests) is not installed' in problem

    def test_availability_is_a_spec_lookup_that_never_imports(self, monkeypatch):
        seen = []
        monkeypatch.setattr(p2p_available.importlib.util, 'find_spec', lambda name: seen.append(name))
        assert app_setup._p2p_available() is False
        assert seen == list(p2p_available.REQUIRED_MODULES)
        assert app_setup._p2p_unavailable_reason() == (
            'this build lacks the Python libraries ' + ', '.join(p2p_available.REQUIRED_MODULES))

    def test_a_broken_check_reads_as_unavailable_and_is_logged(self, monkeypatch):
        def broken():
            raise OSError('permission denied')

        monkeypatch.setattr(app_setup.p2p_available, 'unavailable_reason', broken)
        monkeypatch.setattr(app_setup.app.logger, 'exception', MagicMock())
        assert app_setup._p2p_available() is False
        assert app_setup._p2p_unavailable_reason() == 'the check for Tor failed'
        assert app_setup.app.logger.exception.called


def _save(monkeypatch, config_values, fields=()):
    monkeypatch.setattr(app_setup.setup_manager, 'get_all_fields',
                        lambda _config: [{'name': name, 'value': ''} for name in fields])
    monkeypatch.setattr(app_setup, '_has_admin_user', lambda: True)
    monkeypatch.setattr(app_setup.config, 'AUTH_ENABLED', False)
    monkeypatch.setattr(app_setup.config, 'MEDIASERVER_TYPE', 'jellyfin')
    monkeypatch.setattr(app_setup.config, 'MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE', {'jellyfin': []})
    monkeypatch.setattr(app_setup.config, 'MEDIASERVER_FIELDS_BY_TYPE', {'jellyfin': []})
    monkeypatch.setattr(app_setup.config, 'MEDIASERVER_CONFIG_KEYS', set())
    monkeypatch.setattr(app_setup.config, 'MUSIC_LIBRARIES', '')
    monkeypatch.setattr(app_setup.setup_manager, '_is_valid_server_config', lambda _cfg: True)
    monkeypatch.setattr(app_setup.setup_manager, 'delete_config_values', MagicMock())
    save = MagicMock()
    monkeypatch.setattr(app_setup.setup_manager, 'save_config_values', save)
    monkeypatch.setattr(app_setup.config, 'refresh_config', MagicMock())
    monkeypatch.setattr(mediaserver_registry, 'save_default_server_settings', MagicMock())
    monkeypatch.setattr(app_setup.restart_manager, 'publish_restart_request', MagicMock(return_value=True))
    monkeypatch.setattr(app_setup.restart_manager, 'schedule_flask_restart', MagicMock(return_value=True))

    with app_setup.app.test_request_context('/api/setup', method='POST', json={'config': config_values}):
        response = app_setup.setup_api()
    if isinstance(response, tuple):
        response, status = response[0], response[1]
    else:
        status = response.status_code
    return response, status, save


class TestSaveRoute:
    def test_an_invalid_value_is_refused_before_anything_is_stored(self, monkeypatch, p2p_present):
        response, status, save = _save(monkeypatch, {'P2P_MAX_PARALLEL_UPLOADS': '0'})
        assert status == 400
        assert '1 to 64' in response.get_json()['error']
        save.assert_not_called()

    def test_enabling_while_p2p_cannot_run_is_refused_with_the_reason(self, monkeypatch, p2p_missing):
        response, status, save = _save(monkeypatch, {'P2P_ENABLED': 'true'})
        assert status == 400
        assert response.get_json()['error'] == app_setup.P2P_UNAVAILABLE_MESSAGE % MISSING_REASON
        save.assert_not_called()

    def test_valid_values_are_stored_normalised(self, monkeypatch, p2p_present):
        response, status, save = _save(monkeypatch, {
            'P2P_ENABLED': 'true',
            'P2P_MAX_PARALLEL_DOWNLOADS': '8',
            'P2P_MAX_PARALLEL_UPLOADS': ' 2 ',
            'P2P_MAX_DOWNLOAD_KIBPS': '512',
            'P2P_MAX_UPLOAD_KIBPS': '0',
        })
        assert status == 200
        stored = save.call_args[0][0]
        assert stored == {
            'P2P_ENABLED': True,
            'P2P_MAX_PARALLEL_DOWNLOADS': 8,
            'P2P_MAX_PARALLEL_UPLOADS': 2,
            'P2P_MAX_DOWNLOAD_KIBPS': 512,
            'P2P_MAX_UPLOAD_KIBPS': 0,
        }

    def test_a_removed_network_setting_is_dropped_on_save(self, monkeypatch, p2p_present):
        response, status, save = _save(monkeypatch, {
            'P2P_MAX_PARALLEL_DOWNLOADS': '8',
            'P2P_LISTEN_PORT': '4102',
            'P2P_BOOTSTRAP_PEERS': '/dnsaddr/bootstrap.example.org',
            'P2P_MAIN_SERVER': '/ip4/203.0.113.7/tcp/4101',
        })
        assert status == 200
        assert save.call_args[0][0] == {'P2P_MAX_PARALLEL_DOWNLOADS': 8}

    def test_the_tor_program_and_the_relays_are_never_saved(self, monkeypatch, p2p_present):
        response, status, save = _save(monkeypatch, {
            'P2P_MAX_PARALLEL_DOWNLOADS': '8',
            'TOR_BINARY': '/opt/another/tor',
            'P2P_DIRECTORY_RELAYS': 'relay.example.org',
        }, fields=ENV_ONLY_FIELDS)
        assert status == 200
        assert save.call_args[0][0] == {'P2P_MAX_PARALLEL_DOWNLOADS': 8}


def _get_payload(monkeypatch):
    monkeypatch.setattr(app_setup.setup_manager, 'get_all_fields', lambda _config: [])
    monkeypatch.setattr(app_setup.setup_manager, 'get_raw_overrides', lambda ensure_table=True: {})
    monkeypatch.setattr(app_setup, 'check_setup_needed', lambda: False)
    monkeypatch.setattr(app_setup, '_has_admin_user', lambda: True)
    monkeypatch.setattr(app_setup, 'model_coverage_levels', lambda: {})
    with app_setup.app.test_request_context('/api/setup', method='GET'):
        response = app_setup.setup_api()
    return response.get_json()


class TestSetupPayload:
    def test_every_setting_and_the_availability_flag_are_handed_out(self, monkeypatch, p2p_present):
        fields = _get_payload(monkeypatch)['p2p_fields']
        assert set(fields) == set(CONTRACT_FIELDS) | {'available'}
        assert fields['available'] is True
        assert all(isinstance(fields[key], str) for key in CONTRACT_FIELDS)

    def test_availability_follows_the_p2p_check(self, monkeypatch, p2p_missing):
        assert _get_payload(monkeypatch)['p2p_fields']['available'] is False

    def test_values_are_rendered_for_the_form(self, monkeypatch, p2p_present):
        monkeypatch.setattr(app_setup.config, 'P2P_ENABLED', False)
        monkeypatch.setattr(app_setup.config, 'P2P_MAX_DOWNLOAD_KIBPS', 1024)
        monkeypatch.setattr(app_setup.config, 'P2P_MAX_PARALLEL_UPLOADS', 4)
        fields = _get_payload(monkeypatch)['p2p_fields']
        assert fields['P2P_ENABLED'] == 'false'
        assert fields['P2P_MAX_DOWNLOAD_KIBPS'] == '1024'
        assert fields['P2P_MAX_PARALLEL_UPLOADS'] == '4'

    def test_the_internals_are_not_in_the_section_payload(self, monkeypatch, p2p_present):
        fields = _get_payload(monkeypatch)['p2p_fields']
        for field in INTERNAL_FIELDS:
            assert field not in fields


class _Cursor:
    def __init__(self, row, error):
        self.row = row
        self.error = error
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.error:
            raise self.error
        self.executed.append((sql, params))

    def fetchone(self):
        return self.row


class _Db:
    def __init__(self, row=None, error=None):
        self.cur = _Cursor(row, error)
        self.rollbacks = 0

    def cursor(self):
        return self.cur

    def rollback(self):
        self.rollbacks += 1


def _status(monkeypatch, enabled=True, row=None, error=None, db_error=None):
    monkeypatch.setattr(app_setup.config, 'P2P_ENABLED', enabled)
    db = _Db(row, error)
    opened = []

    def get_db():
        opened.append(True)
        if db_error:
            raise db_error
        return db

    monkeypatch.setattr(app_setup.database, 'get_db', get_db)
    with app_setup.app.test_request_context('/api/setup/p2p/status', method='GET'):
        response, status = app_setup.setup_p2p_status()
    body = response.get_data(as_text=True)
    assert status == 200
    assert set(response.get_json()) == {'state', 'label'}
    assert 'fp_' not in body and 'Traceback' not in body and ONION_ID not in body
    assert response.headers['Cache-Control'] == 'no-store'
    return response.get_json(), db, opened


def _row(state, age=5.0, **extra):
    status = {'state': state, 'reason': 'fp_0123 test', 'peer_id': ONION_ID}
    status.update(extra)
    return (p2p_ipc.dumps(status), age)


class TestStatusRoute:
    def test_off_never_touches_the_database(self, monkeypatch, p2p_present):
        body, _db, opened = _status(monkeypatch, enabled=False)
        assert body == {'state': 'off', 'label': 'Off'}
        assert not opened

    def test_a_build_that_cannot_run_p2p_is_unavailable(self, monkeypatch, p2p_missing):
        body, _db, opened = _status(monkeypatch)
        assert body == {'state': 'unavailable', 'label': 'Unavailable'}
        assert not opened

    def test_no_status_row_is_not_running(self, monkeypatch, p2p_present):
        body, db, _opened = _status(monkeypatch, row=None)
        assert body == {'state': 'not_running', 'label': 'Not running'}
        assert db.rollbacks == 1

    def test_a_fresh_connected_row_is_connected_and_hides_the_peer_id(self, monkeypatch, p2p_present):
        body, db, _opened = _status(monkeypatch, row=_row('connected', served=12))
        assert body == {'state': 'connected', 'label': 'Connected'}
        assert db.cur.executed and db.cur.executed[0][1] == (p2p_ipc.STATUS_KEY,)
        assert db.rollbacks == 1

    @pytest.mark.parametrize('state,label', [
        ('starting', 'Starting'),
        ('no_peers', 'No peers found'),
        ('error', 'Error'),
    ])
    def test_every_node_word_is_passed_through(self, monkeypatch, p2p_present, state, label):
        body, _db, _opened = _status(monkeypatch, row=_row(state))
        assert body == {'state': state, 'label': label}

    def test_a_stale_row_is_not_running(self, monkeypatch, p2p_present):
        body, _db, _opened = _status(monkeypatch, row=_row('connected', age=p2p_ipc.STATUS_FRESH_SECONDS + 60))
        assert body == {'state': 'not_running', 'label': 'Not running'}

    @pytest.mark.parametrize('state', ['weird', ['connected'], None])
    def test_an_unknown_word_is_not_running(self, monkeypatch, p2p_present, state):
        body, _db, _opened = _status(monkeypatch, row=_row(state))
        assert body == {'state': 'not_running', 'label': 'Not running'}

    def test_a_failing_query_is_an_error_word_and_logged(self, monkeypatch, p2p_present):
        logged = MagicMock()
        monkeypatch.setattr(app_setup.app.logger, 'exception', logged)
        body, db, _opened = _status(monkeypatch, error=RuntimeError('secret detail fp_0042'))
        assert body == {'state': 'error', 'label': 'Error'}
        assert logged.called
        assert db.rollbacks == 1

    def test_a_missing_database_is_an_error_word(self, monkeypatch, p2p_present):
        logged = MagicMock()
        monkeypatch.setattr(app_setup.app.logger, 'exception', logged)
        body, _db, _opened = _status(monkeypatch, db_error=RuntimeError('secret detail fp_0042'))
        assert body == {'state': 'error', 'label': 'Error'}
        assert logged.called


class TestRouteDocs:
    def test_the_status_route_has_a_valid_flasgger_block(self):
        doc = app_setup.setup_p2p_status.__doc__
        spec = yaml.safe_load(textwrap.dedent(doc.split('---', 1)[1]))
        assert spec['tags'] == ['Setup']
        assert 200 in spec['responses']

    def test_the_status_route_is_registered_under_the_setup_prefix(self):
        rules = {rule.rule: rule.methods for rule in app_setup.app.url_map.iter_rules()}
        assert 'GET' in rules['/api/setup/p2p/status']
        assert not [rule for rule in rules if rule.startswith('/api/setup/p2p/') and rule != '/api/setup/p2p/status']
