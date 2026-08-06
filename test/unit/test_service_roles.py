# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The supervised service names have exactly one definition, everywhere.

A control request names a service; the native supervisors map it to a role and
the container maps it to a supervisord program. When those tables were
maintained as six hand-edited copies, a rename made ``dispatch_control`` answer
False for an unknown name and every native restart, stop and start silently did
nothing - no exception, no log. The Python copies are now one import, and these
tests cover the two consumers that CANNOT import Python: supervisord.conf and
docker-entrypoint.sh.

Main Features:
* supervisord.conf's [program:...] blocks are exactly the shared boot order
* Each supervisord worker program is launched with its role's queue name
* docker-entrypoint.sh starts exactly the shared worker and flask service lists
* restart_manager and all three native supervisors read the shared objects
* Every role in the table is dispatchable, and the listener survives a stop
"""

import importlib.util
import os
import re
import sys

import pytest

import restart_manager
import service_roles

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
NATIVE_BUILD = os.path.join(REPO_ROOT, 'native-build')

SUPERVISORD_CONF = os.path.join(REPO_ROOT, 'deployment', 'supervisord.conf')
ENTRYPOINT = os.path.join(REPO_ROOT, 'deployment', 'docker-entrypoint.sh')

PLATFORMS = ['linux', 'macos', 'windows']


def _read(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return handle.read()


def _load_supervisor(platform_name):
    for entry in (REPO_ROOT, NATIVE_BUILD):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    mod_name = 'service_roles_supervisor_' + platform_name
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = os.path.join(NATIVE_BUILD, platform_name, 'supervisor.py')
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        pytest.skip(f"{platform_name} supervisor does not import here: {exc!r}")
    return mod


class TestSupervisordConfMatchesTheSharedTable:
    def test_the_program_blocks_are_exactly_the_shared_boot_order(self):
        programs = re.findall(r'^\[program:([^\]]+)\]', _read(SUPERVISORD_CONF), re.MULTILINE)

        assert sorted(programs) == sorted(service_roles.BOOT_ORDER)

    def test_every_worker_program_is_launched_with_its_roles_queue_name(self):
        text = _read(SUPERVISORD_CONF)
        for service, role in service_roles.ROLE_OF.items():
            queue = service_roles.QUEUE_OF_ROLE.get(role)
            if queue is None:
                continue
            block = re.search(
                r'^\[program:' + re.escape(service) + r'\](.*?)(?=^\[|\Z)',
                text,
                re.MULTILINE | re.DOTALL,
            )
            assert block is not None
            assert f'--queue {queue}' in block.group(1)

    def test_the_maintenance_program_runs_the_shared_maintenance_module(self):
        assert f'-m {service_roles.MAINTENANCE_MODULE}' in _read(SUPERVISORD_CONF)

    def test_the_worker_programs_run_the_shared_worker_module(self):
        assert f'-m {service_roles.WORKER_MODULE}' in _read(SUPERVISORD_CONF)


class TestTheEntrypointStartsTheSharedServices:
    def test_the_worker_branch_starts_exactly_the_shared_worker_services(self):
        line = re.search(r'^\s*run_supervisorctl_checked start (queue-.*)$',
                         _read(ENTRYPOINT), re.MULTILINE)

        assert line is not None
        assert sorted(line.group(1).split()) == sorted(service_roles.WORKER_SERVICES)

    def test_the_flask_branch_starts_exactly_the_shared_flask_services(self):
        text = _read(ENTRYPOINT)
        started = re.findall(r'^\s*run_supervisorctl_checked start (.*)$', text, re.MULTILINE)
        flask_lines = [
            parts for parts in (line.split() for line in started)
            if parts == service_roles.FLASK_SERVICES
        ]

        assert flask_lines


class TestThePythonConsumersReadTheSharedTable:
    def test_restart_manager_uses_the_shared_worker_services(self):
        assert restart_manager.WORKER_SERVICES is service_roles.WORKER_SERVICES

    def test_restart_manager_uses_the_shared_flask_services(self):
        assert restart_manager.FLASK_SERVICE is service_roles.FLASK_SERVICES

    @pytest.mark.parametrize('platform_name', PLATFORMS)
    def test_each_native_supervisor_uses_the_shared_role_map(self, platform_name):
        mod = _load_supervisor(platform_name)

        assert mod.ROLE_OF is service_roles.ROLE_OF
        assert mod.BOOT_ORDER is service_roles.BOOT_ORDER


class TestTheTableIsInternallyConsistent:
    def test_the_restart_listener_is_never_stopped_with_the_workers(self):
        assert service_roles.SERVICE_RESTART_LISTENER not in service_roles.WORKER_SERVICES

    def test_the_restart_listener_still_gets_the_worker_environment(self):
        assert service_roles.ROLE_RESTART_LISTENER in service_roles.WORKER_ROLES

    def test_flask_is_never_treated_as_a_worker(self):
        assert service_roles.ROLE_FLASK not in service_roles.WORKER_ROLES
        assert service_roles.SERVICE_FLASK not in service_roles.WORKER_SERVICES

    def test_every_worker_service_is_a_known_service(self):
        for service in service_roles.WORKER_SERVICES + service_roles.FLASK_SERVICES:
            assert service in service_roles.ROLE_OF

    def test_every_role_in_the_table_is_dispatchable(self, monkeypatch):
        dispatched = []
        monkeypatch.setattr(
            service_roles.runpy, 'run_module',
            lambda module, **_kwargs: dispatched.append(module),
        )
        monkeypatch.setattr(service_roles.sys, 'argv', ['launcher', '--role=x'])

        for role in service_roles.ROLE_OF.values():
            if role == service_roles.ROLE_RESTART_LISTENER:
                continue
            service_roles.run_role(role, lambda: dispatched.append('flask'))

        assert 'flask' in dispatched
        assert dispatched.count(service_roles.WORKER_MODULE) == 2
        assert service_roles.MAINTENANCE_MODULE in dispatched

    def test_an_unknown_role_is_refused_rather_than_ignored(self, monkeypatch):
        monkeypatch.setattr(service_roles.sys, 'argv', ['launcher'])

        with pytest.raises(SystemExit):
            service_roles.run_role('not-a-role', lambda: None)

    def test_a_worker_role_is_handed_its_queue_on_argv(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            service_roles.runpy, 'run_module',
            lambda module, **_kwargs: seen.update(argv=list(service_roles.sys.argv)),
        )
        monkeypatch.setattr(service_roles.sys, 'argv', ['launcher', '--role=worker-high'])

        service_roles.run_role(service_roles.ROLE_WORKER_HIGH, lambda: None)

        assert seen['argv'][1:] == ['--queue', service_roles.QUEUE_OF_ROLE['worker-high']]
