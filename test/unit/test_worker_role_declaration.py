# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""A queue-side entrypoint declares its role before it imports config.

``config`` decides at IMPORT time whether it is running Flask-side, and when it
decides that it is, it bootstraps the schema. An entrypoint that imports config
before saying it is a worker therefore runs Flask's DDL from inside a worker
container, and nothing anywhere raises. That makes the ordering a property of
the SOURCE, not of any behaviour a runtime test could observe, so it is read
back out of the module's own statements here: a future tidy-the-imports edit
that sorts the declaration below ``import config`` fails loudly instead.

The shim itself lives once, in ``service_roles``, because it had grown four
spellings across the queue entrypoints. It stays conditional by default because
Flask imports ``taskqueue.control`` to publish a control request and must keep
its own role; a real queue entrypoint forces it.

Whether the WORKER entrypoint forces it is a second, separate property, and it is
read out of the source for the same reason the ordering is: ``python -m
taskqueue.worker`` is nothing but a worker, so on bare metal, where nothing sets
SERVICE_TYPE, a conditional declaration there leaves the Flask role in place and
the worker runs Flask's schema DDL. Nothing raises and no runtime assertion in
this process can see it, because the role is consumed by config at ITS import.
The argument is therefore checked as an argument: a call that loses ``force=True``
fails here rather than in somebody's container.

Main Features:
* Every taskqueue module with a ``__main__`` block declares its role first
* The worker entrypoint declares it through the shared shim, not a local copy
* The worker entrypoint passes force=True, above the config import
* ``force=True`` declares the worker role with no SERVICE_TYPE in the environment
* Without force, that same bare-metal worker would keep the Flask role
* force outranks a SERVICE_TYPE that says otherwise
* Flask's role survives importing a queue module that carries the conditional shim
* config reads back the exact environment variable the shim writes
"""

import ast
import os

import pytest

import service_roles

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
QUEUE_DIR = os.path.join(REPO_ROOT, 'taskqueue')
WORKER_MODULE_PATH = os.path.join(QUEUE_DIR, 'worker.py')
CONFIG_PATH = os.path.join(REPO_ROOT, 'config.py')

DECLARATION_MARKERS = ('declare_worker_role', service_roles.ROLE_ENV)


def _read(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return handle.read()


def _statements(path):
    source = _read(path)
    lines = source.splitlines()
    body = ast.parse(source, filename=path).body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body, lines


def _source_of(stmt, lines):
    return '\n'.join(lines[stmt.lineno - 1:stmt.end_lineno])


def _is_main_block(stmt):
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == '__name__'
        and any(
            isinstance(value, ast.Constant) and value.value == '__main__'
            for value in test.comparators
        )
    )


def _role_declaration_line(body, lines):
    for stmt in body:
        text = _source_of(stmt, lines)
        if any(marker in text for marker in DECLARATION_MARKERS):
            return stmt.lineno
    return None


def _config_import_line(body):
    for stmt in body:
        if isinstance(stmt, ast.Import):
            if any(alias.name.split('.')[0] == 'config' for alias in stmt.names):
                return stmt.lineno
        elif isinstance(stmt, ast.ImportFrom):
            if not stmt.level and (stmt.module or '').split('.')[0] == 'config':
                return stmt.lineno
    return None


def _called_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ''


def _declaration_calls(path):
    tree = ast.parse(_read(path), filename=path)
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node.func) == 'declare_worker_role'
    ]


def _is_forced(call):
    forcing = [keyword.value for keyword in call.keywords if keyword.arg == 'force']
    forcing.extend(call.args[:1])
    return any(
        isinstance(value, ast.Constant) and value.value is True for value in forcing
    )


def _queue_entrypoints():
    found = []
    for name in sorted(os.listdir(QUEUE_DIR)):
        if not name.endswith('.py'):
            continue
        path = os.path.join(QUEUE_DIR, name)
        body, _lines = _statements(path)
        if any(_is_main_block(stmt) for stmt in body):
            found.append(name)
    return found


QUEUE_ENTRYPOINTS = _queue_entrypoints()


@pytest.fixture
def env(monkeypatch):
    fake = {}
    monkeypatch.setattr(service_roles.os, 'environ', fake)
    return fake


class TestQueueEntrypointsDeclareTheirRoleBeforeConfigIsImported:
    def test_the_entrypoint_scan_found_the_queue_modules_it_exists_to_check(self):
        assert 'worker.py' in QUEUE_ENTRYPOINTS
        assert len(QUEUE_ENTRYPOINTS) >= 3

    @pytest.mark.parametrize('module_name', QUEUE_ENTRYPOINTS)
    def test_a_queue_side_entrypoint_declares_its_role_before_it_imports_config(
        self, module_name
    ):
        body, lines = _statements(os.path.join(QUEUE_DIR, module_name))
        config_line = _config_import_line(body)
        declaration_line = _role_declaration_line(body, lines)

        assert config_line is not None
        assert declaration_line is not None, (
            f"taskqueue/{module_name} imports config without declaring its role first; "
            f"config bootstraps the schema at import time when it believes it is Flask-side"
        )
        assert declaration_line < config_line, (
            f"taskqueue/{module_name} declares its role at line {declaration_line}, "
            f"below the config import at line {config_line}"
        )

    def test_the_worker_entrypoint_forces_the_role_instead_of_trusting_service_type(self):
        calls = _declaration_calls(WORKER_MODULE_PATH)

        assert calls, 'taskqueue/worker.py never calls declare_worker_role'
        assert [call.lineno for call in calls if _is_forced(call)], (
            "taskqueue/worker.py declares its role conditionally; a bare-metal "
            "'python -m taskqueue.worker' with no SERVICE_TYPE in the environment then "
            "keeps the Flask role and runs Flask's schema DDL from inside a worker"
        )

    def test_that_forced_declaration_is_itself_above_the_config_import(self):
        body, _lines = _statements(WORKER_MODULE_PATH)
        config_line = _config_import_line(body)
        forced = [call.lineno for call in _declaration_calls(WORKER_MODULE_PATH)
                  if _is_forced(call)]

        assert config_line is not None
        assert forced
        assert min(forced) < config_line

    def test_the_worker_entrypoint_keeps_no_role_spelling_of_its_own(self):
        body, lines = _statements(WORKER_MODULE_PATH)
        local_copies = [
            stmt.lineno for stmt in body
            if service_roles.ROLE_ENV in _source_of(stmt, lines)
        ]

        assert not local_copies, (
            f"taskqueue/worker.py spells {service_roles.ROLE_ENV} itself at line(s) "
            f"{local_copies}; it must call service_roles.declare_worker_role instead"
        )


class TestTheSharedShimDecidesWhoIsAWorker:
    def test_a_queue_entrypoint_declares_the_worker_role_with_no_service_type_set(self, env):
        declared = service_roles.declare_worker_role(force=True)

        assert declared is True
        assert env[service_roles.ROLE_ENV] == service_roles.WORKER_ENV_VALUE

    def test_the_same_bare_metal_process_without_force_keeps_the_flask_role(self, env):
        assert service_roles.declare_worker_role(force=False) is False
        assert service_roles.ROLE_ENV not in env

        assert service_roles.declare_worker_role(force=True) is True
        assert env[service_roles.ROLE_ENV] == service_roles.WORKER_ENV_VALUE

    def test_force_outranks_a_service_type_that_says_this_process_is_flask(self, env):
        env[service_roles.SERVICE_TYPE_ENV] = 'flask'

        declared = service_roles.declare_worker_role(force=True)

        assert declared is True
        assert env[service_roles.ROLE_ENV] == service_roles.WORKER_ENV_VALUE

    def test_a_worker_service_type_declares_the_worker_role(self, env):
        env[service_roles.SERVICE_TYPE_ENV] = 'WORKER'

        declared = service_roles.declare_worker_role()

        assert declared is True
        assert env[service_roles.ROLE_ENV] == service_roles.WORKER_ENV_VALUE

    def test_flask_keeps_its_own_role_when_it_imports_a_queue_module(self, env):
        env[service_roles.SERVICE_TYPE_ENV] = 'flask'

        declared = service_roles.declare_worker_role()

        assert declared is False
        assert service_roles.ROLE_ENV not in env

    def test_an_absent_service_type_declares_nothing_on_its_own(self, env):
        declared = service_roles.declare_worker_role()

        assert declared is False
        assert service_roles.ROLE_ENV not in env

    def test_a_role_already_in_the_environment_is_left_exactly_as_it_is(self, env):
        env[service_roles.SERVICE_TYPE_ENV] = 'worker'
        env[service_roles.ROLE_ENV] = 'already-set'

        service_roles.declare_worker_role()

        assert env[service_roles.ROLE_ENV] == 'already-set'


class TestTheShimWritesWhatConfigReads:
    def test_config_reads_back_the_variable_and_the_value_the_shim_writes(self):
        reads = [
            line for line in _read(CONFIG_PATH).splitlines()
            if service_roles.ROLE_ENV in line
            and f"'{service_roles.WORKER_ENV_VALUE}'" in line
        ]

        assert reads, (
            f"config no longer decides worker mode from {service_roles.ROLE_ENV} == "
            f"{service_roles.WORKER_ENV_VALUE!r}; the shim and its one consumer have drifted"
        )
