# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every module the queue needs at run time is inside the frozen bundle.

The native builds (Windows, macOS, Linux) run the worker and maintenance through
runpy from service_roles, which PyInstaller cannot see, so AudioMuse-AI.spec names
those roots as hidden imports and PyInstaller follows their STATIC imports from
there. A module that only ever reaches the worker through a function-level import
is invisible to that walk and is simply absent from the bundle - the macOS build
lists numeric_bootstrap by hand for exactly that reason.

These tests walk the same eager import graph test_import_architecture pins, from
the roots the spec declares, and assert that the modules the retry contract added
are reachable. They fail on the day someone makes one of those imports lazy.

Main Features:
* The spec still names the queue roots as hidden imports
* taskqueue.retry, taskqueue.errors, task_types, queue_names and tasks.task_run
  are all reachable from those roots by static import alone
"""

import ast
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def _graph():
    path = pathlib.Path(__file__).with_name('test_import_architecture.py')
    spec = importlib.util.spec_from_file_location('test_import_architecture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._graph()

QUEUE_ROOTS = ('taskqueue', 'taskqueue.worker', 'taskqueue.maintenance', 'taskqueue.control')

NEEDED_AT_RUN_TIME = (
    'taskqueue.retry',
    'taskqueue.errors',
    'taskqueue.sql',
    'task_types',
    'queue_names',
    'tasks.task_run',
    'tasks.recovery',
)


def _spec_hidden_imports():
    tree = ast.parse((REPO / 'AudioMuse-AI.spec').read_text(encoding='utf-8'))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == 'hiddenimports' for t in node.targets):
            if isinstance(node.value, ast.List):
                return [
                    elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)
                ]
    return []


def _reachable(graph, roots):
    seen = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, ()))
    return seen


class TestTheSpecStillNamesTheQueueRoots:
    @pytest.mark.parametrize('root', QUEUE_ROOTS)
    def test_the_root_is_a_hidden_import(self, root):
        assert root in _spec_hidden_imports(), (
            f'{root} is launched through runpy, which PyInstaller cannot follow; '
            'without this entry the native builds ship a launcher with no queue'
        )


class TestEveryQueueModuleReachesTheBundle:
    @pytest.mark.parametrize('module', NEEDED_AT_RUN_TIME)
    def test_the_module_is_reachable_by_static_import_alone(self, module):
        modules, graph = _graph()
        roots = [r for r in _spec_hidden_imports() if r in modules]
        roots += [m for m in modules if m == 'tasks' or m.startswith('tasks.')]

        assert module in modules, f'{module} does not exist'
        assert module in _reachable(graph, roots), (
            f'{module} is only reachable through a function-level import, so '
            'PyInstaller leaves it out of the bundle and the frozen worker raises '
            'ModuleNotFoundError the first time it needs it'
        )
