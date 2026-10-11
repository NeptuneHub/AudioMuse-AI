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
* The P2P node roots (p2p.node, and p2p.runtime it imports lazily) are hidden
  imports; p2p, stem, trio and trio_websocket are collected whole and
  coincurve's shared library is bundled, so every library p2p.available
  requires and every third-party module the p2p package imports is in the
  bundle
* libp2p, multiaddr and miniupnpc are gone: no p2p module imports them and
  the spec neither collects, hides nor copies the metadata of any of them
* The model-space modules the node reads (tasks.lyrics_constants,
  tasks.nfp_format) sit in the collected tasks package
"""

import ast
import importlib.util
import pathlib
import sys

import pytest

from p2p import available as p2p_available

REPO = pathlib.Path(__file__).resolve().parents[2]


def _graph():
    path = pathlib.Path(__file__).with_name('test_import_architecture.py')
    spec = importlib.util.spec_from_file_location('test_import_architecture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._graph()

QUEUE_ROOTS = ('taskqueue', 'taskqueue.worker', 'taskqueue.maintenance', 'taskqueue.control')

P2P_ROOTS = ('p2p.node', 'p2p.runtime')

P2P_COLLECTED_PACKAGES = ('p2p', 'stem', 'trio', 'trio_websocket')

P2P_BUNDLED_LIBRARIES = ('coincurve',)

STATICALLY_REACHED = ('numpy', 'psycopg2')

REMOVED_P2P_PACKAGES = ('libp2p', 'multiaddr', 'miniupnpc')

P2P_NODE_TASK_MODULES = ('tasks.lyrics_constants', 'tasks.nfp_format')

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


def _spec_tree():
    return ast.parse((REPO / 'AudioMuse-AI.spec').read_text(encoding='utf-8'))


def _calls_named(name):
    return [
        node for node in ast.walk(_spec_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _loop_packages_feeding(name):
    packages = []
    for node in ast.walk(_spec_tree()):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Tuple):
            continue
        feeds = any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
            for stmt in node.body for call in ast.walk(stmt)
        )
        if feeds:
            packages.extend(
                elt.value for elt in node.iter.elts if isinstance(elt, ast.Constant)
            )
    return packages


def _collected_packages(name):
    direct = [
        call.args[0].value for call in _calls_named(name)
        if call.args and isinstance(call.args[0], ast.Constant)
    ]
    return direct + _loop_packages_feeding(name)


def _p2p_top_level_imports():
    names = set()
    for path in (REPO / 'p2p').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split('.')[0])
            elif isinstance(node, ast.Import):
                names.update(alias.name.split('.')[0] for alias in node.names)
    return names


def _third_party(names):
    return {
        name for name in names
        if name not in sys.stdlib_module_names
        and not (REPO / name).is_dir() and not (REPO / f'{name}.py').is_file()
    }


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


class TestTheSpecBundlesTheP2PNode:
    @pytest.mark.parametrize('root', P2P_ROOTS)
    def test_the_node_root_is_a_hidden_import(self, root):
        assert root in _spec_hidden_imports(), (
            f'{root} is reached only through runpy or a lazy import, so without this '
            'entry the frozen --role=p2p-node child raises ModuleNotFoundError'
        )

    @pytest.mark.parametrize('package', P2P_COLLECTED_PACKAGES)
    def test_the_package_is_collected_whole(self, package):
        assert package in _collected_packages('collect_submodules'), (
            f'{package} loads submodules dynamically (stem controllers, trio '
            'backends); the static walk alone leaves some of them out of the bundle'
        )

    @pytest.mark.parametrize('library', P2P_BUNDLED_LIBRARIES)
    def test_the_shared_library_is_bundled(self, library):
        assert library in _collected_packages('collect_dynamic_libs'), (
            f'{library} ships a C library the static walk does not copy on every platform'
        )

    def test_every_library_the_node_requires_is_in_the_bundle(self):
        bundled = set(_collected_packages('collect_submodules')) | set(_collected_packages('collect_dynamic_libs'))

        assert set(p2p_available.REQUIRED_MODULES) <= bundled, sorted(
            set(p2p_available.REQUIRED_MODULES) - bundled
        )

    def test_every_third_party_module_the_p2p_package_imports_is_bundled(self):
        bundled = set(P2P_COLLECTED_PACKAGES) | set(P2P_BUNDLED_LIBRARIES) | set(STATICALLY_REACHED)
        third_party = _third_party(_p2p_top_level_imports())

        assert third_party, 'the scan found no third-party import in the p2p package'
        assert third_party <= bundled, sorted(third_party - bundled)

    @pytest.mark.parametrize('package', REMOVED_P2P_PACKAGES)
    def test_the_removed_network_stack_is_gone(self, package):
        hidden = _spec_hidden_imports()

        assert package not in _p2p_top_level_imports()
        assert package not in _collected_packages('collect_submodules')
        assert package not in _collected_packages('copy_metadata')
        assert not [name for name in hidden if name == package or name.startswith(package + '.')]

    @pytest.mark.parametrize('module', P2P_NODE_TASK_MODULES)
    def test_the_task_module_the_node_reads_is_collected(self, module):
        assert module.split('.')[0] in _collected_packages('collect_submodules')
        assert (REPO / (module.replace('.', '/') + '.py')).is_file(), module
