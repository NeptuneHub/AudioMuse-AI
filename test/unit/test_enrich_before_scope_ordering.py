# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Guard that every endpoint enriches song features BEFORE it scopes the results.

attach_song_features looks the rows up in the score table by CANONICAL item_id,
while scope_results rewrites each surviving row's item_id to the request server's
own provider id. Called in that order the score lookup misses every row and the
setdefault calls never fire, so mood_vector / other_features / top_genre / top_mood
silently go missing from the response on any canonicalized or multi-server install.

The pairing is done PER RESULT LIST: a scope_results(rows) is only required to be
preceded by an attach_song_features on that same variable, so an endpoint that
shapes two independent lists is judged on each list separately rather than on the
earliest line of either. A list that is never enriched anywhere in the function
carries no ordering requirement at all - plenty of endpoints scope rows that are
not song rows. Calls whose first argument is not a plain name cannot be paired
that way and fall back to requiring some enrich call earlier in the function.
Nested function bodies are attributed to the nested function, not to the endpoint
that encloses it.

Main Features:
* The scanned file list is non-empty and covers every module that scopes results,
  not only the top-level app_*.py blueprints
* Within one function body, a list is never scoped before it is enriched
* A call inside a nested def is not attributed to its enclosing function
* A list that is scoped but never enriched raises no complaint
* An unpairable scope call still requires an earlier enrich in the same function
"""

import ast
import os

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

ENRICH = 'attach_song_features'
SCOPE = 'scope_results'
SKIP_DIRS = {'.git', '.venv', '.venv-windows', 'dist', 'build', 'node_modules', 'model', 'test'}


def _called_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _first_arg_name(node):
    if not node.args:
        return None
    first = node.args[0]
    return first.id if isinstance(first, ast.Name) else None


def _own_body_calls(func_node):
    calls = []
    stack = list(func_node.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            calls.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return calls


def _ordering_offences(func_node):
    enrich_lines_by_name = {}
    enrich_lines = []
    scope_calls = []
    for node in _own_body_calls(func_node):
        name = _called_name(node)
        if name == ENRICH:
            enrich_lines.append(node.lineno)
            enrich_lines_by_name.setdefault(_first_arg_name(node), []).append(node.lineno)
        elif name == SCOPE:
            scope_calls.append(node)

    if not enrich_lines or not scope_calls:
        return []

    offences = []
    for node in scope_calls:
        target = _first_arg_name(node)
        if target is not None:
            candidates = enrich_lines_by_name.get(target)
            if not candidates:
                continue
        else:
            candidates = enrich_lines
        if not any(line < node.lineno for line in candidates):
            offences.append((target or '<expression>', node.lineno, min(candidates)))
    return offences


def _python_sources():
    found = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.')]
        for filename in filenames:
            if filename.endswith('.py'):
                found.append(os.path.join(dirpath, filename))
    return sorted(found)


def _modules_that_scope():
    scoping = []
    for path in _python_sources():
        with open(path, 'r', encoding='utf-8') as handle:
            text = handle.read()
        if SCOPE + '(' in text:
            scoping.append((path, text))
    return scoping


def test_the_scan_reaches_every_module_that_scopes_results():
    paths = [os.path.basename(p) for p, _ in _modules_that_scope()]

    assert len(paths) > 5, 'the ordering guard is scanning almost nothing: %s' % paths


def test_a_call_inside_a_nested_def_is_not_attributed_to_its_enclosing_function():
    tree = ast.parse(
        'def endpoint():\n'
        '    rows = build()\n'
        '    rows = scope_results(rows)\n'
        '    def helper():\n'
        '        attach_song_features(rows)\n'
    )
    endpoint = tree.body[0]

    assert _ordering_offences(endpoint) == []


def test_two_independent_lists_are_judged_separately():
    tree = ast.parse(
        'def endpoint():\n'
        '    a = build()\n'
        '    a = scope_results(a)\n'
        '    b = build()\n'
        '    attach_song_features(b)\n'
        '    b = scope_results(b)\n'
    )

    assert _ordering_offences(tree.body[0]) == []


def test_scoping_a_list_before_enriching_that_same_list_is_reported():
    tree = ast.parse(
        'def endpoint():\n'
        '    rows = build()\n'
        '    rows = scope_results(rows)\n'
        '    attach_song_features(rows)\n'
    )
    offences = _ordering_offences(tree.body[0])

    assert [o[0] for o in offences] == ['rows']


def test_attach_song_features_runs_before_scope_results():
    offenders = []
    for path, text in _modules_that_scope():
        tree = ast.parse(text, filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for target, scope_line, enrich_line in _ordering_offences(node):
                offenders.append(
                    '%s in %s(): %s scoped at line %d, earliest enrich at line %d'
                    % (os.path.relpath(path, REPO_ROOT), node.name, target,
                       scope_line, enrich_line)
                )
    assert not offenders, (
        'attach_song_features must run on canonical ids, before scope_results '
        'rewrites them to provider ids:\n' + '\n'.join(offenders)
    )


@pytest.mark.parametrize('name', [ENRICH, SCOPE])
def test_both_helper_names_still_exist(name):
    import app_helper
    import app_server_context

    assert hasattr(app_helper, name) or hasattr(app_server_context, name), (
        '%s was renamed or moved, so the ordering guard now matches nothing' % name
    )
