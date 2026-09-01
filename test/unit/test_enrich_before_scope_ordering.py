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

Main Features:
* The scanned file list is non-empty so the check cannot silently pass
* In any function calling both, attach_song_features precedes scope_results
"""

import ast
import glob
import os

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

ENRICH = 'attach_song_features'
SCOPE = 'scope_results'


def _called_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _first_call_lines(func_node):
    enrich_lines = []
    scope_lines = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name == ENRICH:
            enrich_lines.append(node.lineno)
        elif name == SCOPE:
            scope_lines.append(node.lineno)
    return enrich_lines, scope_lines


def _app_modules():
    return sorted(glob.glob(os.path.join(REPO_ROOT, 'app_*.py')))


def test_app_modules_are_actually_scanned():
    assert len(_app_modules()) > 5


def test_attach_song_features_runs_before_scope_results():
    offenders = []
    for path in _app_modules():
        with open(path, 'r', encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            enrich_lines, scope_lines = _first_call_lines(node)
            if not enrich_lines or not scope_lines:
                continue
            if min(enrich_lines) > min(scope_lines):
                offenders.append(
                    '%s:%s in %s() enriches at line %d but scopes at line %d'
                    % (os.path.basename(path), node.lineno, node.name,
                       min(enrich_lines), min(scope_lines))
                )
    assert not offenders, (
        'attach_song_features must run on canonical ids, before scope_results '
        'rewrites them to provider ids:\n' + '\n'.join(offenders)
    )
