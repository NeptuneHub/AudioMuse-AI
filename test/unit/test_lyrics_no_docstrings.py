# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Guard that lyrics/*.py carries no docstrings except the required header.

House rule: `.py` files directly under `lyrics/` (not subfolders) must have
no docstrings anywhere in the file body - only the module-level header
docstring is allowed. Function/class docstrings are stripped on sight.

Main Features:
* The candidate file list is non-empty so the scan cannot silently pass
* Each lyrics/*.py file has its required module-level header docstring
* No function, async function, or class in lyrics/*.py carries a docstring
"""

import ast
import glob
import os

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
LYRICS_DIR = os.path.join(REPO_ROOT, 'lyrics')


def _candidate_files():
    return sorted(glob.glob(os.path.join(LYRICS_DIR, '*.py')))


def test_candidate_file_list_is_non_empty():
    assert _candidate_files(), 'no lyrics/*.py files found'


def test_lyrics_files_have_no_body_docstrings():
    failures = []
    for abs_path in _candidate_files():
        rel_path = os.path.relpath(abs_path, REPO_ROOT).replace('\\', '/')
        with open(abs_path, encoding='utf-8') as handle:
            source = handle.read()
        tree = ast.parse(source, filename=rel_path)

        if not ast.get_docstring(tree):
            failures.append(f'{rel_path}: missing required module-level header docstring')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    failures.append(f'{rel_path}:{node.lineno}: {node.name}() has a docstring')

    assert not failures, (
        'lyrics/*.py files may only have the required module-level header '
        'docstring - no function/class docstrings:\n  ' + '\n  '.join(failures)
    )
