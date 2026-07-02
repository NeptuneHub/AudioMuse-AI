# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Repo-wide guard that every tracked .py file carries the house header.

Every file must start with a legalese `#` comment block (AudioMuse-AI, the
repo link, and the AGPL-3.0 SPDX line) followed by a module docstring; every
docstring except bare `__init__.py` package markers must also contain a
`Main Features:` bullet list.

Main Features:
* The candidate file list is non-empty so the scan cannot silently pass
* Every tracked .py file has the legalese header block and a module docstring
* Every non-`__init__.py` module docstring contains a `Main Features:` section
"""

import ast
import os
import subprocess

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

LEGALESE_MARKERS = ('AudioMuse-AI', 'AGPL-3.0')
MAIN_FEATURES_MARKER = 'Main Features:'


def _git_ls_files():
    out = subprocess.check_output(['git', 'ls-files', '*.py'], cwd=REPO_ROOT).decode('utf-8')
    return [line for line in out.splitlines() if line]


def _candidate_files():
    return _git_ls_files()


def _leading_comment_block(source):
    lines = source.splitlines()
    if lines and lines[0].startswith('#!'):
        lines = lines[1:]
    block = []
    for line in lines:
        if line.startswith('#'):
            block.append(line)
            continue
        break
    return '\n'.join(block)


def test_candidate_file_list_is_non_empty():
    assert _candidate_files(), 'no candidate .py files found via git ls-files'


def test_every_file_has_legalese_header_and_docstring():
    failures = []
    for rel_path in _candidate_files():
        abs_path = os.path.join(REPO_ROOT, rel_path)
        try:
            with open(abs_path, encoding='utf-8') as handle:
                source = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f'{rel_path}: could not read file ({exc})')
            continue

        header = _leading_comment_block(source)
        missing_markers = [m for m in LEGALESE_MARKERS if m not in header]
        if missing_markers:
            failures.append(f'{rel_path}: missing header marker(s) {missing_markers}')

        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as exc:
            failures.append(f'{rel_path}: could not parse for docstring check ({exc})')
            continue

        docstring = ast.get_docstring(tree)
        if not docstring:
            failures.append(f'{rel_path}: missing module docstring')
            continue

        if os.path.basename(rel_path) != '__init__.py' and MAIN_FEATURES_MARKER not in docstring:
            failures.append(f'{rel_path}: module docstring missing "{MAIN_FEATURES_MARKER}"')

    assert not failures, (
        'Every tracked .py file must start with the house header (a "#" legalese '
        'block containing "AudioMuse-AI" and "AGPL-3.0") followed by a module '
        'docstring with a "Main Features:" bullet list (package-marker '
        '__init__.py files are exempt from the Main Features requirement):\n  '
        + '\n  '.join(failures)
    )
