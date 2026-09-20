# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Repo-wide guard that every tracked .py file carries the house header.

Every file must start with a legalese `#` comment block (AudioMuse-AI, the
repo link, and the AGPL-3.0 SPDX line) followed by a module docstring that
stays within a character budget; every docstring except bare `__init__.py`
package markers must also contain a `Main Features:` bullet list. That header
is the only prose allowed: `#` comments below it belong to `app*.py`,
`config.py` and tool pragmas alone.

Main Features:
* The candidate file list is non-empty so the scan cannot silently pass
* Every tracked .py file has the legalese header block and a module docstring
* Every non-`__init__.py` module docstring contains a `Main Features:` section
* Every module docstring stays inside the 2500 character budget
* No `#` comment below the header outside `app*.py`, `config.py` and pragmas
"""

import ast
import fnmatch
import io
import os
import re
import subprocess
import tokenize

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

LEGALESE_MARKERS = ('AudioMuse-AI', 'AGPL-3.0')
MAIN_FEATURES_MARKER = 'Main Features:'
MAX_DOCSTRING_CHARS = 2500
COMMENT_EXEMPT_GLOBS = ('app*.py', 'config.py')
PRAGMA_RE = re.compile(
    r'^#\s*(?:noqa|pragma|nosec|type:|fmt:|isort:|black:|pylint:|mypy:|ruff:|coding[:=])'
)


def _git_ls_files():
    out = subprocess.check_output(['git', 'ls-files', '*.py'], cwd=REPO_ROOT).decode('utf-8')
    return [line for line in out.splitlines() if line]


def _candidate_files():
    return _git_ls_files()


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), encoding='utf-8') as handle:
        return handle.read()


def _leading_comment_lines(source):
    lines = source.splitlines()
    index = 1 if lines and lines[0].startswith('#!') else 0
    while index < len(lines) and lines[index].startswith('#'):
        index += 1
    return index


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


def _is_comment_exempt(rel_path):
    base = os.path.basename(rel_path)
    return any(fnmatch.fnmatch(base, pattern) for pattern in COMMENT_EXEMPT_GLOBS)


def _body_comments(source):
    header_end = _leading_comment_lines(source)
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT or token.start[0] <= header_end:
            continue
        text = token.string.strip()
        if PRAGMA_RE.match(text):
            continue
        found.append((token.start[0], text))
    return found


def test_candidate_file_list_is_non_empty():
    assert _candidate_files(), 'no candidate .py files found via git ls-files'


def test_every_file_has_legalese_header_and_docstring():
    failures = []
    for rel_path in _candidate_files():
        try:
            source = _read(rel_path)
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


def test_module_docstring_stays_within_budget():
    failures = []
    for rel_path in _candidate_files():
        try:
            source = _read(rel_path)
            tree = ast.parse(source, filename=rel_path)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            failures.append(f'{rel_path}: could not read file ({exc})')
            continue

        docstring = ast.get_docstring(tree, clean=False) or ''
        if len(docstring) > MAX_DOCSTRING_CHARS:
            failures.append(f'{rel_path}: {len(docstring)} chars')

    assert not failures, (
        f'A module docstring is a summary, not a manual: keep it under '
        f'{MAX_DOCSTRING_CHARS} characters (the legalese "#" block above it does '
        f'not count). Condense the "Main Features:" bullets instead of growing '
        f'them:\n  ' + '\n  '.join(failures)
    )


def test_no_comments_below_the_header():
    failures = []
    for rel_path in _candidate_files():
        if _is_comment_exempt(rel_path):
            continue
        try:
            source = _read(rel_path)
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f'{rel_path}: could not read file ({exc})')
            continue

        try:
            comments = _body_comments(source)
        except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
            failures.append(f'{rel_path}: could not tokenize for comment check ({exc})')
            continue

        for line_no, text in comments:
            failures.append(f'{rel_path}:{line_no}: {text[:70]}')

    assert not failures, (
        'Code below the file header carries no "#" comments: the header docstring '
        'explains the module and the code explains itself. Only '
        + ', '.join(COMMENT_EXEMPT_GLOBS)
        + ' may carry body comments, plus tool pragmas (noqa / pragma / nosec / '
        'type: / fmt: / isort:) anywhere:\n  ' + '\n  '.join(failures)
    )
