# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Repo-wide guard that a backslash inside a SQL string literal is always E-prefixed.

Issue #901: a backslash inside a plain SQL string literal is read one way when
the server has standard_conforming_strings on and another way when it is off.
A backslash ESCAPE clause written as a plain string then fails to parse, and an
inline LIKE pattern silently loses its escaped underscore. The E-prefixed
literal reads as one backslash in both modes, so every such clause and pattern
must use it. Bound parameters need nothing, psycopg2 quotes them per server.

Main Features:
* The candidate file list is non-empty so the scan cannot silently pass
* The scan skips instead of erroring where git ls-files is unavailable
* No tracked Python or SQL file has a plain-string backslash ESCAPE clause, in
  any case or spacing, even when the statement is split across source lines
* No tracked Python or SQL file has an inline LIKE pattern holding a backslash
  in a plain string
"""

import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

BACKSLASH = chr(92)
PLAIN_BACKSLASH_ESCAPE_CLAUSE = re.compile(
    r"\bESCAPE[\s\"]*'" + re.escape(BACKSLASH), re.IGNORECASE
)
PLAIN_BACKSLASH_LIKE_PATTERN = re.compile(
    r"\b(?:LIKE|ILIKE)[\s\"]*'[^'\n]*" + re.escape(BACKSLASH) + r"[^'\n]*'", re.IGNORECASE
)

PATH_FRAGMENT_EXCLUDES = (
    '.venv',
    'node_modules',
    '/vendor/',
)


def _git_ls_files():
    try:
        out = subprocess.check_output(
            ['git', 'ls-files', '*.py', '*.sql'], cwd=REPO_ROOT
        ).decode('utf-8')
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f'git ls-files is unavailable here, source guard skipped: {exc}')
    return [line for line in out.splitlines() if line]


def _is_candidate(rel_path):
    posix = rel_path.replace('\\', '/')
    for frag in PATH_FRAGMENT_EXCLUDES:
        if frag in posix:
            return False
    return True


def _read_text(rel_path):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    try:
        with open(abs_path, encoding='utf-8') as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


def _candidate_files():
    return [f for f in _git_ls_files() if _is_candidate(f)]


def _scan(pattern):
    failures = []
    for rel_path in _candidate_files():
        text = _read_text(rel_path)
        if text is None:
            continue
        for match in pattern.finditer(text):
            failures.append('{0}:{1}'.format(rel_path, text.count('\n', 0, match.start()) + 1))
    return failures


def test_candidate_file_list_is_non_empty():
    assert _candidate_files(), 'no candidate files found via git ls-files'


def test_no_plain_string_backslash_escape_clause_in_repo():
    failures = _scan(PLAIN_BACKSLASH_ESCAPE_CLAUSE)
    assert not failures, (
        "A plain-string backslash ESCAPE clause breaks Postgres with standard_conforming_strings off "
        "(issue #901); write it as an E-prefixed string literal instead:\n  "
        + '\n  '.join(failures)
    )


def test_no_plain_string_backslash_like_pattern_in_repo():
    failures = _scan(PLAIN_BACKSLASH_LIKE_PATTERN)
    assert not failures, (
        "An inline LIKE pattern with a backslash in a plain string loses the backslash when "
        "standard_conforming_strings is off (issue #901); write it as an E-prefixed string literal:\n  "
        + '\n  '.join(failures)
    )
