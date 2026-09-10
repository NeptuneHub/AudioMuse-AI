# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Repo-wide guard that no SQL statement writes the backslash ESCAPE as a plain string.

Issue #901: a backslash ESCAPE clause written as a plain string literal is
valid only while the server runs with standard_conforming_strings on; with it
off the clause is an unterminated string and every affected query fails. The
E-prefixed string literal reads as one backslash in both modes, so every
backslash ESCAPE clause must use it.

The same server setting silently drops the backslash from an inline LIKE
pattern such as fp backslash underscore percent, turning the escaped underscore
into a one-character wildcard, so inline patterns must use the E-prefixed form
as well.

Main Features:
* The candidate file list is non-empty so the scan cannot silently pass
* No tracked Python file contains a plain-string backslash ESCAPE clause
* No tracked Python file has an inline LIKE pattern with a backslash in a plain string
"""

import os
import re
import subprocess

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

BACKSLASH_ESCAPE_CLAUSE = "ESCAPE '" + chr(92) * 2 + "'"
PLAIN_BACKSLASH_LIKE_PATTERN = re.compile(r"\b(?:LIKE|ILIKE)\s+'[^']*" + re.escape(chr(92)), re.IGNORECASE)

PATH_FRAGMENT_EXCLUDES = (
    '.venv',
    'node_modules',
    '/vendor/',
)


def _git_ls_files():
    out = subprocess.check_output(['git', 'ls-files', '*.py'], cwd=REPO_ROOT).decode('utf-8')
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


def test_candidate_file_list_is_non_empty():
    assert _candidate_files(), 'no candidate files found via git ls-files'


def test_no_backslash_escape_clause_in_repo():
    failures = []
    for rel_path in _candidate_files():
        text = _read_text(rel_path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if BACKSLASH_ESCAPE_CLAUSE in line:
                failures.append('{0}:{1}'.format(rel_path, line_no))
    assert not failures, (
        "A plain-string backslash ESCAPE clause breaks Postgres with standard_conforming_strings off "
        "(issue #901); write it as an E-prefixed string literal instead:\n  "
        + '\n  '.join(failures)
    )


def test_no_plain_string_backslash_like_pattern_in_repo():
    failures = []
    for rel_path in _candidate_files():
        text = _read_text(rel_path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if PLAIN_BACKSLASH_LIKE_PATTERN.search(line):
                failures.append('{0}:{1}'.format(rel_path, line_no))
    assert not failures, (
        "An inline LIKE pattern with a backslash in a plain string loses the backslash when "
        "standard_conforming_strings is off (issue #901); write it as an E-prefixed string literal:\n  "
        + '\n  '.join(failures)
    )
