# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Repo-wide guard that no tracked code file contains emoji or icons.

Scans tracked files with code extensions for emoji code points and fails with
the offending paths, since emoji break the Windows build.

Main Features:
* The candidate file list is non-empty so the scan cannot silently pass
* No emoji appears in any tracked code file
"""

import os
import subprocess

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

CODE_EXTENSIONS = (
    '.py',
    '.js',
    '.sh',
    '.bat',
    '.ps1',
    '.yml',
    '.yaml',
    '.toml',
    '.cfg',
    '.ini',
)

EXACT_EXCLUDES = {
    'static/plotly-2.29.1.min.js',
    'lyrics/lyrics_transcriber.py',
}

PATH_FRAGMENT_EXCLUDES = (
    '.venv',
    'node_modules',
    'screenshot/',
)


def _is_pictographic_emoji(codepoint):
    return (0x1F000 <= codepoint <= 0x1FAFF) or codepoint == 0xFE0F


def _git_ls_files():
    out = subprocess.check_output(['git', 'ls-files'], cwd=REPO_ROOT).decode('utf-8')
    return [line for line in out.splitlines() if line]


def _is_candidate(rel_path):
    posix = rel_path.replace('\\', '/')
    if not posix.endswith(CODE_EXTENSIONS):
        return False
    if posix.endswith('.html') or posix.startswith('templates/'):
        return False
    if posix.endswith('.json'):
        return False
    if posix in EXACT_EXCLUDES:
        return False
    if posix.startswith('static/') and posix.endswith('.js'):
        return False
    for frag in PATH_FRAGMENT_EXCLUDES:
        if frag in posix:
            return False
    return True


def _candidate_files():
    return [f for f in _git_ls_files() if _is_candidate(f)]


def _scan_for_emoji(rel_path):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    offenders = []
    try:
        with open(abs_path, encoding='utf-8') as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return offenders
    for line_no, line in enumerate(lines, start=1):
        bad = sorted({ch for ch in line if _is_pictographic_emoji(ord(ch))})
        if bad:
            codes = ' '.join('U+%04X' % ord(c) for c in bad)
            offenders.append((line_no, codes))
    return offenders


def test_candidate_file_list_is_non_empty():
    assert _candidate_files(), 'no candidate code files found via git ls-files'


def test_no_emoji_in_code_files():
    failures = []
    for rel_path in _candidate_files():
        for line_no, codes in _scan_for_emoji(rel_path):
            failures.append('{0}:{1} ({2})'.format(rel_path, line_no, codes))
    assert not failures, (
        'Emoji / pictographic codepoints found in code files '
        '(emoji are only allowed in HTML pages; they crash the Windows '
        'console build):\n  ' + '\n  '.join(failures)
    )
