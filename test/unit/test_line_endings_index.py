# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Repo-wide guard that no tracked git blob carries CRLF line endings.

Reads `git ls-files --eol`, which reports the INDEX blob's line-ending state
directly - deterministic regardless of the working-tree checkout or
core.autocrlf, unlike CI's lint-line-endings.yml grep over the checked-out
tree (which also blindly skips the whole windows/ directory). .gitattributes
forces LF on every extension except the *.bat/*.cmd pair that cmd.exe
requires as CRLF.

Main Features:
* The candidate file list is non-empty so the scan cannot silently pass
* No tracked text blob is CRLF or mixed-eol in the git index
"""

import os
import subprocess

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def _git_ls_files_eol():
    out = subprocess.check_output(['git', 'ls-files', '--eol'], cwd=REPO_ROOT).decode('utf-8')
    entries = []
    for line in out.splitlines():
        if not line:
            continue
        attrs, _, path = line.partition('\t')
        fields = attrs.split()
        index_field = next((f for f in fields if f.startswith('i/')), 'i/none')
        entries.append((path, index_field))
    return entries


def test_candidate_file_list_is_non_empty():
    assert _git_ls_files_eol(), 'no tracked files found via git ls-files --eol'


def test_no_crlf_or_mixed_eol_in_index():
    failures = [
        f'{path} ({index_field})'
        for path, index_field in _git_ls_files_eol()
        if index_field in ('i/crlf', 'i/mixed')
    ]
    assert not failures, (
        'Tracked files with CRLF/mixed line endings in the git index (only '
        '*.bat/*.cmd may be eol=crlf per .gitattributes; everything else must '
        'be LF):\n  ' + '\n  '.join(failures)
    )
