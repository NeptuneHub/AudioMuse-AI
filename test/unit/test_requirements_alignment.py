# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Alignment of test/requirements.txt pins with requirements/common.txt.

security.yml's Trivy scan reads `pip:requirements/.*\\.txt` and explicitly
skips test/requirements.txt and both noavx2 files, so a stale/unpinned
version in test/requirements.txt is invisible to the vulnerability scanner.
This guard is the only mechanism keeping test/requirements.txt honest: every
package that requirements/common.txt pins with `==` must be pinned to that
identical version wherever it also appears in test/requirements.txt.
requirements/common-noavx2.txt is intentionally excluded - it pins older
versions on purpose for CPU compatibility and must not be touched to match.

Main Features:
* Both requirement files parse into non-empty name -> version maps
* Every package `==`-pinned in common.txt is `==`-pinned to the same version
  in test/requirements.txt, wherever that package also appears there
"""

import os
import re

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
COMMON_REQUIREMENTS = os.path.join(REPO_ROOT, 'requirements', 'common.txt')
TEST_REQUIREMENTS = os.path.join(REPO_ROOT, 'test', 'requirements.txt')

_PIN_RE = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)')
_NAME_RE = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*)')


def _normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def _parse_requirements(path):
    """Returns (pinned: {norm_name: version}, all_names: {norm_name})."""
    pinned = {}
    all_names = set()
    with open(path, encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            pin_match = _PIN_RE.match(line)
            if pin_match:
                pinned[_normalize(pin_match.group(1))] = pin_match.group(2)
                all_names.add(_normalize(pin_match.group(1)))
                continue
            name_match = _NAME_RE.match(line)
            if name_match:
                all_names.add(_normalize(name_match.group(1)))
    return pinned, all_names


def test_both_requirement_files_parse_non_empty():
    common_pinned, common_all = _parse_requirements(COMMON_REQUIREMENTS)
    test_pinned, test_all = _parse_requirements(TEST_REQUIREMENTS)
    assert common_pinned and common_all, 'requirements/common.txt parsed as empty'
    assert test_pinned and test_all, 'test/requirements.txt parsed as empty'


def test_shared_packages_pinned_identically_to_common():
    common_pinned, _ = _parse_requirements(COMMON_REQUIREMENTS)
    test_pinned, test_all = _parse_requirements(TEST_REQUIREMENTS)

    failures = []
    for name, common_version in sorted(common_pinned.items()):
        if name not in test_all:
            continue
        test_version = test_pinned.get(name)
        if test_version is None:
            failures.append(
                f'{name}: pinned=={common_version} in common.txt but unpinned in '
                f'test/requirements.txt'
            )
        elif test_version != common_version:
            failures.append(
                f'{name}: common.txt=={common_version} vs test/requirements.txt=='
                f'{test_version}'
            )

    assert not failures, (
        'test/requirements.txt drifted from requirements/common.txt (Trivy skips '
        'test/requirements.txt, so this is the only guard against a stale/'
        'vulnerable pin sneaking into the test file):\n  ' + '\n  '.join(failures)
    )
