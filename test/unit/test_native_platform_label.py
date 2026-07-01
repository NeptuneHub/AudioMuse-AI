# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Guard that each native-build env.py sets AUDIOMUSE_PLATFORM to its own name.

Prevents the copy-paste mislabel bug (linux/windows env.py had hardcoded
"macos") from silently returning, since no runtime code branches on the value
anymore.

Main Features:
* Asserts native-build/<platform>/env.py sets AUDIOMUSE_PLATFORM == "<platform>"
* Scans source text so it needs no platform-specific imports
"""

import os

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


@pytest.mark.parametrize('platform', ['linux', 'macos', 'windows'])
def test_env_sets_matching_platform_label(platform):
    path = os.path.join(REPO_ROOT, 'native-build', platform, 'env.py')
    with open(path, encoding='utf-8') as handle:
        source = handle.read()
    expected = '"AUDIOMUSE_PLATFORM": "%s"' % platform
    assert expected in source, (
        'native-build/%s/env.py must set %s (found a different or missing label)' % (platform, expected)
    )
