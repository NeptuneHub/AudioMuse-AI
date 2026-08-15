# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Guard that every repo-root data file config.py resolves relative to its
own directory is actually bundled by the native build (AudioMuse-AI.spec).

config.py resolves files like GENRE_SUBGENRE_FILE via _bundle_data_root()
(a frozen-aware root: sys._MEIPASS when running under PyInstaller, else
the config module's directory). Those files only exist at runtime in a
PyInstaller build if AudioMuse-AI.spec lists the filename in its bundled
data. Docker/pip installs read the repo checkout directly and are never
affected by this gap; only native Windows/macOS/Linux standalone builds
are, and only until the app is rebuilt with an updated spec. Regression:
genre_subgenre.json shipped in config.py without a matching spec entry,
so every native build silently fell back to the legacy mood-based
Hyperbolic Explorer tree.

Main Features:
* config.py's single-file data resolutions (direct or via _bundle_data_root)
  are discoverable
* Every such file is referenced by name somewhere in AudioMuse-AI.spec
"""

import os
import re

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
CONFIG_PATH = os.path.join(REPO_ROOT, 'config.py')
SPEC_PATH = os.path.join(REPO_ROOT, 'AudioMuse-AI.spec')

ROOT_FILE_PATTERN = re.compile(
    r"os\.path\.join\((?:os\.path\.dirname\(os\.path\.abspath\(__file__\)\)|_bundle_data_root\(\)),\s*"
    r"['\"]([^'\"]+)['\"]\s*\)"
)
QUOTED_FILENAME_PATTERN = re.compile(r"""['"]([^'"]+\.[A-Za-z0-9]+)['"]""")


def _read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def _config_declared_root_files():
    text = _read(CONFIG_PATH)
    names = ROOT_FILE_PATTERN.findall(text)
    return [n for n in names if os.path.isfile(os.path.join(REPO_ROOT, n))]


def _spec_referenced_names():
    text = _read(SPEC_PATH)
    return set(QUOTED_FILENAME_PATTERN.findall(text))


def test_config_declares_at_least_one_root_data_file():
    assert _config_declared_root_files(), 'no __file__-relative data files found in config.py'


def test_every_config_declared_root_data_file_is_bundled_in_spec():
    declared = _config_declared_root_files()
    bundled = _spec_referenced_names()
    missing = [name for name in declared if name not in bundled]
    assert not missing, (
        'config.py resolves these repo-root files relative to itself, but '
        'AudioMuse-AI.spec never references them, so native builds ship '
        'without them: ' + ', '.join(missing)
    )
