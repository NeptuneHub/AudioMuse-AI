# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Filesystem locations of the end-to-end suite.

Every other stack module and the conftest resolve paths through here, so the
repo root, the committed fixture library, the per-run artifact directory that
CI uploads, and the scratch data directory are each defined exactly once. The
scratch data (Postgres cluster, Navidrome database, temp audio, IVF cache) lives
on a native temp filesystem rather than under the repo, because the repo may sit
on a mounted Windows drive where Postgres and SQLite locking misbehave.

Main Features:
* REPO_ROOT, LIBRARY_DIR and MANIFEST_PATH point at the committed inputs
* RUN_DIR holds logs, env.json and browser traces (uploaded as a CI artifact)
* prepare_data_dir returns a fresh temp root, or the directory named by
  AUDIOMUSE_E2E_STATE_DIR when a developer wants to keep state between runs
"""

import os
import shutil
import tempfile

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
)
E2E_DIR = os.path.join(REPO_ROOT, 'test', 'e2e')
LIBRARY_DIR = os.path.join(E2E_DIR, 'library')
MANIFEST_PATH = os.path.join(LIBRARY_DIR, 'manifest.json')
RUN_DIR = os.path.join(E2E_DIR, '.run')
LOG_DIR = os.path.join(RUN_DIR, 'logs')
PLAYWRIGHT_DIR = os.path.join(RUN_DIR, 'playwright')
CACHE_DIR = os.path.join(REPO_ROOT, 'test', '.cache')

STATE_DIR_ENV = 'AUDIOMUSE_E2E_STATE_DIR'

DATA_SUBDIRS = (
    'pg',
    os.path.join('navidrome', 'data'),
    os.path.join('navidrome', 'cache'),
    os.path.join('navidrome2', 'data'),
    os.path.join('navidrome2', 'cache'),
    'temp_audio',
    'ivf_cache',
    'tls',
    'plugins',
    'backups',
    'numba_cache',
)


def prepare_run_dir():
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    for path in (LOG_DIR, PLAYWRIGHT_DIR):
        os.makedirs(path, exist_ok=True)
    return RUN_DIR


def prepare_data_dir():
    keep = os.environ.get(STATE_DIR_ENV, '').strip()
    if keep:
        root = os.path.abspath(keep)
        reused = os.path.isdir(os.path.join(root, 'pg', 'base'))
    else:
        root = tempfile.mkdtemp(prefix='audiomuse_e2e_')
        reused = False
    for name in DATA_SUBDIRS:
        os.makedirs(os.path.join(root, name), exist_ok=True)
    return root, reused


def log_path(name):
    return os.path.join(LOG_DIR, f'{name}.log')
