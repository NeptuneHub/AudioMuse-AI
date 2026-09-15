# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Native library-path handling and child-env builder for the Linux standalone build.

PyInstaller sets ``LD_LIBRARY_PATH`` to the bundle so the frozen app finds its
own libraries; that same path breaks child processes (postgres) that
need the system loader. This module restores the original
``LD_LIBRARY_PATH_ORIG`` for child environments. The model/cache/temp/backup
half of the child environment is shared with macOS and Windows in
``native_common.child_env``; only the platform label and the control socket are
named here.

Main Features:
* ``restore_native_lib_path`` rewrites a child env dict to the system loader path;
  ``native_lib_path_restored`` applies the same fix in-process (no-op when not frozen).
* ``build_child_env`` adds the Linux label and control socket to the shared child
  environment.
* It derives POSTGRES_HOST/POSTGRES_PORT from the URL the embedded server
  reported, so the children reach the socket it actually opened rather than a
  guess: ``config.py`` builds every connection string from those two.
"""

import contextlib
import os
import sys

from linux import paths
from native_common import child_env


def _pg_conn_parts(database_url):
    return child_env.pg_conn_parts(database_url, paths.pgdata_dir)


def restore_native_lib_path(env):
    if not getattr(sys, "frozen", False):
        return env
    orig = env.get("LD_LIBRARY_PATH_ORIG")
    if orig:
        env["LD_LIBRARY_PATH"] = orig
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


@contextlib.contextmanager
def native_lib_path_restored():
    if not getattr(sys, "frozen", False):
        yield
        return
    saved = os.environ.get("LD_LIBRARY_PATH")
    try:
        restore_native_lib_path(os.environ)
        yield
    finally:
        if saved is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = saved


def build_child_env(role, database_url):
    return child_env.build_child_env(
        paths,
        role,
        database_url,
        child_env.embedded_postgres_parts(database_url, paths.pgdata_dir),
        {
            "AUDIOMUSE_PLATFORM": "linux",
            "AUDIOMUSE_CONTROL_SOCKET": paths.control_socket_path(),
        },
    )
