# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Child-process environment builder for the macOS standalone build.

The per-user data, model, cache, temp and backup variables every supervised
child inherits are shared with Linux and Windows in
``native_common.child_env``; what is named here is what only macOS needs: the
platform label, the control socket and the OBJC fork-safety and LC_NUMERIC
settings that keep forked children from aborting.

Main Features:
* Adds the macOS label, control socket and fork-safety settings to the shared
  child environment.
* Derives POSTGRES_HOST/POSTGRES_PORT from the URL the embedded server reported,
  so the children reach the socket it actually opened rather than a guess:
  ``config.py`` builds every connection string from those two.
"""

from macos import paths
from native_common import child_env


def _pg_conn_parts(database_url):
    return child_env.pg_conn_parts(database_url, paths.pgdata_dir)


def build_child_env(role, database_url):
    return child_env.build_child_env(
        paths,
        role,
        database_url,
        child_env.embedded_postgres_parts(database_url, paths.pgdata_dir),
        {
            "AUDIOMUSE_PLATFORM": "macos",
            "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
            "LC_NUMERIC": "C",
            "AUDIOMUSE_CONTROL_SOCKET": paths.control_socket_path(),
        },
    )
