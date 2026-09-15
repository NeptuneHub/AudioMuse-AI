# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Child-process environment builder for the Windows standalone build.

The per-user data, model, cache, temp and backup variables every supervised
child inherits are shared with Linux and macOS in ``native_common.child_env``;
what is named here is what only Windows needs: the platform label, the loopback
control host/port used instead of a control socket, and the database URL built
from the connection mapping the embedded server reports.

Main Features:
* Builds the DATABASE_URL from the embedded connection mapping.
* Adds the Windows label and loopback control host/port to the shared child
  environment.
"""

from urllib.parse import quote

from native_common import child_env
from windows import paths


def build_child_env(role, db_conn):
    database_url = (
        f"postgresql://{quote(db_conn['user'], safe='')}:"
        f"{quote(db_conn['password'], safe='')}"
        f"@{db_conn['host']}:{db_conn['port']}/{db_conn['dbname']}"
    )
    return child_env.build_child_env(
        paths,
        role,
        database_url,
        {
            "host": db_conn["host"],
            "port": str(db_conn["port"]),
            "user": db_conn["user"],
            "password": db_conn["password"],
            "dbname": db_conn["dbname"],
        },
        {
            "AUDIOMUSE_PLATFORM": "windows",
            "AUDIOMUSE_CONTROL_SOCKET": "",
            "AUDIOMUSE_CONTROL_HOST": "127.0.0.1",
            "AUDIOMUSE_CONTROL_PORT": str(paths.control_port()),
        },
    )
