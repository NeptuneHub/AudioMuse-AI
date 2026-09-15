# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Process supervisor for the Linux standalone build.

Boots and monitors the full local stack in dependency order: embedded
PostgreSQL, the Flask/waitress server and the queue worker/maintenance/
restart-listener children (each re-spawned from ``linux.launcher`` with a
``--role=``). The boot order, the restart policy and the teardown all live in
``native_common.supervisor_posix``, which macOS shares; this module only names
the Linux embedded-database, paths and child-environment modules.

Main Features:
* Binds the shared POSIX supervisor to the ``linux`` platform modules.
"""

import service_roles
from linux import db_backend
from linux import env as env_builder
from linux import paths
from native_common.supervisor_posix import PosixSupervisor

ROLE_OF = service_roles.ROLE_OF

BOOT_ORDER = service_roles.BOOT_ORDER


class ProcessSupervisor(PosixSupervisor):
    db_backend = db_backend
    paths = paths
    env_builder = env_builder
