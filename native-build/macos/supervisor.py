# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Process supervisor for the macOS standalone build.

Boots and monitors the full local stack in dependency order: embedded
PostgreSQL (via the shared ``database`` module), the Flask/waitress server and
the queue worker/maintenance/control-listener children (each re-spawned from
``macos.launcher`` with a ``--role=``). The boot order, the restart policy and
the teardown all live in ``native_common.supervisor_posix``, which Linux shares;
this module only names the macOS embedded-database, paths and child-environment
modules.

Main Features:
* Binds the shared POSIX supervisor to the ``macos`` platform modules.
"""

import database
import service_roles
from macos import env as env_builder
from macos import paths
from native_common.supervisor_posix import PosixSupervisor

ROLE_OF = service_roles.ROLE_OF

BOOT_ORDER = service_roles.BOOT_ORDER


class ProcessSupervisor(PosixSupervisor):
    db_backend = database
    paths = paths
    env_builder = env_builder
