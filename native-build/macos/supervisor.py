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
this module names the macOS embedded-database, paths and child-environment
modules and clears the CoreML compile cache before each boot.

Main Features:
* Binds the shared POSIX supervisor to the ``macos`` platform modules.
* Deletes Apple's CoreML compile cache (``com.apple.e5rt.e5bundlecache``) after
  the orphan reap and before the children boot. ONNX Runtime recompiles the
  CLAP audio model to a new temp path on every load, so CoreML adds an entry
  per load that it never reuses and only purges when the disk runs low. No
  child is running yet, so no live CoreML session can lose its cache; a missing
  folder is skipped silently.
"""

import os
import shutil

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

    def _clear_platform_caches(self):
        cache_dir = paths.coreml_cache_dir()
        if not os.path.isdir(cache_dir):
            return
        try:
            shutil.rmtree(cache_dir)
            self._log.info("Deleted the CoreML compile cache %s", cache_dir)
        except Exception:
            self._log.exception("Could not delete the CoreML compile cache %s", cache_dir)
