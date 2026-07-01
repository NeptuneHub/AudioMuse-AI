# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Linux embedded-PostgreSQL backend selector.

Chooses the embedded-database implementation for the Linux standalone build
based on CPU architecture: x86_64 uses the pgserver-backed path in the shared
``database`` module (wrapped in the native library-path context), while other
architectures fall back to the bundled-binary ``linux.embedded_pg`` control.

Main Features:
* Detects architecture and exposes ``using_pgserver`` for callers.
* Dispatches ``start_embedded`` / ``ensure_embedded_running`` to the right backend.
"""

import platform

_USE_PGSERVER = platform.machine() in ("x86_64", "amd64")


def using_pgserver():
    return _USE_PGSERVER


def start_embedded(data_dir):
    if _USE_PGSERVER:
        import database
        from linux import env

        with env.native_lib_path_restored():
            return database.start_embedded(data_dir)
    from linux import embedded_pg

    return embedded_pg.start(data_dir)


def ensure_embedded_running(data_dir):
    if _USE_PGSERVER:
        import database
        from linux import env

        with env.native_lib_path_restored():
            return database.ensure_embedded_running(data_dir)
    from linux import embedded_pg

    return embedded_pg.ensure_running(data_dir)


def stop_embedded():
    if _USE_PGSERVER:
        import database

        return database.stop_embedded()
    from linux import embedded_pg

    return embedded_pg.stop()
