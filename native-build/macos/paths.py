# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Filesystem path resolution for the macOS standalone build.

Resolves the bundle resource root (PyInstaller ``_MEIPASS`` or the source
repo) and the per-user data locations under ``~/Library``, so the macOS
launcher, supervisor and control-socket modules agree on where pgdata, redis,
logs, models and the control socket live. The Linux/Windows ``paths`` modules
are the platform-specific siblings.

Main Features:
* ``resource_root`` plus ``~/Library`` data and ``~/Library/Logs`` directories.
* Control-socket and model paths shared across the macOS supervisor children.
"""

import os
import platform
import sys

APP_NAME = "AudioMuse-AI"


def resource_root():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", _repo_root())
    return _repo_root()


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def app_support_dir():
    return _ensure(os.path.join(os.path.expanduser("~"), "Library", APP_NAME))


def logs_dir():
    return _ensure(os.path.join(os.path.expanduser("~"), "Library", "Logs", APP_NAME))


def pgdata_dir():
    return _ensure(os.path.join(app_support_dir(), "pgdata"))


def redis_dir():
    return _ensure(os.path.join(app_support_dir(), "redis"))


def temp_audio_dir():
    return _ensure(os.path.join(app_support_dir(), "temp_audio"))


def numba_cache_dir():
    return _ensure(os.path.join(app_support_dir(), "numba_cache"))


def backup_dir():
    return _ensure(os.path.join(app_support_dir(), "backup"))


def redis_socket_path():
    return os.path.join(redis_dir(), "redis.sock")


def control_socket_path():
    return os.path.join(app_support_dir(), "control.sock")


def pid_file():
    return os.path.join(app_support_dir(), "supervisor_pids.json")


def log_file():
    return os.path.join(logs_dir(), "audiomuse.log")


def model_dir():
    return os.path.join(resource_root(), "model")


def menubar_icon():
    return os.path.join(resource_root(), "assets", "menubar-icon.png")


def redis_binary():
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "redis-server")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
        "redis",
        platform.machine(),
        "redis-server",
    )


def fpcalc_binary():
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "fpcalc")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
        "fpcalc",
        platform.machine(),
        "fpcalc",
    )


def pg_bin_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "pgserver", "pginstall", "bin")
    import pgserver

    return os.path.join(os.path.dirname(os.path.abspath(pgserver.__file__)), "pginstall", "bin")
