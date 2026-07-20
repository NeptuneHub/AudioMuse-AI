# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Filesystem path resolution for the Windows standalone build.

Resolves the bundle resource root (PyInstaller ``_MEIPASS`` or the source
repo) and the per-user data locations under ``%LOCALAPPDATA%`` (falling back to
``%PROGRAMDATA%`` when the path contains spaces), so the Windows launcher,
supervisor and embedded-Postgres modules agree on where pgdata, redis, logs,
models and temp files live. The Linux/macOS ``paths`` modules are the
platform-specific siblings.

Main Features:
* ``resource_root``, tray-icon path and per-user data directories under LOCALAPPDATA.
* Fixed loopback ports (pg 5432, redis 6379, control 8001) plus persisted random
  DB/Redis passwords cached under a per-user ``secrets`` dir.
"""

import os
import platform
import secrets
import sys
from urllib.parse import quote

APP_NAME = "AudioMuse-AI"


def resource_root():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", _repo_root())
    return _repo_root()


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tray_icon():
    return os.path.join(resource_root(), "assets", "AudioMuse-AI.ico")


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def app_support_dir():
    local_appdata = os.environ.get(
        "LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local")
    )
    root = os.path.join(local_appdata, APP_NAME)
    if " " in root:
        root = os.path.join(os.environ.get("PROGRAMDATA", "C:\\ProgramData"), APP_NAME)
    return _ensure(root)


def logs_dir():
    d = os.path.join(app_support_dir(), "logs")
    return _ensure(d)


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


def secrets_dir():
    return _ensure(os.path.join(app_support_dir(), "secrets"))


def _secret(name):
    path = os.path.join(secrets_dir(), name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            value = fh.read().strip()
        if value:
            return value
    except OSError:
        pass
    value = secrets.token_urlsafe(32)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(value)
    return value


def db_password():
    return _secret("pg_password")


def redis_password():
    return _secret("redis_password")


def redis_port():
    return 6379


def pg_port():
    return 5432


def pg_start_timeout():
    return 120


def redis_url():
    return f"redis://:{quote(redis_password(), safe='')}@127.0.0.1:{redis_port()}/0"


def control_port():
    return 8001


def pid_file():
    return os.path.join(app_support_dir(), "supervisor_pids.json")


def supervisor_lock_path():
    return os.path.join(app_support_dir(), "supervisor.lock")


def log_file():
    return os.path.join(logs_dir(), "audiomuse.log")


def model_dir():
    return os.path.join(resource_root(), "model")


def redis_binary():
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "redis-server.exe")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
        "redis",
        platform.machine().lower(),
        "redis-server.exe",
    )


def fpcalc_binary():
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "fpcalc.exe")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
        "fpcalc",
        platform.machine().lower(),
        "fpcalc.exe",
    )


def _pgserver_pginstall():
    if getattr(sys, "frozen", False):
        cand = os.path.join(resource_root(), "pgserver", "pginstall")
        return cand if os.path.isdir(cand) else None
    try:
        import pgserver

        cand = os.path.join(os.path.dirname(pgserver.__file__), "pginstall")
        return cand if os.path.isdir(cand) else None
    except Exception:
        return None


def pg_bin_dir():
    pginstall = _pgserver_pginstall()
    if pginstall:
        return os.path.join(pginstall, "bin")
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "pgsql", "bin")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
        "postgres",
        platform.machine().lower(),
        "bin",
    )


def pg_lib_dir():
    pginstall = _pgserver_pginstall()
    if pginstall:
        return os.path.join(pginstall, "lib")
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "pgsql", "lib")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
        "postgres",
        platform.machine().lower(),
        "lib",
    )
