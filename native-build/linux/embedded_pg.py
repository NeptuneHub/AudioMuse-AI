# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Bundled-binary embedded PostgreSQL control for the Linux standalone build.

Used on non-x86_64 Linux (where pgserver is unavailable) to drive the vendored
PostgreSQL binaries resolved by ``linux.paths``. It initializes the data
directory on first run and starts/stops the server, mirroring the pgserver path
selected in ``linux.db_backend``.

Main Features:
* Runs initdb, pg_ctl start/stop against the bundled binaries under an RLock.
* Builds the child environment with the correct native library path so the
  bundled PostgreSQL finds its shared libraries.
"""

import logging
import os
import shutil
import subprocess
import threading

from linux import paths

logger = logging.getLogger("audiomuse.embedded_pg")

_lock = threading.RLock()
_data_dir = None

_READY_MARKER = "audiomuse_initialized"


def _pg_env():
    from linux import env as env_builder

    env = env_builder.restore_native_lib_path(dict(os.environ))
    libdir = paths.pg_lib_dir()
    if libdir:
        parts = [libdir, os.path.join(libdir, "postgresql"), env.get("LD_LIBRARY_PATH", "")]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, parts))
    return env


def _bin(name):
    return os.path.join(paths.pg_bin_dir(), name)


def _initialized(data_dir):
    if os.path.exists(os.path.join(data_dir, _READY_MARKER)):
        return True
    if os.path.exists(os.path.join(data_dir, "global", "pg_control")):
        try:
            with open(os.path.join(data_dir, _READY_MARKER), "w", encoding="utf-8") as fh:
                fh.write("ok\n")
        except OSError:
            pass
        return True
    return False


def _has_cluster_data(data_dir):
    return os.path.exists(os.path.join(data_dir, "global", "pg_control"))


def _reset_data_dir(data_dir):
    if not (os.path.isdir(data_dir) and os.listdir(data_dir)):
        return
    if _has_cluster_data(data_dir):
        raise RuntimeError(
            f"Refusing to wipe {data_dir}: it contains an existing PostgreSQL "
            "cluster (global/pg_control present). Back it up or remove it "
            "manually if you really want a fresh start."
        )
    logger.warning("Clearing incomplete PostgreSQL data dir %s before re-init", data_dir)
    for entry in os.listdir(data_dir):
        target = os.path.join(data_dir, entry)
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)
        except OSError:
            logger.exception("Could not remove %s", target)


def _dsn(data_dir):
    return f"postgresql://postgres@/postgres?host={data_dir}"


def _is_running(data_dir, env):
    proc = subprocess.run(
        [_bin("pg_ctl"), "-D", data_dir, "status"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def _run_checked(argv, env):
    proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        out = (proc.stdout or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"{argv[0]} failed (exit {proc.returncode}): {out}")


def _init_cluster(data_dir, env):
    _run_checked(
        [
            _bin("initdb"),
            "-D",
            data_dir,
            "-U",
            "postgres",
            "--auth=trust",
            "-E",
            "UTF8",
            "--locale=C",
        ],
        env,
    )
    conf = os.path.join(data_dir, "postgresql.conf")
    with open(conf, "a", encoding="utf-8") as fh:
        fh.write("\n# --- AudioMuse-AI embedded overrides ---\n")
        fh.write(f"unix_socket_directories = '{data_dir}'\n")
        fh.write("listen_addresses = ''\n")
    with open(os.path.join(data_dir, _READY_MARKER), "w", encoding="utf-8") as fh:
        fh.write("ok\n")


def start(data_dir):
    global _data_dir
    with _lock:
        env = _pg_env()
        if not _initialized(data_dir):
            logger.info("Initializing embedded PostgreSQL cluster at %s", data_dir)
            _reset_data_dir(data_dir)
            _init_cluster(data_dir, env)
        if not _is_running(data_dir, env):
            _run_checked([_bin("pg_ctl"), "-D", data_dir, "-w", "start"], env)
        _data_dir = data_dir
        return _dsn(data_dir)


def ensure_running(data_dir):
    return start(data_dir)


def stop():
    global _data_dir
    with _lock:
        if _data_dir is None:
            return
        env = _pg_env()
        try:
            subprocess.run(
                [_bin("pg_ctl"), "-D", _data_dir, "-w", "-m", "fast", "stop"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.exception("Error stopping embedded PostgreSQL")
        _data_dir = None
