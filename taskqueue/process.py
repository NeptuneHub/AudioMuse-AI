# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Stopping a worker and everything it spawned, identically on every platform.

A job runs in the worker process itself, so cancelling means ending that process
and its children (ONNX sessions, a loky pool); the supervisor respawns the
worker. There is exactly one kill path.

POSIX signals children INDIVIDUALLY first and only signals the process GROUP as
its last act, with SIGKILL. The ordering is the whole point: this process is
itself in the group it signals, and signal.signal raises ValueError off the
main thread (where a cancel arrives), so a group-first signal would kill the
worker before the grace period ran. The group sweep is still needed last because
a reparented grandchild keeps its process group, which killpg reaches.

Windows has no signalable process groups, so it enumerates with psutil before
touching itself. taskkill /T /F is avoided, and os.kill(pid, 0) on win32
TERMINATES the probed process, so liveness probing is psutil-only there.

The temp sweep is pid-scoped because TEMP_DIR is SHARED by both workers in a
container; a folder is removed only once the pid in its name is provably dead.
The two layouts put that pid in different fields
(joblib_memmapping_folder_<pid> vs loky-<pid>-<suffix>).

Main Features:
* stop_hard kills this process tree and exits, on POSIX and Windows
* sweep_stale_temp_dirs clears joblib folders a previous hard kill leaked
"""

import logging
import os
import shutil
import signal
import sys
import time

import config

logger = logging.getLogger(__name__)

_JOBLIB_PREFIXES = ('joblib_memmapping_folder_', 'loky-')


def _kill_tree_posix(grace):
    pgid = os.getpgid(0)
    for child in _live_children():
        try:
            os.kill(child, signal.SIGTERM)
        except Exception:
            logger.debug("SIGTERM to child %s failed", child, exc_info=True)

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _live_children():
            break
        time.sleep(0.1)

    for child in _live_children():
        try:
            os.kill(child, signal.SIGKILL)
        except Exception:
            logger.debug("SIGKILL to child %s failed", child, exc_info=True)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        logger.debug("Flushing before the group sweep failed", exc_info=True)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        logger.debug("SIGKILL to the worker process group failed", exc_info=True)


def _live_children():
    try:
        import psutil

        return [child.pid for child in psutil.Process().children(recursive=True)]
    except Exception:
        return []


def _kill_tree_windows(grace):
    try:
        import psutil
    except Exception:
        logger.exception("psutil is unavailable; cannot reach this worker's children")
        return
    try:
        children = psutil.Process().children(recursive=True)
    except Exception:
        logger.exception("Could not enumerate this worker's children")
        return
    for child in children:
        try:
            child.terminate()
        except Exception:
            logger.debug("Terminating child %s failed", child, exc_info=True)
    _gone, alive = psutil.wait_procs(children, timeout=grace)
    for child in alive:
        try:
            child.kill()
        except Exception:
            logger.debug("Killing child %s failed", child, exc_info=True)


def stop_hard(reason):
    logger.warning("Stopping this worker and its process tree: %s", reason)
    grace = max(0.0, float(config.QUEUE_KILL_GRACE_SECONDS))
    try:
        if sys.platform == 'win32':
            _kill_tree_windows(grace)
        else:
            _kill_tree_posix(grace)
    except Exception:
        logger.exception("Tree kill failed; exiting anyway")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _owner_pid(entry):
    if entry.startswith('loky-'):
        parts = entry.split('-')
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    for part in entry.split('_'):
        if part.isdigit():
            return int(part)
    return None


def _pid_is_alive(pid):
    if sys.platform == 'win32':
        try:
            import psutil

            return psutil.pid_exists(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def sweep_stale_temp_dirs(temp_dir=None):
    base = temp_dir or config.TEMP_DIR
    if not base or not os.path.isdir(base):
        return 0
    removed = 0
    for entry in os.listdir(base):
        if not entry.startswith(_JOBLIB_PREFIXES):
            continue
        owner = _owner_pid(entry)
        if owner is None or _pid_is_alive(owner):
            continue
        path = os.path.join(base, entry)
        try:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        except Exception:
            logger.debug("Could not remove stale temp folder %s", path, exc_info=True)
    if removed:
        logger.info("Removed %d joblib folder(s) whose worker is gone.", removed)
    return removed
