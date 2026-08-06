# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Stopping a worker and everything it spawned, identically on every platform.

A job runs in the worker process itself, so cancelling it means ending that
process, and ending it means taking its children with it: an analysis job holds
ONNX sessions and a loky pool whose workers are separate OS processes, and a
survivor would keep a GPU or a whole CPU busy with work nobody is waiting for.
The supervisor then respawns the worker, which is the same mechanism used to
recycle it after ``QUEUE_MAX_JOBS`` jobs, so there is exactly one kill path.

The POSIX branch signals children INDIVIDUALLY first and only signals the
process GROUP as its last act, with SIGKILL. The ordering is the whole point.
This process is itself in the group it signals, so a group SIGTERM ends it at
that instruction unless the process first sets SIGTERM to SIG_IGN - and
``signal.signal`` raises ValueError off the main thread, which is exactly where
a cancel arrives (the notification is dispatched on the Listener thread). That
made the group signal kill the worker before the grace period, the SIGKILL
escalation and the clean exit could run, leaving alive the loky and ONNX
children this function exists to reap. Doing the per-child work first needs no
signal disposition at all and therefore behaves identically on every thread.

The group sweep is still needed and still last: a child that exits while its own
child lives leaves a grandchild reparented to PID 1, where ``psutil`` can no
longer find it, but which KEEPS its process group - so ``killpg`` reaches it and
enumeration does not. SIGKILL needs no handler, so nothing can be skipped, and
this process dying with the group is the intended exit (every supervisor entry
is ``autorestart=true``, so the exit status does not matter). It is only correct
because the worker is a group leader in every environment that starts it -
supervisord with ``killasgroup=true`` in the container, ``start_new_session=True``
in the Linux and macOS supervisors, ``CREATE_NEW_PROCESS_GROUP`` on Windows - so
a change to any of those spawn flags would make this signal the supervisor's own
group instead. The unit tests assert those flags for that reason.

Windows has no process groups to signal this way, so it enumerates with psutil
BEFORE touching itself, terminates, waits, then kills. Shelling out to
``taskkill /T /F`` was rejected: spawning a console process from inside a frozen
PyInstaller bundle is exactly the fragile path ``frozen_children`` exists to
avoid, and psutil is already a dependency. Liveness probing is psutil-only there
too: CPython implements ``os.kill(pid, 0)`` on win32 as OpenProcess plus
TerminateProcess, so the POSIX "signal 0 is a probe" idiom TERMINATES the process
it asks about - it killed the sibling worker whose joblib folder was being
checked, and reported every genuinely dead pid as alive because OpenProcess
fails there with a plain OSError rather than ProcessLookupError.

The sweep is pid-scoped for the same reason. TEMP_DIR is SHARED by both worker
processes in a container, so an unqualified sweep at boot deleted memmap folders
belonging to the sibling worker that was still using them - and the default
worker recycles every ``QUEUE_MAX_JOBS`` jobs, so that happened routinely
mid-analysis. A folder is removed only once the pid in its name is provably
dead, and an unreadable name is left alone. The two layouts put that pid in
different places: ``joblib_memmapping_folder_<pid>_<...>`` in the first numeric
underscore field, ``loky-<pid>-<random suffix>`` in the SECOND hyphen field
rather than the last, so reading the last field returned the random suffix and
swept no loky folder at all - except when mkdtemp happened to produce an
all-digit suffix, where an unrelated number decided the fate of a live pool.

Main Features:
* ``stop_hard`` kills this process tree and exits, on POSIX and on Windows
* ``sweep_stale_temp_dirs`` clears joblib folders a previous hard kill leaked
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
