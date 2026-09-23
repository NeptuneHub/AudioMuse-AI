# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The supervisor plumbing all three native builds share.

Logging setup, the boot thread, control dispatch, the pid file and the Flask
readiness wait used to be three copies that drifted: only Windows aborted the
readiness wait when a stop was requested, and only Linux and macOS survived an
unwritable pid file. Keeping one implementation removes the drift, and the two
places the platforms genuinely differ stay as class attributes a subclass sets:
``paths`` names the platform paths module and ``join_skips_main_thread`` marks
the Windows tray thread that must never be joined.

Main Features:
* One readiness wait that honours a stop request and reports the last error
* Control dispatch that runs every service and reports the aggregate result
* Pid file writes and removal that tolerate a read-only or missing state dir
* wait_until_stopped lets a quit or a CLI stop block until the stop finished
* references_pgdata tells a postgres/pg_ctl command line that runs OUR data
  directory (its -D argument) from one that runs a sibling such as a backup
  copy, which a substring match would have killed too.
* stale_role_child names a --role= child of this executable that no live
  supervisor owns, so every platform reaps the same orphans at startup
"""

import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request

from native_common.reverse_log import NewestFirstFileHandler

logger = logging.getLogger("audiomuse.supervisor")

FLASK_URL = "http://127.0.0.1:8000/"


def own_executables():
    names = [sys.executable, os.path.realpath(sys.executable)]
    return {os.path.normcase(name) for name in names if name}


def stale_role_child(argv, own_exes, roles, exe_of=None):
    if len(argv) < 2:
        return None
    role = None
    for arg in argv[1:]:
        if arg.startswith("--role="):
            role = arg[len("--role="):]
            break
    if role not in roles:
        return None
    if os.path.isabs(argv[0]) and os.path.normcase(argv[0]) in own_exes:
        return role
    if exe_of is None:
        return None
    try:
        exe = exe_of() or ""
    except Exception:
        return None
    if not exe:
        return None
    images = {os.path.normcase(exe), os.path.normcase(os.path.realpath(exe))}
    return role if not images.isdisjoint(own_exes) else None


def _pgdata_key(path):
    text = str(path).strip().strip('"').replace("\\", "/")
    return os.path.normcase(os.path.normpath(text)).rstrip("/\\")


def references_pgdata(argv, pgdata_dir):
    wanted = _pgdata_key(pgdata_dir)
    for index, arg in enumerate(argv):
        if arg == "-D" and index + 1 < len(argv):
            candidate = argv[index + 1]
        elif arg.startswith("-D") and len(arg) > 2:
            candidate = arg[2:]
        elif arg.startswith("--pgdata="):
            candidate = arg[len("--pgdata="):]
        else:
            continue
        if _pgdata_key(candidate) == wanted:
            return True
    return False


class SupervisorCommonMixin:
    paths = None
    flask_url = FLASK_URL
    join_skips_main_thread = False

    def _setup_logging(self):
        log = logging.getLogger("audiomuse.app")
        log.setLevel(logging.INFO)
        if not log.handlers:
            handler = NewestFirstFileHandler(self.paths.log_file())
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            log.addHandler(handler)
        log.propagate = False
        return log

    def is_running(self):
        return self._state == "running"

    def state(self):
        return self._state

    def wait_until_stopped(self, timeout):
        deadline = time.monotonic() + timeout
        while self._state == "stopping" and time.monotonic() < deadline:
            time.sleep(0.2)
        return self._state == "stopped"

    def start_in_background(self, on_ready=None, on_error=None):
        def _boot():
            try:
                self.start_all()
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)
                return
            if on_ready is not None and self.is_running():
                on_ready()

        self._boot_thread = threading.Thread(target=_boot, name="boot", daemon=True)
        self._boot_thread.start()
        return self._boot_thread

    def _join_workers(self):
        current = threading.current_thread()
        main = threading.main_thread()
        for thread in (self._boot_thread, self._health_thread):
            if thread is None or thread is current or not thread.is_alive():
                continue
            if self.join_skips_main_thread and thread is main:
                continue
            thread.join(timeout=30)

    def dispatch_control(self, action, services):
        if action not in ("restart", "stop", "start"):
            return False
        return self._apply_control(action, list(services))

    def _apply_control(self, action, services):
        operation = {
            "stop": self.stop_child,
            "start": self.start_child,
            "restart": self.restart_child,
        }[action]
        results = []
        for svc in services:
            try:
                results.append(bool(operation(svc)))
            except Exception:
                self._log.exception("Control %s failed for %s", action, svc)
                results.append(False)
        return all(results) if results else False

    def _wait_http(self, url, timeout=180):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            if self._stop_requested.is_set():
                return
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    resp.read(1)
                    return
            except urllib.error.HTTPError:
                return
            except Exception as exc:
                last = exc
            time.sleep(1)
        raise RuntimeError(f"Flask did not become ready at {url}: {last}")

    def _write_pidfile(self):
        with self._lock:
            pids = {name: proc.pid for name, proc in self._children.items() if proc.poll() is None}
        try:
            with open(self.paths.pid_file(), "w") as fh:
                json.dump(pids, fh)
        except OSError:
            logger.exception("Could not write pid file")

    def _clear_pidfile(self):
        try:
            os.unlink(self.paths.pid_file())
        except OSError:
            pass
