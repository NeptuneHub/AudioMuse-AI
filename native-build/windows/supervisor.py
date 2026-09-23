# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Process supervisor for the Windows standalone build.

Boots and monitors the full local stack in dependency order: embedded
PostgreSQL (via ``windows.db_backend``), the Flask/waitress server and the
queue worker/maintenance/control-listener children (each re-spawned from
``windows.launcher`` with a ``--role=``). Logging, the boot thread, control
dispatch, the pid file and the Flask readiness wait come from
``native_common.supervisor_common``; what stays here is what Windows genuinely
does differently: console process groups instead of POSIX process groups, a
restart driven by the log pump rather than the health loop, a loopback TCP
control server, and an embedded database that reports a connection mapping
instead of a URL.

Main Features:
* Ordered boot, health polling and automatic restart of Flask + queue children.
* Runs the TCP control server and writes newest-first rotating logs.
* A console close, logoff or shutdown stops every service and flushes the log,
  instead of letting the auto-restart resurrect children that then outlive the
  supervisor as silent orphans. Windows kills every child attached to the
  console at once and only then reaches this handler, so the log pump never
  restarts a child that exited with STATUS_CONTROL_C_EXIT (killed with the
  console) and otherwise waits one second for a stop request first; the
  handler kills the children outright, stops the embedded PostgreSQL and
  clears the pid file BEFORE the orderly stop and its thread joins, because
  Windows allows it five seconds and CTRL_BREAK has no console left to travel
  through.
* Once a stop is requested no child is started or restarted (health loop
  included, nor after a console-kill exit), and stop_all always ends stopped.
* Startup reaps role children left behind by an earlier supervisor, so a torn
  down instance never leaves orphan workers draining the queue unseen.
"""

import os
import signal
import subprocess
import sys
import threading

import service_roles
from windows import db_backend
from windows import env as env_builder
from windows import paths
from native_common.supervisor_common import (
    SupervisorCommonMixin,
    own_executables,
    references_pgdata,
    stale_role_child,
)
from native_common.supervisor_health import HealthLoopMixin
from windows.control_server import ControlServer

ROLE_OF = service_roles.ROLE_OF

BOOT_ORDER = service_roles.BOOT_ORDER

_CONSOLE_STOP_EVENTS = {2: "console closed", 5: "user logged off", 6: "system shutting down"}
_console_handlers = []
_RESTART_GRACE_SECONDS = 1.0
_CONSOLE_KILL_EXIT_CODES = {0xC000013A, 0xC000013A - (1 << 32)}


class ProcessSupervisor(SupervisorCommonMixin, HealthLoopMixin):
    paths = paths
    join_skips_main_thread = True
    no_restart_exit_codes = frozenset(_CONSOLE_KILL_EXIT_CODES)

    def __init__(self):
        self._lock = threading.RLock()
        self._children = {}
        self._desired = set()
        self._db_conn = None
        self._state = "stopped"
        self._control = ControlServer(
            host="127.0.0.1",
            port=paths.control_port(),
            dispatch=self.dispatch_control,
            supervisor=self,
        )
        self._health_thread = None
        self._health_stop = threading.Event()
        self._stop_requested = threading.Event()
        self._boot_thread = None
        self._log = self._setup_logging()

    def start_all(self):
        with self._lock:
            if self._state in ("running", "starting"):
                return
            self._state = "starting"
            self._stop_requested.clear()
        self._log.info("=== AudioMuse-AI starting ===")
        try:
            self._reap_orphans()
            self._control.start()
            if self._stop_requested.is_set():
                return
            self._db_conn = db_backend.start_embedded(paths.pgdata_dir())
            self._log.info("Embedded PostgreSQL ready")
            if self._stop_requested.is_set():
                return
            for name in BOOT_ORDER:
                if self._stop_requested.is_set():
                    return
                self.start_child(name)
                if name == service_roles.SERVICE_FLASK:
                    self._wait_http(self.flask_url, timeout=180)
            self._write_pidfile()
            with self._lock:
                if self._stop_requested.is_set():
                    return
                self._state = "running"
            self._start_health_loop()
            self._log.info("=== AudioMuse-AI running ===")
        except Exception:
            self._log.exception("Startup failed")
            self.stop_all()
            raise

    def stop_all(self):
        self._stop_requested.set()
        self._health_stop.set()
        with self._lock:
            if self._state in ("stopped", "stopping"):
                return
            self._state = "stopping"
        self._log.info("=== AudioMuse-AI stopping ===")
        try:
            self._join_workers()
            self._control.stop()
            for name in list(self._children.keys()):
                self._stop_child(name)
            db_backend.stop_embedded()
            self._reap_orphans()
            self._clear_pidfile()
        finally:
            with self._lock:
                self._state = "stopped"
        self._log.info("=== AudioMuse-AI stopped ===")

    def install_console_handler(self):
        if sys.platform != "win32":
            return False
        import ctypes
        from ctypes import wintypes

        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        callback = handler_type(self._console_event)
        _console_handlers.append(callback)
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCtrlHandler.argtypes = [handler_type, wintypes.BOOL]
        kernel32.SetConsoleCtrlHandler.restype = wintypes.BOOL
        return bool(kernel32.SetConsoleCtrlHandler(callback, True))

    def _console_event(self, ctrl_type):
        reason = _CONSOLE_STOP_EVENTS.get(int(ctrl_type))
        if reason is None:
            return False
        self._stop_requested.set()
        self._log.warning("=== AudioMuse-AI %s: stopping every service ===", reason)
        self._flush_log()
        self._kill_children()
        try:
            db_backend.stop_embedded()
        except Exception:
            self._log.exception("Stopping the embedded PostgreSQL after the %s event failed", reason)
        self._clear_pidfile()
        self._flush_log()
        try:
            self.stop_all()
        except Exception:
            self._log.exception("Stopping after the %s event failed", reason)
        self._flush_log()
        return True

    def _kill_children(self):
        with self._lock:
            self._desired.clear()
            children = list(self._children.items())
        for name, popen in children:
            try:
                if popen.poll() is None:
                    popen.kill()
                    popen.wait(timeout=2)
            except Exception:
                self._log.exception("Could not kill %s", name)

    def _flush_log(self):
        for handler in list(getattr(self._log, "handlers", None) or []):
            try:
                handler.flush()
            except Exception:
                self._log.debug("Log flush failed", exc_info=True)

    def start_child(self, name):
        role = ROLE_OF.get(name)
        if role is None:
            return False
        with self._lock:
            if self._state not in ("starting", "running") or self._stop_requested.is_set():
                return False
            existing = self._children.get(name)
            if existing is not None and existing.poll() is None:
                self._desired.add(name)
                return True
            self._desired.add(name)
        if not self._claim_start(name):
            return True
        try:
            self._log.info("Starting %s (role=%s)", name, role)
            db_conn = db_backend.ensure_embedded_running(paths.pgdata_dir())
            env = env_builder.build_child_env(role, db_conn)
            exe = sys.executable if not getattr(sys, "frozen", False) else sys.argv[0]
            cmd = [exe, f"--role={role}"]
            if not self._terminate_named(name):
                raise RuntimeError(f"Could not terminate existing child {name}")
            popen = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            with self._lock:
                self._children[name] = popen
            threading.Thread(
                target=self._pump, args=(name, popen), name=f"pump-{name}", daemon=True
            ).start()
            return True
        finally:
            self._release_start(name)

    def _stop_child(self, name):
        with self._lock:
            was_desired = name in self._desired
            self._desired.discard(name)
        stopped = self._terminate_named(name)
        if not stopped and was_desired:
            with self._lock:
                self._desired.add(name)
        return stopped

    def _terminate_named(self, name):
        with self._lock:
            popen = self._children.get(name)
            if popen is None:
                return True
            try:
                already_exited = popen.poll() is not None
            except Exception:
                self._log.exception("Could not inspect %s before termination", name)
                already_exited = False
            if already_exited:
                return True
            self._children.pop(name, None)

        def _stopped_or_restore():
            try:
                stopped = popen.poll() is not None
            except Exception:
                stopped = False
            if not stopped:
                with self._lock:
                    self._children.setdefault(name, popen)
            return stopped

        self._log.info("Stopping %s (pid=%d)", name, popen.pid)
        try:
            if sys.platform == "win32":
                popen.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                popen.send_signal(signal.SIGTERM)
            popen.wait(timeout=15)
            return True
        except Exception:
            try:
                popen.kill()
            except Exception:
                self._log.exception("Could not kill %s", name)
                return _stopped_or_restore()
            try:
                popen.wait(timeout=5)
                return True
            except Exception:
                self._log.exception("Timed out waiting for killed child %s", name)
                return _stopped_or_restore()

    def stop_child(self, name):
        if name not in ROLE_OF:
            return False
        return self._stop_child(name)

    def restart_child(self, name):
        if name not in ROLE_OF or not self._stop_child(name):
            return False
        return self.start_child(name)

    def _pump(self, name, popen):
        for line in popen.stdout:
            self._log.info("[%s] %s", name, line.rstrip())
        popen.wait()
        if popen.returncode in _CONSOLE_KILL_EXIT_CODES:
            self._log.warning("%s was killed with the console; not restarting it", name)
            return
        if self._stop_requested.wait(_RESTART_GRACE_SECONDS):
            return
        with self._lock:
            if self._children.get(name) is not popen:
                return
            restart = (
                name in self._desired
                and self._state == "running"
                and not self._stop_requested.is_set()
            )
        if restart:
            self._log.warning("%s exited unexpectedly -- restarting", name)
            try:
                self.start_child(name)
            except Exception:
                self._log.exception("Failed to restart %s", name)

    def _ensure_postgres_healthy(self):
        if not self._db_conn:
            return
        try:
            healthy = self._probe_postgres(
                host=self._db_conn["host"],
                port=self._db_conn["port"],
                user=self._db_conn["user"],
                password=self._db_conn["password"],
                dbname=self._db_conn["dbname"],
            )
        except Exception:
            healthy = False
        if healthy:
            return
        self._close_probe_conn()
        self._log.warning("Embedded PostgreSQL unhealthy; restarting it")
        try:
            self._db_conn = db_backend.ensure_embedded_running(paths.pgdata_dir())
            self._log.info("Embedded PostgreSQL restarted")
        except Exception:
            self._log.exception("Failed to restart embedded PostgreSQL")

    def _reap_orphans(self):
        try:
            import psutil
        except Exception:
            return
        me = os.getpid()
        pgdata = paths.pgdata_dir()
        own_exes = own_executables()
        with self._lock:
            live_children = {
                popen.pid for popen in self._children.values() if popen.poll() is None
            }
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                pid = proc.info["pid"]
                if pid == me or pid in live_children:
                    continue
                argv = proc.info.get("cmdline") or []
                cmd = " ".join(argv).lower()
                if not cmd:
                    continue
                if ("postgres" in cmd or "pg_ctl" in cmd) and references_pgdata(argv, pgdata):
                    self._log.info(
                        "Reaping orphan %s (pid=%d) referencing our data dir",
                        proc.info.get("name"),
                        pid,
                    )
                    proc.terminate()
                else:
                    role = stale_role_child(argv, own_exes, ROLE_OF.values(), proc.exe)
                    if role:
                        self._log.warning(
                            "Reaping orphan --role=%s (pid=%d) left behind by an earlier supervisor",
                            role,
                            pid,
                        )
                        proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue
