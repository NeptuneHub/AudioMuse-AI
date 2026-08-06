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
``windows.launcher`` with a ``--role=``). It restarts crashed children, serves
the loopback TCP control server, and tears everything down on shutdown. The
Linux/macOS supervisors are the platform-specific siblings.

Main Features:
* Ordered boot, health polling and automatic restart of Flask + queue children.
* Runs the TCP control server and writes newest-first rotating logs.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


import service_roles
from windows import db_backend
from windows import env as env_builder
from windows import paths
from macos.reverse_log import NewestFirstFileHandler
from native_common.supervisor_health import HealthLoopMixin
from windows.control_server import ControlServer

logger = logging.getLogger("audiomuse.supervisor")

FLASK_URL = "http://127.0.0.1:8000/"

ROLE_OF = service_roles.ROLE_OF

BOOT_ORDER = service_roles.BOOT_ORDER


class ProcessSupervisor(HealthLoopMixin):
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

    def _setup_logging(self):
        log = logging.getLogger("audiomuse.app")
        log.setLevel(logging.INFO)
        if not log.handlers:
            handler = NewestFirstFileHandler(paths.log_file())
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            log.addHandler(handler)
        log.propagate = False
        return log

    def is_running(self):
        return self._state == "running"

    def state(self):
        return self._state

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
                    self._wait_http(FLASK_URL, timeout=180)
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
        self._join_workers()
        self._control.stop()
        for name in list(self._children.keys()):
            self._stop_child(name)
        db_backend.stop_embedded()
        self._reap_orphans()
        self._remove_pidfile()
        with self._lock:
            self._state = "stopped"
        self._log.info("=== AudioMuse-AI stopped ===")

    def _join_workers(self):
        current = threading.current_thread()
        main = threading.main_thread()
        for thread in (self._boot_thread, self._health_thread):
            if (
                thread is not None
                and thread is not current
                and thread is not main
                and thread.is_alive()
            ):
                thread.join(timeout=30)

    def start_child(self, name):
        role = ROLE_OF.get(name)
        if role is None:
            return False
        with self._lock:
            if self._state not in ("starting", "running"):
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
            popen = self._children.pop(name, None)
            was_desired = name in self._desired
            self._desired.discard(name)
        if popen is None:
            return True

        def _stopped_or_restore():
            try:
                stopped = popen.poll() is not None
            except Exception:
                stopped = False
            if not stopped:
                with self._lock:
                    self._children.setdefault(name, popen)
                    if was_desired:
                        self._desired.add(name)
            return stopped

        try:
            if popen.poll() is not None:
                return True
        except Exception:
            self._log.exception("Could not inspect %s before termination", name)
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
        with self._lock:
            if self._children.get(name) is not popen:
                return
            self._children.pop(name, None)
            restart = name in self._desired and self._state == "running"
        if restart:
            self._log.warning("%s exited unexpectedly -- restarting", name)
            try:
                self.start_child(name)
            except Exception:
                self._log.exception("Failed to restart %s", name)

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
        while time.time() < deadline:
            if self._stop_requested.is_set():
                return
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except urllib.error.HTTPError:
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"Timed out waiting for {url}")

    def _ensure_postgres_healthy(self):
        if not self._db_conn:
            return
        try:
            import psycopg2

            conn = psycopg2.connect(
                host=self._db_conn["host"],
                port=self._db_conn["port"],
                user=self._db_conn["user"],
                password=self._db_conn["password"],
                dbname=self._db_conn["dbname"],
                connect_timeout=3,
            )
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            finally:
                conn.close()
            return
        except Exception:
            pass
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
        pg_marker = paths.pgdata_dir().lower()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == me:
                    continue
                cmd = " ".join(proc.info.get("cmdline") or []).lower()
                if not cmd:
                    continue
                stale_pg = ("postgres" in cmd or "pg_ctl" in cmd) and pg_marker in cmd
                if stale_pg:
                    self._log.info(
                        "Reaping orphan %s (pid=%d) referencing our data dir",
                        proc.info.get("name"),
                        proc.info["pid"],
                    )
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue

    def _write_pidfile(self):
        pids = {name: proc.pid for name, proc in self._children.items()}
        with open(paths.pid_file(), "w") as fh:
            json.dump(pids, fh)

    def _remove_pidfile(self):
        try:
            os.unlink(paths.pid_file())
        except OSError:
            pass
