# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one process supervisor the Linux and macOS standalone builds both run.

Both builds boot the same stack in the same order over the same POSIX process
group primitives, so they were maintained as two copies of one file and drifted:
the pipe-closing fix in child termination and the escalation from terminate to
kill when reaping a stale server only ever landed on Linux. This base carries
that union, and each platform subclass supplies only what genuinely differs: the
embedded-database module, the paths module and the child environment builder.

Windows is deliberately not a subclass: it has no process groups, restarts a
child from the log pump rather than the health loop, and its embedded database
hands back a connection mapping instead of a URL.

Main Features:
* Ordered boot behind a stop flag that is re-checked before every step
* Child termination that escalates SIGTERM to SIGKILL and closes the log pipe
* Orphan reaping from the pid file left by a previous run, plus any server still
  holding our data directory
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading

import service_roles
from macos.control_ipc import ControlServer
from native_common.supervisor_common import SupervisorCommonMixin
from native_common.supervisor_health import HealthLoopMixin

logger = logging.getLogger("audiomuse.supervisor")

ROLE_OF = service_roles.ROLE_OF

BOOT_ORDER = service_roles.BOOT_ORDER


class PosixSupervisor(SupervisorCommonMixin, HealthLoopMixin):
    db_backend = None
    env_builder = None

    def __init__(self):
        self._lock = threading.RLock()
        self._children = {}
        self._desired = set()
        self._database_url = None
        self._state = "stopped"
        self._control = ControlServer(self.paths.control_socket_path(), self.dispatch_control)
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
            self._database_url = self.db_backend.start_embedded(self.paths.pgdata_dir())
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
            logger.exception("Startup failed")
            self._log.exception("Startup failed")
            self.stop_all()
            raise

    def stop_all(self):
        with self._lock:
            if self._state in ("stopping", "stopped"):
                return
            self._state = "stopping"
            self._stop_requested.set()
            self._desired.clear()
        self._health_stop.set()
        self._join_workers()
        self._close_probe_conn()
        for name in reversed(BOOT_ORDER):
            self._terminate_named(name)
        try:
            self.db_backend.stop_embedded()
        except Exception:
            logger.exception("Error stopping embedded PostgreSQL")
        self._control.stop()
        self._clear_pidfile()
        with self._lock:
            self._state = "stopped"
        self._log.info("=== AudioMuse-AI stopped ===")

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
            argv = [sys.executable, f"--role={role}"]
            child_env = self.env_builder.build_child_env(role, self._database_url)
            self._spawn(name, argv, child_env)
            return True
        finally:
            self._release_start(name)

    def stop_child(self, name):
        if name not in ROLE_OF:
            return False
        with self._lock:
            was_desired = name in self._desired
            self._desired.discard(name)
        stopped = self._terminate_named(name)
        if not stopped and was_desired:
            with self._lock:
                self._desired.add(name)
        return stopped

    def restart_child(self, name):
        if name not in ROLE_OF or not self._terminate_named(name):
            return False
        return self.start_child(name)

    def _spawn(self, name, argv, child_env):
        if not self._terminate_named(name):
            raise RuntimeError(f"Could not terminate existing child {name}")
        proc = subprocess.Popen(
            argv,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=1,
            universal_newlines=True,
        )
        with self._lock:
            self._children[name] = proc
        threading.Thread(
            target=self._pump, args=(name, proc), name=f"log-{name}", daemon=True
        ).start()
        self._log.info("Started %s (pid %s)", name, proc.pid)

    def _pump(self, name, proc):
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                self._log.info("[%s] %s", name, line.rstrip())
        except Exception:
            pass

    def _terminate_named(self, name):
        with self._lock:
            proc = self._children.pop(name, None)
        if proc is None:
            return True
        terminated = False
        try:
            try:
                already_exited = proc.poll() is not None
            except Exception:
                logger.exception("Could not inspect %s before termination", name)
                already_exited = False
            if already_exited:
                terminated = True
                return True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                terminated = True
                return True
            except Exception:
                logger.exception("SIGTERM failed for %s", name)
            try:
                proc.wait(timeout=10)
                terminated = True
                return True
            except Exception:
                pass
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
                terminated = True
                return True
            except Exception:
                logger.exception("SIGKILL failed for %s", name)
                try:
                    terminated = proc.poll() is not None
                except Exception:
                    terminated = False
                if not terminated:
                    with self._lock:
                        self._children.setdefault(name, proc)
                return terminated
        finally:
            if terminated and proc.stdout is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

    def _ensure_postgres_healthy(self):
        if self._database_url is None:
            return
        try:
            pg_host, pg_port = self.env_builder._pg_conn_parts(self._database_url)
            healthy = self._probe_postgres(
                host=pg_host,
                port=pg_port,
                user="postgres",
                dbname="postgres",
            )
        except Exception:
            healthy = False
        if healthy:
            return
        self._close_probe_conn()
        self._log.warning("Embedded PostgreSQL unhealthy; restarting it")
        try:
            self._database_url = self.db_backend.ensure_embedded_running(self.paths.pgdata_dir())
            self._log.info("Embedded PostgreSQL restarted")
        except Exception:
            self._log.exception("Failed to restart embedded PostgreSQL")

    def _is_ours(self, cmdline):
        return (
            self.paths.APP_NAME in cmdline
            or "--role=" in cmdline
            or "postgres" in cmdline
        )

    def _reap_orphans(self):
        path = self.paths.pid_file()
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                pids = json.load(fh)
        except (OSError, ValueError):
            pids = {}
        try:
            import psutil
        except Exception:
            psutil = None
        for name, pid in pids.items():
            try:
                if psutil is not None:
                    proc = psutil.Process(pid)
                    if self._is_ours(" ".join(proc.cmdline())):
                        proc.terminate()
                        self._log.info("Reaped orphan %s (pid %s) from a previous run", name, pid)
                else:
                    comm = subprocess.check_output(
                        ["ps", "-p", str(pid), "-o", "command="], text=True
                    ).strip()
                    if self._is_ours(comm):
                        os.kill(pid, signal.SIGTERM)
                        self._log.info("Reaped orphan %s (pid %s) via ps fallback", name, pid)
            except Exception:
                continue
        self._reap_stale_infra()

    def _reap_stale_infra(self):
        try:
            import psutil
        except Exception:
            return
        me = os.getpid()
        pg_marker = self.paths.pgdata_dir()
        terminated = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == me:
                    continue
                cmd = " ".join(proc.info.get("cmdline") or [])
                if not cmd:
                    continue
                if ("postgres" in cmd or "pg_ctl" in cmd) and pg_marker in cmd:
                    proc.terminate()
                    terminated.append(proc)
                    self._log.info(
                        "Reaped stale %s (pid %s) referencing our data dir",
                        proc.info.get("name"),
                        proc.info["pid"],
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue
        if terminated:
            try:
                _gone, alive = psutil.wait_procs(terminated, timeout=5)
                for p in alive:
                    try:
                        p.kill()
                    except Exception:
                        continue
            except Exception:
                pass
        self._clear_pidfile()
