# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Process supervisor for the macOS standalone build.

Boots and monitors the full local stack in dependency order: embedded
PostgreSQL (via the shared ``database`` module), Redis, the Flask/waitress
server and the RQ worker/janitor/restart-listener children (each re-spawned
from ``macos.launcher`` with a ``--role=``). It restarts crashed children,
serves the Unix-socket control server, and tears everything down on shutdown.
The Linux/Windows supervisors are the platform-specific siblings.

Main Features:
* Ordered boot, health polling and automatic restart of Flask + RQ children.
* Runs the control-socket server and writes newest-first rotating logs.
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

import redis as redis_lib

import database
import taskqueue
from macos import env as env_builder
from macos import paths
from macos.control_ipc import ControlServer
from macos.reverse_log import NewestFirstFileHandler

logger = logging.getLogger("audiomuse.supervisor")

FLASK_URL = "http://127.0.0.1:8000/"

ROLE_OF = {
    "flask": "flask",
    "rq-worker-high": "worker-high",
    "rq-worker-default": "worker-default",
    "rq-janitor": "janitor",
    "restart-listener": "restart-listener",
}

BOOT_ORDER = ["flask", "rq-worker-high", "rq-worker-default", "rq-janitor", "restart-listener"]


class ProcessSupervisor:
    def __init__(self):
        self._lock = threading.RLock()
        self._children = {}
        self._desired = set()
        self._database_url = None
        self._redis_url = None
        self._state = "stopped"
        self._control = ControlServer(paths.control_socket_path(), self.dispatch_control)
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
            self._database_url = database.start_embedded(paths.pgdata_dir())
            self._log.info("Embedded PostgreSQL ready")
            if self._stop_requested.is_set():
                return
            self._start_redis()
            self._log.info("Embedded Redis ready")
            for name in BOOT_ORDER:
                if self._stop_requested.is_set():
                    return
                self.start_child(name)
                if name == "flask":
                    self._wait_http(FLASK_URL, timeout=180)
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
        for name in reversed(BOOT_ORDER):
            self._terminate_named(name)
        self._terminate_named("redis")
        try:
            database.stop_embedded()
        except Exception:
            logger.exception("Error stopping embedded PostgreSQL")
        self._control.stop()
        self._clear_pidfile()
        with self._lock:
            self._state = "stopped"
        self._log.info("=== AudioMuse-AI stopped ===")

    def _join_workers(self):
        current = threading.current_thread()
        for thread in (self._boot_thread, self._health_thread):
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=30)

    def _start_redis(self):
        argv, url = taskqueue.build_embedded_redis_argv(
            paths.redis_binary(), paths.redis_socket_path(), paths.redis_dir()
        )
        self._redis_url = url
        self._spawn("redis", argv, dict(os.environ))
        self._wait_redis(timeout=60)

    def _wait_redis(self, timeout):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                if redis_lib.Redis(unix_socket_path=paths.redis_socket_path()).ping():
                    return
            except Exception as exc:
                last = exc
            time.sleep(0.5)
        raise RuntimeError(f"Embedded Redis did not become ready: {last}")

    def start_child(self, name):
        role = ROLE_OF.get(name)
        if role is None:
            return False
        with self._lock:
            if self._state not in ("starting", "running"):
                return False
            self._desired.add(name)
        argv = [sys.executable, f"--role={role}"]
        child_env = env_builder.build_child_env(role, self._database_url, self._redis_url)
        self._spawn(name, argv, child_env)
        return True

    def stop_child(self, name):
        with self._lock:
            self._desired.discard(name)
        self._terminate_named(name)
        return True

    def restart_child(self, name):
        self._terminate_named(name)
        return self.start_child(name)

    def _spawn(self, name, argv, child_env):
        self._terminate_named(name)
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
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            logger.exception("SIGTERM failed for %s", name)
        try:
            proc.wait(timeout=10)
            return
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass

    def _start_health_loop(self):
        self._health_stop.clear()
        self._health_thread = threading.Thread(target=self._health_loop, name="health", daemon=True)
        self._health_thread.start()

    def _health_loop(self):
        while not self._health_stop.wait(5):
            if self._state != "running":
                continue
            self._ensure_postgres_healthy()
            self._ensure_redis_healthy()
            for name in list(self._desired):
                with self._lock:
                    proc = self._children.get(name)
                if proc is not None and proc.poll() is not None:
                    self._log.warning("%s exited (code %s); restarting", name, proc.returncode)
                    self.start_child(name)

    def _ensure_postgres_healthy(self):
        if self._database_url is None:
            return
        try:
            import psycopg2

            conn = psycopg2.connect(self._database_url, connect_timeout=3)
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
            self._database_url = database.ensure_embedded_running(paths.pgdata_dir())
            self._log.info("Embedded PostgreSQL restarted")
        except Exception:
            self._log.exception("Failed to restart embedded PostgreSQL")

    def _ensure_redis_healthy(self):
        with self._lock:
            proc = self._children.get("redis")
        if proc is not None and proc.poll() is None:
            try:
                if redis_lib.Redis(
                    unix_socket_path=paths.redis_socket_path(),
                    socket_connect_timeout=2,
                    socket_timeout=2,
                ).ping():
                    return
            except Exception:
                pass
        self._log.warning("Embedded Redis unhealthy; restarting it")
        try:
            self._start_redis()
            self._log.info("Embedded Redis restarted")
        except Exception:
            self._log.exception("Failed to restart embedded Redis")

    def dispatch_control(self, action, services):
        if action not in ("restart", "stop", "start"):
            return False
        threading.Thread(
            target=self._apply_control,
            args=(action, list(services)),
            name=f"control-{action}",
            daemon=True,
        ).start()
        return True

    def _apply_control(self, action, services):
        for svc in services:
            try:
                if action == "stop":
                    self.stop_child(svc)
                elif action == "start":
                    self.start_child(svc)
                elif action == "restart":
                    self.restart_child(svc)
            except Exception:
                logger.exception("Control %s failed for %s", action, svc)

    def _wait_http(self, url, timeout):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
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
            with open(paths.pid_file(), "w") as fh:
                json.dump(pids, fh)
        except OSError:
            logger.exception("Could not write pid file")

    def _clear_pidfile(self):
        try:
            if os.path.exists(paths.pid_file()):
                os.unlink(paths.pid_file())
        except OSError:
            pass

    def _reap_orphans(self):
        path = paths.pid_file()
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
                    cmdline = " ".join(proc.cmdline())
                    if (
                        paths.APP_NAME in cmdline
                        or "--role=" in cmdline
                        or "redis-server" in cmdline
                    ):
                        proc.terminate()
                        self._log.info("Reaped orphan %s (pid %s) from a previous run", name, pid)
                else:
                    comm = subprocess.check_output(
                        ["ps", "-p", str(pid), "-o", "command="], text=True
                    ).strip()
                    if (
                        paths.APP_NAME in comm
                        or "--role=" in comm
                        or "redis-server" in comm
                        or "postgres" in comm
                    ):
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
        redis_marker = paths.redis_dir()
        pg_marker = paths.pgdata_dir()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == me:
                    continue
                cmd = " ".join(proc.info.get("cmdline") or [])
                if not cmd:
                    continue
                stale_redis = "redis-server" in cmd and redis_marker in cmd
                stale_pg = ("postgres" in cmd or "pg_ctl" in cmd) and pg_marker in cmd
                if stale_redis or stale_pg:
                    proc.terminate()
                    self._log.info(
                        "Reaped stale %s (pid %s) referencing our data dir",
                        proc.info.get("name"),
                        proc.info["pid"],
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue
        self._clear_pidfile()
