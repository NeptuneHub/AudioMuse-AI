"""Process supervisor for the standalone Windows app.

Windows counterpart of ``native-build/linux/supervisor.py`` -- the logic is platform-agnostic
(stdlib ``subprocess``/``signal``/``os.kill`` + ``psutil``), so this is a near
copy that swaps ``linux`` path/env helpers for the ``windows`` ones and adapts
the control server to use a TCP socket instead of a Unix socket (Windows has no
AF_UNIX).

It owns the full lifecycle of everything the container deployment runs under
supervisord: embedded PostgreSQL (via pgserver or a bundled PostgreSQL),
embedded Redis (the bundled binary), the waitress/Flask web server, the two RQ
workers, the janitor and the restart listener. It boots them in dependency order,
captures every child's output into one rotating log, replaces children that exit
while running (supervisord ``autorestart``), and tears the whole tree down
without leaving orphaned Postgres/Redis processes behind.
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

from windows import db_backend
from windows import env as env_builder
from windows import paths
from macos.reverse_log import NewestFirstFileHandler
from windows.control_server import ControlServer

logger = logging.getLogger("audiomuse.supervisor")

FLASK_URL = "http://127.0.0.1:8000/"

ROLE_OF = {
    "flask": "flask",
    "rq-worker-high": "worker-high",
    "rq-worker-default": "worker-default",
    "rq-janitor": "janitor",
    "restart-listener": "restart-listener",
}

# On Windows, RQ's ``SpawnWorker`` uses ``os.spawnv()`` instead of ``os.fork()``.
# Full worker pool is available on all platforms.
BOOT_ORDER = ["flask", "rq-worker-high", "rq-worker-default", "rq-janitor", "restart-listener"]


class ProcessSupervisor:
    def __init__(self):
        self._lock = threading.RLock()
        self._children = {}
        self._desired = set()
        self._db_conn = None
        self._redis_url = None
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

    def start_all(self):
        with self._lock:
            if self._state in ("running", "starting"):
                return
            self._state = "starting"
            self._stop_requested.clear()
            # Record the boot thread so stop_all can join it before the teardown
            # sweep: a stop racing a boot would otherwise sweep _children while
            # this thread is still spawning, leaking the late children as orphans.
            self._boot_thread = threading.current_thread()
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
        # Wait for any in-flight boot/health work to stop spawning before we
        # sweep: a child spawned after the sweep would survive as an orphan.
        self._join_workers()
        self._control.stop()
        for name in list(self._children.keys()):
            self._stop_child(name)
        db_backend.stop_embedded()
        self._stop_redis()
        self._reap_orphans()
        self._remove_pidfile()
        with self._lock:
            self._state = "stopped"
        self._log.info("=== AudioMuse-AI stopped ===")

    def _join_workers(self):
        """Wait for the boot and health threads to finish before teardown.

        Skips the current thread so stop_all() called from within start_all()'s
        failure handler (boot thread) never deadlocks on itself."""
        current = threading.current_thread()
        for thread in (self._boot_thread, self._health_thread):
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=30)

    def start_child(self, name):
        role = ROLE_OF.get(name)
        if role is None:
            return False
        with self._lock:
            # Refuse to spawn once a stop is in progress -- otherwise the boot
            # loop, the restart pump, or a control request could create a child
            # after stop_all's teardown sweep, leaking it as an orphan.
            if self._state not in ("starting", "running"):
                return False
            self._desired.add(name)
        self._log.info("Starting %s (role=%s)", name, role)
        db_conn = db_backend.ensure_embedded_running(paths.pgdata_dir())
        redis_url = self._ensure_redis_running()
        env = env_builder.build_child_env(role, db_conn, redis_url)
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
        threading.Thread(target=self._pump, args=(name, popen), name=f"pump-{name}", daemon=True).start()
        return True

    def _stop_child(self, name):
        with self._lock:
            popen = self._children.pop(name, None)
            self._desired.discard(name)
        if popen is None:
            return
        self._log.info("Stopping %s (pid=%d)", name, popen.pid)
        try:
            if sys.platform == "win32":
                popen.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                popen.send_signal(signal.SIGTERM)
            popen.wait(timeout=15)
        except Exception:
            try:
                popen.kill()
            except Exception:
                pass

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
        """Handle a restart/stop/start request from the web UI."""
        if action == "restart":
            for svc in services:
                if svc in self._children:
                    self._stop_child(svc)
                    self.start_child(svc)
            return True
        elif action == "stop":
            for svc in services:
                self._stop_child(svc)
            return True
        elif action == "start":
            for svc in services:
                if svc not in self._children:
                    self.start_child(svc)
            return True
        return False

    # --- embedded Redis ---

    def _start_redis(self):
        redis_bin = paths.redis_binary()
        redis_dir = paths.redis_dir()
        os.makedirs(redis_dir, exist_ok=True)
        redis_password = paths.redis_password()
        cmd = [
            redis_bin,
            "--port", str(paths.redis_port()),
            "--bind", "127.0.0.1",
            "--requirepass", redis_password,
            "--dir", redis_dir,
            "--save", "",
            "--appendonly", "no",
            "--loglevel", "warning",
        ]
        self._redis_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        self._redis_url = paths.redis_url()
        # Wait for Redis to be ready.
        for _ in range(60):
            try:
                r = redis_lib.Redis(host="127.0.0.1", port=paths.redis_port(),
                                    password=redis_password, socket_connect_timeout=1)
                r.ping()
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Redis did not start within 30 seconds")

    def _ensure_redis_running(self):
        proc = getattr(self, "_redis_proc", None)
        if self._redis_url is None or proc is None or proc.poll() is not None:
            self._start_redis()
        return self._redis_url

    def _stop_redis(self):
        proc = getattr(self, "_redis_proc", None)
        if proc is None:
            return
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self._redis_proc = None
        self._redis_url = None

    # --- helpers ---

    def _wait_http(self, url, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop_requested.is_set():
                return
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except urllib.error.HTTPError:
                # Server is up (returning 4xx/5xx counts as responding)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"Timed out waiting for {url}")

    def _start_health_loop(self):
        def _loop():
            while not self._health_stop.is_set():
                self._health_stop.wait(30)
                try:
                    urllib.request.urlopen(FLASK_URL, timeout=5)
                except Exception:
                    self._log.warning("Health check failed")
        self._health_thread = threading.Thread(target=_loop, name="health", daemon=True)
        self._health_thread.start()

    def _reap_orphans(self):
        """Kill leftover Postgres/Redis processes from a previous unclean shutdown."""
        import psutil
        for proc in psutil.process_iter(["name"]):
            try:
                name = proc.info.get("name", "").lower()
                if name in ("redis-server.exe", "postgres.exe", "pg_ctl.exe"):
                    self._log.info("Reaping orphan: %s (pid=%d)", proc.info["name"], proc.pid)
                    proc.kill()
            except Exception:
                pass

    def _write_pidfile(self):
        pids = {name: proc.pid for name, proc in self._children.items()}
        with open(paths.pid_file(), "w") as fh:
            json.dump(pids, fh)

    def _remove_pidfile(self):
        try:
            os.unlink(paths.pid_file())
        except OSError:
            pass
