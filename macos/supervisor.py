"""Process supervisor for the standalone macOS app.

Owns the full lifecycle of everything the container deployment runs under
supervisord: embedded PostgreSQL (via pgserver), embedded Redis (the bundled
binary), the waitress/Flask web server, the two RQ workers, the janitor and the
restart listener. It boots them in dependency order, captures every child's
output into one rotating log, replaces children that exit while running
(supervisord ``autorestart``), and tears the whole tree down without leaving
orphaned Postgres/Redis processes behind.
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
        self._log = self._setup_logging()

    def _setup_logging(self):
        log = logging.getLogger("audiomuse.app")
        log.setLevel(logging.INFO)
        if not log.handlers:
            # Newest line on top so opening the log shows the latest activity
            # first. Bounded by line count (~40k) instead of a byte cap; see
            # macos/reverse_log.py for why it's not a literal per-line prepend.
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
        self._log.info("=== AudioMuse-AI starting ===")
        try:
            self._reap_orphans()
            self._control.start()
            self._database_url = database.start_embedded(paths.pgdata_dir())
            self._log.info("Embedded PostgreSQL ready")
            self._start_redis()
            self._log.info("Embedded Redis ready")
            for name in BOOT_ORDER:
                self.start_child(name)
                if name == "flask":
                    self._wait_http(FLASK_URL, timeout=180)
            self._write_pidfile()
            with self._lock:
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
            if self._state == "stopped":
                return
            self._state = "stopping"
            self._desired.clear()
        self._health_stop.set()
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
        argv = [sys.executable, f"--role={role}"]
        child_env = env_builder.build_child_env(role, self._database_url, self._redis_url)
        self._spawn(name, argv, child_env)
        with self._lock:
            self._desired.add(name)
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
        threading.Thread(target=self._pump, args=(name, proc), name=f"log-{name}", daemon=True).start()
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
            # Redis is infrastructure, not a ``_desired`` child, so the child
            # check below never covers it. Without this, a Redis death (crash,
            # OOM, or a stale sibling unlinking the shared unix socket) leaves
            # every worker crash-looping against a missing socket forever.
            self._ensure_redis_healthy()
            for name in list(self._desired):
                with self._lock:
                    proc = self._children.get(name)
                if proc is not None and proc.poll() is not None:
                    self._log.warning("%s exited (code %s); restarting", name, proc.returncode)
                    self.start_child(name)

    def _ensure_redis_healthy(self):
        """Restart embedded Redis if its process died or its socket stopped
        answering (mirrors supervisord ``autorestart`` for the broker)."""
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
                pass  # alive but unreachable (e.g. socket unlinked) -> restart
        self._log.warning("Embedded Redis unhealthy; restarting it")
        try:
            self._start_redis()
            self._log.info("Embedded Redis restarted")
        except Exception:
            self._log.exception("Failed to restart embedded Redis")

    def dispatch_control(self, action, services):
        results = []
        for svc in services:
            if action == "stop":
                results.append(self.stop_child(svc))
            elif action == "start":
                results.append(self.start_child(svc))
            elif action == "restart":
                results.append(self.restart_child(svc))
            else:
                results.append(False)
        return all(results) if results else False

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
                    if paths.APP_NAME in cmdline or "--role=" in cmdline or "redis-server" in cmdline:
                        proc.terminate()
                        self._log.info("Reaped orphan %s (pid %s) from a previous run", name, pid)
                else:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                continue
        self._reap_stale_infra()

    def _reap_stale_infra(self):
        """Kill any leftover embedded Redis/Postgres referencing *our* data dirs.

        The pidfile sweep above misses processes from an unclean exit (force-quit
        or crash, where ``stop_all`` never ran). Multiple ``redis-server``
        instances share one unix-socket path, so a straggler unlinking that
        socket on exit breaks the live broker -- hence we match by our own
        socket/data paths and clear them before starting fresh."""
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
                        proc.info.get("name"), proc.info["pid"],
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue
        self._clear_pidfile()
