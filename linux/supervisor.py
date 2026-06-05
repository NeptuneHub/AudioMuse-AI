"""Process supervisor for the standalone Linux app.

Linux counterpart of ``macos/supervisor.py`` -- the logic is platform-agnostic
(stdlib ``subprocess``/``signal``/``os.killpg`` + ``psutil``), so this is a near
copy that swaps ``macos`` path/env helpers for the ``linux`` ones and reuses the
two genuinely platform-neutral helpers from the macOS package
(``ControlServer``, ``NewestFirstFileHandler``).

It owns the full lifecycle of everything the container deployment runs under
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

import taskqueue
from linux import db_backend
from linux import env as env_builder
from linux import paths
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
        # Set by stop_all() to tell an in-flight start_all() to stop spawning, and
        # owned-thread handles so stop_all() can join the boot/health threads
        # before tearing down (otherwise a stop racing a boot sweeps the child
        # table before boot finished spawning and leaks the late children).
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
        """Boot the stack on a daemon thread, firing the callbacks on
        success/failure. The supervisor owns this thread so ``stop_all`` can join
        it before tearing down -- a stop racing an in-progress boot would
        otherwise sweep the child table before boot finished spawning, leaking
        the late children as detached (``start_new_session``) orphans."""
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
                return  # stop arrived mid-boot; the stopper handles teardown
            self._database_url = db_backend.start_embedded(paths.pgdata_dir())
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
            with self._lock:
                if self._stop_requested.is_set():
                    return
                self._state = "running"
            self._write_pidfile()
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
        # Wait for any in-flight boot/health work to stop spawning before we
        # sweep: a child spawned after the sweep would survive as an orphan in
        # its own session/process group. Joining guarantees everything they
        # started is registered in ``_children`` and gets torn down below.
        self._join_workers()
        for name in reversed(BOOT_ORDER):
            self._terminate_named(name)
        self._terminate_named("redis")
        try:
            db_backend.stop_embedded()
        except Exception:
            logger.exception("Error stopping embedded PostgreSQL")
        self._control.stop()
        self._clear_pidfile()
        with self._lock:
            self._state = "stopped"
        self._log.info("=== AudioMuse-AI stopped ===")

    def _join_workers(self):
        """Wait for the boot and health threads to finish before teardown.

        Skips the current thread so stop_all() called from within start_all()'s
        failure handler (boot thread) or never deadlocks on itself."""
        current = threading.current_thread()
        for thread in (self._boot_thread, self._health_thread):
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=30)

    def _start_redis(self, wait_timeout=60):
        argv, url = taskqueue.build_embedded_redis_argv(
            paths.redis_binary(), paths.redis_socket_path(), paths.redis_dir()
        )
        self._redis_url = url
        # Scrub PyInstaller's LD_LIBRARY_PATH (it points at the bundle's
        # _internal libs) so the bundled redis-server resolves its own libraries
        # via rpath instead of crashing on the frozen app's incompatible ones --
        # same hazard that SIGSEGVs pgserver's initdb. The RQ/flask children
        # re-exec the frozen binary, whose bootloader re-sets the path, so they
        # don't need this; redis is a plain external binary that does.
        self._spawn("redis", argv, env_builder.restore_native_lib_path(dict(os.environ)))
        self._wait_redis(timeout=wait_timeout)

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
            # Refuse to spawn once a stop is in progress -- otherwise the boot
            # loop or the health loop could create a child after stop_all's
            # teardown sweep, leaking it as a detached orphan.
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
        if proc is None:
            return
        try:
            if proc.poll() is not None:
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
                proc.wait(timeout=5)  # reap so it doesn't linger as a zombie
            except Exception:
                pass
        finally:
            # Close the read end of the pipe so _pump's blocking readline wakes
            # up and the log thread exits. Without this, a grand-child (ffmpeg,
            # onnx, ...) that inherited the stdout write fd keeps the pipe open
            # after the worker dies, so _pump never sees EOF and the thread leaks
            # on every restart.
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
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
            # Postgres and Redis are infrastructure, not ``_desired`` children,
            # so the child check below never covers them. Without this, a death
            # of either (crash, OOM, or a stale sibling unlinking Redis's shared
            # unix socket) leaves every worker crash-looping forever.
            self._ensure_postgres_healthy()
            if self._health_stop.is_set():
                return
            self._ensure_redis_healthy()
            for name in list(self._desired):
                if self._health_stop.is_set():
                    return
                with self._lock:
                    proc = self._children.get(name)
                if proc is not None and proc.poll() is not None:
                    self._log.warning("%s exited (code %s); restarting", name, proc.returncode)
                    self.start_child(name)

    def _ensure_postgres_healthy(self):
        """Restart embedded Postgres if it stopped accepting connections."""
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
            pass  # unreachable -> restart below
        self._log.warning("Embedded PostgreSQL unhealthy; restarting it")
        try:
            self._database_url = db_backend.ensure_embedded_running(paths.pgdata_dir())
            self._log.info("Embedded PostgreSQL restarted")
        except Exception:
            self._log.exception("Failed to restart embedded PostgreSQL")

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
            # Use a short readiness wait here (vs. 60s at boot): this runs on the
            # single health thread, so a long block would starve Postgres and
            # child-restart checks for the whole window.
            self._start_redis(wait_timeout=15)
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
                    if (paths.APP_NAME in cmdline or "--role=" in cmdline
                            or "redis-server" in cmdline or "postgres" in cmdline):
                        proc.terminate()
                        self._log.info("Reaped orphan %s (pid %s) from a previous run", name, pid)
                else:
                    # No psutil: verify the PID is still one of ours before
                    # killing -- PIDs get recycled, so a stale pidfile entry could
                    # otherwise name an unrelated process.
                    comm = subprocess.check_output(
                        ["ps", "-p", str(pid), "-o", "command="], text=True
                    ).strip()
                    if (paths.APP_NAME in comm or "--role=" in comm
                            or "redis-server" in comm or "postgres" in comm):
                        os.kill(pid, signal.SIGTERM)
                        self._log.info("Reaped orphan %s (pid %s) via ps fallback", name, pid)
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
        terminated = []
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
                    terminated.append(proc)
                    self._log.info(
                        "Reaped stale %s (pid %s) referencing our data dir",
                        proc.info.get("name"), proc.info["pid"],
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue
        # SIGTERM is asynchronous: wait for the stragglers to actually exit
        # before the caller starts fresh Postgres/Redis, or the new instances
        # race a still-shutting-down process for the data-dir lock / unix
        # socket. Hard-kill anything that ignores SIGTERM within the window.
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
