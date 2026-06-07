"""Standalone Windows entry point (the PyInstaller entry script).

Windows counterpart of ``linux/launcher.py``. There is no native menu-bar agent
here (rumps/AppKit are macOS-only); instead the frozen binary is a small
multi-call launcher:

* ``AudioMuse-AI.exe`` (no args) or ``AudioMuse-AI.exe start`` -- become the single
  foreground supervisor: start embedded PostgreSQL + Redis, the Flask web UI and
  the RQ workers, open the browser, and stay alive until told to stop (Ctrl+C
  sends CTRL_BREAK_EVENT to the process group, or ``AudioMuse-AI.exe stop`` from
  another terminal).
* ``AudioMuse-AI.exe stop`` -- signal the running instance to shut everything down.
* ``AudioMuse-AI.exe status`` -- print whether the stack is up.
* ``AudioMuse-AI.exe open`` -- open the web UI in the browser (starts the stack
  first if it is not already running).
* ``AudioMuse-AI.exe --role=<x>`` -- re-invocation by the supervisor to run one
  child service (the web server, an RQ worker, the janitor or the restart
  listener), reusing the existing entry points unchanged via runpy.
"""

# --- Windows: force multiprocessing 'spawn' before ANY imports ---
# Python's ``multiprocessing`` on Windows does not support ``fork`` (only
# ``spawn``).  RQ's ``scheduler.py`` calls ``get_context('fork')`` at module
# level, and RQ workers call ``os.fork()`` directly.  The monkey-patch below
# redirects ``fork`` requests to ``spawn`` so imports succeed.  Actual
# worker fork() calls will still fail at runtime, so we skip starting
# RQ workers on Windows (see supervisor.py).
import multiprocessing
_orig_get_context = multiprocessing.get_context
def _win_get_context(method=None):
    if method == 'fork':
        method = 'spawn'
    return _orig_get_context(method)
multiprocessing.get_context = _win_get_context

# RQ's SpawnWorker uses several ``os`` functions that only exist on POSIX.
# Monkey-patch them all on Windows so the worker doesn't crash at runtime.
if not hasattr(os, 'wait4'):
    _os_waitpid = os.waitpid
    def _win_wait4(pid, options):
        return _os_waitpid(pid, options) + (None,)
    os.wait4 = _win_wait4
if not hasattr(os, 'WIFEXITED'):
    os.WIFEXITED   = lambda status: True   # Windows: processes always exit, not signalled
if not hasattr(os, 'WIFSIGNALED'):
    os.WIFSIGNALED = lambda status: False  # Windows: no POSIX signals
if not hasattr(os, 'WTERMSIG'):
    os.WTERMSIG    = lambda status: 0      # never called (WIFSIGNALED is always False)
if not hasattr(os, 'WEXITSTATUS'):
    os.WEXITSTATUS = lambda status: status # exit code is already the raw number

# OpenTelemetry's entry-point-based context loading breaks in PyInstaller
# frozen apps because ``importlib.metadata.entry_points()`` can raise
# ``StopIteration`` when no entry points are found (the iterator is
# exhausted).  Patch it to return an empty list instead, then force the
# contextvars runtime context before any import triggers it.
import os as _os
_os.environ.setdefault("OTEL_PYTHON_CONTEXT", "contextvars_context")
# ------------------------------------------------------------------

import os
import runpy
import signal
import subprocess
import sys
import threading
import time
import webbrowser

WEB_URL = "http://127.0.0.1:8000"


def _role_from_argv():
    for arg in sys.argv[1:]:
        if arg.startswith("--role="):
            return arg.split("=", 1)[1]
    return None


def _command_from_argv():
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            return arg
    return None


def _run_flask():
    import waitress
    import app as app_module
    waitress.serve(
        app_module.app,
        host="127.0.0.1",
        port=8000,
        threads=8,
        max_request_body_size=6 * 1024 * 1024 * 1024,
        channel_timeout=300,
    )


def _run_role(role):
    # Strip the launcher's own ``--role=`` flag from argv before handing control
    # to a child module.
    sys.argv = [a for a in sys.argv if not a.startswith("--role=")]
    if role == "flask":
        _run_flask()
    elif role == "worker-high":
        runpy.run_module("rq_worker_high_priority", run_name="__main__")
    elif role == "worker-default":
        runpy.run_module("rq_worker", run_name="__main__")
    elif role == "janitor":
        runpy.run_module("rq_janitor", run_name="__main__")
    elif role == "restart-listener":
        import restart_listener
        restart_listener.main()
    else:
        raise SystemExit(f"Unknown role: {role}")


# --- single-instance lock (Windows named mutex instead of flock) ---

_INSTANCE_LOCK = None


def _acquire_single_instance_lock(paths):
    """Return True if we are the only supervisor (and hold the lock).

    Uses a Windows named mutex (no flock on Windows). The lock file also stores
    the supervisor PID so ``stop`` can find it.
    """
    global _INSTANCE_LOCK
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    mutex_name = r"Global\AudioMuse-AI-Supervisor"
    handle = kernel32.CreateMutexW(None, False, mutex_name)
    if not handle:
        return False
    last_error = kernel32.GetLastError()
    if last_error == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        return False

    _INSTANCE_LOCK = handle

    # Write the supervisor PID to the lock file so ``stop`` can find us.
    lock_path = paths.supervisor_lock_path()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as fh:
        fh.write(str(os.getpid()))
    return True


def _release_single_instance_lock():
    global _INSTANCE_LOCK
    if _INSTANCE_LOCK is not None:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.CloseHandle(_INSTANCE_LOCK)
        _INSTANCE_LOCK = None


def _open_browser(url):
    webbrowser.open(url)


def main():
    role = _role_from_argv()
    if role:
        _run_role(role)
        return

    cmd = _command_from_argv()
    if cmd is None or cmd == "start":
        _start_supervisor()
    elif cmd == "stop":
        _stop_supervisor()
    elif cmd == "status":
        _print_status()
    elif cmd == "open":
        _open_or_start()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Usage: AudioMuse-AI.exe [start|stop|status|open]", file=sys.stderr)
        sys.exit(1)


def _start_supervisor():
    from windows import paths
    from windows.supervisor import ProcessSupervisor

    if not _acquire_single_instance_lock(paths):
        print("Another instance is already running. Opening browser...")
        _open_browser(WEB_URL)
        return

    supervisor = ProcessSupervisor()
    # Install a console control handler so Ctrl+C shuts down cleanly.
    def _on_ctrl(sig):
        print("\nShutting down...")
        supervisor.stop_all()
    signal.signal(signal.SIGINT, _on_ctrl)
    signal.signal(signal.SIGTERM, _on_ctrl)

    try:
        supervisor.start_all()
        _open_browser(WEB_URL)
        print(f"AudioMuse-AI is running at {WEB_URL}")
        print("Press Ctrl+C to stop.")
        # Park the main thread; the supervisor runs on its own threads.
        while supervisor.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
    finally:
        supervisor.stop_all()
        _release_single_instance_lock()


def _stop_supervisor():
    from windows import paths
    import urllib.request

    lock_path = paths.supervisor_lock_path()
    # Signal the running supervisor to stop via the control endpoint.
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{paths.control_port()}/stop",
            method="POST",
            data=b"",
        )
        urllib.request.urlopen(req, timeout=5)
        print("Stop request sent.")
    except Exception:
        # Fall back: try to read the PID from the lock file and terminate.
        try:
            with open(lock_path, "r") as fh:
                pid = int(fh.read().strip())
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(1, False, pid)  # PROCESS_TERMINATE
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
                print("Supervisor terminated.")
        except Exception:
            print("Could not stop supervisor (not running?).", file=sys.stderr)


def _print_status():
    from windows import paths
    import urllib.request

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{paths.control_port()}/status",
            method="GET",
        )
        resp = urllib.request.urlopen(req, timeout=5)
        print(resp.read().decode())
    except Exception:
        print("AudioMuse-AI is not running.")


def _open_or_start():
    from windows import paths

    # Check if already running.
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{paths.control_port()}/status",
            method="GET",
        )
        urllib.request.urlopen(req, timeout=3)
        _open_browser(WEB_URL)
        return
    except Exception:
        pass

    # Not running -- start the supervisor.
    _start_supervisor()


if __name__ == "__main__":
    main()
