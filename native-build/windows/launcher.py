"""Standalone Windows entry point (the PyInstaller entry script).

Windows counterpart of ``native-build/linux/launcher.py``. There is no native menu-bar agent
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
# Python's ``multiprocessing`` on Windows supports only ``spawn`` (no ``fork``),
# but RQ's ``scheduler.py`` calls ``get_context('fork')`` at module level. The
# monkey-patch below redirects those ``fork`` requests to ``spawn`` so imports
# succeed. The workers themselves run as RQ's ``SimpleWorker`` on Windows (no
# ``os.fork()``; each job runs in the worker process), so the full worker pool
# runs here -- see rq_worker.py and supervisor.py.
import multiprocessing
_orig_get_context = multiprocessing.get_context
def _win_get_context(method=None):
    if method == 'fork':
        method = 'spawn'
    return _orig_get_context(method)
multiprocessing.get_context = _win_get_context

# RQ's SpawnWorker uses several ``os`` functions that only exist on POSIX.
# Monkey-patch them all on Windows so the worker doesn't crash at runtime.
import os as _os_patch
if not hasattr(_os_patch, 'wait4'):
    _os_waitpid = _os_patch.waitpid
    def _win_wait4(pid, options):
        return _os_waitpid(pid, options) + (None,)
    _os_patch.wait4 = _win_wait4
if not hasattr(_os_patch, 'WIFEXITED'):
    _os_patch.WIFEXITED   = lambda status: True   # Windows: processes always exit, not signalled
if not hasattr(_os_patch, 'WIFSIGNALED'):
    _os_patch.WIFSIGNALED = lambda status: False  # Windows: no POSIX signals
if not hasattr(_os_patch, 'WTERMSIG'):
    _os_patch.WTERMSIG    = lambda status: 0      # never called (WIFSIGNALED is always False)
if not hasattr(_os_patch, 'WEXITSTATUS'):
    _os_patch.WEXITSTATUS = lambda status: status # exit code is already the raw number

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
        host="0.0.0.0",
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

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    mutex_name = r"Global\AudioMuse-AI-Supervisor"
    handle = kernel32.CreateMutexW(None, False, mutex_name)
    if not handle:
        return False
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
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
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(_INSTANCE_LOCK)
        _INSTANCE_LOCK = None


def _open_browser(url):
    webbrowser.open(url)


def _silence_supervisor_db_probe():
    """Quiet config.py's import-time setup_manager DB probe in the supervisor process.

    The supervisor/CLI process never owns a database connection -- it only
    orchestrates the embedded services -- but importing ``config`` (transitively,
    via ``windows.paths``) runs ``SetupManager`` against the default container DB
    host before the embedded PostgreSQL is up, logging spurious "could not
    translate host name" warnings. Child processes set the embedded
    ``DATABASE_URL`` and connect cleanly, so this only suppresses noise in the
    supervisor's own console and leaves the children's logging untouched.
    """
    import logging
    logging.getLogger("tasks.setup_manager").setLevel(logging.ERROR)


def main():
    if "--run-restore" in sys.argv:
        i = sys.argv.index("--run-restore")
        from app_backup import _run_restore_runner
        sys.exit(_run_restore_runner(sys.argv[i + 1], sys.argv[i + 2]))

    role = _role_from_argv()
    if role:
        _run_role(role)
        return

    _silence_supervisor_db_probe()

    cmd = _command_from_argv()
    if cmd is None or cmd == "tray":
        _run_tray()
    elif cmd == "start":
        _start_supervisor()
    elif cmd == "stop":
        _stop_supervisor()
    elif cmd == "status":
        _print_status()
    elif cmd == "open":
        _open_or_start()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Usage: AudioMuse-AI.exe [tray|start|stop|status|open]", file=sys.stderr)
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


def _hide_console_if_owned():
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        kernel32.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        kernel32.GetConsoleProcessList.restype = wintypes.DWORD
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        buf = (wintypes.DWORD * 2)()
        if kernel32.GetConsoleProcessList(buf, 2) == 1:
            hwnd = kernel32.GetConsoleWindow()
            if hwnd:
                user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


def _run_tray():
    """Run as a notification-area (tray) app: the Windows counterpart of the macOS menu bar."""
    _hide_console_if_owned()

    from windows import paths
    from windows.supervisor import ProcessSupervisor
    import pystray
    from PIL import Image

    if not _acquire_single_instance_lock(paths):
        print("Another instance is already running. Opening browser...")
        _open_browser(WEB_URL)
        return

    supervisor = ProcessSupervisor()
    _labels = {"running": "Running", "starting": "Starting...", "stopping": "Stopping...", "stopped": "Stopped"}

    def _status_title(_item):
        return f"Status: {_labels.get(supervisor.state(), supervisor.state())}"

    def _on_open_browser(icon, _item):
        _open_browser(WEB_URL)

    def _on_open_log(icon, _item):
        try:
            os.startfile(paths.log_file())
        except Exception:
            pass

    def _on_start(icon, _item):
        supervisor.start_in_background()

    def _on_stop(icon, _item):
        threading.Thread(target=supervisor.stop_all, name="tray-stop", daemon=True).start()

    def _on_quit(icon, _item):
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem(_status_title, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open in Browser", _on_open_browser, default=True),
        pystray.MenuItem("Start", _on_start, enabled=lambda _i: supervisor.state() == "stopped"),
        pystray.MenuItem("Stop", _on_stop, enabled=lambda _i: supervisor.state() in ("running", "starting")),
        pystray.MenuItem("Open Log", _on_open_log),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _on_quit),
    )

    icon_path = paths.tray_icon()
    image = Image.open(icon_path) if os.path.exists(icon_path) else None
    icon = pystray.Icon("AudioMuse-AI", icon=image, title="AudioMuse-AI", menu=menu)

    def _on_ready():
        _open_browser(WEB_URL)

    def _on_error(exc):
        print(f"Startup failed: {exc}", file=sys.stderr)

    supervisor.start_in_background(on_ready=_on_ready, on_error=_on_error)
    try:
        icon.run()
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
            os.kill(pid, signal.SIGTERM)
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
    import urllib.request

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
