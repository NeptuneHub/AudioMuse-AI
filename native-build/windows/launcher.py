# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Entry point and role dispatcher for the Windows standalone build.

Single frozen executable that runs as the tray supervisor by default or, with
``--role=``, as one of its child processes: the Flask/waitress server or a queue
worker/maintenance/control-listener. It still forces multiprocessing ``fork`` to
``spawn`` for loky's sake; the POSIX ``os.wait*`` stubs are gone, because nothing
in the queue needs them any more. The Linux/macOS launchers are the
platform-specific siblings.

Main Features:
* Runs Flask via waitress or launches a named queue role in-process.
* Patches multiprocessing so loky's spawn payloads work in the frozen bundle.
* Hands multiprocessing/loky spawn payloads to ``native_common.frozen_children``
  instead of re-entering the launcher as a stray copy of the app.
"""

import multiprocessing

_orig_get_context = multiprocessing.get_context


def _win_get_context(method=None):
    if method == 'fork':
        method = 'spawn'
    return _orig_get_context(method)


multiprocessing.get_context = _win_get_context

import os as _os

_os.environ.setdefault("OTEL_PYTHON_CONTEXT", "contextvars_context")

import os
import signal
import sys
import threading
import time
import webbrowser

import service_roles
from native_common import frozen_children

WEB_URL = "http://127.0.0.1:8000"


def _run_role(role):
    service_roles.run_role(role, service_roles.serve_flask)


_INSTANCE_LOCK = None


def _acquire_single_instance_lock(paths):
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
    if ctypes.get_last_error() == 183:
        kernel32.CloseHandle(handle)
        return False

    _INSTANCE_LOCK = handle

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
    import logging

    logging.getLogger("tasks.setup_manager").setLevel(logging.ERROR)


def main():
    if frozen_children.dispatch_child_invocation(_run_role):
        return

    _silence_supervisor_db_probe()

    cmd = service_roles.command_from_argv()
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
                user32.ShowWindow(hwnd, 0)
    except Exception:
        pass


def _run_tray():
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
    _labels = {
        "running": "Running",
        "starting": "Starting...",
        "stopping": "Stopping...",
        "stopped": "Stopped",
    }

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
        pystray.MenuItem(
            "Stop", _on_stop, enabled=lambda _i: supervisor.state() in ("running", "starting")
        ),
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
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{paths.control_port()}/stop",
            method="POST",
            data=b"",
        )
        urllib.request.urlopen(req, timeout=5)
        print("Stop request sent.")
    except Exception:
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

    _start_supervisor()


if __name__ == "__main__":
    main()
