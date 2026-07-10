# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Entry point and role dispatcher for the Linux standalone build.

Single PyInstaller executable that acts either as the top-level supervisor
(default, holding a single-instance lock and opening the web UI) or, when
invoked with ``--role=``, as one of the child processes it re-spawns: the
Flask/waitress server or an RQ worker/janitor/restart-listener.

Main Features:
* Runs Flask via waitress or launches a named RQ role in-process.
* Enforces single-instance startup with an flock-based supervisor lock.
"""

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
        host="0.0.0.0",
        port=8000,
        threads=8,
        max_request_body_size=6 * 1024 * 1024 * 1024,
        channel_timeout=300,
    )


def _run_role(role):
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


_INSTANCE_LOCK = None


def _acquire_single_instance_lock(paths):
    global _INSTANCE_LOCK
    import fcntl

    lock_path = paths.supervisor_lock_path()
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    fh.seek(0)
    fh.truncate(0)
    fh.write(str(os.getpid()))
    fh.flush()
    _INSTANCE_LOCK = fh
    return True


def _running_supervisor_pid(paths):
    import fcntl

    lock_path = paths.supervisor_lock_path()
    if not os.path.exists(lock_path):
        return None
    try:
        with open(lock_path, "r") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                return None
            except OSError:
                pass
            fh.seek(0)
            pid_text = fh.read().strip()
        return int(pid_text) if pid_text else None
    except (OSError, ValueError):
        return None


def _open_browser():
    try:
        subprocess.Popen(
            ["xdg-open", WEB_URL],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except Exception:
        pass
    try:
        webbrowser.open(WEB_URL)
    except Exception:
        pass


def _run_supervisor(open_browser=True):
    from linux import paths
    from linux.supervisor import ProcessSupervisor

    if not _acquire_single_instance_lock(paths):
        print("AudioMuse-AI is already running at %s" % WEB_URL)
        if open_browser:
            _open_browser()
        return 0

    supervisor = ProcessSupervisor()
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def _on_ready():
        print("AudioMuse-AI is running at %s" % WEB_URL)
        if open_browser:
            _open_browser()

    def _on_error(exc):
        print("AudioMuse-AI failed to start: %s" % exc, file=sys.stderr)
        stop_event.set()

    supervisor.start_in_background(on_ready=_on_ready, on_error=_on_error)

    try:
        while not stop_event.wait(0.5):
            pass
    finally:
        supervisor.stop_all()
    return 0


def _cmd_stop():
    from linux import paths

    pid = _running_supervisor_pid(paths)
    if pid is None:
        print("AudioMuse-AI is not running.")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("AudioMuse-AI is not running.")
        return 0
    except OSError as exc:
        print("Could not stop AudioMuse-AI (pid %s): %s" % (pid, exc), file=sys.stderr)
        return 1
    for _ in range(60):
        if _running_supervisor_pid(paths) is None:
            break
        time.sleep(0.5)
    print("AudioMuse-AI stopped.")
    return 0


def _cmd_status():
    from linux import paths

    pid = _running_supervisor_pid(paths)
    if pid is None:
        print("AudioMuse-AI: stopped")
        return 1
    print("AudioMuse-AI: running (supervisor pid %s, %s)" % (pid, WEB_URL))
    return 0


def _cmd_open():
    from linux import paths

    if _running_supervisor_pid(paths) is None:
        open_browser = os.environ.get("AUDIOMUSE_OPEN_BROWSER", "1") != "0"
        return _run_supervisor(open_browser=open_browser)
    _open_browser()
    return 0


def _refuse_root_for_stack():
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        print(
            "AudioMuse-AI must not be run as root.\n"
            "Run it as your normal user instead:\n"
            "    audiomuse-ai start\n"
            "or enable the per-user service:\n"
            "    systemctl --user enable --now audiomuse-ai",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    if "--run-restore" in sys.argv:
        i = sys.argv.index("--run-restore")
        from app_backup import _run_restore_runner

        sys.exit(_run_restore_runner(sys.argv[i + 1], sys.argv[i + 2]))

    role = _role_from_argv()
    if role:
        _run_role(role)
        return

    command = _command_from_argv()
    if command in (None, "start"):
        _refuse_root_for_stack()
        open_browser = os.environ.get("AUDIOMUSE_OPEN_BROWSER", "1") != "0"
        sys.exit(_run_supervisor(open_browser=open_browser))
    elif command == "stop":
        sys.exit(_cmd_stop())
    elif command == "status":
        sys.exit(_cmd_status())
    elif command == "open":
        _refuse_root_for_stack()
        sys.exit(_cmd_open())
    else:
        print("Usage: AudioMuse-AI [start|stop|status|open]", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
