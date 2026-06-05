"""Standalone Linux entry point (the PyInstaller entry script).

Linux counterpart of ``macos/launcher.py``. There is no native menu-bar agent
here (rumps/AppKit are macOS-only); instead the frozen binary is a small
multi-call launcher:

* ``AudioMuse-AI`` (no args) or ``AudioMuse-AI start`` -- become the single
  foreground supervisor: start embedded PostgreSQL + Redis, the Flask web UI and
  the RQ workers, open the browser, and stay alive until told to stop (SIGINT /
  SIGTERM, or ``AudioMuse-AI stop`` from another terminal). This is what the
  desktop launcher (.desktop) runs.
* ``AudioMuse-AI stop`` -- signal the running instance to shut everything down.
* ``AudioMuse-AI status`` -- print whether the stack is up.
* ``AudioMuse-AI open`` -- open the web UI in the browser (starts the stack first
  if it is not already running).
* ``AudioMuse-AI --role=<x>`` -- re-invocation by the supervisor to run one child
  service (the web server, an RQ worker, the janitor or the restart listener),
  reusing the existing entry points unchanged via runpy.
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
        host="127.0.0.1",
        port=8000,
        threads=8,
        max_request_body_size=6 * 1024 * 1024 * 1024,
        channel_timeout=300,
    )


def _run_role(role):
    # Strip the launcher's own ``--role=`` flag from argv before handing control
    # to a child module: the reused entry points (rq_worker, restart_listener,
    # ...) don't parse argv today, but a future argparse/click in any of them
    # would otherwise choke on the unrecognized flag.
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


# Held for the life of the supervisor process so the flock is not released early.
_INSTANCE_LOCK = None


def _acquire_single_instance_lock(paths):
    """Return True if we are the only supervisor (and hold the lock).

    A second live supervisor is catastrophic: both manage the *same* embedded
    Postgres/Redis, and a newly started ``redis-server`` unlinks the existing
    unix socket out from under the running stack, knocking every worker offline.
    An ``flock`` guarantees only one supervisor runs; the OS releases it if the
    holder dies, so a crash-relaunch cleanly takes over (and reaps the orphans
    on boot). The held FD's file also stores the supervisor PID so ``stop`` can
    find it.
    """
    global _INSTANCE_LOCK
    import fcntl
    lock_path = paths.supervisor_lock_path()
    # Open with "a+" (not "w"): "w" truncates on open, which would erase the live
    # holder's PID from the file before we even try the lock. Only rewrite the
    # PID once we actually own the lock.
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
    _INSTANCE_LOCK = fh  # keep the handle (and thus the lock) alive
    return True


def _running_supervisor_pid(paths):
    """PID of the live supervisor, or None. Verifies the flock is actually held."""
    import fcntl
    lock_path = paths.supervisor_lock_path()
    if not os.path.exists(lock_path):
        return None
    try:
        with open(lock_path, "r") as fh:
            try:
                # If we *can* take the lock, nobody holds it -> not running.
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                return None
            except OSError:
                pass  # locked by a live supervisor
            fh.seek(0)
            pid_text = fh.read().strip()
        return int(pid_text) if pid_text else None
    except (OSError, ValueError):
        return None


def _open_browser():
    # Prefer xdg-open (respects the user's default browser / desktop session);
    # fall back to Python's webbrowser if it is missing.
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
        # Already running -- just surface the UI.
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

    def _boot():
        try:
            supervisor.start_all()
            print("AudioMuse-AI is running at %s" % WEB_URL)
            if open_browser:
                _open_browser()
        except Exception as exc:  # startup already logged; surface a hint too
            print("AudioMuse-AI failed to start: %s" % exc, file=sys.stderr)
            stop_event.set()

    threading.Thread(target=_boot, name="boot", daemon=True).start()

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
    # Give the supervisor a moment to tear the stack down cleanly.
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
        # Not running yet: start it in the foreground (this call becomes the
        # supervisor and opens the browser once it is up).
        return _run_supervisor(open_browser=True)
    _open_browser()
    return 0


def main():
    role = _role_from_argv()
    if role:
        _run_role(role)
        return

    command = _command_from_argv()
    if command in (None, "start"):
        open_browser = os.environ.get("AUDIOMUSE_OPEN_BROWSER", "1") != "0"
        sys.exit(_run_supervisor(open_browser=open_browser))
    elif command == "stop":
        sys.exit(_cmd_stop())
    elif command == "status":
        sys.exit(_cmd_status())
    elif command == "open":
        sys.exit(_cmd_open())
    else:
        print("Usage: AudioMuse-AI [start|stop|status|open]", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
