"""Standalone macOS entry point (the PyInstaller entry script).

Double-clicked, this runs as a background menu-bar agent that supervises the
embedded services and the app. Re-invoked by the supervisor as
``AudioMuse-AI --role=<x>``, the same frozen binary instead runs one child
service -- the web server (waitress serving the existing Flask app) or one of the
RQ worker / janitor / restart-listener entry points, reused unchanged via runpy.
"""

import os
import runpy
import subprocess
import sys
import threading


def _role_from_argv():
    for arg in sys.argv[1:]:
        if arg.startswith("--role="):
            return arg.split("=", 1)[1]
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


# Held for the life of the menu-bar process so the flock is not released early.
_INSTANCE_LOCK = None


def _acquire_single_instance_lock(paths):
    """Return True if we are the only menu-bar agent (and hold the lock).

    A second live supervisor is catastrophic: both manage the *same* embedded
    Postgres/Redis, and a newly started ``redis-server`` unlinks the existing
    unix socket out from under the running stack, knocking every worker offline.
    An ``flock`` guarantees only one agent runs; the OS releases it if the holder
    dies, so a crash-relaunch cleanly takes over (and reaps the orphans on boot).
    """
    global _INSTANCE_LOCK
    import fcntl
    lock_path = os.path.join(paths.app_support_dir(), "supervisor.lock")
    # Open with "a+" (not "w"): "w" truncates on open, which would erase the live
    # holder's PID from the file before we even try the lock. Only rewrite the PID
    # once we actually own the lock.
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


def _run_menubar():
    import rumps
    try:
        from AppKit import NSApp, NSApplicationActivationPolicyAccessory
        NSApp().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    from macos import paths
    from macos.supervisor import ProcessSupervisor

    # Refuse to start a second supervisor; just surface the already-running UI.
    if not _acquire_single_instance_lock(paths):
        subprocess.Popen(["open", "http://127.0.0.1:8000"])
        return

    supervisor = ProcessSupervisor()

    class AudioMuseApp(rumps.App):
        def __init__(self):
            icon = paths.menubar_icon()
            super().__init__(
                "AudioMuse-AI",
                icon=icon if os.path.exists(icon) else None,
                template=True,
                quit_button=None,
            )
            self.status_item = rumps.MenuItem("Status: Starting…")
            self.status_item.set_callback(None)
            self.toggle_item = rumps.MenuItem("Pause Server", callback=self.on_toggle)
            self.menu = [
                self.status_item,
                None,
                rumps.MenuItem("Open in Browser", callback=self.on_open_browser),
                self.toggle_item,
                rumps.MenuItem("Open Log", callback=self.on_open_log),
                None,
                rumps.MenuItem("Quit", callback=self.on_quit),
            ]
            threading.Thread(target=self._boot, name="boot", daemon=True).start()
            rumps.Timer(self._refresh, 3).start()

        def _boot(self):
            try:
                supervisor.start_all()
            except Exception:
                pass

        def on_open_browser(self, _):
            subprocess.Popen(["open", "http://127.0.0.1:8000"])

        def on_open_log(self, _):
            subprocess.Popen(["open", "-a", "Console", paths.log_file()])

        def on_toggle(self, _):
            target = supervisor.stop_all if supervisor.is_running() else supervisor.start_all
            threading.Thread(target=target, daemon=True).start()

        def on_quit(self, _):
            supervisor.stop_all()
            rumps.quit_application()

        def _refresh(self, _):
            labels = {
                "running": "Running",
                "starting": "Starting…",
                "stopping": "Stopping…",
                "stopped": "Stopped",
            }
            self.status_item.title = f"Status: {labels.get(supervisor.state(), supervisor.state())}"
            self.toggle_item.title = "Pause Server" if supervisor.is_running() else "Start Server"

    AudioMuseApp().run()


def main():
    # Force a deterministic C *numeric* locale in every frozen process before any
    # heavy import. NumPy's int->longdouble conversion (used by scipy at import)
    # goes through the locale-sensitive C strtold; a macOS framework changing the
    # locale concurrently otherwise causes an intermittent
    # "Could not parse python long as longdouble" crash. This is macOS-only (the
    # Linux/Docker workers never run launcher.py), so it cannot affect containers.
    try:
        import numeric_bootstrap
        numeric_bootstrap.pin_numeric_locale()
    except Exception:
        pass

    if "--run-restore" in sys.argv:
        i = sys.argv.index("--run-restore")
        from app_backup import _run_restore_runner
        sys.exit(_run_restore_runner(sys.argv[i + 1], sys.argv[i + 2]))

    role = _role_from_argv()
    if role:
        _run_role(role)
    else:
        _run_menubar()


if __name__ == "__main__":
    main()
