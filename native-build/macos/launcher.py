# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Entry point and role dispatcher for the macOS standalone build.

Single frozen executable that runs as the menu-bar supervisor by default or,
with ``--role=``, as one of its child processes: the Flask/waitress server or
an RQ worker/janitor/restart-listener. It also applies the scipy longdouble
warmup before the RQ fork to avoid the macOS newlocale crash. The
Linux/Windows launchers are the platform-specific siblings.

Main Features:
* Runs Flask via waitress or launches a named RQ role in-process.
* Pins the numeric locale early and warms up scipy longdouble for every role
  except janitor and restart-listener (macOS newlocale crash fix).
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
        host="0.0.0.0",
        port=8000,
        threads=8,
        max_request_body_size=6 * 1024 * 1024 * 1024,
        channel_timeout=300,
    )


_NO_LONGDOUBLE_WARMUP_ROLES = {"janitor", "restart-listener"}


def _run_role(role):
    if role not in _NO_LONGDOUBLE_WARMUP_ROLES:
        try:
            import numeric_bootstrap

            numeric_bootstrap.warmup_scipy_longdouble()
        except Exception:
            pass
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

    lock_path = os.path.join(paths.app_support_dir(), "supervisor.lock")
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


def _run_menubar():
    import rumps

    try:
        from AppKit import NSApp, NSApplicationActivationPolicyAccessory

        NSApp().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    from macos import paths
    from macos.supervisor import ProcessSupervisor

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
            supervisor.start_in_background()
            rumps.Timer(self._refresh, 3).start()

        def on_open_browser(self, _):
            subprocess.Popen(["open", "http://127.0.0.1:8000"])

        def on_open_log(self, _):
            subprocess.Popen(["open", "-a", "Console", paths.log_file()])

        def on_toggle(self, _):
            if supervisor.is_running():
                threading.Thread(target=supervisor.stop_all, daemon=True).start()
            else:
                supervisor.start_in_background()

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
