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
a queue worker/maintenance/control-listener. It also applies the scipy longdouble
warmup before the first analysis job to avoid the macOS newlocale crash. The
Linux/Windows launchers are the platform-specific siblings.

Main Features:
* Runs Flask via waitress or launches a named queue role in-process.
* Pins the numeric locale early and warms up scipy longdouble for every role
  except maintenance and restart-listener (macOS newlocale crash fix).
* Hands multiprocessing/loky spawn payloads to ``native_common.frozen_children``
  and rejects any other unknown argv rather than starting a second menu bar.
"""

import os
import subprocess
import sys
import threading

import service_roles
from native_common import frozen_children


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


_NO_LONGDOUBLE_WARMUP_ROLES = {
    service_roles.ROLE_MAINTENANCE,
    service_roles.ROLE_RESTART_LISTENER,
}


def _run_role(role):
    if role not in _NO_LONGDOUBLE_WARMUP_ROLES:
        try:
            import numeric_bootstrap

            numeric_bootstrap.warmup_scipy_longdouble()
        except Exception:
            pass
    service_roles.run_role(role, _run_flask)


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

    if frozen_children.run_frozen_child():
        return

    if "--run-restore" in sys.argv:
        i = sys.argv.index("--run-restore")
        from app_backup import _run_restore_runner

        sys.exit(_run_restore_runner(sys.argv[i + 1], sys.argv[i + 2]))

    role = _role_from_argv()
    if role:
        _run_role(role)
        return

    unknown = _command_from_argv()
    if unknown is not None:
        print(f"Unknown argument: {unknown}", file=sys.stderr)
        print("Usage: AudioMuse-AI [--role=<role>]", file=sys.stderr)
        sys.exit(2)

    _run_menubar()


if __name__ == "__main__":
    main()
