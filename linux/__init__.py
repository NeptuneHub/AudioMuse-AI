"""Standalone Linux packaging for AudioMuse-AI (.deb / .rpm).

This package is the Linux counterpart of the ``macos`` package: everything
specific to the no-container native Linux build -- the launcher, the process
supervisor that runs embedded PostgreSQL/Redis and the app processes, the
path/env helpers, and the PyInstaller build tooling. None of it is imported by
the cloud/container deployment.

Design notes
------------
* It deliberately reuses the two *platform-agnostic* helpers from the ``macos``
  package (``macos.control_ipc.ControlServer`` and
  ``macos.reverse_log.NewestFirstFileHandler``) instead of duplicating them.
  Both are pure-stdlib and contain nothing macOS-specific.
* The child env (``linux/env.py``) reports ``AUDIOMUSE_PLATFORM=macos`` on
  purpose: that value is the *only* platform-keyed branch in the shared
  ``restart_manager.py`` and it selects the unix-socket control-server path
  (which this package's supervisor implements identically). We are not allowed
  to touch shared code to add a separate ``linux`` value, so we ride the
  existing standalone-mode branch. See ``linux/env.py`` for the full rationale.
"""
