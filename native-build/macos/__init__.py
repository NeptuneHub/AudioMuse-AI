"""Standalone macOS packaging for AudioMuse-AI.

This package contains everything specific to the no-container macOS build: the
menu-bar launcher, the process supervisor that runs embedded PostgreSQL/Redis and
the app processes, path/env helpers, the control socket that replaces supervisord,
and the PyInstaller build tooling. None of it is imported by the cloud/container
deployment.
"""
