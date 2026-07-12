# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Active-media-server context for concurrent multi-server support.

Holds the currently selected media server for the running request or worker job
in a ``contextvars.ContextVar``, so several servers can be used concurrently
without sharing mutable global state. When no server is active the accessors
return the caller-supplied fallback (the global ``config`` default), so an unset
context reproduces the historical single-server behaviour exactly.

Main Features:
* ``use_server`` context manager binds a normalized server dict for a scope.
* ``active_type`` / ``active_creds`` / ``active_libraries`` / ``active_server_id``
  let provider backends resolve the active server before falling back to config.
* When a server is bound, ``active_creds`` uses the server's creds as the base
  and lets caller-supplied creds override only non-empty fields, so a bound
  secondary server's URL/token always win over stale defaults.
"""

import contextvars

_active_server = contextvars.ContextVar("audiomuse_active_server", default=None)


def active_server():
    return _active_server.get()


def set_active_server(server):
    return _active_server.set(server)


def reset_active_server(token):
    _active_server.reset(token)


class use_server:
    """Bind ``server`` (a normalized registry dict or None) as active for a scope."""

    def __init__(self, server):
        self._server = server
        self._token = None

    def __enter__(self):
        self._token = _active_server.set(self._server)
        return self._server

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _active_server.reset(self._token)
            self._token = None
        return False


def active_type(default=None):
    server = _active_server.get()
    if server and server.get("server_type"):
        return server["server_type"]
    return default


def active_creds(user_creds=None):
    server = _active_server.get()
    if server is None:
        return user_creds or None
    merged = dict(server.get("creds") or {})
    for key, value in (user_creds or {}).items():
        if value:
            merged[key] = value
    return merged or None


def active_libraries(default=""):
    server = _active_server.get()
    if server is not None and server.get("music_libraries") is not None:
        return server.get("music_libraries")
    return default


def active_server_id():
    server = _active_server.get()
    return server.get("server_id") if server else None
