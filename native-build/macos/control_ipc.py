# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unix-socket control IPC server for the macOS standalone build.

Runs a small line-oriented server over a mode-0600 Unix domain socket so the
menu-bar app and CLI can send control commands (start, stop, restart, status)
to the running ``macos.supervisor``. The Windows sibling uses a TCP control
server instead.

Main Features:
* Accepts JSON control requests on a private Unix socket in a daemon thread.
* Dispatches each request to a supplied handler and returns its JSON reply.
"""

import json
import logging
import os
import socket
import threading

logger = logging.getLogger("audiomuse.control")


class ControlServer:
    def __init__(self, socket_path, dispatch):
        self._socket_path = socket_path
        self._dispatch = dispatch
        self._sock = None
        self._thread = None
        self._running = False

    def start(self):
        self._unlink()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self._socket_path)
        os.chmod(self._socket_path, 0o600)
        self._sock.listen(8)
        self._running = True
        self._thread = threading.Thread(target=self._serve, name="control-server", daemon=True)
        self._thread.start()

    def _serve(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            with conn:
                try:
                    conn.settimeout(15)
                    data = conn.recv(4096).strip()
                    request = json.loads(data.decode("utf-8"))
                    ok = bool(
                        self._dispatch(request.get("action", ""), request.get("services", []))
                    )
                    conn.sendall(b"ok" if ok else b"error")
                except Exception:
                    logger.exception("Control request failed")
                    try:
                        conn.sendall(b"error")
                    except OSError:
                        pass

    def stop(self):
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._unlink()

    def _unlink(self):
        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass
