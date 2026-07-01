# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Loopback TCP control server for the Windows standalone build.

Windows lacks the Unix domain sockets the macOS build uses, so control commands
travel over a loopback-only TCP socket instead. The server runs in a daemon
thread (default 127.0.0.1:8001) and speaks two shapes: HTTP-style ``GET
/status`` and ``POST /stop`` used by the CLI, plus JSON start/stop/restart
requests dispatched to the ``windows.supervisor``.

Main Features:
* Listens on 127.0.0.1 with a 1s accept timeout so ``stop`` unblocks cleanly.
* Answers GET /status and POST /stop directly; dispatches JSON control requests.
"""

import json
import logging
import socket
import threading

logger = logging.getLogger("audiomuse.control")


class ControlServer:
    def __init__(self, host="127.0.0.1", port=8001, dispatch=None, supervisor=None):
        self._host = host
        self._port = port
        self._dispatch = dispatch
        self._supervisor = supervisor
        self._sock = None
        self._thread = None
        self._running = False

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.listen(8)
        self._running = True
        self._thread = threading.Thread(target=self._serve, name="control-server", daemon=True)
        self._thread.start()
        logger.info("Control server listening on %s:%d", self._host, self._port)

    def _serve(self):
        while self._running:
            try:
                self._sock.settimeout(1)
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn, addr), name=f"ctrl-{addr}", daemon=True
            ).start()

    def _handle(self, conn, addr):
        with conn:
            try:
                conn.settimeout(15)
                data = conn.recv(4096).strip()
                text = data.decode("utf-8", errors="replace")

                if text.startswith("GET /status"):
                    state = self._supervisor.state() if self._supervisor else "unknown"
                    conn.sendall(
                        f"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n{state}".encode()
                    )
                    return
                if text.startswith("POST /stop"):
                    if self._supervisor:
                        threading.Thread(target=self._supervisor.stop_all, daemon=True).start()
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nstopping")
                    return

                request = json.loads(text)
                ok = bool(self._dispatch(request.get("action", ""), request.get("services", [])))
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
        self._sock = None
