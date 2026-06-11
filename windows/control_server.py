"""TCP control server for the standalone Windows supervisor.

Replaces the Unix-domain-socket ``macos/control_ipc.py`` on Windows, where
``AF_UNIX`` is not available. The protocol is identical (JSON line -> response),
just the transport is TCP on localhost.

The web UI's "save config -> restart workers" flow publishes to Redis;
``restart_listener`` (a supervised child) receives it and calls
``restart_manager``, which -- on the standalone builds -- sends a single line of
JSON (``{"action": ..., "services": [...]}``) to this server instead of shelling
out to ``supervisorctl``. The supervisor applies the same start/stop/restart
semantics to its managed processes.

Also serves a minimal HTTP endpoint at ``/status`` and ``/stop`` for the
``AudioMuse-AI.exe status`` and ``AudioMuse-AI.exe stop`` CLI commands.
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
            threading.Thread(target=self._handle, args=(conn, addr), name=f"ctrl-{addr}", daemon=True).start()

    def _handle(self, conn, addr):
        with conn:
            try:
                conn.settimeout(15)
                data = conn.recv(4096).strip()
                text = data.decode("utf-8", errors="replace")

                # Minimal HTTP handling for status/stop CLI commands.
                if text.startswith("GET /status"):
                    state = self._supervisor.state() if self._supervisor else "unknown"
                    conn.sendall(f"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n{state}".encode())
                    return
                if text.startswith("POST /stop"):
                    if self._supervisor:
                        threading.Thread(target=self._supervisor.stop_all, daemon=True).start()
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nstopping")
                    return

                # JSON control protocol (same as macos/control_ipc.py).
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
