
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
        self._unlink()

    def _unlink(self):
        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass
