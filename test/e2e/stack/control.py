# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The supervisor side of the app's control plane, served by the harness.

The app restarts its own processes through restart_manager: in the container
it shells out to supervisorctl, on the native builds it sends one JSON line to
the launcher's control socket (AUDIOMUSE_CONTROL_SOCKET) and expects "ok".
This module is that socket server for the test session: it maps the service
names of service_roles onto the harness's process handles, so a worker
restart requested by a config save, a default-server swap or a provider
migration really stops and starts the worker processes, and the control
listener (python -m taskqueue.control) can acknowledge it in the database.

Main Features:
* ControlServer listens on a unix socket and answers "ok" or "error: ..."
* stop / start / restart on flask, queue-worker-high, queue-worker-default,
  queue-maintenance (accepted, nothing to run) and config-restart-listener
* actions are serialized so two overlapping requests cannot interleave
"""

import json
import os
import socket
import threading

from .errors import StackError

SERVICE_FLASK = 'flask'
SERVICE_WORKER_HIGH = 'queue-worker-high'
SERVICE_WORKER_DEFAULT = 'queue-worker-default'
SERVICE_MAINTENANCE = 'queue-maintenance'
SERVICE_LISTENER = 'config-restart-listener'
KNOWN_SERVICES = (
    SERVICE_FLASK, SERVICE_WORKER_HIGH, SERVICE_WORKER_DEFAULT, SERVICE_MAINTENANCE, SERVICE_LISTENER,
)
ACTIONS = ('stop', 'start', 'restart')


class ControlServer:
    def __init__(self, socket_path, resolve_service, log_path):
        self.socket_path = socket_path
        self._resolve = resolve_service
        self._log_path = log_path
        self._lock = threading.Lock()
        self._sock = None
        self._thread = None
        self._halt = threading.Event()
        self.requests = []

    def start(self):
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.socket_path)
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, name='e2e-control', daemon=True)
        self._thread.start()

    def stop(self):
        self._halt.set()
        if self._thread is not None:
            self._thread.join(3)
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

    def _log(self, text):
        with open(self._log_path, 'a', encoding='utf-8') as handle:
            handle.write(text + '\n')

    def _serve(self):
        while not self._halt.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(120)
                    data = b''
                    while not data.endswith(b'\n'):
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    reply = self._handle(data.decode('utf-8', 'replace').strip())
                except Exception as exc:
                    reply = f'error: {exc}'
                try:
                    conn.sendall(reply.encode('utf-8'))
                except OSError:
                    pass

    def _handle(self, line):
        message = json.loads(line or '{}')
        action = message.get('action')
        services = list(message.get('services') or [])
        self.requests.append((action, tuple(services)))
        self._log(f'control request: {action} {services}')
        if action not in ACTIONS:
            return f'error: unknown action {action}'
        unknown = [s for s in services if s not in KNOWN_SERVICES]
        if unknown:
            return f'error: unknown services {unknown}'
        with self._lock:
            for service in services:
                handle = self._resolve(service)
                if handle is None:
                    continue
                if action in ('stop', 'restart'):
                    handle.stop()
                if action in ('start', 'restart'):
                    handle.start()
                    if action == 'restart' and hasattr(handle, 'wait_ready'):
                        try:
                            handle.wait_ready(180)
                        except StackError as exc:
                            self._log(f'control restart of {service} did not become ready: {exc}')
                            return f'error: {service} did not come back'
        self._log(f'control request done: {action} {services}')
        return 'ok'
