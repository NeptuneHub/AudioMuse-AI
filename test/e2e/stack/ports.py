# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""TCP port probes for the end-to-end stack.

Flask always binds 8000 (config.FLASK_BIND_PORT is not tunable), so the harness
must know before it starts anything whether that port is already taken and by
whom; Navidrome's port is tunable and falls back to a free one.

Main Features:
* is_free binds a probe socket, the only reliable answer on every platform
* listener names the pid and process behind a busy port via psutil
* require_free raises StackError with that pid so a stale server is found fast
"""

import socket

import psutil

from .errors import StackError


def is_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def listener(port):
    try:
        connections = psutil.net_connections(kind='tcp')
    except psutil.Error:
        return None, ''
    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or not conn.laddr or conn.laddr.port != port:
            continue
        name = ''
        if conn.pid:
            try:
                name = psutil.Process(conn.pid).name()
            except psutil.Error:
                name = '?'
        return conn.pid, name
    return None, ''


def free_port(host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def require_free(port, what, host='0.0.0.0'):
    if is_free(host, port):
        return
    pid, name = listener(port)
    raise StackError(
        f"port {port} is busy ({what}): pid {pid} ({name or 'unknown'}) - stop it or run the suite elsewhere"
    )
