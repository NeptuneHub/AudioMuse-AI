# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Gunicorn hooks for the container image; gunicorn reads ./gunicorn.conf.py on its own.

Main Features:
* post_worker_init makes every listening socket the worker inherited answer
  HTTPS as well as HTTP on the port it already has, the same way waitress and
  app.run do in the native builds, so the microphone works on a plain-HTTP
  LAN address without any deployment change
"""


def post_worker_init(worker):
    from tls_listener import adopt_listener, prepare_tls

    prepare_tls()
    for listener in worker.sockets:
        listener.sock = adopt_listener(listener.sock)
