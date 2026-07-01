# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Centralized HTTP layer for media-server backends.

Provides a drop-in replacement for the parts of `requests` used by backends,
adopted via `from . import http as requests`.

Main Features:
* Wraps every call with a connection-only retry to survive the macOS first-
  outbound-request race ("[Errno 65] No route to host") right after launch.
* Automatically delegates all other `requests` attributes to the real library.
"""

import requests as _requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_CONNECT_RETRY = Retry(
    total=3,
    connect=3,
    read=0,
    status_forcelist=[],
    backoff_factor=1,
)


def _request(verb, *args, **kwargs):
    with _requests.Session() as s:
        adapter = HTTPAdapter(max_retries=_CONNECT_RETRY)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return getattr(s, verb)(*args, **kwargs)


def get(*args, **kwargs):
    return _request("get", *args, **kwargs)


def post(*args, **kwargs):
    return _request("post", *args, **kwargs)


def put(*args, **kwargs):
    return _request("put", *args, **kwargs)


def delete(*args, **kwargs):
    return _request("delete", *args, **kwargs)


def head(*args, **kwargs):
    return _request("head", *args, **kwargs)


def patch(*args, **kwargs):
    return _request("patch", *args, **kwargs)


def request(*args, **kwargs):
    return _request("request", *args, **kwargs)


def __getattr__(name):
    return getattr(_requests, name)
