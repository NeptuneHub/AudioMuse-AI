# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the plugin subsystem HTTP downloader.

Drives ``plugin.net.download`` with a fake requests Session to verify that
unreachable hosts, timeouts, and HTTP error statuses become a clean
``DownloadError`` (host in the message, no traceback), that a rejected URL and an
oversized body stay ``ValueError``, and that a normal response returns its bytes.

Main Features:
* No network: the SSRF guard and requests.Session are monkeypatched.
* Covers the connection-error, timeout, HTTP-status, size-limit, and success paths.
"""

import pytest
import requests

import plugin.net as net


class _FakeSession:
    def __init__(self, exc=None, resp=None):
        self._exc = exc
        self._resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, *a, **k):
        if self._exc:
            raise self._exc
        return _Ctx(self._resp)


class _Ctx:
    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        return self._obj

    def __exit__(self, *a):
        return False


class _Resp:
    def __init__(self, chunks=(b'',), status=None):
        self._chunks = chunks
        self._status = status

    def raise_for_status(self):
        if self._status:
            err = requests.exceptions.HTTPError(f'HTTP {self._status}')
            err.response = self
            raise err

    @property
    def status_code(self):
        return self._status

    def iter_content(self, chunk_size=0):
        return iter(self._chunks)


def _wire(monkeypatch, session):
    monkeypatch.setattr(net, 'validate_outbound_url', lambda url: (True, ''))
    monkeypatch.setattr(net.requests, 'Session', lambda: session)


class TestDownload:
    def test_connection_error_becomes_download_error(self, monkeypatch):
        _wire(monkeypatch, _FakeSession(exc=requests.exceptions.ConnectionError('boom')))
        with pytest.raises(net.DownloadError) as ei:
            net.download('https://raw.githubusercontent.com/x/y.zip', 1000)
        assert 'raw.githubusercontent.com' in str(ei.value)

    def test_timeout_becomes_download_error(self, monkeypatch):
        _wire(monkeypatch, _FakeSession(exc=requests.exceptions.ConnectTimeout('slow')))
        with pytest.raises(net.DownloadError) as ei:
            net.download('https://example.com/y.zip', 1000)
        assert 'example.com' in str(ei.value)

    def test_http_error_becomes_download_error_with_status(self, monkeypatch):
        _wire(monkeypatch, _FakeSession(resp=_Resp(status=404)))
        with pytest.raises(net.DownloadError) as ei:
            net.download('https://example.com/y.zip', 1000)
        assert '404' in str(ei.value)

    def test_rejected_url_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(net, 'validate_outbound_url', lambda url: (False, 'blocked'))
        with pytest.raises(ValueError):
            net.download('https://bad.example/', 1000)

    def test_size_limit_raises_value_error(self, monkeypatch):
        _wire(monkeypatch, _FakeSession(resp=_Resp(chunks=(b'x' * 100,))))
        with pytest.raises(ValueError):
            net.download('https://example.com/y.zip', 10)

    def test_success_returns_bytes(self, monkeypatch):
        _wire(monkeypatch, _FakeSession(resp=_Resp(chunks=(b'ab', b'cd'))))
        assert net.download('https://example.com/y.zip', 1000) == b'abcd'
