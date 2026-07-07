# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared HTTP download helper for the plugin subsystem.

One SSRF-guarded, size-capped downloader used by both the plugin blueprint
(catalog/manifest fetches) and the plugin manager (package re-downloads) so the
guard, headers, timeouts, and size enforcement can never drift between the two.

Main Features:
* validate_outbound_url SSRF guard, User-Agent, and split connect/read timeouts from config.
* Streamed body assembled with bytearray (linear, not quadratic) and aborted past the byte cap.
* A fresh requests.Session per call so concurrent catalog-fan-out threads never share one Session.
"""

import requests

import config
from ssrf_guard import validate_outbound_url


def download(url, max_bytes):
    """Fetch ``url`` and return its bytes, rejecting SSRF targets and oversized bodies."""
    ok, message = validate_outbound_url(url)
    if not ok:
        raise ValueError(f'URL rejected: {message}')
    headers = {'User-Agent': f'AudioMuse-AI/{config.APP_VERSION}'}
    timeout = (config.PLUGIN_HTTP_CONNECT_TIMEOUT, config.PLUGIN_HTTP_READ_TIMEOUT)
    buffer = bytearray()
    with requests.Session() as session:
        with session.get(url, headers=headers, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=65536):
                buffer.extend(chunk)
                if len(buffer) > max_bytes:
                    raise ValueError('Download exceeds the configured size limit')
    return bytes(buffer)
