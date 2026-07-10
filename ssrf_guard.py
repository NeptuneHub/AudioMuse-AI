# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""SSRF guard for validating user-supplied outbound URLs.

Vets URLs before the application makes outbound HTTP requests on their behalf,
rejecting non-http(s) schemes and hosts that resolve into blocked special-use
address ranges. Note: loopback and private (RFC 1918) addresses are NOT rejected
here; callers needing those blocked must add their own check.

Main Features:
* ``validate_outbound_url`` checks scheme, resolves the host, and inspects every IP.
* Blocks link-local, multicast, reserved, and unspecified resolved addresses.
"""

import ipaddress
import socket
from urllib.parse import urlparse


def validate_outbound_url(url):
    if not url:
        return False, 'URL is required'
    try:
        parsed = urlparse(str(url))
    except Exception:
        return False, 'Invalid URL'
    if parsed.scheme not in ('http', 'https'):
        return False, 'Only http and https URLs are supported'
    host = parsed.hostname
    if not host:
        return False, 'URL host is required'
    try:
        addrinfo = socket.getaddrinfo(
            host,
            parsed.port or (443 if parsed.scheme == 'https' else 80),
            type=socket.SOCK_STREAM,
        )
    except Exception:
        return False, 'Could not resolve host'
    for entry in addrinfo:
        try:
            ip_obj = ipaddress.ip_address(entry[4][0])
        except ValueError:
            return False, 'Resolved host to invalid IP'
        if (
            ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            return False, 'Target host resolves to a disallowed IP address'
    return True, None
