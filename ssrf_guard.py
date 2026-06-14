# ssrf_guard.py
"""SSRF guard for user-supplied outbound URLs.

A dependency-free leaf module (standard library only), so any blueprint, helper,
or task can validate a URL without pulling in the database / task-queue / Flask
layers. Keep it that way -- nothing here may import a project module.
"""
import ipaddress
import socket
from urllib.parse import urlparse


def validate_outbound_url(url):
    """SSRF guard for user-supplied outbound HTTP(S) URLs.

    Returns ``(True, None)`` when the URL is safe to fetch, else
    ``(False, reason)``.
    """
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
            host, parsed.port or (443 if parsed.scheme == 'https' else 80),
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
