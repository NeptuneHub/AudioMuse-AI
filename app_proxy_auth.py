# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Reverse-proxy (forward-auth) authentication.

An identity-aware proxy such as authentik, Authelia or oauth2-proxy logs the
user in and forwards the identity as request headers. This module turns those
headers into an AudioMuse identity (``username``, ``role``) and is consulted by
``app_auth.check_auth_needed`` before the cookie/bearer checks.

Security model:
* The proxy is recognised by its TCP peer address (``AUTH_PROXY_TRUSTED_SOURCES``)
  and/or a shared secret header (``AUTH_PROXY_SHARED_SECRET``). Every configured
  check must pass; with neither configured the feature stays off (fail closed).
  The peer is the raw socket address, never ``X-Forwarded-For`` (ProxyFix
  rewrites ``remote_addr``, so the original value it stashes is used instead).
* Groups map to roles: ``AUTH_PROXY_ADMIN_GROUP`` -> admin,
  ``AUTH_PROXY_USER_GROUP`` (or anyone, when unset) -> user, otherwise 403.
* Proxy auth is ambient (the IdP cookie rides along on cross-site requests),
  so state-changing requests must come from the same origin (CSRF guard).

Main Features:
* ``resolve_proxy_identity`` - header -> (username, role) or a denial.
* ``proxy_csrf_error`` - same-origin check for unsafe methods.
* ``bind_acting_user`` / ``reset_acting_user`` - expose the proxy user to the
  Navidrome provider so Subsonic calls run as that user (ExtAuth passthrough),
  together with the user's choice to share created playlists (private default).
"""

import hmac
import ipaddress
import logging
from urllib.parse import urlsplit

from flask import g, request

from tasks.mediaserver import context as ms_context

logger = logging.getLogger(__name__)

ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
_UNSAFE_METHODS = frozenset({'POST', 'PUT', 'PATCH', 'DELETE'})
_MAX_USERNAME_LEN = 150

_warned_untrusted = set()


PROXY_DENIED_MESSAGE = "Your account is not allowed to use AudioMuse-AI."


class ProxyAuthDenied(Exception):
    """The proxy authenticated the user, but AudioMuse refuses access.

    Callers answer with ``PROXY_DENIED_MESSAGE``, never the exception text.
    """


def _cfg():
    import config

    return config


def _parse_networks(raw):
    networks = []
    for part in (raw or '').split(','):
        part = part.strip()
        if not part:
            continue
        try:
            networks.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            logger.warning("AUTH_PROXY_TRUSTED_SOURCES: ignoring invalid entry %r", part)
    return networks


def _peer_address():
    """Raw socket peer, unaffected by ProxyFix / X-Forwarded-For."""
    orig = request.environ.get('werkzeug.proxy_fix.orig') or {}
    return orig.get('REMOTE_ADDR') or request.environ.get('REMOTE_ADDR')


def is_trusted_peer(addr=None):
    addr = addr if addr is not None else _peer_address()
    networks = _parse_networks(_cfg().AUTH_PROXY_TRUSTED_SOURCES)
    if not addr or not networks:
        return False
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in networks)


def _secret_matches():
    """True when no shared secret is configured or the request carries it."""
    cfg = _cfg()
    expected = cfg.AUTH_PROXY_SHARED_SECRET or ''
    if not expected:
        return True
    supplied = request.headers.get(cfg.AUTH_PROXY_SECRET_HEADER or 'X-Proxy-Secret') or ''
    return hmac.compare_digest(supplied.encode('utf-8'), expected.encode('utf-8'))


def is_trusted_proxy_request():
    """Every configured proxy check (peer address, shared secret) passes."""
    cfg = _cfg()
    has_sources = bool(_parse_networks(cfg.AUTH_PROXY_TRUSTED_SOURCES))
    has_secret = bool(cfg.AUTH_PROXY_SHARED_SECRET)
    if not has_sources and not has_secret:
        return False
    if has_sources and not is_trusted_peer():
        return False
    return _secret_matches()


def _parse_groups(raw):
    # authentik joins groups with '|', most other proxies with ','.
    if not raw:
        return set()
    normalized = raw.replace('|', ',')
    return {g_.strip() for g_ in normalized.split(',') if g_.strip()}


def _valid_username(name):
    return (
        isinstance(name, str)
        and 0 < len(name) <= _MAX_USERNAME_LEN
        and name.isprintable()
        and name == name.strip()
    )


def role_for_groups(groups):
    """Map proxy groups to an AudioMuse role, or None when not allowed."""
    cfg = _cfg()
    admin_group = (cfg.AUTH_PROXY_ADMIN_GROUP or '').strip()
    user_group = (cfg.AUTH_PROXY_USER_GROUP or '').strip()
    if admin_group and admin_group in groups:
        return ROLE_ADMIN
    if not user_group or user_group in groups:
        return ROLE_USER
    return None


def resolve_proxy_identity():
    """Return ``(username, role)`` for a proxy-authenticated request.

    Returns None when proxy auth is disabled, the user header is absent, or
    the request did not pass the trusted-proxy checks (the header is then ignored
    and the normal cookie/bearer checks run). Raises ``ProxyAuthDenied`` when
    the user is authenticated by the proxy but not in an allowed group.
    """
    cfg = _cfg()
    if not cfg.AUTH_PROXY_ENABLED:
        return None
    username = request.headers.get(cfg.AUTH_PROXY_USER_HEADER or 'Remote-User')
    if not username:
        return None
    if not is_trusted_proxy_request():
        peer = _peer_address()
        if peer not in _warned_untrusted:
            _warned_untrusted.add(peer)
            logger.warning(
                "Ignoring %s header from %s: not a trusted proxy "
                "(AUTH_PROXY_TRUSTED_SOURCES / AUTH_PROXY_SHARED_SECRET).",
                cfg.AUTH_PROXY_USER_HEADER, peer,
            )
        return None
    username = username.strip()
    if not _valid_username(username):
        logger.warning("Proxy supplied an invalid username; access denied.")
        raise ProxyAuthDenied("invalid username")
    groups = _parse_groups(request.headers.get(cfg.AUTH_PROXY_GROUPS_HEADER or 'Remote-Groups'))
    role = role_for_groups(groups)
    if role is None:
        logger.warning(
            "A proxy-authenticated user is in neither %s nor %s; access denied.",
            _cfg().AUTH_PROXY_ADMIN_GROUP or '(no admin group)',
            _cfg().AUTH_PROXY_USER_GROUP or '(no user group)',
        )
        raise ProxyAuthDenied("not in an AudioMuse group")
    return username, role


def proxy_csrf_error():
    """Return an error message when an unsafe request looks cross-site."""
    if request.method not in _UNSAFE_METHODS:
        return None
    fetch_site = request.headers.get('Sec-Fetch-Site')
    if fetch_site and fetch_site not in ('same-origin', 'none'):
        return "Cross-site request rejected."
    origin = request.headers.get('Origin') or request.headers.get('Referer')
    if origin and origin != 'null':
        if urlsplit(origin).netloc.lower() != request.host.lower():
            return "Cross-site request rejected."
    elif origin == 'null':
        return "Cross-site request rejected."
    return None


# --- Navidrome passthrough --------------------------------------------------


def passthrough_active():
    """True when media-server calls of this request run as the proxy user."""
    return getattr(g, 'auth_method', None) == 'proxy' and bool(_cfg().NAVIDROME_EXTAUTH_ENABLED)


def _requested_playlist_public():
    """Private unless the request explicitly asks to share the playlist, via
    ``?playlist_public=true`` (set by the UI checkbox) or ``"public": true`` in
    the JSON body (API callers)."""
    if request.args.get('playlist_public') == 'true':
        return True
    if not request.is_json:
        return False
    data = request.get_json(silent=True)
    return isinstance(data, dict) and data.get('public') is True


def bind_acting_user():
    """before_request hook: expose the proxy user to media-server providers."""
    if passthrough_active():
        g._acting_user_token = ms_context.set_acting_user(g.auth_user)
        g._playlist_public_token = ms_context.set_playlist_public(_requested_playlist_public())
        g._passthrough_token = ms_context.begin_passthrough_request()


def reset_acting_user(*_args):
    """teardown_request hook: release everything bind_acting_user bound."""
    token = g.pop('_acting_user_token', None)
    if token is not None:
        ms_context.reset_acting_user(token)
    public_token = g.pop('_playlist_public_token', None)
    if public_token is not None:
        ms_context.reset_playlist_public(public_token)
    passthrough_token = g.pop('_passthrough_token', None)
    if passthrough_token is not None:
        ms_context.end_passthrough_request(passthrough_token)
