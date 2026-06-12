# tasks/mediaserver/http.py
"""Centralized HTTP layer for the media-server clients.

Drop-in stand-in for the parts of ``requests`` the media-server modules use.
Adopt it with a one-line import swap and nothing else:

    import requests
    ->
    from . import http as requests

Every existing ``requests.get(...)`` / ``requests.post(...)`` / etc. call then
gains a *connection-only* retry, and any other attribute
(``requests.exceptions``, ``requests.Session``, ``requests.utils`` ...) keeps
working because it is delegated to the real ``requests`` module below.

Why this exists
---------------
On macOS the app's very first outbound request to a LAN media server often
fails with "[Errno 65] No route to host" right after launch: the OS gates an
app's first local-network connection, so the initial attempt loses the race
and a manual re-click then succeeds. That is not specific to one endpoint or
one media server, so the retry lives here, once, for all of them.

Design notes
------------
* CONNECT-ONLY retry. We retry establishing the TCP connection (``connect=3``)
  but NEVER re-send the request once it is on the wire (``read=0``, no status
  retries). So a blocked/failed connection is retried safely for ANY verb
  (nothing was sent yet), while the real call — SELECT, POST, DELETE — is
  issued exactly once. No risk of a mutating request being applied twice.
* NO shared/persistent session. Each call uses its own throwaway session
  (the same thing ``requests.get()`` does internally), so there is no
  cross-call state — cookies, auth, pooled sockets — shared between callers
  that pass different credentials. This is intentionally stateless.
"""
import requests as _requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Retry only the connection handshake; once connected, do not retry/re-send.
_CONNECT_RETRY = Retry(
    total=3,
    connect=3,            # retry failed connection attempts ("No route to host")
    read=0,               # never re-send a request that already went out
    status_forcelist=[],  # never retry on HTTP error statuses (let them surface)
    backoff_factor=1,     # brief pause between connection attempts (~0s, 1s, 2s)
)


def _request(verb, *args, **kwargs):
    """Issue one request through a throwaway session with a connect-retry adapter.

    Mirrors ``requests.api.request`` (fresh session per call, closed on return)
    so behaviour — including streamed downloads with ``stream=True`` — is
    identical to plain ``requests``; the only addition is the connect-retry.
    """
    with _requests.Session() as s:
        adapter = HTTPAdapter(max_retries=_CONNECT_RETRY)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return getattr(s, verb)(*args, **kwargs)


# --- requests-compatible verb helpers ---
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
    """Delegate anything not defined here (exceptions, Session, utils, codes, ...)
    to the real ``requests`` module, so this stays a safe drop-in replacement."""
    return getattr(_requests, name)
