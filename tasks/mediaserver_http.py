# tasks/mediaserver_http.py
"""Centralized HTTP layer for the media-server clients.

Drop-in stand-in for the parts of ``requests`` the media-server modules use,
backed by a single shared ``Session`` whose adapter retries failed *connection*
attempts. Adopt it with a one-line import swap and nothing else:

    import requests
    ->
    from tasks import mediaserver_http as requests

Every existing ``requests.get(...)`` / ``requests.post(...)`` / etc. call then
transparently gains connection-retry, and any other attribute
(``requests.exceptions``, ``requests.Session``, ``requests.utils`` ...) keeps
working because it is delegated to the real ``requests`` module below.

Why this exists
---------------
On macOS the app's very first outbound request to a LAN media server often
fails with "[Errno 65] No route to host" right after launch: the OS gates an
app's first local-network connection, so the initial attempt loses the race
and a manual re-click then succeeds. That is not specific to one endpoint or
one media server, so the retry lives here, once, for all of them.

Only connection establishment is retried (``connect``). HTTP error statuses and
read timeouts are NOT retried, so genuine failures still surface immediately to
the caller and are never hidden.
"""
import requests as _requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_retry_strategy = Retry(
    total=3,
    connect=3,          # retry failed connection attempts ("No route to host")
    read=0,             # do NOT retry once connected (don't re-run slow reads)
    status_forcelist=[],  # do NOT retry on HTTP error statuses (let them surface)
    backoff_factor=1,   # ~0s, 1s, 2s between connection attempts
)

_session = None


def _get_session():
    """Return the shared retrying session, creating it on first use."""
    global _session
    if _session is None:
        s = _requests.Session()
        adapter = HTTPAdapter(max_retries=_retry_strategy)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _session = s
    return _session


# --- requests-compatible verb helpers (all flow through the shared session) ---
def get(*args, **kwargs):
    return _get_session().get(*args, **kwargs)


def post(*args, **kwargs):
    return _get_session().post(*args, **kwargs)


def put(*args, **kwargs):
    return _get_session().put(*args, **kwargs)


def delete(*args, **kwargs):
    return _get_session().delete(*args, **kwargs)


def head(*args, **kwargs):
    return _get_session().head(*args, **kwargs)


def patch(*args, **kwargs):
    return _get_session().patch(*args, **kwargs)


def request(*args, **kwargs):
    return _get_session().request(*args, **kwargs)


def __getattr__(name):
    """Delegate anything not defined here (exceptions, Session, utils, codes, ...)
    to the real ``requests`` module, so this stays a safe drop-in replacement."""
    return getattr(_requests, name)
