"""Centralized database access, dispatched on ``config.DATABASE_TYPE``.

This mirrors the ``tasks/mediaserver.py`` approach: a single module owns "how do
we talk to the database" so the rest of the code never constructs a connection
itself. ``app_helper`` re-exports :func:`get_db`/:func:`close_db` from here, so
the ~50 modules that do ``from app_helper import get_db`` are untouched.

``postgres`` (default) and ``embedded`` both connect through ``config.DATABASE_URL``
with psycopg2 and behave identically at the call site. The only difference is who
starts the server: with ``embedded`` the macOS supervisor calls :func:`start_embedded`
first (pgserver), exports the resulting DSN as ``DATABASE_URL``, then boots the app.
"""

import logging

import psycopg2
from flask import g

import config

logger = logging.getLogger(__name__)

_embedded_server = None


def get_db():
    """Return a request-scoped psycopg2 connection (cached on Flask ``g``)."""
    if 'db' not in g:
        try:
            g.db = psycopg2.connect(
                config.DATABASE_URL,
                connect_timeout=30,
                keepalives_idle=600,
                keepalives_interval=30,
                keepalives_count=3,
                options='-c statement_timeout=600000'
            )
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    return g.db


def close_db(e=None):
    """Close and drop the request-scoped connection, if any."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def start_embedded(data_dir):
    """Start an embedded PostgreSQL server (pgserver) and return its libpq DSN.

    Used only by the standalone (macOS) supervisor when ``DATABASE_TYPE`` is
    ``embedded``. The data directory must live outside the read-only app bundle
    and its path must not contain spaces (pgserver doubles it as the unix-socket
    dir, which ``postgres`` receives via ``pg_ctl -o '-k <dir>'`` and re-splits on
    whitespace); see ``macos/paths.py::app_support_dir``. Initializes the cluster
    on first run, idempotent afterwards.
    """
    global _embedded_server
    import pgserver
    _embedded_server = pgserver.get_server(data_dir)
    return _embedded_server.get_uri()


def stop_embedded():
    """Cleanly stop the embedded PostgreSQL server started by :func:`start_embedded`."""
    global _embedded_server
    if _embedded_server is not None:
        _embedded_server.cleanup()
        _embedded_server = None
