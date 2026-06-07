"""Embedded-database backend selector for the standalone Windows build.

Two mechanisms provide the embedded PostgreSQL:

* **pgserver available** -- the shared :mod:`database` module's pgserver path
  (pgserver ships a Windows wheel with PostgreSQL + pgvector).
* **Fallback** -- :mod:`windows.embedded_pg`, which manages a from-source,
  relocatable PostgreSQL bundled in the package (if pgserver has no Windows wheel).

Both expose the same ``start_embedded`` / ``ensure_embedded_running`` /
``stop_embedded`` surface and return a libpq DSN, so :mod:`windows.supervisor`
calls these without caring which backend is active. This keeps the
platform-specific branching in one tiny place and changes **no shared code**.
"""

import logging
import os

logger = logging.getLogger("audiomuse.db_backend")

_USE_PGSERVER = None


def _check_pgserver():
    """Return True if pgserver can be imported (has a Windows wheel)."""
    global _USE_PGSERVER
    if _USE_PGSERVER is None:
        try:
            import pgserver
            _USE_PGSERVER = True
            logger.info("Using pgserver for embedded PostgreSQL")
        except ImportError:
            _USE_PGSERVER = False
            logger.info("pgserver not available; using bundled PostgreSQL")
    return _USE_PGSERVER


def using_pgserver():
    return _check_pgserver()


def start_embedded(data_dir):
    if _check_pgserver():
        import database
        from windows import embedded_pg
        try:
            return database.start_embedded(data_dir)
        except Exception:
            # On Windows, a hard kill (taskkill /f) can leave a non-empty
            # pgdata dir behind.  Clear it and retry once.
            if os.path.isdir(data_dir):
                logger.warning("PostgreSQL data dir stale after crash — clearing and retrying")
                import shutil
                shutil.rmtree(data_dir, ignore_errors=True)
                return database.start_embedded(data_dir)
            raise
    from windows import embedded_pg
    return embedded_pg.start(data_dir)


def ensure_embedded_running(data_dir):
    if _check_pgserver():
        import database
        return database.ensure_embedded_running(data_dir)
    from windows import embedded_pg
    return embedded_pg.ensure_running(data_dir)


def stop_embedded():
    if _check_pgserver():
        import database
        return database.stop_embedded()
    from windows import embedded_pg
    return embedded_pg.stop()
