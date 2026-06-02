"""Embedded-database backend selector for the standalone Linux build.

Two mechanisms provide the embedded PostgreSQL, chosen by architecture:

* **x86_64** -- the shared :mod:`database` module's pgserver path (unchanged;
  ``pgserver`` ships a manylinux x86_64 wheel with PostgreSQL + pgvector).
* **aarch64** -- :mod:`linux.embedded_pg`, which manages a from-source,
  relocatable PostgreSQL bundled in the package (pgserver has no arm64 wheel).

Both expose the same ``start_embedded`` / ``ensure_embedded_running`` /
``stop_embedded`` surface and return a libpq DSN, so :mod:`linux.supervisor`
calls these without caring which backend is active. This keeps the
arch-specific branching in one tiny place and changes **no shared code**.
"""

import platform

# pgserver wheels exist for x86_64 (and amd64 is the same machine reported by
# some libcs); everything else uses the bundled-from-source server.
_USE_PGSERVER = platform.machine() in ("x86_64", "amd64")


def using_pgserver():
    return _USE_PGSERVER


def start_embedded(data_dir):
    if _USE_PGSERVER:
        import database
        return database.start_embedded(data_dir)
    from linux import embedded_pg
    return embedded_pg.start(data_dir)


def ensure_embedded_running(data_dir):
    if _USE_PGSERVER:
        import database
        return database.ensure_embedded_running(data_dir)
    from linux import embedded_pg
    return embedded_pg.ensure_running(data_dir)


def stop_embedded():
    if _USE_PGSERVER:
        import database
        return database.stop_embedded()
    from linux import embedded_pg
    return embedded_pg.stop()
