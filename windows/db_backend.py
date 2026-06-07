"""Embedded-database backend selector for the standalone Windows build.

Two mechanisms provide the embedded PostgreSQL:

* **pgserver available** -- the shared :mod:`database` module's pgserver path
  (pgserver ships a Windows wheel with PostgreSQL + pgvector).
* **Fallback** -- :mod:`windows.embedded_pg`, which manages a from-source,
  relocatable PostgreSQL bundled in the package (if pgserver has no Windows wheel).

Both expose the same ``start_embedded`` / ``ensure_embedded_running`` /
``stop_embedded`` surface and return a connection-info dict
(``host``/``port``/``user``/``password``/``dbname``), so :mod:`windows.supervisor`
and :mod:`windows.env` build the child ``DATABASE_URL``/``POSTGRES_*`` from one
source of truth. This keeps the platform-specific branching in one tiny place and
changes **no shared code**.

Unlike macOS/Linux (owner-only unix sockets), Windows has no AF_UNIX, so the
cluster listens on loopback TCP, which any local process can reach. To keep the
same "no unauthenticated local access" guarantee, the cluster is initialized with
``scram-sha-256`` auth and a generated superuser password
(:func:`windows.paths.db_password`) baked in at ``initdb`` time -- so the server
is never reachable without the password and there is no trust window.
"""

import logging
import os
import subprocess
import tempfile
from urllib.parse import urlparse

from windows import paths

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


def _conn(host, port, password, user="postgres", dbname="postgres"):
    return {"host": host, "port": int(port), "user": user,
            "password": password, "dbname": dbname}


def _conn_from_uri(uri, password):
    """Build the conn dict from pgserver's password-less URI plus our password."""
    parsed = urlparse(uri)
    dbname = (parsed.path or "/postgres").lstrip("/") or "postgres"
    return _conn(parsed.hostname or "127.0.0.1", parsed.port or paths.pg_port(),
                 password, dbname=dbname)


def _initdb_bin():
    from pgserver._commands import POSTGRES_BIN_PATH
    name = "initdb.exe" if os.name == "nt" else "initdb"
    return str(POSTGRES_BIN_PATH / name)


def _preinit_scram(data_dir, password):
    """Initialize the cluster with scram auth + ``password`` before pgserver sees it.

    pgserver's ``ensure_pgdata_inited`` runs its own ``initdb --auth=trust`` only
    when ``PG_VERSION`` is absent. Running our own scram ``initdb`` first (with
    pgserver's own bundled binary, so the cluster is version-compatible) makes
    pgserver skip that step and start straight into password-enforced mode -- the
    server is never reachable without the password.
    """
    pwfile = None
    try:
        fd, pwfile = tempfile.mkstemp(prefix="ampg_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(password)
        subprocess.run(
            [_initdb_bin(), "-D", data_dir, "-U", "postgres",
             "--auth-host=scram-sha-256", "--auth-local=scram-sha-256",
             "--encoding=utf8", f"--pwfile={pwfile}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    finally:
        if pwfile and os.path.exists(pwfile):
            try:
                os.unlink(pwfile)
            except OSError:
                pass


def _harden_existing(data_dir, password, uri):
    """Best-effort upgrade of a legacy trust cluster (created before this fix).

    Only triggers when ``PG_VERSION`` already existed and ``pg_hba.conf`` still
    grants ``trust``. Sets the superuser password (PG16 stores a scram verifier by
    default), rewrites the ``trust`` entries to ``scram-sha-256`` and reloads.
    Never blocks startup: failures are logged and ignored.
    """
    hba = os.path.join(data_dir, "pg_hba.conf")
    try:
        with open(hba, "r", encoding="utf-8") as fh:
            content = fh.read()
    except OSError:
        return

    def _is_trust(line):
        s = line.strip()
        return bool(s) and not s.startswith("#") and s.split()[-1] == "trust"

    if not any(_is_trust(line) for line in content.splitlines()):
        return
    try:
        import psycopg2
        conn = psycopg2.connect(uri)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("ALTER USER postgres PASSWORD %s", (password,))
            new_lines = []
            for line in content.splitlines():
                if _is_trust(line):
                    line = line[: line.rfind("trust")] + "scram-sha-256"
                new_lines.append(line)
            with open(hba, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(new_lines) + "\n")
            with conn.cursor() as cur:
                cur.execute("SELECT pg_reload_conf()")
        finally:
            conn.close()
        logger.info("Upgraded legacy trust PostgreSQL cluster to scram-sha-256")
    except Exception:
        logger.exception("Could not upgrade legacy PostgreSQL auth; leaving as-is")


def _clear_stale_data_dir(data_dir):
    """``initdb`` refuses to run against a non-empty target directory.

    A crash mid-init -- or a partial cleanup of an earlier failed start -- can
    leave an un-initialized data dir behind (e.g. just a ``log/`` subdir) that
    has no ``PG_VERSION`` yet still makes the fresh ``initdb`` fail, bricking
    every subsequent start. Wipe such leftovers before initializing.
    """
    if not (os.path.isdir(data_dir) and os.listdir(data_dir)):
        return
    import shutil
    logger.warning("Clearing incomplete PostgreSQL data dir %s before init", data_dir)
    shutil.rmtree(data_dir, ignore_errors=True)


def start_embedded(data_dir):
    pw = paths.db_password()
    if _check_pgserver():
        import database
        fresh = not os.path.exists(os.path.join(data_dir, "PG_VERSION"))
        if fresh:
            _clear_stale_data_dir(data_dir)
            _preinit_scram(data_dir, pw)
        try:
            uri = database.start_embedded(data_dir)
        except Exception:
            if os.path.isdir(data_dir):
                logger.warning("PostgreSQL data dir stale after crash — clearing and retrying")
                import shutil
                shutil.rmtree(data_dir, ignore_errors=True)
                _preinit_scram(data_dir, pw)
                uri = database.start_embedded(data_dir)
            else:
                raise
        if not fresh:
            _harden_existing(data_dir, pw, uri)
        return _conn_from_uri(uri, pw)
    from windows import embedded_pg
    return embedded_pg.start(data_dir, pw)


def ensure_embedded_running(data_dir):
    pw = paths.db_password()
    if _check_pgserver():
        import database
        return _conn_from_uri(database.ensure_embedded_running(data_dir), pw)
    from windows import embedded_pg
    return embedded_pg.ensure_running(data_dir, pw)


def stop_embedded():
    if _check_pgserver():
        import database
        return database.stop_embedded()
    from windows import embedded_pg
    return embedded_pg.stop()
