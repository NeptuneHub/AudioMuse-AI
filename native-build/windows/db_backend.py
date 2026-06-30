import importlib
import logging
import os
import subprocess
import tempfile
import threading
from urllib.parse import urlparse

from windows import paths

logger = logging.getLogger("audiomuse.db_backend")

_USE_PGSERVER = None

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_patch_lock = threading.Lock()
_embedded_lock = threading.Lock()


def _check_pgserver():
    global _USE_PGSERVER
    if _USE_PGSERVER is None:
        try:
            importlib.import_module("pgserver")
            _USE_PGSERVER = True
            logger.info("Using pgserver for embedded PostgreSQL")
        except ImportError:
            _USE_PGSERVER = False
            logger.info("pgserver not available; using bundled PostgreSQL")
    return _USE_PGSERVER


def using_pgserver():
    return _check_pgserver()


def _conn(host, port, password, user="postgres", dbname="postgres"):
    return {"host": host, "port": int(port), "user": user, "password": password, "dbname": dbname}


def _conn_from_uri(uri, password):
    parsed = urlparse(uri)
    dbname = (parsed.path or "/postgres").lstrip("/") or "postgres"
    return _conn(
        parsed.hostname or "127.0.0.1", parsed.port or paths.pg_port(), password, dbname=dbname
    )


def _initdb_bin():
    from pgserver._commands import POSTGRES_BIN_PATH

    name = "initdb.exe" if os.name == "nt" else "initdb"
    return str(POSTGRES_BIN_PATH / name)


def _preinit_scram(data_dir, password):
    pwfile = None
    try:
        fd, pwfile = tempfile.mkstemp(prefix="ampg_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(password)
        subprocess.run(
            [
                _initdb_bin(),
                "-D",
                data_dir,
                "-U",
                "postgres",
                "--auth-host=scram-sha-256",
                "--auth-local=scram-sha-256",
                "--encoding=utf8",
                f"--pwfile={pwfile}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            stdin=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
    finally:
        if pwfile and os.path.exists(pwfile):
            try:
                os.unlink(pwfile)
            except OSError:
                pass


def _harden_existing(data_dir, password, uri):
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


def _has_cluster_data(data_dir):
    return os.path.exists(os.path.join(data_dir, "global", "pg_control"))


def _clear_stale_data_dir(data_dir):
    if not (os.path.isdir(data_dir) and os.listdir(data_dir)):
        return
    if _has_cluster_data(data_dir):
        raise RuntimeError(
            f"Refusing to wipe {data_dir}: it contains an existing PostgreSQL "
            "cluster (global/pg_control present). Back it up or remove it "
            "manually if you really want a fresh start."
        )
    import shutil

    logger.warning("Clearing incomplete PostgreSQL data dir %s before init", data_dir)
    shutil.rmtree(data_dir, ignore_errors=True)


def _patch_pgserver_pg_ctl():
    if os.name != "nt":
        return
    import pgserver.postgres_server as ps

    if getattr(ps, "_audiomuse_pg_ctl_patched", False):
        return
    with _patch_lock:
        if getattr(ps, "_audiomuse_pg_ctl_patched", False):
            return
        original = ps.pg_ctl
        timeout = paths.pg_start_timeout()

        def pg_ctl(args, **kwargs):
            if args and "start" in args:
                if kwargs.get("timeout") is None or kwargs["timeout"] < timeout:
                    kwargs["timeout"] = timeout
                kwargs.setdefault("stdin", subprocess.DEVNULL)
                kwargs["creationflags"] = (kwargs.get("creationflags") or 0) | _NO_WINDOW
                env = kwargs.get("env")
                env = dict(os.environ if env is None else env)
                env.setdefault("PGCTLTIMEOUT", str(timeout))
                kwargs["env"] = env
            return original(args, **kwargs)

        ps.pg_ctl = pg_ctl
        ps._audiomuse_pg_ctl_patched = True


def start_embedded(data_dir):
    pw = paths.db_password()
    if _check_pgserver():
        _patch_pgserver_pg_ctl()
        import database

        fresh = not os.path.exists(os.path.join(data_dir, "PG_VERSION"))
        if fresh:
            _clear_stale_data_dir(data_dir)
            _preinit_scram(data_dir, pw)
        with _embedded_lock:
            try:
                uri = database.start_embedded(data_dir)
            except Exception:
                if not fresh:
                    raise
                logger.warning(
                    "Fresh PostgreSQL cluster failed to start - clearing and retrying once"
                )
                import shutil

                shutil.rmtree(data_dir, ignore_errors=True)
                _preinit_scram(data_dir, pw)
                uri = database.start_embedded(data_dir)
        if not fresh:
            _harden_existing(data_dir, pw, uri)
        return _conn_from_uri(uri, pw)
    from windows import embedded_pg

    return embedded_pg.start(data_dir, pw)


def ensure_embedded_running(data_dir):
    pw = paths.db_password()
    if _check_pgserver():
        _patch_pgserver_pg_ctl()
        import database

        with _embedded_lock:
            return _conn_from_uri(database.ensure_embedded_running(data_dir), pw)
    from windows import embedded_pg

    return embedded_pg.ensure_running(data_dir, pw)


def stop_embedded():
    with _embedded_lock:
        if _check_pgserver():
            import database

            return database.stop_embedded()
        from windows import embedded_pg

        return embedded_pg.stop()
