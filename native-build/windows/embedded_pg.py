"""Embedded PostgreSQL manager for the standalone Windows build (fallback).

Used when ``pgserver`` has no Windows wheel. Manages a bundled relocatable
PostgreSQL with ``initdb`` + ``pg_ctl``, exposing the same ``start`` /
``ensure_running`` / ``stop`` surface that :mod:`windows.db_backend` routes to.

The bundled server is relocatable: PostgreSQL on Windows derives its support-file
paths from the running executable's location, so the tree works wherever the
package is installed (``C:\\Program Files\\AudioMuse-AI\\_internal\\pgsql``).

Unlike macOS/Linux, there are no Unix sockets on Windows -- PostgreSQL listens on
TCP 127.0.0.1.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading

from windows import paths

logger = logging.getLogger("audiomuse.embedded_pg")

_lock = threading.RLock()
_running_proc = None  # the postgres.exe process handle

_READY_MARKER = "audiomuse_initialized"


def _bin(name):
    ext = ".exe" if sys.platform == "win32" else ""
    return os.path.join(paths.pg_bin_dir(), name + ext)


def _conn(password):
    return {"host": "127.0.0.1", "port": paths.pg_port(), "user": "postgres",
            "password": password, "dbname": "postgres"}


def _initialized(data_dir):
    if os.path.exists(os.path.join(data_dir, _READY_MARKER)):
        return True
    if os.path.exists(os.path.join(data_dir, "global", "pg_control")):
        try:
            with open(os.path.join(data_dir, _READY_MARKER), "w", encoding="utf-8") as fh:
                fh.write("ok\n")
        except OSError:
            pass
        return True
    return False


def _has_cluster_data(data_dir):
    """True if data_dir holds a real PostgreSQL cluster that must never be auto-deleted."""
    return (
        os.path.exists(os.path.join(data_dir, "PG_VERSION"))
        or os.path.exists(os.path.join(data_dir, "global", "pg_control"))
        or os.path.isdir(os.path.join(data_dir, "base"))
    )


def _reset_data_dir(data_dir):
    if not (os.path.isdir(data_dir) and os.listdir(data_dir)):
        return
    if _has_cluster_data(data_dir):
        raise RuntimeError(
            f"Refusing to wipe {data_dir}: it contains an existing PostgreSQL "
            "cluster (PG_VERSION/pg_control/base present). Back it up or remove it "
            "manually if you really want a fresh start."
        )
    logger.warning("Clearing incomplete PostgreSQL data dir %s before re-init", data_dir)
    for entry in os.listdir(data_dir):
        target = os.path.join(data_dir, entry)
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)
        except OSError:
            logger.exception("Could not remove %s", target)


def start(data_dir, password):
    global _running_proc
    with _lock:
        os.makedirs(data_dir, exist_ok=True)

        if not _initialized(data_dir):
            _reset_data_dir(data_dir)
            logger.info("Initializing PostgreSQL cluster at %s", data_dir)
            pwfile = None
            try:
                fd, pwfile = tempfile.mkstemp(prefix="ampg_")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(password)
                subprocess.run(
                    [_bin("initdb"), "-D", data_dir, "-U", "postgres",
                     "--auth-host=scram-sha-256", "--auth-local=scram-sha-256",
                     f"--pwfile={pwfile}", "--no-locale", "--encoding=UTF8"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            finally:
                if pwfile and os.path.exists(pwfile):
                    try:
                        os.unlink(pwfile)
                    except OSError:
                        pass
            with open(os.path.join(data_dir, "postgresql.conf"), "a", encoding="utf-8", newline="\n") as fh:
                fh.write("\npassword_encryption = scram-sha-256\n")
            with open(os.path.join(data_dir, _READY_MARKER), "w", encoding="utf-8") as fh:
                fh.write("ok\n")

        port = str(paths.pg_port())
        logger.info("Starting PostgreSQL on 127.0.0.1:%s", port)
        _running_proc = subprocess.Popen(
            [_bin("pg_ctl"), "start", "-D", data_dir,
             "-o", f"-p {port} -h 127.0.0.1",
             "-l", os.path.join(data_dir, "pg.log")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Wait for the server to be ready.
        for _ in range(60):
            result = subprocess.run(
                [_bin("pg_isready"), "-h", "127.0.0.1", "-p", port, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                break
            import time
            time.sleep(0.5)
        else:
            raise RuntimeError("PostgreSQL did not start within 30 seconds")

        return _conn(password)


def ensure_running(data_dir, password):
    port = str(paths.pg_port())
    result = subprocess.run(
        [_bin("pg_isready"), "-h", "127.0.0.1", "-p", port, "-q"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return _conn(password)
    return start(data_dir, password)


def stop():
    global _running_proc
    with _lock:
        if _running_proc is None:
            return
        data_dir = paths.pgdata_dir()
        logger.info("Stopping PostgreSQL")
        try:
            subprocess.run(
                [_bin("pg_ctl"), "stop", "-D", data_dir, "-m", "fast"],
                timeout=15,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            if _running_proc.poll() is None:
                _running_proc.terminate()
        _running_proc = None
