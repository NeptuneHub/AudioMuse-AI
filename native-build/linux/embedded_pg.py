"""Embedded PostgreSQL manager for the standalone Linux **aarch64** build.

``pgserver`` (used on x86_64) publishes no Linux/aarch64 wheel, so on arm64 we
bundle a relocatable PostgreSQL built from source in CI (plus the ``unaccent`` /
``pg_trgm`` contrib modules) and manage its lifecycle here with ``initdb`` +
``pg_ctl``. This exposes the same ``start`` / ``ensure_running`` / ``stop``
surface that :mod:`linux.db_backend` routes to, and returns a libpq DSN over a
unix socket -- mirroring what ``pgserver.get_uri()`` provides on x86_64, so the
rest of the app (which only ever talks to ``config.DATABASE_URL`` via psycopg2)
is identical on both architectures.

The bundled server is relocatable: PostgreSQL derives its support-file paths
from the running executable's location, so the tree works wherever the package
is installed (``/opt/AudioMuse-AI/_internal/pgsql``). The unix socket lives in
the data dir itself (which ``native-build/linux/paths.py`` guarantees is space-free), exactly
like the pgserver/macOS setup.
"""

import logging
import os
import shutil
import subprocess
import threading

from linux import paths

logger = logging.getLogger("audiomuse.embedded_pg")

_lock = threading.RLock()
_data_dir = None  # remembered so stop() can target the right cluster

# Written only after initdb AND our config edits both succeed, so its presence
# proves a *complete* cluster. PG_VERSION alone is not enough: initdb writes it
# partway through, so a crash/SIGKILL mid-initdb leaves a half-built dir that
# looks initialized forever and wedges every later start.
_READY_MARKER = "audiomuse_initialized"


def _pg_env():
    """Environment for the bundled Postgres tools.

    The from-source build is relocatable via rpath, but we also set
    ``LD_LIBRARY_PATH`` defensively so the server/tools always find their own
    bundled shared libraries regardless of how they were invoked.

    We first scrub PyInstaller's injected ``LD_LIBRARY_PATH`` (which points at the
    frozen app's ``_internal`` libs): otherwise the bundle's incompatible
    ``libssl``/``libz``/... would sit on the search path after our pg libdir and
    a tool needing one of those (not shipped in the pg tree) could load the wrong
    copy and crash -- the same hazard that SIGSEGVs pgserver's ``initdb`` on x86_64.
    """
    from linux import env as env_builder
    env = env_builder.restore_native_lib_path(dict(os.environ))
    libdir = paths.pg_lib_dir()
    if libdir:
        parts = [libdir, os.path.join(libdir, "postgresql"), env.get("LD_LIBRARY_PATH", "")]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, parts))
    return env


def _bin(name):
    return os.path.join(paths.pg_bin_dir(), name)


def _initialized(data_dir):
    if os.path.exists(os.path.join(data_dir, _READY_MARKER)):
        return True
    # Legacy clusters created before the completion marker existed: adopt them if
    # initdb actually finished. global/pg_control is written by initdb and is
    # required for the server to start, so treat it as the completeness signal,
    # then stamp the marker so future starts take the fast path.
    if os.path.exists(os.path.join(data_dir, "global", "pg_control")):
        try:
            with open(os.path.join(data_dir, _READY_MARKER), "w", encoding="utf-8") as fh:
                fh.write("ok\n")
        except OSError:
            pass
        return True
    return False


def _has_cluster_data(data_dir):
    """True if data_dir holds an initialized PostgreSQL cluster (never auto-delete it).

    Keyed on ``global/pg_control``: written at the END of initdb and required for
    the server to start, so its presence proves a complete cluster with real data.
    A half-built dir (interrupted initdb, no pg_control) holds no usable data and
    is still cleared normally, so this guard never blocks first-run self-heal.
    """
    return os.path.exists(os.path.join(data_dir, "global", "pg_control"))


def _reset_data_dir(data_dir):
    """Empty a partially-initialized data dir so initdb can retry.

    initdb refuses a non-empty target, so a half-built cluster (interrupted
    initdb, no completion marker) would otherwise wedge every start forever.
    Only ever called when :func:`_initialized` is False; additionally refuses to
    delete a dir that still holds a real cluster, so a transient mis-detection
    can never destroy existing data (it surfaces an error instead)."""
    if not (os.path.isdir(data_dir) and os.listdir(data_dir)):
        return
    if _has_cluster_data(data_dir):
        raise RuntimeError(
            f"Refusing to wipe {data_dir}: it contains an existing PostgreSQL "
            "cluster (global/pg_control present). Back it up or remove it "
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


def _dsn(data_dir):
    # Unix-socket DSN: host is the socket directory (== data dir). trust auth,
    # no password; superuser/db are both ``postgres`` (created by initdb -U).
    return f"postgresql://postgres@/postgres?host={data_dir}"


def _is_running(data_dir, env):
    proc = subprocess.run(
        [_bin("pg_ctl"), "-D", data_dir, "status"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def _run_checked(argv, env):
    proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        out = (proc.stdout or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"{argv[0]} failed (exit {proc.returncode}): {out}")


def _init_cluster(data_dir, env):
    # --auth=trust: local-only socket, no password (matches the embedded model).
    # --locale=C -E UTF8: deterministic, ICU-free (the server is built
    # --without-icu); storage is UTF-8, search uses unaccent/lower anyway.
    _run_checked(
        [_bin("initdb"), "-D", data_dir, "-U", "postgres",
         "--auth=trust", "-E", "UTF8", "--locale=C"],
        env,
    )
    # Pin the socket dir to the data dir and disable TCP. Writing these into
    # postgresql.conf (rather than passing pg_ctl -o "-k ...") avoids postgres
    # re-splitting a single -o string on whitespace -- the same footgun the
    # macOS build documents; the data dir is space-free, but config keys are the
    # clean way regardless.
    conf = os.path.join(data_dir, "postgresql.conf")
    with open(conf, "a", encoding="utf-8") as fh:
        fh.write("\n# --- AudioMuse-AI embedded overrides ---\n")
        fh.write(f"unix_socket_directories = '{data_dir}'\n")
        fh.write("listen_addresses = ''\n")
    # Stamp the completion marker last: only now is the cluster fully usable, so
    # an interruption before this point leaves _initialized() False and triggers
    # a clean re-init on the next start.
    with open(os.path.join(data_dir, _READY_MARKER), "w", encoding="utf-8") as fh:
        fh.write("ok\n")


def start(data_dir):
    """Initialize (first run) and start the cluster; return its libpq DSN."""
    global _data_dir
    with _lock:
        env = _pg_env()
        if not _initialized(data_dir):
            logger.info("Initializing embedded PostgreSQL cluster at %s", data_dir)
            _reset_data_dir(data_dir)  # clear any half-built cluster first
            _init_cluster(data_dir, env)
        if not _is_running(data_dir, env):
            _run_checked([_bin("pg_ctl"), "-D", data_dir, "-w", "start"], env)
        _data_dir = data_dir
        return _dsn(data_dir)


def ensure_running(data_dir):
    """(Re)start the cluster if its postmaster died; return the DSN.

    ``start`` is already idempotent (it no-ops when the server is up and only
    runs initdb on a fresh dir), so the health loop can call this directly.
    ``pg_ctl start`` also clears a stale ``postmaster.pid`` from an unclean exit.
    """
    return start(data_dir)


def stop():
    """Cleanly stop the cluster started by :func:`start` (fast shutdown)."""
    global _data_dir
    with _lock:
        if _data_dir is None:
            return
        env = _pg_env()
        try:
            subprocess.run(
                [_bin("pg_ctl"), "-D", _data_dir, "-w", "-m", "fast", "stop"],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.exception("Error stopping embedded PostgreSQL")
        _data_dir = None
