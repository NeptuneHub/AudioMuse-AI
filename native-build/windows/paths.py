"""Filesystem locations for the standalone Windows build.

Mirrors ``native-build/linux/paths.py`` but uses Windows conventions (``%LOCALAPPDATA%``)
instead of XDG directories.

Read-only resources (ONNX models, Flask templates/static, the bundled
``redis-server.exe``, the bundled Postgres tools) live inside the PyInstaller
bundle -- ``sys._MEIPASS`` when frozen, the repo root in dev. Every *writable*
path (the Postgres data dir, the Redis working dir, transcode scratch, the numba
cache, logs, the control socket and the supervisor pid file) lives under
``%LOCALAPPDATA%\\AudioMuse-AI``; nothing writable ever lands inside the
read-only bundle directory (wherever the zip is extracted).

The writable root is ``%LOCALAPPDATA%\\AudioMuse-AI`` (typically
``C:\\Users\\<user>\\AppData\\Local\\AudioMuse-AI``). Like the macOS/Linux builds,
the path must be *space-free*: pgserver hands the embedded Postgres its data
directory as a command-line argument, and a space in the path makes startup fail.
A normal Windows user profile (``C:\\Users\\<user>``) is space-free; if a user's
profile path somehow contains a space we fall back to ``C:\\ProgramData\\AudioMuse-AI``
so the embedded cluster still starts.
"""

import os
import platform
import secrets
import sys
from urllib.parse import quote

APP_NAME = "AudioMuse-AI"


def resource_root():
    """Directory that holds the bundled read-only resources."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", _repo_root())
    return _repo_root()


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tray_icon():
    return os.path.join(resource_root(), "assets", "AudioMuse-AI.ico")


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def app_support_dir():
    """The writable root for all runtime state.

    Must not contain a space (see module docstring). If the natural location
    under the user's LOCALAPPDATA would contain one, fall back to
    ``C:\\ProgramData\\AudioMuse-AI`` so the embedded Postgres can still start.
    """
    local_appdata = os.environ.get("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local"))
    root = os.path.join(local_appdata, APP_NAME)
    if " " in root:
        root = os.path.join(os.environ.get("PROGRAMDATA", "C:\\ProgramData"), APP_NAME)
    return _ensure(root)


def logs_dir():
    d = os.path.join(app_support_dir(), "logs")
    return _ensure(d)


def pgdata_dir():
    return _ensure(os.path.join(app_support_dir(), "pgdata"))


def redis_dir():
    return _ensure(os.path.join(app_support_dir(), "redis"))


def temp_audio_dir():
    return _ensure(os.path.join(app_support_dir(), "temp_audio"))


def numba_cache_dir():
    return _ensure(os.path.join(app_support_dir(), "numba_cache"))


def backup_dir():
    """Writable dir for pg_dump backups / restore logs (``app_backup.py``)."""
    return _ensure(os.path.join(app_support_dir(), "backup"))


def secrets_dir():
    return _ensure(os.path.join(app_support_dir(), "secrets"))


def _secret(name):
    """Return a persisted random token, generating it on first use.

    Windows has no AF_UNIX, so the embedded Postgres/Redis listen on loopback TCP
    rather than the owner-only unix sockets used on macOS/Linux. A loopback port
    is reachable by any local process, so each service is gated by a generated
    per-install secret instead. The token lives under the user-private
    ``%LOCALAPPDATA%`` profile (default Windows ACLs are owner-only).
    """
    path = os.path.join(secrets_dir(), name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            value = fh.read().strip()
        if value:
            return value
    except OSError:
        pass
    value = secrets.token_urlsafe(32)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(value)
    return value


def db_password():
    """Generated superuser password for the embedded PostgreSQL."""
    return _secret("pg_password")


def redis_password():
    """Generated password for the embedded Redis (``requirepass``)."""
    return _secret("redis_password")


def redis_port():
    """TCP port for the embedded Redis (no unix sockets on Windows)."""
    return 6379


def pg_port():
    """TCP port for embedded PostgreSQL."""
    return 5432


def pg_start_timeout():
    """Seconds to wait for embedded PostgreSQL to report ready before giving up.

    pgserver's built-in 10s is too short when the supervisor boots from the tray
    app's daemon thread under a hidden console, and on a cold first start while
    Windows Defender scans the freshly extracted binaries. A premature timeout
    orphans postgres.exe holding the data dir and bricks the next start, so allow
    a generous window.
    """
    return 120


def redis_url():
    """Password-bearing URL for the embedded Redis (loopback TCP, scram-equivalent gate)."""
    return f"redis://:{quote(redis_password(), safe='')}@127.0.0.1:{redis_port()}/0"


def control_port():
    """TCP port for the control server (replaces Unix socket on Windows)."""
    return 8001


def pid_file():
    return os.path.join(app_support_dir(), "supervisor_pids.json")


def supervisor_lock_path():
    return os.path.join(app_support_dir(), "supervisor.lock")


def log_file():
    return os.path.join(logs_dir(), "audiomuse.log")


def model_dir():
    return os.path.join(resource_root(), "model")


def redis_binary():
    """Path to the embedded ``redis-server.exe`` executable."""
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "redis-server.exe")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "redis", platform.machine().lower(), "redis-server.exe")


def _pgserver_pginstall():
    """Root of pgserver's bundled PostgreSQL install, or None when pgserver is absent.

    pgserver (the default backend) ships the full PostgreSQL tree -- including the
    client tools pg_dump/pg_restore/psql -- under ``pgserver/pginstall``. The
    ``pgsql``/``vendor/postgres`` layout below is only used by the fallback
    ``embedded_pg`` backend.
    """
    if getattr(sys, "frozen", False):
        cand = os.path.join(resource_root(), "pgserver", "pginstall")
        return cand if os.path.isdir(cand) else None
    try:
        import pgserver
        cand = os.path.join(os.path.dirname(pgserver.__file__), "pginstall")
        return cand if os.path.isdir(cand) else None
    except Exception:
        return None


def pg_bin_dir():
    """Directory containing the bundled PostgreSQL client tools (pg_dump, psql, etc.)."""
    pginstall = _pgserver_pginstall()
    if pginstall:
        return os.path.join(pginstall, "bin")
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "pgsql", "bin")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "postgres", platform.machine().lower(), "bin")


def pg_lib_dir():
    """Directory containing the bundled PostgreSQL shared libraries."""
    pginstall = _pgserver_pginstall()
    if pginstall:
        return os.path.join(pginstall, "lib")
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "pgsql", "lib")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "postgres", platform.machine().lower(), "lib")
