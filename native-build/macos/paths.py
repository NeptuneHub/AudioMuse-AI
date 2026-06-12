"""Filesystem locations for the standalone macOS build.

Read-only resources (ONNX models, Flask templates/static, the bundled
``redis-server``, icons) live inside the app bundle -- ``sys._MEIPASS`` when
frozen, the repo root in dev. Every *writable* path (the Postgres data dir, the
Redis socket, transcode scratch, the numba cache, logs, the control socket and
the supervisor pid file) lives under the user's ``~/Library``; nothing writable
ever lands inside the read-only, signed bundle. The writable root is
``~/Library/AudioMuse-AI`` (deliberately *not* ``Application Support`` -- see
``app_support_dir`` for why the path must be space-free).
"""

import os
import platform
import sys

APP_NAME = "AudioMuse-AI"


def resource_root():
    """Directory that holds the bundled read-only resources."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", _repo_root())
    return _repo_root()


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def app_support_dir():
    # NB: ``~/Library/AudioMuse-AI`` rather than the conventional
    # ``~/Library/Application Support/AudioMuse-AI``. The path must not contain a
    # space: pgserver hands the embedded Postgres its unix-socket directory via
    # ``pg_ctl -o '-k <dir>'``, a single string that ``postgres`` re-splits on
    # whitespace -- a space in the path makes startup fail with
    # ``invalid argument``. The cluster's data dir doubles as its socket dir, so
    # the whole writable root stays space-free.
    return _ensure(os.path.join(os.path.expanduser("~"), "Library", APP_NAME))


def logs_dir():
    return _ensure(os.path.join(os.path.expanduser("~"), "Library", "Logs", APP_NAME))


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


def redis_socket_path():
    return os.path.join(redis_dir(), "redis.sock")


def control_socket_path():
    return os.path.join(app_support_dir(), "control.sock")


def pid_file():
    return os.path.join(app_support_dir(), "supervisor_pids.json")


def log_file():
    return os.path.join(logs_dir(), "audiomuse.log")


def model_dir():
    return os.path.join(resource_root(), "model")


def menubar_icon():
    return os.path.join(resource_root(), "assets", "menubar-icon.png")


def redis_binary():
    """Path to the embedded ``redis-server`` executable."""
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "redis-server")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "redis", platform.machine(), "redis-server")


def pg_bin_dir():
    """Directory holding the bundled Postgres client tools (pg_dump, psql, pg_restore).

    Lives next to the pgserver server binaries inside the frozen bundle; in dev it
    resolves from the installed pgserver package.
    """
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "pgserver", "pginstall", "bin")
    import pgserver
    return os.path.join(os.path.dirname(os.path.abspath(pgserver.__file__)), "pginstall", "bin")
