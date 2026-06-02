"""Filesystem locations for the standalone Linux build.

Mirrors ``macos/paths.py`` but follows the XDG Base Directory spec instead of
``~/Library``.

Read-only resources (ONNX models, Flask templates/static, the bundled
``redis-server``, the bundled Postgres tools) live inside the PyInstaller bundle
-- ``sys._MEIPASS`` when frozen, the repo root in dev. Every *writable* path (the
Postgres data dir, the Redis socket, transcode scratch, the numba cache, logs,
the control socket and the supervisor pid file) lives under the user's data dir;
nothing writable ever lands inside the read-only, system-installed bundle
(``/opt/AudioMuse-AI`` on an installed package).

The writable root is ``$XDG_DATA_HOME/AudioMuse-AI`` (default
``~/.local/share/AudioMuse-AI``). Like the macOS build, the path must be
*space-free*: pgserver hands the embedded Postgres its unix-socket directory via
``pg_ctl -o '-k <dir>'`` -- a single string that ``postgres`` re-splits on
whitespace, so a space in the path makes startup fail with ``invalid argument``.
A normal Linux home (``/home/<user>``) is space-free; if a user's home somehow
contains a space we fall back to a space-free location under ``/tmp`` keyed by
uid so the embedded cluster still starts.
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
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _xdg_data_home():
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg and os.path.isabs(xdg):
        return xdg
    return os.path.join(os.path.expanduser("~"), ".local", "share")


def app_support_dir():
    """The writable root for all runtime state.

    Must not contain a space (see module docstring). If the natural location
    under the user's home would contain one, fall back to a per-uid dir under
    ``/tmp`` (always space-free) so the embedded Postgres can still start.
    """
    root = os.path.join(_xdg_data_home(), APP_NAME)
    if " " in root:
        try:
            uid = os.getuid()
        except AttributeError:  # pragma: no cover - non-POSIX, not a build target
            uid = 0
        root = os.path.join("/tmp", "AudioMuse-AI-%d" % uid)
    return _ensure(root)


def _xdg_state_home():
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg and os.path.isabs(xdg):
        return xdg
    return os.path.join(os.path.expanduser("~"), ".local", "state")


def logs_dir():
    d = os.path.join(_xdg_state_home(), APP_NAME, "logs")
    # Logs dir may legally contain a space (nothing re-splits it); but keep it
    # next to the other state by reusing the space-free fallback root when the
    # data root had to move.
    if " " in d:
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


def redis_socket_path():
    return os.path.join(redis_dir(), "redis.sock")


def control_socket_path():
    return os.path.join(app_support_dir(), "control.sock")


def pid_file():
    return os.path.join(app_support_dir(), "supervisor_pids.json")


def supervisor_lock_path():
    return os.path.join(app_support_dir(), "supervisor.lock")


def log_file():
    return os.path.join(logs_dir(), "audiomuse.log")


def model_dir():
    return os.path.join(resource_root(), "model")


def redis_binary():
    """Path to the embedded ``redis-server`` executable."""
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "redis-server")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor", "redis", platform.machine(), "redis-server",
    )


def pg_bin_dir():
    """Directory holding the bundled Postgres client tools (pg_dump, psql, pg_restore).

    Lives next to the pgserver server binaries inside the frozen bundle; in dev it
    resolves from the installed pgserver package.
    """
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "pgserver", "pginstall", "bin")
    import pgserver
    return os.path.join(os.path.dirname(os.path.abspath(pgserver.__file__)), "pginstall", "bin")
