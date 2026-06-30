import os
import platform
import sys

APP_NAME = "AudioMuse-AI"


def resource_root():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", _repo_root())
    return _repo_root()


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _xdg_data_home():
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg and os.path.isabs(xdg):
        return xdg
    return os.path.join(os.path.expanduser("~"), ".local", "share")


def app_support_dir():
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
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "redis-server")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vendor",
        "redis",
        platform.machine(),
        "redis-server",
    )


def _uses_pgserver():
    return platform.machine() in ("x86_64", "amd64")


def pg_install_dir():
    if _uses_pgserver():
        if getattr(sys, "frozen", False):
            return os.path.join(resource_root(), "pgserver", "pginstall")
        import pgserver

        return os.path.join(os.path.dirname(os.path.abspath(pgserver.__file__)), "pginstall")
    if getattr(sys, "frozen", False):
        return os.path.join(resource_root(), "pgsql")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vendor", "postgres", platform.machine()
    )


def pg_bin_dir():
    return os.path.join(pg_install_dir(), "bin")


def pg_lib_dir():
    if _uses_pgserver():
        return None
    return os.path.join(pg_install_dir(), "lib")
