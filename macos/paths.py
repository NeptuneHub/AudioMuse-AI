"""Filesystem locations for the standalone macOS build.

Read-only resources (ONNX models, Flask templates/static, the bundled
``redis-server``, icons) live inside the app bundle -- ``sys._MEIPASS`` when
frozen, the repo root in dev. Every *writable* path (the Postgres data dir, the
Redis socket, transcode scratch, the numba cache, logs, the control socket and
the supervisor pid file) lives under the user's ``~/Library``; nothing writable
ever lands inside the read-only, signed bundle.
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


def app_support_dir():
    return _ensure(os.path.join(os.path.expanduser("~"), "Library", "Application Support", APP_NAME))


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
