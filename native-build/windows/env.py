"""Build the environment handed to each supervised child process (Windows build).

Windows counterpart of ``native-build/linux/env.py``. The standalone-mode overrides are
centralized here so every child imports ``config`` already pointed at the
embedded services and the bundled models.

Why ``AUDIOMUSE_PLATFORM=macos`` on Windows
-------------------------------------------
``restart_manager.py`` (shared code we must not modify) has exactly one
platform-keyed branch: ``if config.AUDIOMUSE_PLATFORM == 'macos'`` it forwards
restart requests to a socket-based *control server* instead of shelling out to
``supervisorctl`` (which only exists in the container). That control-server
protocol is platform-agnostic, and this build's supervisor implements it
identically (via ``macos.control_ipc.ControlServer`` adapted to TCP on Windows).
So reporting ``macos`` here makes the web UI's "save config -> restart workers"
flow work on the native Windows build with **zero shared-code changes**. The
value is an internal "standalone embedded supervisor" signal, not a real OS check.
"""

import os
import sys
from urllib.parse import quote

from windows import paths

_WORKER_ROLES = {"worker-high", "worker-default", "janitor", "restart-listener"}


def build_child_env(role, db_conn, redis_url):
    """Return an ``os.environ`` copy with the embedded-mode overrides for ``role``."""
    env = dict(os.environ)
    model_dir = paths.model_dir()
    database_url = (
        f"postgresql://{quote(db_conn['user'], safe='')}:"
        f"{quote(db_conn['password'], safe='')}"
        f"@{db_conn['host']}:{db_conn['port']}/{db_conn['dbname']}"
    )
    env.update({
        # See module docstring: selects the control-socket restart path in the
        # shared restart_manager.py. Not a real OS check.
        "AUDIOMUSE_PLATFORM": "macos",
        "APP_DATA_DIR": paths.app_support_dir(),
        # Windows has no AF_UNIX.  Set the control socket to empty so
        # restart_manager._send_control bails out safely (logs an error,
        # returns False) instead of crashing on ``socket.AF_UNIX``.
        # Restart-on-config-change is a no-op on Windows; users restart
        # the app manually.
        "AUDIOMUSE_CONTROL_SOCKET": "",
        "AUDIOMUSE_CONTROL_HOST": "127.0.0.1",
        "AUDIOMUSE_CONTROL_PORT": str(paths.control_port()),
        "DATABASE_TYPE": "embedded",
        "QUEUE_TYPE": "embedded",
        "DATABASE_URL": database_url,
        "REDIS_URL": redis_url,
        "TEMP_DIR": paths.temp_audio_dir(),
        "NUMBA_CACHE_DIR": paths.numba_cache_dir(),
        # transformers/huggingface_hub look up models in HF_HOME/hub.
        "HF_HOME": os.path.join(model_dir, "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "EMBEDDING_MODEL_PATH": os.path.join(model_dir, "musicnn_embedding.onnx"),
        "PREDICTION_MODEL_PATH": os.path.join(model_dir, "musicnn_prediction.onnx"),
        "CLAP_AUDIO_MODEL_PATH": os.path.join(model_dir, "model_epoch_36.onnx"),
        "CLAP_TEXT_MODEL_PATH": os.path.join(model_dir, "clap_text_model.onnx"),
        "LYRICS_MODEL_DIR": model_dir,
        "LYRICS_WHISPER_MODEL_DIR": os.path.join(model_dir, "whisper-small-onnx"),
        "SILERO_VAD_ONNX_PATH": os.path.join(model_dir, "silero_vad.onnx"),
        "LYRICS_GTE_ONNX_PATH": os.path.join(model_dir, "gte-multilingual-base-int8.onnx"),
        "LYRICS_GTE_TOKENIZER_DIR": os.path.join(model_dir, "gte-multilingual-base"),
        # Backup/restore. Repoint at the writable data dir.
        "BACKUP_DIR": paths.backup_dir(),
        "RESTORE_LOG_DIR": paths.backup_dir(),
        "POSTGRES_HOST": db_conn["host"],
        "POSTGRES_PORT": str(db_conn["port"]),
        "POSTGRES_USER": db_conn["user"],
        "POSTGRES_PASSWORD": db_conn["password"],
        "POSTGRES_DB": db_conn["dbname"],
        # Put the bundled Postgres client tools on PATH.
        "PATH": paths.pg_bin_dir() + os.pathsep + os.environ.get("PATH", ""),
    })
    if role in _WORKER_ROLES:
        env["AUDIOMUSE_ROLE"] = "worker"
        env["SERVICE_TYPE"] = "worker"
    else:
        env["SERVICE_TYPE"] = "flask"
        env.pop("AUDIOMUSE_ROLE", None)
    return env
