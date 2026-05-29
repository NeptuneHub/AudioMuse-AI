"""Build the environment handed to each supervised child process.

The macOS-specific overrides are centralized here so every child imports
``config`` with all settings already pointed at the embedded services and the
bundled models. Keys mirror the env-driven settings in ``config.py``
(``DATABASE_URL``, ``REDIS_URL``, ``TEMP_DIR``, the ``*_MODEL_PATH`` values,
``LYRICS_MODEL_DIR``) plus the runtime signals the supervisor and
``restart_manager`` agree on (``AUDIOMUSE_PLATFORM``, control socket, roles).
"""

import os

from macos import paths

_WORKER_ROLES = {"worker-high", "worker-default", "janitor", "restart-listener"}


def build_child_env(role, database_url, redis_url):
    """Return an ``os.environ`` copy with the embedded-mode overrides for ``role``."""
    env = dict(os.environ)
    model_dir = paths.model_dir()
    env.update({
        "AUDIOMUSE_PLATFORM": "macos",
        "APP_DATA_DIR": paths.app_support_dir(),
        "AUDIOMUSE_CONTROL_SOCKET": paths.control_socket_path(),
        "DATABASE_TYPE": "embedded",
        "QUEUE_TYPE": "embedded",
        "DATABASE_URL": database_url,
        "REDIS_URL": redis_url,
        "TEMP_DIR": paths.temp_audio_dir(),
        "NUMBA_CACHE_DIR": paths.numba_cache_dir(),
        "EMBEDDING_MODEL_PATH": os.path.join(model_dir, "musicnn_embedding.onnx"),
        "PREDICTION_MODEL_PATH": os.path.join(model_dir, "musicnn_prediction.onnx"),
        "CLAP_AUDIO_MODEL_PATH": os.path.join(model_dir, "model_epoch_36.onnx"),
        "CLAP_TEXT_MODEL_PATH": os.path.join(model_dir, "clap_text_model.onnx"),
        "LYRICS_MODEL_DIR": model_dir,
    })
    if role in _WORKER_ROLES:
        env["AUDIOMUSE_ROLE"] = "worker"
        env["SERVICE_TYPE"] = "worker"
    else:
        env["SERVICE_TYPE"] = "flask"
        env.pop("AUDIOMUSE_ROLE", None)
    return env
