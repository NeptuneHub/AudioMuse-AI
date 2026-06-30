import contextlib
import os
import sys

from linux import paths

_WORKER_ROLES = {"worker-high", "worker-default", "janitor", "restart-listener"}


def restore_native_lib_path(env):
    if not getattr(sys, "frozen", False):
        return env
    orig = env.get("LD_LIBRARY_PATH_ORIG")
    if orig:
        env["LD_LIBRARY_PATH"] = orig
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


@contextlib.contextmanager
def native_lib_path_restored():
    if not getattr(sys, "frozen", False):
        yield
        return
    saved = os.environ.get("LD_LIBRARY_PATH")
    try:
        restore_native_lib_path(os.environ)
        yield
    finally:
        if saved is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = saved


def build_child_env(role, database_url, redis_url):
    env = dict(os.environ)
    model_dir = paths.model_dir()
    env.update(
        {
            "AUDIOMUSE_PLATFORM": "macos",
            "APP_DATA_DIR": paths.app_support_dir(),
            "AUDIOMUSE_CONTROL_SOCKET": paths.control_socket_path(),
            "DATABASE_TYPE": "embedded",
            "QUEUE_TYPE": "embedded",
            "DATABASE_URL": database_url,
            "REDIS_URL": redis_url,
            "TEMP_DIR": paths.temp_audio_dir(),
            "NUMBA_CACHE_DIR": paths.numba_cache_dir(),
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
            "BACKUP_DIR": paths.backup_dir(),
            "RESTORE_LOG_DIR": paths.backup_dir(),
            "POSTGRES_HOST": paths.pgdata_dir(),
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "",
            "POSTGRES_DB": "postgres",
            "PATH": os.pathsep.join(filter(None, [paths.pg_bin_dir(), os.environ.get("PATH")])),
        }
    )
    if role in _WORKER_ROLES:
        env["AUDIOMUSE_ROLE"] = "worker"
        env["SERVICE_TYPE"] = "worker"
    else:
        env["SERVICE_TYPE"] = "flask"
        env.pop("AUDIOMUSE_ROLE", None)
    return env
