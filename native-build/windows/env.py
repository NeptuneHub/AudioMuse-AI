
import os
from urllib.parse import quote

from windows import paths

_WORKER_ROLES = {"worker-high", "worker-default", "janitor", "restart-listener"}


def build_child_env(role, db_conn, redis_url):
    env = dict(os.environ)
    model_dir = paths.model_dir()
    database_url = (
        f"postgresql://{quote(db_conn['user'], safe='')}:"
        f"{quote(db_conn['password'], safe='')}"
        f"@{db_conn['host']}:{db_conn['port']}/{db_conn['dbname']}"
    )
    env.update({
        "AUDIOMUSE_PLATFORM": "macos",
        "APP_DATA_DIR": paths.app_support_dir(),
        "AUDIOMUSE_CONTROL_SOCKET": "",
        "AUDIOMUSE_CONTROL_HOST": "127.0.0.1",
        "AUDIOMUSE_CONTROL_PORT": str(paths.control_port()),
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
        "POSTGRES_HOST": db_conn["host"],
        "POSTGRES_PORT": str(db_conn["port"]),
        "POSTGRES_USER": db_conn["user"],
        "POSTGRES_PASSWORD": db_conn["password"],
        "POSTGRES_DB": db_conn["dbname"],
        "PATH": paths.pg_bin_dir() + os.pathsep + os.environ.get("PATH", ""),
    })
    if role in _WORKER_ROLES:
        env["AUDIOMUSE_ROLE"] = "worker"
        env["SERVICE_TYPE"] = "worker"
    else:
        env["SERVICE_TYPE"] = "flask"
        env.pop("AUDIOMUSE_ROLE", None)
    return env
