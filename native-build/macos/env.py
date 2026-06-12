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
        # RQ workers fork a child per job. On macOS, if any thread in the parent
        # is inside the Objective-C runtime when fork() happens (Foundation gets
        # pulled in transitively via pyobjc/CoreAudio/onnxruntime), the child
        # aborts with "+[NSNumber initialize] may have been in progress in
        # another thread when fork() was called. Crashing instead." -- so jobs
        # never run. This is the documented opt-out of that fork-safety check.
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
        # NumPy converts ints to longdouble via the C library's locale-sensitive
        # strtold. macOS frameworks (Cocoa/CoreAudio) call setlocale() at startup,
        # and a concurrent locale change while scipy imports causes an intermittent
        # "Could not parse python long as longdouble" crash that leaves analysis
        # tasks stuck pending. Pinning the *numeric* locale to C makes strtold
        # deterministic with no transition to race against. Kept to LC_NUMERIC only
        # so LC_CTYPE/UTF-8 (accented file paths) is untouched. The matching
        # in-process pin lives in native-build/macos/launcher.py (numeric_bootstrap.py); this
        # is all macOS-only and does not affect the Linux/Docker images.
        "LC_NUMERIC": "C",
        "APP_DATA_DIR": paths.app_support_dir(),
        "AUDIOMUSE_CONTROL_SOCKET": paths.control_socket_path(),
        "DATABASE_TYPE": "embedded",
        "QUEUE_TYPE": "embedded",
        "DATABASE_URL": database_url,
        "REDIS_URL": redis_url,
        "TEMP_DIR": paths.temp_audio_dir(),
        "NUMBA_CACHE_DIR": paths.numba_cache_dir(),
        # transformers/huggingface_hub look up models (e.g. the CLAP RoBERTa
        # tokenizer, loaded with local_files_only=True) in HF_HOME/hub. The
        # container points HF_HOME at a pre-baked cache; we bundle the same cache
        # at model/huggingface and point there. Offline flags guarantee no
        # network stall on a machine that may be offline -- the bundled cache is
        # complete (the integration suite runs CLAP+lyrics analysis with these
        # same offline flags set).
        "HF_HOME": os.path.join(model_dir, "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "EMBEDDING_MODEL_PATH": os.path.join(model_dir, "musicnn_embedding.onnx"),
        "PREDICTION_MODEL_PATH": os.path.join(model_dir, "musicnn_prediction.onnx"),
        "CLAP_AUDIO_MODEL_PATH": os.path.join(model_dir, "model_epoch_36.onnx"),
        "CLAP_TEXT_MODEL_PATH": os.path.join(model_dir, "clap_text_model.onnx"),
        "LYRICS_MODEL_DIR": model_dir,
        # The lyrics ONNX loaders (lyrics/silero_onnx.py, lyrics/gte_onnx.py)
        # hardcode container ``/app/model/...`` defaults and do NOT derive from
        # LYRICS_MODEL_DIR, so each needs its own override or the frozen app
        # looks at the Linux path. (Whisper does derive from LYRICS_MODEL_DIR,
        # but we set it explicitly too for clarity.)
        "LYRICS_WHISPER_MODEL_DIR": os.path.join(model_dir, "whisper-small-onnx"),
        "SILERO_VAD_ONNX_PATH": os.path.join(model_dir, "silero_vad.onnx"),
        "LYRICS_GTE_ONNX_PATH": os.path.join(model_dir, "gte-multilingual-base-int8.onnx"),
        "LYRICS_GTE_TOKENIZER_DIR": os.path.join(model_dir, "gte-multilingual-base"),
        # Backup/restore (app_backup.py). Its defaults are container-shaped: it
        # writes pg_dump output under /app/backup (read-only in the bundle) and
        # builds pg_dump/psql/pg_restore connection args from the POSTGRES_* vars
        # (TCP host/port that don't match the embedded server). Repoint all of it
        # at the embedded unix socket + the writable Library dir. Only app_backup
        # reads these POSTGRES_* values; the app itself connects via DATABASE_URL,
        # which the supervisor already sets, so this does not affect anything else.
        "BACKUP_DIR": paths.backup_dir(),
        "RESTORE_LOG_DIR": paths.backup_dir(),
        "POSTGRES_HOST": paths.pgdata_dir(),  # libpq treats a /path host as a unix-socket dir
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "",
        "POSTGRES_DB": "postgres",
        # Put the bundled Postgres client tools on PATH so pg_dump et al. resolve.
        "PATH": paths.pg_bin_dir() + os.pathsep + os.environ.get("PATH", ""),
    })
    if role in _WORKER_ROLES:
        env["AUDIOMUSE_ROLE"] = "worker"
        env["SERVICE_TYPE"] = "worker"
    else:
        env["SERVICE_TYPE"] = "flask"
        env.pop("AUDIOMUSE_ROLE", None)
    return env
