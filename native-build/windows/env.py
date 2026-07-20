# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Child-process environment builder for the Windows standalone build.

Assembles the environment variables each supervised child (Flask, RQ workers)
inherits: the embedded database URL built from the connection dict, queue
selection, per-user data and model paths, offline-model flags and the loopback
control host/port (used instead of the macOS control socket). The macOS/Linux
``env`` modules build the equivalent environments for their platforms.

Main Features:
* Builds the DATABASE_URL from the embedded connection and per-user data paths.
* Points model/cache/temp/backup paths at ``windows.paths`` and forces offline models.
"""

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
    env.update(
        {
            "AUDIOMUSE_PLATFORM": "windows",
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
        }
    )
    fpcalc = paths.fpcalc_binary()
    if os.path.exists(fpcalc):
        env["FPCALC"] = fpcalc
    if role in _WORKER_ROLES:
        env["AUDIOMUSE_ROLE"] = "worker"
        env["SERVICE_TYPE"] = "worker"
    else:
        env["SERVICE_TYPE"] = "flask"
        env.pop("AUDIOMUSE_ROLE", None)
    return env
