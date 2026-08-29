# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The child-process environment every native build hands its supervised children.

The model, cache, temp and backup variables are the same on all three platforms
and were maintained as three copies of one dictionary, so a new model path had
to be added three times or a build silently ran without it. Here they are built
once from the platform paths module; each platform still supplies its own label,
its own control transport and its own POSTGRES_* values, because those are the
only parts that genuinely differ.

``pg_conn_parts`` lives here too: config.py assembles every connection string
from the five POSTGRES_* parts, so the standalone builds must report the host
and port the embedded server actually opened. pgserver picks the socket
directory at runtime and falls back to a hashed path when pgdata is too long for
a Unix socket, so the query string is read raw rather than through a URL parser
that would unquote a directory name containing a plus or a hash.

Main Features:
* One model/cache/temp/backup environment shared by Linux, macOS and Windows
* Worker roles get AUDIOMUSE_ROLE/SERVICE_TYPE; everything else runs as flask
* FPCALC is exported only when the vendored binary is actually present
"""

import os
from urllib.parse import unquote, urlsplit

import service_roles

_WORKER_ROLES = service_roles.WORKER_ROLES


def socket_dir_from_url(database_url):
    marker = "?host="
    at = (database_url or "").find(marker)
    if at < 0:
        marker = "&host="
        at = (database_url or "").find(marker)
    if at < 0:
        return ""
    return unquote((database_url or "")[at + len(marker):])


def pg_conn_parts(database_url, fallback_host):
    parts = urlsplit(database_url or "")
    socket_dir = socket_dir_from_url(database_url)
    return (
        socket_dir or parts.hostname or fallback_host(),
        str(parts.port or 5432),
    )


def build_child_env(paths, role, database_url, postgres, extra):
    env = dict(os.environ)
    model_dir = paths.model_dir()
    env.update(
        {
            "APP_DATA_DIR": paths.app_support_dir(),
            "DATABASE_TYPE": "embedded",
            "DATABASE_URL": database_url,
            "TEMP_DIR": paths.temp_audio_dir(),
            "NUMBA_CACHE_DIR": paths.numba_cache_dir(),
            "HF_HOME": os.path.join(model_dir, "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "EMBEDDING_MODEL_PATH": os.path.join(model_dir, "musicnn_embedding.onnx"),
            "PREDICTION_MODEL_PATH": os.path.join(model_dir, "musicnn_prediction.onnx"),
            "CLAP_AUDIO_MODEL_PATH": os.path.join(model_dir, "model_epoch_36.onnx"),
            "CLAP_TEXT_MODEL_PATH": os.path.join(model_dir, "clap_text_model.onnx"),
            "CLAP_SAE_ENCODER_PATH": os.path.join(
                model_dir, "dclap_sae_k20_d1024_best_encoder.onnx"
            ),
            "CLAP_SAE_MODEL_PATH": os.path.join(
                model_dir, "dclap_sae_k20_d1024_best_decoder.onnx"
            ),
            "LYRICS_MODEL_DIR": model_dir,
            "LYRICS_WHISPER_MODEL_DIR": os.path.join(model_dir, "whisper-small-onnx"),
            "SILERO_VAD_ONNX_PATH": os.path.join(model_dir, "silero_vad.onnx"),
            "LYRICS_GTE_ONNX_PATH": os.path.join(model_dir, "gte-multilingual-base-int8.onnx"),
            "LYRICS_GTE_TOKENIZER_DIR": os.path.join(model_dir, "gte-multilingual-base"),
            "BACKUP_DIR": paths.backup_dir(),
            "RESTORE_LOG_DIR": paths.backup_dir(),
            "POSTGRES_HOST": postgres["host"],
            "POSTGRES_PORT": postgres["port"],
            "POSTGRES_USER": postgres["user"],
            "POSTGRES_PASSWORD": postgres["password"],
            "POSTGRES_DB": postgres["dbname"],
            "PATH": os.pathsep.join(filter(None, [paths.pg_bin_dir(), os.environ.get("PATH")])),
        }
    )
    env.update(extra)
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


def embedded_postgres_parts(database_url, fallback_host):
    pg_host, pg_port = pg_conn_parts(database_url, fallback_host)
    return {
        "host": pg_host,
        "port": pg_port,
        "user": "postgres",
        "password": "",
        "dbname": "postgres",
    }
