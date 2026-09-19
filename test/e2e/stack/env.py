# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one environment both app processes of the end-to-end stack are started with.

Every variable set here is a name config.py already reads; the harness never
invents one. The wizard is bypassed by configuring the default media server
from the environment (app_auth.setup_status is satisfied by a valid navidrome
configuration plus AUTH_ENABLED=false, and database._seed_registry_from_legacy_config
seeds the music_servers row at first boot). The small-library knobs are set
before the first boot because app.py persists every missing parameter into
app_config at startup and the database is the source from then on, for the
workers too.

Main Features:
* build_env(role, ...) starts from a scrubbed copy of the parent environment
  so a developer's shell exports cannot leak into the stack
* AUDIOMUSE_CONTROL_SOCKET points restart_manager at the harness control server
* TUNABLES is the single place the catalogue-size knobs live
* masked() renders the environment for env.json without secrets
"""

import os

STRIP_PREFIXES = (
    'POSTGRES_',
    'NAVIDROME_',
    'JELLYFIN_',
    'EMBY_',
    'PLEX_',
    'LYRION_',
    'AUDIOMUSE_',
    'MEDIASERVER_',
    'MUSIC_LIBRARIES',
    'AI_',
    'OPENAI_',
    'GEMINI_',
    'MISTRAL_',
    'OLLAMA_',
    'FLASK_',
    'LYRICS_',
    'CLAP_',
    'NEURAL_',
    'SERVICE_TYPE',
    'HF_',
    'TRANSFORMERS_',
    'TEMP_DIR',
    'IVF_DISK_CACHE_DIR',
    'PLUGINS_',
    'BACKUP_DIR',
    'FPCALC',
    'CUDA_VISIBLE_DEVICES',
    'AUTH_ENABLED',
    'DISABLE_FLASK_RESTART',
)

SECRET_MARKERS = ('PASSWORD', 'TOKEN', 'SECRET', 'API_KEY')

NAVIDROME_ADMIN_USER = 'admin'
NAVIDROME_ADMIN_PASSWORD = 'audiomuse-e2e'

FEATURES = {
    'AUTH_ENABLED': 'false',
    'AI_MODEL_PROVIDER': 'NONE',
    'DISABLE_FLASK_RESTART': 'true',
    'FLASK_BUILTIN_HTTPS': 'false',
    'PLUGINS_ENABLED': 'false',
    'CLAP_ENABLED': 'true',
    'LYRICS_ENABLED': 'true',
    'LYRICS_API_ENABLE': 'false',
    'LYRICS_ASR_ENABLE': 'true',
    'LYRICS_MUSICNN_SKIP': 'false',
    'NEURAL_FINGERPRINT_ENABLED': 'true',
    'CLAP_SAE_STEERING_ENABLED': 'true',
    'CHROMAPRINT_COLLECTION_ENABLED': 'true',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'PYTHONUNBUFFERED': '1',
    'TZ': 'UTC',
    'CUDA_VISIBLE_DEVICES': '',
}

TUNABLES = {
    'MIN_PLAYLIST_SIZE_FOR_TOP_N': '2',
    'NUM_CLUSTERS_MIN': '2',
    'NUM_CLUSTERS_MAX': '4',
    'GMM_N_COMPONENTS_MIN': '2',
    'GMM_N_COMPONENTS_MAX': '4',
    'SPECTRAL_N_CLUSTERS_MIN': '2',
    'SPECTRAL_N_CLUSTERS_MAX': '4',
    'CLUSTERING_RUNS': '10',
    'PCA_COMPONENTS_MAX': '8',
    'TOP_N_CLUSTERING_PLAYLIST': '3',
    'CLUSTERING_AUTO_CALIBRATION': 'false',
    'HYPERBOLIC_MIN_CLUSTER_SIZE': '2',
    'HYPERBOLIC_TARGET_LEAF_SIZE': '5',
    'PER_SONG_MODEL_RELOAD': 'false',
    'ANALYSIS_MONITOR_DB_INTERVAL': '2',
}


def scrubbed_parent_env():
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(STRIP_PREFIXES)
    }


def build_env(role, pg_parts, data_root, navidrome_url, models, fpcalc=None, extra_path=(), control_socket=None):
    env = scrubbed_parent_env()
    env.update(pg_parts)
    env.update(
        {
            'MEDIASERVER_TYPE': 'navidrome',
            'NAVIDROME_URL': navidrome_url,
            'NAVIDROME_USER': NAVIDROME_ADMIN_USER,
            'NAVIDROME_PASSWORD': NAVIDROME_ADMIN_PASSWORD,
            'NAVIDROME_API_KEY': '',
            'MUSIC_LIBRARIES': '',
            'TEMP_DIR': os.path.join(data_root, 'temp_audio'),
            'IVF_DISK_CACHE_DIR': os.path.join(data_root, 'ivf_cache'),
            'FLASK_HTTPS_CERT_DIR': os.path.join(data_root, 'tls'),
            'PLUGINS_DIR': os.path.join(data_root, 'plugins'),
            'BACKUP_DIR': os.path.join(data_root, 'backups'),
            'NUMBA_CACHE_DIR': os.path.join(data_root, 'numba_cache'),
        }
    )
    env.update(FEATURES)
    env.update(models)
    env.update(TUNABLES)
    if fpcalc:
        env['FPCALC'] = fpcalc
    if control_socket:
        env['AUDIOMUSE_CONTROL_SOCKET'] = control_socket
    path_parts = [p for p in extra_path if p] + [env.get('PATH', '')]
    env['PATH'] = os.pathsep.join(path_parts)
    if role == 'flask':
        env['SERVICE_TYPE'] = 'flask'
    else:
        env['SERVICE_TYPE'] = 'worker'
        env['AUDIOMUSE_ROLE'] = 'worker'
    return env


def masked(env):
    return {
        key: ('***' if any(marker in key for marker in SECRET_MARKERS) and value else value)
        for key, value in sorted(env.items())
    }
