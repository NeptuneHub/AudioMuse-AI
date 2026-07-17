# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The analysis package: FOR EACH SERVER -> FOR EACH ALBUM -> FOR EACH SONG.

A lazy facade: every attribute resolves on first use, so importing
``tasks.analysis`` stays free of heavy dependencies. RQ job strings
(``tasks.analysis.run_analysis_task``, ``tasks.analysis.analyze_album_task``,
``tasks.analysis.rebuild_all_indexes_task``) resolve here.

Main Features:
* main: the orchestrator (per-server phases, album dispatch, drain).
* album: the per-album RQ job and its per-song stage sequence.
* song: audio decode, the MusiCNN/CLAP/lyrics models and their persistence.
* helper: planning, fingerprint identity, work map and the task reporter.
* index: the similarity-index rebuild task.
"""

_HOMES = {
    'run_analysis_task': 'main',
    'run_analysis_server_task': 'main',
    '_run_analysis_server_task_impl': 'main',
    '_run_all_index_builds': 'main',
    '_verify_media_server_reachable': 'main',
    '_probe_looks_like_auth_failure': 'main',
    '_phase_outcome': 'main',
    '_albums_per_server': 'main',
    '_enabled_analysis_servers': 'main',
    '_rq_job_still_pending': 'main',
    'clean_temp': 'main',
    'analyze_album_task': 'album',
    '_analyze_album_task_impl': 'album',
    '_record_album_failure_row': 'album',
    'TrackNotAnalyzable': 'album',
    'rebuild_all_indexes_task': 'index',
    'sigmoid': 'song',
    'analyze_track': 'song',
    'robust_load_audio_with_fallback': 'song',
    'make_task_reporter': 'helper',
    '_bind_server_context': 'helper',
}


def __getattr__(name):
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module 'tasks.analysis' has no attribute '{name}'")
    import importlib

    module = importlib.import_module(f'.{home}', __name__)
    return getattr(module, name)
