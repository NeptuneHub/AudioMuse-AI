# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared task-details sanitizer used by every task-status endpoint.

Pins the guarantees the frontend relies on: no traceback or heavyweight internal
keys reach the client, logs are truncated, and a failed task always carries a
well-formed structured error regardless of which endpoint produced it.

Main Features:
* Traceback and analysis-only keys are stripped; logs collapse to the last 10.
* Failed tasks always gain a structured error dict plus a mirrored error_message.
"""

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app_helper import sanitize_task_details
from error import error_dictionary as ed


class TestSanitizeTaskDetails:
    def test_non_dict_passthrough(self):
        assert sanitize_task_details(None, 'FAILURE', 'main_analysis') is None
        assert sanitize_task_details('str', 'SUCCESS', None) == 'str'

    def test_traceback_is_removed(self):
        out = sanitize_task_details({'traceback': 'secret\nstack'}, 'SUCCESS', 'x')
        assert 'traceback' not in out

    def test_checked_album_ids_removed_only_for_analysis(self):
        analysis = sanitize_task_details({'checked_album_ids': [1, 2]}, 'PROGRESS', 'main_analysis')
        assert 'checked_album_ids' not in analysis
        other = sanitize_task_details({'checked_album_ids': [1, 2]}, 'PROGRESS', 'main_clustering')
        assert 'checked_album_ids' in other

    def test_log_truncated_to_last_ten(self):
        out = sanitize_task_details({'log': [f'line {i}' for i in range(25)]}, 'SUCCESS', 'x')
        assert len(out['log']) == 11
        assert 'truncated' in out['log'][0]
        assert out['log'][-1] == 'line 24'

    def test_failure_without_error_backfills_unknown(self):
        out = sanitize_task_details({}, 'FAILURE', 'main_analysis')
        assert out['error']['error_code'] == ed.UNKNOWN_ERROR_CODE
        assert out['error_message'] == out['error']['error_message']

    def test_failure_with_code_only_is_rebuilt(self):
        out = sanitize_task_details({'error': {'error_code': ed.ERR_DB_CONNECTION}}, 'FAILURE', 'x')
        assert out['error']['error_class'] == 'Database Error'
        assert out['error']['error_message']

    def test_failure_with_full_error_is_preserved(self):
        structured = {
            'error_code': ed.ERR_ANALYSIS_FAILED,
            'error_class': 'Analysis Error',
            'error_message': 'Audio analysis failed. detail',
        }
        out = sanitize_task_details({'error': dict(structured)}, 'FAILURE', 'x')
        assert out['error'] == structured

    def test_success_task_is_not_given_an_error(self):
        out = sanitize_task_details({'log': ['ok']}, 'SUCCESS', 'x')
        assert 'error' not in out

    def test_clustering_batch_internal_track_ids_are_stripped(self):
        out = sanitize_task_details(
            {
                'best_score_in_batch': 12.0,
                'final_subset_track_ids': ['fp_3aaa', 'fp_3bbb'],
                'full_best_result_from_batch': {'named_playlists': {'Rock': ['fp_3ccc']}},
            },
            'SUCCESS', 'clustering_batch',
        )
        assert 'final_subset_track_ids' not in out
        assert 'full_best_result_from_batch' not in out
        assert out['best_score_in_batch'] == 12.0

    def test_cleaning_orphaned_track_item_ids_are_stripped_keeping_labels(self):
        out = sanitize_task_details(
            {
                'final_summary_details': {
                    'orphaned_tracks_count': 1,
                    'orphaned_albums': [
                        {'artist': 'A', 'track_count': 1,
                         'tracks': [{'item_id': 'fp_3zzz', 'title': 'T', 'author': 'A'}]},
                    ],
                }
            },
            'SUCCESS', 'cleaning',
        )
        track = out['final_summary_details']['orphaned_albums'][0]['tracks'][0]
        assert 'item_id' not in track
        assert track == {'title': 'T', 'author': 'A'}
        assert out['final_summary_details']['orphaned_tracks_count'] == 1
