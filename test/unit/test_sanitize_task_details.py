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
* A response with no log is given a one-entry one from its own status message, so
  the `details.log` array survives on the wire without any task writing it to the
  task_status row.
* Failed tasks always gain a structured error dict plus a mirrored error_message.
* The main clustering task's best_result loses its per-song playlist composition
  and per-playlist embedding vectors here only - the database copy stays whole,
  because a worker restart resumes from it and it is what actually becomes
  playlists at the end of the run.
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

    def test_a_running_task_without_a_log_gets_one_from_its_status_message(self):
        out = sanitize_task_details(
            {'status_message': 'Analysing album 12 of 40.'}, 'RUNNING', 'main_analysis'
        )
        assert out['log'] == ['Analysing album 12 of 40.']

    def test_a_failed_task_without_a_log_gets_one_from_its_message(self):
        out = sanitize_task_details({'message': 'It broke.'}, 'FAIL', 'main_clustering')
        assert out['log'] == ['It broke.']

    def test_status_message_wins_over_message_when_both_are_present(self):
        out = sanitize_task_details(
            {'status_message': 'live', 'message': 'stale'}, 'RUNNING', 'cleaning'
        )
        assert out['log'] == ['live']

    def test_an_existing_log_is_never_replaced_by_the_status_message(self):
        out = sanitize_task_details(
            {'log': ['kept'], 'status_message': 'ignored'}, 'RUNNING', 'x'
        )
        assert out['log'] == ['kept']

    def test_an_empty_log_list_is_refilled_from_the_status_message(self):
        out = sanitize_task_details({'log': [], 'status_message': 'live'}, 'RUNNING', 'x')
        assert out['log'] == ['live']

    def test_details_with_no_message_at_all_stay_without_a_log(self):
        out = sanitize_task_details({'progress': 40}, 'RUNNING', 'x')
        assert 'log' not in out

    def test_a_non_string_message_is_coerced_into_a_string_log_entry(self):
        out = sanitize_task_details({'message': 42}, 'RUNNING', 'x')
        assert out['log'] == ['42']

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

    def test_cleaning_orphaned_legacy_track_id_is_kept(self):
        out = sanitize_task_details(
            {
                'final_summary_details': {
                    'orphaned_albums': [
                        {'artist': 'A', 'track_count': 1,
                         'tracks': [{'item_id': 'jelly-legacy-1', 'title': 'T', 'author': 'A'}]},
                    ],
                }
            },
            'SUCCESS', 'cleaning',
        )
        # A legacy provider id is not an internal fp_ id, so it must NOT be stripped.
        assert out['final_summary_details']['orphaned_albums'][0]['tracks'][0]['item_id'] == 'jelly-legacy-1'

    def test_non_list_orphaned_albums_does_not_crash(self):
        out = sanitize_task_details(
            {'final_summary_details': {'orphaned_albums': 'unexpected'}}, 'SUCCESS', 'cleaning'
        )
        assert out['final_summary_details']['orphaned_albums'] == 'unexpected'

    def test_the_main_clustering_tasks_best_result_loses_its_heavy_playlist_data(self):
        out = sanitize_task_details(
            {
                'best_result': {
                    'fitness_score': 0.87,
                    'parameters': {'n_clusters': 12},
                    'named_playlists': {'Rock': ['fp_1', 'fp_2']},
                    'playlist_centroids': {'Rock': [0.1, 0.2, 0.3]},
                    'playlist_to_centroid_vector_map': {'Rock': [0.1, 0.2, 0.3]},
                },
            },
            'PROGRESS', 'main_clustering',
        )
        best_result = out['best_result']
        assert 'named_playlists' not in best_result
        assert 'playlist_centroids' not in best_result
        assert 'playlist_to_centroid_vector_map' not in best_result
        assert best_result['fitness_score'] == 0.87
        assert best_result['parameters'] == {'n_clusters': 12}

    def test_a_details_dict_with_no_best_result_is_left_alone(self):
        out = sanitize_task_details({'progress': 40}, 'PROGRESS', 'main_clustering')
        assert 'best_result' not in out

    def test_a_non_dict_best_result_does_not_crash(self):
        out = sanitize_task_details({'best_result': 'unexpected'}, 'PROGRESS', 'main_clustering')
        assert out['best_result'] == 'unexpected'
