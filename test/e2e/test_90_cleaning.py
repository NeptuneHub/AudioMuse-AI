# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Database cleaning (ALGORITHM.md section 15) after files disappear from the server.

One analyzed clip and the broken file are moved out of the served folder and
Navidrome rescans, so the app sees an orphaned catalogue row and a stale
exclusion. A cleaning without catalogue deletion unbinds the mapping and keeps
the row; a cleaning with catalogue deletion removes the row and its embeddings;
after the file comes back, a new analysis re-creates it under the same
catalogue id because identity is derived from the audio. Runs last because it
mutates the library, and it puts every file back.

Main Features:
* unbound mappings and orphans are counted, exclusions of gone files dropped
* clean_catalogue=true deletes the orphaned row and cascades its embeddings
* the restored file is re-analyzed under the identical fp_ id
"""

import pytest

from test.e2e.e2e_helpers import item_id_of, scalar

pytestmark = pytest.mark.e2e

REMOVED_CLIP = 'D03'
CLEANING_TIMEOUT = 900


@pytest.fixture(scope='module')
def removed_files(stack, library, analyzed_library):
    yield
    library.restore_all()
    stack.rescan_library()


def _clean(api, clean_catalogue):
    api.wait_idle(180)
    task_id = api.start_task('/api/cleaning/start', {'clean_catalogue': clean_catalogue})
    final = api.wait_for_task(task_id, timeout=CLEANING_TIMEOUT)
    api.wait_idle(60)
    assert final['task_type_from_db'] == 'cleaning'
    summary = final['details'].get('final_summary_details')
    assert isinstance(summary, dict), final['details']
    return summary


def test_cleaning_unbinds_then_deletes_and_reanalysis_restores(stack, api, db, library, removed_files, golden):
    pid = library.pid(REMOVED_CLIP)
    fp_before = item_id_of(db, pid)
    assert fp_before and fp_before.startswith('fp_')
    exclusions_before = scalar(db, 'SELECT count(*) FROM analysis_exclusions')
    assert exclusions_before == len(library.unanalyzable)

    library.remove_file(REMOVED_CLIP)
    for key in library.unanalyzable:
        library.remove_file(key)
    remaining = stack.expected_files - 1 - len(library.unanalyzable)
    stack.rescan_library(expected_files=remaining)

    summary = _clean(api, clean_catalogue=False)
    golden.check('cleaning summary without catalogue deletion', summary)
    assert summary['catalogue_deletion'] is False
    assert summary['failed_servers'] == [], summary
    assert summary['total_media_server_tracks'] == remaining, summary
    assert summary['unbound_mappings'] == 1, summary
    assert summary['orphaned_tracks_count'] == 1, summary
    assert summary['deleted_count'] == 0, summary
    assert summary['deleted_analysis_exclusions'] == len(library.unanalyzable), summary
    assert item_id_of(db, pid) is None
    assert scalar(db, 'SELECT count(*) FROM score WHERE item_id = %s', (fp_before,)) == 1
    assert scalar(db, 'SELECT count(*) FROM analysis_exclusions') == 0
    assert scalar(db, 'SELECT track_count FROM music_servers WHERE is_default') == remaining

    summary = _clean(api, clean_catalogue=True)
    golden.check('cleaning summary with catalogue deletion', summary)
    assert summary['catalogue_deletion'] is True
    assert summary['deleted_count'] == 1, summary
    assert summary['remaining_orphans_count'] == 0, summary
    assert scalar(db, 'SELECT count(*) FROM score WHERE item_id = %s', (fp_before,)) == 0
    assert scalar(db, 'SELECT count(*) FROM embedding WHERE item_id = %s', (fp_before,)) == 0
    assert scalar(db, 'SELECT count(*) FROM clap_embedding WHERE item_id = %s', (fp_before,)) == 0
    assert api.json('GET', '/api/search_tracks?search_query=' + library.track(REMOVED_CLIP).title[:10]) == []

    library.restore_file(REMOVED_CLIP)
    stack.rescan_library(expected_files=remaining + 1)
    api.wait_idle(120)
    task_id = api.start_task('/api/analysis/start', {'num_recent_albums': 0, 'top_n_moods': 5})
    final = api.wait_for_task(task_id, timeout=900)
    api.wait_idle(120)
    assert final['details'].get('tracks_analyzed') == 1, final['details']
    assert item_id_of(db, library.pid(REMOVED_CLIP)) == fp_before
    assert scalar(db, 'SELECT count(*) FROM score') == stack.catalogue_rows
