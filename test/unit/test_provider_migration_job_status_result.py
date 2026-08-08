# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The migration status endpoint reports the summary the execute task really wrote.

The execute job finishes itself: `_report_migration` merges its summary into the
top level of `details` and writes SUCCESS, so the worker finalize that would have
parked it under `final_summary_details` early-returns on a row that is no longer
RUNNING. Reading only the worker's key answered `result: null` for every execute
job, so the wizard could never render its index-reset note.

Main Features:
* A self-finished execute row surfaces ok / matched / index_rebuild_needed
* The worker-written `final_summary_details` still wins for dry-run style jobs
* A row with no summary at all still answers a null result, not an empty object
"""

import os
import sys
import importlib.util
import pytest
import config
from contextlib import nullcontext
from unittest.mock import MagicMock


def _load_bp_module():
    mod_name = 'app_provider_migration'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'app_provider_migration.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bp_mod():
    return _load_bp_module()


@pytest.fixture
def client(bp_mod, monkeypatch):
    from flask import Flask

    monkeypatch.setattr(bp_mod, 'main_task_start_lock', nullcontext)
    app = Flask(__name__)
    app.register_blueprint(bp_mod.migration_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def task_row(monkeypatch):
    import database

    holder = {'row': None}
    monkeypatch.setattr(
        database, 'get_task_info_from_db', lambda task_id: holder['row'], raising=False
    )
    return holder


@pytest.fixture(autouse=True)
def never_classify_or_restart(bp_mod, monkeypatch):
    monkeypatch.setattr(bp_mod, '_task_is_the_execute_job', lambda task_id: False)
    monkeypatch.setattr(bp_mod, 'get_db', MagicMock())


def test_self_finished_execute_summary_is_read_from_the_top_level_of_details(
    client, task_row
):
    task_row['row'] = {
        'status': config.TASK_STATUS_SUCCESS,
        'details': {
            'message': 'Provider migration complete: 12 tracks repointed.',
            'status_message': 'Provider migration complete: 12 tracks repointed.',
            'ok': True,
            'matched': 12,
            'index_rebuild_needed': True,
        },
    }
    payload = client.get('/api/migration/status/exec-1').get_json()
    assert payload['result'] == {'ok': True, 'matched': 12, 'index_rebuild_needed': True}


def test_the_index_reset_flag_survives_when_it_is_false(client, task_row):
    task_row['row'] = {
        'status': config.TASK_STATUS_SUCCESS,
        'details': {'ok': True, 'matched': 3, 'index_rebuild_needed': False},
    }
    payload = client.get('/api/migration/status/exec-2').get_json()
    assert payload['result']['index_rebuild_needed'] is False


def test_an_already_applied_retry_reports_its_summary_too(client, task_row):
    task_row['row'] = {
        'status': config.TASK_STATUS_SUCCESS,
        'details': {
            'message': 'Provider migration applied and worker restart acknowledged.',
            'ok': True,
            'matched': 7,
            'index_rebuild_needed': False,
            'already_applied': True,
        },
    }
    payload = client.get('/api/migration/status/exec-3').get_json()
    assert payload['result']['already_applied'] is True


def test_the_worker_written_summary_still_wins_for_planner_jobs(client, task_row):
    task_row['row'] = {
        'status': config.TASK_STATUS_SUCCESS,
        'details': {
            'final_summary_details': {'tier_counts': {'path': 4}, 'unmatched': 1},
            'ok': 'this top-level key must not shadow the worker payload',
        },
    }
    payload = client.get('/api/migration/status/dry-1').get_json()
    assert payload['result'] == {'tier_counts': {'path': 4}, 'unmatched': 1}


def test_a_running_row_without_a_summary_reports_a_null_result(client, task_row):
    task_row['row'] = {
        'status': config.TASK_STATUS_STARTED,
        'details': {'message': 'Provider migration started...'},
    }
    payload = client.get('/api/migration/status/exec-4').get_json()
    assert payload['result'] is None


def test_a_missing_task_is_still_a_404(client, task_row):
    task_row['row'] = None
    response = client.get('/api/migration/status/nope')
    assert response.status_code == 404
