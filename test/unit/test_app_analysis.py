# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the app_analysis blueprint endpoints.

Registers the analysis blueprint and drives the analysis and cleaning start
routes with mocked queue and status calls to check parameters and gating.

Main Features:
* Analysis start with defaults, config defaults, and custom parameters.
* Enqueue parameters, pending-status saving, and active-task blocking.
* Cleaning start, prior-task cleanup, enqueue-failure, and method restrictions.
"""

import pytest
from unittest.mock import Mock, patch
from flask import Flask
import config
from app_analysis import analysis_bp


@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(analysis_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestCleaningPage:
    def test_cleaning_page_returns_html(self, client):
        with patch('app_analysis.render_template') as mock_render:
            mock_render.return_value = "<html>Cleaning Page</html>"

            response = client.get('/cleaning')

            assert response.status_code == 200
            mock_render.assert_called_once_with(
                'cleaning.html', title='AudioMuse-AI - Database Cleaning', active='cleaning',
                cleaning_catalogue_default=config.CLEANING_CATALOGUE,
            )


class TestStartAnalysisEndpoint:
    @pytest.fixture(autouse=True)
    def patch_active_analysis_task(self):
        with patch('app_analysis.get_active_main_task', return_value=None) as mock_active_task:
            yield mock_active_task

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_successful_analysis_start_with_defaults(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "test-job-123"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/analysis/start', json={})

        assert response.status_code == 202
        data = response.get_json()
        assert data['task_id'] == "test-job-123"
        assert data['task_type'] == "main_analysis"
        assert data['status'] == "queued"

        mock_cleanup.assert_called_once()

        mock_save_status.assert_called_once()
        save_call_args = mock_save_status.call_args[0]
        assert save_call_args[1] == "main_analysis"

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    @patch('app_analysis.NUM_RECENT_ALBUMS', 5)
    @patch('app_analysis.TOP_N_MOODS', 10)
    def test_analysis_start_uses_config_defaults(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "test-job-456"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/analysis/start', json={})

        assert response.status_code == 202

        mock_queue.enqueue.assert_called_once()
        call_kwargs = mock_queue.enqueue.call_args[1]
        assert call_kwargs['args'] == (5, 10)

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_start_with_custom_params(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "test-job-789"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post(
            '/api/analysis/start', json={'num_recent_albums': 10, 'top_n_moods': 15}
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data['task_id'] == "test-job-789"

        call_kwargs = mock_queue.enqueue.call_args[1]
        assert call_kwargs['args'] == (10, 15)

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_enqueue_task_parameters(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "test-job-abc"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post(
            '/api/analysis/start', json={'num_recent_albums': 3, 'top_n_moods': 5}
        )

        assert response.status_code == 202

        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args
        assert call_args[0][0] == 'tasks.analysis.run_analysis_task'
        assert call_args[1]['description'] == "Main Music Analysis"
        assert call_args[1]['job_timeout'] == -1

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_handles_missing_json(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "test-job-def"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/analysis/start', json={})

        assert response.status_code == 202

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_saves_pending_status(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "test-job-ghi"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/analysis/start', json={})

        assert response.status_code == 202

        mock_save_status.assert_called_once()
        call_args = mock_save_status.call_args[0]
        assert call_args[1] == "main_analysis"


class TestStartCleaningEndpoint:
    @pytest.fixture(autouse=True)
    def patch_active_cleaning_task(self):
        with patch('app_analysis.get_active_main_task', return_value=None) as mock_active_task:
            yield mock_active_task

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_successful_cleaning_start(self, mock_save_status, mock_cleanup, mock_queue, client):
        mock_job = Mock()
        mock_job.id = "clean-job-123"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/cleaning/start')

        assert response.status_code == 202
        data = response.get_json()
        assert data['task_id'] == "clean-job-123"
        assert data['task_type'] == "cleaning"
        assert data['status'] == "queued"

        mock_cleanup.assert_called_once()

        mock_save_status.assert_called_once()

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_enqueue_task_parameters(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "clean-job-456"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/cleaning/start')

        assert response.status_code == 202

        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args
        assert call_args[0][0] == 'tasks.cleaning.identify_and_clean_orphaned_albums_task'
        # The catalogue-deletion opt-in is passed positionally; with no request body it
        # falls back to the CLEANING_CATALOGUE env default (off).
        assert call_args[0][1] == config.CLEANING_CATALOGUE
        assert (
            call_args[1]['description'] == "Database Cleaning (Identify and Delete Orphaned Albums)"
        )
        assert call_args[1]['job_timeout'] == -1

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_forwards_clean_catalogue_flag_from_body(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "clean-job-789"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/cleaning/start', json={'clean_catalogue': True})

        assert response.status_code == 202
        call_args = mock_queue.enqueue.call_args
        assert call_args[0][1] is True

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_saves_pending_status(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "clean-job-789"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/cleaning/start')

        assert response.status_code == 202

        mock_save_status.assert_called_once()
        call_args = mock_save_status.call_args[0]
        assert call_args[1] == "cleaning"

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_cleans_up_previous_tasks(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        mock_job = Mock()
        mock_job.id = "clean-job-abc"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/cleaning/start')

        assert response.status_code == 202

        mock_cleanup.assert_called_once()

    @patch(
        'app_analysis.get_active_main_task',
        return_value={'task_id': 'existing-cleaning-123', 'status': 'STARTED'},
    )
    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_blocks_when_active_task_exists(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        response = client.post('/api/cleaning/start')

        assert response.status_code == 409
        data = response.get_json()
        assert data['task_id'] == 'existing-cleaning-123'
        assert data['status'] == 'STARTED'
        mock_cleanup.assert_not_called()
        mock_queue.enqueue.assert_not_called()


class TestEndpointErrorHandling:
    @patch('app_analysis.get_active_main_task', return_value=None)
    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_handles_enqueue_failure(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        mock_queue.enqueue.side_effect = Exception("Queue error")

        with pytest.raises(Exception, match="Queue error"):
            client.post('/api/analysis/start', json={})

    @patch(
        'app_analysis.get_active_main_task',
        return_value={
            'task_id': 'existing-cleaning-123',
            'status': 'STARTED',
            'task_type': 'cleaning',
        },
    )
    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_blocks_when_another_batch_is_active(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        response = client.post('/api/analysis/start', json={})

        assert response.status_code == 409
        assert response.get_json()['task_id'] == 'existing-cleaning-123'
        assert response.get_json()['status'] == 'STARTED'
        mock_cleanup.assert_not_called()
        mock_queue.enqueue.assert_not_called()

    @patch('app_analysis.get_active_main_task', return_value=None)
    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_handles_enqueue_failure(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        mock_queue.enqueue.side_effect = Exception("Queue error")

        with pytest.raises(Exception, match="Queue error"):
            client.post('/api/cleaning/start')


class TestBlueprintIntegration:
    def test_blueprint_registered_correctly(self, app):
        rules = [str(rule) for rule in app.url_map.iter_rules()]

        assert '/cleaning' in rules
        assert '/api/analysis/start' in rules
        assert '/api/cleaning/start' in rules

    def test_analysis_endpoint_accepts_post_only(self, client):
        response = client.get('/api/analysis/start')
        assert response.status_code == 405

    def test_cleaning_endpoint_accepts_post_only(self, client):
        response = client.get('/api/cleaning/start')
        assert response.status_code == 405

    def test_cleaning_page_accepts_get_only(self, client):
        response = client.post('/cleaning')
        assert response.status_code == 405
