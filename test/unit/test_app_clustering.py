# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the app_clustering blueprint start endpoint.

Registers the clustering blueprint and posts to the start route with mocked
queue and status calls to check enqueueing and active-task gating.

Main Features:
* Starts clustering when no task is active.
* Blocks when a clustering task or another batch is already active.
* Always enqueues with output_server_scope 'all' (batch tasks cover every
  server; a client-supplied scope is ignored).
"""

import pytest
from unittest.mock import Mock, patch
from flask import Flask
from app_clustering import clustering_bp


@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(clustering_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestStartClusteringEndpoint:
    @patch('app_clustering.get_active_main_task', return_value=None)
    @patch('app_clustering.rq_queue_high')
    @patch('app_clustering.clean_up_previous_main_tasks')
    @patch('app_clustering.save_task_status')
    def test_successful_clustering_start_with_no_active_task(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        mock_job = Mock()
        mock_job.id = "cluster-job-123"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/clustering/start', json={})

        assert response.status_code == 202
        data = response.get_json()
        assert data['task_id'] == "cluster-job-123"
        assert data['task_type'] == "main_clustering"
        assert data['status'] == "queued"
        mock_cleanup.assert_called_once()
        mock_save_status.assert_called_once()

    @patch('app_clustering.get_active_main_task', return_value=None)
    @patch('app_clustering.rq_queue_high')
    @patch('app_clustering.clean_up_previous_main_tasks')
    @patch('app_clustering.save_task_status')
    def test_start_enqueues_output_server_scope_all_even_when_a_scope_is_posted(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        mock_job = Mock()
        mock_job.id = "cluster-job-123"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/clustering/start', json={'output_server_scope': 's2'})

        assert response.status_code == 202
        enqueue_kwargs = mock_queue.enqueue.call_args.kwargs['kwargs']
        assert enqueue_kwargs['output_server_scope'] == 'all'

    @patch('app_clustering.get_active_main_task', return_value=None)
    @patch('app_clustering.rq_queue_high')
    @patch('app_clustering.clean_up_previous_main_tasks')
    @patch('app_clustering.save_task_status')
    def test_auto_parameter_discovery_defaults_on_and_can_be_disabled(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        mock_job = Mock()
        mock_job.id = "cluster-job-123"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/clustering/start', json={})
        assert response.status_code == 202
        assert mock_queue.enqueue.call_args.kwargs['kwargs']['auto_calibration_param'] is True

        response = client.post(
            '/api/clustering/start', json={'auto_parameter_discovery': False}
        )
        assert response.status_code == 202
        assert mock_queue.enqueue.call_args.kwargs['kwargs']['auto_calibration_param'] is False

    @patch(
        'app_clustering.get_active_main_task',
        return_value={'task_id': 'existing-clustering-123', 'status': 'STARTED'},
    )
    @patch('app_clustering.rq_queue_high')
    @patch('app_clustering.clean_up_previous_main_tasks')
    @patch('app_clustering.save_task_status')
    def test_clustering_blocks_when_active_task_exists(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        response = client.post('/api/clustering/start', json={})

        assert response.status_code == 409
        data = response.get_json()
        assert data['task_id'] == 'existing-clustering-123'
        assert data['status'] == 'STARTED'
        mock_cleanup.assert_not_called()
        mock_queue.enqueue.assert_not_called()

    @patch(
        'app_clustering.get_active_main_task',
        return_value={
            'task_id': 'existing-cleaning-123',
            'status': 'STARTED',
            'task_type': 'cleaning',
        },
    )
    @patch('app_clustering.rq_queue_high')
    @patch('app_clustering.clean_up_previous_main_tasks')
    @patch('app_clustering.save_task_status')
    def test_clustering_blocks_when_another_batch_is_active(
        self, mock_save_status, mock_cleanup, mock_queue, mock_get_active, client
    ):
        response = client.post('/api/clustering/start', json={})

        assert response.status_code == 409
        data = response.get_json()
        assert data['task_id'] == 'existing-cleaning-123'
        assert data['status'] == 'STARTED'
        mock_cleanup.assert_not_called()
        mock_queue.enqueue.assert_not_called()
