"""Unit tests for app_clustering.py Flask blueprint"""

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
    @patch('app_helper.get_active_main_task', return_value=None)
    @patch('app_helper.rq_queue_high')
    @patch('app_helper.clean_up_previous_main_tasks')
    @patch('app_helper.save_task_status')
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

    @patch('app_helper.get_active_main_task', return_value={'task_id': 'existing-clustering-123', 'status': 'STARTED'})
    @patch('app_helper.rq_queue_high')
    @patch('app_helper.clean_up_previous_main_tasks')
    @patch('app_helper.save_task_status')
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

    @patch('app_helper.get_active_main_task', return_value={'task_id': 'existing-cleaning-123', 'status': 'STARTED', 'task_type': 'cleaning'})
    @patch('app_helper.rq_queue_high')
    @patch('app_helper.clean_up_previous_main_tasks')
    @patch('app_helper.save_task_status')
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
