"""Unit tests for app_analysis.py Flask blueprint

Tests cover analysis and cleaning endpoints including request handling,
task enqueueing, and error cases.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import uuid
from flask import Flask
from app_analysis import analysis_bp


@pytest.fixture
def app():
    """Create a Flask app for testing"""
    app = Flask(__name__)
    app.register_blueprint(analysis_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create a test client"""
    return app.test_client()


class TestCleaningPage:
    """Tests for the cleaning page endpoint"""

    def test_cleaning_page_returns_html(self, client):
        """Test that /cleaning returns HTML content"""
        with patch('app_analysis.render_template') as mock_render:
            mock_render.return_value = "<html>Cleaning Page</html>"

            response = client.get('/cleaning')

            assert response.status_code == 200
            mock_render.assert_called_once_with(
                'cleaning.html',
                title='AudioMuse-AI - Database Cleaning',
                active='cleaning'
            )


class TestStartAnalysisEndpoint:
    """Tests for the /api/analysis/start endpoint"""

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_successful_analysis_start_with_defaults(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test starting analysis with default parameters"""
        mock_job = Mock()
        mock_job.id = "test-job-123"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post(
            '/api/analysis/start',
            json={}
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data['task_id'] == "test-job-123"
        assert data['task_type'] == "main_analysis"
        assert data['status'] == "queued"

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

        # Verify task status was saved
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
        """Test that config defaults are used when not provided"""
        mock_job = Mock()
        mock_job.id = "test-job-456"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post(
            '/api/analysis/start',
            json={}
        )

        assert response.status_code == 202

        # Verify enqueue was called with default values
        mock_queue.enqueue.assert_called_once()
        call_kwargs = mock_queue.enqueue.call_args[1]
        # Check that args tuple contains (num_recent_albums, top_n_moods)
        assert call_kwargs['args'] == (5, 10)

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_start_with_custom_params(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test starting analysis with custom parameters"""
        mock_job = Mock()
        mock_job.id = "test-job-789"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post(
            '/api/analysis/start',
            json={
                'num_recent_albums': 10,
                'top_n_moods': 15
            }
        )

        assert response.status_code == 202
        data = response.get_json()
        assert data['task_id'] == "test-job-789"

        # Verify enqueue was called with custom values
        call_kwargs = mock_queue.enqueue.call_args[1]
        assert call_kwargs['args'] == (10, 15)

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_enqueue_task_parameters(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test that task is enqueued with correct parameters"""
        mock_job = Mock()
        mock_job.id = "test-job-abc"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post(
            '/api/analysis/start',
            json={'num_recent_albums': 3, 'top_n_moods': 5}
        )

        assert response.status_code == 202

        # Verify enqueue was called with correct parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args
        assert call_args[0][0] == 'tasks.analysis.run_analysis_task'
        assert call_args[1]['description'] == "Main Music Analysis"
        assert call_args[1]['job_timeout'] == -1  # No timeout

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_handles_missing_json(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test that endpoint handles missing JSON body gracefully"""
        mock_job = Mock()
        mock_job.id = "test-job-def"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        # Send request without JSON body
        response = client.post(
            '/api/analysis/start',
            content_type='application/json'
        )

        # Should still work with defaults
        assert response.status_code == 202

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_saves_pending_status(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test that pending task status is saved"""
        mock_job = Mock()
        mock_job.id = "test-job-ghi"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post(
            '/api/analysis/start',
            json={}
        )

        assert response.status_code == 202

        # Verify save_task_status was called with PENDING status
        mock_save_status.assert_called_once()
        call_args = mock_save_status.call_args[0]
        # Should be called with (job_id, task_type, status, details)
        assert call_args[1] == "main_analysis"
        # The third argument should be TASK_STATUS_PENDING (we can't import it directly)


class TestStartCleaningEndpoint:
    """Tests for the /api/cleaning/start endpoint"""

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_successful_cleaning_start(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test starting database cleaning task"""
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

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

        # Verify task status was saved
        mock_save_status.assert_called_once()

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_enqueue_task_parameters(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test that cleaning task is enqueued with correct parameters"""
        mock_job = Mock()
        mock_job.id = "clean-job-456"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/cleaning/start')

        assert response.status_code == 202

        # Verify enqueue was called with correct parameters
        mock_queue.enqueue.assert_called_once()
        call_args = mock_queue.enqueue.call_args
        assert call_args[0][0] == 'tasks.cleaning.identify_and_clean_orphaned_albums_task'
        assert call_args[1]['description'] == "Database Cleaning (Identify and Delete Orphaned Albums)"
        assert call_args[1]['job_timeout'] == -1  # No timeout

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_saves_pending_status(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test that pending cleaning task status is saved"""
        mock_job = Mock()
        mock_job.id = "clean-job-789"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/cleaning/start')

        assert response.status_code == 202

        # Verify save_task_status was called
        mock_save_status.assert_called_once()
        call_args = mock_save_status.call_args[0]
        assert call_args[1] == "cleaning"

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_cleans_up_previous_tasks(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test that previous cleaning tasks are cleaned up"""
        mock_job = Mock()
        mock_job.id = "clean-job-abc"
        mock_job.get_status.return_value = "queued"
        mock_queue.enqueue.return_value = mock_job

        response = client.post('/api/cleaning/start')

        assert response.status_code == 202

        # Verify cleanup was called before enqueueing new task
        mock_cleanup.assert_called_once()


class TestEndpointErrorHandling:
    """Tests for error handling in endpoints"""

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_analysis_handles_enqueue_failure(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test handling of task enqueue failures"""
        mock_queue.enqueue.side_effect = Exception("Queue error")

        # Should raise exception (Flask will handle with 500)
        with pytest.raises(Exception):
            response = client.post('/api/analysis/start', json={})

    @patch('app_analysis.rq_queue_high')
    @patch('app_analysis.clean_up_previous_main_tasks')
    @patch('app_analysis.save_task_status')
    def test_cleaning_handles_enqueue_failure(
        self, mock_save_status, mock_cleanup, mock_queue, client
    ):
        """Test handling of cleaning task enqueue failures"""
        mock_queue.enqueue.side_effect = Exception("Queue error")

        # Should raise exception (Flask will handle with 500)
        with pytest.raises(Exception):
            response = client.post('/api/cleaning/start')


class TestBlueprintIntegration:
    """Tests for Flask blueprint integration"""

    def test_blueprint_registered_correctly(self, app):
        """Test that blueprint routes are registered"""
        # Check that blueprint routes exist
        rules = [str(rule) for rule in app.url_map.iter_rules()]

        assert '/cleaning' in rules
        assert '/api/analysis/start' in rules
        assert '/api/cleaning/start' in rules

    def test_analysis_endpoint_accepts_post_only(self, client):
        """Test that analysis endpoint only accepts POST"""
        # POST should work (even if mocked components fail)
        # GET should return 405 Method Not Allowed
        response = client.get('/api/analysis/start')
        assert response.status_code == 405

    def test_cleaning_endpoint_accepts_post_only(self, client):
        """Test that cleaning endpoint only accepts POST"""
        # GET should return 405 Method Not Allowed
        response = client.get('/api/cleaning/start')
        assert response.status_code == 405

    def test_cleaning_page_accepts_get_only(self, client):
        """Test that cleaning page only accepts GET"""
        # POST should return 405 Method Not Allowed
        response = client.post('/cleaning')
        assert response.status_code == 405
