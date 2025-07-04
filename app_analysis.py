# app_analysis.py
from flask import Blueprint, jsonify, request
import uuid
import logging

# Import configuration from the main config.py
from config import JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, NUM_RECENT_ALBUMS, TOP_N_MOODS

# RQ import
from rq import Retry

# Imports from app.py
from app import clean_successful_task_details_on_new_start, save_task_status, TASK_STATUS_PENDING, rq_queue_high

logger = logging.getLogger(__name__)

# Create a Blueprint for analysis-related routes
analysis_bp = Blueprint('analysis_bp', __name__)

@analysis_bp.route('/api/analysis/start', methods=['POST'])
def start_analysis_endpoint():
    """
    Start the music analysis process for recent albums.
    This endpoint enqueues a main analysis task.
    Note: Starting a new analysis task will archive previously successful tasks by setting their status to REVOKED.
    ---
    tags:
      - Analysis
    requestBody:
      description: Configuration for the analysis task.
      required: false
      content:
        application/json:
          schema:
            type: object
            properties:
              jellyfin_url:
                type: string
                description: URL of the Jellyfin server.
                default: "Configured JELLYFIN_URL"
              jellyfin_user_id:
                type: string
                description: Jellyfin User ID.
                default: "Configured JELLYFIN_USER_ID"
              jellyfin_token:
                type: string
                description: Jellyfin API Token.
                default: "Configured JELLYFIN_TOKEN"
              num_recent_albums:
                type: integer
                description: Number of recent albums to process.
                default: "Configured NUM_RECENT_ALBUMS"
              top_n_moods:
                type: integer
                description: Number of top moods to extract per track.
                default: "Configured TOP_N_MOODS"
    responses:
      202:
        description: Analysis task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued main analysis task.
                task_type:
                  type: string
                  description: Type of the task (e.g., main_analysis).
                  example: main_analysis
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
      400:
        description: Invalid input.
      500:
        description: Server error during task enqueue.
    """
    data = request.json or {}
    jellyfin_url = data.get('jellyfin_url', JELLYFIN_URL)
    jellyfin_user_id = data.get('jellyfin_user_id', JELLYFIN_USER_ID)
    jellyfin_token = data.get('jellyfin_token', JELLYFIN_TOKEN)
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))

    job_id = str(uuid.uuid4())

    # Clean up details of previously successful tasks before starting a new one
    clean_successful_task_details_on_new_start()
    save_task_status(job_id, "main_analysis", TASK_STATUS_PENDING, details={"message": "Task enqueued."})

    # Enqueue task using a string path to its function.
    job = rq_queue_high.enqueue(
        'tasks.analysis.run_analysis_task',
        args=(jellyfin_url, jellyfin_user_id, jellyfin_token, num_recent_albums, top_n_moods),
        job_id=job_id,
        description="Main Music Analysis",
        retry=Retry(max=3),
        job_timeout=-1 # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "main_analysis", "status": job.get_status()}), 202
