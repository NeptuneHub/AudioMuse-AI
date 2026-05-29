# app_collection.py
from flask import Blueprint, jsonify, request, render_template
import uuid
import logging
import json
import time
import traceback

from rq import Retry
from psycopg2.extras import DictCursor

from error import error_manager
from error.error_dictionary import ERR_COLLECTION_SYNC_FAILED


logger = logging.getLogger(__name__)

collection_bp = Blueprint('collection_bp', __name__)

@collection_bp.route('/collection')
def collection_page():
    """
    Collection Sync admin page.
    ---
    tags:
      - Collection
    summary: HTML page for synchronizing local song data with a remote PocketBase collection.
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template('collection.html', title = 'AudioMuse-AI - Collection Sync', active='collection')

def collection_task_failure_handler(job, connection, type, value, tb):
    """A failure handler for the main collection sync task, executed by the worker."""
    from app import app
    from app_helper import save_task_status, TASK_STATUS_FAILURE
    with app.app_context():
        task_id = getattr(job, 'id', None) or getattr(job, 'get_id', lambda: None)()
        tb_formatted = "".join(
            tb.format() if isinstance(tb, traceback.StackSummary)
            else traceback.format_exception(type, value, tb)
        )
        error_details = {
            "message": "Task failed permanently after all retries.",
            "error": error_manager.build(ERR_COLLECTION_SYNC_FAILED, str(value)),
            "error_type": str(type.__name__),
            "error_value": str(value),
        }
        save_task_status(
            task_id,
            "main_collection_sync",
            TASK_STATUS_FAILURE,
            progress=100,
            details=error_details
        )
        app.logger.error(f"Main collection sync task {task_id} failed permanently. DB status updated.\n{tb_formatted}")

@collection_bp.route('/api/collection/start', methods=['POST'])
def start_collection_sync():
    """
    Start a collection synchronization task.
    ---
    tags:
      - Collection
    summary: Synchronize local song metadata with a remote PocketBase collection.
    description: |
      Cleans up previously successful or stale main collection sync tasks,
      then enqueues a new `tasks.collection_manager.sync_collections_task`
      job on the high-priority RQ queue. The job is bounded by a 2-hour
      timeout and retried up to twice on failure.
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [url, token, num_albums]
            properties:
              url:
                type: string
                description: PocketBase base URL.
              token:
                type: string
                description: PocketBase authentication token.
              num_albums:
                type: integer
                description: Number of most recent albums to sync.
    responses:
      202:
        description: Sync task enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                task_type:
                  type: string
                  example: main_collection_sync
                status:
                  type: string
      400:
        description: Missing required parameters.
    """
    # Local import to avoid circular dependency
    from app_helper import rq_queue_high
    from app_helper import save_task_status, TASK_STATUS_PENDING, clean_up_previous_main_tasks

    data = request.json
    # MODIFIED: Expect 'token' instead of 'email' and 'password'
    if not data or not all(k in data for k in ['url', 'token', 'num_albums']):
        return jsonify({"message": "Missing required parameters: url, token, num_albums"}), 400
    
    # Clean up previously successful or stale sync tasks before starting a new one
    clean_up_previous_main_tasks()

    pocketbase_url = data['url']
    pocketbase_token = data['token'] # MODIFIED
    num_last_albums = int(data['num_albums'])
    
    job_id = str(uuid.uuid4())
    
    # Save the initial "PENDING" state for the main task
    save_task_status(
        job_id,
        "main_collection_sync",
        TASK_STATUS_PENDING,
        details={"message": "Synchronization task has been enqueued."}
    )

    # Enqueue the main parent task to the high priority queue
    # MODIFIED: Pass the token to the task
    job = rq_queue_high.enqueue(
        'tasks.collection_manager.sync_collections_task',
        args=(pocketbase_url, pocketbase_token, num_last_albums),
        job_id=job_id,
        description="Main Collection Synchronization",
        retry=Retry(max=2),
        job_timeout='2h', # Set a reasonable timeout for the parent task
        on_failure=collection_task_failure_handler
    )

    return jsonify({
        "task_id": job.id,
        "task_type": "main_collection_sync",
        "status": job.get_status()
    }), 202

@collection_bp.route('/api/collection/last_task', methods=['GET'])
def get_last_collection_task():
    """
    Status of the most recent collection sync task.
    ---
    tags:
      - Collection
    summary: Return the most recently started main_collection_sync task and its progress.
    responses:
      200:
        description: |
          Either the most recent task row or `{"status": "NO_PREVIOUS_TASK"}`
          when no sync has ever been run.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                task_type:
                  type: string
                status:
                  type: string
                progress:
                  type: number
                details:
                  type: object
                running_time_seconds:
                  type: number
                  format: float
    """
    from app_helper import get_db # Local import to use the app context's db connection
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute("""
        SELECT task_id, task_type, status, progress, details, start_time, end_time
        FROM task_status 
        WHERE task_type = 'main_collection_sync'
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    last_task_row = cur.fetchone()
    cur.close()

    if last_task_row:
        last_task_data = dict(last_task_row)
        
        # Safely parse details
        if last_task_data.get('details'):
            try:
                details_val = last_task_data['details']
                last_task_data['details'] = json.loads(details_val) if isinstance(details_val, str) else details_val
            except (json.JSONDecodeError, TypeError):
                 last_task_data['details'] = {"error": "Could not parse details."}

        # Calculate running time
        start_time = last_task_data.get('start_time')
        end_time = last_task_data.get('end_time')
        if start_time:
            effective_end_time = end_time if end_time is not None else time.time()
            last_task_data['running_time_seconds'] = max(0, effective_end_time - start_time)
        else:
            last_task_data['running_time_seconds'] = 0.0
        
        last_task_data.pop('start_time', None)
        last_task_data.pop('end_time', None)

        return jsonify(last_task_data), 200
        
    return jsonify({"status": "NO_PREVIOUS_TASK"}), 200
