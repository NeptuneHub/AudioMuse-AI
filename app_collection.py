# app_collection.py
from flask import Blueprint, jsonify, request, render_template, g
import uuid
import logging
import json
import time
import traceback

from rq import Retry
from psycopg2.extras import DictCursor


logger = logging.getLogger(__name__)

collection_bp = Blueprint('collection_bp', __name__)

@collection_bp.route('/collection')
def collection_page():
    """Serves the HTML page for the Collection Sync feature."""
    return render_template('collection.html', title = 'AudioMuse-AI - Collection Sync', active='collection')

def collection_task_failure_handler(job, connection, type, value, tb):
    """A failure handler for the main collection sync task, executed by the worker."""
    from app import app
    from app_helper import save_task_status, TASK_STATUS_FAILURE
    with app.app_context():
        task_id = job.get_id()
        error_details = {
            "message": "Task failed permanently after all retries.",
            "error_type": str(type.__name__),
            "error_value": str(value),
            # --- FIX: Handle different traceback types, especially from rq-janitor ---
            "traceback": "".join(
                tb.format() if isinstance(tb, traceback.StackSummary)
                else traceback.format_exception(type, value, tb)
            )
        }
        save_task_status(
            task_id,
            "main_collection_sync",
            TASK_STATUS_FAILURE,
            progress=100,
            details=error_details
        )
        app.logger.error(f"Main collection sync task {task_id} failed permanently. DB status updated.")

@collection_bp.route('/api/collection/start', methods=['POST'])
def start_collection_sync():
    """
    Starts the process of synchronizing local song data with a remote PocketBase collection.
    This enqueues the main parent task for the synchronization using an auth token.
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
    Get the status of the most recent collection sync task.
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
