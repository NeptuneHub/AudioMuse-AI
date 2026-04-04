# app_analysis.py
from flask import Blueprint, jsonify, request
import uuid
import logging

# Import configuration from the main config.py
from config import NUM_RECENT_ALBUMS, TOP_N_MOODS

# RQ import
from rq import Retry

logger = logging.getLogger(__name__)

# Create a Blueprint for analysis-related routes
analysis_bp = Blueprint('analysis_bp', __name__)

@analysis_bp.route('/cleaning', methods=['GET'])
def cleaning_page():
    """
    Serves the HTML page for the Database Cleaning feature.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the cleaning page.
        content:
          text/html:
            schema:
              type: string
    """
    from flask import render_template
    return render_template('cleaning.html', title = 'AudioMuse-AI - Database Cleaning', active='cleaning')

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
    # Local imports to prevent circular dependency at startup
    from app_helper import rq_queue_high, clean_up_previous_main_tasks, save_task_status, TASK_STATUS_PENDING, get_active_main_task
    from app_setup import get_providers_display as get_providers

    # Clean up stale tasks first, then check for truly active ones.
    # Must run cleanup before the active-task check — otherwise a crashed worker
    # leaves a STARTED/PROGRESS row that blocks new runs forever.
    clean_up_previous_main_tasks()

    active_task = get_active_main_task()
    if active_task:
        return jsonify({
            "error": "An active batch task is already in progress.",
            "task_id": active_task['task_id'],
            "status": active_task['status']
        }), 409

    data = request.json or {}
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))
    logger.info(f"Starting analysis request: num_recent_albums={num_recent_albums}, top_n_moods={top_n_moods}")

    # Pre-flight validation: block analysis in multi-provider setups with path issues
    try:
        providers = get_providers(enabled_only=True)
        is_multi_provider = len(providers) > 1

        if is_multi_provider:
            path_issues = []
            for p in providers:
                cfg = p.get('config_display') or {}
                pname = p.get('name') or p.get('provider_type')
                ptype = p.get('provider_type')
                path_format = cfg.get('path_format', '')

                if ptype == 'navidrome' and path_format == 'relative':
                    path_issues.append({
                        'provider_id': p['id'],
                        'provider_name': pname,
                        'provider_type': ptype,
                        'issue': 'relative_paths',
                        'message': f'{pname} is reporting virtual file paths instead of real filesystem paths.',
                        'instructions': [
                            'In Navidrome, go to Players in the right sidebar',
                            'Click on the AudioMuse-AI player entry',
                            'Toggle "Report Real Path" to enabled',
                            'Back in AudioMuse-AI Settings, click "Rescan Paths" on this provider'
                        ]
                    })
                elif ptype == 'navidrome' and not path_format:
                    path_issues.append({
                        'provider_id': p['id'],
                        'provider_name': pname,
                        'provider_type': ptype,
                        'issue': 'unknown_path_format',
                        'message': f'{pname} has no path format detected yet.',
                        'instructions': [
                            'Go to Settings and click "Rescan Paths" on this provider',
                            'This will detect whether real file paths are being reported'
                        ]
                    })
                elif path_format == 'relative':
                    path_issues.append({
                        'provider_id': p['id'],
                        'provider_name': pname,
                        'provider_type': ptype,
                        'issue': 'relative_paths',
                        'message': f'{pname} is reporting relative file paths. Cross-provider matching will fail.',
                        'instructions': [
                            'Check provider path configuration to enable real filesystem paths',
                            'Go to Settings and click "Rescan Paths" after fixing'
                        ]
                    })

            if path_issues:
                logger.warning(f"⚠️ Analysis BLOCKED: {len(path_issues)} provider path issue(s) in multi-provider setup.")
                for issue in path_issues:
                    logger.warning(
                        f"  [{issue['provider_name']}] {issue['message']} "
                        f"Fix: {' → '.join(issue.get('instructions', []))}"
                    )
                return jsonify({
                    'error': 'provider_path_issue',
                    'blocked': True,
                    'message': 'Analysis blocked: provider path issues detected in multi-provider setup. '
                               'Fix these issues to prevent duplicate tracks.',
                    'issues': path_issues,
                    'action_url': '/settings'
                }), 409
    except Exception as e:
        logger.warning(f"Pre-flight provider validation failed (non-blocking): {e}")

    job_id = str(uuid.uuid4())

    save_task_status(job_id, "main_analysis", TASK_STATUS_PENDING, details={"message": "Task enqueued."})

    # Enqueue task using a string path to its function.
    # MODIFIED: The arguments passed to the task are updated to match the new function signature.
    job = rq_queue_high.enqueue(
        'tasks.analysis.run_analysis_task',
        args=(num_recent_albums, top_n_moods),
        job_id=job_id,
        description="Main Music Analysis",
        retry=Retry(max=3),
        job_timeout=-1 # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "main_analysis", "status": job.get_status()}), 202

@analysis_bp.route('/api/cleaning/start', methods=['POST'])
def start_cleaning_endpoint():
    """
    Identify and automatically clean orphaned albums from the database.
    This endpoint enqueues a cleaning task that both identifies and deletes orphaned albums.
    ---
    tags:
      - Cleaning
    responses:
      202:
        description: Database cleaning task successfully enqueued.
        content:
          application/json:
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: The ID of the enqueued database cleaning task.
                task_type:
                  type: string
                  description: Type of the task (cleaning).
                  example: cleaning
                status:
                  type: string
                  description: The initial status of the job in the queue (e.g., queued).
      500:
        description: Server error during task enqueue.
    """
    # Local imports to prevent circular dependency at startup
    from app_helper import rq_queue_high, clean_up_previous_main_tasks, save_task_status, TASK_STATUS_PENDING, get_active_main_task

    # Clean up stale tasks first, then check for truly active ones.
    clean_up_previous_main_tasks()

    active_task = get_active_main_task()
    if active_task:
        return jsonify({
            "error": "An active batch task is already in progress.",
            "task_id": active_task['task_id'],
            "status": active_task['status']
        }), 409

    job_id = str(uuid.uuid4())
    save_task_status(job_id, "cleaning", TASK_STATUS_PENDING, details={"message": "Database cleaning task enqueued."})

    # Enqueue combined cleaning task
    job = rq_queue_high.enqueue(
        'tasks.cleaning.identify_and_clean_orphaned_albums_task',
        job_id=job_id,
        description="Database Cleaning (Identify and Delete Orphaned Albums)",
        retry=Retry(max=2),
        job_timeout=-1 # No timeout
    )
    return jsonify({"task_id": job.id, "task_type": "cleaning", "status": job.get_status()}), 202
