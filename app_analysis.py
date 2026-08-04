# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for launching library analysis and database cleaning.

Thin route layer that enqueues the long-running main tasks onto the high
priority RQ queue and returns their job id for the UI to poll via the generic
status routes in `app.py`.

Main Features:
* Routes: `/cleaning` page, `/api/analysis/start` (enqueues
  `tasks.analysis.run_analysis_task`) and `/api/cleaning/start`.
* Archives previously successful main tasks to REVOKED on a new start and
  guards against a second concurrent main task via `get_active_main_task`.
"""

from flask import Blueprint, jsonify, request, render_template
import uuid
import logging

# Import configuration from the main config.py
from config import (
    NUM_RECENT_ALBUMS,
    TOP_N_MOODS,
    TASK_STATUS_PENDING,
    TASK_STATUS_FAILURE,
    CLEANING_CATALOGUE,
)

# RQ import
from rq import Retry

# App helper functions
from app_helper import (
    ENQUEUE_MISSING,
    resolve_enqueue_outcome,
    rq_queue_high,
    save_task_status,
)
from database import (
    clean_up_previous_main_tasks,
    get_active_main_task,
    main_task_start_lock,
    prune_task_status_history,
)

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
    return render_template(
        'cleaning.html', title='AudioMuse-AI - Database Cleaning', active='cleaning',
        cleaning_catalogue_default=CLEANING_CATALOGUE,
    )


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
    data = request.json or {}
    # MODIFIED: Removed jellyfin_url, jellyfin_user_id, and jellyfin_token as they are no longer passed to the task.
    # The task now gets these details from the central config.
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))
    logger.info(
        f"Starting analysis request: num_recent_albums={num_recent_albums}, top_n_moods={top_n_moods}"
    )

    job_id = str(uuid.uuid4())

    # The gate, the archival and the claim are one atomic act. Checked separately,
    # two starts (a double click, or a cron tick landing on a manual start) could
    # both see "nothing running" before either had written its row, and then both
    # launch - or one archival could revoke the row the other had just created.
    with main_task_start_lock():
        # Check for any existing active main task to prevent parallel batch runs.
        active_task = get_active_main_task()
        if active_task:
            return jsonify(
                {
                    "error": "An active batch task is already in progress.",
                    "task_id": active_task['task_id'],
                    "status": active_task['status'],
                }
            ), 409

        # Clean up details of previously successful or stale tasks before starting a new one
        clean_up_previous_main_tasks()
        save_task_status(
            job_id, "main_analysis", TASK_STATUS_PENDING,
            details={"message": "Task enqueued."}, raise_on_error=True,
        )

        # Keep the lock through the definitive enqueue result. Otherwise Cancel can
        # wipe the committed claim in the small window before Redis sees the job.
        try:
            job = rq_queue_high.enqueue(
                'tasks.analysis.run_analysis_task',
                args=(num_recent_albums, top_n_moods),
                job_id=job_id,
                description="Main Music Analysis",
                retry=Retry(max=3),
                job_timeout=-1,  # No timeout
            )
        except Exception:
            logger.exception("Could not enqueue the analysis task")
            outcome, rq_status = resolve_enqueue_outcome(job_id)
            if outcome == ENQUEUE_MISSING:
                save_task_status(
                    job_id, "main_analysis", TASK_STATUS_FAILURE,
                    details={"error": "Could not enqueue the task (is Redis reachable?)"},
                    raise_on_error=True,
                )
                prune_task_status_history()
                return jsonify({"error": "Could not enqueue the analysis. Check the logs."}), 500
            # Redis may have accepted the transaction and only lost the reply. Keep
            # the PENDING claim so another start cannot duplicate it; the janitor
            # resolves an actually-missing job once Redis is reachable again.
            return jsonify(
                {
                    "task_id": job_id,
                    "task_type": "main_analysis",
                    "status": rq_status or TASK_STATUS_PENDING,
                }
            ), 202
    # Queue.enqueue returning is the definitive acceptance point. A second Redis
    # read here can lose its reply and turn an accepted job into an HTTP 500.
    return jsonify(
        {"task_id": job.id, "task_type": "main_analysis", "status": "queued"}
    ), 202


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
    # Cleaning is the ONE start that must also refuse while a sweep is running: both
    # prune track_server_map, each against a snapshot of the server's catalogue taken
    # minutes earlier, so an overlap lets cleaning delete the mappings the sweep just
    # wrote. Every other task type may run alongside a sweep, so they keep the
    # default exclusion.
    # Per-run opt-in: when the cleaning page's checkbox is ticked (or CLEANING_CATALOGUE
    # is the env default) the task also DELETES catalogue rows bound to no server;
    # otherwise it only unbinds each server's stale mappings.
    data = request.get_json(silent=True) or {}
    clean_catalogue = bool(data.get('clean_catalogue', CLEANING_CATALOGUE))

    job_id = str(uuid.uuid4())

    with main_task_start_lock():
        active_task = get_active_main_task(exclude_task_types=())
        if active_task:
            return jsonify(
                {
                    "error": "An active batch task is already in progress.",
                    "task_id": active_task['task_id'],
                    "status": active_task['status'],
                }
            ), 409

        # Clean up any previous cleaning tasks
        clean_up_previous_main_tasks()
        save_task_status(
            job_id,
            "cleaning",
            TASK_STATUS_PENDING,
            details={"message": "Database cleaning task enqueued."},
            raise_on_error=True,
        )

        # Enqueue while still serialized with global Cancel.
        try:
            job = rq_queue_high.enqueue(
                'tasks.cleaning.identify_and_clean_orphaned_albums_task',
                clean_catalogue,
                job_id=job_id,
                description="Database Cleaning (Identify and Delete Orphaned Albums)",
                retry=Retry(max=2),
                job_timeout=-1,  # No timeout
            )
        except Exception:
            logger.exception("Could not enqueue the cleaning task")
            outcome, rq_status = resolve_enqueue_outcome(job_id)
            if outcome == ENQUEUE_MISSING:
                save_task_status(
                    job_id, "cleaning", TASK_STATUS_FAILURE,
                    details={"error": "Could not enqueue the task (is Redis reachable?)"},
                    raise_on_error=True,
                )
                prune_task_status_history()
                return jsonify({"error": "Could not enqueue the cleaning. Check the logs."}), 500
            return jsonify(
                {
                    "task_id": job_id,
                    "task_type": "cleaning",
                    "status": rq_status or TASK_STATUS_PENDING,
                }
            ), 202
    return jsonify(
        {"task_id": job.id, "task_type": "cleaning", "status": "queued"}
    ), 202
