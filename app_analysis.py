# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for launching library analysis and database cleaning.

Thin route layer that enqueues the long-running main tasks onto the high
priority task queue and returns their job id for the UI to poll via the generic
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
    TASK_STATUS_NEW,
    CLEANING_CATALOGUE,
)

# Task queue
import taskqueue

# App helper functions
from database import (
    NON_BLOCKING_TASK_TYPES,
    clean_up_previous_main_tasks,
    get_active_main_task,
    main_task_start_lock,
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
    data = request.get_json(silent=True) or {}
    # MODIFIED: Removed jellyfin_url, jellyfin_user_id, and jellyfin_token as they are no longer passed to the task.
    # The task now gets these details from the central config.
    num_recent_albums = int(data.get('num_recent_albums', NUM_RECENT_ALBUMS))
    top_n_moods = int(data.get('top_n_moods', TOP_N_MOODS))
    logger.info(
        f"Starting analysis request: num_recent_albums={num_recent_albums}, top_n_moods={top_n_moods}"
    )

    job_id = str(uuid.uuid4())

    # The gate and the claim are one INSERT. A partial unique index allows exactly
    # one live main task, so a double click or a cron tick landing on a manual
    # start loses the race in Postgres rather than in application code, and the
    # advisory lock that used to serialize check-then-act is gone with it.
    # The gate MUST come before the archive. clean_up_previous_main_tasks
    # REVOKES any live root, so archiving first cancelled the task that was
    # running and then let this one straight through - the unique index could
    # never fire because the row it would have collided with was just retired.
    # Session-scoped and held across the archive's internal commit: the gate,
    # clean_up_previous_main_tasks and the enqueue must be one critical section,
    # or this start's archive can REVOKE a main task another caller (a cron
    # tick, a second tab) enqueued between our gate read and our archive. The
    # unique index only prevents two LIVE rows; it cannot stop an archive from
    # retiring a row that was legitimately accepted first.
    with main_task_start_lock():
        active_task = get_active_main_task()
        if active_task:
            return jsonify(
                {
                    "error": "An active batch task is already in progress.",
                    "task_id": active_task['task_id'],
                    "status": active_task['status'],
                }
            ), 409

        clean_up_previous_main_tasks()
        try:
            taskqueue.enqueue(
                'tasks.analysis.run_analysis_task',
                args=(num_recent_albums, top_n_moods),
                task_id=job_id,
                task_type="main_analysis",
                queue=taskqueue.QUEUE_HIGH,
                details={"message": "Task queued."},
            )
        except taskqueue.TaskAlreadyRunning as exc:
            active_task = get_active_main_task()
            return jsonify(
                {
                    "error": exc.user_message,
                    "task_id": active_task['task_id'] if active_task else None,
                    "status": active_task['status'] if active_task else None,
                }
            ), exc.status_code
        except Exception:
            logger.exception("Could not queue the analysis task")
            return jsonify({"error": "Could not queue the analysis. Check the logs."}), 500
    return jsonify(
        {"task_id": job_id, "task_type": "main_analysis", "status": TASK_STATUS_NEW}
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

    # Cleaning is the one start that also refuses while a SWEEP runs, because the
    # two write the same mappings; that stricter check stays a read, while the
    # unique index enforces the one-live-main-task rule itself.
    # Same critical section as the analysis start: gate, archive and enqueue
    # under the session start lock, so this start's archive cannot REVOKE a main
    # task another caller enqueued between our gate read and our archive.
    with main_task_start_lock():
        active_task = get_active_main_task(exclude_task_types=NON_BLOCKING_TASK_TYPES)
        if active_task:
            return jsonify(
                {
                    "error": "An active batch task is already in progress.",
                    "task_id": active_task['task_id'],
                    "status": active_task['status'],
                }
            ), 409

        clean_up_previous_main_tasks()
        try:
            taskqueue.enqueue(
                'tasks.cleaning.identify_and_clean_orphaned_albums_task',
                args=(clean_catalogue,),
                task_id=job_id,
                task_type="cleaning",
                queue=taskqueue.QUEUE_HIGH,
                details={"message": "Database cleaning task queued."},
            )
        except taskqueue.TaskAlreadyRunning as exc:
            # The INSERT lost the admission race and its savepoint rolled back, so
            # ``job_id`` names a row that was never written. Re-read the task that
            # actually holds the gate - the same contract analysis and clustering use -
            # so an API consumer polling the returned id does not get a 404.
            active_task = get_active_main_task(exclude_task_types=NON_BLOCKING_TASK_TYPES)
            return jsonify(
                {
                    "error": exc.user_message,
                    "task_id": active_task['task_id'] if active_task else None,
                    "status": active_task['status'] if active_task else None,
                }
            ), exc.status_code
        except Exception:
            logger.exception("Could not queue the cleaning task")
            return jsonify({"error": "Could not queue the cleaning. Check the logs."}), 500
    return jsonify(
        {"task_id": job_id, "task_type": "cleaning", "status": TASK_STATUS_NEW}
    ), 202
