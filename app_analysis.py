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

from flask import Blueprint, request, render_template
import uuid
import logging

# Import configuration from the main config.py
from config import (
    NUM_RECENT_ALBUMS,
    TOP_N_MOODS,
    CLEANING_CATALOGUE,
)

# Task queue
import taskqueue

# App helper functions
import database

from app_helper import admit_and_enqueue_main_task

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
    # Gate, archive and enqueue are one session-scoped critical section (see
    # app_helper.admit_and_enqueue_main_task): the archive REVOKES every live
    # root, so it must come after the gate, and the lock must span the archive's
    # internal commit so this start cannot retire a task another caller enqueued
    # between our gate read and our archive.
    return admit_and_enqueue_main_task(
        job_id=job_id,
        task_type="main_analysis",
        busy_label='analysis',
        error_message="Could not queue the analysis. Check the logs.",
        enqueue=lambda: taskqueue.enqueue(
            'tasks.analysis.run_analysis_task',
            args=(num_recent_albums, top_n_moods),
            task_id=job_id,
            task_type="main_analysis",
            queue=taskqueue.QUEUE_HIGH,
            details={"message": "Task queued."},
        ),
    )


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
    return admit_and_enqueue_main_task(
        job_id=job_id,
        task_type="cleaning",
        busy_label='cleaning',
        error_message="Could not queue the cleaning. Check the logs.",
        blocking_gate=lambda: (
            database.get_queue_blocking_task()
            or database.get_active_main_task(task_type='server_sweep')
        ),
        race_read=lambda: database.get_active_main_task(
            exclude_task_types=database.NON_BLOCKING_TASK_TYPES
        ),
        enqueue=lambda: taskqueue.enqueue(
            'tasks.cleaning.identify_and_clean_orphaned_albums_task',
            args=(clean_catalogue,),
            task_id=job_id,
            task_type="cleaning",
            queue=taskqueue.QUEUE_HIGH,
            details={"message": "Database cleaning task queued."},
        ),
    )
