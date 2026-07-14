# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for managing and running cron-scheduled tasks.

Serves the `/cron` UI and CRUD over the `cron` table, plus the tick function
that reads enabled rows and runs the matching task (analysis, clustering,
sonic fingerprint, or alchemy radio) when its cron expression matches now.

Main Features:
* Routes: `/cron` page and `/api/cron` (GET list, POST create/update), rejecting a
  cron expression that could never fire before it is stored as enabled.
* Cron evaluation that ENQUEUES every task type (analysis, clustering, sonic
  fingerprint, alchemy radio, plugin tasks): nothing runs inline on the poll
  thread, so a slow media server cannot swallow a scheduling window.
* Each row is claimed atomically for its wall-clock minute, so a restart or a
  second web process cannot double-fire it.
"""

from flask import Blueprint, render_template, jsonify, request
from psycopg2.extras import DictCursor, Json
from database import get_db, save_task_status, get_active_main_task
from taskqueue import rq_queue_high, rq_queue_default
from config import TASK_STATUS_PENDING, TASK_STATUS_FAILURE
import uuid
import time
import logging
from config import (
    TOP_N_MOODS,
    CLUSTER_ALGORITHM,
    NUM_CLUSTERS_MIN,
    NUM_CLUSTERS_MAX,
    DBSCAN_EPS_MIN,
    DBSCAN_EPS_MAX,
    DBSCAN_MIN_SAMPLES_MIN,
    DBSCAN_MIN_SAMPLES_MAX,
    GMM_N_COMPONENTS_MIN,
    GMM_N_COMPONENTS_MAX,
    SPECTRAL_N_CLUSTERS_MIN,
    SPECTRAL_N_CLUSTERS_MAX,
    PCA_COMPONENTS_MIN,
    PCA_COMPONENTS_MAX,
    CLUSTERING_RUNS,
    MAX_SONGS_PER_CLUSTER,
    TOP_N_PLAYLISTS,
    MIN_SONGS_PER_GENRE_FOR_STRATIFICATION,
    STRATIFIED_SAMPLING_TARGET_PERCENTILE,
    SCORE_WEIGHT_DIVERSITY,
    SCORE_WEIGHT_SILHOUETTE,
    SCORE_WEIGHT_DAVIES_BOULDIN,
    SCORE_WEIGHT_CALINSKI_HARABASZ,
    SCORE_WEIGHT_PURITY,
    SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY,
    SCORE_WEIGHT_OTHER_FEATURE_PURITY,
    AI_MODEL_PROVIDER,
    OLLAMA_SERVER_URL,
    OLLAMA_MODEL_NAME,
    OPENAI_SERVER_URL,
    OPENAI_MODEL_NAME,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    MISTRAL_API_KEY,
    MISTRAL_MODEL_NAME,
    ENABLE_CLUSTERING_EMBEDDINGS,
)

cron_bp = Blueprint('cron_bp', __name__)

logger = logging.getLogger(__name__)

_ENQUEUED_BY_CRON = "Enqueued by cron."


@cron_bp.route('/cron')
def cron_page():
    """
    Scheduled tasks admin page.
    ---
    tags:
      - Cron
    summary: HTML page for managing cron-scheduled tasks (analysis, clustering, sonic fingerprint).
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template('cron.html', title='AudioMuse-AI - Scheduled Tasks', active='cron')


@cron_bp.route('/api/cron', methods=['GET'])
def get_cron_entries():
    """
    List all cron entries.
    ---
    tags:
      - Cron
    summary: Return every row from the `cron` table with its current state.
    responses:
      200:
        description: List of cron entries.
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: integer
                  name:
                    type: string
                  task_type:
                    type: string
                    enum: [analysis, clustering, sonic_fingerprint, alchemy_radio]
                  cron_expr:
                    type: string
                    description: 5-field cron expression "min hour day month dow".
                  enabled:
                    type: boolean
                  last_run:
                    type: number
                    description: Unix timestamp of the most recent enqueue, or null.
                  created_at:
                    type: string
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute(
        "SELECT id, name, task_type, cron_expr, enabled, last_run, created_at, options "
        "FROM cron ORDER BY id"
    )
    rows = cur.fetchall()
    cur.close()
    entries = []
    for r in rows:
        entries.append(
            {
                'id': r['id'],
                'name': r['name'],
                'task_type': r['task_type'],
                'cron_expr': r['cron_expr'],
                'enabled': bool(r['enabled']),
                'last_run': r['last_run'],
                'created_at': str(r['created_at']),
                'options': r['options'] if isinstance(r['options'], dict) else {},
            }
        )
    # Remove the special-case append for sonic_fingerprint; now handled by DB init
    return jsonify(entries), 200


@cron_bp.route('/api/cron', methods=['POST'])
def save_cron_entry():
    """
    Create or update a cron entry.
    ---
    tags:
      - Cron
    summary: Insert a new cron row or update an existing one (when `id` is supplied).
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              id:
                type: integer
                description: Omit to create a new row; include to update an existing one.
              name:
                type: string
              task_type:
                type: string
                enum: [analysis, clustering, sonic_fingerprint, alchemy_radio]
              cron_expr:
                type: string
                description: 5-field cron expression "min hour day month dow".
              enabled:
                type: boolean
    responses:
      200:
        description: Saved.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: saved
    """
    data = request.json or {}
    # Expected fields: id (optional), name, task_type, cron_expr, enabled
    options = data.get('options') or {}
    if not isinstance(options, dict):
        return jsonify({'error': "'options' must be a JSON object"}), 400

    # Coerced, never None: cron_expr is NOT NULL, so a POST that omitted it used to
    # 500 rather than answer.
    cron_expr = (data.get('cron_expr') or '').strip()

    # Validate ONLY when the row is being enabled. cron.html re-POSTs all four
    # built-in rows on every save, so rejecting an empty expression outright would
    # 400 a save that merely cleared a disabled box. The matcher fails closed and
    # silent, so an unvalidated bad expression (a very plausible '0 3 * * MON')
    # was stored, displayed as active, and simply never fired.
    if bool(data.get('enabled')):
        problem = _cron_expr_problem(cron_expr)
        if problem:
            return jsonify({'error': problem}), 400

    db = get_db()
    cur = db.cursor()
    if data.get('id'):
        cur.execute(
            "UPDATE cron SET name=%s, task_type=%s, cron_expr=%s, enabled=%s, options=%s WHERE id=%s",
            (
                data.get('name'),
                data.get('task_type'),
                cron_expr,
                bool(data.get('enabled')),
                Json(options),
                data.get('id'),
            ),
        )
    else:
        # No id supplied: update the existing row for this task_type if one exists,
        # otherwise insert. Prevents duplicate rows when the client cache is stale.
        cur.execute(
            "SELECT id FROM cron WHERE task_type=%s ORDER BY id LIMIT 1",
            (data.get('task_type'),),
        )
        existing = cur.fetchone()
        if existing:
            cur.execute(
                "UPDATE cron SET name=%s, task_type=%s, cron_expr=%s, enabled=%s, options=%s WHERE id=%s",
                (
                    data.get('name'),
                    data.get('task_type'),
                    cron_expr,
                    bool(data.get('enabled')),
                    Json(options),
                    existing[0],
                ),
            )
        else:
            cur.execute(
                "INSERT INTO cron (name, task_type, cron_expr, enabled, options) VALUES (%s,%s,%s,%s,%s)",
                (
                    data.get('name'),
                    data.get('task_type'),
                    cron_expr,
                    bool(data.get('enabled')),
                    Json(options),
                ),
            )
    db.commit()
    cur.close()
    return jsonify({'message': 'saved'}), 200


@cron_bp.route('/api/cron/plugin_tasks', methods=['GET'])
def get_plugin_cron_tasks():
    """
    List the schedulable plugin cron tasks.
    ---
    tags:
      - Cron
    summary: Return every cron task registered by an enabled plugin.
    responses:
      200:
        description: List of plugin cron tasks.
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
                properties:
                  task_type:
                    type: string
                    description: The schedulable task type, plugin.<id>.<name>.
                  plugin:
                    type: string
                  task:
                    type: string
    """
    try:
        from plugin.manager import plugin_manager

        return jsonify(plugin_manager.available_cron_tasks()), 200
    except Exception:
        logger.exception("Failed to list plugin cron tasks")
        return jsonify([]), 200


def _field_matches(field_expr, value, field_min=0):
    # very small cron field matcher supporting '*', single number, list (comma), ranges (a-b), and steps (*/N, a-b/N).
    # field_min is the lowest legal value for this field (0 for minute/hour/dow, 1 for day-of-month/month) so '*/N'
    # anchors at the field minimum like standard cron instead of at 0.
    if field_expr.strip() == '*':
        return True
    parts = field_expr.split(',')
    for p in parts:
        p = p.strip()
        if '/' in p:
            base, step_s = p.split('/', 1)
            try:
                step = int(step_s)
                if step <= 0:
                    continue
                if base.strip() == '*':
                    if value >= field_min and (value - field_min) % step == 0:
                        return True
                elif '-' in base:
                    a, b = base.split('-', 1)
                    lo, hi = int(a), int(b)
                    if lo <= value <= hi and (value - lo) % step == 0:
                        return True
                else:
                    start = int(base)
                    if value >= start and (value - start) % step == 0:
                        return True
            except ValueError:
                continue
        elif '-' in p:
            a, b = p.split('-', 1)
            try:
                if int(a) <= value <= int(b):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(p) == value:
                    return True
            except ValueError:
                continue
    return False


_CRON_FIELD_DOMAINS = (
    ('minute', 0, 59),
    ('hour', 0, 23),
    ('day of month', 1, 31),
    ('month', 1, 12),
    ('day of week', 0, 6),
)


def _cron_expr_problem(expr):
    """A human-readable reason ``expr`` can never fire, or None when it is valid.

    Decided by running each field over its real domain through the SAME matcher the
    scheduler uses, so the validator cannot drift from the thing it validates. Names
    like MON or @daily are not supported: int() raises on them and _field_matches
    swallows the error, so such a row would be stored, shown as active, and silently
    never run.
    """
    if not expr or not str(expr).strip():
        return "Enter a cron expression, or disable the schedule."
    parts = str(expr).strip().split()
    if len(parts) != 5:
        return (
            f"A cron expression needs 5 fields (minute hour day month weekday); "
            f"got {len(parts)}."
        )
    for field_expr, (name, low, high) in zip(parts, _CRON_FIELD_DOMAINS):
        if not any(
            _field_matches(field_expr, value, low) for value in range(low, high + 1)
        ):
            return (
                f"The {name} field '{field_expr}' never matches any value "
                f"({low}-{high}). Use numbers, not names."
            )
    return None


def cron_matches_now(expr, ts=None):
    # expr expected as 'min hour day month dow'
    t = time.localtime(ts) if ts is not None else time.localtime()
    parts = expr.strip().split()
    if len(parts) < 5:
        return False
    minute, hour, dom, month, dow = parts[:5]
    if not _field_matches(minute, t.tm_min):
        return False
    if not _field_matches(hour, t.tm_hour):
        return False
    # day of week: in cron 0=Sun..6=Sat, Python tm_wday 0=Mon..6=Sun -> convert
    py_dow = (t.tm_wday + 1) % 7
    # Per cron semantics, when both dom and dow are restricted (not '*'),
    # the job runs if EITHER matches; otherwise both must match.
    dom_restricted = dom.strip() != '*'
    dow_restricted = dow.strip() != '*'
    dom_ok = _field_matches(dom, t.tm_mday, field_min=1)
    dow_ok = _field_matches(dow, py_dow) or (py_dow == 0 and _field_matches(dow, 7))
    if dom_restricted and dow_restricted:
        if not (dom_ok or dow_ok):
            return False
    else:
        if not dom_ok or not dow_ok:
            return False
    if not _field_matches(month, t.tm_mon, field_min=1):
        return False
    return True


def _claim_cron_minute(db, row_id, minute_start):
    """Claim ``row_id`` for the wall-clock minute starting at ``minute_start``.

    True exactly once per row per minute, however many processes or restarts race
    for it: the predicate on last_run makes the claim atomic. The row is marked run
    BEFORE the work is enqueued, so a crash in between drops that occurrence rather
    than duplicating it - the right trade for a batch task.
    """
    cur = db.cursor()
    try:
        cur.execute(
            "UPDATE cron SET last_run = %s "
            "WHERE id = %s AND (last_run IS NULL OR last_run < %s)",
            (minute_start, row_id, minute_start),
        )
        claimed = cur.rowcount == 1
        db.commit()
        return claimed
    finally:
        cur.close()


def run_due_cron_jobs():
    """Enqueue every enabled cron row whose expression matches this minute.

    Every branch enqueues: nothing runs inline on the poll thread, so a slow media
    server can never swallow a scheduling window.
    """
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute(
        "SELECT id, name, task_type, cron_expr, enabled, last_run, options "
        "FROM cron WHERE enabled = true"
    )
    rows = cur.fetchall()
    now_ts = time.time()
    minute_start = now_ts - (now_ts % 60)
    for r in rows:
        try:
            if cron_matches_now(r['cron_expr'], now_ts):
                # Claim the row for THIS wall-clock minute before doing anything.
                # The old guard read last_run and wrote it after enqueuing, with no
                # predicate and a 55s window narrower than the 60s minute it was
                # protecting, so a restart inside a matching minute could double-fire.
                # Claiming must come AFTER the match: claiming every enabled row on
                # every tick would stamp last_run continuously and corrupt the
                # dashboard's Last-run display.
                if not _claim_cron_minute(db, r['id'], minute_start):
                    continue
                task_type = r['task_type']
                # Batch work always covers every configured server, one server at
                # a time. There is no per-schedule scope: a "default server only"
                # schedule left every other server's exclusive songs unanalyzed
                # and without playlists, silently.
                server_scope = 'all'
                job_id = str(uuid.uuid4())
                if task_type in ('analysis', 'clustering'):
                    # The manual endpoints 409 while any main task is live; cron used
                    # to enqueue regardless, so a nightly row could start a second
                    # full run on top of one still in progress.
                    active = get_active_main_task()
                    if active:
                        logger.info(
                            "Cron: skipping %s, main task %s is still %s",
                            task_type, active['task_id'], active['status'],
                        )
                        continue
                if task_type == 'analysis':
                    # mark queued in task_status
                    save_task_status(
                        job_id,
                        f"main_{task_type}",
                        TASK_STATUS_PENDING,
                        details={"message": _ENQUEUED_BY_CRON},
                    )
                    try:
                        rq_queue_high.enqueue(
                            'tasks.analysis.run_analysis_task',
                            args=(0, TOP_N_MOODS),
                            kwargs={'server_scope': server_scope},
                            job_id=job_id,
                            description='Cron Analysis',
                            job_timeout=-1,
                        )
                        logger.info(f"Cron: enqueued analysis job {job_id}")
                    except Exception:
                        logger.exception("Cron: enqueue failed for analysis")
                        save_task_status(
                            job_id, f"main_{task_type}", TASK_STATUS_FAILURE,
                            details={"error": "Could not enqueue the task (is Redis reachable?)"},
                        )
                elif task_type == 'clustering':
                    # mark queued in task_status
                    save_task_status(
                        job_id,
                        f"main_{task_type}",
                        TASK_STATUS_PENDING,
                        details={"message": _ENQUEUED_BY_CRON},
                    )
                    clustering_kwargs = {
                        "clustering_method": CLUSTER_ALGORITHM,
                        "num_clusters_min": int(NUM_CLUSTERS_MIN),
                        "num_clusters_max": int(NUM_CLUSTERS_MAX),
                        "dbscan_eps_min": float(DBSCAN_EPS_MIN),
                        "dbscan_eps_max": float(DBSCAN_EPS_MAX),
                        "dbscan_min_samples_min": int(DBSCAN_MIN_SAMPLES_MIN),
                        "dbscan_min_samples_max": int(DBSCAN_MIN_SAMPLES_MAX),
                        "gmm_n_components_min": int(GMM_N_COMPONENTS_MIN),
                        "gmm_n_components_max": int(GMM_N_COMPONENTS_MAX),
                        "spectral_n_clusters_min": int(SPECTRAL_N_CLUSTERS_MIN),
                        "spectral_n_clusters_max": int(SPECTRAL_N_CLUSTERS_MAX),
                        "pca_components_min": int(PCA_COMPONENTS_MIN),
                        "pca_components_max": int(PCA_COMPONENTS_MAX),
                        "num_clustering_runs": int(CLUSTERING_RUNS),
                        "max_songs_per_cluster_val": int(MAX_SONGS_PER_CLUSTER),
                        "top_n_playlists_param": int(TOP_N_PLAYLISTS),
                        "min_songs_per_genre_for_stratification_param": int(
                            MIN_SONGS_PER_GENRE_FOR_STRATIFICATION
                        ),
                        "stratified_sampling_target_percentile_param": int(
                            STRATIFIED_SAMPLING_TARGET_PERCENTILE
                        ),
                        "score_weight_diversity_param": float(SCORE_WEIGHT_DIVERSITY),
                        "score_weight_silhouette_param": float(SCORE_WEIGHT_SILHOUETTE),
                        "score_weight_davies_bouldin_param": float(SCORE_WEIGHT_DAVIES_BOULDIN),
                        "score_weight_calinski_harabasz_param": float(
                            SCORE_WEIGHT_CALINSKI_HARABASZ
                        ),
                        "score_weight_purity_param": float(SCORE_WEIGHT_PURITY),
                        "score_weight_other_feature_diversity_param": float(
                            SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY
                        ),
                        "score_weight_other_feature_purity_param": float(
                            SCORE_WEIGHT_OTHER_FEATURE_PURITY
                        ),
                        "ai_model_provider_param": AI_MODEL_PROVIDER,
                        "ollama_server_url_param": OLLAMA_SERVER_URL,
                        "ollama_model_name_param": OLLAMA_MODEL_NAME,
                        "openai_server_url_param": OPENAI_SERVER_URL,
                        "openai_model_name_param": OPENAI_MODEL_NAME,
                        "openai_api_key_param": OPENAI_API_KEY,
                        "gemini_api_key_param": GEMINI_API_KEY,
                        "gemini_model_name_param": GEMINI_MODEL_NAME,
                        "mistral_api_key_param": MISTRAL_API_KEY,
                        "mistral_model_name_param": MISTRAL_MODEL_NAME,
                        "top_n_moods_for_clustering_param": int(TOP_N_MOODS),
                        "enable_clustering_embeddings_param": bool(ENABLE_CLUSTERING_EMBEDDINGS),
                        "output_server_scope": server_scope,
                    }
                    try:
                        rq_queue_high.enqueue(
                            'tasks.clustering.run_clustering_task',
                            kwargs=clustering_kwargs,
                            job_id=job_id,
                            description='Cron Clustering',
                            job_timeout=-1,
                        )
                        logger.info(f"Cron: enqueued clustering job {job_id}")
                    except Exception:
                        logger.exception("Cron: enqueue failed for clustering")
                        save_task_status(
                            job_id, f"main_{task_type}", TASK_STATUS_FAILURE,
                            details={"error": "Could not enqueue the task (is Redis reachable?)"},
                        )
                elif task_type in ('sonic_fingerprint', 'alchemy_radio'):
                    # Enqueued, never run inline: these call the media server once per
                    # server in scope, and doing that on the 60s poll thread let one
                    # unreachable provider swallow whole scheduling windows.
                    dotted = (
                        'tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task'
                        if task_type == 'sonic_fingerprint'
                        else 'tasks.radio_manager.run_radio_playlists_task'
                    )
                    save_task_status(
                        job_id,
                        task_type,
                        TASK_STATUS_PENDING,
                        details={"message": _ENQUEUED_BY_CRON},
                    )
                    try:
                        rq_queue_default.enqueue(
                            dotted,
                            kwargs={'server_scope': server_scope},
                            job_id=job_id,
                            description=f'Cron {task_type}',
                            job_timeout=-1,
                        )
                        logger.info(f"Cron: enqueued {task_type} job {job_id}")
                    except Exception:
                        logger.exception(f"Cron: enqueue failed for {task_type}")
                        save_task_status(
                            job_id, task_type, TASK_STATUS_FAILURE,
                            details={"error": "Could not enqueue the task (is Redis reachable?)"},
                        )
                elif task_type.startswith('plugin.'):
                    from plugin.manager import plugin_manager

                    cron_task = plugin_manager.get_cron_task(task_type)
                    if not cron_task:
                        logger.warning(
                            f"Cron: no registered plugin task for {task_type}; skipping"
                        )
                    else:
                        save_task_status(
                            job_id,
                            task_type,
                            TASK_STATUS_PENDING,
                            details={"message": _ENQUEUED_BY_CRON},
                        )
                        queue = rq_queue_high if cron_task.get('queue') == 'high' else rq_queue_default
                        try:
                            queue.enqueue(
                                'plugin.manager.run_plugin_task',
                                args=(cron_task['dotted'],),
                                kwargs={'server_scope': server_scope},
                                job_id=job_id,
                                description=f'Cron {task_type}',
                                job_timeout=-1,
                            )
                            logger.info(f"Cron: enqueued plugin task {task_type} job {job_id}")
                        except Exception:
                            logger.exception(f"Cron: enqueue failed for {task_type}")
                            save_task_status(
                                job_id, task_type, TASK_STATUS_FAILURE,
                                details={"error": "Could not enqueue the task (is Redis reachable?)"},
                            )
        except Exception:
            db.rollback()
            logger.exception(f"Error processing cron row {r}")
    cur.close()
