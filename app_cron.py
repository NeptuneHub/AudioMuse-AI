# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for managing and running cron-scheduled tasks.

Serves the `/cron` UI and CRUD over the `cron` table, plus the tick function
that runs every enabled row whose cron expression matches now.

Main Features:
* Routes: `/cron` page and `/api/cron` (GET list, POST create/update), rejecting a
  cron expression that could never fire before it is stored as enabled.
* Cron evaluation that ENQUEUES the batch task types (analysis, clustering,
  plugin tasks) on the worker and runs the online ones (alchemy radio, sonic
  fingerprint, album of the week) INLINE here in Flask through one scaffold,
  `_run_inline`, because they query the in-memory similarity index, which only
  this process holds. The playlist builders resolve their function from
  task_types.CRON_INLINE_TASKS through taskqueue.resolve_func.
* Each row is claimed atomically for its wall-clock minute, so a restart or a
  second web process cannot double-fire it. A tick delayed by an inline run
  evaluates every minute it missed, up to CRON_RETRY_MAX_MINUTES (an older batch
  schedule becomes a visible skip); a wall-clock jump, told apart on the
  monotonic clock, evaluates only the current minute. A row fires at most once
  per tick.
* In one tick the batch rows are dispatched first (they only enqueue); then the
  online rows run one after another, each at least 10 seconds after the previous
  one actually started, so schedules sharing a minute never start together.
* A central queue guard makes analysis, clustering, plugin tasks (and the
  manual cleaning / provider migration starts) mutually exclusive; a scheduled
  run blocked by a live queue-guard task is recorded in `cron_retry` and
  re-attempted every `CRON_RETRY_INTERVAL_MINUTES` up to
  `CRON_RETRY_MAX_MINUTES`, then recorded as a visible skip and never started
  (fail-safe); the window is never extended.
* An inline run never gates batch work: its task type is self-managed (no Start
  ever 409s behind it and it never waits for a batch), it has no queue func so
  the maintenance reclaim cannot mistake it for a dead worker's orphan, its
  terminal row merges into the details it reported, and an inline run that
  executed is never re-run by the retry. `reap_interrupted_inline_runs` fails at
  cron-thread startup whatever a restart left non-terminal. An inline run does
  block the cron poll thread while it runs, the accepted trade.
"""

from flask import Blueprint, render_template, jsonify, request
from psycopg2.extensions import TRANSACTION_STATUS_INERROR
from psycopg2.extras import DictCursor, Json
from database import (
    ConnectionLostError,
    get_db,
    get_task_info_from_db,
    save_task_status,
    get_queue_blocking_task,
    clean_up_previous_main_tasks,
    main_task_start_lock,
    record_cron_retry,
    list_pending_cron_retries,
    clear_cron_retry,
    bump_cron_retry,
    cron_retry_task_already_done,
    INLINE_FLASK_TASK_TYPES,
)
import task_types
import taskqueue
from config import (
    TASK_STATUS_FAILURE,
    TASK_STATUS_STARTED,
    TASK_STATUS_PROGRESS,
    TASK_STATUS_SUCCESS,
    TASK_STATUS_REVOKED,
    CRON_RETRY_MAX_MINUTES,
    CRON_RETRY_INTERVAL_MINUTES,
)
import json
import sys
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
    TOP_N_CLUSTERING_PLAYLIST,
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
from error.error_dictionary import (
    ERR_INVALID_REQUEST,
    ERR_SEARCH_FAILED,
)
from error.responses import json_error

cron_bp = Blueprint('cron_bp', __name__)

logger = logging.getLogger(__name__)

_ENQUEUED_BY_CRON = "Enqueued by cron."
_STARTED_BY_CRON = "Started by cron."

# 'enqueued' = handed to the queue, 'ran' = an inline run that EXECUTED (success,
# failure or cancel). Either way this occurrence happened, so the cron retry
# must never start it again.
_DISPATCH_DONE = ('enqueued', 'ran')

# Process-local: the last wall-clock minute run_due_cron_jobs evaluated, and the
# monotonic time of that tick. A gap the monotonic clock also measured is this
# thread having been busy (an inline run), so its minutes are caught up, for at
# most CRON_RETRY_MAX_MINUTES like any other wait. A wall-clock gap the monotonic
# clock did NOT measure is a clock jump (NTP step, a host waking from sleep);
# replaying it would fire every schedule of those hours at once, so such a tick
# evaluates only the current minute, as a first tick does.
_cron_clock = {'last_minute': None, 'last_monotonic': None}
_CLOCK_JUMP_TOLERANCE_SECONDS = 120

# task_type -> wall-clock time its schedule was last enabled or changed on the
# Scheduled Tasks page (see save_cron_entry).
_cron_saved_at = {}

# The catch-up tells a busy thread from a clock jump with time.monotonic, which
# stops while a Linux or macOS host sleeps but keeps counting through sleep on
# Windows. There a wake would look like a busy thread and replay the missed
# schedules, so on Windows a tick evaluates only the current minute, exactly as
# the scheduler did before the catch-up existed.
_CATCH_UP_SUPPORTED = sys.platform != 'win32'

# Online rows due in the same tick start this many seconds apart (owner rule).
_INLINE_STAGGER_SECONDS = 10


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
    # The page's same-minute warning quotes these, so it can never promise a
    # stagger or a retry window the scheduler does not actually apply.
    return render_template(
        'cron.html', title='AudioMuse-AI - Scheduled Tasks', active='cron',
        inline_stagger_seconds=_INLINE_STAGGER_SECONDS,
        cron_retry_max_minutes=CRON_RETRY_MAX_MINUTES,
    )


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
                    enum: [analysis, clustering, sonic_fingerprint, album_of_the_week, alchemy_radio]
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
    now_ts = time.time()
    retry_by_type = {
        retry['task_type']: retry for retry in list_pending_cron_retries(conn=db)
    }
    entries = []
    for r in rows:
        retry = retry_by_type.get(r['task_type'])
        if retry is not None and (retry['retry_until'] is None or retry['retry_until'] > now_ts):
            retry_info = {
                'retry_pending': True,
                'retry_until': retry['retry_until'],
                'retry_attempts': retry['attempts'],
                'retry_blocker_task_type': retry['blocker_task_type'],
            }
        else:
            retry_info = {'retry_pending': False}
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
                **retry_info,
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
                enum: [analysis, clustering, sonic_fingerprint, album_of_the_week, alchemy_radio]
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
        return json_error(ERR_INVALID_REQUEST, "'options' must be a JSON object")

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
            return json_error(ERR_INVALID_REQUEST, problem)

    db = get_db()
    cur = db.cursor()
    enabled = bool(data.get('enabled'))
    if data.get('id'):
        changed = _schedule_changed(cur, data.get('id'), cron_expr, enabled)
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
            changed = _schedule_changed(cur, existing[0], cron_expr, enabled)
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
            changed = enabled
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
    if changed:
        # The schedule starts from this save: a catch-up behind a long online run
        # must not fire it for a minute that passed before it (nor, after an
        # off-and-on, for one that passed while it was off), and must still fire
        # an occurrence due after it. Same process as the cron thread (one
        # gunicorn worker), so process memory is enough and no column is added.
        _cron_saved_at[data.get('task_type')] = time.time()
    return jsonify({'message': 'saved'}), 200


def _schedule_changed(cur, row_id, cron_expr, enabled):
    if not enabled:
        return False
    cur.execute("SELECT cron_expr, enabled FROM cron WHERE id = %s", (row_id,))
    before = cur.fetchone()
    return before is None or not before[1] or before[0] != cron_expr


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


def _claim_cron_minute(db, row_id, minute_start, cron_expr):
    cur = db.cursor()
    try:
        cur.execute(
            "UPDATE cron SET last_run = %s "
            "WHERE id = %s AND enabled = true AND cron_expr = %s "
            "AND (last_run IS NULL OR last_run < %s)",
            (minute_start, row_id, cron_expr, minute_start),
        )
        claimed = cur.rowcount == 1
        db.commit()
        return claimed
    finally:
        cur.close()


def _rollback_quietly(db):
    try:
        db.rollback()
    except Exception:
        logger.exception("Cron: rollback failed; the database connection is gone")


def _usable_db(db):
    if not db.closed:
        return db
    try:
        return get_db()
    except ConnectionLostError:
        return get_db()


def _clear_failed_transaction(db):
    # A failure the run caught internally can leave the shared connection in an
    # aborted transaction; clear it, or the terminal row is refused and the run
    # stays RUNNING until the stale-row sweep fails it.
    try:
        if db.get_transaction_status() == TRANSACTION_STATUS_INERROR:
            db.rollback()
    except Exception:
        logger.exception("Cron: could not clear an aborted transaction")


def _row_details(job_id):
    try:
        info = get_task_info_from_db(job_id)
    except Exception:
        logger.exception("Cron: could not read the details of inline run %s", job_id)
        return {}
    details = (info or {}).get('details')
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except ValueError:
            logger.exception(
                "Cron: the details of inline run %s are not valid JSON; the "
                "terminal row replaces them", job_id,
            )
            return {}
    return dict(details) if isinstance(details, dict) else {}


def _inline_success_details(job_id, summary):
    # The worker's own terminal-row builder, keeping the steps the run reported
    # in its log instead of collapsing them, as a queued SUCCESS does.
    return taskqueue.terminal_details(
        TASK_STATUS_SUCCESS, None, summary, previous=_row_details(job_id), keep_log=True,
    )


def _inline_failure_details(job_id, task_type, exc, error_code):
    return taskqueue.terminal_details(
        TASK_STATUS_FAILURE, f"The {task_type} run failed; check the container logs.",
        taskqueue.failure_record(exc, error_code),
        previous=_row_details(job_id), keep_log=True,
    )


def _write_inline_terminal(job_id, task_type, status, details, db=None):
    if db is not None:
        _clear_failed_transaction(db)
    try:
        written = save_task_status(
            job_id, task_type, status, progress=100, details=details,
            raise_on_error=True,
        )
    except Exception:
        logger.exception(
            "Cron: could not write %s on inline run %s (%s); the stale-row sweep "
            "fails it once QUEUE_INLINE_STALE_SECONDS pass", status, job_id, task_type,
        )
        return
    if not written:
        logger.warning(
            "Cron: inline run %s (%s) finished but its row was already terminal "
            "(cancelled, or failed by the stale-row sweep); %s was not written.",
            job_id, task_type, status,
        )


def _run_inline(db, job_id, task_type, run, error_code):
    # One scaffold for every inline cron row. The row is self-managed: it takes
    # no one-live-main slot and refuses no batch start, so it is written without
    # the start lock and runs beside whatever batch is live.
    try:
        save_task_status(
            job_id, task_type, TASK_STATUS_STARTED, progress=0,
            details={"message": _STARTED_BY_CRON}, raise_on_error=True,
        )
    except Exception:
        logger.exception("Cron: could not write the row of inline run %s", task_type)
        _rollback_quietly(db)
        return 'failed'
    try:
        summary = run()
    except taskqueue.TaskCancelled:
        # The row was revoked (Cancel), so it already carries its verdict.
        logger.info("Cron: inline run %s (%s) was cancelled.", job_id, task_type)
        _rollback_quietly(_usable_db(db))
        return 'ran'
    except Exception as exc:
        logger.exception("Cron: inline run of %s failed", task_type)
        # A database drop during the run moves get_db() to a new connection, so
        # every later step must act on that one, not the tick's dead one.
        db = _usable_db(db)
        _rollback_quietly(db)
        _write_inline_terminal(
            job_id, task_type, TASK_STATUS_FAILURE,
            _inline_failure_details(job_id, task_type, exc, error_code), db,
        )
        return 'ran'
    db = _usable_db(db)
    _clear_failed_transaction(db)
    _write_inline_terminal(
        job_id, task_type, TASK_STATUS_SUCCESS, _inline_success_details(job_id, summary), db,
    )
    logger.info("Cron: ran %s inline (job_id=%s, summary=%s)", task_type, job_id, summary)
    return 'ran'


def _run_playlist_inline(db, job_id, task_type, server_scope):
    dotted = task_types.CRON_INLINE_TASKS[task_type]
    return _run_inline(
        db, job_id, task_type,
        lambda: taskqueue.resolve_func(dotted)(
            server_scope=server_scope, inline_task_id=job_id
        ),
        taskqueue.TASK_FUNC_ERROR_CODES[dotted],
    )


def reap_interrupted_inline_runs():
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "SELECT task_id, task_type FROM task_status "
            "WHERE task_type = ANY(%s) AND parent_task_id IS NULL "
            "AND func IS NULL AND status NOT IN (%s, %s, %s)",
            (
                list(INLINE_FLASK_TASK_TYPES),
                TASK_STATUS_SUCCESS,
                TASK_STATUS_FAILURE,
                TASK_STATUS_REVOKED,
            ),
        )
        rows = cur.fetchall()
    except Exception:
        logger.exception("Cron: could not look up interrupted inline runs")
        db.rollback()
        return 0
    finally:
        cur.close()

    message = (
        "Interrupted by a restart of the web process, which is where this task "
        "runs. It was not completed; it runs again at its next scheduled time."
    )
    reaped = 0
    for row in rows:
        try:
            save_task_status(
                row['task_id'], row['task_type'], TASK_STATUS_FAILURE, progress=100,
                details={'message': message, 'status_message': message, 'error': message},
            )
            reaped += 1
        except Exception:
            logger.exception("Cron: could not fail interrupted run %s", row['task_id'])
    if reaped:
        logger.warning(
            "Cron: failed %d inline run(s) interrupted by a restart.", reaped
        )
    return reaped


def _inline_progress_reporter(job_id, task_type):
    def report(message, progress):
        pct = max(1, min(99, int(progress)))
        try:
            save_task_status(
                job_id, task_type, TASK_STATUS_PROGRESS, progress=pct,
                details={'message': message, 'status_message': message},
            )
        except Exception:
            logger.debug("Cron: inline progress update failed (ignored)", exc_info=True)

    return report


def _enqueue_cron_job(job_id, task_type, enqueue, *, conn=None):
    with main_task_start_lock(conn=conn):
        return _admit_and_enqueue_cron_job(job_id, task_type, enqueue, conn=conn)


def _admit_and_enqueue_cron_job(job_id, task_type, enqueue, *, conn):
    active = get_queue_blocking_task(conn=conn)
    if active:
        logger.info(
            "Cron: skipping %s, task %s is still %s",
            task_type, active['task_id'], active['status'],
        )
        return 'blocked'

    if task_type in ('main_analysis', 'main_clustering'):
        try:
            clean_up_previous_main_tasks()
        except Exception:
            logger.exception("Cron: could not archive previous main tasks; queueing anyway")

    try:
        enqueue()
        # Commit before returning. The enqueue runs on the request connection so
        # that the claim read and the queue row are one atomic act, which means
        # nothing has committed it yet: a later cron row in the same tick raising
        # into `db.rollback()` would otherwise throw this job away while last_run
        # stayed advanced - the schedule fires, nothing runs, and the next
        # occurrence is a minute later.
        if conn is not None:
            conn.commit()
    except taskqueue.TaskAlreadyRunning:
        logger.info("Cron: %s lost the race to another live main task.", task_type)
        return 'blocked'
    except Exception:
        logger.exception("Cron: could not queue %s", task_type)
        return 'failed'
    logger.info("Cron: queued %s job %s", task_type, job_id)
    return 'enqueued'


# Plugin cron rows are not in the registry's cron table (their task_type is the
# plugin's own dotted path, not a fixed name), but they are still admitted
# through the same blocking=True queue guard, so a busy queue can starve them
# exactly like the fixed types - they get the same retry coverage via the
# task_types.PREFIXES checks below.
def _cron_retry_eligible(task_type):
    return task_type in task_types.CRON_RETRY_TASK_TYPES or task_types.matches(
        task_type, prefixes=task_types.PREFIXES
    )


def _queue_type_for_cron_task_type(task_type):
    return task_types.CRON_TASK_TYPE_TO_QUEUE_TYPE.get(task_type, task_type)


def _dispatch_cron_row(db, r):
    task_type = r['task_type']
    # Batch work always covers every configured server, one server at
    # a time. There is no per-schedule scope: a "default server only"
    # schedule left every other server's exclusive songs unanalyzed
    # and without playlists, silently.
    server_scope = 'all'
    job_id = str(uuid.uuid4())
    if task_type == 'analysis':
        return _enqueue_cron_job(
            job_id,
            'main_analysis',
            lambda job_id=job_id, server_scope=server_scope: taskqueue.enqueue(
                'tasks.analysis.run_analysis_task',
                args=(0, TOP_N_MOODS),
                kwargs={'server_scope': server_scope},
                task_id=job_id,
                task_type='main_analysis',
                queue=taskqueue.QUEUE_HIGH,
                details={"message": _ENQUEUED_BY_CRON},
                conn=db,
            ),
            conn=db,
        )
    elif task_type == 'clustering':
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
            "top_n_playlists_param": int(TOP_N_CLUSTERING_PLAYLIST),
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
        return _enqueue_cron_job(
            job_id,
            'main_clustering',
            lambda job_id=job_id, clustering_kwargs=clustering_kwargs: taskqueue.enqueue(
                'tasks.clustering.run_clustering_task',
                kwargs=clustering_kwargs,
                task_id=job_id,
                task_type='main_clustering',
                queue=taskqueue.QUEUE_HIGH,
                details={"message": _ENQUEUED_BY_CRON},
                conn=db,
            ),
            conn=db,
        )
    elif task_type == 'alchemy_radio':
        # Radios run INLINE, here in the Flask process, because they are an
        # online feature: every radio queries the in-memory similarity index,
        # and only this process loads it (app.py skips the load when
        # AUDIOMUSE_ROLE=worker). Enqueued on a worker the index is absent, so
        # every radio failed with "no tracks available on this server". This
        # blocks the poll thread for the length of the run, which is the
        # accepted trade for a schedule that fires once a day.
        #
        # Nothing else can write this row's final status, so it must never
        # gate other work: alchemy_radio is in SELF_MANAGED_TASK_TYPES (so no
        # Start ever 409s behind it), the run heartbeats its progress, and
        # reap_interrupted_inline_runs fails whatever a restart left behind.
        from tasks.radio_manager import run_radio_playlists

        return _run_inline(
            db, job_id, task_type,
            lambda: run_radio_playlists(
                server_scope=server_scope,
                report=_inline_progress_reporter(job_id, task_type),
            ),
            ERR_SEARCH_FAILED,
        )
    elif task_type in task_types.CRON_INLINE_TASKS:
        # Run INLINE here in Flask, like the radios above and for the same
        # reason: both query the in-memory similarity index, and only this
        # process loads it (app.py skips the load when AUDIOMUSE_ROLE=worker).
        # They are online, self-managed types exactly like the radio: they hold
        # no one-live-main slot, so they run beside a live batch and no batch
        # start waits for them. The registry's dotted path is the whole
        # difference between them.
        return _run_playlist_inline(db, job_id, task_type, server_scope)
    elif task_types.matches(task_type, prefixes=task_types.PREFIXES):
        from plugin.manager import plugin_manager

        cron_task = plugin_manager.get_cron_task(task_type)
        if not cron_task:
            logger.warning(
                f"Cron: no registered plugin task for {task_type}; skipping"
            )
            return 'no_handler'
        queue = (
            taskqueue.QUEUE_HIGH
            if cron_task.get('queue') == 'high'
            else taskqueue.QUEUE_DEFAULT
        )
        return _enqueue_cron_job(
            job_id,
            task_type,
            lambda job_id=job_id, server_scope=server_scope, task_type=task_type, queue=queue, cron_task=cron_task: taskqueue.enqueue(
                'plugin.manager.run_plugin_task',
                args=(cron_task['dotted'],),
                kwargs={'server_scope': server_scope},
                task_id=job_id,
                task_type=task_type,
                queue=queue,
                details={"message": _ENQUEUED_BY_CRON},
                conn=db,
            ),
            conn=db,
        )
    return 'unknown'


def _record_cron_retry(db, task_type, due_ts=None):
    if not _cron_retry_eligible(task_type):
        return
    # The window counts from the minute the schedule was due, so a fire that was
    # already late (the cron thread was busy) cannot stretch the total wait past
    # CRON_RETRY_MAX_MINUTES.
    now_ts = time.time()
    first_blocked_at = now_ts if due_ts is None else min(due_ts, now_ts)
    blocker = get_queue_blocking_task(conn=db)
    record_cron_retry(
        task_type,
        first_blocked_at + CRON_RETRY_MAX_MINUTES * 60,
        first_blocked_at=first_blocked_at,
        blocker_task_id=blocker['task_id'] if blocker else None,
        blocker_task_type=blocker['task_type'] if blocker else None,
        conn=db,
    )


def _cron_row_for_retry(db, task_type):
    cur = db.cursor(cursor_factory=DictCursor)
    try:
        cur.execute(
            "SELECT id, name, task_type, cron_expr, enabled, last_run, options "
            "FROM cron WHERE task_type = %s AND enabled = true LIMIT 1",
            (task_type,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        cur.close()


def _record_retry_expired(task_type, entry, message=None, status_message=None):
    job_id = str(uuid.uuid4())
    details = {
        "message": message or (
            f"Scheduled {task_type} did not run: it was blocked for over "
            f"{CRON_RETRY_MAX_MINUTES} minutes and never became free."
        ),
        "status_message": status_message or (
            f"Blocked for over {CRON_RETRY_MAX_MINUTES} minutes; not run."
        ),
        "blocked_by": entry.get('blocker_task_type'),
        "attempts": entry.get('attempts', 0),
    }
    try:
        # The queue type (e.g. main_analysis), not the cron row's own name, so
        # this failed row reads like every other analysis/clustering task
        # instead of an unfamiliar bare "analysis"/"clustering" type.
        save_task_status(
            job_id, _queue_type_for_cron_task_type(task_type), TASK_STATUS_FAILURE,
            progress=100, details=details,
        )
    except Exception:
        logger.exception("Cron: could not record expired retry for %s", task_type)
    logger.warning("Cron: %s not run: %s", task_type, details['status_message'])


def _touch_cron_last_run(db, row_id):
    try:
        cur = db.cursor()
        try:
            cur.execute("UPDATE cron SET last_run = %s WHERE id = %s", (time.time(), row_id))
            db.commit()
        finally:
            cur.close()
    except Exception:
        logger.exception("Cron: could not update last_run for cron row %s", row_id)


def _retry_deadline(entry):
    # retry_until is nullable; a row without one falls back to its first refusal
    # plus the window, and a row with neither expires at once, so no NULL can
    # make an entry wait forever.
    if entry.get('retry_until') is not None:
        return entry['retry_until']
    if entry.get('first_blocked_at') is not None:
        return entry['first_blocked_at'] + CRON_RETRY_MAX_MINUTES * 60
    return 0


def retry_due_cron_jobs():
    pending = list_pending_cron_retries()
    if not pending:
        return 0
    db = get_db()
    now_ts = time.time()
    for entry in pending:
        task_type = entry['task_type']
        if now_ts >= _retry_deadline(entry):
            # The window is over: the schedule becomes a visible skip and is NOT
            # dispatched, not even once more. A run may start only inside
            # CRON_RETRY_MAX_MINUTES of its first refusal, never after, and the
            # window is never extended, so nothing waits forever.
            clear_cron_retry(task_type, conn=db)
            _record_retry_expired(task_type, entry)
            continue
        row = _cron_row_for_retry(db, task_type)
        if row is None:
            clear_cron_retry(task_type, conn=db)
            continue
        if cron_retry_task_already_done(task_type, entry.get('first_blocked_at'), conn=db):
            clear_cron_retry(task_type, conn=db)
            continue
        result = _dispatch_cron_row(db, row)
        if result in _DISPATCH_DONE:
            clear_cron_retry(task_type, conn=db)
            # The claim earlier stamped last_run for the blocked attempt; a
            # retried run actually happening now must move it forward too, or
            # the Scheduled Tasks page keeps showing the old blocked timestamp.
            _touch_cron_last_run(db, row['id'])
        elif result == 'blocked':
            blocker = get_queue_blocking_task(conn=db)
            bump_cron_retry(
                task_type,
                blocker_task_id=blocker['task_id'] if blocker else None,
                blocker_task_type=blocker['task_type'] if blocker else None,
                conn=db,
            )
        elif result == 'no_handler':
            clear_cron_retry(task_type, conn=db)
        else:
            bump_cron_retry(task_type, conn=db)
    return len(pending)


def _minutes_to_evaluate(now_ts, now_monotonic=None):
    # Returns (minutes to evaluate, minutes dropped), oldest first. An inline run
    # blocks this thread, so a tick can land minutes late; evaluating only "now"
    # silently lost every schedule due in between. The first tick after a start
    # evaluates only the current minute. Both clocks are read at the SAME moment
    # (the tick start), or a slow connect would look like a clock jump.
    current = now_ts - (now_ts % 60)
    if now_monotonic is None:
        now_monotonic = time.monotonic()
    previous = _cron_clock['last_minute']
    previous_monotonic = _cron_clock['last_monotonic']
    _cron_clock['last_minute'] = current
    _cron_clock['last_monotonic'] = now_monotonic
    if previous is None or previous >= current or not _CATCH_UP_SUPPORTED:
        return [current], []
    wall_gap = current - previous
    if previous_monotonic is not None and (
        wall_gap > now_monotonic - previous_monotonic + _CLOCK_JUMP_TOLERANCE_SECONDS
    ):
        logger.warning(
            "Cron: the wall clock moved %s minutes further than this thread ran; "
            "treating it as a clock jump and evaluating only the current minute, "
            "so schedules in that span are skipped.",
            int(round(wall_gap / 60)),
        )
        return [current], []
    count = int(round(wall_gap / 60))
    missed = [current - 60 * step for step in range(count - 1, -1, -1)]
    # At least the current minute: a window of 0 must mean "no catch-up", never
    # "missed[-0:]", which is the whole list.
    window = max(1, CRON_RETRY_MAX_MINUTES)
    if count <= window:
        return missed, []
    return missed[-window:], missed[:-window]


def _latest_due_minute(expr, minutes):
    for minute_start in reversed(minutes):
        if cron_matches_now(expr, minute_start):
            return minute_start
    return None


def _fire_cron_row(db, r, minute_start):
    try:
        # A long inline run earlier in this tick may have outlived the tick's
        # connection (a database restart); save_task_status reconnects on its
        # own, so pick up the live connection instead of the dead one.
        db = _usable_db(db)
        # Claim the row for THAT wall-clock minute before doing anything. The
        # old guard read last_run and wrote it after enqueuing, with no
        # predicate and a 55s window narrower than the 60s minute it was
        # protecting, so a restart inside a matching minute could double-fire.
        # Claiming must come AFTER the match: claiming every enabled row on
        # every tick would stamp last_run continuously and corrupt the
        # dashboard's Last-run display. The claim also re-checks that the row is
        # still enabled with the same expression: the SELECT is from the tick
        # start, and a schedule disabled on the page meanwhile must not run.
        if not _claim_cron_minute(db, r['id'], minute_start, r['cron_expr']):
            return False
        result = _dispatch_cron_row(db, r)
        db = _usable_db(db)
        if result == 'blocked':
            _record_cron_retry(db, r['task_type'], minute_start)
        elif result in _DISPATCH_DONE:
            # A later occurrence of the same task type ran: drop any stale
            # retry so it cannot duplicate this fresh run.
            clear_cron_retry(r['task_type'], conn=db)
        return True
    except Exception:
        _rollback_quietly(db)
        logger.exception(f"Error processing cron row {r}")
        return True


def _due_cron_rows(rows, minutes, dropped):
    due = []
    for r in rows:
        try:
            row_minutes, row_dropped = minutes, dropped
            saved_at = _cron_saved_at.get(r['task_type'])
            if saved_at is not None:
                # The schedule starts when it was enabled or changed on the page:
                # only the minutes after that save count, for a fire or a skip.
                row_minutes = [minute for minute in minutes if minute > saved_at]
                row_dropped = [minute for minute in dropped if minute > saved_at]
            # At most ONE fire per row per tick, for the most recent matching
            # minute in the window: an every-minute row behind a slow inline run
            # catches up once, not once per missed minute.
            minute_start = _latest_due_minute(r['cron_expr'], row_minutes)
            if (
                minute_start is None and row_dropped
                and r['task_type'] not in INLINE_FLASK_TASK_TYPES
                and _cron_retry_eligible(r['task_type'])
                and _latest_due_minute(r['cron_expr'], row_dropped) is not None
            ):
                # A batch schedule due longer than CRON_RETRY_MAX_MINUTES ago,
                # while this thread was busy, is past the window any wait gets:
                # it becomes a visible skip, never a silent loss. An online
                # schedule missed that way is simply skipped, as the owner allows.
                _record_retry_expired(
                    r['task_type'], {'attempts': 0},
                    message=(
                        f"Scheduled {r['task_type']} did not run: the scheduler was "
                        f"busy for over {CRON_RETRY_MAX_MINUTES} minutes (a long "
                        "online run, or the database was unreachable), so its start "
                        "time fell outside the window."
                    ),
                    status_message=(
                        f"Missed while the scheduler was busy for over "
                        f"{CRON_RETRY_MAX_MINUTES} minutes; not run."
                    ),
                )
        except Exception:
            logger.exception(f"Error processing cron row {r}")
            continue
        if minute_start is not None:
            due.append((r, minute_start))
    return due


def run_due_cron_jobs():
    tick_start = time.time()
    tick_monotonic = time.monotonic()
    db = get_db()
    cur = db.cursor(cursor_factory=DictCursor)
    cur.execute(
        "SELECT id, name, task_type, cron_expr, enabled, last_run, options "
        "FROM cron WHERE enabled = true ORDER BY id"
    )
    rows = cur.fetchall()
    cur.close()
    minutes, dropped = _minutes_to_evaluate(tick_start, tick_monotonic)
    due = _due_cron_rows(rows, minutes, dropped)
    # Batch rows only enqueue, so they go first (in row order, so which of two
    # batch schedules sharing a minute is queued and which waits is not left to
    # the table scan) and never wait behind an online run. The online rows run
    # here one after another, each starting at least _INLINE_STAGGER_SECONDS
    # after the previous one actually started, so schedules that share a minute
    # never hit the media server and the index all together; a run that already
    # took longer than the gap lets the next start at once. The gap is measured
    # on the monotonic clock: a wall-clock step must never stretch the sleep.
    online = sorted(
        (pair for pair in due if pair[0]['task_type'] in INLINE_FLASK_TASK_TYPES),
        key=lambda pair: (pair[0]['task_type'], pair[0]['id']),
    )
    for r, minute_start in due:
        if r['task_type'] not in INLINE_FLASK_TASK_TYPES:
            _fire_cron_row(db, r, minute_start)
    previous_start = None
    for r, minute_start in online:
        if previous_start is not None:
            wait = previous_start + _INLINE_STAGGER_SECONDS - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        started = time.monotonic()
        if _fire_cron_row(db, r, minute_start):
            previous_start = started


def cron_retry_interval_seconds():
    if CRON_RETRY_INTERVAL_MINUTES >= CRON_RETRY_MAX_MINUTES:
        logger.warning(
            "Cron: CRON_RETRY_INTERVAL_MINUTES (%s) is not below "
            "CRON_RETRY_MAX_MINUTES (%s); clamping so a retry can still run.",
            CRON_RETRY_INTERVAL_MINUTES, CRON_RETRY_MAX_MINUTES,
        )
        return max(1, CRON_RETRY_MAX_MINUTES - 1) * 60
    return CRON_RETRY_INTERVAL_MINUTES * 60
