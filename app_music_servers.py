# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""REST API for the media-server registry (multi-server support).

Lets the setup wizard and the shared server dropdown list, add, edit, test,
delete and set-default the configured media servers, and trigger the
cross-server matching sweep. Every configured server is always active; there
is no per-server enable/disable state. Listing is available to any
authenticated user (credentials masked); every mutation is admin-only,
mirroring the setup page - EXCEPT during first-run setup, where the wizard
itself is the (unauthenticated) caller and /api/setup is already open.

Main Features:
* CRUD over the registry with masked secrets and a preserve-on-mask update.
* Connection testing and per-server catalogue-matching sweep enqueue.
* Usable by the first-run setup wizard, so a fresh install can configure its
  media servers here before any admin account exists.
"""

import json
import logging
import time
import uuid

from flask import Blueprint, g, jsonify, request

import config
import taskqueue
from app_logging import sanitize_log_value
from app_helper import (
    coerce_db_details,
)
from database import (
    get_db,
    missing_required_creds,
    get_active_main_task,
    main_task_start_lock,
    record_task_history,
    stage_pending_task_row,
)
from taskqueue.sql import SWEEP_TASK_TYPE
from app_server_context import (
    merge_creds,
    server_public_dict,
    servers_for_ui,
)
from tasks import provider_probe
from tasks.mediaserver import registry

logger = logging.getLogger(__name__)

music_servers_bp = Blueprint('music_servers_bp', __name__)

_SUPPORTED_TYPES = ('jellyfin', 'emby', 'navidrome', 'lyrion', 'plex')

_SUPERSEDED_SWEEP_MESSAGE = 'Superseded by a new alignment covering all servers.'


def _superseded_sweep_details():
    return {
        'message': _SUPERSEDED_SWEEP_MESSAGE,
        'status_message': _SUPERSEDED_SWEEP_MESSAGE,
        'log': [_SUPERSEDED_SWEEP_MESSAGE],
    }


def _setup_in_progress():
    return bool(getattr(g, 'setup_needed', False))


def _is_admin_caller():
    return (
        _setup_in_progress()
        or (not config.AUTH_ENABLED)
        or getattr(g, 'auth_role', None) == 'admin'
    )


def _forbid_non_admin():
    if _is_admin_caller():
        return None
    return jsonify({"error": "Forbidden"}), 403


def _validate_type(server_type):
    return isinstance(server_type, str) and server_type.lower() in _SUPPORTED_TYPES


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def _apply_default_to_config():
    import restart_manager

    try:
        config.refresh_config()
        return bool(
            restart_manager.publish_restart_request(
                timeout_seconds=restart_manager.CONTROL_ACK_ADVISORY_TIMEOUT_SECONDS
            )
        )
    except Exception:
        logger.exception(
            "Default server was saved, but worker restart acknowledgement failed"
        )
        return False


def _restart_partial_failure(body, status_code=200):
    # The server row is ALREADY committed and the sweep already enqueued, so a 503
    # here made the admin UI print "Save failed." for a change that succeeded - and
    # retrying then hit "already exists" with no way to reach the success path.
    body['restart_acknowledged'] = False
    body['warning'] = (
        "The server change was saved, but worker restart was not acknowledged. "
        "Restart AudioMuse before starting new catalogue work."
    )
    return jsonify(body), status_code


def _revoke_active_sweeps(cur):
    terminal = (
        config.TASK_STATUS_SUCCESS,
        config.TASK_STATUS_FAILURE,
        config.TASK_STATUS_REVOKED,
    )
    cur.execute(
        # Any live sweep, root or child. The provider migration queues its
        # alignment as its own child so it does not take the start path, and a
        # root-only filter left it running while a new sweep started over it.
        "SELECT task_id, start_time FROM task_status WHERE task_type = %s "
        "AND status NOT IN (%s, %s, %s) FOR UPDATE",
        (SWEEP_TASK_TYPE,) + terminal,
    )
    rows = cur.fetchall()
    task_ids = [row[0] for row in rows]
    if task_ids:
        now = time.time()
        cur.execute(
            """
            UPDATE task_status
            SET status = %s, progress = 100, details = %s,
                timestamp = NOW(), end_time = COALESCE(end_time, %s)
            WHERE task_id = ANY(%s)
              AND status NOT IN (%s, %s, %s)
            """,
            (
                config.TASK_STATUS_REVOKED,
                json.dumps(_superseded_sweep_details()),
                now,
                task_ids,
                *terminal,
            ),
        )
        if cur.rowcount != len(task_ids):
            raise RuntimeError(
                "Active sweep set changed while superseding; refusing partial replacement"
            )
    return [
        (
            row[0],
            max(0.0, now - float(row[1])) if row[1] is not None else None,
        )
        for row in rows
    ]


def _record_superseded_sweep_history(records):
    details = _superseded_sweep_details()
    for task_id, duration in records:
        record_task_history(
            task_id,
            SWEEP_TASK_TYPE,
            config.TASK_STATUS_REVOKED,
            duration_seconds=duration,
            details=details,
        )


def _cleanup_superseded_sweep_jobs(task_ids):
    for stale_task_id in task_ids:
        try:
            taskqueue.request_cancel(stale_task_id)
        except Exception:
            logger.exception("Could not signal the superseded sweep %s", stale_task_id)


def _claim_replacement_sweep(task_id):
    db = get_db()
    try:
        with db.cursor() as cur:
            records = _revoke_active_sweeps(cur)
            staged = stage_pending_task_row(
                cur,
                task_id,
                SWEEP_TASK_TYPE,
                {'message': 'Server alignment queued for all servers.'},
            )
            if not staged:
                raise RuntimeError(
                    "Another alignment sweep took the live slot; not replacing it"
                )
        # NO commit here. The staged row only becomes runnable once the enqueue
        # writes its func, and committing between the two published a func-less
        # row that a crash left stranded for the 30-minute stale sweep. The caller
        # enqueues on this same connection and commits both together. The INSERT
        # itself lives in database.stage_pending_task_row so that the status and
        # the absent func stay whatever taskqueue's adoption guard accepts.
    except Exception:
        db.rollback()
        logger.exception("Could not atomically claim replacement sweep %s", task_id)
        raise
    return records


def _live_sweep_row():
    # get_active_main_task only looks at parentless rows, by design: it answers
    # "is a MAIN task running". The provider migration queues its alignment as its
    # own CHILD so it does not take the start path, so that question misses it and
    # a manual align would start a second sweep straight over the top of it.
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT task_id, status FROM task_status WHERE task_type = %s "
            "AND status = ANY(%s) ORDER BY timestamp DESC LIMIT 1",
            (SWEEP_TASK_TYPE, list(config.TASK_STATUS_LIVE)),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {'task_id': row[0], 'status': row[1]}


def _task_blocking_a_sweep():
    # Cleaning and provider migration both rewrite track_server_map, which is
    # exactly what a sweep writes. Migration refuses while a sweep runs; without
    # this the reverse was not true, so a sweep could start mid-repoint.
    for task_type in ('cleaning', 'provider_migration'):
        active = get_active_main_task(task_type=task_type)
        if active:
            return active
    return None


def _enqueue_sweep(at_front=False):
    task_id = str(uuid.uuid4())
    try:
        # Cleaning's own gate sees sweeps (exclude_task_types=()), so without the
        # shared lock a cleaning start and this enqueue could each pass their gate
        # before either had written its row - and both prune track_server_map.
        with main_task_start_lock():
            active = _task_blocking_a_sweep()
            if active:
                logger.warning(
                    "Server alignment not enqueued: %s task %s is still %s. "
                    "Re-run the alignment once it finishes.",
                    active['task_type'], active['task_id'], active['status'],
                )
                return None

            # Revoking only some old sweeps and then continuing would permit two
            # incompatible catalogue snapshots. Revoke the complete set and
            # insert the replacement in one transaction.
            db = get_db()
            records = _claim_replacement_sweep(task_id)
            try:
                taskqueue.enqueue(
                    'tasks.multiserver_sync.sweep_all_secondary_servers',
                    kwargs={'task_id': task_id},
                    task_id=task_id,
                    task_type=SWEEP_TASK_TYPE,
                    queue=taskqueue.QUEUE_HIGH,
                    priority=taskqueue.PRIORITY_FRONT if at_front else 0,
                    details={'message': 'Server alignment queued.'},
                    conn=db,
                )
                db.commit()
            except Exception:
                db.rollback()
                raise
        _record_superseded_sweep_history(records)
        superseded = [stale_id for stale_id, _duration in records]
        _cleanup_superseded_sweep_jobs(superseded)
        if superseded:
            logger.info(
                "Superseded %d active sweep(s) with consolidated alignment %s",
                len(superseded), task_id,
            )
        return task_id
    except Exception:
        logger.exception("Failed to enqueue the server alignment")
        return None


def _latest_sweep_task():
    try:
        db = get_db()
        cur = db.cursor()
        try:
            cur.execute(
                "SELECT task_id, status, progress, details FROM task_status "
                "WHERE task_type = %s ORDER BY timestamp DESC LIMIT 1",
                (SWEEP_TASK_TYPE,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
        if not row:
            return None
        details = coerce_db_details(row[3])
        if not isinstance(details, dict):
            details = {}
        message = details.get('status_message') or details.get('message') or ''
        return {'task_id': row[0], 'status': row[1], 'progress': row[2] or 0, 'message': message}
    except Exception:
        logger.exception("Could not load latest sweep task")
        return None


def _name_taken(name, exclude_server_id=None):
    wanted = (name or '').strip()
    if not wanted:
        return False
    server = registry.get_server_by_name(wanted)
    return server is not None and server['server_id'] != exclude_server_id


def _missing_cred_keys(server_type, creds):
    return missing_required_creds(server_type, creds)


def _placeholder_default():
    try:
        default = registry.get_default_server()
    except Exception:
        logger.exception("Could not read the default server")
        return None
    if default is None:
        return None
    if _missing_cred_keys(default['server_type'], default['creds']):
        return default
    return None


def _drop_unused_placeholder(placeholder):
    try:
        if registry.mapped_count(placeholder['server_id']):
            return False
        registry.delete_server(placeholder['server_id'])
        logger.info(
            "Removed the unconfigured seed server '%s'; '%s' is the default now.",
            placeholder['name'], registry.get_default_server_id(),
        )
        return True
    except Exception:
        logger.exception("Could not remove the unconfigured seed server")
        return False


@music_servers_bp.route('/api/servers', methods=['GET'])
def list_servers():
    payload = servers_for_ui()
    payload['sweep_task'] = _latest_sweep_task()
    if not _is_admin_caller():
        for server in payload['servers']:
            server.pop('creds', None)
    return jsonify(payload)


@music_servers_bp.route('/api/servers', methods=['POST'])
def add_server():
    forbidden = _forbid_non_admin()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    server_type = (data.get('server_type') or '').strip().lower()
    creds = data.get('creds') or {}
    if not name:
        return jsonify({"error": "Server name is required."}), 400
    if not _validate_type(server_type):
        return jsonify({"error": f"server_type must be one of {list(_SUPPORTED_TYPES)}."}), 400
    if not isinstance(creds, dict):
        return jsonify({"error": "creds must be an object."}), 400
    if _name_taken(name):
        return jsonify({"error": f"A server named '{name}' already exists; names must be unique."}), 400
    make_default = _as_bool(data.get('make_default', False))
    missing = _missing_cred_keys(server_type, creds)
    if missing:
        return jsonify(
            {"error": f"Missing required credentials for {server_type}: {', '.join(missing)}."}
        ), 400
    placeholder = _placeholder_default()
    if placeholder is not None and not make_default:
        logger.info(
            "No usable default server is configured; '%s' becomes the default.", name
        )
        make_default = True
    server_id = registry.add_server(
        name=name,
        server_type=server_type,
        creds=creds,
        music_libraries=data.get('music_libraries') or '',
        make_default=make_default,
    )
    sweep_task_id = None
    created = registry.get_server(server_id)
    restart_acknowledged = True
    if created and created['is_default']:
        if placeholder is not None and placeholder['server_id'] != server_id:
            _drop_unused_placeholder(placeholder)
        restart_acknowledged = _apply_default_to_config()
    # The sweep aligns the new server; it does not depend on workers having
    # acknowledged the restart. Skipping it left a committed server row permanently
    # unaligned, with no path to retry it (re-submitting hits "already exists").
    sweep_task_id = _enqueue_sweep()
    body = server_public_dict(created)
    body['sweep_task_id'] = sweep_task_id
    if not restart_acknowledged:
        return _restart_partial_failure(body, 201)
    return jsonify(body), 201


@music_servers_bp.route('/api/servers/<server_id>', methods=['PUT', 'PATCH'])
def update_server(server_id):
    forbidden = _forbid_non_admin()
    if forbidden:
        return forbidden
    existing = registry.get_server(server_id)
    if existing is None:
        return jsonify({"error": "Unknown server."}), 404
    data = request.get_json(silent=True) or {}
    server_type = data.get('server_type')
    if server_type is not None:
        server_type = server_type.strip().lower()
        if not _validate_type(server_type):
            return jsonify({"error": f"server_type must be one of {list(_SUPPORTED_TYPES)}."}), 400
    new_name = data.get('name').strip() if isinstance(data.get('name'), str) else None
    if isinstance(data.get('name'), str) and not new_name:
        return jsonify({"error": "Server name cannot be empty"}), 400
    if new_name and _name_taken(new_name, exclude_server_id=server_id):
        return jsonify({"error": f"A server named '{new_name}' already exists; names must be unique."}), 400
    creds = None
    if 'creds' in data and isinstance(data['creds'], dict):
        creds = merge_creds(existing['creds'], data['creds'])
    is_default = registry.get_default_server_id(get_db()) == server_id
    # The DEFAULT server is validated too: it is the one config projects onto
    # every unbound provider call, so saving it credential-less breaks the whole
    # install (and the providers would silently fall back to stale config values).
    if server_type is not None or creds is not None:
        effective_type = server_type or existing['server_type']
        effective_creds = creds if creds is not None else existing['creds']
        missing = _missing_cred_keys(effective_type, effective_creds)
        if missing:
            return jsonify(
                {"error": f"Missing required credentials for {effective_type}: {', '.join(missing)}."}
            ), 400
    registry.update_server(
        server_id,
        name=new_name,
        server_type=server_type,
        creds=creds,
        music_libraries=data.get('music_libraries'),
    )
    sweep_task_id = None
    restart_acknowledged = _apply_default_to_config() if is_default else True
    # Sweep only on changes that can alter track matching; renames never
    # re-match the catalogue.
    new_libraries = data.get('music_libraries')
    needs_sweep = (
        (server_type is not None and server_type != existing['server_type'])
        or (creds is not None and creds != (existing['creds'] or {}))
        or (new_libraries is not None and new_libraries != existing['music_libraries'])
    )
    if needs_sweep:
        sweep_task_id = _enqueue_sweep()
    body = server_public_dict(registry.get_server(server_id))
    body['sweep_task_id'] = sweep_task_id
    if not restart_acknowledged:
        return _restart_partial_failure(body)
    return jsonify(body)


@music_servers_bp.route('/api/servers/<server_id>', methods=['DELETE'])
def delete_server(server_id):
    forbidden = _forbid_non_admin()
    if forbidden:
        return forbidden
    try:
        deleted = registry.delete_server(server_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not deleted:
        return jsonify({"error": "Unknown server."}), 404
    return jsonify({"deleted": server_id})


@music_servers_bp.route('/api/servers/<server_id>/default', methods=['POST'])
def set_default_server(server_id):
    forbidden = _forbid_non_admin()
    if forbidden:
        return forbidden
    if registry.get_server(server_id) is None:
        return jsonify({"error": "Unknown server."}), 404
    registry.set_default(server_id)
    restart_acknowledged = _apply_default_to_config()
    sweep_task_id = _enqueue_sweep()
    payload = servers_for_ui()
    payload['sweep_task_id'] = sweep_task_id
    if not restart_acknowledged:
        return _restart_partial_failure(payload)
    return jsonify(payload)


def _parse_probe_request():
    forbidden = _forbid_non_admin()
    if forbidden:
        return None, None, forbidden
    data = request.get_json(silent=True) or {}
    server_type = (data.get('server_type') or '').strip().lower()
    creds = data.get('creds') or {}
    if not _validate_type(server_type):
        error = jsonify({"error": f"server_type must be one of {list(_SUPPORTED_TYPES)}."}), 400
        return None, None, error
    server_id = data.get('server_id')
    if server_id:
        existing = registry.get_server(server_id)
        if existing is not None:
            creds = merge_creds(existing['creds'], creds)
    return server_type, creds, None


@music_servers_bp.route('/api/servers/test', methods=['POST'])
def test_server():
    server_type, creds, error = _parse_probe_request()
    if error is not None:
        return error
    try:
        result = provider_probe.test_connection(server_type, creds)
    except Exception:
        logger.exception("Media server test connection failed")
        return jsonify({"ok": False, "error": "Connection test failed; check container logs."}), 200
    return jsonify(result)


@music_servers_bp.route('/api/servers/libraries', methods=['POST'])
def server_libraries():
    server_type, creds, error = _parse_probe_request()
    if error is not None:
        return error
    try:
        return jsonify(provider_probe.list_libraries(server_type, creds))
    except Exception:
        logger.exception("Media server list libraries failed")
        return jsonify({"libraries": [], "unsupported": True}), 200


@music_servers_bp.route('/api/servers/align', methods=['POST'])
def align_servers():
    forbidden = _forbid_non_admin()
    if forbidden:
        return forbidden
    task_id = _enqueue_sweep(at_front=True)
    if task_id is None:
        return jsonify({"error": "Could not enqueue the alignment; check container logs."}), 500
    return jsonify({"enqueued": True, "task_id": task_id}), 202


@music_servers_bp.route('/api/servers/<server_id>/sweep', methods=['POST'])
def sweep_server(server_id):
    forbidden = _forbid_non_admin()
    if forbidden:
        return forbidden
    if registry.get_server(server_id) is None:
        return jsonify({"error": "Unknown server."}), 404
    task_id = str(uuid.uuid4())
    try:
        # This endpoint had no gate at all, so a single-server sweep could start
        # mid-migration and rewrite the mappings it was repointing.
        with main_task_start_lock():
            active = _task_blocking_a_sweep()
            if active:
                return jsonify(
                    {
                        "error": f"A {active['task_type']} task is running. "
                                 f"Re-run the sweep once it finishes.",
                        "task_id": active['task_id'],
                    }
                ), 409
            active_sweep = _live_sweep_row()
            if active_sweep:
                return jsonify(
                    {
                        "error": "A server sweep is already in progress.",
                        "task_id": active_sweep['task_id'],
                        "status": active_sweep['status'],
                    }
                ), 409
            taskqueue.enqueue(
                'tasks.multiserver_sync.sweep_server',
                args=(server_id,),
                kwargs={'task_id': task_id},
                task_id=task_id,
                task_type=SWEEP_TASK_TYPE,
                queue=taskqueue.QUEUE_HIGH,
                details={
                    'message': 'Server matching sweep queued.',
                    'status_message': 'Server matching sweep queued.',
                },
            )
    except taskqueue.TaskAlreadyRunning as exc:
        return jsonify({"error": exc.user_message}), exc.status_code
    except Exception:
        logger.exception(
            "Failed to queue the matching sweep for server %s", sanitize_log_value(server_id)
        )
        return jsonify({"error": "Could not queue the sweep; check container logs."}), 500
    return jsonify({"enqueued": True, "task_id": task_id, "job_id": task_id, "server_id": server_id}), 202
