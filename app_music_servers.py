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
mirroring the setup page.

Main Features:
* CRUD over the registry with masked secrets and a preserve-on-mask update.
* Connection testing and per-server catalogue-matching sweep enqueue.
"""

import json
import logging
import uuid

from flask import Blueprint, g, jsonify, request
from rq.job import Job

import config
from app_helper import redis_conn, rq_queue_default, save_task_status, send_stop_job_command
from database import get_db
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


def _forbid_non_admin():
    if not config.AUTH_ENABLED:
        return None
    if getattr(g, 'auth_role', None) != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    return None


def _validate_type(server_type):
    return isinstance(server_type, str) and server_type.lower() in _SUPPORTED_TYPES


def _apply_default_to_config():
    """Propagate a default-server change to every process.

    The registry row that just changed IS the source of truth; the config module
    globals are only its projection. Reload them here for this process and
    request a restart so workers re-import config and re-project the row. No
    values are written anywhere - the registry was already updated by the caller.
    """
    import restart_manager

    config.refresh_config()
    restart_manager.publish_restart_request()


def _cancel_active_sweeps():
    """Revoke queued/running alignment sweeps so a consolidated one replaces them.

    Surgical per-sweep cancel: touches ONLY the stale sweep jobs, never the RQ
    queues or other task_status rows, so a running analysis (or any other job)
    keeps going when a server is added or edited. The REVOKED row is written
    first so the sweep's cooperative cancellation check picks it up even when
    the RQ commands fail.
    """
    cancelled = []
    try:
        db = get_db()
        cur = db.cursor()
        try:
            cur.execute(
                "SELECT task_id FROM task_status WHERE task_type = 'server_sweep' "
                "AND status NOT IN (%s, %s, %s)",
                (config.TASK_STATUS_SUCCESS, config.TASK_STATUS_FAILURE,
                 config.TASK_STATUS_REVOKED),
            )
            rows = cur.fetchall()
        finally:
            cur.close()
    except Exception:
        logger.exception("Could not look up active sweeps to supersede")
        return cancelled
    for row in rows:
        stale_task_id = row[0]
        try:
            save_task_status(
                stale_task_id, 'server_sweep', config.TASK_STATUS_REVOKED,
                progress=100,
                details={'message': 'Superseded by a new alignment covering all servers.'},
            )
            cancelled.append(stale_task_id)
        except Exception:
            logger.exception("Could not revoke superseded sweep %s", stale_task_id)
            continue
        try:
            job = Job.fetch(stale_task_id, connection=redis_conn)
            status = job.get_status(refresh=True)
            # str(JobStatus.QUEUED) is 'JobStatus.QUEUED'; .value is 'queued'.
            status_value = getattr(status, 'value', None) or str(status)
            if status_value in ('queued', 'deferred', 'scheduled'):
                job.cancel()
            elif status_value == 'started':
                send_stop_job_command(redis_conn, stale_task_id)
        except Exception:
            logger.exception("RQ cleanup failed for superseded sweep %s", stale_task_id)
    return cancelled


def _enqueue_sweep(at_front=False):
    """Replace any queued/running sweep with one alignment of every server.

    Adding several servers back to back cancels the previous alignment each time
    and starts a fresh one, so the newest sweep always covers every not-yet-aligned
    server and no stale sweep for an outdated server set keeps running.
    """
    superseded = _cancel_active_sweeps()
    task_id = str(uuid.uuid4())
    try:
        save_task_status(
            task_id, 'server_sweep', config.TASK_STATUS_PENDING,
            details={'message': 'Server alignment queued for all servers.'},
        )
        rq_queue_default.enqueue(
            'tasks.multiserver_sync.sweep_all_secondary_servers',
            kwargs={'task_id': task_id},
            job_id=task_id,
            job_timeout=-1,
            at_front=at_front,
        )
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
                "WHERE task_type = 'server_sweep' ORDER BY timestamp DESC LIMIT 1"
            )
            row = cur.fetchone()
        finally:
            cur.close()
        if not row:
            return None
        details = row[3]
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except ValueError:
                details = {}
        message = (details or {}).get('status_message') or (details or {}).get('message') or ''
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
    """Required-but-empty cred keys for ``server_type`` (url/token/... style keys)."""
    required = [
        config.MEDIASERVER_CRED_KEY_BY_FIELD[field]
        for field in config.MEDIASERVER_FIELDS_BY_TYPE.get(server_type, [])
        if field in config.MEDIASERVER_CRED_KEY_BY_FIELD
    ]
    creds = creds or {}
    return [key for key in required if not creds.get(key)]


@music_servers_bp.route('/api/servers', methods=['GET'])
def list_servers():
    """List configured media servers plus the default id.

    Admins receive each server's masked credentials (to prefill the setup editor);
    non-admins receive only the fields the menu dropdown needs, with no creds.
    """
    payload = servers_for_ui()
    payload['sweep_task'] = _latest_sweep_task()
    is_admin = (not config.AUTH_ENABLED) or getattr(g, 'auth_role', None) == 'admin'
    if not is_admin:
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
    make_default = bool(data.get('make_default', False))
    missing = _missing_cred_keys(server_type, creds)
    if missing:
        return jsonify(
            {"error": f"Missing required credentials for {server_type}: {', '.join(missing)}."}
        ), 400
    server_id = registry.add_server(
        name=name,
        server_type=server_type,
        creds=creds,
        music_libraries=data.get('music_libraries') or '',
        make_default=make_default,
    )
    sweep_task_id = None
    created = registry.get_server(server_id)
    if created and created['is_default']:
        _apply_default_to_config()
    sweep_task_id = _enqueue_sweep()
    body = server_public_dict(created)
    body['sweep_task_id'] = sweep_task_id
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
    if not is_default and (server_type is not None or creds is not None):
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
    if is_default:
        _apply_default_to_config()
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
    sweep_task_id = _enqueue_sweep()
    _apply_default_to_config()
    payload = servers_for_ui()
    payload['sweep_task_id'] = sweep_task_id
    return jsonify(payload)


@music_servers_bp.route('/api/servers/test', methods=['POST'])
def test_server():
    forbidden = _forbid_non_admin()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True) or {}
    server_type = (data.get('server_type') or '').strip().lower()
    creds = data.get('creds') or {}
    if not _validate_type(server_type):
        return jsonify({"error": f"server_type must be one of {list(_SUPPORTED_TYPES)}."}), 400
    server_id = data.get('server_id')
    if server_id:
        existing = registry.get_server(server_id)
        if existing is not None:
            creds = merge_creds(existing['creds'], creds)
    try:
        result = provider_probe.test_connection(server_type, creds)
    except Exception:
        logger.exception("Media server test connection failed")
        return jsonify({"ok": False, "error": "Connection test failed; check container logs."}), 200
    return jsonify(result)


@music_servers_bp.route('/api/servers/libraries', methods=['POST'])
def server_libraries():
    forbidden = _forbid_non_admin()
    if forbidden:
        return forbidden
    data = request.get_json(silent=True) or {}
    server_type = (data.get('server_type') or '').strip().lower()
    creds = data.get('creds') or {}
    if not _validate_type(server_type):
        return jsonify({"error": f"server_type must be one of {list(_SUPPORTED_TYPES)}."}), 400
    server_id = data.get('server_id')
    if server_id:
        existing = registry.get_server(server_id)
        if existing is not None:
            creds = merge_creds(existing['creds'], creds)
    try:
        return jsonify(provider_probe.list_libraries(server_type, creds))
    except Exception:
        logger.exception("Media server list libraries failed")
        return jsonify({"libraries": [], "unsupported": True}), 200


@music_servers_bp.route('/api/servers/align', methods=['POST'])
def align_servers():
    """Align every secondary server against the default (no-op when aligned)."""
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
        save_task_status(
            task_id, 'server_sweep', config.TASK_STATUS_PENDING,
            details={'message': 'Server matching sweep queued.'},
        )
        rq_queue_default.enqueue(
            'tasks.multiserver_sync.sweep_server',
            args=(server_id,),
            kwargs={'task_id': task_id},
            job_id=task_id,
            job_timeout=-1,
        )
    except Exception:
        logger.exception("Failed to enqueue matching sweep for server %s", server_id)
        return jsonify({"error": "Could not enqueue the sweep; check container logs."}), 500
    return jsonify({"enqueued": True, "task_id": task_id, "job_id": task_id, "server_id": server_id}), 202
