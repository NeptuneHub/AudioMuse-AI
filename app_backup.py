# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for database backup and restore.

Serves the `/backup` UI and drives `pg_dump`/`psql` against the configured
Postgres instance, coordinating with `restart_manager` to bounce the app and
workers around a restore.

Main Features:
* Routes: `/backup` page, `/api/backup/create`, `/api/backup/download/<filename>`,
  `/api/backup/restore`.
* Connects exactly like the rest of the app: `pg_dump`/`psql` are handed
  `config.DATABASE_URL`, with the password moved into PGPASSWORD so it never
  reaches argv or the restore log.
* Serializes restores across containers with a self-releasing Redis lock and
  strips the PG17+ `SET transaction_timeout` prologue line that PG15/16 reject.
* The restore runs detached, so `/api/backup/restore` answers "started" long
  before the outcome is known; the runner appends a `RESTORE-RESULT:` marker that
  `/api/backup/restore-status` reports back to the page.
* Backups are compressed to .zip; restore accepts .sql or .zip uploads
  (zip detected by magic bytes and extracted before psql).
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import logging
import tempfile
import zipfile
from datetime import datetime
from functools import lru_cache
from urllib.parse import unquote, urlsplit, urlunsplit
from flask import Blueprint, render_template, jsonify, request, send_file
from redis.exceptions import RedisError
import config
from sanitization import sanitize_for_log
import restart_manager
from taskqueue import new_redis_connection
from error import error_manager
from error.error_dictionary import (
    ERR_BACKUP_VERSION_MISMATCH,
    ERR_BACKUP_FAILED,
    ERR_RESTORE_FAILED,
)

logger = logging.getLogger(__name__)

backup_bp = Blueprint('backup_bp', __name__)

BACKUP_DIR = os.environ.get("BACKUP_DIR", "/app/backup")
RESTORE_LOG_DIR = os.environ.get("RESTORE_LOG_DIR", BACKUP_DIR)

# Cross-container restore lock. Only one restore may run at a time.
# TTL self-releases on crash; runner releases explicitly on clean exit.
RESTORE_LOCK_KEY = 'audiomuse:restore_lock'
RESTORE_LOCK_TTL_SECONDS = 60 * 60  # 1 hour

# Machine-readable outcome the detached runner appends to its log. The HTTP
# request answers "started" long before the runner finishes, so this marker is
# the only way the page can tell an abort from a completed restore.
RESTORE_RESULT_MARKER = 'RESTORE-RESULT:'
RESTORE_RESULT_ABORTED = 'aborted'
RESTORE_RESULT_FAILED = 'failed'
RESTORE_RESULT_COMPLETED = 'completed'
RESTORE_RESULT_COMPLETED_DEGRADED = 'completed_degraded'
RESTORE_LOG_NAME_PATTERN = re.compile(r'^restore_\d{8}_\d{6}\.log$')


def _acquire_restore_lock():
    """SET NX EX. Returns True if we got the lock, False if held or Redis is down."""
    try:
        client = new_redis_connection(
            socket_connect_timeout=5, socket_timeout=5, decode_responses=True
        )
        return bool(client.set(RESTORE_LOCK_KEY, '1', nx=True, ex=RESTORE_LOCK_TTL_SECONDS))
    except RedisError:
        logger.exception("Redis unavailable while acquiring restore lock; failing closed.")
        return False


def _release_restore_lock():
    try:
        new_redis_connection(socket_connect_timeout=5, socket_timeout=5).delete(RESTORE_LOCK_KEY)
    except RedisError:
        logger.exception("Redis unavailable while releasing restore lock; relying on TTL.")


def _restore_lock_held():
    """Returns True if a restore lock is currently set in Redis.

    On Redis errors, returns True to fail closed - better to refuse a chunk
    than to let it write into a possibly-orphaned chunks_dir.
    """
    try:
        client = new_redis_connection(
            socket_connect_timeout=5, socket_timeout=5, decode_responses=True
        )
        return bool(client.exists(RESTORE_LOCK_KEY))
    except RedisError:
        logger.exception("Redis unavailable while checking restore lock; failing closed.")
        return True


@lru_cache(maxsize=1)
def _split_conninfo(database_url):
    parts = urlsplit(database_url)
    userinfo, at, hostport = parts.netloc.rpartition('@')
    user, _, password = userinfo.partition(':')
    conninfo = urlunsplit(
        (parts.scheme, f"{user}{at}{hostport}", parts.path, parts.query, parts.fragment)
    )
    return conninfo, unquote(password)


def _pg_conninfo():
    return _split_conninfo(config.DATABASE_URL)


def _pg_env():
    """Return a copy of os.environ with PGPASSWORD set."""
    _, password = _pg_conninfo()
    env = os.environ.copy()
    env['PGPASSWORD'] = password
    return env


def _pg_cmd(tool, *extra_args):
    """Build a pg command list with the connection string."""
    conninfo, _ = _pg_conninfo()
    return [tool, '-d', conninfo, *extra_args]


# Only files created by create_backup may be served by the download route.
_BACKUP_FILENAME_RE = re.compile(r'audiomuse_backup_\d{8}_\d{6}\.(sql|zip)')

# pg_dump 17+ writes `SET transaction_timeout = 0;` in the dump prologue; that
# GUC does not exist before PG 17, so a dump from the bundled client 18 cannot
# be replayed into a PG 15/16 server. Drop the line on the way into psql.
_TXN_TIMEOUT_RE = re.compile(rb'(?m)^SET transaction_timeout\b[^\n]*\n')


def _is_contained(path, *allowed_dirs):
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    for base in allowed_dirs:
        try:
            base_real = os.path.realpath(base)
        except OSError:
            continue
        if os.path.commonpath([base_real, real]) == base_real:
            return True
    return False


def _unlink_restore_artifact(path):
    """Delete a restore working file, refusing anything outside our own dirs."""
    if not path:
        return
    if not _is_contained(path, BACKUP_DIR, RESTORE_LOG_DIR, tempfile.gettempdir()):
        logger.error(
            'Refusing to delete restore artifact outside the backup/temp dirs: %s',
            sanitize_for_log(path),
        )
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _feed_dump(stdin, dump_file, result):
    """Stream the dump into psql; record delivery in result so a short feed isn't reported as success."""
    try:
        with open(dump_file, 'rb') as src:
            head = _TXN_TIMEOUT_RE.sub(b'', src.read(1024 * 1024), count=1)
            stdin.write(b'DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;\n')
            stdin.write(head)
            shutil.copyfileobj(src, stdin, 1024 * 1024)
        result['ok'] = True
    except BrokenPipeError:
        result['error'] = 'psql closed the input stream before the dump finished'
    except OSError as exc:
        result['error'] = str(exc)
    finally:
        try:
            stdin.close()
        except OSError:
            pass


def _extract_sql_if_zip(dump_file, log):
    try:
        with open(dump_file, 'rb') as fh:
            if fh.read(4) != b'PK\x03\x04':
                return dump_file, None
    except OSError as exc:
        log.write(f"Restore FAILED: could not read backup file: {exc}\n")
        return None, None
    log.write("Backup file is a zip archive; extracting SQL dump.\n")
    tmp = None
    try:
        with zipfile.ZipFile(dump_file) as zf:
            member = next((n for n in zf.namelist() if n.lower().endswith('.sql')), None)
            if member is None:
                log.write("Restore FAILED: no .sql file found inside the zip archive.\n")
                return None, None
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sql')
            with zf.open(member) as src:
                shutil.copyfileobj(src, tmp, 1024 * 1024)
            tmp.close()
            log.write(f"Extracted {member} from the zip archive.\n")
            return tmp.name, tmp.name
    except (zipfile.BadZipFile, OSError) as exc:
        log.write(f"Restore FAILED: could not extract zip archive: {exc}\n")
        if tmp is not None:
            try:
                tmp.close()
                os.unlink(tmp.name)
            except OSError:
                pass
        return None, None


def _publish_worker_start(log=None):
    """Request worker recovery and report the actual ACK result."""
    try:
        started = restart_manager.publish_start_request()
    except Exception as exc:
        if log is not None:
            log.write(f"Failed to request worker start: {exc}\n")
            log.flush()
        else:
            logger.exception("Failed to request worker start")
        return False
    if log is not None:
        if started:
            log.write("Worker start request completed successfully.\n")
        else:
            log.write("Worker start request FAILED or was not acknowledged.\n")
        log.flush()
    elif not started:
        logger.error("Worker start request failed or was not acknowledged")
    return bool(started)


def _write_restore_result(log, result, message):
    try:
        log.write(f"{RESTORE_RESULT_MARKER} {result} {message}\n")
        log.flush()
    except OSError:
        logger.exception("Could not record the restore outcome marker")


def _run_restore_runner(dump_file, log_file):
    """Run the restore outside the Flask request in a detached process."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a', encoding='utf-8', errors='ignore') as log:
        log.write(f"Restore runner started at {datetime.now().isoformat()}\n")
        log.write(f"Dump file: {dump_file}\n")
        log.flush()

        try:
            env = _pg_env()
        except ValueError as exc:
            log.write(f"Restore FAILED: the configured database connection is unusable: {exc}\n")
            log.flush()
            _publish_worker_start(log)
            _unlink_restore_artifact(dump_file)
            _write_restore_result(
                log,
                RESTORE_RESULT_FAILED,
                'The database was NOT touched: the configured database connection is unusable.',
            )
            _release_restore_lock()
            return 1

        flask_stopped = False
        try:
            flask_stopped = bool(restart_manager.stop_local_flask_service())
            if flask_stopped:
                log.write("Stopped local Flask service.\n")
                log.flush()
            else:
                log.write("Restore ABORTED: local Flask service did not confirm it stopped.\n")
                log.flush()
        except Exception as exc:
            log.write(f"Restore ABORTED: failed to stop local Flask service: {exc}\n")
            log.flush()

        if not flask_stopped:
            # Never run psql while the application may still be writing. The
            # endpoint already stopped workers, so put them back before exiting.
            _publish_worker_start(log)
            try:
                if restart_manager.start_local_flask_service():
                    log.write("Ensured local Flask service is started after abort.\n")
                else:
                    log.write("Could not confirm local Flask start after abort.\n")
                log.flush()
            except Exception as exc:
                log.write(f"Failed to request local Flask start after abort: {exc}\n")
                log.flush()
            # Keep the uploaded dump. Deleting it forced the user to re-upload a
            # multi-gigabyte file after an abort they did not cause, and the HTTP
            # request already answered "restore started".
            log.write(
                f"THE DATABASE WAS NOT TOUCHED. The uploaded dump is kept at "
                f"{dump_file}; retry the restore once the services are healthy.\n"
            )
            log.flush()
            _write_restore_result(
                log,
                RESTORE_RESULT_ABORTED,
                'The database was NOT touched: the local Flask service did not '
                'confirm it stopped. Your uploaded dump was kept - retry once the '
                'services are healthy.',
            )
            _release_restore_lock()
            return 1

        try:
            from tasks.mcp_helper import _ensure_ai_chat_db_user

            _ensure_ai_chat_db_user()
            log.write("Ensured AI chat DB role exists before restore.\n")
            log.flush()
        except Exception as exc:
            log.write(f"Could not ensure AI chat DB role exists: {exc}; continuing anyway.\n")
            log.flush()

        restore_cmd = _pg_cmd(
            'psql',
            '-v',
            'ON_ERROR_STOP=1',
            '--single-transaction',
        )
        log.write(f"Running restore command: {' '.join(restore_cmd)} < {dump_file} (via stdin)\n")
        log.write(
            "Streaming dump via stdin (stripping pg_dump 17+ transaction_timeout for old-server compatibility).\n"
        )
        log.flush()

        sql_source, extracted = _extract_sql_if_zip(dump_file, log)
        log.flush()

        proc = None
        feeder = None
        feed_result = {}
        ret = -1
        if sql_source is None:
            log.write("Restore aborted: no usable SQL dump to feed psql.\n")
            log.flush()
        else:
            try:
                proc = subprocess.Popen(
                    restore_cmd,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
                feeder = threading.Thread(
                    target=_feed_dump, args=(proc.stdin, sql_source, feed_result), daemon=True
                )
                feeder.start()
                ret = proc.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                if proc is not None:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
                    proc.kill()
                    proc.wait()
                ret = -1
                log.write("Restore command timed out after 3600 seconds and was killed.\n")
                log.flush()
            except Exception as exc:
                log.write(f"Failed to execute restore command: {exc}\n")
                log.flush()
            finally:
                if feeder is not None:
                    feeder.join(timeout=10)
        if ret == 0 and not feed_result.get('ok'):
            ret = 1
            log.write(
                "Restore FAILED: dump was not fully streamed to psql (%s); database may be incomplete.\n"
                % feed_result.get('error', 'feeder did not finish')
            )
            log.flush()
        log.write(f"Restore command finished with return code {ret}\n")
        log.flush()

        if ret == 0:
            try:
                from database import USERS_PASSWORD_CHANGED_AT_DDL

                ensure_cmd = _pg_cmd(
                    'psql', '-v', 'ON_ERROR_STOP=1',
                    '-c', USERS_PASSWORD_CHANGED_AT_DDL,
                )
                ensure_ret = -1
                for attempt in (1, 2):
                    ensure_ret = subprocess.run(
                        ensure_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120
                    ).returncode
                    if ensure_ret == 0:
                        log.write("Ensured users session schema after restore.\n")
                        break
                    log.write(f"Users session schema ensure attempt {attempt} failed (rc={ensure_ret}).\n")
                    log.flush()
                    if attempt == 1:
                        time.sleep(5)
                if ensure_ret != 0:
                    log.write(
                        "WARNING: users session schema was not ensured; if logins fail "
                        "after this restore, restart the container to re-run schema init.\n"
                    )
                log.flush()
            except Exception as exc:
                log.write(
                    f"Could not ensure users session schema after restore: {exc}; "
                    f"restart the container if logins fail.\n"
                )
                log.flush()

        workers_started = False
        flask_started = False
        try:
            try:
                workers_started = _publish_worker_start(log)
            except Exception:
                # Cleanup and the local Flask start must still run if worker
                # start reporting itself unexpectedly fails.
                logger.exception("Unexpected worker-start reporting failure")

            try:
                flask_started = restart_manager.start_local_flask_service()
                if flask_started:
                    log.write("Started local Flask service.\n")
                else:
                    log.write("Local Flask start request FAILED.\n")
                log.flush()
            except Exception as exc:
                log.write(f"Failed to start local Flask service: {exc}\n")
                log.flush()

            for path in (dump_file, extracted):
                if not path:
                    continue
                try:
                    os.unlink(path)
                    log.write(f"Deleted temporary dump file {path}\n")
                    log.flush()
                except Exception as exc:
                    log.write(f"Could not delete temporary dump file {path}: {exc}\n")
                    log.flush()
        finally:
            _release_restore_lock()
            try:
                log.write("Released restore lock.\n")
                log.flush()
            except OSError:
                pass

        log.write(f"Restore runner finished at {datetime.now().isoformat()}\n")
        if ret == 0 and not (workers_started and flask_started):
            log.write(
                "Database restore completed, but service recovery FAILED; "
                "the restored database was committed and services require manual recovery.\n"
            )
            _write_restore_result(
                log,
                RESTORE_RESULT_COMPLETED_DEGRADED,
                'The database WAS restored, but the services did not come back. '
                'Restart AudioMuse manually.',
            )
            return 2
        if ret == 0:
            _write_restore_result(
                log,
                RESTORE_RESULT_COMPLETED,
                'Database restore completed and the services were restarted.',
            )
        else:
            _write_restore_result(
                log,
                RESTORE_RESULT_FAILED,
                'The restore command failed. Check the restore log and the '
                'container logs; the database may be incomplete.',
            )

    return ret


@backup_bp.route('/backup')
def backup_page():
    """
    Backup & restore admin page.
    ---
    tags:
      - Backup
    summary: HTML page for creating and restoring database backups.
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template('backup.html', title='AudioMuse-AI - Backup & Restore', active='backup')


@backup_bp.route('/api/backup/create', methods=['POST'])
def create_backup():
    """
    Create a database backup.
    ---
    tags:
      - Backup
    summary: Run pg_dump on the application database and return the backup file name.
    description: |
      Removes any prior `audiomuse_backup_*` files in BACKUP_DIR, then runs
      `pg_dump --clean --if-exists` and compresses the dump into
      `audiomuse_backup_<TIMESTAMP>.zip`. pg_dump is bounded by a 600 second
      timeout. The archive itself is fetched in a second step via
      GET /api/backup/download/<filename> so the browser can stream it
      natively to disk.
    responses:
      200:
        description: pg_dump succeeded; the response carries the file name to download.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                filename:
                  type: string
                size_bytes:
                  type: integer
      500:
        description: pg_dump failed, was not installed, or timed out.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Remove old backup files
    for old in os.listdir(BACKUP_DIR):
        if old.startswith('audiomuse_backup_') and old.endswith(('.sql', '.zip')):
            try:
                os.remove(os.path.join(BACKUP_DIR, old))
            except OSError:
                pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"audiomuse_backup_{timestamp}.sql"
    filepath = os.path.join(BACKUP_DIR, filename)

    cmd = _pg_cmd('pg_dump', '--clean', '--if-exists', '--no-owner', '--no-acl')

    try:
        with open(filepath, 'w') as f:
            result = subprocess.run(
                cmd, env=_pg_env(), stdout=f, stderr=subprocess.PIPE, text=True, timeout=600
            )
        if result.returncode != 0:
            logger.error("pg_dump failed: %s", result.stderr)
            if os.path.exists(filepath):
                os.remove(filepath)
            stderr = result.stderr or ""
            if "server version mismatch" in stderr.lower():
                err = error_manager.build(ERR_BACKUP_VERSION_MISMATCH, stderr)
            else:
                err = error_manager.build(ERR_BACKUP_FAILED, stderr)
            return jsonify({**err, 'error': err['error_message']}), 500
    except FileNotFoundError:
        logger.exception("pg_dump not found on system PATH")
        err = error_manager.build(ERR_BACKUP_FAILED, "pg_dump is not installed or not on PATH.")
        return jsonify({**err, 'error': err['error_message']}), 500
    except subprocess.TimeoutExpired:
        logger.exception("pg_dump timed out")
        if os.path.exists(filepath):
            os.remove(filepath)
        err = error_manager.build(ERR_BACKUP_FAILED, "pg_dump timed out after 600 seconds.")
        return jsonify({**err, 'error': err['error_message']}), 500

    zip_filename = f"audiomuse_backup_{timestamp}.zip"
    zip_filepath = os.path.join(BACKUP_DIR, zip_filename)
    try:
        # Fastest deflate level: dumps are multi-GB text, level 1 still compresses
        # them well and the default level 6 is several times slower.
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            zf.write(filepath, arcname=filename)
        os.remove(filepath)
    except OSError:
        logger.exception("Failed to compress backup")
        for path in (filepath, zip_filepath):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                logger.warning("Could not delete %s after compression failure", path, exc_info=True)
        err = error_manager.build(ERR_BACKUP_FAILED, "Failed to compress the backup file.")
        return jsonify({**err, 'error': err['error_message']}), 500

    logger.info("Backup created: %s", zip_filepath)
    return jsonify(
        {'success': True, 'filename': zip_filename, 'size_bytes': os.path.getsize(zip_filepath)}
    )


@backup_bp.route('/api/backup/download/<filename>', methods=['GET'])
def download_backup(filename):
    """
    Download a previously created backup file.
    ---
    tags:
      - Backup
    summary: Stream a backup file created by /api/backup/create as an attachment.
    parameters:
      - in: path
        name: filename
        required: true
        schema:
          type: string
        description: File name returned by /api/backup/create (audiomuse_backup_<TIMESTAMP>.zip).
    responses:
      200:
        description: The backup file is returned as an attachment.
        content:
          application/zip:
            schema:
              type: string
              format: binary
      404:
        description: Invalid or unknown backup file name.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
    """
    if not _BACKUP_FILENAME_RE.fullmatch(filename):
        return jsonify({'error': 'Invalid backup file name.'}), 404
    filepath = os.path.join(BACKUP_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({'error': 'Backup file not found. Create a new backup first.'}), 404
    return send_file(filepath, as_attachment=True, download_name=filename)


def _read_restore_result_from(log_path):
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as log:
            lines = log.readlines()
    except OSError:
        logger.exception("Could not read restore log %s", log_path)
        return None
    for line in reversed(lines):
        if not line.startswith(RESTORE_RESULT_MARKER):
            continue
        remainder = line[len(RESTORE_RESULT_MARKER):].strip()
        result, _, message = remainder.partition(' ')
        return {'result': result, 'message': message.strip()}
    return None


@backup_bp.route('/api/backup/restore-status', methods=['GET'])
def restore_status():
    """
    Report the outcome of a detached restore run.
    ---
    tags:
      - Backup
    summary: Poll the outcome the detached restore runner recorded in its log.
    description: |
      `/api/backup/restore` answers as soon as the detached runner is spawned, so
      the caller cannot tell an aborted restore from a completed one. The runner
      appends a machine-readable outcome marker to its log when it exits; this
      endpoint reports it. While the runner is still working the state is
      `running`, and during a successful restore this endpoint is unreachable
      because the local Flask service is stopped on purpose.
    parameters:
      - in: query
        name: log
        required: true
        schema:
          type: string
        description: The `restore_log_name` returned by /api/backup/restore.
    responses:
      200:
        description: Current restore state.
      400:
        description: Missing or malformed log name.
    """
    log_name = os.path.basename(request.args.get('log', ''))
    if not RESTORE_LOG_NAME_PATTERN.fullmatch(log_name):
        return jsonify({'error': 'Invalid restore log name.'}), 400

    base_dir_real = os.path.realpath(RESTORE_LOG_DIR)
    log_path_real = os.path.realpath(os.path.join(base_dir_real, log_name))
    if os.path.commonpath([base_dir_real, log_path_real]) != base_dir_real:
        return jsonify({'error': 'Invalid restore log name.'}), 400

    outcome = _read_restore_result_from(log_path_real)
    if outcome is None:
        return jsonify({'state': 'running'})
    return jsonify({'state': outcome['result'], 'message': outcome['message']})


@backup_bp.route('/api/backup/restore', methods=['POST'])
def restore_backup():
    """
    Restore the database from an uploaded .sql dump.
    ---
    tags:
      - Backup
    summary: Upload a backup (.sql or .zip, single file or chunked) and replay it via psql.
    description: |
      Acquires a 1-hour Redis lock (`audiomuse:restore_lock`) to prevent
      concurrent restores. The endpoint accepts either:

      - A single full upload (no `chunk_num`/`total_chunks` form fields).
      - A chunked upload where each request carries one 1 GB chunk plus the
        chunk index. Chunks are saved into `BACKUP_DIR/chunks/` and reassembled
        when the last chunk arrives. The first chunk wipes any leftover chunks
        from a previous attempt.

      When all data has been received, a detached subprocess runs psql with
      `--single-transaction` and `ON_ERROR_STOP=1` against the configured
      Postgres database, then restarts the local Flask service. A .zip upload
      (detected by magic bytes) is extracted to the inner .sql before replay.
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            required: [confirmation, file]
            properties:
              confirmation:
                type: string
                description: Must equal "I want to restore the database from the backup. This action is not reversible".
              file:
                type: string
                format: binary
                description: The backup as .sql or .zip (or one chunk of it).
              chunk_num:
                type: integer
                description: 1-indexed chunk number; omit for single-file upload.
              total_chunks:
                type: integer
                description: Total number of chunks; omit for single-file upload.
    responses:
      200:
        description: |
          Either an intermediate "chunk received" acknowledgement or, on the
          last chunk / single upload, confirmation that the detached restore
          subprocess started.
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                message:
                  type: string
                all_chunks_received:
                  type: boolean
                chunk_num:
                  type: integer
                total_chunks:
                  type: integer
                received_chunks:
                  type: array
                  items:
                    type: integer
                missing_chunks:
                  type: array
                  items:
                    type: integer
                restore_pid:
                  type: integer
                restore_log:
                  type: string
      400:
        description: Confirmation phrase missing or chunk numbers invalid.
      409:
        description: A restore is already in progress (Redis lock held), or the chunked-upload session was overtaken / expired mid-upload.
      500:
        description: Server-side failure during chunk save, reassembly, or runner spawn.
      503:
        description: Lock service (Redis) unreachable.
    """
    confirmation = request.form.get('confirmation', '')
    expected = "I want to restore the database from the backup. This action is not reversible"
    if confirmation != expected:
        return jsonify({'error': 'Confirmation text does not match.'}), 400

    uploaded = request.files.get('file')
    if not uploaded or not uploaded.filename:
        return jsonify({'error': 'No file uploaded.'}), 400

    # Check if this is a chunked upload
    chunk_num = request.form.get('chunk_num')
    total_chunks = request.form.get('total_chunks')

    restore_file = None
    restore_log = None
    restore_pid = None
    workers_require_recovery = False

    try:
        if chunk_num and total_chunks:
            # Chunked upload mode
            try:
                chunk_num = int(chunk_num)
                total_chunks = int(total_chunks)
            except ValueError:
                return jsonify({'error': 'chunk_num and total_chunks must be integers.'}), 400

            if chunk_num < 1 or chunk_num > total_chunks or total_chunks < 1:
                return jsonify(
                    {
                        'error': f'Invalid chunk numbers: chunk_num={chunk_num}, total_chunks={total_chunks}'
                    }
                ), 400

            chunks_dir = os.path.join(BACKUP_DIR, 'chunks')
            os.makedirs(chunks_dir, exist_ok=True)

            # Cross-container restore lock: chunk 1 acquires it, later chunks
            # verify it is still held - protects against the lock auto-expiring
            # mid-upload and a different session taking over.
            if chunk_num == 1:
                if not _acquire_restore_lock():
                    logger.warning("Refusing chunk 1: restore lock already held.")
                    return jsonify(
                        {
                            'error': 'A database restore is already in progress. '
                            'Wait for it to finish, or wait up to 1 hour for the lock to auto-release.'
                        }
                    ), 409
            else:
                if not _restore_lock_held():
                    logger.warning("Refusing chunk %s: restore lock no longer held.", chunk_num)
                    return jsonify(
                        {
                            'error': 'Restore session expired or was overtaken. '
                            'Restart the upload from chunk 1.'
                        }
                    ), 409

            chunk_file = os.path.join(chunks_dir, f'backup_{chunk_num}_of_{total_chunks}.sql')

            # The first chunk marks the start of a new upload session: wipe any
            # leftovers so a previous failed run cannot leak stale data into the
            # reassembled file (chunks may match total_chunks but have different
            # contents).
            if chunk_num == 1:
                for f in os.listdir(chunks_dir):
                    if f.startswith('backup_') and f.endswith('.sql'):
                        try:
                            os.unlink(os.path.join(chunks_dir, f))
                        except Exception:
                            logger.warning("Could not delete leftover chunk %s", f, exc_info=True)

            # Save the current chunk
            try:
                uploaded.save(chunk_file)
                logger.info(f"Saved chunk {chunk_num}/{total_chunks}")
            except Exception:
                logger.exception("Failed to save chunk %s", chunk_num)
                err = error_manager.build(ERR_RESTORE_FAILED, f"Failed to save chunk {chunk_num}.")
                return jsonify({**err, 'error': err['error_message']}), 500

            # Rebuild the received set from disk (only chunks belonging to this session)
            received_chunks = set()
            for f in os.listdir(chunks_dir):
                if f.startswith('backup_') and f.endswith(f'_of_{total_chunks}.sql'):
                    try:
                        parts = f.replace('backup_', '').replace('.sql', '').split('_of_')
                        if len(parts) == 2:
                            received_chunks.add(int(parts[0]))
                    except (ValueError, IndexError):
                        pass

            logger.info(f"Received chunks: {sorted(received_chunks)}/{total_chunks}")

            # If all chunks received, reassemble
            if len(received_chunks) == total_chunks and all(
                i in received_chunks for i in range(1, total_chunks + 1)
            ):
                logger.info(f"All {total_chunks} chunks received. Reassembling...")

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sql')
                restore_file = tmp.name

                try:
                    for i in range(1, total_chunks + 1):
                        chunk_path = os.path.join(chunks_dir, f'backup_{i}_of_{total_chunks}.sql')
                        if not os.path.exists(chunk_path):
                            raise Exception(f"Chunk {i} is missing during reassembly!")
                        try:
                            bytes_read = 0
                            with open(chunk_path, 'rb') as chunk_f:
                                while True:
                                    buf = chunk_f.read(1024 * 1024)  # 1MB stream buffer
                                    if not buf:
                                        break
                                    tmp.write(buf)
                                    bytes_read += len(buf)
                            if bytes_read == 0:
                                raise Exception(f"Chunk {i} is empty!")
                        except IOError as e:
                            raise Exception(f"Error reading chunk {i}: {str(e)}") from e

                    tmp.close()
                    file_size = os.path.getsize(restore_file)
                    logger.info(f"Reassembly complete: {restore_file} ({file_size} bytes)")

                    # Clean up chunk files
                    for i in range(1, total_chunks + 1):
                        try:
                            os.unlink(os.path.join(chunks_dir, f'backup_{i}_of_{total_chunks}.sql'))
                        except Exception:
                            logger.warning("Could not delete chunk %s", i, exc_info=True)

                    # Start restore with reassembled file
                    all_chunks_received = True
                except Exception:
                    logger.exception("Failed to reassemble uploaded backup chunks")
                    if tmp:
                        try:
                            tmp.close()
                        except Exception:
                            pass
                    if restore_file and os.path.exists(restore_file):
                        os.unlink(restore_file)
                    # Free disk immediately - chunks are 1GB each.
                    for i in range(1, total_chunks + 1):
                        chunk_path = os.path.join(chunks_dir, f'backup_{i}_of_{total_chunks}.sql')
                        try:
                            if os.path.exists(chunk_path):
                                os.unlink(chunk_path)
                        except OSError:
                            logger.warning(
                                "Could not delete chunk %s after reassembly failure",
                                i,
                                exc_info=True,
                            )
                    _release_restore_lock()
                    return jsonify(
                        {'error': 'Failed to reassemble chunks due to an internal error.'}
                    ), 500
            else:
                # Still waiting for more chunks
                missing_chunks = [i for i in range(1, total_chunks + 1) if i not in received_chunks]
                return jsonify(
                    {
                        'success': True,
                        'message': f'Chunk {chunk_num}/{total_chunks} received. Waiting for chunks: {missing_chunks}',
                        'chunk_num': chunk_num,
                        'total_chunks': total_chunks,
                        'received_chunks': sorted(received_chunks),
                        'missing_chunks': missing_chunks,
                        'all_chunks_received': False,
                    }
                )
        else:
            # Single file upload (non-chunked)
            if not _acquire_restore_lock():
                logger.warning("Refusing non-chunked restore: lock already held.")
                return jsonify(
                    {
                        'error': 'A database restore is already in progress. '
                        'Wait for it to finish, or wait up to 1 hour for the lock to auto-release.'
                    }
                ), 409
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sql')
            uploaded.save(tmp)
            tmp.close()
            restore_file = tmp.name
            all_chunks_received = True

        # Start restore only if all chunks received or single file upload
        if restore_file and all_chunks_received:
            workers_require_recovery = True
            stop_requested = restart_manager.publish_stop_request()
            if not stop_requested:
                logger.error(
                    'Restore aborted: worker stop request failed or was not acknowledged'
                )
                _publish_worker_start()
                workers_require_recovery = False
                if restore_file and os.path.exists(restore_file):
                    os.unlink(restore_file)
                _release_restore_lock()
                return jsonify({
                    'error': 'Restore was not started because workers did not confirm they stopped.'
                }), 503
            logger.info('Worker stop request completed successfully')

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            restore_log = os.path.join(RESTORE_LOG_DIR, f"restore_{timestamp}.log")
            os.makedirs(RESTORE_LOG_DIR, exist_ok=True)

            if getattr(sys, 'frozen', False):
                restore_cmd = [sys.executable, '--run-restore', restore_file, restore_log]
            else:
                restore_cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    '--run-restore',
                    restore_file,
                    restore_log,
                ]
            popen_kwargs = dict(
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            if sys.platform == 'win32':
                popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(restore_cmd, **popen_kwargs)
            # The detached runner now owns worker recovery and lock cleanup.
            workers_require_recovery = False
            restore_pid = proc.pid
            logger.info("Restore started in detached process %s", restore_pid)

            return jsonify(
                {
                    'success': True,
                    'message': 'Database restore started.',
                    'restore_pid': restore_pid,
                    'restore_log': restore_log,
                    'restore_log_name': os.path.basename(restore_log),
                    'all_chunks_received': True,
                }
            )

    except FileNotFoundError:
        logger.exception("Python executable not found for restore runner")
        if workers_require_recovery:
            _publish_worker_start()
        if restore_file and os.path.exists(restore_file):
            os.unlink(restore_file)
        _release_restore_lock()
        err = error_manager.build(ERR_RESTORE_FAILED, "Python executable not found for restore runner.")
        return jsonify({**err, 'error': err['error_message']}), 500
    except Exception:
        logger.exception("Restore failed")
        if workers_require_recovery:
            _publish_worker_start()
        if restore_file and os.path.exists(restore_file):
            os.unlink(restore_file)
        _release_restore_lock()
        err = error_manager.build(ERR_RESTORE_FAILED, "Restore failed. Check server logs.")
        return jsonify({**err, 'error': err['error_message']}), 500


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--run-restore':
        dump_path = sys.argv[2]
        log_path = sys.argv[3]
        sys.exit(_run_restore_runner(dump_path, log_path))
    else:
        print('This module is intended to be imported by the Flask app.')
