import os
import subprocess
import sys
import time
import logging
import tempfile
from datetime import datetime
from flask import Blueprint, render_template, jsonify, request, send_file, after_this_request
from config import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB
import restart_manager

logger = logging.getLogger(__name__)

backup_bp = Blueprint('backup_bp', __name__)

BACKUP_DIR = os.environ.get("BACKUP_DIR", "/app/backup")
RESTORE_LOG_DIR = os.environ.get("RESTORE_LOG_DIR", BACKUP_DIR)


def _pg_env():
    """Return a copy of os.environ with PGPASSWORD set."""
    env = os.environ.copy()
    env['PGPASSWORD'] = POSTGRES_PASSWORD
    return env


def _pg_cmd(tool, *extra_args):
    """Build a pg command list with common connection args."""
    return [
        tool,
        '-h', POSTGRES_HOST,
        '-p', POSTGRES_PORT,
        '-U', POSTGRES_USER,
        *extra_args,
    ]


def _run_restore_runner(dump_file, log_file):
    """Run the restore outside the Flask request in a detached process."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    env = _pg_env()
    with open(log_file, 'a', encoding='utf-8', errors='ignore') as log:
        log.write(f"Restore runner started at {datetime.now().isoformat()}\n")
        log.write(f"Dump file: {dump_file}\n")
        log.flush()

        # Worker stop is published by the Flask restore endpoint before this
        # detached runner starts. The runner only waits briefly to allow
        # workers to settle before stopping the local Flask service.
        time.sleep(5)
        log.write("Wait complete. Proceeding with local Flask shutdown.\n")
        log.flush()

        try:
            if not restart_manager.stop_local_flask_service():
                log.write("Failed to stop local Flask service. Continuing restore anyway.\n")
                log.flush()
            else:
                log.write("Stopped local Flask service.\n")
                log.flush()
        except Exception as exc:
            log.write(f"Failed to stop local Flask service: {exc}\n")
            log.flush()
            log.write("Continuing restore despite local Flask stop failure.\n")
            log.flush()

        restore_cmd = _pg_cmd('psql', '-d', POSTGRES_DB, '-v', 'ON_ERROR_STOP=1', '-f', dump_file)
        log.write(f"Running restore command: {' '.join(restore_cmd)}\n")
        log.flush()

        proc = None
        ret = -1
        try:
            proc = subprocess.Popen(
                restore_cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                close_fds=True,
            )
            ret = proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
                proc.wait()
            ret = -1
            log.write("Restore command timed out after 3600 seconds and was killed.\n")
            log.flush()
        except Exception as exc:
            log.write(f"Failed to execute restore command: {exc}\n")
            log.flush()
            ret = -1
        log.write(f"Restore command finished with return code {ret}\n")
        log.flush()

        try:
            restart_manager.publish_start_request()
            log.write("Published worker start request.\n")
            log.flush()
        except Exception as exc:
            log.write(f"Failed to publish worker start request: {exc}\n")
            log.flush()

        try:
            restart_manager.start_local_flask_service()
            log.write("Started local Flask service.\n")
            log.flush()
        except Exception as exc:
            log.write(f"Failed to start local Flask service: {exc}\n")
            log.flush()

        try:
            os.unlink(dump_file)
            log.write(f"Deleted temporary dump file {dump_file}\n")
            log.flush()
        except Exception as exc:
            log.write(f"Could not delete temporary dump file {dump_file}: {exc}\n")
            log.flush()

        log.write(f"Restore runner finished at {datetime.now().isoformat()}\n")
        log.flush()

    return ret


@backup_bp.route('/backup')
def backup_page():
    return render_template('backup.html', title='AudioMuse-AI - Backup & Restore', active='backup')


@backup_bp.route('/api/backup/create', methods=['POST'])
def create_backup():
    """Full pg_dump of the application database and return the .sql file."""
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Remove old backup files
    for old in os.listdir(BACKUP_DIR):
        if old.startswith('audiomuse_backup_') and old.endswith('.sql'):
            try:
                os.remove(os.path.join(BACKUP_DIR, old))
            except OSError:
                pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"audiomuse_backup_{timestamp}.sql"
    filepath = os.path.join(BACKUP_DIR, filename)

    cmd = _pg_cmd('pg_dump', '--clean', '--if-exists', '-d', POSTGRES_DB)

    try:
        with open(filepath, 'w') as f:
            result = subprocess.run(cmd, env=_pg_env(), stdout=f, stderr=subprocess.PIPE, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("pg_dump failed: %s", result.stderr)
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'pg_dump failed: {result.stderr}'}), 500
    except FileNotFoundError:
        logger.error("pg_dump not found on system PATH")
        return jsonify({'error': 'pg_dump is not installed or not on PATH'}), 500
    except subprocess.TimeoutExpired:
        logger.error("pg_dump timed out")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'pg_dump timed out after 600 seconds'}), 500

    logger.info("Backup created: %s", filepath)
    return send_file(filepath, as_attachment=True, download_name=filename)


@backup_bp.route('/api/backup/restore', methods=['POST'])
def restore_backup():
    """Restore the database from an uploaded .sql dump file via psql."""
    confirmation = request.form.get('confirmation', '')
    expected = "I want to restore the database from the backup. This action is not reversible"
    if confirmation != expected:
        return jsonify({'error': 'Confirmation text does not match.'}), 400

    uploaded = request.files.get('file')
    if not uploaded or not uploaded.filename:
        return jsonify({'error': 'No file uploaded.'}), 400

    restore_file = None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sql')
    restore_log = None
    restore_pid = None
    try:
        uploaded.save(tmp)
        tmp.close()
        restore_file = tmp.name

        # Publish worker stop as soon as the upload has been persisted and before
        # starting the detached restore runner. This reduces the window between
        # upload completion and the stop request being sent.
        stop_requested = restart_manager.publish_stop_request()
        logger.info('Published worker stop request before restore runner start: %s', stop_requested)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        restore_log = os.path.join(RESTORE_LOG_DIR, f"restore_{timestamp}.log")
        os.makedirs(RESTORE_LOG_DIR, exist_ok=True)

        restore_cmd = [sys.executable, os.path.abspath(__file__), '--run-restore', restore_file, restore_log]
        proc = subprocess.Popen(
            restore_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        restore_pid = proc.pid
        logger.info("Restore started in detached process %s, log=%s", restore_pid, restore_log)

        return jsonify({
            'success': True,
            'message': 'Database restore started.',
            'restore_pid': restore_pid,
            'restore_log': restore_log,
        })
    except FileNotFoundError:
        logger.error("Python executable not found for restore runner")
        if restore_file and os.path.exists(restore_file):
            os.unlink(restore_file)
        return jsonify({'error': 'Python executable not found for restore runner.'}), 500
    except Exception:
        logger.exception("Restore launch failed")
        if restore_file and os.path.exists(restore_file):
            os.unlink(restore_file)
        if restore_log and os.path.exists(restore_log):
            os.unlink(restore_log)
        return jsonify({'error': 'Restore launch failed. Check server logs.'}), 500


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--run-restore':
        dump_path = sys.argv[2]
        log_path = sys.argv[3]
        sys.exit(_run_restore_runner(dump_path, log_path))
    else:
        print('This module is intended to be imported by the Flask app.')
