import os
import subprocess
import logging
import tempfile
from datetime import datetime
from flask import Blueprint, render_template, jsonify, request, send_file
from config import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB

logger = logging.getLogger(__name__)

backup_bp = Blueprint('backup_bp', __name__)

BACKUP_DIR = os.environ.get("BACKUP_DIR", "/app/backup")


def _pg_env():
    """Return a copy of os.environ with PGPASSWORD set."""
    env = os.environ.copy()
    env['PGPASSWORD'] = POSTGRES_PASSWORD
    return env


@backup_bp.route('/backup')
def backup_page():
    return render_template('backup.html', title='Backup & Restore', active='backup')


@backup_bp.route('/api/backup/create', methods=['POST'])
def create_backup():
    """Full pg_dumpall and return the .sql file for download."""
    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"audiomuse_backup_{timestamp}.sql"
    filepath = os.path.join(BACKUP_DIR, filename)

    cmd = [
        'pg_dumpall', '-c',
        '-h', POSTGRES_HOST,
        '-p', POSTGRES_PORT,
        '-U', POSTGRES_USER,
    ]

    try:
        with open(filepath, 'w') as f:
            result = subprocess.run(cmd, env=_pg_env(), stdout=f, stderr=subprocess.PIPE, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("pg_dumpall failed: %s", result.stderr)
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'pg_dumpall failed: {result.stderr}'}), 500
    except FileNotFoundError:
        logger.error("pg_dumpall not found on system PATH")
        return jsonify({'error': 'pg_dumpall is not installed or not on PATH'}), 500
    except subprocess.TimeoutExpired:
        logger.error("pg_dumpall timed out")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'pg_dumpall timed out after 600 seconds'}), 500

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

    # Save the uploaded file to a temporary location
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sql')
    try:
        uploaded.save(tmp)
        tmp.close()

        cmd = [
            'psql',
            '-h', POSTGRES_HOST,
            '-p', POSTGRES_PORT,
            '-U', POSTGRES_USER,
            '-d', POSTGRES_DB,
            '-f', tmp.name,
        ]

        result = subprocess.run(cmd, env=_pg_env(), capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("psql restore failed: %s", result.stderr)
            return jsonify({'error': f'psql restore failed: {result.stderr}'}), 500

        logger.info("Database restored from uploaded file: %s", uploaded.filename)
        return jsonify({'success': True, 'message': 'Database restored successfully.'})
    except FileNotFoundError:
        logger.error("psql not found on system PATH")
        return jsonify({'error': 'psql is not installed or not on PATH'}), 500
    except subprocess.TimeoutExpired:
        logger.error("psql restore timed out")
        return jsonify({'error': 'psql restore timed out after 600 seconds'}), 500
    except Exception:
        logger.exception("Restore failed")
        return jsonify({'error': 'Restore failed. Check server logs.'}), 500
    finally:
        os.unlink(tmp.name)
