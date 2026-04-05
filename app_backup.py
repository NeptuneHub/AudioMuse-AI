import os
import subprocess
import logging
from datetime import datetime
from flask import Blueprint, render_template, jsonify, request, send_file
from config import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB

logger = logging.getLogger(__name__)

backup_bp = Blueprint('backup_bp', __name__)

BACKUP_DIR = os.environ.get("BACKUP_DIR", "/app/backup")


@backup_bp.route('/backup')
def backup_page():
    return render_template('backup.html', title='Backup & Restore', active='backup')


@backup_bp.route('/api/backup/create', methods=['POST'])
def create_backup():
    """Dump the entire PostgreSQL database and return the file for download."""
    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"audiomuse_backup_{timestamp}.sql"
    filepath = os.path.join(BACKUP_DIR, filename)

    env = os.environ.copy()
    env['PGPASSWORD'] = POSTGRES_PASSWORD

    cmd = [
        'pg_dump',
        '-h', POSTGRES_HOST,
        '-p', POSTGRES_PORT,
        '-U', POSTGRES_USER,
        '-d', POSTGRES_DB,
        '-F', 'c',          # custom format (compressed, supports pg_restore)
        '-f', filepath,
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("pg_dump failed: %s", result.stderr)
            return jsonify({'error': f'pg_dump failed: {result.stderr}'}), 500
    except FileNotFoundError:
        logger.error("pg_dump not found on system PATH")
        return jsonify({'error': 'pg_dump is not installed or not on PATH'}), 500
    except subprocess.TimeoutExpired:
        logger.error("pg_dump timed out")
        return jsonify({'error': 'pg_dump timed out after 600 seconds'}), 500

    logger.info("Backup created: %s", filepath)
    return send_file(filepath, as_attachment=True, download_name=filename)


@backup_bp.route('/api/backup/list', methods=['GET'])
def list_backups():
    """Return list of existing backup files."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    files = []
    for f in sorted(os.listdir(BACKUP_DIR), reverse=True):
        if f.startswith('audiomuse_backup_') and f.endswith('.sql'):
            full = os.path.join(BACKUP_DIR, f)
            stat = os.stat(full)
            files.append({
                'filename': f,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            })
    return jsonify(files)


@backup_bp.route('/api/backup/restore', methods=['POST'])
def restore_backup():
    """Restore the database from a selected backup file."""
    data = request.get_json(silent=True) or {}
    filename = data.get('filename', '')
    confirmation = data.get('confirmation', '')

    expected = "I want to restore the database from the backup. This action is not reversible"
    if confirmation != expected:
        return jsonify({'error': 'Confirmation text does not match.'}), 400

    # Validate filename: must be a known backup file, no path traversal
    if not filename or '/' in filename or '\\' in filename or '..' in filename:
        return jsonify({'error': 'Invalid filename.'}), 400

    filepath = os.path.join(BACKUP_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({'error': 'Backup file not found.'}), 404

    env = os.environ.copy()
    env['PGPASSWORD'] = POSTGRES_PASSWORD

    # Drop and recreate all objects, then restore
    cmd = [
        'pg_restore',
        '-h', POSTGRES_HOST,
        '-p', POSTGRES_PORT,
        '-U', POSTGRES_USER,
        '-d', POSTGRES_DB,
        '--clean',           # drop existing objects before restore
        '--if-exists',       # don't error if objects don't exist yet
        filepath,
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        # pg_restore may return warnings on --clean --if-exists; only treat returncode > 1 as fatal
        if result.returncode > 1:
            logger.error("pg_restore failed: %s", result.stderr)
            return jsonify({'error': f'pg_restore failed: {result.stderr}'}), 500
    except FileNotFoundError:
        logger.error("pg_restore not found on system PATH")
        return jsonify({'error': 'pg_restore is not installed or not on PATH'}), 500
    except subprocess.TimeoutExpired:
        logger.error("pg_restore timed out")
        return jsonify({'error': 'pg_restore timed out after 600 seconds'}), 500

    logger.info("Database restored from: %s", filepath)
    return jsonify({'success': True, 'message': f'Database restored from {filename}'})
