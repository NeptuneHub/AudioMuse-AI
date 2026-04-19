import os
import subprocess
import logging
import tempfile
from datetime import datetime
from flask import Blueprint, render_template, jsonify, request, send_file, after_this_request
from config import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB
import restart_manager

logger = logging.getLogger(__name__)

backup_bp = Blueprint('backup_bp', __name__)

BACKUP_DIR = os.environ.get("BACKUP_DIR", "/app/backup")


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

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sql')
    try:
        uploaded.save(tmp)
        tmp.close()

        env = _pg_env()

        # First: drop ALL tables in the public schema so the restore starts clean
        drop_sql = (
            "DO $$ DECLARE r RECORD; BEGIN "
            "FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP "
            "EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE'; "
            "END LOOP; END $$;"
        )
        drop_cmd = _pg_cmd('psql', '-d', POSTGRES_DB, '-c', drop_sql)
        drop_result = subprocess.run(drop_cmd, env=env, capture_output=True, text=True, timeout=60)
        if drop_result.returncode != 0:
            logger.error("Failed to drop tables before restore: %s", drop_result.stderr)
            return jsonify({'error': f'Failed to clean database: {drop_result.stderr}'}), 500
        logger.info("All existing tables dropped before restore.")

        # Then: restore from the dump
        restore_cmd = _pg_cmd('psql', '-d', POSTGRES_DB, '-f', tmp.name)
        result = subprocess.run(restore_cmd, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("psql restore failed: %s", result.stderr)
            return jsonify({'error': f'psql restore failed: {result.stderr}'}), 500

        logger.info("Database restored from uploaded file: %s", uploaded.filename)

        # Trigger the same automatic restart flow used by the setup page so
        # the user lands on a freshly-rebooted Flask (and workers) after the
        # restore — the schema the code expects may have changed.
        restart_requested = False
        try:
            restart_manager.publish_restart_request()
            restart_requested = True
        except Exception as e:
            logger.warning("restore: publish_restart_request failed: %s", e)

        @after_this_request
        def _schedule_restart(response):
            if restart_requested:
                try:
                    restart_manager.schedule_flask_restart()
                except Exception as e:
                    logger.warning("restore: schedule_flask_restart failed: %s", e)
            return response

        return jsonify({
            'success': True,
            'message': 'Database restored successfully.',
            'restart_requested': restart_requested,
        })
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
