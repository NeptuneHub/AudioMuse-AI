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

        restore_cmd = _pg_cmd(
            'psql',
            '-d', POSTGRES_DB,
            '-v', 'ON_ERROR_STOP=1',
            '--single-transaction',
            '-c', 'DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;',
            '-f', dump_file,
        )
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
    """Restore the database from an uploaded .sql dump file via psql.

    Supports both:
    - Full file upload (chunk_num=None)
    - Chunked upload (chunk_num=X, total_chunks=Y) - each chunk saved, reassembled when all received
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

    try:
        if chunk_num and total_chunks:
            # Chunked upload mode
            try:
                chunk_num = int(chunk_num)
                total_chunks = int(total_chunks)
            except ValueError:
                return jsonify({'error': 'chunk_num and total_chunks must be integers.'}), 400

            if chunk_num < 1 or chunk_num > total_chunks or total_chunks < 1:
                return jsonify({'error': f'Invalid chunk numbers: chunk_num={chunk_num}, total_chunks={total_chunks}'}), 400

            chunks_dir = os.path.join(BACKUP_DIR, 'chunks')
            os.makedirs(chunks_dir, exist_ok=True)

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
                        except Exception as exc:
                            logger.warning(f"Could not delete leftover chunk {f}: {exc}")

            # Save the current chunk
            try:
                uploaded.save(chunk_file)
                logger.info(f"Saved chunk {chunk_num}/{total_chunks}")
            except Exception:
                logger.exception("Failed to save chunk %s", chunk_num)
                return jsonify({'error': f'Failed to save chunk {chunk_num}.'}), 500

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
            if len(received_chunks) == total_chunks and all(i in received_chunks for i in range(1, total_chunks + 1)):
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
                            raise Exception(f"Error reading chunk {i}: {str(e)}")

                    tmp.close()
                    file_size = os.path.getsize(restore_file)
                    logger.info(f"Reassembly complete: {restore_file} ({file_size} bytes)")

                    # Clean up chunk files
                    for i in range(1, total_chunks + 1):
                        try:
                            os.unlink(os.path.join(chunks_dir, f'backup_{i}_of_{total_chunks}.sql'))
                        except Exception as e:
                            logger.warning(f"Could not delete chunk {i}: {e}")

                    # Start restore with reassembled file
                    all_chunks_received = True
                except Exception as e:
                    logger.exception("Failed to reassemble uploaded backup chunks")
                    if tmp:
                        try:
                            tmp.close()
                        except:
                            pass
                    if restore_file and os.path.exists(restore_file):
                        os.unlink(restore_file)
                    return jsonify({'error': 'Failed to reassemble chunks due to an internal error.'}), 500
            else:
                # Still waiting for more chunks
                missing_chunks = [i for i in range(1, total_chunks + 1) if i not in received_chunks]
                return jsonify({
                    'success': True,
                    'message': f'Chunk {chunk_num}/{total_chunks} received. Waiting for chunks: {missing_chunks}',
                    'chunk_num': chunk_num,
                    'total_chunks': total_chunks,
                    'received_chunks': sorted(received_chunks),
                    'missing_chunks': missing_chunks,
                    'all_chunks_received': False,
                })
        else:
            # Single file upload (non-chunked)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sql')
            uploaded.save(tmp)
            tmp.close()
            restore_file = tmp.name
            all_chunks_received = True

        # Start restore only if all chunks received or single file upload
        if restore_file and all_chunks_received:
            stop_requested = restart_manager.publish_stop_request()
            logger.info('Published worker stop request: %s', stop_requested)

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
            logger.info("Restore started in detached process %s", restore_pid)

            return jsonify({
                'success': True,
                'message': 'Database restore started.',
                'restore_pid': restore_pid,
                'restore_log': restore_log,
                'all_chunks_received': True,
            })

    except FileNotFoundError:
        logger.error("Python executable not found for restore runner")
        if restore_file and os.path.exists(restore_file):
            os.unlink(restore_file)
        return jsonify({'error': 'Python executable not found for restore runner.'}), 500
    except Exception:
        logger.exception("Restore failed")
        if restore_file and os.path.exists(restore_file):
            os.unlink(restore_file)
        return jsonify({'error': 'Restore failed. Check server logs.'}), 500


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--run-restore':
        dump_path = sys.argv[2]
        log_path = sys.argv[3]
        sys.exit(_run_restore_runner(dump_path, log_path))
    else:
        print('This module is intended to be imported by the Flask app.')
