import logging
import os
import signal
import subprocess
import threading
import time

from redis import Redis
import config

RESTART_CHANNEL = os.environ.get('AUDIO_MUSE_CONFIG_RESTART_CHANNEL', 'audiomuse:config_restart')
WORKER_RESTART_PATTERNS = [
    '/app/rq_worker.py',
    '/app/rq_worker_high_priority.py',
    '/app/rq_janitor.py',
]
logger = logging.getLogger(__name__)


def publish_restart_request():
    """Publish a restart request to worker containers via Redis."""
    try:
        redis_conn = Redis.from_url(
            config.REDIS_URL,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
            retry_on_timeout=True,
            decode_responses=True,
        )
        redis_conn.publish(RESTART_CHANNEL, 'restart')
        return True
    except Exception as exc:
        logger.exception('Could not publish restart request to Redis: %s', exc)
        return False


FLASK_RESTART_PATTERNS = [
    'gunicorn',
    'app:app',
]


def _find_flask_pids():
    pids = set()
    for pattern in FLASK_RESTART_PATTERNS:
        try:
            result = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
            if result.returncode != 0:
                continue
            for line in result.stdout.splitlines():
                try:
                    pid = int(line.strip())
                except ValueError:
                    continue
                if pid != os.getpid():
                    pids.add(pid)
        except FileNotFoundError:
            logger.warning('pgrep not available in this environment; cannot find Flask process PIDs for pattern: %s', pattern)
            return set()
        except Exception:
            logger.exception('Error while finding Flask processes for pattern: %s', pattern)
    return pids


def _kill_pids(pids, sig):
    for pid in sorted(pids):
        try:
            os.kill(pid, sig)
            logger.info('Sent %s to PID %s', sig.name, pid)
        except ProcessLookupError:
            logger.warning('PID %s disappeared before signal could be sent', pid)
        except Exception:
            logger.exception('Failed to signal PID %s', pid)


def _terminate_flask_processes():
    pids = _find_flask_pids()
    if not pids:
        parent_pid = os.getppid()
        if parent_pid and parent_pid != 1:
            logger.info('No Flask processes found by pattern; terminating parent PID %s', parent_pid)
            try:
                os.kill(parent_pid, signal.SIGKILL)
                return True
            except Exception as exc:
                logger.exception('Failed to terminate Flask parent PID %s: %s', parent_pid, exc)
                return False
        logger.info('No Flask process candidates found; terminating current PID %s', os.getpid())
        try:
            os.kill(os.getpid(), signal.SIGKILL)
            return True
        except Exception as exc:
            logger.exception('Failed to terminate Flask current PID %s: %s', os.getpid(), exc)
            return False

    logger.info('Terminating Flask PIDs: %s', sorted(pids))
    _kill_pids(pids, signal.SIGTERM)
    time.sleep(0.2)
    remaining = {pid for pid in pids if os.path.exists(f'/proc/{pid}')}
    if remaining:
        logger.warning('Flask PIDs still alive after SIGTERM: %s; sending SIGKILL', sorted(remaining))
        _kill_pids(remaining, signal.SIGKILL)
    return True


def schedule_flask_restart(delay_seconds=1.0):
    """Schedule a Flask container restart after the current response completes.

    Initial setup can take longer to settle, so use a longer delay to avoid
    killing the running process before the response is fully delivered.
    """
    if os.environ.get('SERVICE_TYPE', '').lower() != 'flask':
        return False

    if os.environ.get('DISABLE_FLASK_RESTART', 'false').lower() == 'true':
        return False

    timer = threading.Timer(delay_seconds, _terminate_flask_processes)
    timer.daemon = True
    timer.start()
    return True


def _find_worker_pids():
    pids = set()
    for pattern in WORKER_RESTART_PATTERNS:
        try:
            result = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
            if result.returncode != 0:
                continue
            for line in result.stdout.splitlines():
                try:
                    pid = int(line.strip())
                except ValueError:
                    continue
                if pid != os.getpid():
                    pids.add(pid)
        except FileNotFoundError:
            logger.warning('pgrep not available in this environment; cannot find worker process PIDs for pattern: %s', pattern)
            return set()
        except Exception:
            logger.exception('Error while finding processes for pattern: %s', pattern)
    return pids


def _terminate_worker_processes():
    pids = _find_worker_pids()
    if not pids:
        logger.warning('No supervised worker processes found to terminate.')
        return False

    logger.info('Terminating worker PIDs: %s', sorted(pids))
    success = False
    for pid in sorted(pids):
        try:
            os.kill(pid, signal.SIGTERM)
            success = True
        except ProcessLookupError:
            logger.warning('Worker PID %s disappeared before termination', pid)
        except Exception:
            logger.exception('Failed to terminate worker PID %s', pid)
    return success


def restart_supervisor_workers():
    """Restart supervised worker programs inside the current worker container."""
    return _terminate_worker_processes()
