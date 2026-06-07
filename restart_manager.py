import json
import logging
import os
import socket
import subprocess
import threading

from redis import Redis
import config

RESTART_CHANNEL = os.environ.get('AUDIO_MUSE_CONFIG_RESTART_CHANNEL', 'audiomuse:config_restart')
SUPERVISORCTL_CMD = os.environ.get('SUPERVISORCTL_CMD', '/usr/bin/supervisorctl')
SUPERVISOR_CONF = os.environ.get('SUPERVISOR_CONF', '/etc/supervisor/conf.d/supervisord.conf')
logger = logging.getLogger(__name__)

FLASK_SERVICE = ['flask']
WORKER_SERVICES = ['rq-worker-default', 'rq-worker-high', 'rq-janitor']
ALL_SUPERVISOR_SERVICES = FLASK_SERVICE + WORKER_SERVICES


def publish_control_request(action):
    """Publish a control request to worker containers via Redis."""
    try:
        redis_conn = Redis.from_url(
            config.REDIS_URL,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
            retry_on_timeout=True,
            decode_responses=True,
        )
        redis_conn.publish(RESTART_CHANNEL, action)
        return True
    except Exception as exc:
        logger.exception('Could not publish %s request to Redis: %s', action, exc)
        return False


def publish_restart_request():
    return publish_control_request('restart')


def publish_stop_request():
    return publish_control_request('stop')


def publish_start_request():
    return publish_control_request('start')


def _send_control(arguments):
    """Forward an ``[action, *services]`` request to the standalone supervisor.

    On macOS/Linux the supervisor listens on a unix socket
    (``AUDIOMUSE_CONTROL_SOCKET``).  On Windows it listens on TCP
    (``AUDIOMUSE_CONTROL_HOST`` / ``AUDIOMUSE_CONTROL_PORT``) because
    ``AF_UNIX`` is not available.  The JSON-line protocol is identical.
    """
    if not arguments:
        return False

    control_host = config.AUDIOMUSE_CONTROL_HOST
    control_port = config.AUDIOMUSE_CONTROL_PORT

    payload = json.dumps({'action': arguments[0], 'services': list(arguments[1:])}).encode('utf-8')
    try:
        if control_host and control_port:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(15)
                sock.connect((str(control_host), int(control_port)))
                sock.sendall(payload + b'\n')
                response = sock.recv(1024).strip()
        elif config.AUDIOMUSE_CONTROL_SOCKET:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(15)
                sock.connect(config.AUDIOMUSE_CONTROL_SOCKET)
                sock.sendall(payload + b'\n')
                response = sock.recv(1024).strip()
        else:
            logger.error('Neither AUDIOMUSE_CONTROL_SOCKET nor AUDIOMUSE_CONTROL_HOST/PORT set; cannot dispatch %s', arguments)
            return False
    except Exception:
        target = f'{control_host}:{control_port}' if (control_host and control_port) else config.AUDIOMUSE_CONTROL_SOCKET
        logger.exception('Failed to send control command %s to %s', arguments, target)
        return False
    if response == b'ok':
        logger.info('Control command succeeded: %s', arguments)
        return True
    logger.error('Control server rejected %s: %s', arguments, response)
    return False


def _run_supervisorctl(arguments):
    if config.AUDIOMUSE_PLATFORM == 'macos':
        return _send_control(arguments)
    cmd = [SUPERVISORCTL_CMD, '-c', SUPERVISOR_CONF] + arguments
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            logger.error('supervisorctl failed (%s): %s', result.returncode, stderr or stdout)
            return False
        logger.info('supervisorctl succeeded: %s', stdout)
        return True
    except FileNotFoundError:
        logger.exception('supervisorctl command not found at %s', SUPERVISORCTL_CMD)
        return False
    except Exception:
        logger.exception('Failed to run supervisorctl command: %s', cmd)
        return False

def stop_local_flask_service():
    """Stop the supervised Flask service locally."""
    logger.info('Stopping supervised Flask service')
    return _run_supervisorctl(['stop'] + FLASK_SERVICE)


def start_local_flask_service():
    """Start the supervised Flask service locally."""
    logger.info('Starting supervised Flask service')
    return _run_supervisorctl(['start'] + FLASK_SERVICE)


def stop_supervisor_workers():
    """Stop supervised worker processes via supervisorctl."""
    logger.info('Stopping supervised worker services: %s', WORKER_SERVICES)
    return _run_supervisorctl(['stop'] + WORKER_SERVICES)


def start_supervisor_workers():
    """Start supervised worker processes via supervisorctl."""
    logger.info('Starting supervised worker services: %s', WORKER_SERVICES)
    return _run_supervisorctl(['start'] + WORKER_SERVICES)


def _spawn_supervisorctl(arguments):
    if config.AUDIOMUSE_PLATFORM == 'macos':
        return _send_control(arguments)
    cmd = [SUPERVISORCTL_CMD, '-c', SUPERVISOR_CONF] + arguments
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info('Spawned detached supervisorctl: %s', ' '.join(cmd))
        return True
    except FileNotFoundError:
        logger.exception('supervisorctl command not found at %s', SUPERVISORCTL_CMD)
        return False
    except Exception:
        logger.exception('Failed to spawn supervisorctl command: %s', cmd)
        return False


def _restart_flask_program():
    logger.info('Restarting supervised Flask program via supervisorctl')
    return _spawn_supervisorctl(['restart', 'flask'])


def schedule_flask_restart(delay_seconds=2.5):
    """Schedule a Flask container restart after the current response completes."""
    if os.environ.get('SERVICE_TYPE', '').lower() != 'flask':
        return False

    if os.environ.get('DISABLE_FLASK_RESTART', 'false').lower() == 'true':
        return False

    timer = threading.Timer(delay_seconds, _restart_flask_program)
    timer.daemon = True
    timer.start()
    return True


def restart_supervisor_workers():
    """Restart supervised worker programs inside the current worker container."""
    if os.environ.get('SERVICE_TYPE', '').lower() != 'worker':
        logger.info('SERVICE_TYPE is not worker; skipping supervised worker restart')
        return True

    logger.info('Restarting supervised worker programs via supervisorctl')
    return _run_supervisorctl(['restart', 'rq-worker-default', 'rq-worker-high', 'rq-janitor'])
