# /home/guido/Music/AudioMuse-AI/rq_worker_high_priority.py
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Signal to app.py that we are an RQ worker, so it should skip index loading and background threads
os.environ['AUDIOMUSE_ROLE'] = 'worker'

from rq import Worker

try:
    from app_helper import redis_conn
    from config import APP_VERSION
    # Provider migration tool: this worker deliberately does NOT import app.py,
    # but still needs to pick up any runtime active-provider override written by
    # the migration tool. Push a minimal Flask app context so get_db() works,
    # then apply the override and start the pub/sub listener.
    from flask import Flask as _Flask
    _mig_app = _Flask(__name__)
    with _mig_app.app_context():
        from app_provider_migration import apply_provider_overrides_from_db, subscribe_to_provider_migrated_channel
        apply_provider_overrides_from_db()
        subscribe_to_provider_migrated_channel()
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

# This worker ONLY listens to the 'high' queue.
queues_to_listen = ['high']

if __name__ == '__main__':
    print(f"🚀 DEDICATED HIGH PRIORITY RQ Worker starting. Version: {APP_VERSION}. Listening ONLY on queues: {queues_to_listen}")
    print(f"Using Redis connection: {redis_conn.connection_pool.connection_kwargs}")

    # High priority worker doesn't analyze songs, so no CLAP preload needed
    # Only rq_worker.py (default queue) handles song analysis tasks

    worker = Worker(
        queues_to_listen,
        connection=redis_conn,
        # --- Resilience Settings for Kubernetes ---
        worker_ttl=30,  # Consider worker dead if no heartbeat for 30 seconds.
        job_monitoring_interval=10 # Check for dead workers every 10 seconds.
    )

    # Memory leak prevention: restart after N jobs
    # Higher than default worker since this doesn't load CLAP model
    max_jobs_before_restart = int(os.getenv('RQ_MAX_JOBS_HIGH', '100'))

    logging_level = os.getenv("RQ_LOGGING_LEVEL", "INFO").upper()
    print(f"RQ Worker logging level set to: {logging_level}")
    print(f"Worker will restart after {max_jobs_before_restart} jobs to prevent memory leaks")

    try:
        # The job function itself is responsible for creating an app context if needed.
        worker.work(logging_level=logging_level, max_jobs=max_jobs_before_restart)
    except Exception as e:
        print(f"High Priority RQ Worker failed to start or encountered an error: {e}")
        sys.exit(1)