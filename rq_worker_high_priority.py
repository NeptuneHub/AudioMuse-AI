# /home/guido/Music/AudioMuse-AI/rq_worker_high_priority.py
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rq import Worker

try:
    from app_helper import redis_conn
    from config import APP_VERSION
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

    logging_level = os.getenv("RQ_LOGGING_LEVEL", "INFO").upper()
    print(f"RQ Worker logging level set to: {logging_level}")

    try:
        # The job function itself is responsible for creating an app context if needed.
        worker.work(logging_level=logging_level)
    except Exception as e:
        print(f"High Priority RQ Worker failed to start or encountered an error: {e}")
        sys.exit(1)