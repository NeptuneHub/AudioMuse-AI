# /home/guido/Music/AudioMuse-AI/rq_worker_high_priority.py
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Signal to app.py that we are an RQ worker, so it should skip index loading and background threads
os.environ['AUDIOMUSE_ROLE'] = 'worker'

# Cap thread pools used by ML/numeric libraries (numpy/OpenBLAS/MKL/OMP) BEFORE
# any of them get imported. This worker is the LIGHT scheduler ('high' queue),
# but it still imports numpy via app_helper and forks children per job — without
# this cap libgomp/OpenBLAS default to all CPUs and forked children spin
# libgomp wait-loops at 100% CPU. Use cpu_count // 3 with floor 1.
_cpu_count = os.cpu_count() or 1
_max_threads = max(1, _cpu_count // 3)
for _env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_env_key] = str(_max_threads)
print(f"High-priority worker CPU thread cap = {_max_threads} (cpu_count // 3, min 1)")

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