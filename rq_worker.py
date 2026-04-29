# /home/guido/Music/AudioMuse-AI/rq_worker.py
import os
import sys

# Ensure the /app directory (where app.py and tasks.py are) is in the Python path
# This is important if rq_worker.py is in the root and app.py/tasks.py are in /app
# In your Docker setup, PYTHONPATH already includes /app, but this is good for local dev too.
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Adds the current directory
# If app.py is in a subdirectory like 'app_module' relative to rq_worker.py, you'd adjust:
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app_module'))

# Signal to app.py that we are an RQ worker, so it should skip index loading and background threads
os.environ['AUDIOMUSE_ROLE'] = 'worker'

# Limit CPU threads for worker-local model inference to half the available cores.
# This caps Whisper / LLaMA / transformer / BLAS thread usage and avoids runaway CPU saturation.
_cpu_count = os.cpu_count() or 1
_max_lyrics_threads = max(1, _cpu_count // 2)
for _env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_env_key] = str(_max_lyrics_threads)
print(f"Worker CPU thread cap set to {_max_lyrics_threads} threads for CPU-bound libraries")

# Import Worker from rq
from rq import Worker

# Import the redis_conn, rq_queue (which is the 'default' queue),
# and the Flask app instance from your main app.py.
# This ensures the worker uses the same Redis connection, queue configuration,
# and application context as your Flask app.
try:
    # Import the specific queues we defined
    from app import app
    from app_helper import redis_conn
    import config
    from config import APP_VERSION
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

# The queues the worker will listen on.
# The order is important! Workers will always check 'high' before 'default'.
queues_to_listen = ['default']

# NOTE: Do NOT preload Whisper / transformers / llama_cpp here in the parent
# process. RQ uses os.fork() to spawn each job's child process. PyTorch and
# OpenMP (libgomp / libomp) are NOT fork-safe: any thread pool initialized in
# the parent becomes corrupted in the child and the first call into the model
# deadlocks at 0% CPU. Models are lazy-loaded inside the child on first use
# via the module-level caches in lyrics.lyrics_transcriber (so jobs 2..N in
# the same child are free).


if __name__ == '__main__':
    # The redis_conn is already initialized when imported from app.py.
    # The queues_to_listen are already configured with this connection.

    # Use the list of names directly for the log message
    print(f"DEFAULT RQ Worker starting. Version: {APP_VERSION}. Listening on queues: {queues_to_listen}")
    print(f"Using Redis connection: {redis_conn.connection_pool.connection_kwargs}")

    # Create a worker instance, explicitly passing the connection.
    # The 'app' object is passed to `with app.app_context():` within the tasks themselves
    # if they need it. RQ's default job execution doesn't automatically push an app context.
    # Tasks should be designed to handle this, e.g., by calling `with app.app_context():`
    # or by using functions from app.py that manage their own context.
    worker = Worker(
        queues_to_listen,
        connection=redis_conn,
        # --- Resilience Settings for Kubernetes ---
        worker_ttl=120,  # Consider worker dead if no heartbeat for 120 seconds.
        job_monitoring_interval=30 # Check for dead workers every 30 seconds.
    )

    # Memory leak prevention: restart after N jobs
    # RQ will automatically respawn via supervisord
    # Balance: High enough to avoid frequent CLAP reloads, low enough to prevent memory leaks
    max_jobs_before_restart = int(os.getenv('RQ_MAX_JOBS', '50'))

    # Start the worker.
    # You can set logging_level for more verbose output.
    # Common levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging_level = os.getenv("RQ_LOGGING_LEVEL", "INFO").upper()
    print(f"RQ Worker logging level set to: {logging_level}")
    print(f"Worker will restart after {max_jobs_before_restart} jobs to prevent memory leaks")

    try:
        # The `with app.app_context():` here is generally NOT how RQ workers are run.
        # RQ jobs are executed in separate processes. If a job needs app context,
        # the job function itself should establish it.
        # However, if there's any setup *for the worker process itself* that needs app context,
        # it could be done here, but it's uncommon.
        # For tasks needing app context (like DB access), they should handle it internally:
        #
        # In tasks.py:
        # from app import app, get_db
        # def my_task():
        #     with app.app_context():
        #         db = get_db()
        #         # ... do work ...

        worker.work(logging_level=logging_level, max_jobs=max_jobs_before_restart)
    except Exception as e:
        print(f"RQ Worker failed to start or encountered an error: {e}")
        sys.exit(1)
