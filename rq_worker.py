# /home/guido/Music/AudioMuse-AI/rq_worker.py
import os
import sys

# Ensure the /app directory (where app.py and tasks.py are) is in the Python path
# This is important if rq_worker.py is in the root and app.py/tasks.py are in /app
# In your Docker setup, PYTHONPATH already includes /app, but this is good for local dev too.
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Adds the current directory
# If app.py is in a subdirectory like 'app_module' relative to rq_worker.py, you'd adjust:
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app_module'))

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
    from config import APP_VERSION
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

# The queues the worker will listen on.
# The order is important! Workers will always check 'high' before 'default'.
queues_to_listen = ['default']

if __name__ == '__main__':
    # The redis_conn is already initialized when imported from app.py.
    # The queues_to_listen are already configured with this connection.

    # Use the list of names directly for the log message
    print(f"DEFAULT RQ Worker starting. Version: {APP_VERSION}. Listening on queues: {queues_to_listen}")
    print(f"Using Redis connection: {redis_conn.connection_pool.connection_kwargs}")

    # Preload CLAP model to avoid loading delays on first text search
    try:
        print("Preloading CLAP model for this worker...")
        from tasks.clap_analyzer import initialize_clap_model
        initialize_clap_model()
        print("✓ CLAP model preloaded successfully")
    except Exception as e:
        print(f"⚠ CLAP model preload failed, will retry on first use: {e}")

    # Create a worker instance, explicitly passing the connection.
    # The 'app' object is passed to `with app.app_context():` within the tasks themselves
    # if they need it. RQ's default job execution doesn't automatically push an app context.
    # Tasks should be designed to handle this, e.g., by calling `with app.app_context():`
    # or by using functions from app.py that manage their own context.
    worker = Worker(
        queues_to_listen,
        connection=redis_conn,
        # --- Resilience Settings for Kubernetes ---
        worker_ttl=30,  # Consider worker dead if no heartbeat for 30 seconds.
        job_monitoring_interval=10 # Check for dead workers every 10 seconds.
    )

    # Start the worker.
    # You can set logging_level for more verbose output.
    # Common levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging_level = os.getenv("RQ_LOGGING_LEVEL", "INFO").upper()
    print(f"RQ Worker logging level set to: {logging_level}")

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

        worker.work(logging_level=logging_level)
    except Exception as e:
        print(f"RQ Worker failed to start or encountered an error: {e}")
        sys.exit(1)
