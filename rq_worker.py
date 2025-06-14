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
    from app import app, redis_conn, rq_queue
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

# The queues the worker will listen on.
# We use rq_queue directly, which is already configured with redis_conn.
# You can add more queues here if you define them in app.py, e.g., high_priority_queue
queues_to_listen = [rq_queue]

if __name__ == '__main__':
    # The redis_conn is already initialized when imported from app.py.
    # The queues_to_listen are already configured with this connection.

    print(f"RQ Worker starting. Listening on queues: {[q.name for q in queues_to_listen]}")
    print(f"Using Redis connection: {redis_conn.connection_pool.connection_kwargs}")

    # Create a worker instance, explicitly passing the connection.
    # The 'app' object is passed to `with app.app_context():` within the tasks themselves
    # if they need it. RQ's default job execution doesn't automatically push an app context.
    # Tasks should be designed to handle this, e.g., by calling `with app.app_context():`
    # or by using functions from app.py that manage their own context.
    worker = Worker(
        queues_to_listen,
        connection=redis_conn
        # default_timeout=3600, # Optional: set a default timeout for jobs
        # log_job_description=True # Optional: logs job descriptions
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
