# /home/guido/Music/AudioMuse-AI/rq_janitor.py
import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # We need the queue objects to get their registries
    from app_helper import redis_conn, rq_queue_high, rq_queue_default
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]- %(message)s')

if __name__ == '__main__':
    logging.info("🧹 RQ Janitor process starting. Cleaning registries every 10 seconds.")
    queues_to_clean = [rq_queue_high, rq_queue_default]
    while True:
        try:
            for queue in queues_to_clean:
                # 1. Clean StartedJobRegistry - orphaned jobs from dead workers
                started_registry = queue.started_job_registry
                started_before = started_registry.count
                started_registry.cleanup()
                started_after = started_registry.count
                started_cleaned = started_before - started_after
                if started_cleaned > 0:
                    logging.info("Janitor cleaned %d orphaned jobs from '%s' started_job_registry.", started_cleaned, queue.name)
                
                # 2. Clean FinishedJobRegistry - completed jobs older than TTL (default 500s)
                # CRITICAL: This prevents memory/thread leaks from accumulated finished jobs
                finished_registry = queue.finished_job_registry
                finished_before = finished_registry.count
                finished_registry.cleanup()
                finished_after = finished_registry.count
                finished_cleaned = finished_before - finished_after
                if finished_cleaned > 0:
                    logging.info("Janitor cleaned %d expired finished jobs from '%s' finished_job_registry.", finished_cleaned, queue.name)
                
                # 3. Clean FailedJobRegistry - failed jobs older than TTL
                failed_registry = queue.failed_job_registry
                failed_before = failed_registry.count
                failed_registry.cleanup()
                failed_after = failed_registry.count
                failed_cleaned = failed_before - failed_after
                if failed_cleaned > 0:
                    logging.info("Janitor cleaned %d expired failed jobs from '%s' failed_job_registry.", failed_cleaned, queue.name)
        except Exception as e:
            logging.error("Error in RQ Janitor loop: %s", e, exc_info=True)
        
        # Sleep for the desired monitoring interval.
        time.sleep(10)