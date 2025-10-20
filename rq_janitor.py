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
                # The StartedJobRegistry is where orphaned jobs from dead workers live.
                # Cleaning this registry finds workers that have not sent a heartbeat
                # within their TTL and moves their jobs back to the queue or to failed.
                # This is the primary mechanism for recovering from unclean shutdowns.

                # The .cleanup() method in many RQ versions does not return a count.
                # To log the count, we check the size before and after.
                registry = queue.started_job_registry
                count_before = registry.count

                registry.cleanup() # This is the important part

                count_after = registry.count
                cleaned_count = count_before - count_after

                if cleaned_count > 0:
                    logging.info("Janitor cleaned %d orphaned jobs from the '%s' queue's started_job_registry.", cleaned_count, queue.name)
        except Exception as e:
            logging.error("Error in RQ Janitor loop: %s", e, exc_info=True)
        
        # Sleep for the desired monitoring interval.
        time.sleep(10)