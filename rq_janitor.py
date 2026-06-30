import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app_helper import rq_queue_high, rq_queue_default
    from app_logging import configure_logging
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

configure_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("RQ Janitor process starting. Cleaning registries every 10 seconds.")
    queues_to_clean = [rq_queue_high, rq_queue_default]
    while True:
        try:
            for queue in queues_to_clean:
                started_registry = queue.started_job_registry
                started_before = started_registry.count
                started_registry.cleanup()
                started_after = started_registry.count
                started_cleaned = started_before - started_after
                if started_cleaned > 0:
                    logger.info(
                        "Janitor cleaned %d orphaned jobs from '%s' started_job_registry.",
                        started_cleaned,
                        queue.name,
                    )

                finished_registry = queue.finished_job_registry
                finished_before = finished_registry.count
                finished_registry.cleanup()
                finished_after = finished_registry.count
                finished_cleaned = finished_before - finished_after
                if finished_cleaned > 0:
                    logger.info(
                        "Janitor cleaned %d expired finished jobs from '%s' finished_job_registry.",
                        finished_cleaned,
                        queue.name,
                    )

                failed_registry = queue.failed_job_registry
                failed_before = failed_registry.count
                failed_registry.cleanup()
                failed_after = failed_registry.count
                failed_cleaned = failed_before - failed_after
                if failed_cleaned > 0:
                    logger.info(
                        "Janitor cleaned %d expired failed jobs from '%s' failed_job_registry.",
                        failed_cleaned,
                        queue.name,
                    )
        except Exception as e:
            logger.exception("Error in RQ Janitor loop: %s", e)

        time.sleep(10)
