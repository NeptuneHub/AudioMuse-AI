# /home/guido/Music/AudioMuse-AI/rq_janitor.py
import os
import sys
import time
import logging
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # We need the queue objects to get their registries
    from app_helper import redis_conn, rq_queue_high, rq_queue_default, get_db
    from app_helper import TASK_STATUS_PENDING, TASK_STATUS_STARTED, TASK_STATUS_REVOKED
except ImportError as e:
    print(f"Error importing from app.py: {e}")
    print("Please ensure app.py is in the Python path and does not have top-level errors.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]- %(message)s')

# Threshold for considering a task stale (5 minutes in seconds)
STALE_TASK_THRESHOLD_SECONDS = 300

def clean_stale_database_tasks():
    """
    Clean up stale PENDING or STARTED tasks in the database.
    A task is considered stale if it's been in PENDING or STARTED state for more than STALE_TASK_THRESHOLD_SECONDS.
    """
    try:
        from flask import Flask
        from app import app

        with app.app_context():
            db = get_db()
            cur = db.cursor()

            current_time = time.time()
            threshold_time = current_time - STALE_TASK_THRESHOLD_SECONDS

            # Find main tasks (main_analysis, main_clustering) that are stale
            # Only clean up tasks that start with 'main_' to avoid affecting other tasks like fetch_playlists
            cur.execute("""
                SELECT task_id, task_type, status, start_time, timestamp
                FROM task_status
                WHERE parent_task_id IS NULL
                  AND task_type LIKE %s
                  AND status IN (%s, %s)
                  AND start_time IS NOT NULL
                  AND start_time < %s
            """, ('main_%', TASK_STATUS_PENDING, TASK_STATUS_STARTED, threshold_time))

            stale_tasks = cur.fetchall()

            if stale_tasks:
                for task_row in stale_tasks:
                    task_id, task_type, status, start_time, timestamp = task_row
                    age_seconds = int(current_time - start_time) if start_time else 0

                    # Mark as REVOKED
                    revoked_details = {
                        "message": f"Task revoked by janitor: stuck in {status} for {age_seconds}s (threshold: {STALE_TASK_THRESHOLD_SECONDS}s)",
                        "original_status": status,
                        "revoked_by": "rq_janitor",
                        "log": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Janitor detected stale task (stuck in {status} for {age_seconds} seconds). Marking as REVOKED."]
                    }

                    cur.execute("""
                        UPDATE task_status
                        SET status = %s, details = %s, progress = 100, timestamp = NOW(), end_time = %s
                        WHERE task_id = %s
                    """, (TASK_STATUS_REVOKED, json.dumps(revoked_details), current_time, task_id))

                    logging.info(f"Janitor cleaned stale database task: {task_id} (type: {task_type}, was in {status} for {age_seconds}s)")

                db.commit()

            cur.close()

    except Exception as e:
        logging.error(f"Error cleaning stale database tasks: {e}", exc_info=True)


if __name__ == '__main__':
    logging.info("🧹 RQ Janitor process starting. Cleaning RQ registries and database tasks every 10 seconds.")
    queues_to_clean = [rq_queue_high, rq_queue_default]
    while True:
        try:
            # 1. Clean RQ registries (existing functionality)
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

            # 2. Clean stale database tasks (new functionality)
            clean_stale_database_tasks()

        except Exception as e:
            logging.error("Error in RQ Janitor loop: %s", e, exc_info=True)

        # Sleep for the desired monitoring interval.
        time.sleep(10)