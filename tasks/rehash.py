# tasks/rehash.py
"""
Background RQ task for rehashing provider track paths.

Used after a Navidrome user enables "Report Real Path" — this task fetches
the current (now-absolute) paths from the provider API and updates all
track identity records (file_path_hash, normalized_path) so cross-provider
matching works correctly.
"""

import logging
import time
import uuid

from rq import get_current_job

logger = logging.getLogger(__name__)


def rehash_provider_tracks_task(provider_id):
    """
    RQ task: Re-fetch paths from a provider and rehash all linked tracks.

    Steps:
    1. Fetch all songs from the provider API (current paths)
    2. For each provider_track record, find the new path
    3. Recompute normalized_path and file_path_hash
    4. Update track table records
    5. Merge duplicates (two tracks that now hash to the same value)

    Args:
        provider_id: The provider ID to rehash tracks for
    """
    from app import app
    from app_helper import (
        redis_conn, get_db, save_task_status,
        normalize_provider_path, _compute_file_path_hash,
        TASK_STATUS_STARTED, TASK_STATUS_PROGRESS,
        TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE
    )
    from app_setup import get_provider_by_id

    current_job = get_current_job(redis_conn)
    current_task_id = current_job.id if current_job else str(uuid.uuid4())

    with app.app_context():
        save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_STARTED,
                         progress=0, details={"message": "Starting track rehash..."})

        try:
            # Get provider info
            provider = get_provider_by_id(provider_id)
            if not provider:
                save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_FAILURE,
                                 progress=0, details={"message": f"Provider {provider_id} not found"})
                return {"status": "FAILURE", "message": "Provider not found"}

            provider_type = provider['provider_type']
            provider_name = provider.get('name') or provider_type
            config_data = provider.get('config') or {}

            # Verify path_format is now 'absolute' (user must fix setting first)
            if config_data.get('path_format') == 'relative':
                msg = (
                    f"Provider '{provider_name}' still has relative paths. "
                    f"Enable 'Report Real Path' in Navidrome and run 'Rescan Paths' first."
                )
                save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_FAILURE,
                                 progress=0, details={"message": msg})
                return {"status": "FAILURE", "message": msg}

            save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_PROGRESS,
                             progress=5, details={"message": f"Fetching all songs from {provider_name}..."})

            # Fetch all songs from the provider to get current paths
            from tasks.mediaserver import get_sample_tracks_from_provider
            all_tracks = get_sample_tracks_from_provider(provider_type, config_data, limit=99999)

            if not all_tracks:
                save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_FAILURE,
                                 progress=10, details={"message": "Could not fetch tracks from provider"})
                return {"status": "FAILURE", "message": "Could not fetch tracks from provider"}

            # Build a lookup: item_id -> current file_path from provider
            # get_sample_tracks_from_provider returns dicts with 'id', 'title', 'artist', 'file_path'
            path_by_item_id = {}
            for t in all_tracks:
                item_id = t.get('id')
                file_path = t.get('file_path')
                if item_id and file_path:
                    path_by_item_id[str(item_id)] = file_path

            save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_PROGRESS,
                             progress=15, details={
                                 "message": f"Fetched {len(path_by_item_id)} tracks with paths. Rehashing..."
                             })

            # Get all provider_track records for this provider
            db = get_db()
            with db.cursor() as cur:
                cur.execute("""
                    SELECT pt.item_id, pt.track_id, t.file_path_hash, t.normalized_path, t.file_path
                    FROM provider_track pt
                    JOIN track t ON pt.track_id = t.id
                    WHERE pt.provider_id = %s
                """, (provider_id,))
                provider_tracks = cur.fetchall()

            total = len(provider_tracks)
            rehashed = 0
            skipped = 0
            merged_duplicates = 0
            errors = 0

            for idx, (item_id, track_id, old_hash, old_normalized, old_file_path) in enumerate(provider_tracks):
                progress = 15 + int(80 * ((idx + 1) / max(total, 1)))

                if idx % 100 == 0:
                    save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_PROGRESS,
                                     progress=progress, details={
                                         "message": f"Rehashing track {idx + 1}/{total}...",
                                         "rehashed": rehashed, "skipped": skipped,
                                         "merged": merged_duplicates, "errors": errors
                                     })

                # Get new path from provider
                new_path = path_by_item_id.get(str(item_id))
                if not new_path:
                    skipped += 1
                    continue

                # Compute new normalized path and hash
                new_normalized = normalize_provider_path(new_path, provider_id)
                if not new_normalized:
                    skipped += 1
                    continue

                new_hash = _compute_file_path_hash(new_path, provider_id)
                if not new_hash:
                    skipped += 1
                    continue

                # If hash hasn't changed, skip
                if new_hash == old_hash:
                    skipped += 1
                    continue

                try:
                    with db.cursor() as cur:
                        # Check if a track with this new hash already exists
                        cur.execute("SELECT id FROM track WHERE file_path_hash = %s AND id != %s",
                                    (new_hash, track_id))
                        existing = cur.fetchone()

                        if existing:
                            # Merge: another track record already has this hash
                            # Keep the existing one (it likely has analysis data)
                            existing_track_id = existing[0]

                            # Re-point this provider_track to the existing track
                            cur.execute("""
                                UPDATE provider_track SET track_id = %s
                                WHERE provider_id = %s AND item_id = %s
                            """, (existing_track_id, provider_id, item_id))

                            # In the new schema, track_id IS the score PK.
                            # If the existing track already has a score row, delete the old one.
                            # If the existing track has NO score row, re-point the old one.
                            cur.execute("SELECT 1 FROM score WHERE track_id = %s", (existing_track_id,))
                            if cur.fetchone():
                                # Existing track has analysis — delete the duplicate
                                cur.execute("DELETE FROM score WHERE track_id = %s", (track_id,))
                            else:
                                # No analysis for existing track — can't re-key (PK conflict)
                                # Just delete the orphan score since existing track will get re-analyzed
                                cur.execute("DELETE FROM score WHERE track_id = %s", (track_id,))

                            # Check if old track_id is still referenced
                            cur.execute("""
                                SELECT COUNT(*) FROM provider_track WHERE track_id = %s
                            """, (track_id,))
                            ref_count = cur.fetchone()[0]

                            if ref_count == 0:
                                # Safe to delete orphaned track record
                                cur.execute("DELETE FROM track WHERE id = %s", (track_id,))

                            db.commit()
                            merged_duplicates += 1
                            rehashed += 1
                        else:
                            # Update the track record with new hash and path
                            cur.execute("""
                                UPDATE track
                                SET file_path_hash = %s, normalized_path = %s,
                                    file_path = %s, updated_at = NOW()
                                WHERE id = %s
                            """, (new_hash, new_normalized, new_path, track_id))

                            # Also update score.file_path if it exists
                            cur.execute("""
                                UPDATE score SET file_path = %s
                                WHERE track_id = %s
                            """, (new_normalized, track_id))

                            db.commit()
                            rehashed += 1

                except Exception as e:
                    logger.error(f"Error rehashing track {item_id} (track_id={track_id}): {e}")
                    errors += 1
                    try:
                        db.rollback()
                    except Exception:
                        pass

            summary = {
                "message": f"Rehash complete for {provider_name}",
                "rehashed": rehashed,
                "skipped": skipped,
                "merged_duplicates": merged_duplicates,
                "errors": errors,
                "total": total
            }

            save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_SUCCESS,
                             progress=100, details=summary)

            # Clear needs_rehash flag now that rehash is complete
            try:
                with db.cursor() as cur2:
                    cur2.execute(
                        "UPDATE provider SET config = config - 'needs_rehash' WHERE id = %s",
                        (provider_id,)
                    )
                    db.commit()
            except Exception:
                pass  # Non-critical — flag will just remain

            logger.info(f"Track rehash complete for provider {provider_id}: {summary}")
            return {"status": "SUCCESS", **summary}

        except Exception as e:
            logger.error(f"Track rehash failed for provider {provider_id}: {e}", exc_info=True)
            save_task_status(current_task_id, "rehash_tracks", TASK_STATUS_FAILURE,
                             progress=0, details={"message": f"Rehash failed: {str(e)}"})
            return {"status": "FAILURE", "message": str(e)}
