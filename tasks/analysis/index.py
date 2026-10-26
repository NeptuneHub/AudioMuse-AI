# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Similarity index rebuilds: the visible queue task and the nine build steps.

Main Features:
* _run_all_index_builds: audio IVF (fatal), CLAP text, lyrics, lyrics axes,
  SemGrove, artist similarity, song map, artist map and the Hyperbolic Explorer
  projection backfill + tree build (computed and persisted here, worker-side,
  exactly like the map's UMAP fit - the Flask process only ever loads the
  persisted result via the index-reload NOTIFY, never reclusters); non-fatal
  failures are recorded through the central error registry at WARNING and the
  run continues.
* rebuild_all_indexes_task: the queue entry point, reporting into task_status
  and re-raising so its enqueue-time Retry policy fires.
* The shared g.db connection is CLOSED between build steps. A Postgres backend
  never returns memory to the OS while its session lives, so running all nine
  builds on one connection made that single backend accumulate the union of
  every step's peak: measured on a real server, one connection writing 200MB of
  index blobs climbed to 145MB RSS and stayed there, while a fresh connection
  per build peaked at 69MB and released it on close. Recycling costs one cheap
  reconnect per step and keeps Postgres' idle footprint flat after an analysis.
* Every step runs under a row_heartbeat on the task's own row. The wedged-main
  nudge reads task_status.timestamp and nothing else, but a step is ONE opaque
  call that writes no row while it runs, so on a big library the IVF, artist-GMM
  or hyperbolic backfill outlived QUEUE_WEDGED_MAIN_TASK_MINUTES and had a
  healthy analysis cancelled out from under it. The heartbeat names the running
  step at WARNING each time it fires, bumps from this process so reclaim still
  takes the row of a worker that dies, and is BOUNDED: a build that never
  returns stops being propped up after several windows, or the nudge could never
  fire and one wedged rebuild would lock out every main task.
* After the nine builds finish, the worker issues a CHECKPOINT on its own
  connection and closes it. Postgres only releases WAL segments and flushed
  dirty buffers at a checkpoint, so without it the hundreds of MB written as
  index blobs keep the WAL and buffer cache inflated after the run; forcing the
  checkpoint (plus the per-step recycle) leaves Postgres back at its idle floor
  once the worker's child process exits.
* Each step drops the map, artist-map and hyperbolic tree caches, so an inline
  worker never keeps an index in RAM.
"""

import gc
import hashlib
import json
import logging
import os
import uuid

import taskqueue

import database
from flask_app import app
from app_helper import (
    build_and_store_map_projection,
    build_and_store_artist_projection,
)
import config
from database import close_db, get_db
from config import QUEUE_WEDGED_MAIN_TASK_MINUTES

from error import error_manager
from error.error_dictionary import ERR_INDEX_BUILD

from ..memory_utils import release_memory_to_os
from ..recovery import row_heartbeat, slow_step_budget_minutes
from .helper import make_task_reporter


logger = logging.getLogger(__name__)

_INDEX_BUILD_SIGNATURE_VERSION = 1
_INDEX_BUILD_CONFIG_KEYS = (
    "INDEX_NAME",
    "EMBEDDING_DIMENSION",
    "CLAP_EMBEDDING_DIMENSION",
    "LYRICS_EMBEDDING_DIMENSION",
    "LYRICS_ENABLED",
    "IVF_METRIC",
    "IVF_NLIST_MAX",
    "IVF_TRAIN_POINTS_PER_CELL",
    "IVF_MAX_CELL_MB",
    "IVF_MAX_PART_SIZE_MB",
    "IVF_STORAGE_DTYPE",
    "GMM_N_COMPONENTS_MIN",
    "GMM_N_COMPONENTS_MAX",
    "GMM_COVARIANCE_TYPE",
    "SEM_GROVE_WEIGHT_AUDIO",
    "SEM_GROVE_WEIGHT_LYRICS",
    "HYPERBOLIC_RADIUS_SCALE",
    "HYPERBOLIC_RADIUS_PERCENTILE",
    "HYPERBOLIC_MIN_CLUSTER_SIZE",
    "HYPERBOLIC_RADIAL_SPREAD",
    "HYPERBOLIC_TARGET_LEAF_SIZE",
    "OTHER_FEATURE_LABELS",
    "STRATIFIED_GENRES",
    "VOICE_VOCAB",
    "MOOD_CENTROIDS_FILE",
    "GENRE_SUBGENRE_FILE",
    "NEURAL_FINGERPRINT_ENABLED",
    "NEURAL_FINGERPRINT_MODEL_PATH",
    "NEURAL_FINGERPRINT_CODEBOOK_PATH",
    "NEURAL_FINGERPRINT_INDEX_STRIDE",
    "NEURAL_FINGERPRINT_TRAIN_ROWS",
    "NEURAL_FINGERPRINT_RETRAIN_GROWTH",
)


def _index_build_signature():
    settings = {
        name: getattr(config, name, None) for name in _INDEX_BUILD_CONFIG_KEYS
    }
    settings["version"] = _INDEX_BUILD_SIGNATURE_VERSION
    settings["asset_files"] = {}
    for name in (
        "MOOD_CENTROIDS_FILE",
        "GENRE_SUBGENRE_FILE",
        "NEURAL_FINGERPRINT_MODEL_PATH",
        "NEURAL_FINGERPRINT_CODEBOOK_PATH",
    ):
        path = getattr(config, name, None)
        if not path:
            continue
        try:
            stat = os.stat(path)
            settings["asset_files"][name] = (path, stat.st_size, stat.st_mtime_ns)
        except OSError:
            settings["asset_files"][name] = (path, None, None)
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_index_build_state():
    db = get_db()
    try:
        with db.cursor() as cur:
            cur.execute("""
                SELECT
                    (SELECT array_agg(revision ORDER BY shard)
                     FROM index_input_revision),
                    state.input_revisions,
                    state.build_signature
                FROM (VALUES (TRUE)) AS singleton(value)
                LEFT JOIN index_build_state AS state
                    ON state.singleton = singleton.value
            """)
            row = cur.fetchone()
    except Exception:
        try:
            db.rollback()
        except Exception:
            logger.debug("Could not roll back index build state read", exc_info=True)
        raise

    revisions = row[0] if row else None
    if revisions is None or len(revisions) != 256:
        return None, None, None
    if row[1] is None:
        return tuple(revisions), None, None
    return tuple(revisions), tuple(row[1]), row[2]


def _record_index_build_state(revisions, build_signature):
    db = get_db()
    try:
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO index_build_state "
                "(singleton, input_revisions, build_signature, updated_at) "
                "VALUES (TRUE, %s, %s, CURRENT_TIMESTAMP) "
                "ON CONFLICT (singleton) DO UPDATE SET "
                "input_revisions = EXCLUDED.input_revisions, "
                "build_signature = EXCLUDED.build_signature, "
                "updated_at = EXCLUDED.updated_at",
                (list(revisions), build_signature),
            )
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            logger.debug("Could not roll back index build state update", exc_info=True)
        raise


def _recycle_db_connection():
    try:
        close_db()
    except Exception:
        logger.debug("Index-build connection recycle failed", exc_info=True)


def _checkpoint_postgres():
    try:
        db = get_db()
        try:
            with db.cursor() as cur:
                cur.execute("CHECKPOINT")
            db.commit()
        finally:
            close_db()
    except Exception:
        logger.debug("Post-analysis Postgres CHECKPOINT failed", exc_info=True)


def _run_all_index_builds(
    log_fn=None, progress_start=95, progress_end=98, task_id=None,
    skip_if_unchanged=False,
):
    from ..ivf_manager import build_and_store_ivf_index
    from ..clap_text_search import build_and_store_clap_index
    from ..lyrics_manager import build_and_store_lyrics_index, build_and_store_lyrics_axes_index
    from ..sem_grove_manager import build_and_store_sem_grove_index
    from ..artist_gmm_manager import build_and_store_artist_index
    from ..hyperbolic_manager import (
        backfill_hyperbolic_columns,
        build_hyperbolic_tree_cache,
        reset_hyperbolic_tree_cache,
    )
    from ..hyperbolic_index import build_and_store_hyperbolic_index
    from ..neural_fingerprint_index import build_and_store_neural_fingerprint_index

    def _build_hyperbolic():
        backfill_hyperbolic_columns()
        build_hyperbolic_tree_cache()
        build_and_store_hyperbolic_index(get_db())

    steps = (
        ("IVF index rebuilt", "Building IVF audio index...",
         lambda: build_and_store_ivf_index(get_db()), True),
        ("CLAP text search index", "Building CLAP text search index...",
         lambda: build_and_store_clap_index(get_db()), False),
        ("Lyrics search index", "Building lyrics search index...",
         lambda: build_and_store_lyrics_index(get_db()), False),
        ("Lyrics axes index", "Building lyrics axes index...",
         lambda: build_and_store_lyrics_axes_index(get_db()), False),
        ("SemGrove merged index rebuilt", "Building SemGrove merged index...",
         lambda: build_and_store_sem_grove_index(get_db()), False),
        ("Artist similarity index rebuilt", "Building artist similarity index...",
         lambda: build_and_store_artist_index(get_db()), False),
        ("Song map projection rebuilt", "Building song map projection...",
         lambda: build_and_store_map_projection('main_map'), False),
        ("Artist component projection rebuilt", "Building artist component projection...",
         lambda: build_and_store_artist_projection('artist_map'), False),
        ("Hyperbolic Explorer projections rebuilt", "Building hyperbolic projections...",
         _build_hyperbolic, False),
        ("Neural fingerprint index rebuilt", "Building neural fingerprint index...",
         lambda: build_and_store_neural_fingerprint_index(get_db()), False),
    )
    span = max(0, progress_end - progress_start)

    def safe_log(message, progress):
        if not log_fn:
            return
        try:
            log_fn(message, progress)
        except Exception:
            logger.debug("Index-build progress callback failed", exc_info=True)

    build_signature = _index_build_signature()
    input_revisions = None
    previous_revisions = None
    previous_signature = None
    try:
        input_revisions, previous_revisions, previous_signature = _read_index_build_state()
    except Exception:
        logger.debug("Could not read index input revisions; running a full rebuild", exc_info=True)

    if (
        skip_if_unchanged
        and input_revisions is not None
        and input_revisions == previous_revisions
        and build_signature == previous_signature
    ):
        logger.info("Index inputs and build settings are unchanged; skipping rebuild.")
        safe_log("Index inputs are unchanged; skipping index rebuild.", progress_end)
        release_memory_to_os()
        return

    safe_log("Rebuilding similarity indexes...", progress_start)
    running = ['the index rebuild']
    step_failed = False
    with row_heartbeat(
        task_id, lambda: running[0],
        stop_after_minutes=slow_step_budget_minutes(QUEUE_WEDGED_MAIN_TASK_MINUTES),
    ):
        for index, (label, banner, build, fatal) in enumerate(steps):
            running[0] = f"{label} ({index + 1}/{len(steps)})"
            safe_log(
                f"{banner} ({index + 1}/{len(steps)})",
                progress_start + (span * index) // len(steps),
            )
            try:
                build()
                logger.info(f"OK {label}")
            except Exception as e:
                step_failed = True
                error_manager.record(
                    error_manager.classify(e, ERR_INDEX_BUILD),
                    f"{label}: {e}", exc=e, logger=logger, level=logging.WARNING,
                )
                if fatal:
                    raise
            finally:
                database.MAP_PROJECTION_CACHE = None
                database.ARTIST_PROJECTION_CACHE = None
                reset_hyperbolic_tree_cache()
                _recycle_db_connection()
                gc.collect()
    if input_revisions is not None and not step_failed:
        try:
            _record_index_build_state(input_revisions, build_signature)
        except Exception:
            logger.warning("Could not record the completed index input revision", exc_info=True)
    try:
        taskqueue.publish_event('index-reload')
    except Exception as e:
        logger.warning(f'Could not publish reload message: {e}')

    _checkpoint_postgres()
    release_memory_to_os()


def rebuild_all_indexes_task(parent_task_id=None, skip_if_unchanged=False):
    logger.info("Starting index rebuild task...")
    claimed_task_id = taskqueue.current_task_id()
    current_task_id = claimed_task_id or str(uuid.uuid4())

    with app.app_context():
        from ..task_run import TaskCancelled, cancel_guard

        with cancel_guard(claimed_task_id, parent_task_id) as cancel:
            cancel(force=True)
            log_and_update = make_task_reporter(
                current_task_id, "index_rebuild",
                "Index rebuild started.", parent_task_id=parent_task_id,
                prefix=f"IndexRebuild-{current_task_id}",
            )

            def report_between_builds(message, progress, **kwargs):
                cancel()
                return log_and_update(message, progress, **kwargs)

            try:
                _run_all_index_builds(
                    log_fn=report_between_builds, progress_start=0, progress_end=99,
                    task_id=current_task_id, skip_if_unchanged=skip_if_unchanged,
                )
            except TaskCancelled:
                raise
            except Exception as e:
                err = error_manager.from_exception(
                    e, code=error_manager.classify(e, ERR_INDEX_BUILD), logger=logger
                )
                log_and_update(
                    "Index rebuild failed. Check the container logs for details.",
                    log_and_update.state['progress'], error=err,
                )
                raise
        message = "All similarity indexes rebuilt."
        log_and_update(message, 100)
        logger.info("OK Index rebuild task completed successfully")
        return {"message": message}
