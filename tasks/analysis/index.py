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
* rebuild_all_indexes_task: the queue entry point, reporting into task_status and
  re-raising on failure so its enqueue-time Retry policy actually fires.
"""

import gc
import logging
import uuid

import taskqueue

from flask_app import app
from app_helper import (
    build_and_store_map_projection,
    build_and_store_artist_projection,
)
from database import get_db
from config import TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE

from error import error_manager
from error.error_dictionary import ERR_INDEX_BUILD

from ..memory_utils import release_memory_to_os
from .helper import make_task_reporter


logger = logging.getLogger(__name__)


def _run_all_index_builds(log_fn=None, progress_start=95, progress_end=98):
    from ..ivf_manager import build_and_store_ivf_index
    from ..clap_text_search import build_and_store_clap_index
    from ..lyrics_manager import build_and_store_lyrics_index, build_and_store_lyrics_axes_index
    from ..sem_grove_manager import build_and_store_sem_grove_index
    from ..artist_gmm_manager import build_and_store_artist_index
    from ..hyperbolic_manager import backfill_hyperbolic_columns, build_hyperbolic_tree_cache
    from ..hyperbolic_index import build_and_store_hyperbolic_index

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
    )
    span = max(0, progress_end - progress_start)

    def safe_log(message, progress):
        if not log_fn:
            return
        try:
            log_fn(message, progress)
        except Exception:
            logger.debug("Index-build progress callback failed", exc_info=True)

    safe_log("Rebuilding similarity indexes...", progress_start)
    for index, (label, banner, build, fatal) in enumerate(steps):
        safe_log(
            f"{banner} ({index + 1}/{len(steps)})",
            progress_start + (span * index) // len(steps),
        )
        try:
            build()
            logger.info(f"OK {label}")
        except Exception as e:
            error_manager.record(
                error_manager.classify(e, ERR_INDEX_BUILD),
                f"{label}: {e}", exc=e, logger=logger, level=logging.WARNING,
            )
            if fatal:
                raise
        finally:
            gc.collect()
    try:
        taskqueue.publish_event('index-reload')
    except Exception as e:
        logger.warning(f'Could not publish reload message: {e}')

    release_memory_to_os()


def rebuild_all_indexes_task(parent_task_id=None):
    logger.info("Starting index rebuild task...")
    claimed_task_id = taskqueue.current_task_id()
    current_task_id = claimed_task_id or str(uuid.uuid4())

    with app.app_context():
        if claimed_task_id and parent_task_id:
            from database import get_task_statuses
            from config import TASK_STATUS_REVOKED

            statuses = get_task_statuses([parent_task_id])
            if parent_task_id not in statuses or statuses.get(parent_task_id) in (
                TASK_STATUS_SUCCESS,
                TASK_STATUS_FAILURE,
                TASK_STATUS_REVOKED,
            ):
                logger.info(
                    "Index rebuild %s will not start because parent %s was cancelled.",
                    current_task_id,
                    parent_task_id,
                )
                return {
                    "status": TASK_STATUS_REVOKED,
                    "message": "Parent analysis was cancelled.",
                }
        log_and_update = make_task_reporter(
            current_task_id, "index_rebuild",
            "Index rebuild started.", parent_task_id=parent_task_id,
            prefix=f"IndexRebuild-{current_task_id}",
        )
        try:
            _run_all_index_builds(
                log_fn=log_and_update, progress_start=0, progress_end=99
            )
        except Exception as e:
            err = error_manager.from_exception(
                e, code=error_manager.classify(e, ERR_INDEX_BUILD), logger=logger
            )
            log_and_update(
                "Index rebuild failed. Check the container logs for details.",
                100, task_state=TASK_STATUS_FAILURE, error=err,
            )
            raise
        log_and_update(
            "All similarity indexes rebuilt.", 100, task_state=TASK_STATUS_SUCCESS
        )
        logger.info("OK Index rebuild task completed successfully")
        return {"status": "SUCCESS", "message": "All indexes rebuilt"}
