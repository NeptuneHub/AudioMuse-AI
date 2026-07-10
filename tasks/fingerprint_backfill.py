# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Backfill content fingerprints for already-analyzed tracks that lack one.

Newly analyzed tracks get their fingerprint during analysis from the audio in
hand. This module fills the gap for legacy rows created before fingerprinting
existed: it downloads each track from the default media server, computes the
same 64-bit SimHash catalogue id, and stores it. It is bounded per run and best
effort - a failed download or an unavailable fingerprinter never raises.

Main Features:
* ``backfill_fingerprints`` fingerprints up to ``limit`` fingerprint-less tracks.
* Uses the existing media-server download path and cleans up temp audio.
"""

import logging
import os

import config
from database import connect_raw, set_track_fingerprint
from tasks import audio_fingerprint

logger = logging.getLogger(__name__)


def _fingerprint_one(item_id):
    from tasks import mediaserver

    path = None
    try:
        path = mediaserver.download_track(config.TEMP_DIR, {'Id': item_id, 'id': item_id})
        if not path:
            return None
        return audio_fingerprint.canonical_fingerprint_file(path)
    except Exception:
        logger.debug("Fingerprint backfill failed for %s", item_id, exc_info=True)
        return None
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def backfill_fingerprints(limit=None, conn=None):
    """Fingerprint up to ``limit`` analyzed tracks that have no fingerprint yet."""
    own_conn = conn is None
    db = conn or connect_raw()
    try:
        cur = db.cursor()
        try:
            if limit and limit > 0:
                cur.execute(
                    "SELECT item_id FROM score WHERE fingerprint IS NULL LIMIT %s", (limit,)
                )
            else:
                cur.execute("SELECT item_id FROM score WHERE fingerprint IS NULL")
            item_ids = [r[0] for r in cur.fetchall()]
        finally:
            cur.close()

        done = 0
        for item_id in item_ids:
            fingerprint = _fingerprint_one(item_id)
            if fingerprint is not None:
                set_track_fingerprint(item_id, fingerprint, conn=db)
                done += 1
        if item_ids:
            logger.info(
                "Fingerprint backfill: %d/%d fingerprint-less tracks updated", done, len(item_ids)
            )
        return {'candidates': len(item_ids), 'fingerprinted': done}
    finally:
        if own_conn:
            db.close()
