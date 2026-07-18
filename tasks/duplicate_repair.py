# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One-time catalogue-id duplicate check for installs migrated by an early 3.0 build.

There are two upgrade paths, told apart entirely by the state of the tables, so
neither needs a stored flag (a flag in app_config is purged as an unknown key on
the next boot, which made this run on every restart):

* From < 3.0.0: score.item_id are provider ids. The legacy catalogue migration
  (fingerprint_canonicalize) computes content ids AND backfills score.duration,
  so its duplicate groups are already duration-confirmed. Nothing here to do.
* From an early 3.0.0: score.item_id are already fp_ content ids, but they were
  merged from the MusiCNN embedding alone, so different recordings collapsed into
  one id and their survivor row carries NO duration. THOSE are what this fixes.

The signal is score.duration: a survivor WITH a duration was already confirmed
(by the legacy migration or by a previous run of this check), a survivor with a
NULL duration was not. So this only ever looks at fp_ groups whose survivor has
no duration; when a group turns out to be a real duplicate it writes the
survivor's duration, so the group is never examined again; when it is a false
duplicate its provider files are unmapped so the next analysis re-analyzes them
under their own correct ids. Durations come from ONE metadata listing per
server (no audio downloads). A server that cannot be listed is skipped and its
groups stay NULL, so the check simply retries them on the next start. It never
deletes a score or embedding row.

Main Features:
* Table-derived, marker-free idempotency via score.duration - a no-op once every
  survivor is confirmed, so it is instant after the first successful pass.
* Duration-consensus per group; real groups get their duration stamped, false
  groups lose only their track_server_map rows.
* Progress logs at every ~10% plus a final real-vs-false summary.
"""

import logging

from psycopg2.extras import execute_values

import config
from database import connect_raw
from tasks import provider_probe
from tasks.mediaserver import context as ms_context
from tasks.mediaserver import registry
from tasks.simhash import CANONICAL_ID_LEN

logger = logging.getLogger(__name__)

_DELETE_CHUNK = 5000
_MIN_KNOWN_DURATION_RATIO = 0.5
# Its own lock, distinct from the legacy migration's, so exactly one Flask
# replica runs this check on a multi-replica boot instead of every replica
# pulling each server's catalogue at once.
_REPAIR_ADVISORY_LOCK = 726354823


def _groups_needing_check(cur):
    """Current-scheme fp_2 groups (>1 file) whose survivor has NO duration yet.

    A survivor WITH a duration was already confirmed - by the legacy migration or
    a previous run - so it is skipped. This is what makes the check a table-derived
    one-time step with no stored flag. Only current-scheme signature ids qualify:
    an fp_0 (no-signature) id is always a single file, and an fp_1 (retired
    scheme) id is relabelled by the migration first, so neither should be here.
    """
    cur.execute(
        "SELECT tsm.server_id, s.item_id, array_agg(tsm.provider_track_id) "
        "FROM track_server_map tsm "
        "JOIN score s ON s.item_id = tsm.item_id "
        "WHERE s.item_id LIKE 'fp\\_2%%' AND length(s.item_id) = %s "
        "AND s.duration IS NULL "
        "GROUP BY tsm.server_id, s.item_id "
        "HAVING count(*) > 1",
        (CANONICAL_ID_LEN,),
    )
    groups = {}
    for server_id, item_id, provider_ids in cur.fetchall():
        groups.setdefault(str(server_id), {})[str(item_id)] = [
            str(provider_id) for provider_id in provider_ids
        ]
    return groups


def _server_durations(server):
    with ms_context.use_server(server):
        tracks = provider_probe.fetch_all_tracks(
            server['server_type'], server['creds'], apply_filter=False
        )
    return {
        str(track['id']): track['duration']
        for track in tracks
        if track.get('id') is not None and track.get('duration') is not None
    }


def _group_duration(provider_ids, durations):
    """The consensus duration of a real duplicate group, or None if it is false.

    Real means every member's length is known and they all agree within
    DURATION_TOLERANCE_SECONDS; the stamped value is the smallest, deterministic
    and within tolerance of the survivor's true length.
    """
    values = [durations.get(provider_id) for provider_id in provider_ids]
    if any(value is None for value in values):
        return None
    if (max(values) - min(values)) > config.DURATION_TOLERANCE_SECONDS:
        return None
    return min(values)


def _server_label(server, server_id):
    return (server or {}).get('name') or server_id


def _stamp_real_durations(cur, real_durations):
    if not real_durations:
        return
    execute_values(
        cur,
        "UPDATE score SET duration = data.duration "
        "FROM (VALUES %s) AS data(item_id, duration) "
        "WHERE score.item_id = data.item_id AND score.duration IS NULL",
        list(real_durations.items()),
    )


def _unmap_false_groups(cur, server_id, false_ids):
    removed = 0
    for begin in range(0, len(false_ids), _DELETE_CHUNK):
        chunk = false_ids[begin:begin + _DELETE_CHUNK]
        cur.execute(
            "DELETE FROM track_server_map "
            "WHERE server_id = %s AND item_id = ANY(%s)",
            (server_id, chunk),
        )
        removed += cur.rowcount
    return removed


def repair_duplicate_track_maps(conn=None):
    own_conn = conn is None
    db = conn or connect_raw()
    acquired = False
    cur = None
    try:
        try:
            db.autocommit = False
        except Exception:
            logger.debug("Could not force autocommit off", exc_info=True)
        cur = db.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_REPAIR_ADVISORY_LOCK,))
        acquired = bool(cur.fetchone()[0])
        if not acquired:
            logger.info(
                "Catalogue id duplicate check: another replica already holds the "
                "lock; skipping on this one."
            )
            return {'skipped': 'locked'}
        try:
            groups_by_server = _groups_needing_check(cur)
            total_groups = sum(len(groups) for groups in groups_by_server.values())
            if not total_groups:
                return {'checked': 0, 'real': 0, 'false': 0, 'removed': 0}

            logger.info("=" * 64)
            logger.info(
                "START OF CATALOGUE ID DUPLICATE CHECK ON %d SONGS: these catalogue "
                "ids map more than one file on their server and were merged before "
                "track length was considered (%d server(s) involved).",
                total_groups, len(groups_by_server),
            )
            logger.info(
                "One-time step: durations come from the music server's metadata "
                "listing, no audio is downloaded. Real duplicates (same audio, same "
                "length) are kept; false duplicates are unmapped so the next analysis "
                "run re-analyzes them under their own correct ids."
            )
            logger.info("=" * 64)

            step = max(1, total_groups // 10)
            checked = real = false = removed = 0

            for server_id, groups in groups_by_server.items():
                server = registry.get_server(server_id, conn=db)
                if server is None:
                    logger.warning(
                        "Catalogue id duplicate check: server %s no longer exists; "
                        "leaving its %d songs to a later start.", server_id, len(groups),
                    )
                    checked += len(groups)
                    continue
                try:
                    durations = _server_durations(server)
                except Exception:
                    logger.exception(
                        "Catalogue id duplicate check: could not list tracks from "
                        "server '%s'; its %d songs stay unconfirmed and the check "
                        "retries them on the next start.",
                        _server_label(server, server_id), len(groups),
                    )
                    checked += len(groups)
                    continue

                member_ids = [
                    provider_id
                    for provider_ids in groups.values()
                    for provider_id in provider_ids
                ]
                known = sum(1 for provider_id in member_ids if provider_id in durations)
                if known < _MIN_KNOWN_DURATION_RATIO * len(member_ids):
                    logger.warning(
                        "Catalogue id duplicate check: server '%s' returned durations "
                        "for only %d of %d mapped files; listing looks unreliable, "
                        "retrying this server on the next start.",
                        _server_label(server, server_id), known, len(member_ids),
                    )
                    checked += len(groups)
                    continue

                real_durations = {}
                false_ids = []
                for item_id, provider_ids in groups.items():
                    consensus = _group_duration(provider_ids, durations)
                    if consensus is None:
                        false += 1
                        false_ids.append(item_id)
                    else:
                        real += 1
                        real_durations[item_id] = consensus
                    checked += 1
                    if checked % step == 0 or checked == total_groups:
                        logger.info(
                            "Catalogue id duplicate check: %d%% (%d/%d songs checked; "
                            "%d real duplicates, %d false so far)",
                            int(round(100.0 * checked / total_groups)),
                            checked, total_groups, real, false,
                        )

                _stamp_real_durations(cur, real_durations)
                removed += _unmap_false_groups(cur, server_id, false_ids)
                if false_ids:
                    cur.execute(
                        "UPDATE music_servers SET updated_at = now() "
                        "WHERE server_id = %s",
                        (server_id,),
                    )
                db.commit()

            logger.info("=" * 64)
            logger.info(
                "CATALOGUE ID DUPLICATE CHECK COMPLETE: of the initial %d duplicated "
                "songs, %d are REAL duplicates (kept, their length recorded) and %d "
                "were FALSE duplicates from the old id calculation; %d file "
                "mapping(s) removed. The next analysis run will re-analyze those "
                "files and give each one its own correct catalogue id.",
                total_groups, real, false, removed,
            )
            logger.info("=" * 64)
            return {'checked': checked, 'real': real, 'false': false, 'removed': removed}
        except Exception:
            try:
                db.rollback()
            except Exception:
                logger.debug("Rollback failed", exc_info=True)
            logger.exception(
                "Catalogue id duplicate check failed; it retries on the next start"
            )
            raise
    finally:
        if cur is not None:
            if acquired:
                try:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (_REPAIR_ADVISORY_LOCK,))
                    db.commit()
                except Exception:
                    logger.debug("Advisory unlock failed", exc_info=True)
            cur.close()
        if own_conn:
            db.close()
