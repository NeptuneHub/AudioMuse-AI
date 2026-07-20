# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""One-time catalogue duration backfill + duplicate repair at Flask startup.

Every fp_2 content id needs its track LENGTH in score.duration: it is the anchor
of identity confirmation (durations_compatible splits when either side is NULL),
so a row with no duration can never be a duration-confirmed merge target - the
same audio re-added, or the same song on another server (the whole point of the
N:1 track_server_map design), would fail the check and mint a duplicate id. So
this gives EVERY fp_2 row a duration, not only the duplicates.

Upgrade paths, told apart by the shape of score.item_id, all converge here
(no stored flag - a flag in app_config is purged as an unknown key on the next
boot, which made an earlier version run on every restart):

* From < 3.0.0 (item_id are provider ids): the legacy migration
  (fingerprint_canonicalize) relabels to the current scheme AND backfills
  score.duration for every relabelled row in one shot. This step then no-ops.
* From an early 3.0.0 (item_id are ALREADY fp_2): the legacy migration no-ops,
  and THIS step does the work - it backfills the length of every fp_2 row and
  fixes the embedding-only false merges.
* From a later scheme bump (e.g. fp_3 -> fp_4, which tightened the duration
  tolerance): every EXISTING merge is re-confirmed in place at the current
  tolerance and the ones whose files now differ in length by more than it are
  unmapped; the stored length is kept, never dropped. Then the scheme relabel
  bumps every row up (minting the next id on a collision), so it runs once.

The signal is score.duration: a row WITH a duration was already confirmed, a row
with a NULL duration was not. Every fp_2 NULL-duration row is looked at exactly
once. A row mapping ONE file just gets its length stamped. A row mapping MORE
than one file is a merge: real (all lengths agree) keeps its length stamped and
its mappings; false (lengths differ) is unmapped so the next analysis re-analyzes
each file under its own correct id. A single file whose server reports no length
is stamped with a 0 sentinel so the whole catalogue is not re-listed for it on
every boot - 0 behaves exactly like NULL for identity (never confirms a merge)
and a later re-analysis overwrites it with the real length. Durations come from
ONE metadata listing per server (no per-id/batch fetch, no audio downloads),
fetched concurrently across servers. It never deletes a score or embedding row.

Main Features:
* Table-derived, marker-free idempotency via score.duration - instant no-op once
  every row carries a length.
* Backfills the length of EVERY fp_2 row (single-file and duplicate survivors),
  not only duplicates, from the same one whole-catalogue listing per server.
* Duplicate consensus: real groups keep + stamp their length, false groups lose
  only their track_server_map rows; single-file rows are never unmapped.
* Concurrent per-server fetch, progress logs at every ~10%, final summary.
* split_chromaprint_false_merges: cleaning-time Path B - splits a same-server
  duplicate group whose stored Chromaprints prove its files are different
  recordings (skip-if-missing); unmaps only, never deletes a catalogue row.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from psycopg2.extras import execute_values

import config
from database import connect_raw
from tasks import provider_probe
from tasks import simhash
from tasks.chromaprint import chromaprints_agree
from tasks.mediaserver import context as ms_context
from tasks.mediaserver import registry

logger = logging.getLogger(__name__)

_DELETE_CHUNK = 5000
_STAMP_PAGE = 5000
_MIN_KNOWN_DURATION_RATIO = 0.5
_MAX_FETCH_THREADS = 6
# Written to score.duration for a single-file row whose server reports no length,
# so it is no longer NULL (never re-listed) yet still never confirms a merge
# (durations_compatible rejects <= 0). A re-analysis overwrites it with the real
# length via COALESCE, so it is self-healing.
_NO_SERVER_DURATION = 0.0
# Its own lock, distinct from the legacy migration's, so exactly one Flask
# replica runs this check on a multi-replica boot instead of every replica
# pulling each server's catalogue at once.
_REPAIR_ADVISORY_LOCK = 726354823
_SAME_FOLDER_ADVISORY_LOCK = 726354824
_CHROMAPRINT_ADVISORY_LOCK = 726354825


def _empty_totals():
    return {
        'checked': 0, 'backfilled': 0, 'no_length': 0,
        'real': 0, 'false': 0, 'removed': 0, 'relabelled': 0,
    }


def _old_scheme_where(alias='s'):
    """(sql, params) matching an OLDER-version signature id that must be bumped up.

    Delegates to the single spelling of the predicate in ``simhash.signature_id_sql``
    so a future scheme bump touches one place.
    """
    return simhash.signature_id_sql(alias)


def _old_scheme_rows_exist(cur):
    """The one-time gate: are there any older-version ids left to migrate?

    Once the migration has bumped every id to the current scheme this is false and
    the whole step is skipped instantly - no server is ever contacted again. This
    is the hard version gate: it does not infer 'done' from durations, so orphans
    the server has no length for can never re-trigger it.
    """
    where, params = _old_scheme_where('score')
    cur.execute("SELECT EXISTS (SELECT 1 FROM score WHERE " + where + ")", params)
    return bool(cur.fetchone()[0])


def _groups_needing_check(cur):
    """Older-version groups to look at, grouped (server, item_id) -> files.

    Two kinds come back: single-file rows with NO length yet (backfill), and EVERY
    multi-file group (a merge to re-confirm at the CURRENT DURATION_TOLERANCE_SECONDS,
    whether or not its length is already stamped). The second kind is what lets a
    scheme bump re-verify existing merges in place - the survivor keeps its stored
    length untouched (the stamp only fills NULLs), and a group whose files now differ
    by more than the tolerance is unmapped. The file-list size tells the two apart.
    """
    where, params = _old_scheme_where('s')
    cur.execute(
        "SELECT tsm.server_id, s.item_id, array_agg(tsm.provider_track_id), "
        "array_agg(tsm.file_path) "
        "FROM track_server_map tsm "
        "JOIN score s ON s.item_id = tsm.item_id "
        "WHERE " + where + " "
        "GROUP BY tsm.server_id, s.item_id "
        "HAVING count(*) > 1 OR bool_or(s.duration IS NULL)",
        params,
    )
    groups = {}
    for server_id, item_id, provider_ids, file_paths in cur.fetchall():
        groups.setdefault(str(server_id), {})[str(item_id)] = (
            [str(provider_id) for provider_id in provider_ids],
            [path for path in file_paths if path],
        )
    return groups


def _server_durations(server):
    # apply_filter=True so a server whose AudioMuse config is a subset of a much
    # larger media library only lists the configured folders, not the whole
    # server. The duplicate provider ids are all analyzed tracks, so they live in
    # the configured libraries and are always covered.
    with ms_context.use_server(server):
        tracks = provider_probe.fetch_all_tracks(
            server['server_type'], server['creds'], apply_filter=True
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


def _stamp_durations(cur, durations_to_write):
    if not durations_to_write:
        return
    execute_values(
        cur,
        "UPDATE score SET duration = data.duration "
        "FROM (VALUES %s) AS data(item_id, duration) "
        "WHERE score.item_id = data.item_id AND score.duration IS NULL",
        list(durations_to_write.items()),
        page_size=_STAMP_PAGE,
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


def _force_no_autocommit(db):
    try:
        db.autocommit = False
    except Exception:
        logger.debug("Could not force autocommit off", exc_info=True)


def _rollback(db):
    try:
        db.rollback()
    except Exception:
        logger.debug("Rollback failed", exc_info=True)


def _release(cur, db, acquired, own_conn, lock=_REPAIR_ADVISORY_LOCK):
    if cur is not None:
        if acquired:
            try:
                cur.execute("SELECT pg_advisory_unlock(%s)", (lock,))
                if own_conn:
                    db.commit()
            except Exception:
                logger.debug("Advisory unlock failed", exc_info=True)
        cur.close()
    if own_conn:
        db.close()


def _log_start_banner(total_groups, server_count):
    logger.info("=" * 64)
    logger.info(
        "START OF CATALOGUE DUPLICATE RE-VERIFY ON %d group(s) across %d server(s): "
        "existing merges are re-confirmed at the current duration tolerance (a group "
        "whose files now differ in length by more than it is split), and any row "
        "still without a length gets one. Stored durations are kept, not dropped.",
        total_groups, server_count,
    )
    logger.info(
        "One-time step per scheme version: lengths come from the music server's "
        "metadata listing, no audio is downloaded. A real duplicate keeps its "
        "mappings; a false one is unmapped so the next analysis re-analyzes each "
        "file under its own correct id."
    )
    logger.info("=" * 64)


def _log_progress(totals, total_groups):
    logger.info(
        "Catalogue duplicate re-verify: %d%% (%d/%d groups; %d confirmed, "
        "%d split so far)",
        int(round(100.0 * totals['checked'] / total_groups)),
        totals['checked'], total_groups,
        totals['backfilled'] + totals['real'], totals['false'],
    )


def _log_complete(total_groups, totals):
    logger.info("=" * 64)
    logger.info(
        "CATALOGUE DUPLICATE RE-VERIFY COMPLETE: of %d group(s) checked, %d single "
        "songs and %d real duplicates were confirmed (lengths kept or stamped); %d "
        "were split as false merges (%d mapping(s) unmapped) and %d had no length on "
        "the server. The next analysis re-analyzes the unmapped files under their "
        "own correct ids.",
        total_groups, totals['backfilled'], totals['real'], totals['false'],
        totals['removed'], totals['no_length'],
    )
    logger.info("=" * 64)


def _fetch_all_server_durations(db, groups_by_server, prefetched=None):
    """Every server's duration map, all fetched CONCURRENTLY.

    The catalogue listing per server is one slow HTTP round trip and they are
    independent, so they run in a thread pool: the wall-clock is the slowest
    single server, not the sum. It is safe to thread because each server's fetch
    is HTTP-only and reads its creds/libraries from a per-thread context var
    (`use_server` inside `_server_durations`), never the shared Flask DB handle -
    `registry.get_server` (which uses `db`) is resolved here on the main thread
    first. A server that no longer exists or cannot be listed maps to None, so
    its groups are left for the next start. NOT a per-id/batch fetch: still one
    whole-catalogue listing per server, just no longer serialized.

    ``prefetched`` is the legacy migration's own whole-server listing from earlier
    this same boot: a server already in it is NOT listed again (that second full
    listing was minutes of pure waste on a mixed upgrade). The legacy listing is
    unfiltered, a superset of the filtered one, so it can only widen coverage.
    """
    prefetched = prefetched or {}
    durations = {sid: prefetched[sid] for sid in groups_by_server if sid in prefetched}
    if durations:
        logger.info(
            "Catalogue duration backfill: reusing the legacy migration's listing "
            "for %d server(s) - not listing them again.", len(durations),
        )
    servers = {}
    for server_id in groups_by_server:
        if server_id in durations:
            continue
        server = registry.get_server(server_id, conn=db)
        if server is None:
            logger.warning(
                "Catalogue id duplicate check: server %s no longer exists; "
                "leaving its %d songs to a later start.",
                server_id, len(groups_by_server[server_id]),
            )
        else:
            servers[server_id] = server
    if not servers:
        return durations
    workers = min(len(servers), _MAX_FETCH_THREADS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_server_durations, server): server_id
            for server_id, server in servers.items()
        }
        for future in futures:
            server_id = futures[future]
            try:
                durations[server_id] = future.result()
            except Exception:
                logger.exception(
                    "Catalogue id duplicate check: could not list tracks from "
                    "server '%s'; its %d songs stay unconfirmed and the check "
                    "retries them on the next start.",
                    _server_label(servers[server_id], server_id),
                    len(groups_by_server[server_id]),
                )
    return durations


def _classify_group(item_id, provider_ids, file_paths, durations, totals, to_stamp, false_ids):
    consensus = _group_duration(provider_ids, durations)
    is_duplicate = len(provider_ids) > 1
    if is_duplicate and simhash.folder_conflict_in_group(file_paths):
        false_ids.append(item_id)
        totals['false'] += 1
    elif consensus is not None:
        # single file -> its length; real duplicate -> the agreed length
        to_stamp[item_id] = consensus
        totals['real' if is_duplicate else 'backfilled'] += 1
    elif is_duplicate:
        # lengths disagree or are missing -> a false merge; unmap and re-analyze
        false_ids.append(item_id)
        totals['false'] += 1
    else:
        # single file the server has no length for: stamp the sentinel so the
        # catalogue is not re-listed for it forever (never unmap a single file)
        to_stamp[item_id] = _NO_SERVER_DURATION
        totals['no_length'] += 1


def _process_server(db, cur, server_id, groups, durations, totals, total_groups, step):
    if durations is None:
        # server gone or unreachable (already logged) - retried next start
        totals['checked'] += len(groups)
        return
    # Reliability is measured against the server's WHOLE catalogue vs how many
    # tracks we have mapped for it - NOT against the NULL rows. The NULL rows are
    # exactly the ones the server could not give a length for last time, i.e. the
    # orphans (files deleted from the server) we now want to stamp with the
    # sentinel; measuring "known of the NULL rows" declared a perfectly healthy
    # server unreliable whenever its leftovers were orphans, skipped it, never
    # stamped the sentinel, and re-listed the whole catalogue on every restart.
    cur.execute(
        "SELECT count(*) FROM track_server_map WHERE server_id = %s", (server_id,)
    )
    mapped = cur.fetchone()[0]
    if mapped and len(durations) < _MIN_KNOWN_DURATION_RATIO * mapped:
        logger.warning(
            "Catalogue duration backfill: server '%s' listed only %d of %d mapped "
            "tracks; the listing looks unreliable, retrying it on the next start.",
            server_id, len(durations), mapped,
        )
        totals['checked'] += len(groups)
        return
    to_stamp = {}
    false_ids = []
    for item_id, (provider_ids, file_paths) in groups.items():
        _classify_group(item_id, provider_ids, file_paths, durations, totals, to_stamp, false_ids)
        totals['checked'] += 1
        if totals['checked'] % step == 0 or totals['checked'] == total_groups:
            _log_progress(totals, total_groups)
    _stamp_durations(cur, to_stamp)
    totals['removed'] += _unmap_false_groups(cur, server_id, false_ids)
    if false_ids:
        cur.execute(
            "UPDATE music_servers SET updated_at = now() WHERE server_id = %s",
            (server_id,),
        )
    db.commit()


def _run_backfill(db, cur, prefetched=None):
    """Give every older-version NULL-duration row a length (real or sentinel)."""
    groups_by_server = _groups_needing_check(cur)
    total_groups = sum(len(groups) for groups in groups_by_server.values())
    totals = _empty_totals()
    if not total_groups:
        return totals
    _log_start_banner(total_groups, len(groups_by_server))
    step = max(1, total_groups // 10)
    # Fetch every server's catalogue concurrently, THEN write sequentially on
    # the single DB cursor (the fetch is the slow part; the DB writes are not
    # thread-safe and stay on this thread).
    logger.info(
        "Catalogue duration backfill: listing %d server catalogue(s) for track "
        "lengths - this is the slow part (metadata only, no audio downloaded); "
        "startup continues as soon as it returns...", len(groups_by_server),
    )
    durations_by_server = _fetch_all_server_durations(db, groups_by_server, prefetched)
    logger.info("Catalogue duration backfill: server listing done, writing lengths...")
    for server_id, groups in groups_by_server.items():
        _process_server(
            db, cur, server_id, groups, durations_by_server.get(server_id),
            totals, total_groups, step,
        )
    _log_complete(total_groups, totals)
    return totals


def _run_migration(db, cur, prefetched=None):
    try:
        totals = _run_backfill(db, cur, prefetched)
        # HARD version gate: bump every older id that now carries a length (plus any
        # orphan no server maps) up to the current scheme. Rows a skipped/unreliable
        # server left NULL keep their old id and retry next boot; everything else
        # becomes current, so the gate above goes false and this step is skipped
        # forever - an unmappable orphan can no longer keep it alive.
        from tasks.fingerprint_canonicalize import relabel_scheme_to_current
        totals['relabelled'] = relabel_scheme_to_current(cur, only_with_duration=True)
        db.commit()
        return totals
    except Exception:
        _rollback(db)
        logger.exception(
            "Catalogue duration migration failed; it retries on the next start"
        )
        raise


def repair_duplicate_track_maps(conn=None, prefetched_durations=None):
    own_conn = conn is None
    db = conn or connect_raw()
    acquired = False
    cur = None
    try:
        _force_no_autocommit(db)
        cur = db.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_REPAIR_ADVISORY_LOCK,))
        acquired = bool(cur.fetchone()[0])
        if not acquired:
            logger.info(
                "Catalogue duration migration: another replica already holds the "
                "lock; skipping on this one."
            )
            return {'skipped': 'locked'}
        # Hard version gate: no older-scheme ids left -> already migrated -> instant
        # no-op, the server is never listed again (survives orphans with no length).
        if not _old_scheme_rows_exist(cur):
            return {'skipped': 'up_to_date'}
        return _run_migration(db, cur, prefetched_durations)
    finally:
        _release(cur, db, acquired, own_conn)


def _same_folder_conflicts(cur):
    cur.execute(
        "SELECT server_id, item_id FROM track_server_map "
        "WHERE file_path IS NOT NULL "
        "GROUP BY server_id, item_id, regexp_replace(file_path, '/[^/]+$', '') "
        "HAVING count(DISTINCT file_path) > 1"
    )
    by_server = {}
    for server_id, item_id in cur.fetchall():
        by_server.setdefault(str(server_id), set()).add(str(item_id))
    return {server_id: list(ids) for server_id, ids in by_server.items()}


def split_same_folder_merges(conn=None):
    own_conn = conn is None
    db = conn or connect_raw()
    acquired = False
    cur = None
    try:
        _force_no_autocommit(db)
        cur = db.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_SAME_FOLDER_ADVISORY_LOCK,))
        acquired = bool(cur.fetchone()[0])
        if not acquired:
            return {'skipped': 'locked'}
        by_server = _same_folder_conflicts(cur)
        if not by_server:
            return {'split': 0, 'removed': 0}
        split = 0
        removed = 0
        for server_id, item_ids in by_server.items():
            removed += _unmap_false_groups(cur, server_id, item_ids)
            split += len(item_ids)
            cur.execute(
                "UPDATE music_servers SET updated_at = now() WHERE server_id = %s",
                (server_id,),
            )
        db.commit()
        logger.info(
            "Same-folder cleanup: split %d merged group(s) that held two distinct "
            "files from one folder (%d mapping(s) unmapped; each re-analyzes under "
            "its own id).",
            split, removed,
        )
        return {'split': split, 'removed': removed}
    except Exception:
        _rollback(db)
        logger.exception("Same-folder cleanup failed; it retries on the next start")
        return {'error': 'failed'}
    finally:
        _release(cur, db, acquired, own_conn, _SAME_FOLDER_ADVISORY_LOCK)


def _group_chromaprints_disagree(fingerprints):
    # True only when TWO stored fingerprints in the group definitively DISAGREE
    # (chromaprints_agree returns False). Missing/undecodable ones return None and
    # are ignored, so a group we cannot fully judge is left merged (skip-if-missing).
    present = [fp for fp in fingerprints if fp is not None]
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            if chromaprints_agree(present[i], present[j]) is False:
                return True
    return False


def _chromaprint_false_merges(cur):
    # Merges to split: a group of TWO OR MORE files on ONE server sharing an item_id
    # whose stored Chromaprints prove at least one pair is a different recording.
    # Only same-server duplicate groups are considered - the same song legitimately
    # maps across servers, which count(*) > 1 per (server_id, item_id) never sees.
    cur.execute(
        "SELECT tsm.server_id, tsm.item_id, cp.fingerprint "
        "FROM track_server_map tsm "
        "JOIN ( "
        "  SELECT server_id, item_id FROM track_server_map "
        "  GROUP BY server_id, item_id HAVING count(*) > 1 "
        ") dup ON dup.server_id = tsm.server_id AND dup.item_id = tsm.item_id "
        "LEFT JOIN chromaprint cp "
        "  ON cp.server_id = tsm.server_id AND cp.provider_track_id = tsm.provider_track_id"
    )
    groups = {}
    for server_id, item_id, fingerprint in cur.fetchall():
        groups.setdefault((str(server_id), str(item_id)), []).append(fingerprint)
    by_server = {}
    for (server_id, item_id), fingerprints in groups.items():
        if _group_chromaprints_disagree(fingerprints):
            by_server.setdefault(server_id, []).append(item_id)
    return by_server


def split_chromaprint_false_merges(conn=None):
    own_conn = conn is None
    db = conn or connect_raw()
    acquired = False
    cur = None
    try:
        _force_no_autocommit(db)
        cur = db.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_CHROMAPRINT_ADVISORY_LOCK,))
        acquired = bool(cur.fetchone()[0])
        if not acquired:
            return {'skipped': 'locked'}
        by_server = _chromaprint_false_merges(cur)
        if not by_server:
            return {'split': 0, 'removed': 0}
        split = 0
        removed = 0
        for server_id, item_ids in by_server.items():
            removed += _unmap_false_groups(cur, server_id, item_ids)
            split += len(item_ids)
            cur.execute(
                "UPDATE music_servers SET updated_at = now() WHERE server_id = %s",
                (server_id,),
            )
        db.commit()
        logger.info(
            "Chromaprint dedup: thanks to Chromaprint, %d false merge(s) were split - "
            "merged groups whose files Chromaprint proved are different recordings "
            "(%d mapping(s) unmapped; each re-analyzes under its own id).",
            split, removed,
        )
        return {'split': split, 'removed': removed}
    except Exception:
        _rollback(db)
        logger.exception("Chromaprint dedup failed; it retries on the next run")
        return {'error': 'failed'}
    finally:
        _release(cur, db, acquired, own_conn, _CHROMAPRINT_ADVISORY_LOCK)
