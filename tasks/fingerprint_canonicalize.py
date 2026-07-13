# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Relabel legacy catalogue rows so item_id becomes the embedding signature.

The canonical id is the 200-bit per-dimension sign signature of each track's
stored MusiCNN embedding (tasks.simhash), so this is a pure database operation:
no downloads, no binaries, no audio decoding. It runs ONCE per lifetime of a
legacy row, at Flask container startup, and is an instant no-op afterwards;
analysis mints canonical ids directly at analyze time so nothing here runs
during analysis. Signatures are computed vectorized in chunks across a small
thread pool so large legacy installs migrate as fast as the machine allows.
The rewrite uses the same proven transactional key-rewrite the
provider-migration feature uses (score, playlist, and all embedding tables,
with the embedding foreign keys dropped and re-added around it). A legacy row
merges into an existing catalogue row ONLY when its signature is within
tolerance AND the exact raw-embedding cosine confirms it is the same audio
(the Similar Songs duplicate rule) - identity is decided purely by the
content. The source server's real ids are preserved in track_server_map so
output can be translated back; rows without an embedding keep their provider
id and keep working via the identity translation fallback.

Main Features:
* One-time, idempotent startup relabel of legacy rows, resolved for the WHOLE
  catalogue in one vectorized pass (~5s for 200k tracks, where the per-track
  loop it replaces took ~10 minutes: that loop was quadratic and, being Python
  bit twiddling under the GIL, could not be rescued by any thread pool).
* Cosine-confirmed duplicate merge into existing canonical rows.
* Records the source-server mapping in track_server_map, streamed in with COPY.
"""

import io
import logging
import time

import numpy as np

from database import connect_raw
from tasks import simhash
from tasks.mediaserver import registry
from tasks.provider_migration_tasks import (
    find_fk,
    _drop_fk_constraints,
    _readd_fk_constraints,
)

logger = logging.getLogger(__name__)

_CHUNK_ROWS = 20000
_CURRENT_SCHEME_SQL = "(s.item_id LIKE 'fp\\_2%%' AND length(s.item_id) = %s)"
_LEGACY_ROW_SQL = "NOT " + _CURRENT_SCHEME_SQL
_RELABEL_ADVISORY_LOCK = 726354822


def _load_catalogue(cur, sql, params, count, ids, matrix, offset):
    """Stream ``count`` (item_id, embedding) rows into ``matrix`` from ``offset``.

    A server-side cursor so the whole result never lands in the client at once;
    each blob is written straight into its row of the preallocated float32
    matrix, so the embeddings exist exactly once in memory.
    """
    scan = cur.connection.cursor(name='migration_scan_%d' % offset)
    scan.itersize = _CHUNK_ROWS
    try:
        scan.execute(sql, params)
        row_index = offset
        while True:
            rows = scan.fetchmany(_CHUNK_ROWS)
            if not rows:
                break
            for item_id, blob in rows:
                vector = np.frombuffer(blob, dtype=np.float32)
                if vector.size != simhash.SIGNATURE_BITS:
                    continue
                matrix[row_index] = vector
                ids.append(str(item_id))
                row_index += 1
        return row_index - offset
    finally:
        scan.close()


def _build_mapping(cur):
    """{legacy_id: canonical_id} to relabel plus {legacy_id: existing_id} to merge.

    Legacy rows are everything whose item_id is not a current-scheme signature
    id: provider ids and ids minted by retired schemes alike. The legacy COUNT
    runs FIRST, so a fully migrated catalogue returns instantly without loading
    anything.

    Identity is resolved for the WHOLE catalogue in one vectorized pass
    (``simhash.resolve_catalog``): the signatures are hashed as one matrix, the
    banded blocking and the Hamming filter are a handful of big array
    operations, and the exact cosine confirms every surviving pair at once. The
    per-track loop this replaces was quadratic AND single-core - it spent its
    life in Python bit twiddling under the GIL, which no thread pool can help -
    and it dominated the migration (~9.5 minutes for 188k tracks, versus a few
    seconds here). The answer is identical: a track still merges into the
    nearest earlier row that the cosine confirms.
    """
    head_len = simhash.CANONICAL_ID_LEN
    cur.execute(
        "SELECT COUNT(*) FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        "WHERE e.embedding IS NOT NULL AND " + _LEGACY_ROW_SQL,
        (head_len,),
    )
    total = cur.fetchone()[0]
    if not total:
        return {}, {}
    cur.execute(
        "SELECT COUNT(*) FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        "WHERE e.embedding IS NOT NULL AND " + _CURRENT_SCHEME_SQL,
        (head_len,),
    )
    canonical_total = cur.fetchone()[0]

    logger.info("=" * 64)
    logger.info(
        "LEGACY CATALOGUE MIGRATION STARTING: computing content ids for "
        "%d tracks from their stored embeddings.", total,
    )
    logger.info(
        "One-time step (first start after upgrade only); no downloads, pure "
        "database work. Holding %d MB of embeddings.",
        ((total + canonical_total) * simhash.SIGNATURE_BITS * 4) // (1024 * 1024),
    )
    logger.info("=" * 64)

    ids = []
    matrix = np.zeros(
        (total + canonical_total, simhash.SIGNATURE_BITS), dtype=np.float32
    )

    started = time.monotonic()
    # Canonical rows first: "earlier wins", so an existing catalogue id is always
    # the one a legacy duplicate merges INTO, never the other way round.
    canonical_loaded = _load_catalogue(
        cur,
        "SELECT s.item_id, e.embedding FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        "WHERE e.embedding IS NOT NULL AND " + _CURRENT_SCHEME_SQL,
        (head_len,), canonical_total, ids, matrix, 0,
    )
    legacy_loaded = _load_catalogue(
        cur,
        "SELECT s.item_id, e.embedding FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        "WHERE e.embedding IS NOT NULL AND " + _LEGACY_ROW_SQL,
        (head_len,), total, ids, matrix, canonical_loaded,
    )
    loaded = canonical_loaded + legacy_loaded
    matrix = matrix[:loaded]
    logger.info(
        "Legacy catalogue migration: read %d embeddings in %.1fs; resolving identities...",
        loaded, time.monotonic() - started,
    )

    resolved_at = time.monotonic()
    packed, valid = simhash.signature_matrix(matrix)
    # A canonical row's id already encodes its signature - keep using it, exactly
    # as the streaming resolver did when it registered those rows by id alone.
    for row in range(canonical_loaded):
        signature = simhash.signature_from_canonical_id(ids[row])
        if signature is None:
            valid[row] = False
            continue
        packed[row] = simhash._pack_signature(signature)
        valid[row] = True

    parent = simhash.resolve_catalog(packed, valid, matrix)

    mapping = {}
    duplicate_mapping = {}
    canonical_of = dict(enumerate(ids[:canonical_loaded]))
    taken = set(ids[:canonical_loaded])
    for row in range(canonical_loaded, loaded):
        if not valid[row]:
            continue
        legacy_id = ids[row]
        target = int(parent[row])
        if target != row:
            duplicate_mapping[legacy_id] = canonical_of[target]
            continue
        minted = simhash.mint_canonical_id(simhash._unpack_signature(packed[row]), taken)
        canonical_of[row] = minted
        taken.add(minted)
        mapping[legacy_id] = minted
    logger.info(
        "Legacy catalogue migration: resolved %d tracks in %.1fs "
        "(%d new content ids, %d duplicates merged).",
        legacy_loaded, time.monotonic() - resolved_at,
        len(mapping), len(duplicate_mapping),
    )
    return mapping, duplicate_mapping


def _merge_duplicate_rows(cur, duplicate_mapping):
    """Merge provider-keyed duplicate analysis rows into existing canonical rows.

    The source track_server_map rows are snapshotted and deleted before the
    canonical copies are inserted, so the per-server provider-id unique index
    is never violated while both keys exist.
    """
    if not duplicate_mapping:
        return
    cur.execute(
        "CREATE TEMP TABLE duplicate_item_id_map ("
        "old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL) ON COMMIT DROP"
    )
    _copy_pairs(cur, 'duplicate_item_id_map', duplicate_mapping)
    cur.execute(
        "CREATE TEMP TABLE duplicate_server_map_rows ON COMMIT DROP AS "
        "SELECT d.new_id, t.server_id, t.provider_track_id, t.match_tier "
        "FROM track_server_map t JOIN duplicate_item_id_map d ON d.old_id = t.item_id"
    )
    cur.execute(
        "DELETE FROM track_server_map t USING duplicate_item_id_map d "
        "WHERE t.item_id = d.old_id"
    )
    cur.execute(
        "INSERT INTO track_server_map "
        "(item_id, server_id, provider_track_id, match_tier, updated_at) "
        "SELECT r.new_id, r.server_id, r.provider_track_id, r.match_tier, now() "
        "FROM duplicate_server_map_rows r "
        "ON CONFLICT (item_id, server_id) DO NOTHING"
    )
    cur.execute(
        "INSERT INTO playlist (playlist_name, item_id, title, author) "
        "SELECT p.playlist_name, d.new_id, p.title, p.author "
        "FROM playlist p JOIN duplicate_item_id_map d ON d.old_id = p.item_id "
        "ON CONFLICT (playlist_name, item_id) DO NOTHING"
    )
    cur.execute(
        "DELETE FROM playlist p USING duplicate_item_id_map d WHERE p.item_id = d.old_id"
    )
    cur.execute(
        "DELETE FROM score s USING duplicate_item_id_map d WHERE s.item_id = d.old_id"
    )


def _default_provider_ids(cur, default_id, item_ids):
    """Preserve current default-server ids before catalogue keys are rewritten."""
    if not item_ids:
        return {}
    cur.execute(
        "SELECT item_id, provider_track_id FROM track_server_map "
        "WHERE server_id = %s AND item_id = ANY(%s)",
        (default_id, list(item_ids)),
    )
    return {str(item_id): str(provider_id) for item_id, provider_id in cur.fetchall()}


def _copy_pairs(cur, table, mapping):
    """COPY a {old_id: new_id} mapping into ``table`` (id, id) - one stream, no
    per-row round trips."""
    buffer = io.StringIO()
    for old_id, new_id in mapping.items():
        buffer.write(
            "%s\t%s\n"
            % (
                str(old_id).replace('\t', ' ').replace('\n', ' '),
                str(new_id).replace('\t', ' ').replace('\n', ' '),
            )
        )
    buffer.seek(0)
    cur.copy_expert("COPY %s (old_id, new_id) FROM STDIN" % table, buffer)


def _populate_relabel_map(cur, mapping):
    cur.execute(
        "CREATE TEMP TABLE item_id_relabel_map ("
        "old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL UNIQUE) ON COMMIT DROP"
    )
    _copy_pairs(cur, 'item_id_relabel_map', mapping)
    cur.execute("ANALYZE item_id_relabel_map")


def _relabel_item_ids(cur, lyrics_exists):
    """Single-pass key rewrite: every table is written exactly once.

    New fp_2 signature ids can never equal any legacy id (different shape) and
    are unique among themselves, so the collision-safe two-phase prefix rewrite
    the provider-migration uses is unnecessary here - skipping the second pass
    halves the write volume on the embedding tables, which dominate the
    migration time.
    """
    tables = ["score", "playlist", "embedding", "clap_embedding"]
    if lyrics_exists:
        tables.append("lyrics_embedding")
    for table in tables:
        cur.execute(
            f"UPDATE {table} t SET item_id = m.new_id "
            f"FROM item_id_relabel_map m WHERE t.item_id = m.old_id"
        )
        logger.info(
            "Legacy catalogue migration: relabelled %d rows in %s",
            cur.rowcount, table,
        )


def _copy_track_server_map(cur, source_id, all_changes, default_provider_ids):
    """Stream the preserved provider ids in with COPY, not row-by-row INSERTs.

    One 200k-row COPY into an unlogged staging table beats tens of thousands of
    parameterised VALUES tuples: the client does no per-row round trip and the
    server does no per-row parse.
    """
    if not all_changes:
        return
    buffer = io.StringIO()
    for old_id, canonical in all_changes.items():
        provider_id = default_provider_ids.get(str(old_id), str(old_id))
        buffer.write(
            "%s\t%s\t%s\n"
            % (
                canonical.replace('\t', ' '),
                source_id,
                str(provider_id).replace('\t', ' ').replace('\n', ' '),
            )
        )
    buffer.seek(0)
    cur.execute(
        "CREATE TEMP TABLE incoming_default_map "
        "(item_id TEXT, server_id TEXT, provider_track_id TEXT) ON COMMIT DROP"
    )
    cur.copy_expert(
        "COPY incoming_default_map (item_id, server_id, provider_track_id) FROM STDIN",
        buffer,
    )
    cur.execute(
        "INSERT INTO track_server_map "
        "(item_id, server_id, provider_track_id, match_tier, updated_at) "
        "SELECT item_id, server_id, provider_track_id, 'default', now() "
        "FROM incoming_default_map "
        "ON CONFLICT (item_id, server_id) DO NOTHING"
    )


def _enqueue_index_rebuild():
    """Queue one index rebuild after a successful relabel so similarity
    features resolve the new ids without waiting for the next analysis."""
    try:
        from app_helper import rq_queue_default

        rq_queue_default.enqueue(
            'tasks.analysis.rebuild_all_indexes_task',
            job_timeout=-1,
            description='Post-migration index rebuild',
        )
        logger.info(
            "Enqueued the post-migration index rebuild; similarity features "
            "will use the new catalogue ids as soon as it completes."
        )
    except Exception:
        logger.warning(
            "Could not enqueue the post-migration index rebuild; the next "
            "analysis will rebuild the indexes instead.",
            exc_info=True,
        )


def canonicalize_fingerprinted_ids(conn=None, log_fn=None, source_server_id=None):
    """Relabel legacy item_ids to the canonical signature id.

    Pure database alignment: no downloads. When rows were actually relabelled,
    an index rebuild job is enqueued right after the commit so the similarity
    features work on the new ids immediately instead of waiting for the next
    analysis. ``log_fn`` receives ``(message, progress)`` step updates for a
    caller's progress bar. The session's statement_timeout is lifted and
    autocommit forced off for the rewrite (both restored on a caller-provided
    connection) so large catalogues are not cancelled mid-relabel.
    """
    def _log(message):
        if log_fn is not None:
            try:
                log_fn(message, None)
            except Exception:
                logger.debug("Canonicalization progress callback failed", exc_info=True)

    own_conn = conn is None
    db = conn or connect_raw()
    prev_autocommit = getattr(db, 'autocommit', None) if not own_conn else None
    try:
        db.autocommit = False
    except Exception:
        pass
    cur = db.cursor()
    relabelled = 0
    duplicates = 0
    prev_timeout = None
    try:
        if not own_conn:
            cur.execute("SHOW statement_timeout")
            prev_timeout = cur.fetchone()[0]
        cur.execute("SET statement_timeout = 0")
        # Several Flask replicas boot at once on a multi-replica deployment.
        # This lock makes exactly one of them do the relabel: the others wait,
        # then find nothing left to migrate and return immediately, instead of
        # racing the same key rewrite and DDL through the FK drop/re-add.
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_RELABEL_ADVISORY_LOCK,))
        source_id = source_server_id or registry.get_default_server_id(db)
        if source_id is None:
            logger.warning(
                "Canonicalization skipped: no default server row exists to preserve the "
                "provider ids; relabelling now would lose them"
            )
            return {'skipped': 'no_default'}
        _log("Computing canonical ids from stored embeddings...")
        mapping, duplicate_mapping = _build_mapping(cur)
        duplicates = len(duplicate_mapping)
        if not mapping and not duplicate_mapping:
            db.commit()
            return {'relabelled': 0, 'duplicates': duplicates}
        _log(
            f"Rewriting {len(mapping)} catalogue keys and merging "
            f"{duplicates} duplicate rows..."
        )
        logger.info(
            "Legacy catalogue migration: rewriting %d catalogue keys and merging "
            "%d duplicate rows (single pass per table)...",
            len(mapping), duplicates,
        )
        cur.execute("SET LOCAL synchronous_commit = off")
        all_changes = dict(mapping)
        all_changes.update(duplicate_mapping)
        default_provider_ids = _default_provider_ids(cur, source_id, all_changes)

        fk_embedding = find_fk(cur, 'embedding', 'item_id')
        fk_clap = find_fk(cur, 'clap_embedding', 'item_id')
        cur.execute("SELECT to_regclass('public.lyrics_embedding') IS NOT NULL")
        lyrics_exists = bool(cur.fetchone()[0])
        fk_lyrics = find_fk(cur, 'lyrics_embedding', 'item_id') if lyrics_exists else None

        _drop_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
        if mapping:
            _populate_relabel_map(cur, mapping)
            _relabel_item_ids(cur, lyrics_exists)
        _readd_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
        _merge_duplicate_rows(cur, duplicate_mapping)

        _log("Preserving the server's real track ids in track_server_map...")
        _copy_track_server_map(cur, source_id, all_changes, default_provider_ids)
        cur.execute(
            "UPDATE music_servers SET updated_at = now() WHERE server_id = %s",
            (source_id,),
        )
        db.commit()
        relabelled = len(mapping) + duplicates
        logger.info("=" * 64)
        logger.info(
            "LEGACY CATALOGUE MIGRATION COMPLETE: %d tracks relabelled to "
            "content ids, %d duplicate rows merged into existing ones, "
            "provider ids preserved in track_server_map.",
            len(mapping), duplicates,
        )
        logger.info("=" * 64)
        _enqueue_index_rebuild()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.exception("Fingerprint canonicalization failed; catalogue left unchanged")
        raise
    finally:
        if not own_conn and prev_timeout is not None:
            try:
                cur.execute("SET statement_timeout = %s", (prev_timeout,))
                db.commit()
            except Exception:
                logger.debug("Could not restore statement_timeout", exc_info=True)
        cur.close()
        if not own_conn and prev_autocommit is not None:
            try:
                db.autocommit = prev_autocommit
            except Exception:
                logger.debug("Could not restore autocommit", exc_info=True)
        if own_conn:
            db.close()

    return {'relabelled': relabelled, 'duplicates': duplicates}
