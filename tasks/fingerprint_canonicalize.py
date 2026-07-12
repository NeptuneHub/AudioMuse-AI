# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Relabel legacy catalogue rows so item_id becomes the embedding fingerprint.

The canonical id is the similarity-preserving 64-bit SimHash of each track's
stored MusiCNN embedding, so this is a pure database operation: no downloads,
no binaries, no audio decoding, seconds even on a 180k-track legacy install.
It runs ONCE per lifetime of a legacy row, at Flask container startup, and is
an instant no-op afterwards; analysis mints canonical ids directly at analyze
time so nothing here runs during analysis. The rewrite uses the same proven
transactional key-rewrite the provider-migration feature uses (score, playlist,
and all embedding tables, with the embedding foreign keys dropped and re-added
around it). Rows whose hash lands within Hamming tolerance of an existing
catalogue row are the same content and are merged into it, keeping every
server mapping. The source server's real ids are preserved in track_server_map
so output can be translated back; rows without an embedding keep their provider
id and keep working via the identity translation fallback.

Main Features:
* One-time, idempotent startup relabel of embedding-bearing legacy rows.
* Hamming-tolerant duplicate merge into existing canonical rows.
* Records the source-server mapping in track_server_map.
"""

import logging

from database import connect_raw
from tasks import audio_fingerprint
from tasks.mediaserver import registry
from tasks.provider_migration_tasks import (
    find_fk,
    _drop_fk_constraints,
    _populate_migration_map_table,
    _readd_fk_constraints,
    _rewrite_item_ids,
)

logger = logging.getLogger(__name__)


def _build_mapping(cur):
    """{legacy_id: canonical_id} to relabel plus {legacy_id: existing_id} to merge.

    Legacy rows are everything whose item_id is not a current-scheme embedding
    hash id: provider ids and ids minted by retired schemes alike. A legacy row
    whose hash lands within Hamming tolerance of a row already in the catalogue
    (or of another legacy row processed first) is the same content and is merged
    into that id instead of claiming its own.
    """
    cur.execute(
        "SELECT s.item_id FROM score s "
        "WHERE s.item_id LIKE 'fp\\_%' AND length(s.item_id) = 19"
    )
    index = audio_fingerprint.FingerprintIndex.from_item_ids(
        r[0] for r in cur.fetchall()
    )
    cur.execute(
        "SELECT s.item_id, e.embedding FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        "WHERE e.embedding IS NOT NULL "
        "AND (s.item_id NOT LIKE 'fp\\_%' OR length(s.item_id) <> 19)"
    )
    rows = cur.fetchall()
    total = len(rows)
    if total:
        logger.info("=" * 64)
        logger.info(
            "LEGACY CATALOGUE MIGRATION STARTING: computing content ids for "
            "%d tracks from their stored embeddings.", total,
        )
        logger.info(
            "One-time step (first start after upgrade only); no downloads, "
            "pure database work. Progress every 10%%."
        )
        logger.info("=" * 64)
    mapping = {}
    duplicate_mapping = {}
    next_pct = 10
    for done, (old_id, embedding_blob) in enumerate(rows, 1):
        fingerprint = audio_fingerprint.embedding_fingerprint(embedding_blob)
        canonical = audio_fingerprint.canonical_id_str(fingerprint)
        if canonical is not None:
            existing = index.find(fingerprint)
            if existing is not None:
                duplicate_mapping[str(old_id)] = existing
            else:
                mapping[str(old_id)] = canonical
                index.add(canonical, fingerprint)
        if total >= 10 and done * 100 >= next_pct * total:
            logger.info(
                "Legacy catalogue migration: %d%% (%d/%d content ids computed)",
                next_pct, done, total,
            )
            next_pct += 10
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
    rows = list(duplicate_mapping.items())
    for i in range(0, len(rows), 1000):
        chunk = rows[i:i + 1000]
        placeholders = ",".join(["(%s,%s)"] * len(chunk))
        cur.execute(
            "INSERT INTO duplicate_item_id_map (old_id, new_id) VALUES " + placeholders,
            [value for pair in chunk for value in pair],
        )
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


def canonicalize_fingerprinted_ids(
    conn=None, rebuild=True, log_fn=None, source_server_id=None
):
    """Relabel fingerprinted item_ids to the canonical fingerprint id.

    ``rebuild=False`` skips the index rebuild - used when the caller (analysis)
    rebuilds the indexes itself right afterwards, so they build once. ``log_fn``
    receives ``(message, progress)`` step updates for a caller's progress bar.
    The session's statement_timeout is lifted and autocommit forced off for the
    rewrite (both restored on a caller-provided connection) so large catalogues
    are not cancelled mid-relabel.
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
            "%d duplicate rows (transactional, may take a moment)...",
            len(mapping), duplicates,
        )
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
            _populate_migration_map_table(cur, mapping)
            _rewrite_item_ids(cur, lyrics_exists)
        _readd_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
        _merge_duplicate_rows(cur, duplicate_mapping)

        _log("Preserving the server's real track ids in track_server_map...")
        rows = [
            (canonical, source_id, default_provider_ids.get(str(old_id), str(old_id)))
            for old_id, canonical in all_changes.items()
        ]
        for i in range(0, len(rows), 1000):
            chunk = rows[i:i + 1000]
            placeholders = ",".join(["(%s, %s, %s, 'default', now())"] * len(chunk))
            flat = [v for row in chunk for v in row]
            cur.execute(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier, updated_at) "
                "VALUES " + placeholders +  # nosec B608 - %s-placeholder string only; values are bound params
                " ON CONFLICT (item_id, server_id) DO NOTHING",
                flat,
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

    if rebuild and relabelled:
        from tasks.analysis import rebuild_all_indexes_task
        rebuild_all_indexes_task(log_fn=log_fn)

    return {'relabelled': relabelled, 'duplicates': duplicates}
