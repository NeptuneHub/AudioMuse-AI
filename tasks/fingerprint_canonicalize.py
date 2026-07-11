# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Relabel the catalogue so item_id becomes the content fingerprint.

The canonical id is the SimHash-LSH of each track's stored MusiCNN embedding, so
this is a pure database operation - no downloads, no audio decoding, instant
even on a 180k-track legacy install. It runs at the end of every analysis and at
the start of every server sweep, rewriting each analyzed track's item_id from
the media server's id to the canonical ``fp_<hex>`` id using the same proven,
transactional key-rewrite the provider-migration feature uses (score, playlist,
and all embedding tables, with the embedding foreign keys dropped and re-added
around it). The server's real ids are preserved in track_server_map so output
can be translated back; rows without an embedding keep their provider id and
keep working via the identity translation fallback.

Main Features:
* Idempotent, collision-safe relabel of embedding-bearing rows (duplicates skipped).
* Records the default server mapping and rebuilds all similarity indexes.
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
    cur.execute(
        "SELECT s.item_id, e.embedding FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        "WHERE s.item_id NOT LIKE 'fp\\_%' AND e.embedding IS NOT NULL"
    )
    mapping = {}
    seen = set()
    duplicates = 0
    for old_id, embedding_blob in cur.fetchall():
        canonical = audio_fingerprint.embedding_canonical_id(embedding_blob)
        if canonical is None or canonical == old_id:
            continue
        if canonical in seen:
            duplicates += 1
            continue
        seen.add(canonical)
        mapping[old_id] = canonical
    if mapping:
        canonicals = list(mapping.values())
        cur.execute("SELECT item_id FROM score WHERE item_id = ANY(%s)", (canonicals,))
        taken = {r[0] for r in cur.fetchall()}
        if taken:
            duplicates += sum(1 for c in mapping.values() if c in taken)
            mapping = {o: c for o, c in mapping.items() if c not in taken}
    return mapping, duplicates


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


def canonicalize_fingerprinted_ids(conn=None, rebuild=True, log_fn=None):
    """Relabel fingerprinted item_ids to the canonical fingerprint id.

    ``rebuild=False`` skips the index rebuild - used when the caller (analysis)
    rebuilds the indexes itself right afterwards, so they build once. ``log_fn``
    receives ``(message, progress)`` step updates for a caller's progress bar.
    """
    def _log(message):
        if log_fn is not None:
            try:
                log_fn(message, None)
            except Exception:
                logger.debug("Canonicalization progress callback failed", exc_info=True)

    own_conn = conn is None
    db = conn or connect_raw()
    try:
        db.autocommit = False
    except Exception:
        pass
    cur = db.cursor()
    relabelled = 0
    duplicates = 0
    try:
        default_id = registry.get_default_server_id(db)
        if default_id is None:
            logger.warning(
                "Canonicalization skipped: no default server row exists to preserve the "
                "provider ids; relabelling now would lose them"
            )
            return {'skipped': 'no_default'}
        _log("Computing canonical ids from stored embeddings...")
        mapping, duplicates = _build_mapping(cur)
        if not mapping:
            db.commit()
            return {'relabelled': 0, 'duplicates': duplicates}
        _log(f"Rewriting catalogue keys for {len(mapping)} tracks...")
        default_provider_ids = _default_provider_ids(cur, default_id, mapping)

        fk_embedding = find_fk(cur, 'embedding', 'item_id')
        fk_clap = find_fk(cur, 'clap_embedding', 'item_id')
        cur.execute("SELECT to_regclass('public.lyrics_embedding') IS NOT NULL")
        lyrics_exists = bool(cur.fetchone()[0])
        fk_lyrics = find_fk(cur, 'lyrics_embedding', 'item_id') if lyrics_exists else None

        _drop_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
        _populate_migration_map_table(cur, mapping)
        _rewrite_item_ids(cur, lyrics_exists)
        _readd_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)

        _log("Preserving the server's real track ids in track_server_map...")
        rows = [
            (canonical, default_id, default_provider_ids.get(str(old_id), str(old_id)))
            for old_id, canonical in mapping.items()
        ]
        for i in range(0, len(rows), 1000):
            chunk = rows[i:i + 1000]
            placeholders = ",".join(["(%s, %s, %s, 'default', now())"] * len(chunk))
            flat = [v for row in chunk for v in row]
            cur.execute(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier, updated_at) "
                "VALUES " + placeholders +  # nosec B608 - %s-placeholder string only; values are bound params
                " ON CONFLICT (item_id, server_id) DO UPDATE SET "
                "provider_track_id = EXCLUDED.provider_track_id, updated_at = now()",
                flat,
            )
        db.commit()
        relabelled = len(mapping)
        logger.info(
            "Fingerprint canonicalization: relabelled %d item_ids (%d duplicates skipped)",
            relabelled, duplicates,
        )
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.exception("Fingerprint canonicalization failed; catalogue left unchanged")
        raise
    finally:
        cur.close()
        if own_conn:
            db.close()

    if rebuild and relabelled:
        try:
            from tasks.analysis import rebuild_all_indexes_task
            rebuild_all_indexes_task(log_fn=log_fn)
        except Exception:
            logger.exception("Index rebuild after canonicalization failed; run a rebuild manually")

    return {'relabelled': relabelled, 'duplicates': duplicates}
