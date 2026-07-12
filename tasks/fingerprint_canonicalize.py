# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Relabel the catalogue so item_id becomes the content fingerprint.

The canonical id is a SHA-256 digest of each track's stored Chromaprint. The
Chromaprint itself remains intact in ``score.chromaprint`` for future matching
or migrations. It runs at the end of every analysis and at
the start of every server sweep, rewriting each analyzed track's item_id from
the media server's id to the canonical ``fp_<hex>`` id using the same proven,
transactional key-rewrite the provider-migration feature uses (score, playlist,
and all embedding tables, with the embedding foreign keys dropped and re-added
around it). The server's real ids are preserved in track_server_map so output
can be translated back; rows without a Chromaprint keep their provider id and
keep working via the identity translation fallback.

Main Features:
* Idempotent relabel of Chromaprint-bearing rows, merging duplicate content rows.
* Records the source-server mapping and rebuilds all similarity indexes.
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
        "SELECT s.item_id, s.chromaprint FROM score s "
        "WHERE s.chromaprint IS NOT NULL AND s.chromaprint <> ''"
    )
    candidates = []
    for old_id, chromaprint in cur.fetchall():
        canonical = audio_fingerprint.chromaprint_canonical_id(chromaprint)
        if canonical is None or canonical == old_id:
            continue
        candidates.append((str(old_id), canonical))
    mapping = {}
    duplicate_mapping = {}
    if candidates:
        canonicals = list({canonical for _, canonical in candidates})
        cur.execute("SELECT item_id FROM score WHERE item_id = ANY(%s)", (canonicals,))
        taken = {r[0] for r in cur.fetchall()}
        claimed = set(taken)
        for old_id, canonical in candidates:
            if canonical in claimed:
                duplicate_mapping[old_id] = canonical
            else:
                mapping[old_id] = canonical
                claimed.add(canonical)
    return mapping, duplicate_mapping


def _merge_duplicate_rows(cur, duplicate_mapping, lyrics_exists):
    """Merge provider-keyed duplicate analysis rows into existing canonical rows."""
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
        "INSERT INTO track_server_map "
        "(item_id, server_id, provider_track_id, match_tier, updated_at) "
        "SELECT d.new_id, t.server_id, t.provider_track_id, t.match_tier, now() "
        "FROM track_server_map t JOIN duplicate_item_id_map d ON d.old_id = t.item_id "
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
    # Deleting score now cascades the redundant embeddings and old mappings;
    # the canonical analysis row and copied mappings remain intact.
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
        source_id = source_server_id or registry.get_default_server_id(db)
        if source_id is None:
            logger.warning(
                "Canonicalization skipped: no default server row exists to preserve the "
                "provider ids; relabelling now would lose them"
            )
            return {'skipped': 'no_default'}
        _log("Computing canonical ids from stored Chromaprints...")
        mapping, duplicate_mapping = _build_mapping(cur)
        duplicates = len(duplicate_mapping)
        if not mapping and not duplicate_mapping:
            db.commit()
            return {'relabelled': 0, 'duplicates': duplicates}
        _log(
            f"Rewriting {len(mapping)} catalogue keys and merging "
            f"{duplicates} duplicate rows..."
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
        _merge_duplicate_rows(cur, duplicate_mapping, lyrics_exists)

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
