# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Relabel the catalogue so item_id becomes the content fingerprint.

When CATALOG_FINGERPRINT_AS_ID is on, this rewrites every fingerprinted track's
item_id from the default media server's id to the canonical fingerprint id, using
the same proven, transactional key-rewrite the provider-migration feature uses
(score, playlist, and all embedding tables, with the embedding foreign keys
dropped and re-added around it). The default server's real ids are preserved in
track_server_map so output can be translated back, and the six IVF indexes are
rebuilt afterwards because their on-disk cells still hold the old ids.

Main Features:
* Idempotent, collision-safe relabel of fingerprinted rows (duplicates skipped).
* Records the default server mapping and rebuilds all similarity indexes.
"""

import logging

import config
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
    cur.execute("SELECT item_id, fingerprint FROM score WHERE fingerprint IS NOT NULL")
    mapping = {}
    seen = set()
    duplicates = 0
    for old_id, fp in cur.fetchall():
        canonical = audio_fingerprint.canonical_id_str(fp)
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


def canonicalize_fingerprinted_ids(conn=None):
    """Relabel fingerprinted item_ids to the canonical fingerprint id."""
    if not config.CATALOG_FINGERPRINT_AS_ID:
        return {'skipped': 'disabled'}

    own_conn = conn is None
    db = conn or connect_raw()
    try:
        db.autocommit = False
    except Exception:
        pass
    cur = db.cursor()
    try:
        default_id = registry.get_default_server_id(db)
        mapping, duplicates = _build_mapping(cur)
        if not mapping:
            db.commit()
            return {'relabelled': 0, 'duplicates': duplicates}

        fk_embedding = find_fk(cur, 'embedding', 'item_id')
        fk_clap = find_fk(cur, 'clap_embedding', 'item_id')
        cur.execute("SELECT to_regclass('public.lyrics_embedding') IS NOT NULL")
        lyrics_exists = bool(cur.fetchone()[0])
        fk_lyrics = find_fk(cur, 'lyrics_embedding', 'item_id') if lyrics_exists else None

        _drop_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
        _populate_migration_map_table(cur, mapping)
        _rewrite_item_ids(cur, lyrics_exists)
        _readd_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)

        if default_id:
            for old_id, canonical in mapping.items():
                cur.execute(
                    "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier, updated_at) "
                    "VALUES (%s, %s, %s, 'default', now()) "
                    "ON CONFLICT (item_id, server_id) DO UPDATE SET "
                    "provider_track_id = EXCLUDED.provider_track_id, updated_at = now()",
                    (canonical, default_id, old_id),
                )
        db.commit()
        logger.info(
            "Fingerprint canonicalization: relabelled %d item_ids (%d duplicates skipped); rebuilding indexes",
            len(mapping), duplicates,
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

    try:
        from tasks.analysis import rebuild_all_indexes_task
        rebuild_all_indexes_task()
    except Exception:
        logger.exception("Index rebuild after canonicalization failed; run a rebuild manually")

    return {'relabelled': len(mapping), 'duplicates': duplicates}
