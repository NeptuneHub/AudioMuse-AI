# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Relabel legacy catalogue rows so item_id becomes the embedding signature.

The canonical id is the 200-bit per-dimension sign signature of each track's
stored MusiCNN embedding (tasks.simhash), so this is a pure database
operation: no downloads, binaries or audio decoding. It runs ONCE per lifetime of
a legacy row at Flask startup and is an instant no-op afterwards; identity comes
from the embedding, so swapping the model re-mints every id.

Duplicate candidates come from the audio IVF index the library already built:
only tracks sharing an IVF cell are compared, keeping peak memory linear in the
library. The rewrite uses the same transactional key-rewrite as provider
migration, and repoints the similarity indexes in the same transaction. A legacy
row merges only when it shares an IVF cluster AND the exact raw-embedding cosine
confirms the same audio AND the durations agree within
DURATION_TOLERANCE_SECONDS. Durations come from one paged metadata listing
and are backfilled into score.duration; an unreachable server or missing IVF
index simply merges nothing, which is always safe. Unsignable rows are relabelled
to the server-scoped fp_0 id, so no corruption shape can re-run the migration
forever.

Main Features:
* One-time, idempotent startup relabel; cosine-confirmed duplicate merge
* Source-server mapping recorded in track_server_map, streamed in with COPY
"""

import io
import json
import logging

import taskqueue
import time

import numpy as np

import config
from database import connect_raw
from sanitization import sanitize_string_for_db
from tasks import simhash
from tasks.mediaserver import registry
from tasks.provider_migration_tasks import (
    find_fk,
    _drop_fk_constraints,
    _readd_fk_constraints,
)

logger = logging.getLogger(__name__)

_CHUNK_ROWS = 10000
_CONFIRM_PAIRS = 50000

# Indexes keyed by track id, which a relabel therefore invalidates. The artist
# index and the artist projection are keyed by artist NAME, which a relabel does
# not touch, so they are deliberately absent.
_TRACK_KEYED_INDEXES = (
    config.INDEX_NAME,
    'clap_index',
    'lyrics_index',
    'lyrics_axes_index',
    'sem_grove_index',
)
# Any signature content id (fp_1..fp_9), current or older, is already a resolved
# catalogue row - canonicalize only turns PROVIDER ids into content ids and never
# re-resolves an existing one; bumping the scheme version (fp_2 -> fp_3) is the
# duration migration's cheap relabel, not a re-hash from embeddings.
_CURRENT_SCHEME_SQL = (
    "(s.item_id LIKE 'fp\\_%%' AND length(s.item_id) = %s "
    "AND substring(s.item_id from 4 for 1) BETWEEN '1' AND '9')"
)
# Analysis deliberately keeps a track whose embedding yields no usable signature
# (non-finite or constant) under its PROVIDER id, and records that with the
# 'analysis' match tier. Such a row can never be relabelled, so counting it as
# legacy work made this "one-time" migration re-hash the whole catalogue on EVERY
# boot and relabel nothing.
# The fp_0 head IS the marker for a row minted as unsignable, and unlike the map
# row it cannot be taken away: a provider migration unbinds an unmatched song from
# its server, which used to strip the only evidence and hand the row straight back
# to this migration as legacy work it can never relabel.
_UNSIGNABLE_SQL = (
    "((s.item_id LIKE 'fp\\_0%%' AND length(s.item_id) = "
    + str(simhash.CANONICAL_ID_LEN)
    + ") OR EXISTS (SELECT 1 FROM track_server_map t "
    "WHERE t.item_id = s.item_id AND t.match_tier = 'analysis'))"
)
_LEGACY_ROW_SQL = "NOT " + _CURRENT_SCHEME_SQL + " AND NOT " + _UNSIGNABLE_SQL
_RELABEL_ADVISORY_LOCK = 726354822


def _hash_catalogue(cur, sql, params, ids, packed, valid, offset):
    scan = cur.connection.cursor(name='migration_scan_%d' % offset)
    scan.itersize = _CHUNK_ROWS
    row_index = offset
    try:
        scan.execute(sql, params)
        while True:
            rows = scan.fetchmany(_CHUNK_ROWS)
            if not rows:
                break
            batch = np.zeros((len(rows), simhash.SIGNATURE_BITS), dtype=np.float32)
            kept = 0
            for item_id, blob in rows:
                if blob is not None and len(blob) % 4 == 0:
                    vector = np.frombuffer(blob, dtype=np.float32)
                    if vector.size == simhash.SIGNATURE_BITS:
                        batch[kept] = vector
                ids.append(str(item_id))
                kept += 1
            if not kept:
                continue
            batch_packed, batch_valid = simhash.signature_matrix(batch[:kept])
            packed[row_index:row_index + kept] = batch_packed
            valid[row_index:row_index + kept] = batch_valid
            row_index += kept
        return row_index - offset
    finally:
        scan.close()


def _fetch_provider_durations(source_id, conn):
    from tasks import provider_probe
    from tasks.mediaserver import context as ms_context

    try:
        server = registry.get_server(source_id, conn=conn)
        if server is None:
            logger.warning(
                "Legacy catalogue migration: no server row for %s; track durations "
                "unavailable, duplicate merging disabled for this run.", source_id,
            )
            return {}
        logger.info(
            "Legacy catalogue migration: fetching track durations from the music "
            "server (metadata listing only, no downloads)..."
        )
        with ms_context.use_server(server):
            tracks = provider_probe.fetch_all_tracks(
                server['server_type'], server['creds'], apply_filter=False
            )
        durations = {
            sanitize_string_for_db(str(track['id'])): track['duration']
            for track in tracks
            if track.get('id') is not None and track.get('duration') is not None
        }
        logger.info(
            "Legacy catalogue migration: got durations for %d of %d server tracks.",
            len(durations), len(tracks),
        )
        return durations
    except Exception:
        logger.exception(
            "Legacy catalogue migration: could not fetch track durations from the "
            "music server; duplicate merging disabled for this run (every legacy "
            "track keeps its own id, nothing is lost)."
        )
        return {}


def _durations_for_rows(cur, ids, rows, provider_durations, source_id):
    wanted = list({ids[int(row)] for row in rows})
    durations = {}
    for begin in range(0, len(wanted), _CHUNK_ROWS):
        chunk = wanted[begin:begin + _CHUNK_ROWS]
        cur.execute(
            "SELECT item_id, duration FROM score "
            "WHERE duration IS NOT NULL AND item_id = ANY(%s)",
            (chunk,),
        )
        for item_id, duration in cur.fetchall():
            durations[str(item_id)] = float(duration)
    unresolved = [i for i in wanted if i not in durations]
    for item_id in unresolved:
        value = provider_durations.get(sanitize_string_for_db(item_id))
        if value is not None:
            durations[item_id] = value
    unresolved = [i for i in unresolved if i not in durations]
    if unresolved and provider_durations:
        for begin in range(0, len(unresolved), _CHUNK_ROWS):
            chunk = unresolved[begin:begin + _CHUNK_ROWS]
            cur.execute(
                "SELECT item_id, provider_track_id FROM track_server_map "
                "WHERE server_id = %s AND item_id = ANY(%s)",
                (source_id, chunk),
            )
            for item_id, provider_id in cur.fetchall():
                value = provider_durations.get(sanitize_string_for_db(str(provider_id)))
                if value is not None:
                    durations.setdefault(str(item_id), value)
    return durations


def _confirm_slice(fetch, ids, left_slice, right_slice, duration_of):
    rows = np.unique(np.concatenate((left_slice, right_slice)))
    vectors = np.zeros((rows.size, simhash.SIGNATURE_BITS), dtype=np.float32)
    slot_of = {ids[int(row)]: slot for slot, row in enumerate(rows)}
    fingerprint_of = {}
    for chunk in range(0, rows.size, _CHUNK_ROWS):
        wanted = [ids[int(row)] for row in rows[chunk:chunk + _CHUNK_ROWS]]
        fetch.execute(
            "SELECT item_id, embedding FROM embedding WHERE item_id = ANY(%s)",
            (wanted,),
        )
        for item_id, blob in fetch.fetchall():
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.size == simhash.SIGNATURE_BITS:
                vectors[slot_of[str(item_id)]] = vector
        if config.CHROMAPRINT_GATE_ENABLED:
            fetch.execute(
                "SELECT DISTINCT ON (m.item_id) m.item_id, c.fingerprint "
                "FROM track_server_map m JOIN chromaprint c "
                "ON c.server_id = m.server_id "
                "AND c.provider_track_id = m.provider_track_id "
                "WHERE m.item_id = ANY(%s) AND c.fingerprint IS NOT NULL",
                (wanted,),
            )
            for item_id, blob in fetch.fetchall():
                if blob is not None:
                    fingerprint_of[str(item_id)] = bytes(blob)
    confirmed = simhash.confirm_pairs(
        vectors[np.searchsorted(rows, left_slice)],
        vectors[np.searchsorted(rows, right_slice)],
        left_durations=[duration_of.get(ids[int(row)]) for row in left_slice],
        right_durations=[duration_of.get(ids[int(row)]) for row in right_slice],
        left_fingerprints=(
            [fingerprint_of.get(ids[int(row)]) for row in left_slice]
            if fingerprint_of else None
        ),
        right_fingerprints=(
            [fingerprint_of.get(ids[int(row)]) for row in right_slice]
            if fingerprint_of else None
        ),
    )
    if not confirmed.any():
        empty = np.empty(0, dtype=left_slice.dtype)
        return empty, empty
    return left_slice[confirmed], right_slice[confirmed]


def _ivf_candidate_pairs(cur, ids, valid, loaded, provider_durations, source_id):
    from .paged_ivf import IVF_DIR_TABLE, unpack_directory
    from .index_build_helpers import load_segmented_blob

    empty = np.empty(0, dtype=np.int64)
    conn = cur.connection
    blob = load_segmented_blob(conn, IVF_DIR_TABLE, "%s__ivf_dir" % config.INDEX_NAME)
    if not blob:
        logger.warning(
            "Legacy catalogue migration: no audio IVF index found; every legacy "
            "track keeps its own id (no duplicate merging this run). Rebuild the "
            "similarity index (run analysis) and restart to merge duplicates."
        )
        return empty, empty
    _centroids, id2cell, index_item_ids = unpack_directory(blob)[:3]
    row_of = {ids[row]: row for row in range(loaded) if valid[row]}
    cell_of_row = []
    rows_present = []
    for vec_id, item_id in enumerate(index_item_ids):
        row = row_of.get(item_id)
        if row is not None:
            cell_of_row.append(int(id2cell[vec_id]))
            rows_present.append(row)
    if len(rows_present) < 2:
        return empty, empty
    cells = np.asarray(cell_of_row, dtype=np.int64)
    rows_arr = np.asarray(rows_present, dtype=np.int64)
    order_pos = np.argsort(cells, kind="stable")
    order = rows_arr[order_pos]
    _uniq, starts, sizes = np.unique(
        cells[order_pos], return_index=True, return_counts=True
    )
    crowded = sizes > 1
    if not crowded.any():
        return empty, empty
    group_of_pos = np.repeat(np.arange(sizes.size), sizes)
    involved = order[crowded[group_of_pos]]
    duration_of = _durations_for_rows(cur, ids, involved, provider_durations, source_id)
    row_duration = np.full(loaded, np.nan)
    for item_id, value in duration_of.items():
        row = row_of.get(item_id)
        if row is not None and value is not None:
            row_duration[row] = value
    logger.info(
        "Legacy catalogue migration: %d IVF cluster(s) hold multiple tracks; "
        "confirming duplicates within each cluster (slices of %d)...",
        int(crowded.sum()), _CONFIRM_PAIRS,
    )
    kept_left = []
    kept_right = []
    pending_first = []
    pending_second = []
    pending = 0
    fetch = conn.cursor()

    def _flush():
        nonlocal pending
        if not pending_first:
            return
        kept_l, kept_r = _confirm_slice(
            fetch, ids,
            np.concatenate(pending_first), np.concatenate(pending_second),
            duration_of,
        )
        pending_first.clear()
        pending_second.clear()
        pending = 0
        if kept_l.size:
            kept_left.append(kept_l)
            kept_right.append(kept_r)

    try:
        for first, second in simhash._iter_group_pairs(
            order, starts[crowded], sizes[crowded], limit=_CONFIRM_PAIRS
        ):
            first = first.astype(np.int64)
            second = second.astype(np.int64)
            compatible = simhash.duration_mask_arrays(
                row_duration[first], row_duration[second]
            )
            if not compatible.any():
                continue
            pending_first.append(first[compatible])
            pending_second.append(second[compatible])
            pending += int(compatible.sum())
            if pending >= _CONFIRM_PAIRS:
                _flush()
        _flush()
    finally:
        fetch.close()
    if not kept_left:
        return empty, empty
    left = np.concatenate(kept_left)
    right = np.concatenate(kept_right)
    return np.minimum(left, right), np.maximum(left, right)


def _folders_for_rows(cur, ids, left, right, count):
    folders = [None] * count
    if left.size == 0:
        return folders
    involved = sorted({int(row) for row in np.concatenate((left, right))})
    path_of = {}
    wanted = list({ids[row] for row in involved})
    for begin in range(0, len(wanted), _CHUNK_ROWS):
        chunk = wanted[begin:begin + _CHUNK_ROWS]
        cur.execute(
            "SELECT item_id, file_path FROM score "
            "WHERE file_path IS NOT NULL AND item_id = ANY(%s)",
            (chunk,),
        )
        for item_id, file_path in cur.fetchall():
            path_of[str(item_id)] = file_path
    for row in involved:
        folders[row] = simhash.folder_key(path_of.get(ids[row]))
    return folders


def _build_mapping(cur, source_id):
    head_len = simhash.CANONICAL_ID_LEN
    cur.execute(
        "SELECT COUNT(*) FROM score s "
        "LEFT JOIN embedding e ON e.item_id = s.item_id "
        "WHERE " + _LEGACY_ROW_SQL,
        (head_len,),
    )
    total = cur.fetchone()[0]
    if not total:
        return {}, {}, {}
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
        "One-time step (first start after upgrade only); no audio downloads, "
        "database work plus one metadata listing. Streamed in batches of %d tracks.",
        _CHUNK_ROWS,
    )
    logger.info("=" * 64)

    provider_durations = _fetch_provider_durations(source_id, cur.connection)

    ids = []
    rows_total = total + canonical_total
    packed = np.zeros((rows_total, simhash.SIGNATURE_BYTES), dtype=np.uint8)
    valid = np.zeros(rows_total, dtype=bool)

    started = time.monotonic()
    canonical_loaded = _hash_catalogue(
        cur,
        "SELECT s.item_id, e.embedding FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id "
        "WHERE e.embedding IS NOT NULL AND " + _CURRENT_SCHEME_SQL,
        (head_len,), ids, packed, valid, 0,
    )
    legacy_loaded = _hash_catalogue(
        cur,
        "SELECT s.item_id, e.embedding FROM score s "
        "LEFT JOIN embedding e ON e.item_id = s.item_id "
        "WHERE " + _LEGACY_ROW_SQL,
        (head_len,), ids, packed, valid, canonical_loaded,
    )
    loaded = canonical_loaded + legacy_loaded
    packed = packed[:loaded]
    valid = valid[:loaded]

    for row in range(canonical_loaded):
        signature = simhash.signature_from_canonical_id(ids[row])
        if signature is None:
            valid[row] = False
            continue
        packed[row] = simhash._pack_signature(signature)
        valid[row] = True
    logger.info(
        "Legacy catalogue migration: hashed %d embeddings in %.1fs; resolving identities...",
        loaded, time.monotonic() - started,
    )

    resolved_at = time.monotonic()

    left, right = _ivf_candidate_pairs(
        cur, ids, valid, loaded, provider_durations, source_id
    )
    keep = right >= canonical_loaded
    left, right = left[keep], right[keep]
    folders = _folders_for_rows(cur, ids, left, right, loaded)
    parent = simhash.merge_pairs(loaded, packed, left, right, folders=folders)

    mapping = {}
    duplicate_mapping = {}
    canonical_of = dict(enumerate(ids[:canonical_loaded]))
    taken = set(ids[:canonical_loaded])
    unsignable_ids = [
        ids[row] for row in range(canonical_loaded, loaded) if not valid[row]
    ]
    unsignable_provider_of = (
        _default_provider_ids(cur, source_id, unsignable_ids)
        if unsignable_ids else {}
    )
    unsignable = 0
    for row in range(canonical_loaded, loaded):
        legacy_id = ids[row]
        if not valid[row]:
            minted = simhash.unsignable_canonical_id(
                source_id, unsignable_provider_of.get(legacy_id, legacy_id)
            )
            while minted in taken:
                minted = simhash.unsignable_canonical_id(source_id, minted)
            taken.add(minted)
            mapping[legacy_id] = minted
            unsignable += 1
            continue
        target = int(parent[row])
        if target != row:
            duplicate_mapping[legacy_id] = canonical_of[target]
            continue
        minted = simhash.mint_canonical_id(simhash._unpack_signature(packed[row]), taken)
        canonical_of[row] = minted
        taken.add(minted)
        mapping[legacy_id] = minted
    if unsignable:
        logger.warning(
            "Legacy catalogue migration: %d track(s) have no usable embedding "
            "signature (constant or non-finite embedding); catalogued under "
            "server-scoped fp_0 ids, excluded from duplicate matching.",
            unsignable,
        )
    logger.info(
        "Legacy catalogue migration: resolved %d tracks in %.1fs "
        "(%d new content ids, %d duplicates merged).",
        legacy_loaded, time.monotonic() - resolved_at,
        len(mapping), len(duplicate_mapping),
    )
    return mapping, duplicate_mapping, provider_durations


def _merge_duplicate_rows(cur, duplicate_mapping):
    if not duplicate_mapping:
        return
    cur.execute(
        "CREATE TEMP TABLE duplicate_item_id_map ("
        "old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL) ON COMMIT DROP"
    )
    _copy_pairs(cur, 'duplicate_item_id_map', duplicate_mapping)
    cur.execute(
        "CREATE TEMP TABLE duplicate_server_map_rows ON COMMIT DROP AS "
        "SELECT d.new_id, t.server_id, t.provider_track_id, t.match_tier, t.file_path "
        "FROM track_server_map t JOIN duplicate_item_id_map d ON d.old_id = t.item_id"
    )
    cur.execute(
        "DELETE FROM track_server_map t USING duplicate_item_id_map d "
        "WHERE t.item_id = d.old_id"
    )
    cur.execute(
        "INSERT INTO track_server_map "
        "(item_id, server_id, provider_track_id, match_tier, file_path, updated_at) "
        "SELECT r.new_id, r.server_id, r.provider_track_id, r.match_tier, r.file_path, now() "
        "FROM duplicate_server_map_rows r "
        "ON CONFLICT (server_id, provider_track_id) DO NOTHING"
    )
    cur.execute(
        "INSERT INTO playlist (playlist_name, item_id, title, author, server_id) "
        "SELECT DISTINCT ON (p.playlist_name, d.new_id, p.server_id) "
        "p.playlist_name, d.new_id, p.title, p.author, p.server_id "
        "FROM playlist p JOIN duplicate_item_id_map d ON d.old_id = p.item_id "
        "WHERE NOT EXISTS (SELECT 1 FROM playlist q "
        "WHERE q.playlist_name = p.playlist_name AND q.item_id = d.new_id "
        "AND q.server_id IS NOT DISTINCT FROM p.server_id) "
        "ORDER BY p.playlist_name, d.new_id, p.server_id, p.item_id "
        "ON CONFLICT (playlist_name, item_id, server_id) DO NOTHING"
    )
    cur.execute(
        "DELETE FROM playlist p USING duplicate_item_id_map d WHERE p.item_id = d.old_id"
    )
    cur.execute(
        "DELETE FROM score s USING duplicate_item_id_map d WHERE s.item_id = d.old_id"
    )


def _default_provider_ids(cur, default_id, item_ids):
    if not item_ids:
        return {}
    cur.execute(
        "SELECT item_id, provider_track_id FROM track_server_map "
        "WHERE server_id = %s AND item_id = ANY(%s)",
        (default_id, list(item_ids)),
    )
    return {str(item_id): str(provider_id) for item_id, provider_id in cur.fetchall()}


_COPY_ESCAPES = {0x5C: '\\\\', 0x09: '\\t', 0x0A: '\\n', 0x0D: '\\r'}


def _copy_escape(value):
    return str(value).translate(_COPY_ESCAPES)


def _copy_pairs(cur, table, mapping):
    buffer = io.StringIO()
    for old_id, new_id in mapping.items():
        buffer.write(
            "%s\t%s\n"
            % (
                _copy_escape(old_id),
                _copy_escape(new_id),
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


def _legacy_paths_by_item_id(cur):
    cur.execute("SELECT item_id, file_path FROM score WHERE file_path IS NOT NULL")
    return {str(item_id): path for item_id, path in cur.fetchall()}


def _copy_track_server_map(cur, source_id, all_changes, default_provider_ids,
                           legacy_paths, provider_durations=None):
    if not all_changes:
        return
    provider_durations = provider_durations or {}
    buffer = io.StringIO()
    for old_id, canonical in all_changes.items():
        provider_id = str(default_provider_ids.get(str(old_id), str(old_id)))
        path = legacy_paths.get(str(old_id))
        duration = provider_durations.get(provider_id)
        tier = 'analysis' if simhash.is_unsignable_id(canonical) else 'default'
        buffer.write(
            "%s\t%s\t%s\t%s\t%s\t%s\n"
            % (
                _copy_escape(canonical),
                _copy_escape(source_id),
                _copy_escape(provider_id),
                tier,
                r'\N' if not path else _copy_escape(path),
                r'\N' if duration is None else repr(float(duration)),
            )
        )
    buffer.seek(0)
    cur.execute(
        "CREATE TEMP TABLE incoming_default_map "
        "(item_id TEXT, server_id TEXT, provider_track_id TEXT, match_tier TEXT, "
        "file_path TEXT, duration DOUBLE PRECISION) "
        "ON COMMIT DROP"
    )
    cur.copy_expert(
        "COPY incoming_default_map "
        "(item_id, server_id, provider_track_id, match_tier, file_path, duration) "
        "FROM STDIN",
        buffer,
    )
    cur.execute(
        "INSERT INTO track_server_map "
        "(item_id, server_id, provider_track_id, match_tier, file_path, updated_at) "
        "SELECT item_id, server_id, provider_track_id, match_tier, file_path, now() "
        "FROM incoming_default_map "
        "ON CONFLICT (server_id, provider_track_id) DO UPDATE SET "
        "match_tier = CASE WHEN EXCLUDED.match_tier = 'analysis' "
        "THEN EXCLUDED.match_tier ELSE track_server_map.match_tier END, "
        "file_path = COALESCE(EXCLUDED.file_path, track_server_map.file_path)"
    )
    cur.execute(
        "UPDATE score s SET duration = i.duration FROM incoming_default_map i "
        "WHERE s.item_id = i.item_id AND i.duration IS NOT NULL "
        "AND s.duration IS NULL"
    )
    if cur.rowcount:
        logger.info(
            "Legacy catalogue migration: backfilled track duration for %d "
            "catalogue rows from the server metadata.", cur.rowcount,
        )


def _repoint_indexes(cur, renames):
    from .paged_ivf import (
        IVF_DIR_TABLE,
        invalidate_global_cell_cache,
        pack_directory,
        unpack_directory,
    )
    from .index_build_helpers import load_segmented_blob, store_segmented_blob

    if not renames:
        return
    started = time.monotonic()
    conn = cur.connection
    repointed = []
    for name in _TRACK_KEYED_INDEXES:
        try:
            blob = load_segmented_blob(conn, IVF_DIR_TABLE, f"{name}__ivf_dir")
            if not blob:
                continue
            centroids, id2cell, item_ids, dim, metric, normalized, storage = (
                unpack_directory(blob)
            )
            updated = [renames.get(item_id, item_id) for item_id in item_ids]
            if updated == item_ids:
                continue
            store_segmented_blob(
                conn,
                IVF_DIR_TABLE,
                f"{name}__ivf_dir",
                pack_directory(
                    centroids, id2cell, updated, dim, metric,
                    normalized=normalized, storage_dtype=storage,
                ),
                max_part_size_mb=config.IVF_MAX_PART_SIZE_MB,
            )
            invalidate_global_cell_cache(name)
            repointed.append(f"{name} ({len(updated)})")
        except Exception:
            logger.exception(
                "Could not repoint index '%s' at the new ids; it will be rebuilt "
                "by the next analysis", name,
            )

    cur.execute("SELECT index_name, id_map_json FROM map_projection_data")
    for index_name, id_map_json in cur.fetchall():
        try:
            item_ids = json.loads(id_map_json)
            updated = [renames.get(item_id, item_id) for item_id in item_ids]
            if updated == item_ids:
                continue
            cur.execute(
                "UPDATE map_projection_data SET id_map_json = %s WHERE index_name = %s",
                (json.dumps(updated), index_name),
            )
            repointed.append(f"{index_name} ({len(updated)})")
        except Exception:
            logger.exception(
                "Could not repoint map projection '%s' at the new ids", index_name
            )

    logger.info(
        "Legacy catalogue migration: repointed %s at the new catalogue ids in %.1fs "
        "(no re-clustering; every vector, cell and centroid is unchanged).",
        ", ".join(repointed) if repointed else "no index",
        time.monotonic() - started,
    )


def relabel_scheme_to_current(cur, only_with_duration=True):
    from tasks import simhash

    head = simhash.CURRENT_ID_HEAD
    guard, params = simhash.signature_id_sql()
    if only_with_duration:
        guard += (
            " AND (duration IS NOT NULL OR NOT EXISTS ("
            "SELECT 1 FROM track_server_map t WHERE t.item_id = score.item_id))"
        )
    cur.execute("SELECT item_id FROM score WHERE " + guard, params)
    old_ids = [row[0] for row in cur.fetchall()]
    if not old_ids:
        return 0
    cur.execute("SELECT item_id FROM score")
    taken = {row[0] for row in cur.fetchall()}
    taken.difference_update(old_ids)
    mapping = {}
    for old in old_ids:
        new = simhash.mint_canonical_id(simhash.signature_from_canonical_id(old), taken)
        taken.add(new)
        mapping[old] = new

    fk_embedding = find_fk(cur, 'embedding', 'item_id') or 'embedding_item_id_fkey'
    fk_clap = find_fk(cur, 'clap_embedding', 'item_id') or 'clap_embedding_item_id_fkey'
    cur.execute("SELECT to_regclass('public.lyrics_embedding') IS NOT NULL")
    lyrics_exists = bool(cur.fetchone()[0])
    fk_lyrics = (
        (find_fk(cur, 'lyrics_embedding', 'item_id') or 'lyrics_embedding_item_id_fkey')
        if lyrics_exists else None
    )
    _drop_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
    _populate_relabel_map(cur, mapping)
    _relabel_item_ids(cur, lyrics_exists)
    _readd_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
    _repoint_indexes(cur, mapping)
    logger.info(
        "Catalogue scheme relabel: bumped %d ids up to %s.", len(mapping), head,
    )
    return len(mapping)


class CanonicalizationVerificationError(RuntimeError):
    pass


def _index_id_map_lengths(cur):
    from .paged_ivf import IVF_DIR_TABLE, unpack_directory
    from .index_build_helpers import load_segmented_blob

    lengths = {}
    conn = cur.connection
    for name in _TRACK_KEYED_INDEXES:
        try:
            blob = load_segmented_blob(conn, IVF_DIR_TABLE, f"{name}__ivf_dir")
            if not blob:
                continue
            lengths[f"ivf:{name}"] = len(unpack_directory(blob)[2])
        except Exception:
            logger.exception("Could not read the id list of index '%s'", name)
    cur.execute("SELECT index_name, id_map_json FROM map_projection_data")
    for index_name, id_map_json in cur.fetchall():
        try:
            lengths[f"projection:{index_name}"] = len(json.loads(id_map_json))
        except Exception:
            logger.exception("Could not read the id map of projection '%s'", index_name)
    return lengths


def _verify_migration(cur, score_before, duplicates, index_lengths_before):
    problems = []

    cur.execute(
        "SELECT count(*) FROM score s WHERE " + _LEGACY_ROW_SQL, (simhash.CANONICAL_ID_LEN,)
    )
    legacy_left = cur.fetchone()[0]
    if legacy_left:
        problems.append(f"{legacy_left} legacy id(s) survived the relabel")

    cur.execute("SELECT count(*) FROM score")
    score_after = cur.fetchone()[0]
    expected = score_before - duplicates
    if score_after != expected:
        problems.append(
            f"score holds {score_after} rows, expected {expected} "
            f"({score_before} before minus {duplicates} merged)"
        )

    for table in ('embedding', 'clap_embedding', 'lyrics_embedding'):
        cur.execute("SELECT to_regclass(%s)", (f'public.{table}',))
        if cur.fetchone()[0] is None:
            continue
        cur.execute(
            f"SELECT count(*) FROM {table} e "
            "WHERE NOT EXISTS (SELECT 1 FROM score s WHERE s.item_id = e.item_id)"
        )
        orphans = cur.fetchone()[0]
        if orphans:
            problems.append(f"{orphans} {table} row(s) lost their score parent")

    lengths_after = _index_id_map_lengths(cur)
    for name, before in index_lengths_before.items():
        after = lengths_after.get(name)
        if after is None:
            problems.append(f"index '{name}' disappeared during the rewrite")
        elif after != before:
            problems.append(
                f"index '{name}' carried {before} ids before and {after} after"
            )

    if problems:
        raise CanonicalizationVerificationError("; ".join(problems))


def _publish_index_reload():
    try:

        taskqueue.publish_event('index-reload')
        logger.info(
            "Similarity indexes now answer to the new catalogue ids; asked Flask "
            "to reload them."
        )
    except Exception:
        logger.warning(
            "Could not publish the index reload; a running Flask will pick the "
            "repointed indexes up on its next restart.",
            exc_info=True,
        )


def canonicalize_fingerprinted_ids(conn=None, log_fn=None, source_server_id=None):
    def _log(message):
        logger.info("[CatalogueMigration] %s", message)
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
    source_id = None
    provider_durations = {}
    prev_timeout = None
    try:
        if not own_conn:
            cur.execute("SHOW statement_timeout")
            prev_timeout = cur.fetchone()[0]
        cur.execute("SET statement_timeout = 0")
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_RELABEL_ADVISORY_LOCK,))
        source_id = source_server_id or registry.get_default_server_id(db)
        if not source_id:
            logger.warning(
                "Canonicalization skipped: no default server row exists to preserve the "
                "provider ids; relabelling now would lose them"
            )
            return {'skipped': 'no_default'}
        _log("Computing canonical ids from stored embeddings...")
        mapping, duplicate_mapping, provider_durations = _build_mapping(cur, source_id)
        duplicates = len(duplicate_mapping)
        if not mapping and not duplicate_mapping:
            db.commit()
            return {'relabelled': 0, 'duplicates': duplicates}
        _log(
            f"Rewriting {len(mapping)} catalogue keys and merging "
            f"{duplicates} duplicate rows..."
        )
        cur.execute("SET LOCAL synchronous_commit = off")
        all_changes = dict(mapping)
        all_changes.update(duplicate_mapping)
        default_provider_ids = _default_provider_ids(cur, source_id, all_changes)

        cur.execute("SELECT count(*) FROM score")
        score_before = cur.fetchone()[0]
        index_lengths_before = _index_id_map_lengths(cur)
        legacy_paths = _legacy_paths_by_item_id(cur)

        fk_embedding = find_fk(cur, 'embedding', 'item_id') or 'embedding_item_id_fkey'
        fk_clap = (
            find_fk(cur, 'clap_embedding', 'item_id') or 'clap_embedding_item_id_fkey'
        )
        cur.execute("SELECT to_regclass('public.lyrics_embedding') IS NOT NULL")
        lyrics_exists = bool(cur.fetchone()[0])
        fk_lyrics = (
            (find_fk(cur, 'lyrics_embedding', 'item_id') or 'lyrics_embedding_item_id_fkey')
            if lyrics_exists else None
        )

        _drop_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
        if mapping:
            _populate_relabel_map(cur, mapping)
            _relabel_item_ids(cur, lyrics_exists)
        _readd_fk_constraints(cur, fk_embedding, fk_clap, lyrics_exists, fk_lyrics)
        _merge_duplicate_rows(cur, duplicate_mapping)

        _log("Preserving the server's real track ids in track_server_map...")
        _copy_track_server_map(
            cur, source_id, all_changes, default_provider_ids, legacy_paths,
            provider_durations,
        )
        cur.execute(
            "UPDATE music_servers SET updated_at = now() WHERE server_id = %s",
            (source_id,),
        )
        _log("Pointing the similarity indexes at the new ids...")
        _repoint_indexes(cur, all_changes)

        _log("Verifying the rewritten catalogue...")
        _verify_migration(cur, score_before, duplicates, index_lengths_before)

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
        _publish_index_reload()
    except CanonicalizationVerificationError as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.critical("=" * 64)
        logger.critical(
            "CATALOGUE MIGRATION ROLLED BACK - the rewrite did not hold up: %s. "
            "The catalogue is EXACTLY as it was; nothing was committed.", e,
        )
        logger.critical("=" * 64)
        raise
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

    return {
        'relabelled': relabelled,
        'duplicates': duplicates,
        'source_server_id': source_id,
        'server_durations': ({source_id: provider_durations}
                             if source_id and provider_durations else {}),
    }
