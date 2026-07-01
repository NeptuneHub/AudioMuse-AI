# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared building blocks for constructing and persisting the IVF indexes.

Provides the streaming embedding readers and the segmented-blob storage layer
used by tasks.ivf_manager and tasks.lyrics_manager to build any of the six
disk-paged IVF indexes without holding whole libraries in RAM. Sits below
tasks.paged_ivf, which owns the on-disk IVF format and query path.

Main Features:
* stream_embeddings_to_buffer / iter_embedding_batches: read pgvector columns
  over a read-only side connection with a server-side named cursor.
* build_and_store_index_streaming and the segmented-blob helpers: split large
  id maps and index payloads into row-sized fragments (SQL identifiers are
  regex-validated before interpolation), plus artist-metadata pack/unpack.
"""

from __future__ import annotations

import io
import json
import logging
import re
import struct
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import psycopg2

import config

logger = logging.getLogger(__name__)


class EmptyIndexError(ValueError):
    pass


_STREAM_ITERSIZE = 5000

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sql_identifier(ident: str, kind: str) -> None:
    if not isinstance(ident, str) or not _IDENT_RE.match(ident):
        raise ValueError(f"Invalid SQL {kind}: {ident!r}")


def _open_side_connection() -> "psycopg2.extensions.connection":
    conn = psycopg2.connect(
        config.DATABASE_URL,
        connect_timeout=30,
        keepalives_idle=600,
        keepalives_interval=30,
        keepalives_count=3,
        options="-c statement_timeout=0",
    )
    try:
        conn.set_session(readonly=True)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise
    return conn


def stream_embeddings_to_buffer(
    table: str,
    column: str,
    dim: int,
    where_clause: Optional[str] = None,
    cursor_name: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    _validate_sql_identifier(table, "table")
    _validate_sql_identifier(column, "column")
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int, got {dim!r}")

    where_sql = f" WHERE {where_clause}" if where_clause else ""
    count_sql = f"SELECT COUNT(*) FROM {table}{where_sql}"
    select_sql = f"SELECT item_id, {column} FROM {table}{where_sql} ORDER BY item_id"
    cname = cursor_name or f"_idx_stream_{table}_{column}"
    _validate_sql_identifier(cname, "cursor name")

    side_conn = _open_side_connection()
    try:
        with side_conn.cursor() as count_cur:
            count_cur.execute(count_sql)
            n_hint = int(count_cur.fetchone()[0])

        if n_hint == 0:
            return np.empty((0, dim), dtype=np.float32), []

        buf = np.empty((n_hint, dim), dtype=np.float32)
        item_ids: List[str] = []
        write_idx = 0
        skipped_null = 0
        skipped_dim = 0

        with side_conn.cursor(name=cname) as sc:
            sc.itersize = _STREAM_ITERSIZE
            sc.execute(select_sql)
            for item_id, blob in sc:
                if blob is None:
                    skipped_null += 1
                    continue
                if len(blob) != dim * 4:
                    skipped_dim += 1
                    continue
                vec = np.frombuffer(blob, dtype=np.float32)
                if write_idx >= buf.shape[0]:
                    new_size = max(buf.shape[0] * 2, write_idx + 1)
                    grown = np.empty((new_size, dim), dtype=np.float32)
                    grown[:write_idx] = buf[:write_idx]
                    buf = grown
                buf[write_idx] = vec
                item_ids.append(item_id)
                write_idx += 1

        if write_idx == 0:
            return np.empty((0, dim), dtype=np.float32), []

        if write_idx < buf.shape[0]:
            buf = buf[:write_idx].copy()

        if skipped_null or skipped_dim:
            logger.warning(
                "stream_embeddings_to_buffer(%s.%s): kept=%d skipped_null=%d skipped_dim=%d",
                table,
                column,
                write_idx,
                skipped_null,
                skipped_dim,
            )
        else:
            logger.info(
                "stream_embeddings_to_buffer(%s.%s): loaded %d rows (dim=%d).",
                table,
                column,
                write_idx,
                dim,
            )

        return buf, item_ids
    finally:
        try:
            side_conn.close()
        except Exception:
            pass


def iter_embedding_batches(
    table: str,
    column: str,
    dim: int,
    batch_size: int = 5000,
    where_clause: Optional[str] = None,
    cursor_name: Optional[str] = None,
) -> Iterator[Tuple[np.ndarray, List[str]]]:
    _validate_sql_identifier(table, "table")
    _validate_sql_identifier(column, "column")
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int, got {dim!r}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")

    where_sql = f" WHERE {where_clause}" if where_clause else ""
    select_sql = f"SELECT item_id, {column} FROM {table}{where_sql} ORDER BY item_id"
    cname = cursor_name or f"_idx_iter_{table}_{column}"
    _validate_sql_identifier(cname, "cursor name")

    side_conn = _open_side_connection()
    try:
        batch_buf = np.empty((batch_size, dim), dtype=np.float32)
        batch_ids: List[str] = []
        write_idx = 0
        total_kept = 0
        total_skipped_null = 0
        total_skipped_dim = 0
        batch_no = 0

        with side_conn.cursor(name=cname) as sc:
            sc.itersize = min(_STREAM_ITERSIZE, batch_size)
            sc.execute(select_sql)
            for item_id, blob in sc:
                if blob is None:
                    total_skipped_null += 1
                    continue
                if len(blob) != dim * 4:
                    total_skipped_dim += 1
                    continue
                vec = np.frombuffer(blob, dtype=np.float32)
                batch_buf[write_idx] = vec
                batch_ids.append(item_id)
                write_idx += 1
                if write_idx >= batch_size:
                    batch_no += 1
                    total_kept += write_idx
                    logger.info(
                        "iter_embedding_batches(%s.%s): batch %d yielded (%d rows).",
                        table,
                        column,
                        batch_no,
                        write_idx,
                    )
                    yield batch_buf, batch_ids
                    batch_buf = np.empty((batch_size, dim), dtype=np.float32)
                    batch_ids = []
                    write_idx = 0

        if write_idx > 0:
            batch_no += 1
            total_kept += write_idx
            logger.info(
                "iter_embedding_batches(%s.%s): batch %d yielded (%d rows, final).",
                table,
                column,
                batch_no,
                write_idx,
            )
            yield batch_buf[:write_idx].copy(), batch_ids

        if total_skipped_null or total_skipped_dim:
            logger.warning(
                "iter_embedding_batches(%s.%s): kept=%d skipped_null=%d skipped_dim=%d across %d batch(es).",
                table,
                column,
                total_kept,
                total_skipped_null,
                total_skipped_dim,
                batch_no,
            )
        else:
            logger.info(
                "iter_embedding_batches(%s.%s): streamed %d rows across %d batch(es), dim=%d.",
                table,
                column,
                total_kept,
                batch_no,
                dim,
            )
    finally:
        try:
            side_conn.close()
        except Exception:
            pass


def _split_bytes(data: bytes, part_size: int) -> List[bytes]:
    return [data[i : i + part_size] for i in range(0, len(data), part_size)]


def _split_text(text: str, max_part_bytes: int) -> List[str]:
    if not text:
        return [""]
    if len(text.encode("utf-8")) <= max_part_bytes:
        return [text]
    step = max(1, max_part_bytes // 4)
    return [text[i : i + step] for i in range(0, len(text), step)]


def reassemble_segmented_id_map(fragments: Iterable[Tuple[int, Optional[str]]]) -> str:
    return "".join(frag or "" for _, frag in sorted(fragments, key=lambda p: p[0]))


def build_and_store_index_streaming(
    db_conn,
    source_table: str,
    source_column: str,
    dim: int,
    target_table: str,
    index_name: str,
    metric: str,
    where_clause: Optional[str] = None,
    label: Optional[str] = None,
) -> bool:
    label = label or index_name

    try:
        from .paged_ivf import build_and_store_paged_ivf

        logger.info("Building %s IVF index (disk-paged)...", label)
        buf, item_ids = stream_embeddings_to_buffer(
            table=source_table,
            column=source_column,
            dim=dim,
            where_clause=where_clause,
        )
        if buf.shape[0] == 0:
            logger.warning("No valid %s vectors found for IVF index build.", label)
            return False
        ok = build_and_store_paged_ivf(db_conn, index_name, buf, item_ids, dim, metric)
        if ok:
            db_conn.commit()
            logger.info("%s IVF index build successful.", label)
        return ok
    except Exception as e:
        logger.error("Failed to build/store %s IVF index: %s", label, e, exc_info=True)
        try:
            db_conn.rollback()
        except Exception:
            pass
        return False


def store_ivf_index_segmented(
    db_conn,
    target_table: str,
    index_name: str,
    index_bytes: bytes,
    id_map: dict,
    embedding_dimension: int,
    max_part_size_mb: Optional[int] = None,
    binary_column: str = "index_data",
) -> None:
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(index_name, "index_name")
    _validate_sql_identifier(binary_column, "column")
    if not index_bytes:
        raise ValueError("index_bytes is empty; refusing to persist an empty index")

    mb = config.IVF_MAX_PART_SIZE_MB if max_part_size_mb is None else int(max_part_size_mb)
    max_part_size = mb * 1024 * 1024
    id_map_json = json.dumps(id_map)

    delete_sql = (
        f"DELETE FROM {target_table} WHERE index_name = %s OR index_name LIKE %s ESCAPE '\\'"
    )
    like_pattern = index_name.replace("_", r"\_") + r"\_%\_%"

    upsert_sql = (
        f"INSERT INTO {target_table} "
        f"(index_name, {binary_column}, id_map_json, embedding_dimension, created_at) "
        f"VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) "
        f"ON CONFLICT (index_name) DO UPDATE SET "
        f"{binary_column} = EXCLUDED.{binary_column}, "
        f"id_map_json = EXCLUDED.id_map_json, "
        f"embedding_dimension = EXCLUDED.embedding_dimension, "
        f"created_at = EXCLUDED.created_at"
    )
    insert_sql = (
        f"INSERT INTO {target_table} "
        f"(index_name, {binary_column}, id_map_json, embedding_dimension, created_at) "
        f"VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) "
        f"ON CONFLICT (index_name) DO UPDATE SET "
        f"{binary_column} = EXCLUDED.{binary_column}, "
        f"id_map_json = EXCLUDED.id_map_json, "
        f"embedding_dimension = EXCLUDED.embedding_dimension, "
        f"created_at = EXCLUDED.created_at"
    )

    id_map_fits = len(id_map_json.encode("utf-8")) <= max_part_size

    with db_conn.cursor() as cur:
        cur.execute(delete_sql, (index_name, like_pattern))

        if len(index_bytes) <= max_part_size and id_map_fits:
            cur.execute(
                upsert_sql,
                (index_name, psycopg2.Binary(index_bytes), id_map_json, embedding_dimension),
            )
            logger.info("Stored '%s' as a single row in %s.", index_name, target_table)
        else:
            bin_parts = _split_bytes(index_bytes, max_part_size)
            id_map_parts = _split_text(id_map_json, max_part_size)
            num_parts = max(len(bin_parts), len(id_map_parts))
            for idx in range(1, num_parts + 1):
                part_name = f"{index_name}_{idx}_{num_parts}"
                bin_frag = bin_parts[idx - 1] if idx - 1 < len(bin_parts) else b""
                id_map_frag = id_map_parts[idx - 1] if idx - 1 < len(id_map_parts) else ""
                cur.execute(
                    insert_sql,
                    (part_name, psycopg2.Binary(bin_frag), id_map_frag, embedding_dimension),
                )
            logger.info(
                "Stored '%s' in %d segmented rows in %s (binary=%d parts, id_map=%d parts).",
                index_name,
                num_parts,
                target_table,
                len(bin_parts),
                len(id_map_parts),
            )


def rewrite_segmented_id_map(
    cur,
    target_table: str,
    index_name: str,
    rewrite_fn,
    max_part_size_mb: Optional[int] = None,
) -> bool:
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(index_name, "index_name")
    mb = config.IVF_MAX_PART_SIZE_MB if max_part_size_mb is None else int(max_part_size_mb)
    max_part_size = mb * 1024 * 1024

    cur.execute(
        f"SELECT id_map_json FROM {target_table} WHERE index_name = %s",
        (index_name,),
    )
    single_row = cur.fetchone()
    if single_row is not None:
        old_json = single_row[0]
        new_json = rewrite_fn(old_json)
        if new_json == old_json:
            return False
        cur.execute(
            f"UPDATE {target_table} SET id_map_json = %s WHERE index_name = %s",
            (new_json, index_name),
        )
        return True

    like_pattern = index_name.replace("_", r"\_") + r"\_%\_%"
    cur.execute(
        f"SELECT index_name, id_map_json FROM {target_table} WHERE index_name LIKE %s ESCAPE '\\'",
        (like_pattern,),
    )
    seg_pattern = re.compile(rf"^{re.escape(index_name)}_(\d+)_(\d+)$")
    parts = []
    for name, frag in cur.fetchall() or []:
        m = seg_pattern.match(name)
        if m:
            parts.append((int(m.group(1)), name, frag))
    if not parts:
        return False
    parts.sort(key=lambda p: p[0])
    num_parts = len(parts)

    old_full = reassemble_segmented_id_map((p[0], p[2]) for p in parts)
    new_full = rewrite_fn(old_full)
    if new_full == old_full:
        return False

    new_frags = _split_text(new_full, max_part_size)
    if len(new_frags) > num_parts:
        raise ValueError(
            f"rewritten id_map for '{index_name}' needs {len(new_frags)} part rows "
            f"but the index has {num_parts}; rebuild the index instead of rewriting in place."
        )
    for position, (_, name, _) in enumerate(parts):
        frag = new_frags[position] if position < len(new_frags) else ""
        cur.execute(
            f"UPDATE {target_table} SET id_map_json = %s WHERE index_name = %s",
            (frag, name),
        )
    return True


def build_id_map(item_ids: Iterable[str]) -> dict:
    return {i: item_id for i, item_id in enumerate(item_ids)}


def store_segmented_blob(
    db_conn,
    target_table: str,
    name: str,
    blob: bytes,
    max_part_size_mb: Optional[int] = None,
) -> None:
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(name, "name")
    if not blob:
        raise ValueError("blob is empty; refusing to persist an empty payload")

    mb = config.IVF_MAX_PART_SIZE_MB if max_part_size_mb is None else int(max_part_size_mb)
    max_part_size = mb * 1024 * 1024

    delete_sql = f"DELETE FROM {target_table} WHERE name = %s OR name LIKE %s ESCAPE '\\'"
    like_pattern = name.replace("_", r"\_") + r"\_%\_%"

    upsert_sql = (
        f"INSERT INTO {target_table} (name, blob_data, created_at) "
        f"VALUES (%s, %s, CURRENT_TIMESTAMP) "
        f"ON CONFLICT (name) DO UPDATE SET "
        f"blob_data = EXCLUDED.blob_data, "
        f"created_at = EXCLUDED.created_at"
    )
    insert_sql = (
        f"INSERT INTO {target_table} (name, blob_data, created_at) "
        f"VALUES (%s, %s, CURRENT_TIMESTAMP) "
        f"ON CONFLICT (name) DO UPDATE SET "
        f"blob_data = EXCLUDED.blob_data, "
        f"created_at = EXCLUDED.created_at"
    )

    with db_conn.cursor() as cur:
        cur.execute(delete_sql, (name, like_pattern))

        if len(blob) <= max_part_size:
            cur.execute(upsert_sql, (name, psycopg2.Binary(blob)))
            logger.info("Stored '%s' as a single row in %s.", name, target_table)
        else:
            parts = _split_bytes(blob, max_part_size)
            num_parts = len(parts)
            for idx, part in enumerate(parts, start=1):
                part_name = f"{name}_{idx}_{num_parts}"
                cur.execute(insert_sql, (part_name, psycopg2.Binary(part)))
            logger.info(
                "Stored '%s' in %d segmented rows in %s.",
                name,
                num_parts,
                target_table,
            )


def load_segmented_blob(
    db_conn,
    target_table: str,
    name: str,
) -> Optional[bytes]:
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(name, "name")

    select_single_sql = f"SELECT blob_data FROM {target_table} WHERE name = %s"
    select_segments_sql = (
        f"SELECT name, blob_data FROM {target_table} WHERE name LIKE %s ESCAPE '\\'"
    )
    like_pattern = name.replace("_", r"\_") + r"\_%\_%"
    seg_pattern = re.compile(rf"^{re.escape(name)}_(\d+)_(\d+)$")

    with db_conn.cursor() as cur:
        cur.execute(select_single_sql, (name,))
        row = cur.fetchone()
        if row and row[0]:
            data = row[0]
            return bytes(data)

        cur.execute(select_segments_sql, (like_pattern,))
        rows = cur.fetchall()

    if not rows:
        return None

    parts: List[Tuple[int, bytes]] = []
    total_expected: Optional[int] = None
    for row_name, row_blob in rows:
        m = seg_pattern.match(row_name)
        if not m:
            continue
        part_no = int(m.group(1))
        total = int(m.group(2))
        if total_expected is None:
            total_expected = total
        elif total_expected != total:
            raise ValueError(
                f"Segment total mismatch for '{name}' in {target_table}: "
                f"saw {total_expected} and {total}."
            )
        parts.append((part_no, bytes(row_blob) if row_blob else b""))

    if total_expected is None or len(parts) != total_expected:
        raise ValueError(
            f"Incomplete segmented blob for '{name}' in {target_table}: "
            f"expected {total_expected}, found {len(parts)}."
        )

    parts.sort(key=lambda p: p[0])
    return b"".join(part_data for _, part_data in parts)


_ARTIST_META_MAGIC = b"ARMD"
_ARTIST_META_VERSION = 1
_ARTIST_META_HEADER_FMT = "<4sIIIII"
_ARTIST_META_HEADER_SIZE = struct.calcsize(_ARTIST_META_HEADER_FMT)


def pack_artist_metadata(
    artist_map: Dict[int, str],
    artist_gmms: Dict[str, Dict],
) -> bytes:
    buf = io.BytesIO()
    buf.write(b"\x00" * _ARTIST_META_HEADER_SIZE)

    artist_map_offset = buf.tell()
    buf.write(struct.pack("<I", len(artist_map)))
    for vec_id, artist_name in artist_map.items():
        name_bytes = artist_name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError(
                f"artist_name too long ({len(name_bytes)} bytes) for uint16 length prefix"
            )
        buf.write(struct.pack("<IH", int(vec_id), len(name_bytes)))
        buf.write(name_bytes)

    gmm_params_offset = buf.tell()
    buf.write(struct.pack("<I", len(artist_gmms)))
    for artist_name, gmm in artist_gmms.items():
        name_bytes = artist_name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError(
                f"artist_name too long ({len(name_bytes)} bytes) for uint16 length prefix"
            )

        tracks_hash = gmm.get("tracks_hash", "")
        tracks_hash_bytes = tracks_hash.encode("ascii") if tracks_hash else b""
        if len(tracks_hash_bytes) > 0xFF:
            raise ValueError(
                f"tracks_hash too long ({len(tracks_hash_bytes)} bytes) for uint8 length prefix"
            )

        n_components = int(gmm["n_components"])
        n_features = int(gmm["n_features"])
        n_tracks = int(gmm.get("n_tracks", 0))
        is_few_songs = 1 if gmm.get("is_few_songs", False) else 0

        means = np.ascontiguousarray(np.asarray(gmm["means"], dtype=np.float32))
        weights = np.ascontiguousarray(np.asarray(gmm["weights"], dtype=np.float32))
        if means.shape != (n_components, n_features):
            raise ValueError(
                f"means shape {means.shape} != ({n_components}, {n_features}) "
                f"for artist '{artist_name}'"
            )
        if weights.shape != (n_components,):
            raise ValueError(
                f"weights shape {weights.shape} != ({n_components},) for artist '{artist_name}'"
            )

        buf.write(struct.pack("<H", len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack("<B", len(tracks_hash_bytes)))
        buf.write(tracks_hash_bytes)
        buf.write(struct.pack("<BHHI", is_few_songs, n_components, n_features, n_tracks))
        buf.write(means.tobytes())
        buf.write(weights.tobytes())

    payload = buf.getvalue()
    header = struct.pack(
        _ARTIST_META_HEADER_FMT,
        _ARTIST_META_MAGIC,
        _ARTIST_META_VERSION,
        len(artist_gmms),
        artist_map_offset,
        gmm_params_offset,
        0,
    )
    return header + payload[_ARTIST_META_HEADER_SIZE:]


def unpack_artist_metadata(blob: bytes) -> Tuple[Dict[int, str], Dict[str, Dict]]:
    if len(blob) < _ARTIST_META_HEADER_SIZE:
        raise ValueError(f"artist metadata blob too short ({len(blob)} bytes)")

    magic, version, artist_count, artist_map_offset, gmm_params_offset, _reserved = struct.unpack(
        _ARTIST_META_HEADER_FMT, blob[:_ARTIST_META_HEADER_SIZE]
    )
    if magic != _ARTIST_META_MAGIC:
        raise ValueError(f"artist metadata magic mismatch: {magic!r}")
    if version != _ARTIST_META_VERSION:
        raise ValueError(f"unsupported artist metadata version: {version}")
    if artist_map_offset < _ARTIST_META_HEADER_SIZE or gmm_params_offset < artist_map_offset:
        raise ValueError("artist metadata section offsets are inconsistent")

    artist_map: Dict[int, str] = {}
    pos = artist_map_offset
    (map_count,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    for _ in range(map_count):
        vec_id, name_len = struct.unpack_from("<IH", blob, pos)
        pos += 6
        name = blob[pos : pos + name_len].decode("utf-8")
        pos += name_len
        artist_map[int(vec_id)] = name

    artist_gmms: Dict[str, Dict] = {}
    pos = gmm_params_offset
    (gmm_count,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    if gmm_count != artist_count:
        raise ValueError(f"header artist_count={artist_count} != gmm section count={gmm_count}")
    for _ in range(gmm_count):
        (name_len,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        name = blob[pos : pos + name_len].decode("utf-8")
        pos += name_len
        (tracks_hash_len,) = struct.unpack_from("<B", blob, pos)
        pos += 1
        tracks_hash = blob[pos : pos + tracks_hash_len].decode("ascii") if tracks_hash_len else ""
        pos += tracks_hash_len
        is_few_songs, n_components, n_features, n_tracks = struct.unpack_from("<BHHI", blob, pos)
        pos += 1 + 2 + 2 + 4

        means_size = n_components * n_features * 4
        weights_size = n_components * 4
        means = np.frombuffer(blob, dtype=np.float32, count=n_components * n_features, offset=pos)
        means = means.reshape(n_components, n_features)
        pos += means_size
        weights = np.frombuffer(blob, dtype=np.float32, count=n_components, offset=pos)
        pos += weights_size

        artist_gmms[name] = {
            "means": means.tolist(),
            "weights": weights.tolist(),
            "n_components": int(n_components),
            "n_features": int(n_features),
            "n_tracks": int(n_tracks),
            "is_few_songs": bool(is_few_songs),
            "tracks_hash": tracks_hash,
        }

    return artist_map, artist_gmms
