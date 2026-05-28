"""
Centralized helpers for building and persisting Voyager indexes.

Every index builder at the 96% stage (CLAP, lyrics, lyrics-axes, SemGrove,
audio Voyager) follows the same three-phase pattern:

    1. Load all embeddings from a BYTEA column into a contiguous numpy buffer.
    2. Add the buffer to a freshly built voyager.Index and serialize it.
    3. Persist the serialized bytes to an ``*_index_data`` table, splitting
       into multiple rows if larger than ``VOYAGER_MAX_PART_SIZE_MB``.

This module factors all three phases into reusable functions so any future
RAM, snapshot, or storage change happens in exactly one place.

Phase 1 design notes
--------------------
The previous streaming implementation opened a Postgres server-side cursor
on the worker's shared connection, inside the outer build transaction. That
left the worker connection idle-in-transaction for the entire build (often
many minutes), which broke whenever another worker on another container
modified the same embedding tables, or when Postgres' idle-in-tx timeout
fired.

``stream_embeddings_to_buffer`` opens its own short-lived connection in
autocommit + read-only mode, runs the SELECT through a named cursor, fills
a pre-allocated float32 buffer, and closes the connection before the
caller writes anything. Concurrent writes from other workers cannot affect
the iteration: the SELECT runs against a stable PG snapshot, and the
side connection holds no locks the rest of the build needs. The worst
realistic outcome is an index that omits a handful of rows committed
during the SELECT window -- acceptable because the next batch rebuild
picks them up.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import tempfile
from typing import Iterable, List, Optional, Tuple

import numpy as np
import psycopg2

import config

logger = logging.getLogger(__name__)


_STREAM_ITERSIZE = 5000

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sql_identifier(ident: str, kind: str) -> None:
    """Reject anything that isn't a bare SQL identifier.

    psycopg2 cannot parameterize table/column names, so they must be
    interpolated -- which means callers must not be able to pass arbitrary
    strings here. Builders always pass module-level literals; this guard is
    defense in depth.
    """
    if not isinstance(ident, str) or not _IDENT_RE.match(ident):
        raise ValueError(f"Invalid SQL {kind}: {ident!r}")


def _open_side_connection() -> "psycopg2.extensions.connection":
    """Open a fresh autocommit, read-only Postgres connection for streaming.

    Statement timeout is disabled here because builds over large libraries
    can legitimately exceed the default 10-minute global limit. Keepalives
    match ``app_helper.get_db`` so dead TCP sockets are detected.
    """
    conn = psycopg2.connect(
        config.DATABASE_URL,
        connect_timeout=30,
        keepalives_idle=600,
        keepalives_interval=30,
        keepalives_count=3,
        options="-c statement_timeout=0",
    )
    try:
        conn.set_session(autocommit=True, readonly=True)
    except Exception:
        try:
            conn.close()
        finally:
            raise
    return conn


def stream_embeddings_to_buffer(
    table: str,
    column: str,
    dim: int,
    where_clause: Optional[str] = None,
    cursor_name: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Stream a fixed-width float32 embedding column into a numpy buffer.

    Args:
        table: source table (must be a bare SQL identifier).
        column: BYTEA column holding float32 little-endian vectors
            (must be a bare SQL identifier).
        dim: expected number of float32 elements per row.
        where_clause: optional raw SQL fragment appended after ``WHERE``.
            Must not contain user input -- only module-level literals are
            allowed (e.g. ``"embedding IS NOT NULL"``).
        cursor_name: override for the server-side cursor name (default
            ``"_idx_stream_<table>_<column>"``).

    Returns:
        ``(buf, item_ids)`` where ``buf`` is a contiguous ``np.ndarray`` of
        shape ``(N, dim)`` and dtype ``float32``, and ``item_ids`` is a list
        of N item_id strings in the same row order. Rows with NULL or
        wrong-dimension blobs are skipped (counted, logged at warning).
        Returns an empty buffer and list if the source table has no rows.

    Raises:
        ValueError: if ``table`` or ``column`` is not a bare SQL identifier.
        psycopg2.Error: on connection or query failure.
    """
    _validate_sql_identifier(table, "table")
    _validate_sql_identifier(column, "column")
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int, got {dim!r}")

    where_sql = f" WHERE {where_clause}" if where_clause else ""
    count_sql = f"SELECT COUNT(*) FROM {table}{where_sql}"
    select_sql = f"SELECT item_id, {column} FROM {table}{where_sql}"
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
                vec = np.frombuffer(bytes(blob), dtype=np.float32)
                if vec.shape[0] != dim:
                    skipped_dim += 1
                    continue
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
                table, column, write_idx, skipped_null, skipped_dim,
            )
        else:
            logger.info(
                "stream_embeddings_to_buffer(%s.%s): loaded %d rows (dim=%d).",
                table, column, write_idx, dim,
            )

        return buf, item_ids
    finally:
        try:
            side_conn.close()
        except Exception:
            pass


def _resolve_voyager_space(metric: str):
    """Translate the config metric string into a voyager.Space enum value."""
    import voyager

    metric_l = (metric or "angular").lower()
    if metric_l == "angular":
        return voyager.Space.Cosine
    if metric_l == "euclidean":
        return voyager.Space.Euclidean
    if metric_l == "dot":
        return voyager.Space.InnerProduct
    logger.warning("Unknown Voyager metric '%s'; defaulting to Cosine.", metric)
    return voyager.Space.Cosine


def build_voyager_index_bytes(
    buf: np.ndarray,
    dim: int,
    metric: str = "angular",
    m: Optional[int] = None,
    ef_construction: Optional[int] = None,
) -> bytes:
    """Build a Voyager HNSW index over ``buf`` and serialize it to bytes.

    Item ids are assigned densely as ``0..N-1``; callers persist their own
    ``{voyager_id: item_id}`` map alongside the index bytes (see
    ``store_voyager_index_segmented``).

    Args:
        buf: contiguous ``np.ndarray`` of shape ``(N, dim)`` and dtype
            ``float32``. N must be >= 1.
        dim: number of dimensions, must equal ``buf.shape[1]``.
        metric: ``"angular"`` (cosine), ``"euclidean"``, or ``"dot"``.
        m, ef_construction: HNSW graph parameters. Default to the values in
            ``config.VOYAGER_M`` / ``config.VOYAGER_EF_CONSTRUCTION``.

    Returns:
        Serialized index bytes.

    Raises:
        ImportError: if the ``voyager`` library is not installed.
        ValueError: if ``buf`` is empty or has the wrong shape/dtype.
    """
    import voyager

    if not isinstance(buf, np.ndarray) or buf.ndim != 2:
        raise ValueError("buf must be a 2-D numpy array")
    if buf.shape[0] == 0:
        raise ValueError("buf is empty; refusing to build an empty index")
    if buf.shape[1] != dim:
        raise ValueError(
            f"buf has dim={buf.shape[1]} but caller declared dim={dim}"
        )
    if buf.dtype != np.float32:
        buf = buf.astype(np.float32, copy=False)
    if not buf.flags["C_CONTIGUOUS"]:
        buf = np.ascontiguousarray(buf)

    m_val = config.VOYAGER_M if m is None else int(m)
    ef_val = config.VOYAGER_EF_CONSTRUCTION if ef_construction is None else int(ef_construction)

    space = _resolve_voyager_space(metric)
    builder = voyager.Index(
        space=space,
        num_dimensions=dim,
        M=m_val,
        ef_construction=ef_val,
    )

    n = buf.shape[0]
    ids = np.arange(n, dtype=np.int64)
    builder.add_items(buf, ids=ids)

    temp_file_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".voyager") as tmp:
            temp_file_path = tmp.name
        builder.save(temp_file_path)
        del builder
        gc.collect()
        with open(temp_file_path, "rb") as f:
            return f.read()
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


def _split_bytes(data: bytes, part_size: int) -> List[bytes]:
    return [data[i:i + part_size] for i in range(0, len(data), part_size)]


def store_voyager_index_segmented(
    db_conn,
    target_table: str,
    index_name: str,
    index_bytes: bytes,
    id_map: dict,
    embedding_dimension: int,
    max_part_size_mb: Optional[int] = None,
) -> None:
    """Persist a serialized Voyager index to a chunked ``*_index_data`` table.

    Atomically replaces any existing rows for ``index_name`` (single or
    segmented). If the binary is small enough it is written as a single
    row; otherwise it is split into rows named
    ``<index_name>_<part>_<total>``. The id_map JSON is stored in full on
    the first part to keep subsequent parts small.

    The caller's ``db_conn`` is used as-is and is **not** committed by this
    function -- the caller controls the transaction boundary (matching the
    existing builders, which commit at the very end).

    Args:
        db_conn: psycopg2 connection (the build's main connection).
        target_table: name of the ``*_index_data`` table to write to
            (must be a bare SQL identifier).
        index_name: logical index name (e.g. ``"clap_index"``). Must be a
            bare SQL identifier so the LIKE-escape pattern is unambiguous.
        index_bytes: serialized index payload (from
            ``build_voyager_index_bytes``).
        id_map: ``{voyager_id_int: item_id_str}`` mapping. Serialized to
            JSON for the first row.
        embedding_dimension: stored alongside the index for validation on
            load.
        max_part_size_mb: override for ``config.VOYAGER_MAX_PART_SIZE_MB``.
    """
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(index_name, "index_name")
    if not index_bytes:
        raise ValueError("index_bytes is empty; refusing to persist an empty index")

    mb = config.VOYAGER_MAX_PART_SIZE_MB if max_part_size_mb is None else int(max_part_size_mb)
    max_part_size = mb * 1024 * 1024
    id_map_json = json.dumps(id_map)

    delete_sql = (
        f"DELETE FROM {target_table} "
        f"WHERE index_name = %s OR index_name LIKE %s ESCAPE '\\'"
    )
    like_pattern = index_name.replace("_", r"\_") + r"\_%\_%"

    upsert_sql = (
        f"INSERT INTO {target_table} "
        f"(index_name, index_data, id_map_json, embedding_dimension, created_at) "
        f"VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) "
        f"ON CONFLICT (index_name) DO UPDATE SET "
        f"index_data = EXCLUDED.index_data, "
        f"id_map_json = EXCLUDED.id_map_json, "
        f"embedding_dimension = EXCLUDED.embedding_dimension, "
        f"created_at = EXCLUDED.created_at"
    )
    insert_sql = (
        f"INSERT INTO {target_table} "
        f"(index_name, index_data, id_map_json, embedding_dimension, created_at) "
        f"VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)"
    )

    with db_conn.cursor() as cur:
        cur.execute(delete_sql, (index_name, like_pattern))

        if len(index_bytes) <= max_part_size:
            cur.execute(
                upsert_sql,
                (index_name, psycopg2.Binary(index_bytes), id_map_json, embedding_dimension),
            )
            logger.info("Stored '%s' as a single row in %s.", index_name, target_table)
        else:
            parts = _split_bytes(index_bytes, max_part_size)
            num_parts = len(parts)
            for idx, part in enumerate(parts, start=1):
                part_name = f"{index_name}_{idx}_{num_parts}"
                part_id_map = id_map_json if idx == 1 else ""
                cur.execute(
                    insert_sql,
                    (part_name, psycopg2.Binary(part), part_id_map, embedding_dimension),
                )
            logger.info(
                "Stored '%s' in %d segmented rows in %s.",
                index_name, num_parts, target_table,
            )


def build_id_map(item_ids: Iterable[str]) -> dict:
    """Return ``{int_voyager_id: item_id_str}`` matching the row order."""
    return {i: item_id for i, item_id in enumerate(item_ids)}
