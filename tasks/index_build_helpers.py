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

``stream_embeddings_to_buffer`` opens its own dedicated short-lived
read-only connection, runs the SELECT through a server-side named cursor
inside that connection's own implicit transaction, fills a pre-allocated
float32 buffer, and closes the connection (auto-rolling-back the read-only
transaction) before the caller writes anything. Two independent guarantees:

* **Snapshot consistency.** The named cursor sees a stable PG snapshot for
  its entire lifetime, so concurrent writes from other workers cannot make
  fetches inconsistent. We deliberately do NOT use ``autocommit=True`` --
  psycopg2 still permits named cursors in autocommit mode but PG does not
  hold a snapshot across fetches, which would silently return mixed data.
* **No fate-sharing with the build's main transaction.** The streaming
  transaction lives only on the side connection and only for the duration
  of the SELECT. The worker's main connection (where the index is
  ultimately written) is never put in idle-in-transaction state by this
  helper, which is what caused the previous streaming revert.

The worst realistic outcome is an index that omits a handful of rows
committed AFTER the side connection's snapshot was taken -- acceptable
because the next batch rebuild picks them up.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import re
import struct
import tempfile
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import psycopg2

import config

logger = logging.getLogger(__name__)


class EmptyIndexError(ValueError):
    """Raised when an index builder is asked to serialize zero items.

    Distinguishing this from a generic ``ValueError`` lets callers downgrade
    "empty source" to a warning while still surfacing real programming errors
    (wrong dim, batch shape mismatch, mismatched ids length) as exceptions.
    Subclassing ``ValueError`` preserves backward compatibility with any code
    that already catches ``ValueError`` here.
    """


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
    """Open a fresh read-only Postgres connection for streaming reads.

    The connection runs in the default transactional mode (NOT autocommit):
    psycopg2 with ``autocommit=True`` still permits server-side named
    cursors, but PG does not hold a stable snapshot across fetches in that
    mode -- concurrent writes from other workers could make iteration
    return inconsistent data. With autocommit off, psycopg2 issues an
    implicit ``BEGIN`` before the first statement and the named cursor
    inherits that transaction's snapshot for its entire lifetime.

    The transaction is closed cheaply: ``conn.close()`` issues an implicit
    ``ROLLBACK`` (read-only, nothing to commit), and because this is a
    dedicated short-lived connection, the open transaction never overlaps
    with the rest of the build the way the previous streaming revert did.

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


def iter_embedding_batches(
    table: str,
    column: str,
    dim: int,
    batch_size: int = 5000,
    where_clause: Optional[str] = None,
    cursor_name: Optional[str] = None,
) -> Iterator[Tuple[np.ndarray, List[str]]]:
    """Yield ``(batch_buf, batch_ids)`` pairs from a BYTEA float32 column.

    Same snapshot-safe pattern as :func:`stream_embeddings_to_buffer`
    (dedicated short-lived read-only side connection, default transactional
    mode, server-side named cursor inside the connection's own implicit
    ``BEGIN``), but each batch is yielded and freed before the next is
    fetched. Peak RAM per batch is ``batch_size * dim * 4`` bytes plus the
    per-row item_id strings -- e.g. ~15 MB for a 5000-row batch at 768 dim.

    Each yielded ``batch_buf`` is a fresh, contiguous float32 ndarray of
    shape ``(actual_batch_n, dim)`` where ``actual_batch_n <= batch_size``
    (last batch may be partial; rows with NULL or wrong-dim blobs are
    skipped silently and counted in an aggregate warning at end).

    The side connection is closed in the generator's ``finally`` so it is
    released both on normal completion and on early ``GeneratorExit``
    (consumer breaks out of the loop or hits an exception).

    Args:
        table: source table (must be a bare SQL identifier).
        column: BYTEA column holding float32 little-endian vectors
            (must be a bare SQL identifier).
        dim: expected number of float32 elements per row.
        batch_size: maximum rows per yielded batch. Defaults to 5000 to
            match ``_STREAM_ITERSIZE``.
        where_clause: optional raw SQL fragment appended after ``WHERE``.
            Must not contain user input.
        cursor_name: override for the server-side cursor name.

    Yields:
        ``(batch_buf, batch_ids)`` tuples. Yields nothing if the source has
        no rows.

    Raises:
        ValueError: if ``table``/``column`` is not a bare SQL identifier or
            ``dim``/``batch_size`` is not a positive int.
        psycopg2.Error: on connection or query failure.
    """
    _validate_sql_identifier(table, "table")
    _validate_sql_identifier(column, "column")
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int, got {dim!r}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")

    where_sql = f" WHERE {where_clause}" if where_clause else ""
    select_sql = f"SELECT item_id, {column} FROM {table}{where_sql}"
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
                        table, column, batch_no, write_idx,
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
                table, column, batch_no, write_idx,
            )
            yield batch_buf[:write_idx].copy(), batch_ids

        if total_skipped_null or total_skipped_dim:
            logger.warning(
                "iter_embedding_batches(%s.%s): kept=%d skipped_null=%d skipped_dim=%d across %d batch(es).",
                table, column, total_kept, total_skipped_null, total_skipped_dim, batch_no,
            )
        else:
            logger.info(
                "iter_embedding_batches(%s.%s): streamed %d rows across %d batch(es), dim=%d.",
                table, column, total_kept, batch_no, dim,
            )
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


def build_voyager_index_bytes_streaming(
    batch_iter: Iterable[Tuple[np.ndarray, List[str]]],
    dim: int,
    metric: str = "angular",
    m: Optional[int] = None,
    ef_construction: Optional[int] = None,
) -> Tuple[bytes, List[str]]:
    """Build a Voyager HNSW index incrementally and serialize to bytes.

    Consumes batches from ``batch_iter`` (typically the generator returned by
    :func:`iter_embedding_batches`) and calls ``voyager.Index.add_items``
    once per batch with dense ``0..N-1`` ids assigned across batches. The
    Voyager index auto-grows on each ``add_items`` call, so no upfront row
    count is required.

    Peak RAM during build is ``1 batch + voyager's internal HNSW storage``
    rather than ``full library buffer + voyager's internal HNSW storage`` --
    i.e. the input-side memory is essentially zero compared to the
    unavoidable index storage.

    Args:
        batch_iter: iterable yielding ``(batch_buf, batch_ids)`` tuples.
            Each ``batch_buf`` must be a float32 ndarray of shape
            ``(batch_n, dim)`` (any dtype is silently coerced; non-2-D or
            wrong-dim batches raise ``ValueError`` immediately).
        dim: vector dimensionality. Must equal each batch's second axis.
        metric: ``"angular"`` (cosine), ``"euclidean"``, or ``"dot"``.
        m, ef_construction: HNSW graph parameters. Default to
            ``config.VOYAGER_M`` / ``config.VOYAGER_EF_CONSTRUCTION``.

    Returns:
        ``(index_bytes, item_ids)`` -- the serialized Voyager index and the
        flat list of item_id strings in row order. Items dropped by the
        generator (NULL/wrong-dim blobs) do not appear in either output.

    Raises:
        ImportError: if the ``voyager`` library is not installed.
        ValueError: if the iterator yields no batches at all, or a batch
            has the wrong shape/dim.
    """
    import voyager

    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int, got {dim!r}")

    m_val = config.VOYAGER_M if m is None else int(m)
    ef_val = config.VOYAGER_EF_CONSTRUCTION if ef_construction is None else int(ef_construction)
    space = _resolve_voyager_space(metric)

    builder = voyager.Index(
        space=space,
        num_dimensions=dim,
        M=m_val,
        ef_construction=ef_val,
    )

    all_ids: List[str] = []
    next_voyager_id = 0
    saw_any = False

    for batch_buf, batch_ids in batch_iter:
        if not isinstance(batch_buf, np.ndarray) or batch_buf.ndim != 2:
            raise ValueError(
                "build_voyager_index_bytes_streaming: each batch_buf must be a 2-D ndarray"
            )
        if batch_buf.shape[1] != dim:
            raise ValueError(
                f"build_voyager_index_bytes_streaming: batch dim={batch_buf.shape[1]} != declared dim={dim}"
            )
        if batch_buf.shape[0] == 0:
            continue
        if batch_buf.dtype != np.float32:
            batch_buf = batch_buf.astype(np.float32, copy=False)
        if not batch_buf.flags["C_CONTIGUOUS"]:
            batch_buf = np.ascontiguousarray(batch_buf)
        if len(batch_ids) != batch_buf.shape[0]:
            raise ValueError(
                f"build_voyager_index_bytes_streaming: batch_ids len={len(batch_ids)} "
                f"!= batch_buf rows={batch_buf.shape[0]}"
            )

        ids = np.arange(next_voyager_id, next_voyager_id + batch_buf.shape[0], dtype=np.int64)
        builder.add_items(batch_buf, ids=ids)
        all_ids.extend(batch_ids)
        next_voyager_id += batch_buf.shape[0]
        saw_any = True

    if not saw_any or next_voyager_id == 0:
        del builder
        gc.collect()
        raise EmptyIndexError("build_voyager_index_bytes_streaming: no items added; refusing to serialize empty index")

    temp_file_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".voyager") as tmp:
            temp_file_path = tmp.name
        builder.save(temp_file_path)
        del builder
        gc.collect()
        with open(temp_file_path, "rb") as f:
            return f.read(), all_ids
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


def _split_bytes(data: bytes, part_size: int) -> List[bytes]:
    return [data[i:i + part_size] for i in range(0, len(data), part_size)]


def _split_text(text: str, max_part_bytes: int) -> List[str]:
    if not text:
        return [""]
    if len(text.encode("utf-8")) <= max_part_bytes:
        return [text]
    step = max(1, max_part_bytes // 4)
    return [text[i:i + step] for i in range(0, len(text), step)]


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
    """Stream embeddings, build a Voyager index, and persist it.

    Wraps the canonical build pipeline shared by the CLAP, lyrics, and
    lyrics-axes builders: ``iter_embedding_batches`` ->
    ``build_voyager_index_bytes_streaming`` -> ``store_voyager_index_segmented``.
    Commits on success and rolls back on failure using the caller's
    ``db_conn``. Returns True on success, False when the source table has no
    usable vectors or the build/store fails.
    """
    label = label or index_name
    try:
        import voyager  # type: ignore  # noqa: F401
    except ImportError:
        logger.warning("Voyager library is unavailable; cannot build %s index.", label)
        return False

    try:
        logger.info("Building %s voyager index (streaming)...", label)
        batches = iter_embedding_batches(
            table=source_table,
            column=source_column,
            dim=dim,
            where_clause=where_clause,
        )
        try:
            index_bytes, item_ids = build_voyager_index_bytes_streaming(
                batches, dim, metric=metric,
            )
        except EmptyIndexError as ve:
            logger.warning("No valid %s vectors found for index build: %s", label, ve)
            return False
        gc.collect()

        if not index_bytes:
            logger.error("Generated %s index binary is empty; aborting storage.", label)
            return False

        id_map = build_id_map(item_ids)
        store_voyager_index_segmented(
            db_conn,
            target_table=target_table,
            index_name=index_name,
            index_bytes=index_bytes,
            id_map=id_map,
            embedding_dimension=dim,
        )

        db_conn.commit()
        logger.info("%s index build successful.", label)
        return True
    except Exception as e:
        logger.error("Failed to build/store %s index: %s", label, e, exc_info=True)
        try:
            db_conn.rollback()
        except Exception:
            pass
        return False


def load_voyager_index_from_db(
    conn,
    table: str,
    index_name: str,
    expected_dim: int,
    query_ef: int,
    label: Optional[str] = None,
):
    """Load a Voyager index persisted by ``store_voyager_index_segmented``.

    Tries the classic single-row layout first, then the segmented
    ``<index_name>_<part>_<total>`` layout. Returns ``(index, id_map,
    reverse_id_map)`` on success or ``None`` when the index is missing,
    incomplete, or fails validation. DB and deserialization exceptions
    propagate to the caller, which is expected to log and treat the load
    as failed (the historical per-manager behavior).
    """
    label = label or index_name
    _validate_sql_identifier(table, "table")
    _validate_sql_identifier(index_name, "index_name")

    try:
        import voyager  # type: ignore
    except ImportError:
        logger.warning("Voyager library is unavailable; cannot load %s index.", label)
        return None

    with conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout = 0")
        cur.execute(
            f"SELECT index_data, id_map_json, embedding_dimension FROM {table} "
            f"WHERE index_name = %s",
            (index_name,),
        )
        row = cur.fetchone()

        index_stream = None
        try:
            if row:
                binary, id_map_json, db_dim = row
                index_stream = tempfile.TemporaryFile()
                index_stream.write(binary)
                index_stream.seek(0)
            else:
                seg_pattern = re.compile(rf'^{re.escape(index_name)}_(\d+)_(\d+)$')
                parts = []
                total_expected = None
                with conn.cursor(name=f'{index_name}_segments') as seg_cur:
                    seg_cur.itersize = 50
                    seg_cur.execute(
                        f"SELECT index_name, index_data, id_map_json, embedding_dimension "
                        f"FROM {table} WHERE index_name LIKE %s ESCAPE '\\'",
                        (index_name.replace('_', r'\_') + r'\_%\_%',),
                    )
                    for name, part_data, part_id_map, part_dim in seg_cur:
                        m = seg_pattern.match(name)
                        if not m:
                            continue
                        part_no = int(m.group(1))
                        total = int(m.group(2))
                        if total_expected is None:
                            total_expected = total
                        elif total_expected != total:
                            logger.error(
                                "%s index segment total mismatch: %s vs %s",
                                label, total_expected, total,
                            )
                            return None
                        parts.append((part_no, part_data, part_id_map, part_dim))

                if total_expected is None or len(parts) != total_expected:
                    logger.info(
                        "No complete persisted %s index found (expected %s, have %d).",
                        label, total_expected, len(parts),
                    )
                    return None

                parts.sort(key=lambda p: p[0])
                db_dim = parts[0][3]
                index_stream = tempfile.TemporaryFile()
                for _, part_data, _, _ in parts:
                    index_stream.write(part_data)
                index_stream.seek(0)
                id_map_json = reassemble_segmented_id_map((p[0], p[2]) for p in parts)

            if index_stream is None or not id_map_json:
                logger.info("%s index not found or empty in the database.", label)
                return None
            if db_dim != expected_dim:
                logger.error(
                    "%s index dimension mismatch: db=%s expected=%s",
                    label, db_dim, expected_dim,
                )
                return None

            loaded_index = voyager.Index.load(index_stream)
            loaded_index.ef = query_ef
        finally:
            if index_stream is not None:
                try:
                    index_stream.close()
                except Exception:
                    pass

    id_map = {int(k): v for k, v in json.loads(id_map_json).items()}
    if not id_map:
        logger.warning("%s index id_map is empty.", label)
        return None
    reverse_id_map = {v: k for k, v in id_map.items()}
    return loaded_index, id_map, reverse_id_map


def store_voyager_index_segmented(
    db_conn,
    target_table: str,
    index_name: str,
    index_bytes: bytes,
    id_map: dict,
    embedding_dimension: int,
    max_part_size_mb: Optional[int] = None,
    binary_column: str = "index_data",
) -> None:
    """Persist a serialized Voyager index to a chunked ``*_index_data`` table.

    Atomically replaces any existing rows for ``index_name`` (single or
    segmented). If the binary is small enough it is written as a single
    row; otherwise it is split into rows named
    ``<index_name>_<part>_<total>``. The id_map JSON is itself split across
    the same part rows (one fragment per row, reassembled in part order by
    :func:`reassemble_segmented_id_map`) so neither the binary nor the id_map
    can exceed PG's 1 GB field cap at any library size. For libraries whose
    id_map still fits in one part (the common case) the whole map lands on
    part 1 with the rest empty -- byte-identical to the previous layout, so
    older readers stay compatible.

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
            ``build_voyager_index_bytes_streaming``).
        id_map: ``{voyager_id_int: item_id_str}`` mapping. Serialized to
            JSON for the first row.
        embedding_dimension: stored alongside the index for validation on
            load.
        max_part_size_mb: override for ``config.VOYAGER_MAX_PART_SIZE_MB``.
    """
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(index_name, "index_name")
    _validate_sql_identifier(binary_column, "column")
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
        f"VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)"
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
                index_name, num_parts, target_table, len(bin_parts), len(id_map_parts),
            )


def rewrite_segmented_id_map(
    cur,
    target_table: str,
    index_name: str,
    rewrite_fn,
    max_part_size_mb: Optional[int] = None,
) -> bool:
    """Rewrite ``id_map_json`` for a possibly-segmented index in place.

    :func:`store_voyager_index_segmented` splits the id_map JSON across part
    rows, so each segmented row holds a partial-JSON *fragment* rather than a
    standalone document. A naive per-row ``json.loads`` rewrite therefore
    silently no-ops on every fragment. This helper instead reassembles the
    full id_map across all part rows (via :func:`reassemble_segmented_id_map`),
    applies ``rewrite_fn`` to the whole JSON string, then writes the result
    back -- re-split across the SAME part rows for a segmented index (the
    binary columns are left untouched) or as a single field for a single-row
    index.

    ``rewrite_fn`` receives the full id_map JSON string and returns the
    rewritten string; returning it unchanged means "nothing to do".

    Every statement runs on the caller's ``cur`` so the rewrite joins the
    caller's transaction. Returns True if any row was updated.

    Raises ``ValueError`` if the rewritten id_map needs more part rows than the
    index currently has -- only possible when the id_map dominates the part
    count and the new ids are substantially longer. The caller should rebuild
    the index from scratch in that case rather than risk a partial write.
    """
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(index_name, "index_name")
    mb = config.VOYAGER_MAX_PART_SIZE_MB if max_part_size_mb is None else int(max_part_size_mb)
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
        f"SELECT index_name, id_map_json FROM {target_table} "
        f"WHERE index_name LIKE %s ESCAPE '\\'",
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
    """Return ``{int_voyager_id: item_id_str}`` matching the row order."""
    return {i: item_id for i, item_id in enumerate(item_ids)}


def store_segmented_blob(
    db_conn,
    target_table: str,
    name: str,
    blob: bytes,
    max_part_size_mb: Optional[int] = None,
) -> None:
    """Persist a single BYTEA payload to a ``(name, blob_data, created_at)`` table.

    Mirrors :func:`store_voyager_index_segmented` but for the simpler 2-column
    schema used by ``artist_metadata_data``: ``name VARCHAR PRIMARY KEY``,
    ``blob_data BYTEA NOT NULL``, ``created_at TIMESTAMP``. Any previous rows
    matching ``name`` or ``name_<part>_<total>`` are deleted in the same
    transaction before the new payload is written, so readers never see
    partial state.

    Payloads larger than ``max_part_size_mb`` are split into rows named
    ``<name>_<part>_<total>``. This is what insulates the artist metadata
    blob from PG's 1 GB MaxAllocSize cap at any library size: a 2.4 GB blob
    becomes ~48 rows of 50 MB each, all well under the cap.

    The caller's ``db_conn`` is used as-is and is **not** committed by this
    function. The caller controls the transaction boundary.

    Args:
        db_conn: psycopg2 connection (caller's main connection).
        target_table: ``(name VARCHAR PK, blob_data BYTEA, created_at TIMESTAMP)``
            target table (must be a bare SQL identifier).
        name: logical blob name (must be a bare SQL identifier so the
            LIKE-escape pattern is unambiguous).
        blob: payload to persist. Empty bytes raises ``ValueError``.
        max_part_size_mb: override for ``config.VOYAGER_MAX_PART_SIZE_MB``
            (the existing global tunable for segmented-row sizing).
    """
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(name, "name")
    if not blob:
        raise ValueError("blob is empty; refusing to persist an empty payload")

    mb = config.VOYAGER_MAX_PART_SIZE_MB if max_part_size_mb is None else int(max_part_size_mb)
    max_part_size = mb * 1024 * 1024

    delete_sql = (
        f"DELETE FROM {target_table} "
        f"WHERE name = %s OR name LIKE %s ESCAPE '\\'"
    )
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
        f"VALUES (%s, %s, CURRENT_TIMESTAMP)"
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
                name, num_parts, target_table,
            )


def load_segmented_blob(
    db_conn,
    target_table: str,
    name: str,
) -> Optional[bytes]:
    """Reassemble a payload previously written by :func:`store_segmented_blob`.

    Tries the single-row form first (``name = <name>``), then the segmented
    form (``name LIKE <name>_<part>_<total>``). Returns ``None`` if neither
    yields a row -- the loader can use that to detect a legacy deployment
    that has not yet rebuilt onto the new table and fall back to whatever
    older storage existed.

    Validates that all expected segments are present before returning. A
    missing or duplicated segment raises ``ValueError``; this is treated as
    a corruption signal rather than silently returning a partial blob.

    Args:
        db_conn: psycopg2 connection.
        target_table: ``(name VARCHAR PK, blob_data BYTEA, created_at TIMESTAMP)``
            source table (must be a bare SQL identifier).
        name: logical blob name (must be a bare SQL identifier).
    """
    _validate_sql_identifier(target_table, "table")
    _validate_sql_identifier(name, "name")

    select_single_sql = f"SELECT blob_data FROM {target_table} WHERE name = %s"
    select_segments_sql = (
        f"SELECT name, blob_data FROM {target_table} "
        f"WHERE name LIKE %s ESCAPE '\\'"
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
    """Serialize the artist index's auxiliary metadata into a single bytes blob.

    Produces a self-describing little-endian binary container that replaces
    the previous JSON-of-floats storage. Format documented in the plan; in
    short:

    * 24-byte header (magic ``ARMD``, version=1, artist_count, two section
      offsets).
    * Artist-map section: per (voyager_id, artist_name) tuple.
    * GMM-params section: per artist, ``means`` and ``weights`` as raw
      float32 little-endian bytes. ``covariances`` is deliberately not
      stored -- nothing reads it.

    Args:
        artist_map: ``{voyager_id_int: artist_name_str}``.
        artist_gmms: ``{artist_name_str: {means, weights, n_components,
            n_features, n_tracks, is_few_songs, tracks_hash}}``. Extra keys
            are ignored; missing keys raise ``KeyError``.
    """
    buf = io.BytesIO()
    buf.write(b"\x00" * _ARTIST_META_HEADER_SIZE)

    artist_map_offset = buf.tell()
    buf.write(struct.pack("<I", len(artist_map)))
    for voyager_id, artist_name in artist_map.items():
        name_bytes = artist_name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError(f"artist_name too long ({len(name_bytes)} bytes) for uint16 length prefix")
        buf.write(struct.pack("<IH", int(voyager_id), len(name_bytes)))
        buf.write(name_bytes)

    gmm_params_offset = buf.tell()
    buf.write(struct.pack("<I", len(artist_gmms)))
    for artist_name, gmm in artist_gmms.items():
        name_bytes = artist_name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError(f"artist_name too long ({len(name_bytes)} bytes) for uint16 length prefix")

        tracks_hash = gmm.get("tracks_hash", "")
        tracks_hash_bytes = tracks_hash.encode("ascii") if tracks_hash else b""
        if len(tracks_hash_bytes) > 0xFF:
            raise ValueError(f"tracks_hash too long ({len(tracks_hash_bytes)} bytes) for uint8 length prefix")

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
                f"weights shape {weights.shape} != ({n_components},) "
                f"for artist '{artist_name}'"
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
    """Inverse of :func:`pack_artist_metadata`.

    Returns ``(artist_map, artist_gmms)`` reconstructed from the binary blob.
    ``artist_gmms`` entries have the same in-memory shape the rest of the
    code expects (``means`` and ``weights`` as Python lists, plus the
    metadata fields). ``covariances`` and ``covariance_type`` are NOT
    re-introduced -- nothing live reads them.

    Raises ``ValueError`` if the magic / version don't match or if the blob
    is truncated.
    """
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
        voyager_id, name_len = struct.unpack_from("<IH", blob, pos)
        pos += 6
        name = blob[pos:pos + name_len].decode("utf-8")
        pos += name_len
        artist_map[int(voyager_id)] = name

    artist_gmms: Dict[str, Dict] = {}
    pos = gmm_params_offset
    (gmm_count,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    if gmm_count != artist_count:
        raise ValueError(
            f"header artist_count={artist_count} != gmm section count={gmm_count}"
        )
    for _ in range(gmm_count):
        (name_len,) = struct.unpack_from("<H", blob, pos)
        pos += 2
        name = blob[pos:pos + name_len].decode("utf-8")
        pos += name_len
        (tracks_hash_len,) = struct.unpack_from("<B", blob, pos)
        pos += 1
        tracks_hash = blob[pos:pos + tracks_hash_len].decode("ascii") if tracks_hash_len else ""
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
