# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Blob-table space reclaim against a real Postgres database.

Autovacuum's threshold counts ROWS, so a bytea table with a handful of dead rows
sits below it and is never vacuumed; this sweep is what reaches it. Confirms
against real Postgres that the detection query sees such a table, that the
sweep VACUUMs it back to zero dead rows, and that the snapshot probe stands down
only for a session that actually holds a snapshot.

Main Features:
* A bytea table whose dead rows autovacuum would not collect is detected and swept
* After the sweep the table's dead-row count is back to zero
* The snapshot probe defers when another session holds a snapshot
* The snapshot probe ignores a read-committed session with no snapshot
* The snapshot probe passes when no other session holds one
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

pytestmark = pytest.mark.integration


def _force_stats_flush(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT current_setting('server_version_num')::int")
        if cur.fetchone()[0] >= 150000:
            cur.execute("SELECT pg_stat_force_next_flush()")


@pytest.fixture
def blob_reclaim_db(shared_pg_dsn, task_status_ddl):
    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS blob_reclaim_probe CASCADE")
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
        cur.execute(task_status_ddl)
        cur.execute(
            "CREATE TABLE blob_reclaim_probe (id SERIAL PRIMARY KEY, blob_data BYTEA)"
        )
        for _ in range(20):
            cur.execute(
                "INSERT INTO blob_reclaim_probe (blob_data) VALUES (%s)",
                (psycopg2.Binary(b"x" * 4096),),
            )
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS blob_reclaim_probe CASCADE")
        cur.execute("DROP TABLE IF EXISTS task_status CASCADE")
    conn.close()


class TestSweepsRealDeadRows:
    def test_a_bytea_table_autovacuum_cannot_reach_is_detected_and_swept(
        self, blob_reclaim_db
    ):
        from taskqueue import maintenance, sql

        conn = blob_reclaim_db
        with conn.cursor() as cur:
            cur.execute("DELETE FROM blob_reclaim_probe WHERE id <= 15")
        _force_stats_flush(conn)

        with conn.cursor() as cur:
            candidates = sql.blob_tables_autovacuum_cannot_reach(cur)
        names = [name for name, _dead, _size in candidates]
        assert "public.blob_reclaim_probe" in names

        reclaimed = maintenance.reclaim_blob_space(conn)
        assert "public.blob_reclaim_probe" in reclaimed

        with conn.cursor() as cur:
            cur.execute(
                "SELECT n_dead_tup FROM pg_stat_user_tables "
                "WHERE schemaname = 'public' AND relname = 'blob_reclaim_probe'"
            )
            assert cur.fetchone()[0] == 0


class TestSnapshotStandDown:
    def test_the_probe_defers_for_a_session_holding_a_snapshot(
        self, blob_reclaim_db, shared_pg_dsn
    ):
        from taskqueue import sql

        holder = psycopg2.connect(shared_pg_dsn)
        try:
            holder.set_session(isolation_level='REPEATABLE READ', autocommit=False)
            with holder.cursor() as cur:
                cur.execute("SELECT 1")
            probe = psycopg2.connect(shared_pg_dsn)
            try:
                with probe.cursor() as cur:
                    assert (
                        sql.snapshot_holder_blocking_reclaim(cur, grace_seconds=0) is True
                    )
            finally:
                probe.close()
        finally:
            holder.close()

    def test_the_probe_ignores_a_read_committed_session_with_no_snapshot(
        self, blob_reclaim_db, shared_pg_dsn
    ):
        from taskqueue import sql

        holder = psycopg2.connect(shared_pg_dsn)
        try:
            with holder.cursor() as cur:
                cur.execute("SELECT 1")
            probe = psycopg2.connect(shared_pg_dsn)
            try:
                with probe.cursor() as cur:
                    assert (
                        sql.snapshot_holder_blocking_reclaim(cur, grace_seconds=0) is False
                    )
            finally:
                probe.close()
        finally:
            holder.close()

    def test_the_probe_passes_when_no_session_holds_a_snapshot(self, blob_reclaim_db):
        from taskqueue import sql

        with blob_reclaim_db.cursor() as cur:
            assert sql.snapshot_holder_blocking_reclaim(cur, grace_seconds=0) is False
