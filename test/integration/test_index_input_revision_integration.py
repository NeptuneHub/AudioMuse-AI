"""Postgres coverage for durable index-input change tracking."""

import uuid

import psycopg2
import pytest
from psycopg2 import sql

import database


@pytest.fixture
def revision_db(shared_pg_dsn):
    schema = f"index_revision_{uuid.uuid4().hex}"
    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            for ddl in (
                "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
                "album TEXT, album_artist TEXT, tempo DOUBLE PRECISION, key TEXT, "
                "scale TEXT, mood_vector TEXT, energy DOUBLE PRECISION, "
                "other_features TEXT, year INTEGER, rating INTEGER, duration DOUBLE PRECISION, "
                "file_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
                "CREATE TABLE embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, "
                "poincare_embedding BYTEA, hyperbolic_radius DOUBLE PRECISION, "
                "neural_fingerprint BYTEA)",
                "CREATE TABLE clap_embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
                "CREATE TABLE lyrics_embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, "
                "axis_vector BYTEA, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
                "CREATE TABLE track_server_map (item_id TEXT, server_id TEXT, "
                "provider_track_id TEXT, match_tier TEXT, file_path TEXT, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
                "CREATE TABLE artist_server_map (artist_name TEXT, server_id TEXT, "
                "provider_artist_id TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
                "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
                "server_type TEXT, creds JSONB, music_libraries TEXT, is_default BOOLEAN, "
                "track_count INTEGER, created_at TIMESTAMP, updated_at TIMESTAMP)",
                "CREATE TABLE app_config (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            ):
                cur.execute(ddl)
            database._ensure_index_input_revision_tracking(cur)
    finally:
        conn.close()

    yield shared_pg_dsn, schema

    cleanup = psycopg2.connect(shared_pg_dsn)
    cleanup.autocommit = True
    try:
        with cleanup.cursor() as cur:
            cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    finally:
        cleanup.close()


def _connect(dsn, schema):
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
    conn.commit()
    return conn


def _revision_snapshot(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT array_agg(revision ORDER BY shard) FROM index_input_revision")
        return tuple(cur.fetchone()[0])


def test_source_revisions_are_transactional_and_ignore_derived_writes(revision_db):
    dsn, schema = revision_db
    writer = _connect(dsn, schema)
    reader = _connect(dsn, schema)
    try:
        initial = _revision_snapshot(reader)

        with writer.cursor() as cur:
            cur.execute("INSERT INTO score (item_id, title) VALUES ('track-1', 'Original')")
        assert _revision_snapshot(writer) != initial
        assert _revision_snapshot(reader) == initial
        writer.rollback()
        assert _revision_snapshot(reader) == initial

        with writer.cursor() as cur:
            cur.execute("INSERT INTO score (item_id, title) VALUES ('track-1', 'Original')")
            cur.execute(
                "INSERT INTO embedding (item_id, embedding) VALUES "
                "('track-1', decode('0102', 'hex'))"
            )
        writer.commit()
        after_source_insert = _revision_snapshot(reader)
        assert after_source_insert != initial
        assert (
            sum(current != previous for current, previous in zip(after_source_insert, initial)) == 1
        )

        with writer.cursor() as cur:
            cur.execute("UPDATE score SET title = title WHERE item_id = 'track-1'")
            cur.execute(
                "UPDATE embedding SET poincare_embedding = decode('0304', 'hex'), "
                "hyperbolic_radius = 0.5 WHERE item_id = 'track-1'"
            )
        writer.commit()
        assert _revision_snapshot(reader) == after_source_insert

        with writer.cursor() as cur:
            cur.execute("UPDATE score SET title = 'Changed' WHERE item_id = 'track-1'")
        writer.commit()
        after_metadata_change = _revision_snapshot(reader)
        assert after_metadata_change != after_source_insert

        with writer.cursor() as cur:
            cur.execute(
                "INSERT INTO track_server_map "
                "(item_id, server_id, provider_track_id, file_path) "
                "VALUES ('track-1', 'server-1', 'provider-1', '/music/a.flac')"
            )
        writer.commit()
        after_membership_change = _revision_snapshot(reader)
        assert after_membership_change != after_metadata_change

        with writer.cursor() as cur:
            cur.execute(
                "UPDATE track_server_map SET updated_at = CURRENT_TIMESTAMP "
                "WHERE item_id = 'track-1'"
            )
        writer.commit()
        assert _revision_snapshot(reader) == after_membership_change

        with writer.cursor() as cur:
            cur.execute("INSERT INTO app_config (key, value) VALUES ('ivf_metric', 'angular')")
        writer.commit()
        after_config_change = _revision_snapshot(reader)
        assert after_config_change != after_membership_change

        tracked_inserts = (
            (
                "INSERT INTO clap_embedding (item_id, embedding) "
                "VALUES ('track-1', decode('0506', 'hex'))"
            ),
            (
                "INSERT INTO lyrics_embedding (item_id, embedding, axis_vector) "
                "VALUES ('track-1', decode('0708', 'hex'), decode('090a', 'hex'))"
            ),
            (
                "INSERT INTO artist_server_map "
                "(artist_name, server_id, provider_artist_id) "
                "VALUES ('Artist', 'server-1', 'artist-1')"
            ),
            (
                "INSERT INTO music_servers "
                "(server_id, name, server_type, creds, music_libraries, is_default) "
                "VALUES ('server-1', 'Server', 'navidrome', '{}'::jsonb, '', FALSE)"
            ),
        )
        for statement in tracked_inserts:
            before = _revision_snapshot(reader)
            with writer.cursor() as cur:
                cur.execute(statement)
            writer.commit()
            assert _revision_snapshot(reader) != before

        before_delete = _revision_snapshot(reader)
        with writer.cursor() as cur:
            cur.execute("DELETE FROM clap_embedding WHERE item_id = 'track-1'")
        writer.commit()
        assert _revision_snapshot(reader) != before_delete
    finally:
        writer.close()
        reader.close()
