# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every database statement the P2P node runs.

The node only READS the catalogue (on a read-only session) and only WRITES its
own app_config runtime rows: status, the servers it met and, once, its onion
service key.
It owns two autocommit connections named ipc.NODE_APP_NAME: the "ops" session
that also holds the node's advisory lock (so losing it ends the node), and a
read session that reconnects on its own. All calls are blocking; the runtime
runs them in worker threads, one at a time per connection.

Main Features:
* schema_ready / try_lock / lock_alive for the startup gate in node.py
* onion_key / save_onion_key: the Tor onion service key stored as
  "ED25519-V3:<base64>", inserted once with ON CONFLICT DO NOTHING and never
  rewritten; a malformed row is an error
* peer_cache / save_peer_cache: the AudioMuse-AI servers met so far, so the
  node finds them again after a restart without the directory
* local_track reads the MusiCNN embedding, duration and chromaprint of a song
  the worker asks about
* shareable_page / shareable_since page through the rows the node may index
  (signature ids with a known duration and a MusiCNN embedding); serve_rows
  re-reads rows, outputs, output_origin stamps and a chromaprint in ONE
  statement
* write_status and notify_all (the reply parts through
  taskqueue.sql.notify_p2p) on the ops session
"""

import base64
import binascii
import json
import logging
import threading
from typing import NamedTuple

import numpy as np
import psycopg2

import config
import database
from p2p import ipc
from taskqueue import sql as queue_sql
from tasks.simhash import CANONICAL_ID_LEN

logger = logging.getLogger(__name__)

NODE_LOCK_KEY = 7421536190083101
ONION_KEY_TYPE = 'ED25519-V3'
ONION_KEY_BYTES = 64
PAGE_ROWS = 20000
_SIGNATURE_ID_LEN = CANONICAL_ID_LEN

_SIGNATURE_ROW_SQL = (
    "left(s.item_id, 3) = 'fp_' AND length(s.item_id) = %s "
    "AND substring(s.item_id from 4 for 1) BETWEEN '1' AND '9' "
    "AND s.duration > 0 AND e.embedding IS NOT NULL"
)
_SHAREABLE_COLUMNS = (
    "SELECT s.item_id, s.duration, s.created_at, e.output_origin->'m' "
    "FROM score s JOIN embedding e ON e.item_id = s.item_id "
)


class LocalTrack(NamedTuple):
    embedding: np.ndarray
    duration: float
    chromaprint: bytes


class ShareableRow(NamedTuple):
    item_id: str
    duration: float
    created_at: object
    musicnn_origin: object


class ServeRow(NamedTuple):
    item_id: str
    duration: float
    embedding: bytes
    origin: dict
    neural: bytes
    clap: bytes
    lyrics: bytes
    axis: bytes
    chromaprint: bytes


class LockLost(Exception):
    pass


def onion_key_text(content):
    return f'{ONION_KEY_TYPE}:{content}'


def parse_onion_key(text):
    prefix = ONION_KEY_TYPE + ':'
    if not isinstance(text, str) or not text.startswith(prefix):
        raise ValueError('the P2P onion key row is not an ED25519-V3 key')
    content = text[len(prefix):]
    try:
        raw = base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError('the P2P onion key row is not valid base64') from exc
    if len(raw) != ONION_KEY_BYTES:
        raise ValueError(f'the P2P onion key row does not hold a {ONION_KEY_BYTES}-byte key')
    return content


def _origin(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _bytes(value):
    return bytes(value) if value is not None else None


def connect(read_only=False):
    conn = database.connect_raw(application_name=ipc.NODE_APP_NAME, read_only=read_only)
    conn.autocommit = True
    return conn


def schema_ready(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public.app_config') IS NOT NULL, EXISTS (SELECT 1 FROM "
            "information_schema.columns WHERE table_schema = 'public' AND table_name = 'embedding' "
            "AND column_name = 'output_origin')"
        )
        row = cur.fetchone()
    return bool(row and row[0] and row[1])


def try_lock(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (NODE_LOCK_KEY,))
        return bool(cur.fetchone()[0])


def write_status_on(conn, status):
    database.set_app_config_value(ipc.STATUS_KEY, json.dumps(status, separators=(',', ':')), conn=conn)


class Store:
    def __init__(self, ops_conn):
        self._ops = ops_conn
        self._ops_lock = threading.Lock()
        self._read = None
        self._read_lock = threading.Lock()

    def close(self):
        for conn in (self._read,):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    logger.exception("Could not close a P2P node database connection")
        self._read = None

    def _on_ops(self, operation):
        with self._ops_lock:
            try:
                with self._ops.cursor() as cur:
                    return operation(cur)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                raise LockLost('the P2P node lost the connection that holds its lock') from exc

    def _on_read(self, operation):
        with self._read_lock:
            for attempt in (1, 2):
                if self._read is None or self._read.closed:
                    self._read = connect(read_only=True)
                try:
                    with self._read.cursor() as cur:
                        return operation(cur)
                except (psycopg2.OperationalError, psycopg2.InterfaceError):
                    try:
                        self._read.close()
                    except Exception:
                        logger.exception("Could not close the dropped P2P read connection")
                    self._read = None
                    if attempt == 2:
                        raise
                    logger.warning("The P2P node read connection dropped; reconnecting once", exc_info=True)

    def lock_alive(self):
        def check(cur):
            cur.execute(
                "SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid() "
                "AND classid = %s::bigint::oid AND objid = %s::bigint::oid AND objsubid = 1 AND granted",
                (NODE_LOCK_KEY >> 32, NODE_LOCK_KEY & 0xFFFFFFFF),
            )
            return cur.fetchone() is not None
        return self._on_ops(check)

    def onion_key(self):
        text = self._on_ops(lambda cur: database.get_app_config_value(ipc.ONION_KEY_KEY, conn=cur.connection))
        return None if text is None else parse_onion_key(text)

    def save_onion_key(self, content):
        def insert(cur):
            cur.execute(
                "INSERT INTO app_config (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
                (ipc.ONION_KEY_KEY, onion_key_text(content)),
            )
            cur.execute("SELECT value FROM app_config WHERE key = %s", (ipc.ONION_KEY_KEY,))
            row = cur.fetchone()
            return row[0] if row else None
        return parse_onion_key(self._on_ops(insert))

    def peer_cache(self):
        raw = self._on_ops(lambda cur: database.get_app_config_value(ipc.PEERS_KEY, conn=cur.connection))
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            logger.exception("The %s app_config row is not valid JSON; starting with no known servers",
                             ipc.PEERS_KEY)
            return {}
        return data if isinstance(data, dict) else {}

    def save_peer_cache(self, data):
        text = json.dumps(data, separators=(',', ':'))
        self._on_ops(lambda cur: database.set_app_config_value(ipc.PEERS_KEY, text, conn=cur.connection))

    def write_status(self, status):
        self._on_ops(lambda cur: write_status_on(cur.connection, status))

    def notify_all(self, channel, payloads):
        def send(cur):
            for payload in payloads:
                queue_sql.notify_p2p(cur, channel, payload)
        self._on_ops(send)

    def local_track(self, item_id, server_id, provider_id):
        def read(cur):
            cur.execute(
                "SELECT e.embedding, s.duration, COALESCE("
                "(SELECT cp.fingerprint FROM chromaprint cp WHERE cp.server_id = %s "
                "AND cp.provider_track_id = %s AND cp.fingerprint IS NOT NULL), "
                "(SELECT cp.fingerprint FROM track_server_map t JOIN chromaprint cp "
                "ON cp.server_id = t.server_id AND cp.provider_track_id = t.provider_track_id "
                "WHERE t.item_id = s.item_id AND cp.fingerprint IS NOT NULL ORDER BY t.server_id LIMIT 1)) "
                "FROM score s JOIN embedding e ON e.item_id = s.item_id WHERE s.item_id = %s",
                (server_id, provider_id, str(item_id)),
            )
            return cur.fetchone()
        row = self._on_read(read)
        if not row or row[0] is None or row[1] is None:
            return None
        raw = bytes(row[0])
        if len(raw) != config.EMBEDDING_DIMENSION * 4:
            return None
        embedding = np.frombuffer(raw, dtype=np.float32).copy()
        return LocalTrack(embedding, float(row[1]), _bytes(row[2]))

    def _shareable(self, cur, where, params):
        cur.execute(
            _SHAREABLE_COLUMNS + "WHERE " + _SIGNATURE_ROW_SQL + where,
            (_SIGNATURE_ID_LEN,) + tuple(params),
        )
        return [
            ShareableRow(row[0], float(row[1]), row[2], _origin(row[3]) if row[3] is not None else None)
            for row in cur.fetchall()
        ]

    def shareable_page(self, after_item_id, limit=PAGE_ROWS):
        return self._on_read(lambda cur: self._shareable(
            cur, " AND s.item_id > %s ORDER BY s.item_id LIMIT %s", (after_item_id or '', int(limit)),
        ))

    def shareable_since(self, created_after, after_item_id='', limit=PAGE_ROWS):
        return self._on_read(lambda cur: self._shareable(
            cur, " AND (s.created_at, s.item_id) > (%s, %s) ORDER BY s.created_at, s.item_id LIMIT %s",
            (created_after, after_item_id or '', int(limit)),
        ))

    def serve_rows(self, item_ids):
        ids = [str(item_id) for item_id in item_ids]
        if not ids:
            return {}

        def read(cur):
            cur.execute(
                "SELECT s.item_id, s.duration, e.embedding, e.output_origin, e.neural_fingerprint, "
                "c.embedding, l.embedding, l.axis_vector, "
                "(SELECT cp.fingerprint FROM track_server_map t JOIN chromaprint cp "
                "ON cp.server_id = t.server_id AND cp.provider_track_id = t.provider_track_id "
                "WHERE t.item_id = s.item_id AND cp.fingerprint IS NOT NULL ORDER BY t.server_id LIMIT 1) "
                "FROM score s JOIN embedding e ON e.item_id = s.item_id "
                "LEFT JOIN clap_embedding c ON c.item_id = s.item_id "
                "LEFT JOIN lyrics_embedding l ON l.item_id = s.item_id "
                "WHERE s.item_id = ANY(%s)",
                (ids,),
            )
            return cur.fetchall()

        rows = {}
        for row in self._on_read(read):
            if row[1] is None or row[2] is None:
                continue
            rows[row[0]] = ServeRow(
                row[0], float(row[1]), bytes(row[2]), _origin(row[3]), _bytes(row[4]),
                _bytes(row[5]), _bytes(row[6]), _bytes(row[7]), _bytes(row[8]),
            )
        return rows
