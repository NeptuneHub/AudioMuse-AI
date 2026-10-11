# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The output provenance stamps in database.py, against a recording cursor.

Every model output carries the AudioMuse-AI version that computed it in the one
nullable embedding.output_origin column, whether P2P is on or off, so outputs
computed before P2P is enabled can still be shared later. A result supplied by
a peer is saved by the same save function with origin=(version, peer id), and
its stamp is written in the same transaction as the data. These tests pin the
statements each save runs, in which transaction, and what it returns; the real
SQL semantics are covered by test/integration/test_p2p_database_integration.py.

Main Features:
* init_db adds output_origin to embedding (never to score)
* each local save stamps its own key with config.APP_VERSION on the same cursor
  before the commit: MusiCNN only with a real embedding, neural only when the
  row was written, nothing for an empty CLAP or lyrics vector
* origin=(version, peer) makes the CLAP, lyrics and neural saves stamp
  {"v": version, "p": peer} for their key instead, in the same single commit
* the stamp replaces the whole key, so a later local save drops the peer id
* a failing stamp rolls the whole save back, local or peer
"""

import inspect
import re

import numpy as np
import pytest

import config
import database

PEER = 'peeralphatestonly' + 'a' * 39


class _Cursor:
    def __init__(self, rowcount=1, fail_on=None):
        self.executed = []
        self.rowcount = rowcount
        self._fail_on = fail_on
        self.closed = False

    def execute(self, query, params=None):
        if self._fail_on and self._fail_on in query:
            raise RuntimeError('statement failed')
        self.executed.append((' '.join(query.split()), params))

    def close(self):
        self.closed = True


class _Conn:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.events = []

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.events.append(('commit', len(self.cursor_obj.executed)))

    def rollback(self):
        self.events.append(('rollback', len(self.cursor_obj.executed)))


@pytest.fixture
def db(monkeypatch):
    def install(**kwargs):
        cursor = _Cursor(**kwargs)
        conn = _Conn(cursor)
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        return conn, cursor
    return install


def _stamps(cursor):
    return [(query, params) for query, params in cursor.executed
            if query.startswith('UPDATE embedding SET output_origin')]


def _stamp_params(cursor):
    return [params for _, params in _stamps(cursor)]


def _no_hyperbolic(monkeypatch):
    monkeypatch.setattr(database, '_persist_hyperbolic_inline', lambda item_id, vector, cur: None)


LOCAL_SQL = (
    "UPDATE embedding SET output_origin = COALESCE(output_origin, '{}'::jsonb) || "
    "jsonb_build_object(%s, jsonb_build_object('v', %s)) WHERE item_id = %s"
)
PEER_SQL = (
    "UPDATE embedding SET output_origin = COALESCE(output_origin, '{}'::jsonb) || "
    "jsonb_build_object(%s, jsonb_build_object('v', %s, 'p', %s)) WHERE item_id = %s"
)


class TestSchema:
    def test_init_db_adds_the_output_origin_column_to_embedding(self):
        source = inspect.getsource(database.init_db)
        assert "_ensure_column(cur, 'embedding', 'output_origin', 'output_origin JSONB')" in source
        assert not re.search(r"'score',\s*'output_origin'", source)

    def test_the_keys_are_the_documented_letters(self):
        assert (database.OUTPUT_MUSICNN, database.OUTPUT_CLAP, database.OUTPUT_LYRICS,
                database.OUTPUT_NEURAL) == ('m', 'c', 'l', 'n')

    def test_both_stamps_replace_one_whole_key_and_keep_the_others(self):
        assert ' '.join(database._STAMP_LOCAL_OUTPUT_SQL.split()) == LOCAL_SQL
        assert ' '.join(database._STAMP_PEER_OUTPUT_SQL.split()) == PEER_SQL


class TestLocalStamps:
    def test_musicnn_save_stamps_m_before_the_commit(self, db, monkeypatch):
        _no_hyperbolic(monkeypatch)
        conn, cursor = db()
        database.save_track_analysis_and_embedding(
            'item-1', 'Title', 'Author', 120.0, 'C', 'major', {'happy': 0.5}, np.ones(4, dtype=np.float32)
        )
        assert _stamps(cursor) == [(LOCAL_SQL, ('m', config.APP_VERSION, 'item-1'))]
        stamp_index = next(i for i, (q, _) in enumerate(cursor.executed) if q.startswith('UPDATE embedding SET output_origin'))
        assert cursor.executed[stamp_index - 1][0].startswith('INSERT INTO embedding (item_id, embedding)')
        assert conn.events == [('commit', len(cursor.executed))]

    def test_musicnn_save_without_an_embedding_writes_no_stamp(self, db, monkeypatch):
        _no_hyperbolic(monkeypatch)
        conn, cursor = db()
        database.save_track_analysis_and_embedding(
            'item-1', 'Title', 'Author', 120.0, 'C', 'major', {'happy': 0.5}, np.zeros(0, dtype=np.float32)
        )
        assert _stamps(cursor) == []
        assert conn.events == [('commit', 1)]

    def test_the_stamp_uses_the_running_version(self, db, monkeypatch):
        monkeypatch.setattr(config, 'APP_VERSION', 'v9.8.7')
        conn, cursor = db()
        database.save_clap_embedding('item-2', np.ones(512, dtype=np.float32))
        assert _stamp_params(cursor) == [('c', 'v9.8.7', 'item-2')]

    def test_clap_save_stamps_c_in_the_same_transaction(self, db):
        conn, cursor = db()
        database.save_clap_embedding('item-2', np.ones(512, dtype=np.float32))
        assert cursor.executed[0][0].startswith('INSERT INTO clap_embedding')
        assert _stamps(cursor) == [(LOCAL_SQL, ('c', config.APP_VERSION, 'item-2'))]
        assert conn.events == [('commit', 2)]

    def test_empty_clap_writes_nothing(self, db):
        conn, cursor = db()
        database.save_clap_embedding('item-2', np.zeros(0, dtype=np.float32))
        database.save_clap_embedding('item-2', None)
        assert cursor.executed == [] and conn.events == []

    def test_lyrics_save_stamps_l_in_the_same_transaction(self, db):
        conn, cursor = db()
        database.save_lyrics_embedding('item-3', np.ones(768, dtype=np.float32), np.ones(27, dtype=np.float32))
        assert cursor.executed[0][0].startswith('INSERT INTO lyrics_embedding')
        assert _stamps(cursor) == [(LOCAL_SQL, ('l', config.APP_VERSION, 'item-3'))]
        assert conn.events == [('commit', 2)]

    def test_empty_lyrics_writes_nothing(self, db):
        conn, cursor = db()
        database.save_lyrics_embedding('item-3', None)
        assert cursor.executed == [] and conn.events == []

    def test_neural_save_stamps_n_only_when_the_row_was_written(self, db):
        conn, cursor = db(rowcount=1)
        assert database.save_neural_fingerprint('item-4', b'NFP2blob') is True
        assert _stamps(cursor) == [(LOCAL_SQL, ('n', config.APP_VERSION, 'item-4'))]
        assert conn.events == [('commit', 2)]

        conn, cursor = db(rowcount=0)
        assert database.save_neural_fingerprint('item-5', b'NFP2blob') is False
        assert _stamps(cursor) == []
        assert conn.events == [('commit', 1)]

    def test_empty_neural_blob_writes_nothing(self, db):
        conn, cursor = db()
        assert database.save_neural_fingerprint('item-4', b'') is False
        assert cursor.executed == []


class TestPeerOrigin:
    def test_a_peer_clap_is_stamped_with_the_peer_version_and_id_in_one_commit(self, db):
        conn, cursor = db()
        database.save_clap_embedding('item-2', np.ones(512, dtype=np.float32), origin=('v3.1.0', PEER))
        assert cursor.executed[0][0].startswith('INSERT INTO clap_embedding')
        assert _stamps(cursor) == [(PEER_SQL, ('c', 'v3.1.0', PEER, 'item-2'))]
        assert conn.events == [('commit', 2)]

    def test_a_peer_lyrics_result_is_stamped_in_one_commit(self, db):
        conn, cursor = db()
        database.save_lyrics_embedding(
            'item-3', np.ones(768, dtype=np.float32), np.ones(27, dtype=np.float32), origin=('v3.5.2', PEER)
        )
        assert cursor.executed[0][0].startswith('INSERT INTO lyrics_embedding')
        assert _stamps(cursor) == [(PEER_SQL, ('l', 'v3.5.2', PEER, 'item-3'))]
        assert conn.events == [('commit', 2)]

    def test_a_peer_neural_fingerprint_is_stamped_only_when_the_row_was_written(self, db):
        conn, cursor = db(rowcount=1)
        assert database.save_neural_fingerprint('item-4', b'NFP2blob', origin=('v3.6.0', PEER)) is True
        assert _stamps(cursor) == [(PEER_SQL, ('n', 'v3.6.0', PEER, 'item-4'))]
        assert conn.events == [('commit', 2)]

        conn, cursor = db(rowcount=0)
        assert database.save_neural_fingerprint('item-5', b'NFP2blob', origin=('v3.6.0', PEER)) is False
        assert _stamps(cursor) == []

    def test_the_origin_values_are_written_as_text(self, db):
        conn, cursor = db()
        database.save_clap_embedding('item-2', np.ones(512, dtype=np.float32), origin=(3, 7))
        assert _stamp_params(cursor) == [('c', '3', '7', 'item-2')]

    def test_a_local_save_after_a_peer_save_stamps_the_local_version_without_a_peer(self, db):
        conn, cursor = db()
        database.save_clap_embedding('item-2', np.ones(512, dtype=np.float32), origin=('v3.1.0', PEER))
        database.save_clap_embedding('item-2', np.ones(512, dtype=np.float32))
        assert _stamps(cursor) == [
            (PEER_SQL, ('c', 'v3.1.0', PEER, 'item-2')),
            (LOCAL_SQL, ('c', config.APP_VERSION, 'item-2')),
        ]


class TestRollback:
    @pytest.mark.parametrize('origin', [None, ('v3.1.0', PEER)])
    @pytest.mark.parametrize('save, args', [
        (database.save_clap_embedding, ('item-6', np.ones(512, dtype=np.float32))),
        (database.save_lyrics_embedding, ('item-6', np.ones(768, dtype=np.float32), None)),
        (database.save_neural_fingerprint, ('item-6', b'NFP2blob')),
    ])
    def test_a_failing_stamp_rolls_the_whole_save_back(self, db, save, args, origin):
        conn, cursor = db(fail_on='output_origin')
        with pytest.raises(RuntimeError):
            save(*args, origin=origin)
        assert [event for event, _ in conn.events] == ['rollback']
        assert cursor.closed
