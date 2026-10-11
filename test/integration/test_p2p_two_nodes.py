# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Two real P2P nodes that reach each other as Tor onion services.

Server A holds analysed songs; server B holds a slightly different copy of
each (a re-encode: signature within the Hamming tolerance, cosine within the
duplicate threshold) plus a decoy 5 s longer. Each server runs the real
`python -m p2p.node` on its own Postgres database, as its supervisor would, and
each node starts its own real Tor process and publishes its onion service on
the live Tor network. No node listens on anything but 127.0.0.1: B reaches A
only through Tor, exactly as two servers behind NAT. The public directory is
off (P2P_DIRECTORY_RELAYS empty) so nothing is published outside the test; B
learns A's onion address from its saved list of servers. B's analysis side is
the real worker client (p2p.client.Session) talking to B's node over Postgres
NOTIFY.

Main Features:
* neither node nor its Tor process listens on a non-loopback address; B
  reaches Connected; B's worker gets exactly A's CLAP, lyrics (with the axis
  vector) and neural fingerprint for the copy, each with A's version and A's
  onion address as the peer
* a result A received from another peer is served with the version that peer
  stamped, and a result A analysed before stamps existed with A's own
  APP_VERSION
* the +5 s decoy is refused, and a CLAP stamped v2.9.0 on A is never served
* A restarts: it comes back with the same onion address (its key is stored)
  and B gets A's results again with no restart of its own
* neither node writes a catalogue table on either side
* SIGTERM stops a node within seconds and leaves the Not running status
"""

import json
import os
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import psutil
import psycopg2
import pytest
from psycopg2.extensions import make_dsn, parse_dsn

import config
from p2p import client, ipc
from tasks import neural_fingerprint, nfp_format, simhash

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

pytestmark = pytest.mark.integration

A_VERSION = 'v3.6.3'
LYRICS_VERSION = 'v3.5.0'
OLD_VERSION = 'v2.9.0'
RECEIVED_M_VERSION = 'v3.2.0'
RECEIVED_C_VERSION = 'v3.4.1'
THIRD_PEER = 'thirdserver' * 5 + 'x'
START_SECONDS = 300
CONNECT_SECONDS = 180
STOP_SECONDS = 15
DURATION = 200.0

_DDL = (
    "CREATE TABLE app_config (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, other_features TEXT, "
    "duration DOUBLE PRECISION, created_at TIMESTAMP NOT NULL DEFAULT now())",
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) ON DELETE CASCADE, "
    "embedding BYTEA, neural_fingerprint BYTEA, output_origin JSONB)",
    "CREATE TABLE clap_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) ON DELETE CASCADE, "
    "embedding BYTEA)",
    "CREATE TABLE lyrics_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) ON DELETE CASCADE, "
    "embedding BYTEA, axis_vector BYTEA, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE track_server_map (item_id TEXT NOT NULL REFERENCES score (item_id) ON DELETE CASCADE, "
    "server_id TEXT NOT NULL, provider_track_id TEXT NOT NULL, file_path TEXT, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (item_id, server_id))",
    "CREATE TABLE chromaprint (server_id TEXT NOT NULL, provider_track_id TEXT NOT NULL, fingerprint BYTEA, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (server_id, provider_track_id))",
)
_CATALOGUE = ('score', 'embedding', 'clap_embedding', 'lyrics_embedding', 'track_server_map', 'chromaprint')


def _unit(dim, seed):
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _axis(seed):
    rng = np.random.default_rng(seed)
    parts = [rng.random(size).astype(np.float32) for size in (6, 6, 6, 5, 4)]
    return np.concatenate([part / part.sum() for part in parts]).astype(np.float32)


def _musicnn(seed):
    return (np.random.default_rng(seed).random(200) * 0.5).astype(np.float32)


def _neural(book_id, duration, seed):
    count = int(duration * 2)
    header = nfp_format.HEADER.pack(nfp_format.BLOB_MAGIC, nfp_format.DIM, count, nfp_format.PQ_SUBSPACES, book_id)
    codes = np.random.default_rng(seed).integers(0, 256, count * nfp_format.CODE_BYTES, dtype=np.uint8)
    return header + codes.tobytes()


def _item_id(embedding):
    return simhash.canonical_id_str(simhash.embedding_signature(embedding))


def _insert(conn, item_id, embedding, duration, origin, server, track, clap=None, lyrics=None, neural=None):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO score (item_id, title, author, duration) VALUES (%s, 'Title', 'Author', %s)",
                    (item_id, duration))
        cur.execute("INSERT INTO embedding (item_id, embedding, neural_fingerprint, output_origin) "
                    "VALUES (%s, %s, %s, %s)",
                    (item_id, embedding.tobytes(), neural, json.dumps(origin)))
        if clap is not None:
            cur.execute("INSERT INTO clap_embedding (item_id, embedding) VALUES (%s, %s)", (item_id, clap.tobytes()))
        if lyrics is not None:
            cur.execute("INSERT INTO lyrics_embedding (item_id, embedding, axis_vector) VALUES (%s, %s, %s)",
                        (item_id, lyrics[0].tobytes(), lyrics[1].tobytes()))
        cur.execute("INSERT INTO track_server_map (item_id, server_id, provider_track_id) VALUES (%s, %s, %s)",
                    (item_id, server, track))
    conn.commit()


def _counts(conn):
    counts = {}
    with conn.cursor() as cur:
        for table in _CATALOGUE:
            cur.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
        cur.execute("SELECT md5(string_agg(item_id || coalesce(output_origin::text, ''), ',' ORDER BY item_id)) "
                    "FROM embedding")
        counts['origin_digest'] = cur.fetchone()[0]
    conn.commit()
    return counts


def _status(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM app_config WHERE key = 'p2p_status'")
        row = cur.fetchone()
    conn.commit()
    return json.loads(row[0]) if row else {}


def _log_tail(path, lines=60):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            return ''.join(handle.readlines()[-lines:])
    except OSError:
        return ''


def _wait(predicate, seconds, what, logs=()):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(1.0)
    tails = '\n'.join(f'--- {path} ---\n{_log_tail(path)}' for path in logs)
    pytest.fail(f'Timed out after {seconds} s waiting for {what}\n{tails}')


def _network_listeners(pid):
    found = []
    process = psutil.Process(pid)
    for item in [process] + process.children(recursive=True):
        for conn in item.net_connections(kind='inet'):
            if conn.status == psutil.CONN_LISTEN and conn.laddr.ip not in ('127.0.0.1', '::1'):
                found.append((item.name(), conn.laddr))
    return found


def _children(pid):
    return [child.name() for child in psutil.Process(pid).children(recursive=True)]


def _seed_peers(conn, onion):
    payload = json.dumps({'v': 1, 'peers': {onion: {'heard': int(time.time()), 'ok': None}}})
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO app_config (key, value) VALUES ('p2p_peers', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (payload,),
        )
    conn.commit()


def _stop(process, handle):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=STOP_SECONDS)
    handle.close()
    return process.returncode


class Node:
    def __init__(self, name, dsn_parts, dbname, models, log_dir):
        self.name = name
        self.log_path = str(log_dir / f'node_{name}.log')
        temp_dir = log_dir / f'tmp_{name}'
        temp_dir.mkdir()
        env = dict(os.environ)
        env.pop('DATABASE_URL', None)
        env.update({
            'SERVICE_TYPE': 'flask',
            'POSTGRES_HOST': dsn_parts.get('host') or 'localhost',
            'POSTGRES_PORT': str(dsn_parts.get('port') or '5432'),
            'POSTGRES_USER': dsn_parts.get('user') or 'postgres',
            'POSTGRES_PASSWORD': dsn_parts.get('password') or '',
            'POSTGRES_DB': dbname,
            'P2P_ENABLED': 'true',
            'P2P_DIRECTORY_RELAYS': '',
            'TEMP_DIR': str(temp_dir),
            'PYTHONUNBUFFERED': '1',
        })
        env.update(models)
        self.handle = open(self.log_path, 'w', encoding='utf-8')
        self.process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'p2p.node'], cwd=_REPO_ROOT, env=env,
            stdout=self.handle, stderr=subprocess.STDOUT,
        )

    def stop(self):
        return _stop(self.process, self.handle)


@pytest.fixture
def two_databases(shared_pg_dsn):
    parts = parse_dsn(shared_pg_dsn)
    suffix = uuid.uuid4().hex[:8]
    names = {'a': f'audiomuse_p2p_a_{suffix}', 'b': f'audiomuse_p2p_b_{suffix}'}
    admin = psycopg2.connect(shared_pg_dsn)
    admin.autocommit = True
    with admin.cursor() as cur:
        for name in names.values():
            cur.execute(f'CREATE DATABASE {name}')
    conns = {}
    for key, name in names.items():
        conn = psycopg2.connect(**dict(parts, dbname=name))
        with conn.cursor() as cur:
            for ddl in _DDL:
                cur.execute(ddl)
        conn.commit()
        conns[key] = conn
    yield parts, names, conns
    for conn in conns.values():
        conn.close()
    with admin.cursor() as cur:
        for name in names.values():
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,))
            cur.execute(f'DROP DATABASE IF EXISTS {name}')
    admin.close()


@pytest.fixture
def models(tmp_path):
    model_dir = tmp_path / 'models'
    (model_dir / 'gte').mkdir(parents=True)
    book = np.random.default_rng(11).standard_normal(
        (nfp_format.PQ_SUBSPACES, nfp_format.PQ_CENTROIDS, nfp_format.PQ_SUBDIM)).astype(np.float32)
    np.savez(model_dir / 'pq.npz', codebook=book)
    for name, content in (('musicnn.onnx', b'musicnn stand-in'), ('clap.onnx', b'clap stand-in'),
                          ('neural.onnx', b'neural stand-in'), ('gte.onnx', b'gte stand-in'),
                          ('gte/tokenizer.json', b'{"stand-in": true}')):
        (model_dir / name).write_bytes(content)
    env = {
        'EMBEDDING_MODEL_PATH': str(model_dir / 'musicnn.onnx'),
        'CLAP_AUDIO_MODEL_PATH': str(model_dir / 'clap.onnx'),
        'NEURAL_FINGERPRINT_MODEL_PATH': str(model_dir / 'neural.onnx'),
        'NEURAL_FINGERPRINT_CODEBOOK_PATH': str(model_dir / 'pq.npz'),
        'LYRICS_GTE_ONNX_PATH': str(model_dir / 'gte.onnx'),
        'LYRICS_GTE_TOKENIZER_DIR': str(model_dir / 'gte'),
    }
    return env, nfp_format.codebook_id(book)


def test_two_nodes_share_results_only_within_the_rules(two_databases, models, tmp_path, monkeypatch):
    parts, names, conns = two_databases
    model_env, book_id = models
    conn_a, conn_b = conns['a'], conns['b']

    song = _musicnn(1)
    other = _musicnn(2)
    legacy = _musicnn(10)
    relayed = _musicnn(12)
    clap = _unit(512, 3)
    lyrics = (_unit(768, 4), _axis(5))
    neural = _neural(book_id, DURATION, 6)
    legacy_clap = _unit(512, 11)
    relayed_clap = _unit(512, 13)
    stamped = {'m': {'v': A_VERSION}, 'c': {'v': A_VERSION}, 'l': {'v': LYRICS_VERSION}, 'n': {'v': A_VERSION}}
    received = {'m': {'v': RECEIVED_M_VERSION, 'p': THIRD_PEER}, 'c': {'v': RECEIVED_C_VERSION, 'p': THIRD_PEER}}
    _insert(conn_a, _item_id(song), song, DURATION, stamped, 'srv-a', 'trk-song',
            clap=clap, lyrics=lyrics, neural=neural)
    _insert(conn_a, _item_id(other), other, 180.0, {'m': {'v': A_VERSION}, 'c': {'v': OLD_VERSION}},
            'srv-a', 'trk-other', clap=_unit(512, 7))
    _insert(conn_a, _item_id(legacy), legacy, 190.0, {}, 'srv-a', 'trk-legacy', clap=legacy_clap)
    _insert(conn_a, _item_id(relayed), relayed, 170.0, received, 'srv-a', 'trk-relayed', clap=relayed_clap)

    def _copy(embedding, seed):
        return embedding + np.random.default_rng(seed).standard_normal(200).astype(np.float32) * 1e-3

    copy, other_copy = _copy(song, 8), _copy(other, 9)
    legacy_copy, relayed_copy = _copy(legacy, 14), _copy(relayed, 15)
    decoy = song.copy()
    decoy[0] += 1e-4
    copy_id, other_copy_id = _item_id(copy), _item_id(other_copy)
    legacy_copy_id, relayed_copy_id = _item_id(legacy_copy), _item_id(relayed_copy)
    decoy_id = simhash.mint_canonical_id(
        simhash.embedding_signature(decoy), {copy_id, other_copy_id, legacy_copy_id, relayed_copy_id},
    )
    local = {'m': {'v': A_VERSION}}
    _insert(conn_b, copy_id, copy, DURATION + 0.4, local, 'srv-b', 'trk-copy')
    _insert(conn_b, other_copy_id, other_copy, 180.2, local, 'srv-b', 'trk-other')
    _insert(conn_b, legacy_copy_id, legacy_copy, 190.3, local, 'srv-b', 'trk-legacy')
    _insert(conn_b, relayed_copy_id, relayed_copy, 170.1, local, 'srv-b', 'trk-relayed')
    _insert(conn_b, decoy_id, decoy, DURATION + 5.0, local, 'srv-b', 'trk-decoy')
    before = {'a': _counts(conn_a), 'b': _counts(conn_b)}

    node_a = Node('a', parts, names['a'], model_env, tmp_path)
    nodes = [node_a]
    try:
        logs = [node_a.log_path]
        _wait(lambda: _status(conn_a).get('peer_id') and _status(conn_a).get('index') == 4, START_SECONDS,
              "node A to publish its onion service and index its four songs", logs)
        peer_a = _status(conn_a)['peer_id']
        _seed_peers(conn_b, peer_a)
        node_b = Node('b', parts, names['b'], model_env, tmp_path)
        nodes.append(node_b)
        logs.append(node_b.log_path)
        _wait(lambda: _status(conn_b).get('state') == ipc.STATE_CONNECTED, START_SECONDS + CONNECT_SECONDS,
              "node B to reach node A through Tor", logs)
        for node in nodes:
            assert 'tor' in _children(node.process.pid), _log_tail(node.log_path)
            assert _network_listeners(node.process.pid) == [], _log_tail(node.log_path)

        monkeypatch.setattr(config, 'DATABASE_URL', make_dsn(**dict(parts, dbname=names['b'])))
        monkeypatch.setattr(config, 'P2P_ENABLED', True)
        monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', model_env['NEURAL_FINGERPRINT_CODEBOOK_PATH'])
        for key in ('codebook', 'codebook_id', 'codebook_bias'):
            monkeypatch.setitem(neural_fingerprint._STATE, key, None)

        session = client.Session.open(lambda: None)
        assert session is not None and session.live, _log_tail(node_b.log_path)
        try:
            supply = session.ask(copy_id, 'srv-b', 'trk-copy', ['c', 'l', 'n'])
            assert supply is not None, _log_tail(node_b.log_path) + _log_tail(node_a.log_path)
            assert supply.peer == peer_a
            assert supply.versions == {'m': A_VERSION, 'c': A_VERSION, 'l': LYRICS_VERSION, 'n': A_VERSION}
            assert np.array_equal(supply.outputs['c'], clap)
            assert np.array_equal(supply.outputs['l'], lyrics[0])
            assert np.array_equal(supply.outputs['a'], lyrics[1])
            assert supply.outputs['n'] == neural

            legacy_supply = session.ask(legacy_copy_id, 'srv-b', 'trk-legacy', ['c'])
            assert legacy_supply is not None, _log_tail(node_a.log_path)
            assert legacy_supply.versions == {'m': config.APP_VERSION, 'c': config.APP_VERSION}
            assert np.array_equal(legacy_supply.outputs['c'], legacy_clap)

            relayed_supply = session.ask(relayed_copy_id, 'srv-b', 'trk-relayed', ['c'])
            assert relayed_supply is not None, _log_tail(node_a.log_path)
            assert relayed_supply.peer == peer_a
            assert relayed_supply.versions == {'m': RECEIVED_M_VERSION, 'c': RECEIVED_C_VERSION}
            assert np.array_equal(relayed_supply.outputs['c'], relayed_clap)

            assert session.ask(decoy_id, 'srv-b', 'trk-decoy', ['c', 'l', 'n']) is None
            assert session.ask(other_copy_id, 'srv-b', 'trk-other', ['c']) is None

            first_log = node_a.log_path
            assert node_a.stop() == 0, _log_tail(node_a.log_path)
            assert _status(conn_a).get('state') == ipc.STATE_NOT_RUNNING
            node_a = Node('a_again', parts, names['a'], model_env, tmp_path)
            nodes.append(node_a)
            logs.append(node_a.log_path)
            _wait(lambda: _status(conn_a).get('peer_id') == peer_a and _status(conn_a).get('index') == 4,
                  START_SECONDS, "node A to come back with the same onion address", logs)
            again = _wait(lambda: session.ask(copy_id, 'srv-b', 'trk-copy', ['c']), CONNECT_SECONDS * 2,
                          "node B to get A's results again after A restarted", logs)
            assert again.peer == peer_a
            assert np.array_equal(again.outputs['c'], clap)
        finally:
            session.close()

        assert _counts(conn_a) == before['a']
        assert _counts(conn_b) == before['b']
        served = [
            line for path in (first_log, node_a.log_path)
            for line in _log_tail(path, 2000).splitlines() if 'P2P served' in line
        ]
        assert served and not [line for line in served if OLD_VERSION in line]

        assert node_b.stop() == 0, _log_tail(node_b.log_path)
        assert _status(conn_b).get('state') == ipc.STATE_NOT_RUNNING
    finally:
        for node in nodes:
            node.stop()
