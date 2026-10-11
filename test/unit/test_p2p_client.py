# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for p2p.client, the analysis worker's side of the P2P requests.

A fake Postgres connection plays the node: it records every NOTIFY the client
sends and schedules the reply parts on a fake monotonic clock that a fake
select() advances, so deadlines, late parts and the cancel check are exercised
without sleeping or a database.

Main Features:
* open: off means None without a connection, a status other than Connected
  gives a not-live session (7002), Connected LISTENs on a private channel
* request payloads fit one NOTIFY; big replies arrive as split parts, in any
  order, and are reassembled; parts after the deadline or of another request
  are ignored and no reply state outlives its request
* the cancel check runs on every wait loop and TaskCancelled propagates
* per output, only what cannot be stored is dropped (7003): a wrong size, a
  non-finite value, the lyrics sentinel, a neural blob of another codebook or
  a malformed one, a version below v3.0.0 (wire.version_allowed); a MusiCNN
  version below v3.0.0 drops the whole reply; any other finite value and any
  blob length is kept
* two consecutive timeouts set skipping; the exception policy lets database
  connectivity errors through and turns anything else into a logged None
"""

import json
import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest
from psycopg2 import InterfaceError, OperationalError

import config
from p2p import client, ipc, wire
from taskqueue.errors import TaskCancelled
from tasks import neural_fingerprint, nfp_format

PEER = 'peeralphatestonly' + 'a' * 39
CRC = 0x1234ABCD
ITEM = 'fp_2' + '0' * 49 + '1'
GOOD_VERSIONS = {'m': 'v3.6.0', 'c': 'v3.6.0', 'l': 'v3.5.2', 'n': 'v3.6.0'}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else repr(query)
        self.conn.executed.append(text)
        if self.conn.execute_error is not None:
            raise self.conn.execute_error
        if 'pg_notify' in text:
            channel, payload = params
            self.conn.sent.append((channel, payload))
            if channel == ipc.NODE_CHANNEL and self.conn.responder is not None:
                message = json.loads(payload)
                if message.get('op') == ipc.OP_SUPPLY:
                    self.conn.responder(self.conn, message)
        elif 'app_config' in text:
            self._row = self.conn.status_row

    def fetchone(self):
        return self._row

    def close(self):
        pass


class FakeConn:
    def __init__(self, clock, state=ipc.STATE_CONNECTED, age=3.0, responder=None):
        self.clock = clock
        self.status_row = (json.dumps({'state': state}), age)
        self.responder = responder
        self.notifies = []
        self.scheduled = []
        self.executed = []
        self.sent = []
        self.closed = False
        self.autocommit = False
        self.execute_error = None
        self.poll_error = None

    def cursor(self):
        return FakeCursor(self)

    def fileno(self):
        return 42

    def deliver(self, delay, channel, payload):
        self.scheduled.append((self.clock.now + delay, channel, payload))

    def next_due(self):
        return min((at for at, _, _ in self.scheduled), default=None)

    def poll(self):
        if self.poll_error is not None:
            raise self.poll_error
        due = sorted((s for s in self.scheduled if s[0] <= self.clock.now), key=lambda s: s[0])
        self.scheduled = [s for s in self.scheduled if s[0] > self.clock.now]
        for _, channel, payload in due:
            self.notifies.append(SimpleNamespace(channel=channel, payload=payload, pid=7))

    def close(self):
        self.closed = True


def _unit(dim, seed):
    vec = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _axis():
    return np.random.default_rng(4).random(wire.AXIS_DIM).astype(np.float32)


def _neural_blob(count, crc=CRC):
    header = nfp_format.HEADER.pack(
        nfp_format.BLOB_MAGIC, nfp_format.DIM, count, nfp_format.PQ_SUBSPACES, crc
    )
    return header + bytes(count * nfp_format.CODE_BYTES)


def _good_outputs(clap=None, lyrics=None, axis=None, blob=None):
    return {
        'c': ipc.encode_floats(_unit(config.CLAP_EMBEDDING_DIMENSION, 1) if clap is None else clap),
        'l': ipc.encode_floats(_unit(config.LYRICS_EMBEDDING_DIMENSION, 2) if lyrics is None else lyrics),
        'a': ipc.encode_floats(_axis() if axis is None else axis),
        'n': ipc.encode_bytes(_neural_blob(300) if blob is None else blob),
    }


def responder_for(outputs=None, versions=None, status=ipc.REPLY_OK, peer=PEER, delay=0.3,
                  step=0.01, reverse=False):
    def respond(conn, request):
        reply = ipc.supply_reply(
            request['rid'], status, peer=peer,
            versions=GOOD_VERSIONS if versions is None else versions,
            outputs=_good_outputs() if outputs is None else outputs,
        )
        parts = ipc.split_reply(reply)
        if reverse:
            parts = parts[::-1]
        for index, part in enumerate(parts):
            conn.deliver(delay + index * step, request['reply'], part)

    return respond


@pytest.fixture
def rig(monkeypatch):
    clock = FakeClock()
    holder = SimpleNamespace(clock=clock, conn=None, cancels=0, codebook_calls=0, connects=0)

    def fake_select(readable, writable, errored, timeout):
        due = holder.conn.next_due() if holder.conn is not None else None
        if due is not None and due <= clock.now + timeout:
            clock.now = max(clock.now, due)
            return readable, [], []
        clock.now += timeout
        return [], [], []

    def cancel_check():
        holder.cancels += 1

    def codebook():
        holder.codebook_calls += 1
        return None, CRC, None

    def connect_raw(**kwargs):
        holder.connects += 1
        holder.connect_kwargs = kwargs
        return holder.conn

    monkeypatch.setattr(config, 'P2P_ENABLED', True)
    monkeypatch.setattr(config, 'P2P_LOOKUP_TIMEOUT_SECONDS', 5.0)
    monkeypatch.setattr(client, 'time', SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(client, 'select', SimpleNamespace(select=fake_select))
    monkeypatch.setattr(client.database, 'connect_raw', connect_raw)
    monkeypatch.setattr(neural_fingerprint, 'codebook', codebook)
    holder.cancel_check = cancel_check

    def open_session(state=ipc.STATE_CONNECTED, responder=None, age=3.0):
        holder.conn = FakeConn(clock, state=state, age=age, responder=responder)
        return client.Session.open(holder.cancel_check)

    holder.open = open_session
    return holder


def _ask(session, wants=('c', 'l', 'n')):
    return session.ask(ITEM, 'srv-1', 'prov-1', list(wants))


def test_open_returns_none_and_never_connects_when_sharing_is_off(rig, monkeypatch):
    monkeypatch.setattr(config, 'P2P_ENABLED', False)
    assert client.Session.open(rig.cancel_check) is None
    assert rig.connects == 0


def test_a_node_that_is_not_connected_gives_a_not_live_session(rig, caplog):
    caplog.set_level(logging.WARNING)
    session = rig.open(state=ipc.STATE_NO_PEERS)
    assert session is not None and not session.live
    assert rig.conn.closed
    assert not any('LISTEN' in text for text in rig.conn.executed)
    assert _ask(session) is None
    assert rig.conn.sent == []
    assert "No peers found" in session.reason
    assert any('[7002]' in r.getMessage() for r in caplog.records)


def test_a_stale_status_row_reads_as_not_running(rig):
    session = rig.open(state=ipc.STATE_CONNECTED, age=ipc.STATUS_FRESH_SECONDS + 5)
    assert not session.live
    assert session.state == ipc.STATE_NOT_RUNNING


def test_a_connected_node_gives_a_live_session_listening_on_a_private_channel(rig):
    session = rig.open()
    assert session.live and not session.skipping
    assert re.fullmatch(r'audiomuse_p2p_r_[0-9a-f]{16}', session.channel)
    assert rig.conn.autocommit is True
    assert any('LISTEN' in text and session.channel in text for text in rig.conn.executed)
    assert rig.connect_kwargs['application_name'] == client.CLIENT_APP_NAME
    session.close()
    assert rig.conn.closed and not session.live
    session.close()


def test_open_lets_a_lost_database_through_and_logs_anything_else(rig, monkeypatch, caplog):
    def refused(**kwargs):
        raise OperationalError('connection refused')

    monkeypatch.setattr(client.database, 'connect_raw', refused)
    with pytest.raises(OperationalError):
        client.Session.open(rig.cancel_check)

    def broken(**kwargs):
        raise RuntimeError('bad dsn')

    monkeypatch.setattr(client.database, 'connect_raw', broken)
    caplog.set_level(logging.ERROR)
    session = client.Session.open(rig.cancel_check)
    assert session is not None and not session.live
    assert session.state == ipc.STATE_ERROR
    assert any('[7002]' in r.getMessage() and r.exc_info for r in caplog.records)


def test_a_full_supply_is_decoded_and_the_request_fits_one_notify(rig):
    session = rig.open(responder=responder_for())
    supply = _ask(session)

    assert isinstance(supply, client.Supply)
    assert supply.peer == PEER
    assert supply.versions == GOOD_VERSIONS
    assert set(supply.outputs) == {'c', 'l', 'a', 'n'}
    assert supply.outputs['c'].dtype == np.float32 and supply.outputs['c'].shape == (512,)
    assert supply.outputs['l'].shape == (config.LYRICS_EMBEDDING_DIMENSION,)
    assert supply.outputs['a'].shape == (wire.AXIS_DIM,)
    assert supply.outputs['n'] == _neural_blob(300)
    np.testing.assert_allclose(supply.outputs['c'], _unit(512, 1))

    channel, payload = rig.conn.sent[-1]
    assert channel == ipc.NODE_CHANNEL
    assert len(payload.encode('utf-8')) <= ipc.MAX_PAYLOAD_BYTES
    request = json.loads(payload)
    assert request['op'] == ipc.OP_SUPPLY
    assert request['item'] == ITEM and request['server'] == 'srv-1' and request['provider'] == 'prov-1'
    assert request['wants'] == ['c', 'l', 'n']
    assert request['reply'] == session.channel
    assert request['ttl'] == 5.0
    assert session.timeouts == 0


def test_a_reply_bigger_than_one_notify_is_reassembled_from_parts_in_any_order(rig):
    blob = _neural_blob(400)
    outputs = _good_outputs(blob=blob)
    reply = ipc.supply_reply('probe', ipc.REPLY_OK, peer=PEER, versions=GOOD_VERSIONS, outputs=outputs)
    parts = ipc.split_reply(reply)
    assert len(parts) > 2
    assert all(len(part.encode('utf-8')) <= ipc.MAX_PAYLOAD_BYTES for part in parts)

    session = rig.open(responder=responder_for(outputs=outputs, reverse=True))
    supply = _ask(session)
    assert supply is not None
    assert supply.outputs['n'] == blob
    assert session._assembler._parts == {}


def test_parts_after_the_deadline_are_never_read(rig, caplog):
    caplog.set_level(logging.WARNING)
    session = rig.open(responder=responder_for(delay=4.9, step=0.2))
    assert _ask(session) is None
    assert session.timeouts == 1
    assert any('[7002]' in r.getMessage() for r in caplog.records)
    assert session._assembler._parts == {}


def test_late_parts_of_an_earlier_request_are_ignored(rig):
    session = rig.open(responder=responder_for(delay=5.2))
    assert _ask(session) is None
    rig.conn.responder = responder_for(delay=0.5)
    supply = _ask(session)
    assert supply is not None and supply.peer == PEER
    assert session.timeouts == 0
    assert session._assembler._parts == {}


def test_notifications_of_another_channel_or_request_are_skipped(rig):
    def respond(conn, request):
        stray = ipc.supply_reply('another-rid', ipc.REPLY_OK, peer='straypeer' + 'b' * 47, versions=GOOD_VERSIONS,
                                 outputs=_good_outputs())
        for part in ipc.split_reply(stray):
            conn.deliver(0.1, request['reply'], part)
        conn.deliver(0.1, 'audiomuse_p2p_r_ffffffffffffffff', 'not json')
        responder_for(delay=0.2)(conn, request)

    session = rig.open(responder=respond)
    supply = _ask(session)
    assert supply is not None and supply.peer == PEER


def test_the_cancel_check_runs_on_every_wait_loop_and_cancel_propagates(rig):
    session = rig.open(responder=responder_for(delay=2.0))
    before = rig.cancels
    assert _ask(session) is not None
    assert rig.cancels - before >= 8

    def cancelled():
        raise TaskCancelled('stopped')

    session._cancel_check = cancelled
    with pytest.raises(TaskCancelled):
        _ask(session)
    assert session.timeouts == 0
    assert session._assembler._parts == {}


def test_two_consecutive_timeouts_make_the_session_skip(rig, caplog):
    caplog.set_level(logging.WARNING)
    session = rig.open()
    assert _ask(session) is None
    assert not session.skipping
    assert _ask(session) is None
    assert session.skipping
    sent = len(rig.conn.sent)
    rig.conn.responder = responder_for()
    assert _ask(session) is None
    assert len(rig.conn.sent) == sent
    assert 'did not answer 2 requests in a row' in session.reason
    assert sum('[7002]' in r.getMessage() for r in caplog.records) >= 3


def test_a_reply_resets_the_timeout_streak(rig):
    session = rig.open()
    assert _ask(session) is None
    rig.conn.responder = responder_for(status=ipc.REPLY_NONE, outputs={}, versions={})
    assert _ask(session) is None
    assert session.timeouts == 0
    rig.conn.responder = None
    assert _ask(session) is None
    assert not session.skipping


def test_a_busy_node_answer_is_logged_and_not_supplied(rig, caplog):
    caplog.set_level(logging.WARNING)
    session = rig.open(responder=responder_for(status=ipc.REPLY_BUSY, outputs={}, versions={}))
    assert _ask(session) is None
    assert any('[7002]' in r.getMessage() and 'busy' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('bad', [
    {'c': 'broken'},
    {'c': ipc.encode_floats(np.concatenate([[np.nan], np.zeros(511)]).astype(np.float32))},
    {'c': ipc.encode_floats(_unit(256, 3))},
    {'l': ipc.encode_floats(np.asarray(config.LYRICS_INSTRUMENTAL_EMBEDDING, dtype=np.float32))},
    {'l': ipc.encode_floats(np.full(768, np.inf, dtype=np.float32))},
    {'a': ipc.encode_floats(np.full(26, 0.5, dtype=np.float32))},
    {'a': ipc.encode_floats(np.full(27, np.nan, dtype=np.float32))},
    {'n': ipc.encode_bytes(_neural_blob(300, crc=CRC + 1))},
    {'n': ipc.encode_bytes(b'NFP2 truncated')},
])
def test_an_output_that_cannot_be_stored_is_dropped_and_the_rest_kept(rig, caplog, bad):
    caplog.set_level(logging.WARNING)
    outputs = dict(_good_outputs(), **bad)
    session = rig.open(responder=responder_for(outputs=outputs))
    supply = _ask(session)
    dropped = 'l' if 'a' in bad else next(iter(bad))
    assert supply is not None
    assert dropped not in supply.outputs
    assert dropped not in supply.versions
    kept = {'c', 'l', 'n'} - {dropped}
    assert kept <= set(supply.outputs)
    assert any('[7003]' in r.getMessage() for r in caplog.records)


def test_any_finite_value_and_any_blob_length_is_taken_as_it_is(rig, caplog):
    caplog.set_level(logging.WARNING)
    clap = np.full(512, 0.5, dtype=np.float32)
    lyrics = np.full(768, 3.0, dtype=np.float32)
    axis = np.full(27, 0.5, dtype=np.float32)
    blob = _neural_blob(5000)
    session = rig.open(responder=responder_for(outputs=_good_outputs(clap=clap, lyrics=lyrics, axis=axis, blob=blob)))
    supply = _ask(session)
    assert supply is not None
    np.testing.assert_array_equal(supply.outputs['c'], clap)
    np.testing.assert_array_equal(supply.outputs['l'], lyrics)
    np.testing.assert_array_equal(supply.outputs['a'], axis)
    assert supply.outputs['n'] == blob
    assert not any('[7003]' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('version', ['v2.9.9', 'v1.0.0', 'dev', None, '', 'v3'])
def test_an_output_below_v3_or_unparsable_is_dropped(rig, caplog, version):
    caplog.set_level(logging.WARNING)
    versions = dict(GOOD_VERSIONS, c=version)
    session = rig.open(responder=responder_for(versions=versions))
    supply = _ask(session)
    assert supply is not None
    assert 'c' not in supply.outputs and 'c' not in supply.versions
    assert {'l', 'a', 'n'} <= set(supply.outputs)
    assert any('[7003]' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('version', ['v2.9.9', None, 'unknown'])
def test_a_musicnn_version_below_v3_drops_the_whole_reply(rig, version):
    versions = dict(GOOD_VERSIONS, m=version)
    session = rig.open(responder=responder_for(versions=versions))
    assert _ask(session) is None


def test_v3_0_0_itself_is_accepted(rig):
    versions = {key: 'v3.0.0' for key in GOOD_VERSIONS}
    session = rig.open(responder=responder_for(versions=versions))
    supply = _ask(session)
    assert supply is not None and supply.versions['c'] == 'v3.0.0'


def test_unwanted_outputs_are_ignored_and_the_codebook_is_read_only_for_neural(rig):
    session = rig.open(responder=responder_for())
    supply = _ask(session, wants=('c',))
    assert set(supply.outputs) == {'c'}
    assert set(supply.versions) == {'m', 'c'}
    assert rig.codebook_calls == 0
    assert json.loads(rig.conn.sent[-1][1])['wants'] == ['c']

    supply = _ask(session, wants=('n', 'c'))
    assert set(supply.outputs) == {'c', 'n'}
    assert rig.codebook_calls == 1
    _ask(session, wants=('n',))
    assert rig.codebook_calls == 1


def test_an_unreadable_codebook_refuses_peer_neural_fingerprints(rig, monkeypatch, caplog):
    def missing():
        raise RuntimeError('codebook missing')

    monkeypatch.setattr(neural_fingerprint, 'codebook', missing)
    caplog.set_level(logging.WARNING)
    session = rig.open(responder=responder_for())
    supply = _ask(session)
    assert 'n' not in supply.outputs and 'c' in supply.outputs


def test_a_malformed_reply_is_rejected(rig):
    session = rig.open(responder=responder_for(peer=None))
    assert _ask(session) is None


def test_a_lost_database_during_a_request_propagates(rig):
    session = rig.open(responder=responder_for())
    rig.conn.execute_error = OperationalError('server closed the connection')
    with pytest.raises(OperationalError):
        _ask(session)
    rig.conn.execute_error = None
    rig.conn.poll_error = InterfaceError('connection already closed')
    with pytest.raises(InterfaceError):
        _ask(session)


def test_any_other_failure_is_logged_with_its_traceback_and_ends_as_none(rig, caplog):
    caplog.set_level(logging.ERROR)
    session = rig.open(responder=responder_for())
    rig.conn.poll_error = RuntimeError('unexpected')
    assert _ask(session) is None
    assert any('[7002]' in r.getMessage() and r.exc_info for r in caplog.records)
    rig.conn.poll_error = None
    assert _ask(session) is not None


def test_nothing_wanted_sends_nothing(rig):
    session = rig.open(responder=responder_for())
    assert _ask(session, wants=()) is None
    assert _ask(session, wants=('m', 'x')) is None
    assert rig.conn.sent == []
