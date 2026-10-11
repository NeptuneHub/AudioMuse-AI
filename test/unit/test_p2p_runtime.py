# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The running P2P node's own logic (p2p.runtime), with Tor, the relays and the database faked.

Main Features:
* _on_connection serves an inbound Tor stream as the anonymous member through
  a metered StreamIO and never lets an exception escape: a malformed frame is
  logged 7003, a stream that ends early stays quiet, anything else is logged
  with its traceback; the stream is always closed, even when closing fails
* _probe sends an empty PEERS then END, marks the server succeeded and learns
  the onion addresses it lists (never this server's own, never junk; an
  undecodable entry is logged 7003); a server Tor cannot reach, a malformed
  answer, a closed stream or no answer in time marks it failed
* current_state is Starting until the onion service is up, Connected while a
  server answers, No peers found after the startup grace; the status row
  carries the onion address as peer_id and no main network server
* _come_online starts Tor, serves on 127.0.0.1 only, publishes that port,
  stores a new onion key once (the stored key wins) and loads the saved list
  after it knows its own address
* _shutdown saves the list only once online, writes Not running and stops Tor
  even when the database write fails
* the directory loop is off without relays; otherwise it publishes the signed
  record once per 3 hours (again next round when no relay accepted it), reads
  the records every 10 minutes while few servers answer and hourly after,
  learns them, wakes the probe loop and survives a failing round
"""

import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import trio
import trio.testing

import config
from p2p import directory, ipc, limits, peers, runtime, wire
from p2p import tor as p2p_tor

SPACES = {key: bytes([index + 1]) * 32 for index, key in enumerate(wire.SPACE_KEYS)}
OWN = 'ownnode' + 'n' * 49
PEER = 'probed' + 'p' * 50
LISTED = 'listed' + 'l' * 50
ANOTHER = 'another' + 'a' * 49
ONION_KEY = base64.b64encode(bytes(range(64))).decode('ascii')
MALFORMED_PEERS = bytes([wire.WIRE_VERSION, wire.T_PEERS, 1, 5])


class FakeStream:
    def __init__(self, close_error=None):
        self.closed = False
        self.close_error = close_error

    async def receive_some(self, max_bytes=None):
        return b''

    async def send_all(self, data):
        return None

    async def aclose(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class RecordingServer:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def handle_stream(self, io, peer):
        self.calls.append((io, peer))
        if self.error is not None:
            raise self.error


class FakeIO:
    def __init__(self, answer):
        self.data = bytearray(answer)
        self.written = []

    async def read_exactly(self, size):
        chunk = bytes(self.data[:size])
        del self.data[:size]
        if len(chunk) < size:
            raise limits.StreamClosed('the peer closed the stream mid-frame')
        return chunk

    async def write(self, data):
        self.written.append(bytes(data))


class FakeStore:
    def __init__(self, stored_key=None, status_error=None):
        self.stored_key = stored_key
        self.status_error = status_error
        self.saved_keys = []
        self.caches = []
        self.statuses = []

    def save_onion_key(self, content):
        self.saved_keys.append(content)
        return self.stored_key or content

    def save_peer_cache(self, data):
        self.caches.append(data)

    def write_status(self, status):
        if self.status_error is not None:
            raise self.status_error
        self.statuses.append(status)


def _limits():
    return limits.Limits(upload_kibps=0, download_kibps=0, parallel_uploads=1, parallel_downloads=1)


def _serving_node(server):
    node = runtime.Runtime(store=None)
    node.server = server
    node.limits = _limits()
    return node


class TestInboundConnections:
    @pytest.mark.parametrize('error', [
        None,
        wire.Violation('a length of 9 exceeds its cap of 4'),
        limits.StreamClosed('the peer closed the stream mid-frame'),
        trio.BrokenResourceError(),
        trio.ClosedResourceError(),
        RuntimeError('boom'),
        KeyError('missing'),
        ValueError('bad value'),
    ])
    def test_no_exception_escapes_and_the_stream_is_always_closed(self, error):
        server = RecordingServer(error)
        node = _serving_node(server)
        stream = FakeStream()

        assert trio.run(node._on_connection, stream) is None

        assert stream.closed
        ((io, peer),) = server.calls
        assert peer == runtime.ANONYMOUS == 'member'
        assert isinstance(io, limits.MeteredStream)
        assert isinstance(io.stream, p2p_tor.StreamIO) and io.stream.stream is stream

    def test_a_malformed_frame_is_logged_as_a_rejected_request(self, caplog):
        node = _serving_node(RecordingServer(wire.Violation('the frame carries trailing bytes')))

        with caplog.at_level(logging.WARNING):
            trio.run(node._on_connection, FakeStream())

        assert '7003' in caplog.text
        assert 'malformed request from peer an AudioMuse-AI server' in caplog.text

    @pytest.mark.parametrize('error', [
        limits.StreamClosed('closed'), trio.BrokenResourceError(), trio.ClosedResourceError(),
    ])
    def test_a_stream_that_ends_early_stays_quiet(self, caplog, error):
        node = _serving_node(RecordingServer(error))

        with caplog.at_level(logging.INFO):
            trio.run(node._on_connection, FakeStream())

        assert not [record for record in caplog.records if record.levelno >= logging.INFO]

    def test_any_other_failure_is_logged_with_its_traceback(self, caplog):
        node = _serving_node(RecordingServer(RuntimeError('boom')))

        with caplog.at_level(logging.ERROR, logger='p2p.runtime'):
            trio.run(node._on_connection, FakeStream())

        (record,) = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert record.getMessage() == 'Serving a P2P request failed' and record.exc_info

    def test_a_close_that_fails_is_contained_too(self):
        node = _serving_node(RecordingServer(RuntimeError('boom')))
        stream = FakeStream(close_error=OSError('connection reset'))

        assert trio.run(node._on_connection, stream) is None
        assert stream.closed


def _prober(monkeypatch, answer=b'', error=None, hang=False):
    node = runtime.Runtime(store=None)
    node.book = peers.PeerBook(OWN, SPACES, 4)
    node.book.learn(PEER)
    opened = []

    @asynccontextmanager
    async def open_stream(onion):
        if error is not None:
            raise error
        if hang:
            await trio.sleep_forever()
        io = FakeIO(answer)
        opened.append((onion, io))
        yield io

    monkeypatch.setattr(node, '_open', open_stream)
    return node, opened


def _probe(node, onion=PEER):
    async def run():
        await node._probe(onion, trio.CapacityLimiter(1))

    trio.run(run)


class TestProbe:
    def test_an_answer_marks_the_server_live_and_teaches_the_servers_it_lists(self, monkeypatch, caplog):
        answer = wire.encode_peers([
            wire.PeerEntry(LISTED.encode('ascii'), ()),
            wire.PeerEntry(OWN.encode('ascii'), ()),
            wire.PeerEntry(b'not-an-onion', ()),
            wire.PeerEntry(PEER.encode('ascii'), ()),
            wire.PeerEntry(b'\xff\xfe', ()),
        ])
        node, opened = _prober(monkeypatch, answer)

        with caplog.at_level(logging.INFO):
            _probe(node)

        ((onion, io),) = opened
        assert onion == PEER
        assert io.written == [wire.encode_peers([]), wire.encode_end()]
        peer = node.book.peers[PEER]
        assert node.book.is_live(peer) and peer.failures == 0
        assert set(node.book.peers) == {PEER, LISTED}
        assert node.book.peers[LISTED].ok is None
        assert f'Reached AudioMuse-AI server {PEER} through Tor (1 servers answer)' in caplog.text
        assert '7003' in caplog.text and 'PEERS entry' in caplog.text

    def test_an_answer_of_another_kind_still_counts_as_an_answer(self, monkeypatch):
        node, _opened = _prober(monkeypatch, wire.encode_no(wire.NO_NOT_FOUND))

        _probe(node)

        assert node.book.is_live(node.book.peers[PEER])
        assert set(node.book.peers) == {PEER}

    def test_a_live_server_is_not_announced_again(self, monkeypatch, caplog):
        node, _opened = _prober(monkeypatch, wire.encode_peers([]))
        node.book.succeeded(PEER)

        with caplog.at_level(logging.INFO):
            _probe(node)

        assert 'Reached AudioMuse-AI server' not in caplog.text

    def test_a_server_tor_cannot_reach_is_marked_failed(self, monkeypatch, caplog):
        error = p2p_tor.TorConnectError(f'Tor could not reach {PEER}.onion: onion service unreachable')
        node, opened = _prober(monkeypatch, error=error)

        with caplog.at_level(logging.INFO):
            _probe(node)

        assert opened == []
        assert node.book.peers[PEER].failures == 1 and node.book.peers[PEER].ok is None
        assert f'AudioMuse-AI server {PEER} did not answer through Tor' in caplog.text
        assert 'onion service unreachable' in caplog.text

    def test_a_malformed_answer_is_logged_and_marked_failed(self, monkeypatch, caplog):
        node, _opened = _prober(monkeypatch, MALFORMED_PEERS)

        with caplog.at_level(logging.WARNING):
            _probe(node)

        assert node.book.peers[PEER].failures == 1
        assert '7003' in caplog.text and 'PEERS answer' in caplog.text

    def test_a_stream_closed_without_an_answer_is_marked_failed(self, monkeypatch):
        node, _opened = _prober(monkeypatch, b'')

        _probe(node)

        assert node.book.peers[PEER].failures == 1

    def test_no_answer_in_time_is_marked_failed(self, monkeypatch):
        monkeypatch.setattr(runtime, 'PROBE_SECONDS', 0.05)
        node, _opened = _prober(monkeypatch, hang=True)

        _probe(node)

        assert node.book.peers[PEER].failures == 1


class TestState:
    def test_starting_until_the_onion_service_is_up_then_the_peers_decide(self):
        node = runtime.Runtime(store=None)
        assert node.current_state() == ipc.STATE_STARTING
        node.book = peers.PeerBook(OWN, SPACES, 4)
        node.server = SimpleNamespace(state=None, size=0, counters={})
        assert node.current_state() == ipc.STATE_STARTING
        node.server.state = object()
        assert node.current_state() == ipc.STATE_STARTING
        node.onion = OWN
        assert node.current_state() == ipc.STATE_STARTING

        node.started_at -= runtime.STARTUP_GRACE_SECONDS + 1
        assert node.current_state() == ipc.STATE_NO_PEERS
        node.book.succeeded(PEER)
        assert node.current_state() == ipc.STATE_CONNECTED

    def test_a_live_server_means_connected_even_during_the_grace(self):
        node = runtime.Runtime(store=None)
        node.book = peers.PeerBook(OWN, SPACES, 4)
        node.server = SimpleNamespace(state=object(), size=0, counters={})
        node.onion = OWN
        node.book.succeeded(PEER)

        assert node.current_state() == ipc.STATE_CONNECTED

    def test_the_status_row_names_the_onion_address_and_the_counters(self):
        node = runtime.Runtime(store=None)
        node.book = peers.PeerBook(OWN, SPACES, 4)
        node.book.succeeded(PEER)
        node.book.learn(LISTED)
        node.server = SimpleNamespace(state=object(), size=7, counters={'queries': 2})
        node.limits = _limits()
        node.onion = OWN
        node.reason = 'running'

        status = node.status()

        assert status == {
            'state': ipc.STATE_CONNECTED, 'reason': 'running', 'peer_id': OWN, 'version': config.APP_VERSION,
            'peers': 1, 'known': 2, 'index': 7, 'served': {'queries': 2}, 'bytes': {},
        }
        assert 'main_server' not in status

    def test_a_fresh_runtime_reports_starting_with_no_address(self):
        status = runtime.Runtime(store=None).status()

        assert status == {'state': ipc.STATE_STARTING, 'reason': 'starting', 'peer_id': '',
                          'version': config.APP_VERSION}


def _come_online(monkeypatch, key=None, stored_key=None, cache=None):
    store = FakeStore(stored_key=stored_key)
    node = runtime.Runtime(store)
    node.key = key
    node.running_reason = 'running'
    node.book = peers.PeerBook('', SPACES, 4)
    node.limits = _limits()
    served = []
    reasons = []
    published = []
    hosts = []
    tor = SimpleNamespace(socks_port=9050)

    async def handle_stream(io, peer):
        served.append(peer)

    node.server = SimpleNamespace(handle_stream=handle_stream)

    async def start(binary, on_progress):
        on_progress(40)
        reasons.append(node.reason)
        return tor

    async def publish(tor_process, local_port, key_content):
        published.append((tor_process, local_port, key_content))
        reasons.append(node.reason)
        return OWN, key_content or 'NEWKEYCONTENT'

    original = trio.serve_tcp

    async def serve_tcp(handler, port, *, host=None, task_status=trio.TASK_STATUS_IGNORED, **kwargs):
        hosts.append(host)
        return await original(handler, port, host=host, task_status=task_status, **kwargs)

    monkeypatch.setattr(runtime.p2p_tor, 'start', start)
    monkeypatch.setattr(runtime.p2p_tor, 'publish', publish)
    monkeypatch.setattr(trio, 'serve_tcp', serve_tcp)

    async def run():
        async with trio.open_nursery() as nursery:
            node.nursery = nursery
            await node._come_online('/usr/bin/tor', cache or {})
            stream = await trio.open_tcp_stream('127.0.0.1', published[0][1])
            with trio.fail_after(5):
                while not served:
                    await trio.sleep(0.01)
            await stream.aclose()
            nursery.cancel_scope.cancel()

    trio.run(run)
    return SimpleNamespace(node=node, store=store, tor=tor, reasons=reasons, published=published, hosts=hosts,
                           served=served)


class TestComingOnline:
    def test_a_new_key_is_stored_once_and_the_local_port_is_published(self, monkeypatch):
        online = _come_online(monkeypatch)

        ((tor, port, key),) = online.published
        assert tor is online.tor and port > 0 and key is None
        assert online.hosts == ['127.0.0.1']
        assert online.store.saved_keys == ['NEWKEYCONTENT']
        assert online.node.key == 'NEWKEYCONTENT'
        assert (online.node.onion, online.node.book.own_onion) == (OWN, OWN)
        assert online.reasons == ['connecting to the Tor network (40%)', 'publishing the onion service']
        assert online.node.reason == 'running'
        assert online.served == [runtime.ANONYMOUS]

    def test_a_key_stored_meanwhile_wins_over_the_new_one(self, monkeypatch):
        online = _come_online(monkeypatch, stored_key='STOREDFIRST')

        assert online.store.saved_keys == ['NEWKEYCONTENT']
        assert online.node.key == 'STOREDFIRST'

    def test_a_stored_key_is_published_and_never_stored_again(self, monkeypatch):
        online = _come_online(monkeypatch, key=ONION_KEY)

        assert online.published[0][2] == ONION_KEY
        assert online.store.saved_keys == []
        assert online.node.key == ONION_KEY

    def test_the_saved_list_is_loaded_after_the_own_address_is_known(self, monkeypatch):
        now = int(time.time())
        cache = {'v': 1, 'peers': {PEER: {'heard': now, 'ok': now}, OWN: {'heard': now, 'ok': None}}}

        online = _come_online(monkeypatch, cache=cache)

        assert set(online.node.book.peers) == {PEER}


class TestShutdown:
    def test_the_list_and_the_stopped_status_are_written_and_tor_is_stopped(self):
        store = FakeStore()
        node = runtime.Runtime(store)
        node.book = peers.PeerBook(OWN, SPACES, 4)
        node.book.succeeded(PEER)
        node.onion = OWN
        closed = []
        node.tor = SimpleNamespace(close=lambda: closed.append(True))

        trio.run(node._shutdown)

        assert store.caches == [node.book.cache_payload()]
        assert store.statuses == [{'state': ipc.STATE_NOT_RUNNING, 'reason': 'stopped', 'peer_id': OWN}]
        assert closed == [True]

    def test_a_node_that_never_came_online_keeps_the_saved_list(self):
        store = FakeStore()
        node = runtime.Runtime(store)
        node.book = peers.PeerBook('', SPACES, 4)

        trio.run(node._shutdown)

        assert store.caches == []
        assert store.statuses == [{'state': ipc.STATE_NOT_RUNNING, 'reason': 'stopped', 'peer_id': ''}]

    def test_tor_is_stopped_even_when_the_database_write_fails(self, caplog):
        node = runtime.Runtime(FakeStore(status_error=RuntimeError('database gone')))
        closed = []
        node.tor = SimpleNamespace(close=lambda: closed.append(True))

        with caplog.at_level(logging.ERROR, logger='p2p.runtime'):
            trio.run(node._shutdown)

        assert closed == [True]
        assert 'Could not record the P2P node shutdown' in caplog.text


def _directory_node(max_peers=4):
    node = runtime.Runtime(store=None)
    node.key = ONION_KEY
    node.onion = OWN
    node.spaces = SPACES
    node.tor = SimpleNamespace(socks_port=9050)
    node.book = peers.PeerBook(OWN, SPACES, max_peers)
    return node


def _run_directory(node, seconds):
    async def run():
        node.probe_due = trio.Event()
        with trio.move_on_after(seconds):
            await node._directory_loop()
        return node.probe_due.is_set()

    return trio.run(run, clock=trio.testing.MockClock(autojump_threshold=0))


class TestDirectoryLoop:
    def _relays(self, monkeypatch, accepted=True, records=(), fetch_errors=()):
        published = []
        fetched = []
        errors = list(fetch_errors)
        monkeypatch.setattr(config, 'P2P_DIRECTORY_RELAYS', 'one.relay.org,two.relay.org')

        async def publish(socks_port, relays, event):
            published.append((socks_port, relays, event))
            return {relay: accepted for relay in relays}

        async def fetch(socks_port, relays, now=None):
            fetched.append((socks_port, relays))
            if errors:
                raise errors.pop(0)
            return list(records)

        monkeypatch.setattr(runtime.directory, 'publish', publish)
        monkeypatch.setattr(runtime.directory, 'fetch', fetch)
        return published, fetched

    def test_without_relays_the_directory_is_off(self, monkeypatch, caplog):
        monkeypatch.setattr(config, 'P2P_DIRECTORY_RELAYS', '')
        node = _directory_node()

        with caplog.at_level(logging.INFO, logger='p2p.runtime'):
            assert _run_directory(node, 1.0) is False

        assert 'P2P directory off' in caplog.text

    def test_the_record_is_published_once_and_read_every_ten_minutes(self, monkeypatch):
        record = directory.Record(PEER, 'v3.7.0', SPACES, 1234)
        published, fetched = self._relays(monkeypatch, records=[record])
        node = _directory_node()

        woke = _run_directory(node, runtime.READ_FAST_SECONDS * 2 + 1)

        assert woke
        assert len(published) == 1 and len(fetched) == 3
        socks_port, relays, event = published[0]
        assert (socks_port, relays) == (9050, ['one.relay.org', 'two.relay.org'])
        assert json.loads(event['content'])['onion'] == OWN
        assert event['pubkey'] == directory.signing_key(ONION_KEY).public_key_xonly.format().hex()
        learned = node.book.peers[PEER]
        assert (learned.spaces, learned.version, learned.heard) == (SPACES, 'v3.7.0', 1234)

    def test_an_unaccepted_record_is_published_again_next_round(self, monkeypatch, caplog):
        published, fetched = self._relays(monkeypatch, accepted=False)
        node = _directory_node()

        with caplog.at_level(logging.WARNING, logger='p2p.runtime'):
            _run_directory(node, runtime.READ_FAST_SECONDS + 1)

        assert len(published) == 2 and len(fetched) == 2
        assert "No directory relay accepted this server's record" in caplog.text

    def test_with_enough_servers_answering_it_reads_hourly(self, monkeypatch):
        published, fetched = self._relays(monkeypatch)
        node = _directory_node(max_peers=1)
        node.book.succeeded(ANOTHER)

        _run_directory(node, runtime.READ_FAST_SECONDS * 2 + 1)

        assert len(fetched) == 1

    def test_a_failing_round_is_logged_and_the_next_one_runs(self, monkeypatch, caplog):
        published, fetched = self._relays(monkeypatch, fetch_errors=[RuntimeError('relays down')])
        node = _directory_node()

        with caplog.at_level(logging.ERROR, logger='p2p.runtime'):
            _run_directory(node, runtime.READ_FAST_SECONDS + 1)

        assert len(fetched) == 2
        assert 'A P2P directory round failed' in caplog.text
