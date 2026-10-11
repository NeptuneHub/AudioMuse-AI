# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""What the P2P node serves, and how its requesting side takes it (p2p.serve, p2p.fetch).

The server offers every stored result whose version is v3.0.0 or newer, computed
here or received from a peer; the requester confirms the song before it
fetches and takes each result as it is once it decodes. Both halves run here
against stub stores over in-memory trio streams, with no database and no
network.

Main Features:
* an unstamped result is offered with this install's APP_VERSION, a stamped one
  with its stamp; a v2.x stamp is never offered; a result received from a peer
  (a "p" in its stamp) is served with the version that peer stamped
* the lyrics sentinel is never served; a neural fingerprint only with the
  local codebook (through Tor like every result); nothing else is checked
  beyond decoding the bytes
* tokens are one-time, bound to the peer string the runtime passes (every
  inbound Tor stream is the anonymous member) and expire after 60 s; the
  QUERY budget shared by every caller and full upload slots answer BUSY; a
  mismatched model space NO
* the index is built from pages, caught up by created_at into a delta index,
  and answers NO until it exists; an unknown wire version gets NO; PEERS lists
  the onion addresses the peer book reached recently (none without a book)
* a malformed frame only ends that stream (logged 7003) on either side; the
  next stream of the same server is served and asked again
* the requester confirms offers (a +5 s decoy and a different song are
  refused, a missing peer chromaprint too when the gate is on), picks among
  them in random order and drops unasked, old, foreign-space or undecodable
  results
* the Fetcher answers the worker with exactly the served outputs and their
  versions, merges a duplicate request and drops an expired one; every QUERY
  tells the peer book whether that server answered, and a server Tor cannot
  reach is counted in the lookup summary
"""

import datetime
import logging
import random
import time
from contextlib import asynccontextmanager

import numpy as np
import pytest
import trio
import trio.testing

import config
from p2p import fetch, ipc, limits, runtime, serve, store as p2p_store, wire
from p2p import tor as p2p_tor
from tasks import nfp_format, simhash

SPACES = {key: bytes([index + 1]) * 32 for index, key in enumerate(wire.SPACE_KEYS)}
CODEBOOK = 0x0BADC0DE
DURATION = 200.0
PEER_A = 'peera' + 'a' * 51
PEER_B = 'peerb' + 'b' * 51
LISTED = ('listedone' + 'c' * 47, 'listedtwo' + 'd' * 47)
BASE_TIME = datetime.datetime(2026, 9, 1, 12, 0, 0)
MALFORMED_QUERY = bytes([wire.WIRE_VERSION, wire.T_QUERY, 3]) + b'\x00\x00\x00'
MALFORMED_OFFER = bytes([wire.WIRE_VERSION, wire.T_OFFER, 1, 0])


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _unit(dim, seed):
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _axis(seed):
    rng = np.random.default_rng(seed)
    parts = [rng.random(size).astype(np.float32) for size in (6, 6, 6, 5, 4)]
    return np.concatenate([part / part.sum() for part in parts]).astype(np.float32)


def _musicnn(seed):
    return (np.random.default_rng(seed).random(200) * 0.5).astype(np.float32)


def _neural(duration=DURATION, book=CODEBOOK):
    count = int(duration * 2)
    header = nfp_format.HEADER.pack(nfp_format.BLOB_MAGIC, nfp_format.DIM, count, nfp_format.PQ_SUBSPACES, book)
    return header + bytes(range(256)) * (count * nfp_format.CODE_BYTES // 256) + bytes(
        count * nfp_format.CODE_BYTES % 256)


def _local(version='v3.6.3'):
    return {'v': version}


def _received(version='v3.4.1', peer=PEER_B):
    return {'v': version, 'p': peer}


def _row(seed=0, origin=None, duration=DURATION, clap=True, lyrics=True, neural=True,
         lyrics_vector=None, neural_blob=None, chromaprint=None, created=0):
    embedding = _musicnn(seed)
    item_id = simhash.canonical_id_str(simhash.embedding_signature(embedding))
    if origin is None:
        origin = {'m': _local(), 'c': _local(), 'l': _local('v3.5.0'), 'n': _local()}
    row = p2p_store.ServeRow(
        item_id, duration, embedding.tobytes(), origin,
        (neural_blob or _neural(duration)) if neural else None,
        _unit(512, seed + 1).tobytes() if clap else None,
        (lyrics_vector if lyrics_vector is not None else _unit(768, seed + 2)).tobytes() if lyrics else None,
        _axis(seed + 3).tobytes() if lyrics else None,
        chromaprint,
    )
    return row, BASE_TIME + datetime.timedelta(seconds=created)


class StubStore:
    def __init__(self, rows=(), local=None, delay=0.0):
        self.rows = {}
        self.created = {}
        for row, created in rows:
            self.add(row, created)
        self.local = local
        self.delay = delay
        self.serve_calls = 0

    def add(self, row, created):
        self.rows[row.item_id] = row
        self.created[row.item_id] = created

    def _shareable(self, item_id):
        row = self.rows[item_id]
        return p2p_store.ShareableRow(item_id, row.duration, self.created[item_id], row.origin.get('m'))

    def shareable_page(self, after, limit=2):
        ids = sorted(item_id for item_id in self.rows if item_id > (after or ''))
        return [self._shareable(item_id) for item_id in ids[:limit]]

    def shareable_since(self, created_after, after_item_id='', limit=2):
        keys = sorted((self.created[item_id], item_id) for item_id in self.rows)
        keys = [key for key in keys if key > (created_after, after_item_id or '')]
        return [self._shareable(item_id) for _created, item_id in keys[:limit]]

    def serve_rows(self, item_ids):
        self.serve_calls += 1
        return {item_id: self.rows[item_id] for item_id in item_ids if item_id in self.rows}

    def local_track(self, item_id, server_id, provider_id):
        time.sleep(self.delay)
        return self.local


def _limits(clock=None, parallel=2):
    return limits.Limits(upload_kibps=0, download_kibps=0, parallel_uploads=parallel,
                         parallel_downloads=parallel, clock=clock)


def _server(rows, clock=None, spaces=SPACES, codebook=CODEBOOK, parallel=2, peers_provider=None):
    clock = clock or Clock()
    return serve.Server(StubStore(rows), spaces, codebook, _limits(clock, parallel),
                        peers_provider=peers_provider, clock=clock)


def _query(row, wants=('c', 'l', 'n'), spaces=SPACES, duration=None):
    signature = simhash.signature_to_bytes(simhash.embedding_signature(np.frombuffer(row.embedding, np.float32)))
    return wire.Query(signature, duration or row.duration, tuple(wants), {k: spaces[k] for k in ('m',) + tuple(wants)})


def _query_frame(seed=0, wants=('c', 'l')):
    signature = simhash.signature_to_bytes(simhash.embedding_signature(_musicnn(seed)))
    return wire.encode_query(signature, DURATION, list(wants), SPACES)


def _decode(frame):
    return trio.run(wire.read_frame, _Buffer(frame))


async def _adecode(frame):
    return await wire.read_frame(_Buffer(frame))


class _Buffer:
    def __init__(self, data):
        self.data = bytearray(data)

    async def read_exactly(self, size):
        chunk = bytes(self.data[:size])
        del self.data[:size]
        return chunk


async def _built(server):
    await server.build_index()
    return server


def _ask(server, query, peer=PEER_A):
    async def run():
        await _built(server)
        return await server.answer_query(peer, query)
    return _decode(trio.run(run))


def _fetch_all(server, row, wants=('c', 'l', 'n')):
    async def run():
        await _built(server)
        offer = (await _adecode(await server.answer_query(PEER_A, _query(row, wants))))[1]
        fetch_frame = wire.Fetch(offer.candidates[0].token, tuple(wants), {k: SPACES[k] for k in wants})
        return offer, await _adecode(await server.answer_fetch(PEER_A, fetch_frame))
    return trio.run(run)


class TestVersionRules:
    def test_an_unstamped_row_is_offered_with_this_install_version(self, monkeypatch):
        monkeypatch.setattr(config, 'APP_VERSION', 'v3.7.2')
        row, created = _row(origin={})
        server = _server([(row, created)])

        frame_type, message = _ask(server, _query(row))

        assert server.size == 1
        assert frame_type == wire.T_OFFER
        assert message.candidates[0].versions == {'m': 'v3.7.2', 'c': 'v3.7.2', 'l': 'v3.7.2', 'n': 'v3.7.2'}

    def test_an_unstamped_output_next_to_stamped_ones_takes_the_install_version(self, monkeypatch):
        monkeypatch.setattr(config, 'APP_VERSION', 'v3.7.2')
        row, created = _row(origin={'m': _local('v3.1.0'), 'c': _local('v3.2.0')})

        offer, (frame_type, message) = _fetch_all(_server([(row, created)]), row)

        assert offer.candidates[0].versions == {'m': 'v3.1.0', 'c': 'v3.2.0', 'l': 'v3.7.2', 'n': 'v3.7.2'}
        assert frame_type == wire.T_OUTPUTS
        assert {key: item.version for key, item in message.items.items()} == {
            'c': 'v3.2.0', 'l': 'v3.7.2', 'n': 'v3.7.2'}

    def test_a_v2_stamp_is_never_offered(self):
        old = {'m': _local('v2.9.0'), 'c': _local('v2.9.0')}
        row, created = _row(origin=old)
        server = _server([(row, created)])

        frame_type, _message = _ask(server, _query(row))

        assert server.size == 0
        assert frame_type == wire.T_NO

    def test_an_old_output_is_left_out_of_a_current_row(self):
        row, created = _row(origin={'m': _local(), 'c': _local('v2.6.2'), 'l': _local(), 'n': _local()})
        frame_type, message = _ask(_server([(row, created)]), _query(row))

        assert frame_type == wire.T_OFFER
        assert sorted(message.candidates[0].versions) == ['l', 'm', 'n']

    def test_received_outputs_are_served_with_their_stamp_version(self):
        origin = {'m': _received('v3.3.0'), 'c': _received('v3.4.1'), 'l': _received('v3.0.0'), 'n': _local()}
        row, created = _row(origin=origin)

        offer, (frame_type, message) = _fetch_all(_server([(row, created)]), row)

        assert offer.candidates[0].versions == {'m': 'v3.3.0', 'c': 'v3.4.1', 'l': 'v3.0.0', 'n': 'v3.6.3'}
        assert frame_type == wire.T_OUTPUTS
        assert {key: item.version for key, item in message.items.items()} == {
            'c': 'v3.4.1', 'l': 'v3.0.0', 'n': 'v3.6.3'}
        assert np.array_equal(message.items['c'].value, np.frombuffer(row.clap, np.float32))

    def test_a_received_v2_output_is_never_served(self):
        row, created = _row(origin={'m': _local(), 'c': _received('v2.9.0')}, lyrics=False, neural=False)

        frame_type, message = _ask(_server([(row, created)]), _query(row, wants=('c',)))

        assert (frame_type, message) == (wire.T_NO, wire.No(wire.NO_OUTPUTS))

    def test_a_received_musicnn_row_is_shareable(self):
        row, _created = _row(origin={'m': _received('v3.6.3')})
        shareable = p2p_store.ShareableRow(row.item_id, row.duration, BASE_TIME, row.origin['m'])

        assert serve.shareable(shareable)
        assert serve.shareable(shareable._replace(musicnn_origin=None))
        assert not serve.shareable(shareable._replace(musicnn_origin=_local('v2.9.0')))

    @pytest.mark.parametrize('stamp', ['v3.6.3', {'p': PEER_B}, {'v': 'three'}, {'v': None}])
    def test_an_unreadable_stamp_is_never_served(self, stamp):
        row, _created = _row(origin={'m': _local(), 'c': stamp})

        assert 'c' not in serve.servable_outputs(row, ('c',), SPACES, CODEBOOK)

    def test_the_lyrics_sentinel_is_never_served(self):
        sentinel = np.asarray(config.LYRICS_INSTRUMENTAL_EMBEDDING, dtype=np.float32)
        row, created = _row(lyrics_vector=sentinel)

        assert 'l' not in serve.servable_outputs(row, ('c', 'l', 'n'), SPACES, CODEBOOK)
        frame_type, message = _ask(_server([(row, created)]), _query(row))
        assert frame_type == wire.T_OFFER and 'l' not in message.candidates[0].versions

    def test_a_received_lyrics_sentinel_is_never_served_either(self):
        sentinel = np.asarray(config.LYRICS_INSTRUMENTAL_EMBEDDING, dtype=np.float32)
        row, _created = _row(origin={'m': _local(), 'l': _received()}, lyrics_vector=sentinel)

        assert serve.servable_outputs(row, ('l',), SPACES, CODEBOOK) == {}

    def test_a_neural_fingerprint_needs_the_local_codebook(self):
        row, _created = _row(neural_blob=_neural(book=CODEBOOK + 1))

        assert 'n' not in serve.servable_outputs(row, ('n',), SPACES, CODEBOOK)
        assert 'n' not in serve.servable_outputs(_row()[0], ('n',), SPACES, None)
        assert 'n' in serve.servable_outputs(_row()[0], ('n',), SPACES, CODEBOOK)

    def test_an_output_without_a_local_space_is_never_served(self):
        spaces = dict(SPACES, c=None)

        assert 'c' not in serve.servable_outputs(_row()[0], ('c', 'l'), spaces, CODEBOOK)

    def test_an_undecodable_clap_vector_is_never_served(self):
        row, _created = _row()
        not_finite = np.full(512, np.nan, dtype=np.float32).tobytes()

        assert 'c' not in serve.servable_outputs(row._replace(clap=not_finite), ('c',), SPACES, CODEBOOK)
        assert 'c' not in serve.servable_outputs(row._replace(clap=b'\x00' * 100), ('c',), SPACES, CODEBOOK)

    def test_a_stored_result_is_served_as_it_is(self):
        row, _created = _row()
        loose = row._replace(
            clap=(np.ones(512, dtype=np.float32) * 3).tobytes(),
            axis=np.ones(27, dtype=np.float32).tobytes(),
            neural=_neural(duration=DURATION * 4),
        )

        served = serve.servable_outputs(loose, ('c', 'l', 'n'), SPACES, CODEBOOK)

        assert sorted(served) == ['c', 'l', 'n']
        assert np.array_equal(served['c'][1], np.ones(512, dtype=np.float32) * 3)


class TestQueryAnswers:
    def test_no_while_the_index_is_not_built(self):
        row, created = _row()
        server = _server([(row, created)])

        frame = trio.run(server.answer_query, PEER_A, _query(row))

        assert _decode(frame) == (wire.T_NO, wire.No(wire.NO_BUILDING))

    def test_a_foreign_musicnn_space_gets_no(self):
        row, created = _row()
        foreign = dict(SPACES, m=b'\xee' * 32)

        assert _ask(_server([(row, created)]), _query(row, spaces=foreign)) == (wire.T_NO, wire.No(wire.NO_SPACE))

    def test_a_foreign_output_space_narrows_the_offer(self):
        row, created = _row()
        foreign = dict(SPACES, c=b'\xee' * 32)

        frame_type, message = _ask(_server([(row, created)]), _query(row, spaces=foreign))

        assert frame_type == wire.T_OFFER
        assert 'c' not in message.candidates[0].versions

    def test_a_duration_outside_the_tolerance_finds_nothing(self):
        row, created = _row()

        frame_type, _message = _ask(_server([(row, created)]), _query(row, duration=DURATION + 5.0))

        assert frame_type == wire.T_NO

    def test_the_offer_carries_the_embedding_duration_and_chromaprint(self):
        row, created = _row(chromaprint=b'\x78\x9c\x01')
        frame_type, message = _ask(_server([(row, created)]), _query(row))
        candidate = message.candidates[0]

        assert np.array_equal(candidate.embedding, np.frombuffer(row.embedding, np.float32))
        assert candidate.duration == DURATION
        assert candidate.chromaprint == b'\x78\x9c\x01'
        assert len(candidate.token) == wire.TOKEN_BYTES

    def test_the_rate_limit_answers_busy(self):
        row, created = _row()
        clock = Clock()
        server = _server([(row, created)], clock=clock)
        server._limits.queries = limits.QueryRateLimiter(rate=1, clock=clock)

        async def run():
            await _built(server)
            return [await server.answer_query(PEER_A, _query(row)) for _ in range(4)]

        types = [_decode(frame)[0] for frame in trio.run(run)]

        assert types == [wire.T_OFFER, wire.T_OFFER, wire.T_BUSY, wire.T_BUSY]
        assert server.counters['busy'] == 2

    def test_the_query_budget_is_shared_by_every_caller(self):
        row, created = _row()
        clock = Clock()
        server = _server([(row, created)], clock=clock)
        server._limits.queries = limits.QueryRateLimiter(rate=1, clock=clock)

        async def run():
            await _built(server)
            answers = [await server.answer_query(peer, _query(row)) for peer in (PEER_A, PEER_B, PEER_B)]
            clock.now += 1.0
            answers.append(await server.answer_query(PEER_B, _query(row)))
            return answers

        types = [_decode(frame)[0] for frame in trio.run(run)]

        assert types == [wire.T_OFFER, wire.T_OFFER, wire.T_BUSY, wire.T_OFFER]


class TestTokens:
    def _offer_then_fetch(self, server, row, fetches, wants=('c', 'l', 'n')):
        async def run():
            await _built(server)
            offer = (await _adecode(await server.answer_query(PEER_A, _query(row, wants))))[1]
            token = offer.candidates[0].token
            answers = []
            for peer, advance in fetches:
                server._clock.now += advance
                fetch_frame = wire.Fetch(token, tuple(wants), {k: SPACES[k] for k in wants})
                answers.append(await _adecode(await server.answer_fetch(peer, fetch_frame)))
            return answers
        return trio.run(run)

    def test_a_token_is_good_once(self):
        row, created = _row()
        first, second = self._offer_then_fetch(_server([(row, created)]), row, [(PEER_A, 0), (PEER_A, 0)])

        assert first[0] == wire.T_OUTPUTS
        assert sorted(first[1].items) == ['c', 'l', 'n']
        assert second == (wire.T_NO, wire.No(wire.NO_TOKEN))

    def test_a_token_expires_after_60_seconds(self):
        row, created = _row()

        (answer,) = self._offer_then_fetch(_server([(row, created)]), row, [(PEER_A, 61.0)])

        assert answer == (wire.T_NO, wire.No(wire.NO_TOKEN))

    def test_a_token_is_bound_to_the_peer_that_asked(self):
        row, created = _row()

        (answer,) = self._offer_then_fetch(_server([(row, created)]), row, [(PEER_B, 0)])

        assert answer == (wire.T_NO, wire.No(wire.NO_TOKEN))

    def test_the_outputs_are_the_stored_values_with_their_versions(self):
        row, created = _row()
        ((frame_type, message),) = self._offer_then_fetch(_server([(row, created)]), row, [(PEER_A, 0)])

        assert frame_type == wire.T_OUTPUTS
        assert message.items['c'].version == 'v3.6.3'
        assert message.items['l'].version == 'v3.5.0'
        assert message.items['c'].space == SPACES['c']
        assert np.array_equal(message.items['c'].value, np.frombuffer(row.clap, np.float32))
        assert np.array_equal(message.items['l'].value[1], np.frombuffer(row.axis, np.float32))
        assert message.items['n'].value == row.neural

    def test_a_row_that_changed_since_the_offer_is_re_checked(self):
        row, created = _row()
        server = _server([(row, created)])

        async def run():
            await _built(server)
            offer = (await _adecode(await server.answer_query(PEER_A, _query(row))))[1]
            server._store.rows[row.item_id] = row._replace(origin={'m': _local(), 'c': _local('v2.9.0')})
            fetch_frame = wire.Fetch(offer.candidates[0].token, ('c',), {'c': SPACES['c']})
            return await _adecode(await server.answer_fetch(PEER_A, fetch_frame))

        assert trio.run(run) == (wire.T_NO, wire.No(wire.NO_OUTPUTS))


class TestIndex:
    def test_the_index_is_built_across_pages(self):
        rows = [_row(seed) for seed in range(5)]
        server = _server(rows)

        trio.run(server.build_index)

        assert server.size == 5

    def test_catch_up_adds_new_rows_to_the_delta_index(self):
        server = _server([_row(0, created=0), _row(1, created=10)])
        late, created = _row(7, created=500)

        async def run():
            await server.build_index()
            before = server.size
            server._store.add(late, created)
            added = await server.catch_up()
            again = await server.catch_up()
            return before, added, again

        before, added, again = trio.run(run)

        assert (before, added, again) == (2, 1, 0)
        assert server.size == 3
        assert [item for item, _ in server.find(simhash.embedding_signature(_musicnn(7)), DURATION)] == [late.item_id]

    def test_an_unshareable_late_row_is_retried_while_in_the_overlap(self):
        server = _server([_row(0, created=0)])
        pending, created = _row(8, origin={'m': _local('v2.9.0')}, created=30)

        async def run():
            await server.build_index()
            server._store.add(pending, created)
            first = await server.catch_up()
            server._store.rows[pending.item_id] = pending._replace(origin={'m': _local()})
            second = await server.catch_up()
            return first, second

        assert trio.run(run) == (0, 1)


class _Pipe:
    def __init__(self, stream):
        self.stream = stream

    async def read(self, size):
        return await self.stream.receive_some(size)

    async def write(self, data):
        await self.stream.send_all(data)


class _FakeTorStream:
    def __init__(self, data):
        self.data = bytearray(data)
        self.written = bytearray()
        self.closed = False

    async def receive_some(self, max_bytes=None):
        size = len(self.data) if max_bytes is None else max_bytes
        chunk = bytes(self.data[:size])
        del self.data[:size]
        return chunk

    async def send_all(self, data):
        self.written += data

    async def aclose(self):
        self.closed = True


class TestStreams:
    def _exchange(self, server, frames):
        async def run():
            await _built(server)
            client, remote = trio.testing.memory_stream_pair()
            bucket = limits.TokenBucket(0)
            client_io = limits.MeteredStream(_Pipe(client), bucket, bucket)
            remote_io = limits.MeteredStream(_Pipe(remote), bucket, bucket)
            answers = []
            async with trio.open_nursery() as nursery:
                nursery.start_soon(server.handle_stream, remote_io, PEER_A)
                for frame in frames:
                    payload = frame(answers) if callable(frame) else frame
                    await client_io.write(payload)
                    answers.append(await wire.read_frame(client_io))
                await client_io.write(wire.encode_end())
            return answers
        return trio.run(run)

    def test_query_then_fetch_on_one_stream(self):
        row, created = _row()

        def fetch_frame(answers):
            token = answers[0][1].candidates[0].token
            return wire.encode_fetch(token, ['c', 'l'], SPACES)

        answers = self._exchange(_server([(row, created)]), [_query_frame(), fetch_frame])

        assert [frame_type for frame_type, _ in answers] == [wire.T_OFFER, wire.T_OUTPUTS]
        assert sorted(answers[1][1].items) == ['c', 'l']

    def test_an_unknown_wire_version_gets_no(self):
        row, created = _row()

        answers = self._exchange(_server([(row, created)]), [bytes([wire.WIRE_VERSION + 1, wire.T_QUERY, 0])])

        assert answers == [(wire.T_NO, wire.No(wire.NO_VERSION))]

    def test_peers_lists_the_onion_addresses_the_book_reached(self):
        row, created = _row()
        asked = [wire.PeerEntry(PEER_B.encode('ascii'), ())]
        server = _server([(row, created)], peers_provider=lambda: list(LISTED))

        answers = self._exchange(server, [wire.encode_peers(asked), _query_frame()])

        assert answers[0] == (wire.T_PEERS, wire.Peers(tuple(
            wire.PeerEntry(onion.encode('ascii'), ()) for onion in LISTED
        )))
        assert answers[1][0] == wire.T_OFFER

    def test_without_a_peer_book_peers_is_an_empty_list(self):
        row, created = _row()

        answers = self._exchange(_server([(row, created)]), [wire.encode_peers([])])

        assert answers == [(wire.T_PEERS, wire.Peers(()))]
        assert _server([(row, created)]).answer_peers() == wire.encode_peers([])

    def test_full_upload_slots_answer_busy(self):
        row, created = _row()
        server = _server([(row, created)], parallel=1)

        async def run():
            await _built(server)
            server._limits.uploads.acquire_on_behalf_of_nowait('someone-else')
            client, remote = trio.testing.memory_stream_pair()
            bucket = limits.TokenBucket(0)
            client_io = limits.MeteredStream(_Pipe(client), bucket, bucket)
            async with trio.open_nursery() as nursery:
                nursery.start_soon(server.handle_stream, limits.MeteredStream(_Pipe(remote), bucket, bucket), PEER_A)
                await client_io.write(wire.encode_fetch(b'\x01' * 16, ['c'], SPACES))
                answer = await wire.read_frame(client_io)
                await client_io.write(wire.encode_end())
            return answer

        assert trio.run(run) == (wire.T_BUSY, wire.Busy())

    def test_a_malformed_frame_raises_a_violation(self):
        row, created = _row()
        server = _server([(row, created)])

        async def run():
            await _built(server)
            client, remote = trio.testing.memory_stream_pair()
            bucket = limits.TokenBucket(0)
            await client.send_all(MALFORMED_QUERY)
            await server.handle_stream(limits.MeteredStream(_Pipe(remote), bucket, bucket), PEER_A)

        with pytest.raises(wire.Violation):
            trio.run(run)

    def test_a_malformed_request_only_ends_that_stream(self, caplog):
        row, created = _row()
        server = _server([(row, created)])
        node = runtime.Runtime(store=None)
        node.server = server
        node.limits = server._limits
        bad = _FakeTorStream(MALFORMED_QUERY)
        good = _FakeTorStream(_query_frame() + wire.encode_end())

        async def run():
            await _built(server)
            await node._on_connection(bad)
            await node._on_connection(good)

        with caplog.at_level(logging.WARNING):
            trio.run(run)

        assert bad.closed and not bad.written
        assert '7003' in caplog.text and 'malformed request' in caplog.text
        assert good.closed
        assert _decode(bytes(good.written))[0] == wire.T_OFFER
        assert server.counters['queries'] == 1

    def test_an_inbound_stream_is_served_as_the_anonymous_member(self):
        row, created = _row()
        server = _server([(row, created)])
        node = runtime.Runtime(store=None)
        node.server = server
        node.limits = server._limits
        first = _FakeTorStream(_query_frame() + wire.encode_end())

        async def run():
            await _built(server)
            await node._on_connection(first)
            token = (await _adecode(bytes(first.written)))[1].candidates[0].token
            second = _FakeTorStream(wire.encode_fetch(token, ['c'], SPACES) + wire.encode_end())
            await node._on_connection(second)
            return second

        second = trio.run(run)

        assert runtime.ANONYMOUS == 'member'
        assert _decode(bytes(second.written))[0] == wire.T_OUTPUTS
        assert node.limits.counters['bytes_in'] > 0 and node.limits.counters['bytes_out'] > 0


class TestRequesterRules:
    def _candidate(self, seed=0, duration=DURATION, chromaprint=b'', versions=None, embedding=None):
        return wire.Candidate(b'\x01' * 16, _musicnn(seed) if embedding is None else embedding, duration,
                              versions or {'m': 'v3.6.3', 'c': 'v3.6.3', 'l': 'v3.0.0', 'n': 'v3.6.3'},
                              chromaprint)

    def _local(self, seed=0, noise=0.0, duration=DURATION + 0.4, chromaprint=None):
        embedding = _musicnn(seed) + np.random.default_rng(99).standard_normal(200).astype(np.float32) * noise
        return p2p_store.LocalTrack(embedding.astype(np.float32), duration, chromaprint)

    def test_a_perturbed_copy_is_confirmed(self):
        assert fetch.confirm(self._local(noise=1e-3), self._candidate())

    def test_a_five_second_decoy_is_refused(self):
        assert not fetch.confirm(self._local(), self._candidate(duration=DURATION + 5.0))

    def test_a_different_song_is_refused(self):
        assert not fetch.confirm(self._local(seed=0), self._candidate(seed=1))

    def test_an_old_musicnn_version_is_refused(self):
        assert not fetch.confirm(self._local(), self._candidate(versions={'m': 'v2.9.0', 'c': 'v3.6.3'}))

    def test_a_missing_peer_chromaprint_is_refused_when_the_gate_is_on(self, monkeypatch):
        monkeypatch.setattr(config, 'CHROMAPRINT_GATE_ENABLED', True)
        local = self._local(chromaprint=b'local-print')

        assert not fetch.confirm(local, self._candidate(chromaprint=b''))

    def test_a_missing_peer_chromaprint_is_fine_when_the_gate_is_off(self, monkeypatch):
        monkeypatch.setattr(config, 'CHROMAPRINT_GATE_ENABLED', False)
        local = self._local(chromaprint=b'local-print')

        assert fetch.confirm(local, self._candidate(chromaprint=b''))

    def test_offered_keys_drop_old_versions(self):
        candidate = self._candidate(versions={'m': 'v3.6.3', 'c': 'v2.0.0', 'l': 'v3.1.0'})

        assert fetch.offered_keys(candidate, ('c', 'l', 'n')) == ('l',)

    def test_the_pick_is_a_plain_shuffle_of_every_offer(self):
        offers = [fetch.Offered(f'peer{index}', None, ('c',)) for index in range(8)]
        expected = list(offers)
        random.Random(7).shuffle(expected)

        picked = fetch.pick_order(offers, random.Random(7))

        assert picked == expected
        assert sorted(offer.peer for offer in picked) == sorted(offer.peer for offer in offers)
        assert [offer.peer for offer in offers] == [f'peer{index}' for index in range(8)]

    def test_every_offer_can_come_first(self):
        offers = [fetch.Offered(f'peer{index}', None, ('c',)) for index in range(3)]

        firsts = {fetch.pick_order(offers, random.Random(seed))[0].peer for seed in range(40)}

        assert firsts == {'peer0', 'peer1', 'peer2'}

    def _outputs(self, **overrides):
        items = {
            'c': wire.Output(SPACES['c'], 'v3.6.3', _unit(512, 1)),
            'l': wire.Output(SPACES['l'], 'v3.5.0', (_unit(768, 2), _axis(3))),
            'n': wire.Output(SPACES['n'], 'v3.6.3', _neural()),
        }
        items.update(overrides)
        return wire.Outputs(items)

    def _check(self, outputs, keys=('c', 'l', 'n')):
        accepted, rejected = fetch.check_outputs(outputs, keys, SPACES, CODEBOOK)
        return sorted(accepted), sorted(key for key, _reason in rejected)

    def test_good_outputs_are_accepted(self):
        assert self._check(self._outputs()) == (['c', 'l', 'n'], [])

    def test_an_old_version_is_dropped(self):
        outputs = self._outputs(c=wire.Output(SPACES['c'], 'v2.6.2', _unit(512, 1)))

        assert self._check(outputs) == (['l', 'n'], ['c'])

    def test_a_foreign_space_is_dropped(self):
        outputs = self._outputs(l=wire.Output(b'\xee' * 32, 'v3.5.0', (_unit(768, 2), _axis(3))))

        assert self._check(outputs) == (['c', 'n'], ['l'])

    def test_an_unasked_output_is_dropped_and_the_rest_kept(self):
        assert self._check(self._outputs(), keys=('c',)) == (['c'], ['l', 'n'])

    def test_an_undecodable_output_is_dropped_and_the_rest_kept(self):
        broken = np.full(512, np.nan, dtype=np.float32)
        outputs = self._outputs(c=wire.Output(SPACES['c'], 'v3.6.3', broken),
                                n=wire.Output(SPACES['n'], 'v3.6.3', _neural(book=CODEBOOK + 1)))

        assert self._check(outputs) == (['l'], ['c', 'n'])

    def test_a_result_is_taken_as_it_is(self):
        loose_axis = np.ones(27, dtype=np.float32)
        outputs = self._outputs(c=wire.Output(SPACES['c'], 'v3.6.3', np.ones(512, dtype=np.float32) * 3),
                                l=wire.Output(SPACES['l'], 'v3.5.0', (_unit(768, 2) * 2, loose_axis)),
                                n=wire.Output(SPACES['n'], 'v3.6.3', _neural(duration=DURATION * 4)))

        assert self._check(outputs) == (['c', 'l', 'n'], [])

    def test_the_lyrics_sentinel_is_dropped(self):
        sentinel = np.asarray(config.LYRICS_INSTRUMENTAL_EMBEDDING, dtype=np.float32)
        outputs = self._outputs(l=wire.Output(SPACES['l'], 'v3.5.0', (sentinel, _axis(3))))

        assert self._check(outputs) == (['c', 'n'], ['l'])


class Book:
    def __init__(self, peers):
        self.peers = peers
        self.asked = []
        self.succeeded_peers = []
        self.failed_peers = []

    def candidates(self, wants):
        self.asked.append(tuple(wants))
        return [(peer, tuple(wants)) for peer in self.peers]

    def succeeded(self, peer):
        self.succeeded_peers.append(peer)

    def failed(self, peer):
        self.failed_peers.append(peer)


class _Unreachable:
    async def build_index(self):
        return None


class _Garbled:
    def __init__(self, server, bad_streams=1):
        self.server = server
        self.bad_streams = bad_streams

    async def build_index(self):
        await self.server.build_index()

    async def handle_stream(self, io, peer):
        if self.bad_streams:
            self.bad_streams -= 1
            await wire.read_frame(io)
            await io.write(MALFORMED_OFFER)
            return
        await self.server.handle_stream(io, peer)


class TestFetcher:
    def _run(self, servers, local, requests, delay=0.0, sequential=False):
        replies = []

        async def run():
            async with trio.open_nursery() as nursery:
                for server in servers.values():
                    await server.build_index()

                @asynccontextmanager
                async def opener(peer):
                    if isinstance(servers[peer], _Unreachable):
                        raise p2p_tor.TorConnectError(f'Tor could not reach {peer}.onion: onion service unreachable')
                    client, remote = trio.testing.memory_stream_pair()
                    bucket = limits.TokenBucket(0)
                    nursery.start_soon(servers[peer].handle_stream,
                                       limits.MeteredStream(_Pipe(remote), bucket, bucket), 'requester')
                    yield limits.MeteredStream(_Pipe(client), bucket, bucket)

                async def send_reply(channel, reply):
                    parts = ipc.split_reply(reply)
                    assembler = ipc.PartAssembler()
                    assembler.expect(reply['rid'])
                    done = [assembler.add(ipc.parse(part)) for part in parts][-1]
                    replies.append((channel, done))

                self.book = Book(list(servers))
                fetcher = fetch.Fetcher(StubStore(local=local, delay=delay), self.book, SPACES, CODEBOOK,
                                        _limits(), opener, send_reply)
                if sequential:
                    for request, offset in requests:
                        await fetcher.handle(request, trio.current_time() + offset)
                else:
                    async with trio.open_nursery() as inner:
                        for request, offset in requests:
                            inner.start_soon(fetcher.handle, request, trio.current_time() + offset)
                            await trio.sleep(0.05)
                nursery.cancel_scope.cancel()
                self.counters = fetcher.counters
        trio.run(run)
        return replies

    def _request(self, rid='r1', item='fp_item', wants=('c', 'l', 'n')):
        return {'rid': rid, 'item': item, 'server': 's1', 'provider': 't1', 'wants': list(wants),
                'reply': 'audiomuse_p2p_r_0123456789abcdef', 'ttl': 5.0}

    def _local(self, seed=0, duration=DURATION + 0.4):
        embedding = _musicnn(seed) + np.random.default_rng(5).standard_normal(200).astype(np.float32) * 1e-3
        return p2p_store.LocalTrack(embedding.astype(np.float32), duration, None)

    def test_the_worker_gets_exactly_the_served_outputs_and_versions(self):
        row, created = _row()
        replies = self._run({PEER_A: _server([(row, created)])}, self._local(), [(self._request(), 5.0)])

        ((channel, reply),) = replies
        assert channel == 'audiomuse_p2p_r_0123456789abcdef'
        assert reply['status'] == ipc.REPLY_OK
        assert reply['peer'] == PEER_A
        assert reply['versions'] == {'m': 'v3.6.3', 'c': 'v3.6.3', 'l': 'v3.5.0', 'n': 'v3.6.3'}
        assert np.array_equal(ipc.decode_floats(reply['outputs']['c'], 512), np.frombuffer(row.clap, np.float32))
        assert np.array_equal(ipc.decode_floats(reply['outputs']['a'], 27), np.frombuffer(row.axis, np.float32))
        assert ipc.decode_bytes(reply['outputs']['n']) == row.neural

    def test_an_unstamped_row_travels_with_the_serving_install_version(self, monkeypatch):
        monkeypatch.setattr(config, 'APP_VERSION', 'v3.7.2')
        row, created = _row(origin={})

        replies = self._run({PEER_A: _server([(row, created)])}, self._local(), [(self._request(), 5.0)])

        assert replies[0][1]['versions'] == {'m': 'v3.7.2', 'c': 'v3.7.2', 'l': 'v3.7.2', 'n': 'v3.7.2'}

    def test_a_received_output_travels_with_its_stamp_version(self):
        row, created = _row(origin={'m': _local(), 'c': _received('v3.2.0'), 'l': _local('v3.5.0'), 'n': _local()})

        replies = self._run({PEER_A: _server([(row, created)])}, self._local(), [(self._request(), 5.0)])

        assert replies[0][1]['status'] == ipc.REPLY_OK
        assert replies[0][1]['peer'] == PEER_A
        assert replies[0][1]['versions'] == {'m': 'v3.6.3', 'c': 'v3.2.0', 'l': 'v3.5.0', 'n': 'v3.6.3'}

    def test_only_the_wanted_outputs_travel(self):
        row, created = _row()
        replies = self._run({PEER_A: _server([(row, created)])}, self._local(),
                            [(self._request(wants=('c',)), 5.0)])

        assert sorted(replies[0][1]['outputs']) == ['c']
        assert replies[0][1]['versions'] == {'m': 'v3.6.3', 'c': 'v3.6.3'}

    def test_a_decoy_five_seconds_longer_gets_none(self):
        row, created = _row()
        replies = self._run({PEER_A: _server([(row, created)])}, self._local(duration=DURATION + 5.0),
                            [(self._request(), 5.0)])

        assert replies[0][1]['status'] == ipc.REPLY_NONE
        assert self.book.succeeded_peers == [PEER_A] and self.book.failed_peers == []

    def test_no_compatible_peer_answers_no_peers(self):
        replies = self._run({}, self._local(), [(self._request(), 5.0)])

        assert replies[0][1]['status'] == ipc.REPLY_NO_PEERS
        assert self.book.asked == [('c', 'l', 'n')]

    def test_every_query_tells_the_book_whether_the_server_answered(self):
        row, created = _row()

        replies = self._run({PEER_A: _server([(row, created)]), PEER_B: _Unreachable()}, self._local(),
                            [(self._request(), 5.0)])

        assert replies[0][1]['status'] == ipc.REPLY_OK
        assert replies[0][1]['peer'] == PEER_A
        assert self.book.succeeded_peers == [PEER_A]
        assert self.book.failed_peers == [PEER_B]

    def test_a_server_tor_cannot_reach_is_counted_in_the_lookup_summary(self, caplog):
        with caplog.at_level(logging.INFO, logger='p2p.fetch'):
            replies = self._run({PEER_A: _Unreachable()}, self._local(), [(self._request(), 5.0)])

        assert replies[0][1]['status'] == ipc.REPLY_NONE
        assert self.book.failed_peers == [PEER_A] and self.book.succeeded_peers == []
        assert '1 servers asked, 1 unreachable through Tor, 0 offers' in caplog.text
        assert 'onion service unreachable' in caplog.text

    def test_the_query_and_fetch_limits(self):
        assert fetch.QUERY_SECONDS == 10.0
        assert fetch.FETCH_MAX_SECONDS == 20.0

    def test_an_expired_request_is_dropped_without_a_reply(self):
        row, created = _row()
        replies = self._run({PEER_A: _server([(row, created)])}, self._local(), [(self._request(), -1.0)])

        assert replies == []
        assert self.counters['dropped'] == 1

    def test_a_second_request_for_the_same_item_is_merged(self):
        row, created = _row()
        server = _server([(row, created)])
        requests = [(self._request('r1'), 5.0), (self._request('r2', wants=('c',)), 5.0)]

        replies = self._run({PEER_A: server}, self._local(), requests, delay=0.3)

        assert sorted(reply['rid'] for _channel, reply in replies) == ['r1', 'r2']
        assert all(reply['status'] == ipc.REPLY_OK for _channel, reply in replies)
        assert server.counters['fetches'] == 1
        assert self.counters['merged'] == 1

    def test_an_undecodable_result_only_loses_that_result(self, monkeypatch, caplog):
        row, created = _row()
        server = _server([(row, created)])
        honest = serve.servable_outputs

        def broken(row, keys, spaces, codebook_crc):
            served = honest(row, keys, spaces, codebook_crc)
            if 'c' in served:
                served['c'] = (served['c'][0], np.full(512, np.nan, dtype=np.float32))
            return served

        monkeypatch.setattr(serve, 'servable_outputs', broken)

        with caplog.at_level(logging.WARNING, logger='p2p.fetch'):
            replies = self._run({PEER_A: server}, self._local(), [(self._request(), 5.0)])

        assert replies[0][1]['status'] == ipc.REPLY_OK
        assert sorted(replies[0][1]['versions']) == ['l', 'm', 'n']
        assert '7003' in caplog.text

    def test_a_malformed_answer_only_ends_that_stream(self, caplog):
        row, created = _row()
        garbled = _Garbled(_server([(row, created)]))
        requests = [(self._request('r1', item='fp_first'), 5.0), (self._request('r2', item='fp_second'), 5.0)]

        with caplog.at_level(logging.WARNING, logger='p2p.fetch'):
            replies = self._run({PEER_A: garbled}, self._local(), requests, sequential=True)

        assert [reply['status'] for _channel, reply in replies] == [ipc.REPLY_NONE, ipc.REPLY_OK]
        assert replies[1][1]['peer'] == PEER_A
        assert '7003' in caplog.text and 'malformed answer to a QUERY' in caplog.text
        assert self.book.failed_peers == [PEER_A] and self.book.succeeded_peers == [PEER_A]
