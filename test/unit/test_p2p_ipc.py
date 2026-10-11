# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The worker / Flask / P2P node message contract in p2p.ipc.

Postgres NOTIFY carries at most 8000 bytes, so a node reply is split into parts
the worker reassembles. These tests pin the builders, the part split and its
size bound, reassembly in any order, the refusal of every corrupt part, the
vector codecs and the status read that turns a stale heartbeat into "not
running" by the database clock.

Main Features:
* supply_request / supply_reply shapes; parse refuses other protocol versions
  and non-object payloads
* split_reply keeps every part within MAX_PAYLOAD_BYTES (and notify_p2p's
  limit) for replies up to and beyond 60 KB; PartAssembler rebuilds an
  expected reply from parts in any order and ignores duplicates of a part
* PartAssembler keeps no state for a request id it was not told to expect,
  one it discarded or one it already completed, so late or foreign replies
  never pile up
* a corrupt part (bad base64, changed total, foreign rid, wrong op, bad index)
  never yields a reply
* encode/decode round trips for float32 vectors and neural blobs
* read_status: missing row, unreadable row, fresh, stale and off states
* taskqueue.sql.notify_p2p publishes only on the node channel or a well-formed
  reply channel and refuses a payload over 7900 bytes, counted in bytes
* the channel names agree with taskqueue.sql, the output keys with database and
  the runtime keys (p2p_status, p2p_onion_key, p2p_peers) with the startup
  prune
"""

import base64
import json
import random

import numpy as np
import pytest

import config
import database
from p2p import ipc
from taskqueue import sql

PEER = 'peeralphatestonly' + 'a' * 39


def _reply(size, rid=None):
    rng = np.random.default_rng(size)
    outputs = {
        ipc.OUT_CLAP: ipc.encode_floats(rng.standard_normal(512)),
        ipc.OUT_LYRICS: ipc.encode_floats(rng.standard_normal(768)),
        ipc.OUT_AXIS: ipc.encode_floats(rng.random(27)),
        ipc.OUT_NEURAL: ipc.encode_bytes(rng.integers(0, 256, size, dtype=np.uint8).tobytes()),
    }
    versions = {ipc.OUT_MUSICNN: 'v3.1.0', ipc.OUT_CLAP: 'v3.1.0', ipc.OUT_LYRICS: 'v3.2.0', ipc.OUT_NEURAL: 'v3.6.3'}
    return ipc.supply_reply(rid or ipc.new_request_id(), ipc.REPLY_OK, peer=PEER, versions=versions,
                            outputs=outputs)


def _assembler(*rids):
    assembler = ipc.PartAssembler()
    for rid in rids:
        assembler.expect(rid)
    return assembler


class _FakeCursor:
    def __init__(self, row):
        self.row = row
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.row


class TestBuilders:
    def test_supply_request_keeps_only_suppliable_outputs_in_protocol_order(self):
        request = ipc.supply_request('rid1', 'item-1', 7, 'prov-9', ['n', 'x', 'c', 'm', 'c'], 'chan', 5)
        assert request == {
            'v': ipc.PROTOCOL_VERSION,
            'op': ipc.OP_SUPPLY,
            'rid': 'rid1',
            'item': 'item-1',
            'server': '7',
            'provider': 'prov-9',
            'wants': ['c', 'n'],
            'reply': 'chan',
            'ttl': 5.0,
        }

    def test_supply_request_keeps_missing_server_and_provider_as_none(self):
        request = ipc.supply_request('rid1', 'item-1', None, None, ['l'], 'chan', 2)
        assert request['server'] is None
        assert request['provider'] is None
        assert request['wants'] == ['l']

    def test_supply_reply_defaults_to_empty_maps(self):
        reply = ipc.supply_reply('r', ipc.REPLY_NONE)
        assert reply['versions'] == {} and reply['outputs'] == {} and reply['peer'] is None
        assert reply['op'] == ipc.OP_SUPPLIED

    def test_new_ids_and_reply_channels_are_unique_and_accepted_by_notify(self):
        channels = {ipc.new_reply_channel() for _ in range(50)}
        assert len(channels) == 50
        for channel in channels:
            assert sql._P2P_CHANNEL_RE.fullmatch(channel)
        assert len({ipc.new_request_id() for _ in range(50)}) == 50

    def test_dumps_parse_round_trip(self):
        message = ipc.supply_request('rid1', 'item-1', 7, 'prov-9', ['c'], 'chan', 5)
        assert ipc.parse(ipc.dumps(message)) == message

    @pytest.mark.parametrize('payload', [
        None, '', 'not json', '[1, 2]', '"text"', '{"v": 99, "op": "supply"}', '{"op": "supply"}',
        '{"v": 1}', '{"v": 1, "op": 3}',
    ])
    def test_parse_refuses_anything_but_a_current_protocol_object(self, payload):
        assert ipc.parse(payload) is None


class TestSplitAndAssemble:
    @pytest.mark.parametrize('blob_size', [0, 100, 20000, 45000, 60000, 90000])
    def test_every_part_fits_one_notify(self, blob_size):
        reply = _reply(blob_size)
        parts = ipc.split_reply(reply)
        total = len(ipc.dumps(reply).encode('ascii'))
        if total > 60000:
            assert len(parts) > 1
        for part in parts:
            assert len(part.encode('utf-8')) <= ipc.MAX_PAYLOAD_BYTES
            assert len(part.encode('utf-8')) <= sql.P2P_MAX_PAYLOAD_BYTES

    def test_sixty_kilobyte_reply_round_trips(self):
        reply = _reply(60000)
        assembler = _assembler(reply['rid'])
        parts = ipc.split_reply(reply)
        results = [assembler.add(ipc.parse(part)) for part in parts]
        assert all(result is None for result in results[:-1])
        assert results[-1] == reply

    def test_out_of_order_parts_round_trip(self):
        reply = _reply(60000)
        parts = ipc.split_reply(reply)
        shuffled = parts[:]
        random.Random(3).shuffle(shuffled)
        if shuffled == parts:
            shuffled.reverse()
        assembler = _assembler(reply['rid'])
        results = [assembler.add(ipc.parse(part)) for part in shuffled]
        assert [result for result in results if result is not None] == [reply]
        assert results[-1] == reply

    def test_a_repeated_part_does_not_complete_the_reply_early(self):
        reply = _reply(30000)
        parts = ipc.split_reply(reply)
        assert len(parts) >= 3
        assembler = _assembler(reply['rid'])
        assert assembler.add(ipc.parse(parts[0])) is None
        assert assembler.add(ipc.parse(parts[0])) is None
        for part in parts[1:-1]:
            assert assembler.add(ipc.parse(part)) is None
        assert assembler.add(ipc.parse(parts[-1])) == reply

    def test_interleaved_expected_replies_reassemble_independently(self):
        first, second = _reply(20000), _reply(25000)
        first_parts, second_parts = ipc.split_reply(first), ipc.split_reply(second)
        assembler = _assembler(first['rid'], second['rid'])
        done = []
        for index in range(max(len(first_parts), len(second_parts))):
            for parts in (first_parts, second_parts):
                if index < len(parts):
                    result = assembler.add(ipc.parse(parts[index]))
                    if result is not None:
                        done.append(result)
        assert sorted(done, key=lambda r: r['rid']) == sorted([first, second], key=lambda r: r['rid'])

    def test_a_small_reply_is_one_part(self):
        reply = ipc.supply_reply('abc', ipc.REPLY_NONE)
        parts = ipc.split_reply(reply)
        assert len(parts) == 1
        assert _assembler('abc').add(ipc.parse(parts[0])) == reply


class TestExpect:
    def test_parts_of_a_request_nobody_expects_are_ignored_and_never_kept(self):
        reply = _reply(30000)
        assembler = ipc.PartAssembler()
        results = [assembler.add(ipc.parse(part)) for part in ipc.split_reply(reply)]
        assert results == [None] * len(results)
        assert assembler._parts == {}

    def test_only_the_expected_request_is_assembled(self):
        wanted, foreign = _reply(20000), _reply(20000)
        assembler = _assembler(wanted['rid'])
        assert [assembler.add(ipc.parse(part)) for part in ipc.split_reply(foreign)] == [None] * len(
            ipc.split_reply(foreign)
        )
        assert set(assembler._parts) == {wanted['rid']}
        assert [assembler.add(ipc.parse(part)) for part in ipc.split_reply(wanted)][-1] == wanted
        assert assembler._parts == {}

    def test_discard_forgets_a_partial_reply_and_ignores_its_late_parts(self):
        reply = _reply(30000)
        parts = ipc.split_reply(reply)
        assembler = _assembler(reply['rid'])
        for part in parts[:-1]:
            assembler.add(ipc.parse(part))
        assembler.discard(reply['rid'])
        assert assembler._parts == {}
        assert assembler.add(ipc.parse(parts[-1])) is None
        for part in parts:
            assert assembler.add(ipc.parse(part)) is None
        assert assembler._parts == {}

    def test_a_completed_reply_leaves_no_state_and_its_duplicates_are_ignored(self):
        reply = _reply(30000)
        parts = ipc.split_reply(reply)
        assembler = _assembler(reply['rid'])
        assert [assembler.add(ipc.parse(part)) for part in parts][-1] == reply
        assert assembler._parts == {}
        assert assembler.add(ipc.parse(parts[0])) is None
        assert assembler._parts == {}

    def test_discarding_an_unknown_request_is_harmless(self):
        assembler = _assembler('r1')
        assembler.discard('never-expected')
        assert set(assembler._parts) == {'r1'}

    def test_expecting_twice_keeps_the_parts_already_received(self):
        reply = _reply(30000)
        parts = ipc.split_reply(reply)
        assembler = _assembler(reply['rid'])
        assembler.add(ipc.parse(parts[0]))
        assembler.expect(reply['rid'])
        results = [assembler.add(ipc.parse(part)) for part in parts[1:]]
        assert results[-1] == reply

    @pytest.mark.parametrize('rid', [None, 7, b'r1'])
    def test_only_a_text_request_id_can_be_expected(self, rid):
        assembler = _assembler(rid)
        assert assembler._parts == {}


def _parts_as_dicts(reply):
    return [json.loads(part) for part in ipc.split_reply(reply)]


class TestCorruptParts:
    def test_bad_base64_body_yields_nothing(self):
        part = _parts_as_dicts(ipc.supply_reply('r1', ipc.REPLY_NONE))[0]
        part['d'] = '!!!not base64!!!'
        assert _assembler('r1').add(part) is None

    def test_valid_base64_of_non_json_yields_nothing(self):
        part = _parts_as_dicts(ipc.supply_reply('r1', ipc.REPLY_NONE))[0]
        part['d'] = base64.b64encode(b'not json at all').decode('ascii')
        assert _assembler('r1').add(part) is None

    def test_a_body_whose_rid_differs_from_the_part_rid_yields_nothing(self):
        part = _parts_as_dicts(ipc.supply_reply('r1', ipc.REPLY_NONE))[0]
        part['rid'] = 'someone-else'
        assert _assembler('r1', 'someone-else').add(part) is None

    def test_a_body_that_is_not_a_supply_reply_yields_nothing(self):
        request = ipc.supply_request('r1', 'item-1', None, None, ['c'], 'chan', 1)
        body = base64.b64encode(ipc.dumps(request).encode('ascii')).decode('ascii')
        part = {'v': ipc.PROTOCOL_VERSION, 'op': ipc.OP_PART, 'rid': 'r1', 'i': 0, 'n': 1, 'd': body}
        assert _assembler('r1').add(part) is None

    def test_a_changed_total_drops_the_partial_reply(self):
        reply = _reply(30000)
        parts = _parts_as_dicts(reply)
        assembler = _assembler(reply['rid'])
        assert assembler.add(parts[0]) is None
        forged = dict(parts[1], n=parts[1]['n'] + 1)
        assert assembler.add(forged) is None
        for part in parts[1:]:
            assert assembler.add(part) is None
        assert assembler._parts == {}

    @pytest.mark.parametrize('change', [
        {'op': 'supplied'},
        {'i': -1},
        {'i': 5, 'n': 5},
        {'n': 0},
        {'i': '0'},
        {'n': None},
        {'rid': 7},
        {'d': 12},
    ])
    def test_malformed_part_fields_yield_nothing(self, change):
        part = _parts_as_dicts(ipc.supply_reply('r1', ipc.REPLY_NONE))[0]
        part.update(change)
        assert _assembler('r1').add(part) is None

    def test_none_and_empty_messages_are_ignored(self):
        assembler = _assembler('r1')
        assert assembler.add(None) is None
        assert assembler.add({}) is None


class TestCodecs:
    def test_float_round_trip_is_little_endian_float32(self):
        vector = np.linspace(-1, 1, 512).astype(np.float64)
        text = ipc.encode_floats(vector)
        assert base64.b64decode(text) == vector.astype('<f4').tobytes()
        decoded = ipc.decode_floats(text, 512)
        assert decoded.dtype == np.float32
        assert np.array_equal(decoded, vector.astype(np.float32))

    def test_decoded_floats_are_writable_copies(self):
        decoded = ipc.decode_floats(ipc.encode_floats(np.ones(4)), 4)
        decoded[0] = 5.0
        assert decoded[0] == 5.0

    def test_wrong_dimension_or_garbage_decodes_to_none(self):
        text = ipc.encode_floats(np.ones(10))
        assert ipc.decode_floats(text, 11) is None
        assert ipc.decode_floats('%%%', 10) is None
        assert ipc.decode_floats(None, 10) is None

    def test_bytes_round_trip(self):
        blob = bytes(range(256)) * 3
        assert ipc.decode_bytes(ipc.encode_bytes(blob)) == blob
        assert ipc.decode_bytes(ipc.encode_bytes(memoryview(blob))) == blob
        assert ipc.decode_bytes(12) is None
        assert ipc.decode_bytes('a') is None


class TestReadStatus:
    def test_missing_row_reads_as_not_running(self):
        cur = _FakeCursor(None)
        assert ipc.read_status(cur) == {'state': ipc.STATE_NOT_RUNNING}
        query, params = cur.executed[0]
        assert 'now() - updated_at' in query
        assert params == (ipc.STATUS_KEY,)

    @pytest.mark.parametrize('value', ['not json', '[1]', None])
    def test_unreadable_row_reads_as_not_running(self, value):
        assert ipc.read_status(_FakeCursor((value, 1.0))) == {'state': ipc.STATE_NOT_RUNNING}

    def test_fresh_connected_status_is_returned_with_its_age(self):
        row = (json.dumps({'state': ipc.STATE_CONNECTED, 'reason': 'running', 'peer_id': PEER, 'peers': 3}), 4.5)
        status = ipc.read_status(_FakeCursor(row))
        assert status == {'state': ipc.STATE_CONNECTED, 'reason': 'running', 'peer_id': PEER, 'peers': 3}

    @pytest.mark.parametrize('state', [ipc.STATE_CONNECTED, ipc.STATE_STARTING, ipc.STATE_NO_PEERS, ipc.STATE_ERROR])
    def test_stale_live_states_read_as_not_running(self, state):
        row = (json.dumps({'state': state}), ipc.STATUS_FRESH_SECONDS + 1)
        assert ipc.read_status(_FakeCursor(row))['state'] == ipc.STATE_NOT_RUNNING

    def test_a_live_state_without_an_age_reads_as_not_running(self):
        row = (json.dumps({'state': ipc.STATE_CONNECTED}), None)
        status = ipc.read_status(_FakeCursor(row))
        assert status['state'] == ipc.STATE_NOT_RUNNING

    @pytest.mark.parametrize('state', [ipc.STATE_OFF, ipc.STATE_UNAVAILABLE])
    def test_off_and_unavailable_never_go_stale(self, state):
        row = (json.dumps({'state': state}), 10 * ipc.STATUS_FRESH_SECONDS)
        assert ipc.read_status(_FakeCursor(row))['state'] == state

    def test_status_labels_are_words_and_unknown_is_not_running(self):
        assert ipc.status_label(ipc.STATE_CONNECTED) == 'Connected'
        assert ipc.status_label(ipc.STATE_NO_PEERS) == 'No peers found'
        assert ipc.status_label('bogus') == 'Not running'
        assert set(ipc.STATE_LABELS) == {
            ipc.STATE_OFF, ipc.STATE_UNAVAILABLE, ipc.STATE_STARTING, ipc.STATE_CONNECTED,
            ipc.STATE_NO_PEERS, ipc.STATE_ERROR, ipc.STATE_NOT_RUNNING,
        }


class TestNotifyP2P:
    def test_the_node_channel_and_a_reply_channel_are_published(self):
        cur = _FakeCursor(None)
        reply_channel = ipc.new_reply_channel()
        sql.notify_p2p(cur, sql.CHANNEL_P2P, '{"a":1}')
        sql.notify_p2p(cur, reply_channel, {'b': [1, 2]})
        assert cur.executed == [
            ('SELECT pg_notify(%s, %s)', ('audiomuse_p2p', '{"a":1}')),
            ('SELECT pg_notify(%s, %s)', (reply_channel, '{"b":[1,2]}')),
        ]

    @pytest.mark.parametrize('channel', [
        None, '', 'audiomuse_job', 'audiomuse_p2p_r_', 'audiomuse_p2p_r_0123456789abcde',
        'audiomuse_p2p_r_0123456789abcdef0', 'audiomuse_p2p_r_0123456789ABCDEF', 'audiomuse_p2p\n',
        'audiomuse_p2p_r_0123456789abcdef\n', 'xaudiomuse_p2p', 'audiomuse_p2p; LISTEN x',
    ])
    def test_any_other_channel_is_refused(self, channel):
        cur = _FakeCursor(None)
        with pytest.raises(ValueError):
            sql.notify_p2p(cur, channel, '{}')
        assert cur.executed == []

    def test_the_payload_limit_counts_bytes(self):
        cur = _FakeCursor(None)
        sql.notify_p2p(cur, sql.CHANNEL_P2P, 'x' * sql.P2P_MAX_PAYLOAD_BYTES)
        with pytest.raises(ValueError):
            sql.notify_p2p(cur, sql.CHANNEL_P2P, 'x' * (sql.P2P_MAX_PAYLOAD_BYTES + 1))
        with pytest.raises(ValueError):
            sql.notify_p2p(cur, sql.CHANNEL_P2P, chr(233) * (sql.P2P_MAX_PAYLOAD_BYTES // 2 + 1))
        assert len(cur.executed) == 1

    def test_every_part_of_a_large_reply_is_accepted(self):
        cur = _FakeCursor(None)
        channel = ipc.new_reply_channel()
        parts = ipc.split_reply(_reply(60000))
        for part in parts:
            sql.notify_p2p(cur, channel, part)
        assert len(cur.executed) == len(parts)


class TestSharedNames:
    def test_channels_match_the_queue_sql(self):
        assert ipc.NODE_CHANNEL == sql.CHANNEL_P2P
        assert ipc.REPLY_CHANNEL_PREFIX == sql.P2P_REPLY_CHANNEL_PREFIX
        assert ipc.MAX_PAYLOAD_BYTES == sql.P2P_MAX_PAYLOAD_BYTES

    def test_the_runtime_keys_are_exactly_the_ones_the_startup_prune_keeps(self):
        keys = {ipc.STATUS_KEY, ipc.ONION_KEY_KEY, ipc.PEERS_KEY}
        assert keys == {'p2p_status', 'p2p_onion_key', 'p2p_peers'}
        assert {key for key in config.APP_CONFIG_RUNTIME_KEYS if key.startswith('p2p_')} == keys
        assert not hasattr(ipc, 'NODE_KEY_KEY')

    def test_output_keys_match_the_database_origin_keys(self):
        assert (ipc.OUT_MUSICNN, ipc.OUT_CLAP, ipc.OUT_LYRICS, ipc.OUT_NEURAL) == (
            database.OUTPUT_MUSICNN, database.OUTPUT_CLAP, database.OUTPUT_LYRICS, database.OUTPUT_NEURAL,
        )
        assert ipc.OUT_AXIS == 'a'
