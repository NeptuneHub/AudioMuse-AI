# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The /audiomuse-ai/1.0.0 frame format and the gossip HELLO (p2p.wire).

Every frame carries the wire version and its type, and its length is checked
against the cap of that type before a byte of payload is read. These tests pin
the round trips, the refusal paths and the privacy promise of the format.

Main Features:
* QUERY, NO, OFFER, FETCH, OUTPUTS, BUSY, END, PEERS and HELLO survive an
  encode/decode round trip bit for bit (vectors as float32)
* an unknown wire version or frame type raises UnknownVersion / UnknownType,
  which the server answers with NO, never a Violation
* a declared length over the cap is refused before the payload is read;
  truncated, padded or unknown-bit payloads of a known version are Violations
* the version rule lives here: MIN_OUTPUT_VERSION = (3, 0, 0), parse_version
  reads 'v3.0.0', '3.10.2' and 'v2.6.2' numerically and refuses garbage,
  version_allowed admits exactly v3.0.0 and newer (the running version too)
* a version below v3.0.0 or unreadable only fails version_passes
* no frame ever carries an fp_ catalogue id or any text but version strings
* wire imports no AudioMuse-AI module (no config, no p2p.compare), so any
  process can speak the protocol without a database
"""

import ast
import inspect
import struct

import numpy as np
import pytest
import trio

import config
from p2p import wire

SPACES = {key: bytes([index + 1]) * 32 for index, key in enumerate(wire.SPACE_KEYS)}


def _payload(frame):
    cursor = wire._Cursor(frame[2:])
    size = cursor.uvarint(2 ** 28)
    return cursor.take(size)


def _frame(frame_type, payload):
    return bytes([wire.WIRE_VERSION, frame_type]) + wire.encode_uvarint(len(payload)) + bytes(payload)


def _unit(dim, seed=0):
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _axis(seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.random(size).astype(np.float32) for size in (6, 6, 6, 5, 4)]
    return np.concatenate([part / part.sum() for part in parts]).astype(np.float32)


class _Reader:
    def __init__(self, data):
        self.data = bytearray(data)
        self.requested = []

    async def read_exactly(self, size):
        self.requested.append(size)
        if len(self.data) < size:
            raise EOFError('short read')
        chunk = bytes(self.data[:size])
        del self.data[:size]
        return chunk


def _read(data):
    return trio.run(wire.read_frame, _Reader(data))


def _candidate(seed=0, versions=None, chromaprint=b'\x01\x02\x03'):
    return wire.Candidate(
        bytes([seed]) * wire.TOKEN_BYTES, _unit(wire.MUSICNN_DIM, seed), 200.25 + seed,
        versions or {'m': 'v3.6.3', 'c': 'v3.6.3', 'l': 'v3.0.0'}, chromaprint,
    )


def _outputs():
    return {
        'c': wire.Output(SPACES['c'], 'v3.6.3', _unit(wire.CLAP_DIM, 1)),
        'l': wire.Output(SPACES['l'], 'v3.5.0', (_unit(wire.LYRICS_DIM, 2), _axis(3))),
        'n': wire.Output(SPACES['n'], 'v3.6.3', b'NFP2' + bytes(200)),
    }


class TestRoundTrips:
    def test_query(self):
        frame = wire.encode_query(b'\x5a' * 25, 187.5, ['n', 'c'], SPACES)
        frame_type, message = _read(frame)

        assert frame_type == wire.T_QUERY
        assert message.signature == b'\x5a' * 25
        assert message.duration == 187.5
        assert message.wants == ('c', 'n')
        assert message.spaces == {'m': SPACES['m'], 'c': SPACES['c'], 'n': SPACES['n']}

    @pytest.mark.parametrize('reason', sorted(wire.NO_REASONS))
    def test_no(self, reason):
        assert _read(wire.encode_no(reason)) == (wire.T_NO, wire.No(reason))

    def test_an_unknown_no_reason_reads_as_not_found(self):
        assert _read(_frame(wire.T_NO, b'\xee'))[1] == wire.No(wire.NO_NOT_FOUND)

    def test_busy_and_end(self):
        assert _read(wire.encode_busy()) == (wire.T_BUSY, wire.Busy())
        assert _read(wire.encode_end()) == (wire.T_END, wire.End())

    def test_offer_with_two_candidates(self):
        first = _candidate(0)
        second = _candidate(1, versions={'m': 'v3.0.0', 'n': 'v3.1.0'}, chromaprint=b'')
        frame_type, message = _read(wire.encode_offer([first, second]))

        assert frame_type == wire.T_OFFER
        assert len(message.candidates) == 2
        got = message.candidates[0]
        assert got.token == first.token
        assert np.array_equal(got.embedding, first.embedding)
        assert got.embedding.dtype == np.float32
        assert got.duration == first.duration
        assert got.versions == first.versions
        assert got.chromaprint == first.chromaprint
        assert message.candidates[1].versions == {'m': 'v3.0.0', 'n': 'v3.1.0'}
        assert message.candidates[1].chromaprint == b''

    def test_fetch(self):
        frame_type, message = _read(wire.encode_fetch(b'\x07' * 16, ['l', 'c'], SPACES))

        assert frame_type == wire.T_FETCH
        assert message.token == b'\x07' * 16
        assert message.wants == ('c', 'l')
        assert message.spaces == {'c': SPACES['c'], 'l': SPACES['l']}

    def test_outputs_carry_space_version_and_exact_values(self):
        sent = _outputs()
        frame_type, message = _read(wire.encode_outputs(sent))

        assert frame_type == wire.T_OUTPUTS
        assert sorted(message.items) == ['c', 'l', 'n']
        for key, item in sent.items():
            got = message.items[key]
            assert got.space == item.space
            assert got.version == item.version
        assert np.array_equal(message.items['c'].value, sent['c'].value)
        assert np.array_equal(message.items['l'].value[0], sent['l'].value[0])
        assert np.array_equal(message.items['l'].value[1], sent['l'].value[1])
        assert message.items['n'].value == sent['n'].value

    def test_peers(self):
        entries = [wire.PeerEntry(b'\x00\x24' + b'\x11' * 36, (b'\x04\x7f\x00\x00\x01\x06\x10\x05',))]
        frame_type, message = _read(wire.encode_peers(entries))

        assert frame_type == wire.T_PEERS
        assert message.peers == tuple(entries)
        assert _read(wire.encode_peers([]))[1] == wire.Peers(())

    def test_hello(self):
        hello = wire.decode_hello(wire.encode_hello('v3.6.3', SPACES, 1790000000))

        assert hello == wire.Hello('v3.6.3', SPACES, 1790000000)

    def test_hello_skips_absent_spaces(self):
        hello = wire.decode_hello(wire.encode_hello('v3.6.3', {'m': SPACES['m'], 'c': None}, 5))

        assert hello.spaces == {'m': SPACES['m']}


class TestRefusals:
    def test_an_unknown_wire_version_is_not_a_violation(self):
        frame = bytes([wire.WIRE_VERSION + 1, wire.T_QUERY]) + b'\x00'

        with pytest.raises(wire.UnknownVersion):
            _read(frame)
        assert not issubclass(wire.UnknownVersion, wire.Violation)

    def test_an_unknown_frame_type_is_not_a_violation(self):
        with pytest.raises(wire.UnknownType):
            _read(bytes([wire.WIRE_VERSION, 99, 0]))
        assert not issubclass(wire.UnknownType, wire.Violation)

    def test_a_hello_type_is_refused_on_a_stream(self):
        with pytest.raises(wire.UnknownType):
            _read(wire.encode_hello('v3.6.3', SPACES, 1))

    def test_a_length_over_the_cap_is_refused_before_reading_the_payload(self):
        declared = wire.FRAME_CAPS[wire.T_QUERY] + 1
        frame = bytes([wire.WIRE_VERSION, wire.T_QUERY]) + wire.encode_uvarint(declared)
        reader = _Reader(frame)

        with pytest.raises(wire.Violation):
            trio.run(wire.read_frame, reader)
        assert declared not in reader.requested
        assert max(reader.requested) <= 2

    def test_a_huge_uvarint_is_refused(self):
        frame = bytes([wire.WIRE_VERSION, wire.T_OUTPUTS]) + b'\xff' * 6

        with pytest.raises(wire.Violation):
            _read(frame)

    def test_trailing_bytes_are_a_violation(self):
        with pytest.raises(wire.Violation):
            _read(_frame(wire.T_NO, b'\x00\x01'))

    def test_a_truncated_payload_is_a_violation(self):
        payload = _payload(wire.encode_query(b'\x01' * 25, 10.0, ['c'], SPACES))

        with pytest.raises(wire.Violation):
            _read(_frame(wire.T_QUERY, payload[:-5]))

    def test_an_unknown_output_bit_is_a_violation(self):
        payload = b'\x01' * 25 + struct.pack('<d', 10.0) + bytes([0x08]) + SPACES['m']

        with pytest.raises(wire.Violation):
            _read(_frame(wire.T_QUERY, payload))

    @pytest.mark.parametrize('duration', [0.0, -1.0, float('nan'), float('inf')])
    def test_a_query_needs_a_positive_finite_duration(self, duration):
        payload = b'\x01' * 25 + struct.pack('<d', duration) + bytes([1]) + SPACES['m'] + SPACES['c']

        with pytest.raises(wire.Violation):
            _read(_frame(wire.T_QUERY, payload))

    def test_an_offer_with_three_candidates_is_a_violation(self):
        one = _payload(wire.encode_offer([_candidate(0)]))[1:]

        with pytest.raises(wire.Violation):
            _read(_frame(wire.T_OFFER, bytes([3]) + one * 3))

    def test_an_empty_offer_is_a_violation(self):
        with pytest.raises(wire.Violation):
            _read(_frame(wire.T_OFFER, b'\x00'))

    def test_the_encoder_refuses_an_oversized_neural_blob(self):
        outputs = {'n': wire.Output(SPACES['n'], 'v3.6.3', bytes(wire.MAX_NEURAL_BYTES + 1))}

        with pytest.raises(ValueError):
            wire.encode_outputs(outputs)

    def test_a_malformed_hello_raises_a_wire_error(self):
        with pytest.raises(wire.WireError):
            wire.decode_hello(b'\x01')
        with pytest.raises(wire.UnknownVersion):
            wire.decode_hello(bytes([9, wire.T_HELLO, 0]))


class TestVersionRule:
    @pytest.mark.parametrize('text, parsed', [
        ('v3.0.0', (3, 0, 0)),
        ('3.10.2', (3, 10, 2)),
        ('v2.6.2', (2, 6, 2)),
        ('v3.6.3', (3, 6, 3)),
        (' v4.0.1 ', (4, 0, 1)),
        ('v3.1.0-rc1', (3, 1, 0)),
    ])
    def test_versions_parse_numerically(self, text, parsed):
        assert wire.parse_version(text) == parsed

    @pytest.mark.parametrize('text', [None, '', 'garbage', 'v3', 'v3.0', 'version 3.0.0', 'x3.0.0', 300, b'v3.0.0'])
    def test_garbage_does_not_parse(self, text):
        assert wire.parse_version(text) is None

    @pytest.mark.parametrize('text', ['v3.0.0', '3.0.0', 'v3.0.1', 'v3.10.0', '3.10.2', 'v10.0.0', 'v3.6.3'])
    def test_v3_and_newer_are_allowed(self, text):
        assert wire.version_allowed(text)

    @pytest.mark.parametrize('text', ['v2.6.2', '2.99.99', 'v0.1.0', None, 'garbage', ''])
    def test_older_or_unreadable_versions_are_refused(self, text):
        assert not wire.version_allowed(text)

    def test_the_minimum_is_v3_0_0(self):
        assert wire.MIN_OUTPUT_VERSION == (3, 0, 0)

    def test_the_running_version_is_allowed(self):
        assert wire.version_allowed(config.APP_VERSION)


class TestVersionFloor:
    @pytest.mark.parametrize('text', ['v3.0.0', 'v3.6.3', '3.10.2', 'v4.0.0-rc1'])
    def test_v3_and_newer_pass(self, text):
        assert wire.version_passes(text)

    @pytest.mark.parametrize('text', ['v2.9.9', 'v0.1.0', '', 'garbage', None, 'v3.0', 'v3.0.0 x'])
    def test_older_or_unreadable_versions_do_not_pass(self, text):
        assert not wire.version_passes(text)

    def test_an_output_below_the_floor_decodes_and_only_fails_the_version_check(self):
        sent = {'c': wire.Output(SPACES['c'], 'v2.6.2', _unit(wire.CLAP_DIM))}
        message = _read(wire.encode_outputs(sent))[1]

        assert message.items['c'].version == 'v2.6.2'
        assert not wire.version_passes(message.items['c'].version)

    def test_a_non_ascii_version_reads_as_unreadable_not_as_a_violation(self):
        frame = bytearray(wire.encode_outputs({'c': wire.Output(SPACES['c'], 'v3.6.3', _unit(wire.CLAP_DIM))}))
        index = frame.index(b'v3.6.3')
        frame[index] = 0xC3

        message = _read(bytes(frame))[1]

        assert not wire.version_passes(message.items['c'].version)


class TestPrivacy:
    def test_no_frame_carries_a_catalogue_id(self):
        frames = [
            wire.encode_query(b'\x01' * 25, 200.0, ['c', 'l', 'n'], SPACES),
            wire.encode_offer([_candidate(0), _candidate(1)]),
            wire.encode_fetch(b'\x02' * 16, ['c', 'l', 'n'], SPACES),
            wire.encode_outputs(_outputs()),
            wire.encode_peers([wire.PeerEntry(b'\x00\x24' + b'\x11' * 36, (b'\x04\x7f\x00\x00\x01',))]),
            wire.encode_hello('v3.6.3', SPACES, 1),
            wire.encode_no(wire.NO_NOT_FOUND),
            wire.encode_busy(),
            wire.encode_end(),
        ]

        for frame in frames:
            assert b'fp_' not in frame

    def test_the_encoders_take_no_identifiers(self):
        for name in ('encode_query', 'encode_offer', 'encode_fetch', 'encode_outputs', 'encode_hello'):
            params = inspect.signature(getattr(wire, name)).parameters
            assert not {'item_id', 'server_id', 'provider_id', 'path', 'title'} & set(params)


class TestHelpers:
    @pytest.mark.parametrize('value', [0, 1, 127, 128, 300, 16383, 16384, 2 ** 21])
    def test_uvarint_round_trip(self, value):
        encoded = wire.encode_uvarint(value) + b'\x00'
        cursor = wire._Cursor(encoded)

        assert cursor.uvarint(2 ** 28) == value

    def test_bits_and_keys_are_inverse(self):
        for keys in [(), ('c',), ('l',), ('n',), ('c', 'l'), ('c', 'l', 'n')]:
            assert wire.keys_of(wire.bits_of(keys)) == keys


class TestImports:
    def test_wire_imports_no_audiomuse_module(self):
        with open(wire.__file__, 'r', encoding='utf-8') as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or '').split('.')[0])

        assert imported == {'re', 'struct', 'typing', 'numpy'}
