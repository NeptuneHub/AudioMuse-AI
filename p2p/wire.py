# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The byte format of the /audiomuse-ai/1.0.0 stream protocol and the gossip HELLO.

Every frame starts with the wire version byte and a type byte, then a uvarint
payload length that is checked against the cap of that type BEFORE the payload
is read, so a peer can never make the node buffer more than a frame's worth.
Integers and floats are little-endian; vectors travel as float32.

A frame names model spaces by their full 32-byte ids and every result by the
AudioMuse-AI version that computed it. Nothing on the wire identifies a
library: no catalogue, provider or server id, no path, title or lyric text.

Main Features:
* QUERY / NO / OFFER / FETCH / OUTPUTS / BUSY / END / PEERS frames and the HELLO
  gossip message, each with encode_* and a decoder behind read_frame
* UnknownVersion and UnknownType are answered with NO or ignored; Violation
  marks a malformed frame of a known version, which only ends that stream
* MIN_OUTPUT_VERSION (v3.0.0), parse_version and version_allowed: only results
  computed by AudioMuse-AI v3.0.0 or newer are exchanged (older releases used
  other models); version_passes applies the floor every install agrees on, a
  result below it or unparsable is dropped as incompatible
* imports no AudioMuse-AI configuration: the protocol is plain data, usable by
  any process
"""

import re
import struct
from typing import NamedTuple

import numpy as np

MIN_OUTPUT_VERSION = (3, 0, 0)
WIRE_VERSION = 1
PROTOCOL_ID = '/audiomuse-ai/1.0.0'
TOPIC = 'audiomuse-ai'

T_QUERY = 1
T_NO = 2
T_OFFER = 3
T_FETCH = 4
T_OUTPUTS = 5
T_BUSY = 6
T_END = 7
T_PEERS = 8
T_HELLO = 9
STREAM_TYPES = frozenset({T_QUERY, T_NO, T_OFFER, T_FETCH, T_OUTPUTS, T_BUSY, T_END, T_PEERS})

NO_NOT_FOUND = 0
NO_VERSION = 1
NO_SPACE = 2
NO_BUILDING = 3
NO_TOKEN = 4
NO_OUTPUTS = 5
NO_REASONS = frozenset({NO_NOT_FOUND, NO_VERSION, NO_SPACE, NO_BUILDING, NO_TOKEN, NO_OUTPUTS})

OUT_MUSICNN = 'm'
OUT_CLAP = 'c'
OUT_LYRICS = 'l'
OUT_NEURAL = 'n'
OUTPUT_KEYS = (OUT_CLAP, OUT_LYRICS, OUT_NEURAL)
SPACE_KEYS = (OUT_MUSICNN, OUT_CLAP, OUT_LYRICS, OUT_NEURAL)
OUTPUT_BITS = {OUT_CLAP: 1, OUT_LYRICS: 2, OUT_NEURAL: 4}
ALL_BITS = sum(OUTPUT_BITS.values())

SIGNATURE_BYTES = 25
SPACE_ID_BYTES = 32
TOKEN_BYTES = 16
MUSICNN_DIM = 200
CLAP_DIM = 512
LYRICS_DIM = 768
AXIS_DIM = 27
MAX_CANDIDATES = 2
MAX_VERSION_CHARS = 32
MAX_CHROMAPRINT_BYTES = 64 * 1024
MAX_NEURAL_BYTES = 1024 * 1024
MAX_PEERS_LISTED = 32
MAX_PEER_ID_BYTES = 64
MAX_ADDRS_PER_PEER = 8
MAX_ADDR_BYTES = 128
HELLO_MAX_BYTES = 1024
UVARINT_MAX_BYTES = 4

FRAME_CAPS = {
    T_QUERY: 256,
    T_NO: 16,
    T_BUSY: 16,
    T_END: 16,
    T_FETCH: 256,
    T_OFFER: 160 * 1024,
    T_OUTPUTS: MAX_NEURAL_BYTES + 16 * 1024,
    T_PEERS: 64 * 1024,
    T_HELLO: HELLO_MAX_BYTES,
}

_HEADER = struct.Struct('<BB')
_F64 = struct.Struct('<d')
_U64 = struct.Struct('<Q')
_VERSION_TEXT_RE = re.compile(r'^[A-Za-z0-9.+-]{1,32}$')
_VERSION_RE = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)')


class WireError(Exception):
    pass


class UnknownVersion(WireError):
    pass


class UnknownType(WireError):
    pass


class Violation(WireError):
    pass


class Query(NamedTuple):
    signature: bytes
    duration: float
    wants: tuple
    spaces: dict


class No(NamedTuple):
    reason: int


class Busy(NamedTuple):
    pass


class End(NamedTuple):
    pass


class Candidate(NamedTuple):
    token: bytes
    embedding: np.ndarray
    duration: float
    versions: dict
    chromaprint: bytes


class Offer(NamedTuple):
    candidates: tuple


class Fetch(NamedTuple):
    token: bytes
    wants: tuple
    spaces: dict


class Output(NamedTuple):
    space: bytes
    version: str
    value: object


class Outputs(NamedTuple):
    items: dict


class PeerEntry(NamedTuple):
    peer_id: bytes
    addrs: tuple


class Peers(NamedTuple):
    peers: tuple


class Hello(NamedTuple):
    app_version: str
    spaces: dict
    issued_at: int


def encode_uvarint(value):
    value = int(value)
    if value < 0:
        raise ValueError('uvarint cannot encode a negative value')
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def bits_of(keys):
    mask = 0
    for key in keys:
        mask |= OUTPUT_BITS[key]
    return mask


def keys_of(mask):
    return tuple(key for key in OUTPUT_KEYS if mask & OUTPUT_BITS[key])


def parse_version(text):
    if not isinstance(text, str):
        return None
    match = _VERSION_RE.match(text.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def version_allowed(text):
    parsed = parse_version(text)
    return parsed is not None and parsed >= MIN_OUTPUT_VERSION


def version_text_ok(text):
    return isinstance(text, str) and bool(_VERSION_TEXT_RE.match(text))


def version_passes(text):
    return version_text_ok(text) and version_allowed(text)


def _frame(frame_type, payload):
    body = bytes(payload)
    if len(body) > FRAME_CAPS[frame_type]:
        raise ValueError(f'frame type {frame_type} payload of {len(body)} bytes exceeds its cap')
    return _HEADER.pack(WIRE_VERSION, frame_type) + encode_uvarint(len(body)) + body


def _space(value):
    raw = bytes(value)
    if len(raw) != SPACE_ID_BYTES:
        raise ValueError('a space id is exactly 32 bytes')
    return raw


def _floats(vector, dim):
    array = np.ascontiguousarray(np.asarray(vector, dtype='<f4').reshape(-1))
    if array.size != dim:
        raise ValueError(f'expected {dim} floats, got {array.size}')
    return array.tobytes()


def _short_text(text):
    raw = str(text).encode('ascii')
    if not raw or len(raw) > MAX_VERSION_CHARS:
        raise ValueError('a version text is 1 to 32 ASCII characters')
    return bytes([len(raw)]) + raw


def _blob(value, cap):
    raw = bytes(value or b'')
    if len(raw) > cap:
        raise ValueError(f'blob of {len(raw)} bytes exceeds its cap of {cap}')
    return encode_uvarint(len(raw)) + raw


def encode_query(signature, duration, wants, spaces):
    signature = bytes(signature)
    if len(signature) != SIGNATURE_BYTES:
        raise ValueError('a signature is exactly 25 bytes')
    keys = tuple(key for key in OUTPUT_KEYS if key in set(wants))
    body = bytearray(signature)
    body += _F64.pack(float(duration))
    body.append(bits_of(keys))
    body += _space(spaces[OUT_MUSICNN])
    for key in keys:
        body += _space(spaces[key])
    return _frame(T_QUERY, body)


def encode_no(reason):
    return _frame(T_NO, bytes([int(reason)]))


def encode_busy():
    return _frame(T_BUSY, b'')


def encode_end():
    return _frame(T_END, b'')


def encode_offer(candidates):
    candidates = list(candidates)
    if not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError('an offer carries one or two candidates')
    body = bytearray([len(candidates)])
    for candidate in candidates:
        token = bytes(candidate.token)
        if len(token) != TOKEN_BYTES:
            raise ValueError('a token is exactly 16 bytes')
        available = tuple(key for key in OUTPUT_KEYS if key in candidate.versions)
        body += token
        body += _floats(candidate.embedding, MUSICNN_DIM)
        body += _F64.pack(float(candidate.duration))
        body.append(bits_of(available))
        body += _short_text(candidate.versions[OUT_MUSICNN])
        for key in available:
            body += _short_text(candidate.versions[key])
        body += _blob(candidate.chromaprint, MAX_CHROMAPRINT_BYTES)
    return _frame(T_OFFER, body)


def encode_fetch(token, wants, spaces):
    token = bytes(token)
    if len(token) != TOKEN_BYTES:
        raise ValueError('a token is exactly 16 bytes')
    keys = tuple(key for key in OUTPUT_KEYS if key in set(wants))
    body = bytearray(token)
    body.append(bits_of(keys))
    for key in keys:
        body += _space(spaces[key])
    return _frame(T_FETCH, body)


def encode_outputs(items):
    keys = tuple(key for key in OUTPUT_KEYS if key in items)
    body = bytearray([bits_of(keys)])
    for key in keys:
        item = items[key]
        body += _space(item.space)
        body += _short_text(item.version)
        if key == OUT_CLAP:
            body += _floats(item.value, CLAP_DIM)
        elif key == OUT_LYRICS:
            vector, axis = item.value
            body += _floats(vector, LYRICS_DIM)
            body += _floats(axis, AXIS_DIM)
        else:
            body += _blob(item.value, MAX_NEURAL_BYTES)
    return _frame(T_OUTPUTS, body)


def encode_peers(peers):
    peers = list(peers)[:MAX_PEERS_LISTED]
    body = bytearray([len(peers)])
    for entry in peers:
        peer_id = bytes(entry.peer_id)
        if not peer_id or len(peer_id) > MAX_PEER_ID_BYTES:
            raise ValueError('a peer id is 1 to 64 bytes')
        addrs = [bytes(addr) for addr in entry.addrs if 0 < len(bytes(addr)) <= MAX_ADDR_BYTES]
        addrs = addrs[:MAX_ADDRS_PER_PEER]
        body += encode_uvarint(len(peer_id)) + peer_id
        body.append(len(addrs))
        for addr in addrs:
            body += encode_uvarint(len(addr)) + addr
    return _frame(T_PEERS, body)


def encode_hello(app_version, spaces, issued_at):
    body = bytearray(_short_text(app_version))
    present = [key for key in SPACE_KEYS if spaces.get(key)]
    body.append(len(present))
    for key in present:
        body.append(SPACE_KEYS.index(key))
        body += _space(spaces[key])
    body += _U64.pack(int(issued_at))
    return _frame(T_HELLO, body)


class _Cursor:
    def __init__(self, data):
        self._data = memoryview(bytes(data))
        self._at = 0

    def take(self, count):
        count = int(count)
        if count < 0 or self._at + count > len(self._data):
            raise Violation('the frame ends before its declared content')
        chunk = self._data[self._at:self._at + count].tobytes()
        self._at += count
        return chunk

    def u8(self):
        return self.take(1)[0]

    def f64(self):
        return _F64.unpack(self.take(8))[0]

    def u64(self):
        return _U64.unpack(self.take(8))[0]

    def uvarint(self, cap):
        value = 0
        for index in range(UVARINT_MAX_BYTES):
            byte = self.u8()
            value |= (byte & 0x7F) << (7 * index)
            if not byte & 0x80:
                if value > cap:
                    raise Violation(f'a length of {value} exceeds its cap of {cap}')
                return value
        raise Violation('a uvarint is longer than its cap allows')

    def floats(self, dim):
        return np.frombuffer(self.take(dim * 4), dtype='<f4', count=dim).astype(np.float32)

    def text(self):
        size = self.u8()
        if size < 1 or size > MAX_VERSION_CHARS:
            raise Violation('a version text has an invalid length')
        raw = self.take(size)
        try:
            return raw.decode('ascii')
        except UnicodeDecodeError:
            return ''

    def mask(self):
        value = self.u8()
        if value & ~ALL_BITS:
            raise Violation('the output bitmask sets an unknown bit')
        return value

    def done(self):
        if self._at != len(self._data):
            raise Violation('the frame carries trailing bytes')


def _finite_duration(value):
    if not np.isfinite(value) or value <= 0:
        raise Violation('a duration must be a positive finite number')
    return float(value)


def _decode_query(cursor):
    signature = cursor.take(SIGNATURE_BYTES)
    duration = _finite_duration(cursor.f64())
    keys = keys_of(cursor.mask())
    spaces = {OUT_MUSICNN: cursor.take(SPACE_ID_BYTES)}
    for key in keys:
        spaces[key] = cursor.take(SPACE_ID_BYTES)
    return Query(signature, duration, keys, spaces)


def _decode_no(cursor):
    reason = cursor.u8()
    return No(reason if reason in NO_REASONS else NO_NOT_FOUND)


def _decode_offer(cursor):
    count = cursor.u8()
    if not 1 <= count <= MAX_CANDIDATES:
        raise Violation('an offer carries one or two candidates')
    candidates = []
    for _ in range(count):
        token = cursor.take(TOKEN_BYTES)
        embedding = cursor.floats(MUSICNN_DIM)
        duration = _finite_duration(cursor.f64())
        keys = keys_of(cursor.mask())
        versions = {OUT_MUSICNN: cursor.text()}
        for key in keys:
            versions[key] = cursor.text()
        chromaprint = cursor.take(cursor.uvarint(MAX_CHROMAPRINT_BYTES))
        candidates.append(Candidate(token, embedding, duration, versions, chromaprint))
    return Offer(tuple(candidates))


def _decode_fetch(cursor):
    token = cursor.take(TOKEN_BYTES)
    keys = keys_of(cursor.mask())
    spaces = {key: cursor.take(SPACE_ID_BYTES) for key in keys}
    return Fetch(token, keys, spaces)


def _decode_outputs(cursor):
    keys = keys_of(cursor.mask())
    items = {}
    for key in keys:
        space = cursor.take(SPACE_ID_BYTES)
        version = cursor.text()
        if key == OUT_CLAP:
            value = cursor.floats(CLAP_DIM)
        elif key == OUT_LYRICS:
            value = (cursor.floats(LYRICS_DIM), cursor.floats(AXIS_DIM))
        else:
            value = cursor.take(cursor.uvarint(MAX_NEURAL_BYTES))
        items[key] = Output(space, version, value)
    return Outputs(items)


def _decode_peers(cursor):
    count = cursor.u8()
    if count > MAX_PEERS_LISTED:
        raise Violation('a PEERS frame lists too many peers')
    peers = []
    for _ in range(count):
        size = cursor.uvarint(MAX_PEER_ID_BYTES)
        if size < 1:
            raise Violation('a listed peer has an empty id')
        peer_id = cursor.take(size)
        addr_count = cursor.u8()
        if addr_count > MAX_ADDRS_PER_PEER:
            raise Violation('a listed peer has too many addresses')
        addrs = []
        for _ in range(addr_count):
            addr_size = cursor.uvarint(MAX_ADDR_BYTES)
            if addr_size < 1:
                raise Violation('a listed address is empty')
            addrs.append(cursor.take(addr_size))
        peers.append(PeerEntry(peer_id, tuple(addrs)))
    return Peers(tuple(peers))


def _decode_hello(cursor):
    app_version = cursor.text()
    count = cursor.u8()
    if count > len(SPACE_KEYS):
        raise Violation('a HELLO lists too many spaces')
    spaces = {}
    for _ in range(count):
        kind = cursor.u8()
        space = cursor.take(SPACE_ID_BYTES)
        if kind < len(SPACE_KEYS):
            spaces[SPACE_KEYS[kind]] = space
    return Hello(app_version, spaces, cursor.u64())


_DECODERS = {
    T_QUERY: _decode_query,
    T_NO: _decode_no,
    T_OFFER: _decode_offer,
    T_FETCH: _decode_fetch,
    T_OUTPUTS: _decode_outputs,
    T_BUSY: lambda cursor: Busy(),
    T_END: lambda cursor: End(),
    T_PEERS: _decode_peers,
    T_HELLO: _decode_hello,
}


def decode_payload(frame_type, payload):
    decoder = _DECODERS.get(frame_type)
    if decoder is None:
        raise UnknownType(f'unknown frame type {frame_type}')
    cursor = _Cursor(payload)
    try:
        message = decoder(cursor)
    except (struct.error, ValueError) as exc:
        raise Violation(f'malformed frame of type {frame_type}: {exc}') from exc
    cursor.done()
    return message


def check_header(version, frame_type, allowed=STREAM_TYPES):
    if version != WIRE_VERSION:
        raise UnknownVersion(f'wire version {version} is not supported')
    if frame_type not in allowed:
        raise UnknownType(f'frame type {frame_type} is not expected here')


async def _read_uvarint(reader, cap):
    value = 0
    for index in range(UVARINT_MAX_BYTES):
        byte = (await reader.read_exactly(1))[0]
        value |= (byte & 0x7F) << (7 * index)
        if not byte & 0x80:
            if value > cap:
                raise Violation(f'a frame length of {value} exceeds its cap of {cap}')
            return value
    raise Violation('a frame length is longer than its cap allows')


async def read_frame(reader, allowed=STREAM_TYPES):
    version, frame_type = _HEADER.unpack(await reader.read_exactly(_HEADER.size))
    check_header(version, frame_type, allowed)
    size = await _read_uvarint(reader, FRAME_CAPS[frame_type])
    payload = await reader.read_exactly(size) if size else b''
    return frame_type, decode_payload(frame_type, payload)


def decode_hello(data):
    raw = bytes(data)
    if len(raw) < _HEADER.size + 1:
        raise Violation('a HELLO is too short')
    version, frame_type = _HEADER.unpack_from(raw)
    check_header(version, frame_type, frozenset({T_HELLO}))
    cursor = _Cursor(raw[_HEADER.size:])
    size = cursor.uvarint(FRAME_CAPS[T_HELLO])
    payload = cursor.take(size)
    cursor.done()
    return decode_payload(T_HELLO, payload)
