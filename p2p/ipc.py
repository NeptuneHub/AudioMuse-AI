# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The contract between the analysis worker, Flask and the P2P node.

They talk through Postgres NOTIFY (the same broker the task queue uses), which
caps one payload at 8000 bytes, and through a few app_config runtime rows. This
module is the only definition of the channel names, message shapes, keys and
states; it imports nothing heavier than numpy so every process can use it.

A supply request names one catalogue item and the outputs the worker still
needs; the node answers on the worker's private reply channel with the peer
results base64-encoded and split into parts that each fit one NOTIFY. The
worker is the only writer of analysis data, so a reply that arrives after the
worker stopped waiting is simply never read.

Main Features:
* channel names, the node's application_name and the runtime app_config keys
* MIN/MAX_REQUEST_TTL_SECONDS bound how long a worker may ask the node to try
  (P2P_LOOKUP_TIMEOUT_SECONDS); the wizard and the node use the same limits
* supply_request / supply_reply builders, dumps and parse
* split_reply and PartAssembler turn one reply into NOTIFY-sized parts and back;
  the assembler keeps parts only for request ids it was told to expect, so a
  late or foreign reply never piles up in memory
* encode_floats / decode_floats / encode_bytes / decode_bytes for vectors and
  neural fingerprint blobs
* read_status reads the node status row with its age measured by the database
  clock, so a status older than STATUS_FRESH_SECONDS reads as not running
"""

import base64
import binascii
import json
import uuid

import numpy as np

PROTOCOL_VERSION = 1
NODE_CHANNEL = 'audiomuse_p2p'
REPLY_CHANNEL_PREFIX = 'audiomuse_p2p_r_'
MAX_PAYLOAD_BYTES = 7900
MIN_REQUEST_TTL_SECONDS = 1.0
MAX_REQUEST_TTL_SECONDS = 120.0
_PART_OVERHEAD = 200
NODE_APP_NAME = 'audiomuse-p2p-node'

STATUS_KEY = 'p2p_status'
ONION_KEY_KEY = 'p2p_onion_key'
PEERS_KEY = 'p2p_peers'
STATUS_FRESH_SECONDS = 60

STATE_OFF = 'off'
STATE_UNAVAILABLE = 'unavailable'
STATE_STARTING = 'starting'
STATE_CONNECTED = 'connected'
STATE_NO_PEERS = 'no_peers'
STATE_ERROR = 'error'
STATE_NOT_RUNNING = 'not_running'
STATE_LABELS = {
    STATE_OFF: 'Off',
    STATE_UNAVAILABLE: 'Unavailable',
    STATE_STARTING: 'Starting',
    STATE_CONNECTED: 'Connected',
    STATE_NO_PEERS: 'No peers found',
    STATE_ERROR: 'Error',
    STATE_NOT_RUNNING: 'Not running',
}

OUT_MUSICNN = 'm'
OUT_CLAP = 'c'
OUT_LYRICS = 'l'
OUT_AXIS = 'a'
OUT_NEURAL = 'n'
SUPPLIABLE_OUTPUTS = (OUT_CLAP, OUT_LYRICS, OUT_NEURAL)

OP_SUPPLY = 'supply'
OP_SUPPLIED = 'supplied'
OP_PART = 'part'

REPLY_OK = 'ok'
REPLY_NONE = 'none'
REPLY_NO_PEERS = 'no_peers'
REPLY_BUSY = 'busy'
REPLY_ERROR = 'error'


def new_request_id():
    return uuid.uuid4().hex


def new_reply_channel():
    return REPLY_CHANNEL_PREFIX + uuid.uuid4().hex[:16]


def supply_request(rid, item_id, server_id, provider_id, wants, reply_channel, ttl):
    return {
        'v': PROTOCOL_VERSION,
        'op': OP_SUPPLY,
        'rid': rid,
        'item': str(item_id),
        'server': None if server_id is None else str(server_id),
        'provider': None if provider_id is None else str(provider_id),
        'wants': [key for key in SUPPLIABLE_OUTPUTS if key in set(wants)],
        'reply': reply_channel,
        'ttl': float(ttl),
    }


def supply_reply(rid, status, peer=None, versions=None, outputs=None):
    return {
        'v': PROTOCOL_VERSION,
        'op': OP_SUPPLIED,
        'rid': rid,
        'status': status,
        'peer': peer,
        'versions': dict(versions or {}),
        'outputs': dict(outputs or {}),
    }


def dumps(message):
    return json.dumps(message, separators=(',', ':'), ensure_ascii=True)


def parse(payload):
    try:
        message = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(message, dict) or message.get('v') != PROTOCOL_VERSION:
        return None
    if not isinstance(message.get('op'), str):
        return None
    return message


def split_reply(reply):
    body = base64.b64encode(dumps(reply).encode('ascii')).decode('ascii')
    size = MAX_PAYLOAD_BYTES - _PART_OVERHEAD
    chunks = [body[start:start + size] for start in range(0, len(body), size)] or ['']
    return [
        dumps({'v': PROTOCOL_VERSION, 'op': OP_PART, 'rid': reply['rid'], 'i': index, 'n': len(chunks), 'd': chunk})
        for index, chunk in enumerate(chunks)
    ]


class PartAssembler:
    def __init__(self):
        self._parts = {}

    def expect(self, rid):
        if isinstance(rid, str) and rid not in self._parts:
            self._parts[rid] = None

    def add(self, message):
        if not message or message.get('op') != OP_PART:
            return None
        rid, index, total = message.get('rid'), message.get('i'), message.get('n')
        if not isinstance(rid, str) or rid not in self._parts:
            return None
        if not isinstance(index, int) or not isinstance(total, int):
            return None
        if total < 1 or not 0 <= index < total or not isinstance(message.get('d'), str):
            return None
        slot = self._parts[rid]
        if slot is None:
            slot = self._parts[rid] = {'n': total, 'parts': {}}
        if slot['n'] != total:
            self._parts.pop(rid, None)
            return None
        slot['parts'][index] = message['d']
        if len(slot['parts']) < total:
            return None
        self._parts.pop(rid, None)
        try:
            text = base64.b64decode(''.join(slot['parts'][i] for i in range(total)), validate=True)
            reply = json.loads(text.decode('ascii'))
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return None
        if not isinstance(reply, dict) or reply.get('op') != OP_SUPPLIED or reply.get('rid') != rid:
            return None
        return reply

    def discard(self, rid):
        self._parts.pop(rid, None)


def encode_floats(vector):
    array = np.ascontiguousarray(np.asarray(vector, dtype='<f4').reshape(-1))
    return base64.b64encode(array.tobytes()).decode('ascii')


def decode_floats(text, dim):
    raw = decode_bytes(text)
    if raw is None or len(raw) != int(dim) * 4:
        return None
    return np.frombuffer(raw, dtype='<f4', count=int(dim)).astype(np.float32)


def encode_bytes(blob):
    return base64.b64encode(bytes(blob)).decode('ascii')


def decode_bytes(text):
    if not isinstance(text, str):
        return None
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return None


def read_status(cur):
    cur.execute(
        "SELECT value, EXTRACT(EPOCH FROM (now() - updated_at)) FROM app_config WHERE key = %s",
        (STATUS_KEY,),
    )
    row = cur.fetchone()
    if not row:
        return {'state': STATE_NOT_RUNNING}
    try:
        status = json.loads(row[0])
    except (TypeError, ValueError):
        return {'state': STATE_NOT_RUNNING}
    if not isinstance(status, dict):
        return {'state': STATE_NOT_RUNNING}
    age = float(row[1]) if row[1] is not None else None
    if status.get('state') not in (STATE_OFF, STATE_UNAVAILABLE) and (age is None or age > STATUS_FRESH_SECONDS):
        status = dict(status, state=STATE_NOT_RUNNING)
    return status


def status_label(state):
    return STATE_LABELS.get(state, STATE_LABELS[STATE_NOT_RUNNING])
