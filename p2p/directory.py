# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The directory where AudioMuse-AI servers find each other: public Nostr relays.

Nobody keeps a list of members: no AudioMuse-AI server and no project server.
Each member publishes one small signed record on a few public Nostr relays run
by third parties (P2P_DIRECTORY_RELAYS), always through Tor, and reads the
records of the last day from the same relays. A record is addressable (kind
34101 with the d tag "audiomuse-ai-p2p"), so every member has exactly one,
replaced each time it publishes. It holds public data only: the onion address,
the AudioMuse-AI version and the model space ids. The relays never see an IP
address, a song or a result.

Main Features:
* signing_key: derived from the onion service key, so a member keeps one
  identity; events are signed with BIP340 Schnorr (coincurve)
* build_event / parse_record: the record and its checks (kind, age, a valid
  onion address, a readable version, 32-byte space ids); anything else is
  skipped
* verified: the event id and the Schnorr signature must match, so a relay
  cannot forge a newer record in another server's name and hide it
* publish: the record to every relay, best effort, each relay on its own Tor
  connection with its own time limit; fetch: the fresh, verified records of
  every relay, newest per author
* relay_names parses P2P_DIRECTORY_RELAYS
"""

import base64
import hashlib
import json
import logging
import re
import ssl
import time
import uuid
from contextlib import asynccontextmanager
from typing import NamedTuple

import trio
from coincurve import PrivateKey, PublicKeyXOnly
from trio_websocket import wrap_client_stream

import config
from p2p import tor as p2p_tor, wire

logger = logging.getLogger(__name__)

KIND = 34101
TAG = 'audiomuse-ai-p2p'
RECORD_MAX_AGE_SECONDS = 24 * 3600
FUTURE_SKEW_SECONDS = 600
RELAY_SECONDS = 60.0
FETCH_LIMIT = 500
MAX_CONTENT_CHARS = 4096
HTTPS_PORT = 443
KEY_DOMAIN = b'audiomuse-ai-p2p-directory'
_RELAY_RE = re.compile(r'^[A-Za-z0-9.-]{1,253}$')
_SPLIT_RE = re.compile(r'[,\s]+')
_SPACE_RE = re.compile(r'^[0-9a-f]{64}$')


class Record(NamedTuple):
    onion: str
    version: str
    spaces: dict
    created_at: int


def failure_reason(exc):
    while getattr(exc, 'exceptions', None):
        exc = exc.exceptions[0]
    status = getattr(exc, 'status_code', None)
    if isinstance(status, int):
        return f'{type(exc).__name__}: HTTP {status}'
    return f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__


def relay_names(text=None):
    raw = config.P2P_DIRECTORY_RELAYS if text is None else text
    return [name for name in _SPLIT_RE.split(str(raw or '')) if name and _RELAY_RE.match(name)]


def signing_key(onion_key_content):
    return PrivateKey(hashlib.sha256(KEY_DOMAIN + base64.b64decode(onion_key_content)).digest())


def record_content(onion, version, spaces):
    present = {key: bytes(value).hex() for key, value in spaces.items() if value}
    return json.dumps({'onion': onion, 'v': version, 's': present}, separators=(',', ':'), sort_keys=True)


def event_id(pubkey, created_at, kind, tags, content):
    body = json.dumps([0, pubkey, created_at, kind, tags, content], separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def verified(event):
    try:
        expected = event_id(event['pubkey'], event['created_at'], event['kind'], event['tags'], event['content'])
        if expected != event.get('id'):
            return False
        public = PublicKeyXOnly(bytes.fromhex(event['pubkey']))
        return bool(public.verify(bytes.fromhex(event['sig']), bytes.fromhex(expected)))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def build_event(key, onion, version, spaces, created_at=None):
    pubkey = key.public_key_xonly.format().hex()
    created = int(time.time() if created_at is None else created_at)
    tags = [['d', TAG], ['t', TAG]]
    content = record_content(onion, version, spaces)
    event_id_hex = event_id(pubkey, created, KIND, tags, content)
    return {
        'id': event_id_hex, 'pubkey': pubkey, 'created_at': created, 'kind': KIND, 'tags': tags,
        'content': content, 'sig': key.sign_schnorr(bytes.fromhex(event_id_hex)).hex(),
    }


def parse_record(event, now=None):
    now = time.time() if now is None else now
    try:
        if event.get('kind') != KIND:
            return None
        created = int(event.get('created_at'))
        if created < now - RECORD_MAX_AGE_SECONDS or created > now + FUTURE_SKEW_SECONDS:
            return None
        content = event.get('content')
        if not isinstance(content, str) or len(content) > MAX_CONTENT_CHARS:
            return None
        data = json.loads(content)
        onion = data.get('onion')
        version = data.get('v')
        raw_spaces = data.get('s') or {}
    except (AttributeError, TypeError, ValueError):
        return None
    if not p2p_tor.valid_onion(onion) or not wire.version_text_ok(version) or not isinstance(raw_spaces, dict):
        return None
    spaces = {
        key: bytes.fromhex(value) for key, value in raw_spaces.items()
        if key in wire.SPACE_KEYS and isinstance(value, str) and _SPACE_RE.match(value)
    }
    return Record(onion, version, spaces, created)


def newest_records(events, now=None):
    newest = {}
    for event in events:
        if not isinstance(event, dict) or not verified(event):
            continue
        record = parse_record(event, now)
        author = event.get('pubkey')
        if record is None or not isinstance(author, str):
            continue
        current = newest.get(author)
        if current is None or current.created_at < record.created_at:
            newest[author] = record
    return list(newest.values())


@asynccontextmanager
async def _relay(socks_port, host):
    raw = await p2p_tor.connect(socks_port, host, HTTPS_PORT)
    tls = trio.SSLStream(raw, ssl.create_default_context(), server_hostname=host)
    await tls.do_handshake()
    async with trio.open_nursery() as nursery:
        socket = await wrap_client_stream(nursery, tls, host, '/')
        try:
            yield socket
        finally:
            with trio.CancelScope(shield=True):
                with trio.move_on_after(2):
                    await socket.aclose()
            nursery.cancel_scope.cancel()


async def _publish_one(socks_port, host, event, results):
    try:
        with trio.fail_after(RELAY_SECONDS):
            async with _relay(socks_port, host) as socket:
                await socket.send_message(json.dumps(['EVENT', event]))
                while True:
                    message = json.loads(await socket.get_message())
                    if isinstance(message, list) and len(message) >= 3 and message[0] == 'OK' and message[1] == event['id']:
                        results[host] = bool(message[2])
                        if not message[2]:
                            logger.warning("Directory relay %s refused this server's record: %s", host,
                                           message[3] if len(message) > 3 else '')
                        return
    except Exception as exc:
        if host in results:
            logger.debug("The directory relay %s closed untidily after answering", host, exc_info=True)
            return
        results[host] = False
        logger.warning("Could not publish this server's record on the directory relay %s: %s", host, failure_reason(exc))


async def _fetch_one(socks_port, host, since, events):
    subscription = uuid.uuid4().hex[:16]
    answered = []
    try:
        with trio.fail_after(RELAY_SECONDS):
            async with _relay(socks_port, host) as socket:
                await socket.send_message(json.dumps([
                    'REQ', subscription, {'kinds': [KIND], '#d': [TAG], 'since': int(since), 'limit': FETCH_LIMIT},
                ]))
                while True:
                    message = json.loads(await socket.get_message())
                    if not isinstance(message, list) or len(message) < 2 or message[1] != subscription:
                        continue
                    if message[0] == 'EVENT' and len(message) >= 3:
                        events.append(message[2])
                    elif message[0] in ('EOSE', 'CLOSED'):
                        answered.append(True)
                        await socket.send_message(json.dumps(['CLOSE', subscription]))
                        return
    except Exception as exc:
        if answered:
            logger.debug("The directory relay %s closed untidily after answering", host, exc_info=True)
            return
        logger.warning("Could not read the directory relay %s: %s", host, failure_reason(exc))


async def publish(socks_port, relays, event):
    results = {}
    async with trio.open_nursery() as nursery:
        for host in relays:
            nursery.start_soon(_publish_one, socks_port, host, event, results)
    return results


async def fetch(socks_port, relays, now=None):
    now = time.time() if now is None else now
    events = []
    async with trio.open_nursery() as nursery:
        for host in relays:
            nursery.start_soon(_fetch_one, socks_port, host, now - RECORD_MAX_AGE_SECONDS, events)
    return newest_records(events, now)
