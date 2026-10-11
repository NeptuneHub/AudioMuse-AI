# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The Nostr directory where AudioMuse-AI servers find each other (p2p.directory).

The records are built and read here exactly as a Nostr relay sees them, and
the relay connection itself is replaced by a fake websocket, so no Tor, no
TLS and no network is involved.

Main Features:
* build_event: the id is the sha256 of the NIP-01 serialization, the BIP340
  Schnorr signature verifies with the x-only public key (coincurve), the
  record carries only the onion address, the version and the present space
  ids, under kind 34101 with the d and t tags
* signing_key is derived from the onion service key, the same for the same
  key and domain separated from a plain hash of it
* parse_record accepts a good record (the age and clock skew bounds
  included) and rejects a wrong kind, a record older than a day or from the
  future, a bad onion address, an unreadable version, oversize or broken
  content; malformed space ids are dropped one by one
* verified checks the event id and the Schnorr signature: any changed,
  broken or missing field fails
* newest_records keeps the newest verified record of every author, so a
  forged newer record in another server's name or a record changed after
  signing is ignored
* relay_names keeps host names and drops URLs, junk and overlong names
* failure_reason names the type and message of the first failure inside any
  exception group, the type alone when the message is empty, and only the HTTP
  status of a rejected websocket handshake
* publish sends the record to every relay: an OK is True, a refusal, a relay
  that raises or never answers is False with a WARNING naming the reason; a
  relay that answered and then closed untidily keeps its answer, no WARNING
* fetch sends one REQ per relay, keeps only the events of its subscription,
  closes it at EOSE or CLOSED, logs a failing relay (not one that answered
  and then closed untidily), and returns the newest record per author across
  relays
"""

import base64
import hashlib
import json
import logging
from contextlib import asynccontextmanager

import pytest
import trio
from coincurve import PublicKeyXOnly
from trio_websocket import ConnectionRejected

import config
from p2p import directory, wire
from p2p import tor as p2p_tor

SPACES = {key: bytes([index + 1]) * 32 for index, key in enumerate(wire.SPACE_KEYS)}
ONION_KEY = base64.b64encode(bytes(range(64))).decode('ascii')
OTHER_KEY = base64.b64encode(bytes(range(64, 128))).decode('ascii')
ONION = 'directory' + 'd' * 47
OTHER_ONION = 'directory' + 'e' * 47
NOW = 1_790_000_000
DAY = directory.RECORD_MAX_AGE_SECONDS


def _event(created=NOW - 60, key=ONION_KEY, onion=ONION, version='v3.6.3', spaces=SPACES):
    return directory.build_event(directory.signing_key(key), onion, version, spaces, created_at=created)


def _raw(content, created=NOW - 60, kind=directory.KIND, pubkey='ab' * 32):
    return {'id': '00' * 32, 'pubkey': pubkey, 'created_at': created, 'kind': kind,
            'tags': [['d', directory.TAG]], 'content': content, 'sig': '00' * 64}


def _content(**fields):
    base = {'onion': ONION, 'v': 'v3.6.3', 's': {'m': 'ab' * 32}}
    base.update(fields)
    return json.dumps(base)


class TestBuildEvent:
    def test_the_id_is_the_hash_of_the_nip01_serialization(self):
        event = _event(created=NOW)
        serial = json.dumps(
            [0, event['pubkey'], NOW, directory.KIND, [['d', directory.TAG], ['t', directory.TAG]], event['content']],
            separators=(',', ':'), ensure_ascii=False,
        )

        assert event['id'] == hashlib.sha256(serial.encode('utf-8')).hexdigest()
        assert set(event) == {'id', 'pubkey', 'created_at', 'kind', 'tags', 'content', 'sig'}
        assert (event['kind'], event['created_at']) == (34101, NOW)
        assert event['tags'] == [['d', 'audiomuse-ai-p2p'], ['t', 'audiomuse-ai-p2p']]

    def test_the_schnorr_signature_verifies_with_the_x_only_key(self):
        event = _event()
        public = PublicKeyXOnly(bytes.fromhex(event['pubkey']))

        assert len(event['pubkey']) == 64 and len(event['sig']) == 128
        assert public.verify(bytes.fromhex(event['sig']), bytes.fromhex(event['id']))

    def test_a_changed_record_no_longer_matches_its_signature(self):
        event = _event()
        forged = json.loads(event['content'])
        forged['onion'] = OTHER_ONION
        content = json.dumps(forged, separators=(',', ':'), sort_keys=True)
        serial = json.dumps([0, event['pubkey'], event['created_at'], event['kind'], event['tags'], content],
                            separators=(',', ':'), ensure_ascii=False)
        forged_id = hashlib.sha256(serial.encode('utf-8')).digest()

        assert not PublicKeyXOnly(bytes.fromhex(event['pubkey'])).verify(bytes.fromhex(event['sig']), forged_id)

    def test_the_record_holds_only_public_data(self):
        event = _event(spaces=dict(SPACES, l=None))

        assert json.loads(event['content']) == {
            'onion': ONION, 'v': 'v3.6.3', 's': {key: value.hex() for key, value in SPACES.items() if key != 'l'},
        }
        assert event['content'] == json.dumps(json.loads(event['content']), separators=(',', ':'), sort_keys=True)

    def test_the_current_time_is_the_default(self, monkeypatch):
        monkeypatch.setattr(directory.time, 'time', lambda: 1234.9)

        assert directory.build_event(directory.signing_key(ONION_KEY), ONION, 'v3.6.3', SPACES)['created_at'] == 1234


class TestSigningKey:
    def test_the_same_onion_key_gives_the_same_identity(self):
        assert directory.signing_key(ONION_KEY).secret == directory.signing_key(ONION_KEY).secret
        assert _event()['pubkey'] == _event(created=NOW - 5)['pubkey']

    def test_another_onion_key_gives_another_identity(self):
        assert directory.signing_key(ONION_KEY).secret != directory.signing_key(OTHER_KEY).secret
        assert _event()['pubkey'] != _event(key=OTHER_KEY)['pubkey']

    def test_the_key_is_domain_separated(self):
        raw = base64.b64decode(ONION_KEY)
        secret = directory.signing_key(ONION_KEY).secret

        assert secret == hashlib.sha256(b'audiomuse-ai-p2p-directory' + raw).digest()
        assert secret != hashlib.sha256(raw).digest()


class TestParseRecord:
    def test_a_good_record_is_read(self):
        record = directory.parse_record(_event(created=NOW), now=NOW)

        assert record == directory.Record(ONION, 'v3.6.3', SPACES, NOW)

    @pytest.mark.parametrize('created, kept', [
        (NOW - DAY, True), (NOW - DAY - 1, False),
        (NOW + directory.FUTURE_SKEW_SECONDS, True), (NOW + directory.FUTURE_SKEW_SECONDS + 1, False),
    ])
    def test_the_age_and_clock_skew_bounds(self, created, kept):
        assert (directory.parse_record(_event(created=created), now=NOW) is not None) is kept

    @pytest.mark.parametrize('event', [
        _raw(_content(), kind=1),
        _raw(_content(), kind='34101'),
        _raw(_content(), created='soon'),
        _raw(_content(), created=None),
        _raw({'onion': ONION}),
        _raw(_content(padding='a' * directory.MAX_CONTENT_CHARS)),
        _raw('{not json'),
        _raw('[1, 2]'),
        _raw('"text"'),
        _raw(_content(onion='x')),
        _raw(_content(onion=ONION + '.onion')),
        _raw(_content(onion=ONION.upper())),
        _raw(_content(onion=None)),
        _raw(_content(v='')),
        _raw(_content(v='v3.6.3 beta')),
        _raw(_content(v='v' * 33)),
        _raw(_content(v=None)),
        _raw(_content(v=3)),
        _raw(_content(s=['m'])),
        None,
        [],
        'event',
    ])
    def test_anything_else_is_skipped(self, event):
        assert directory.parse_record(event, now=NOW) is None

    def test_malformed_space_ids_are_dropped_one_by_one(self):
        spaces = {'m': 'zz' * 32, 'c': 'ab' * 32, 'x': 'ab' * 32, 'l': 'AB' * 32, 'n': 5}

        record = directory.parse_record(_raw(_content(s=spaces)), now=NOW)

        assert record.spaces == {'c': bytes.fromhex('ab' * 32)}

    @pytest.mark.parametrize('spaces', [{}, [], None])
    def test_a_record_without_space_ids_is_still_a_member(self, spaces):
        record = directory.parse_record(_raw(_content(s=spaces)), now=NOW)

        assert record.onion == ONION and record.spaces == {}

    def test_a_short_space_id_is_dropped(self):
        record = directory.parse_record(_raw(_content(s={'m': 'ab' * 31})), now=NOW)

        assert record.spaces == {}


class TestNewestRecords:
    def test_the_newest_record_of_every_author_wins(self):
        events = [
            _event(created=NOW - 10, onion=OTHER_ONION),
            _event(created=NOW - 100, onion=ONION),
            _event(created=NOW - 50, key=OTHER_KEY, onion=ONION),
        ]

        records = directory.newest_records(events, now=NOW)

        assert sorted((record.onion, record.created_at) for record in records) == [
            (ONION, NOW - 50), (OTHER_ONION, NOW - 10),
        ]

    def test_malformed_events_and_events_without_an_author_are_skipped(self):
        anonymous = _event()
        anonymous.pop('pubkey')
        numbered = dict(_event(key=OTHER_KEY), pubkey=7)

        records = directory.newest_records([None, 'event', anonymous, numbered, _raw('{bad')], now=NOW)

        assert records == []

    def test_a_forged_newer_record_in_another_servers_name_is_ignored(self):
        genuine = _event(created=NOW - 100, onion=ONION)
        forged = dict(_event(created=NOW - 10, key=OTHER_KEY, onion=OTHER_ONION), pubkey=genuine['pubkey'])

        records = directory.newest_records([genuine, forged], now=NOW)

        assert [(record.onion, record.created_at) for record in records] == [(ONION, NOW - 100)]

    def test_a_record_changed_after_signing_is_dropped(self):
        tampered = dict(_event(), content=directory.record_content(OTHER_ONION, 'v3.6.3', SPACES))

        assert directory.newest_records([tampered], now=NOW) == []


class TestVerified:
    def test_a_built_event_verifies(self):
        assert directory.verified(_event()) is True

    @pytest.mark.parametrize('field, value', [
        ('id', '00' * 32),
        ('sig', '00' * 64),
        ('sig', 'zz'),
        ('pubkey', 'ab' * 32),
        ('pubkey', 'nothex'),
        ('created_at', NOW - 61),
        ('tags', [['d', 'other']]),
    ])
    def test_any_changed_or_broken_field_fails(self, field, value):
        assert directory.verified(dict(_event(), **{field: value})) is False

    @pytest.mark.parametrize('missing', ['id', 'pubkey', 'created_at', 'kind', 'tags', 'content', 'sig'])
    def test_a_missing_field_fails(self, missing):
        event = _event()
        event.pop(missing)

        assert directory.verified(event) is False


class TestRelayNames:
    def test_host_names_are_split_on_commas_and_spaces(self):
        assert directory.relay_names('relay.one.org, relay.two.org  relay-three.org,,\n') == [
            'relay.one.org', 'relay.two.org', 'relay-three.org',
        ]

    def test_urls_junk_and_overlong_names_are_dropped(self):
        text = 'wss://relay.url.org, bad_name.org, ' + 'a' * 254 + ', kept.org, relay.org:443'

        assert directory.relay_names(text) == ['kept.org']

    @pytest.mark.parametrize('text', ['', '  ,  ', None])
    def test_an_empty_setting_is_no_relay(self, monkeypatch, text):
        monkeypatch.setattr(config, 'P2P_DIRECTORY_RELAYS', text)

        assert directory.relay_names(text) == []
        assert directory.relay_names() == []

    def test_the_setting_is_the_default(self, monkeypatch):
        monkeypatch.setattr(config, 'P2P_DIRECTORY_RELAYS', 'one.example.org,two.example.org')

        assert directory.relay_names() == ['one.example.org', 'two.example.org']


class TestFailureReason:
    def test_a_message_is_named_with_its_type(self):
        assert directory.failure_reason(OSError('connection reset')) == 'OSError: connection reset'

    def test_an_empty_message_is_the_type_name(self):
        assert directory.failure_reason(trio.TooSlowError()) == 'TooSlowError'

    def test_an_exception_group_is_read_through_to_its_first_failure(self):
        nested = ExceptionGroup('outer', [ExceptionGroup('inner', [ValueError('bad frame')]), KeyError('x')])

        assert directory.failure_reason(nested) == 'ValueError: bad frame'

    def test_a_rejected_websocket_handshake_names_only_its_http_status(self):
        rejected = ConnectionRejected(503, ((b'date', b'Sun, 27 Sep 2026 19:15:51 GMT'),), b'busy')

        assert directory.failure_reason(ExceptionGroup('relay', [rejected])) == 'ConnectionRejected: HTTP 503'


class FakeSocket:
    def __init__(self, script):
        self.script = list(script)
        self.sent = []
        self.subscription = None

    async def send_message(self, text):
        message = json.loads(text)
        self.sent.append(message)
        if message[0] == 'REQ':
            self.subscription = message[1]

    async def get_message(self):
        if not self.script:
            await trio.sleep_forever()
        item = self.script.pop(0)
        if callable(item):
            item = item(self)
        return item if isinstance(item, str) else json.dumps(item)


def _fake_relays(monkeypatch, behaviours):
    opened = []

    @asynccontextmanager
    async def relay(socks_port, host):
        opened.append((socks_port, host))
        behaviour = behaviours[host]
        if isinstance(behaviour, Exception):
            raise behaviour
        yield behaviour
        if getattr(behaviour, 'close_error', None) is not None:
            raise behaviour.close_error

    monkeypatch.setattr(directory, '_relay', relay)
    monkeypatch.setattr(directory, 'RELAY_SECONDS', 0.3)
    return opened


class TestPublish:
    def test_each_relay_answers_for_itself(self, monkeypatch, caplog):
        event = _event()
        good = FakeSocket([
            ['NOTICE', 'welcome'],
            ['OK', 'ff' * 32, True, ''],
            ['OK', event['id'], True, ''],
        ])
        refusing = FakeSocket([['OK', event['id'], False, 'blocked: proof of work required']])
        silent = FakeSocket([])
        broken = p2p_tor.TorConnectError('Tor could not reach broken.relay.org: general failure')
        opened = _fake_relays(monkeypatch, {
            'good.relay.org': good, 'refusing.relay.org': refusing,
            'silent.relay.org': silent, 'broken.relay.org': broken,
        })
        relays = ['good.relay.org', 'refusing.relay.org', 'silent.relay.org', 'broken.relay.org']

        with caplog.at_level(logging.WARNING, logger='p2p.directory'):
            results = trio.run(directory.publish, 9050, relays, event)

        assert results == {
            'good.relay.org': True, 'refusing.relay.org': False,
            'silent.relay.org': False, 'broken.relay.org': False,
        }
        assert good.sent == [['EVENT', event]]
        assert sorted(opened) == sorted((9050, host) for host in relays)
        warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert any('refusing.relay.org refused' in text and 'proof of work' in text for text in warnings)
        assert any(text.endswith('broken.relay.org: TorConnectError: Tor could not reach broken.relay.org: '
                                 'general failure') for text in warnings)
        assert any(text.endswith('silent.relay.org: TooSlowError') for text in warnings)
        assert not any('good.relay.org' in text for text in warnings)

    def test_no_relay_is_no_result(self):
        assert trio.run(directory.publish, 9050, [], _event()) == {}

    def test_a_relay_that_closes_untidily_after_its_ok_keeps_the_ok(self, monkeypatch, caplog):
        event = _event()
        relay = FakeSocket([['OK', event['id'], True, '']])
        relay.close_error = RuntimeError('Event Pong cannot be sent in state LOCAL_CLOSING')
        _fake_relays(monkeypatch, {'a.relay.org': relay})

        with caplog.at_level(logging.WARNING, logger='p2p.directory'):
            results = trio.run(directory.publish, 9050, ['a.relay.org'], event)

        assert results == {'a.relay.org': True}
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


class TestFetch:
    def test_the_newest_record_per_author_across_relays(self, monkeypatch, caplog):
        first_old = _event(created=NOW - 500, onion=ONION)
        first_new = _event(created=NOW - 5, onion=OTHER_ONION)
        second = _event(created=NOW - 50, key=OTHER_KEY, onion=ONION)
        stranger = _event(created=NOW - 1, key=base64.b64encode(bytes(64)).decode('ascii'),
                          onion='stranger' + 's' * 48)
        relay_a = FakeSocket([
            lambda sock: ['EVENT', sock.subscription, first_old],
            ['EVENT', 'another-subscription', stranger],
            '"not a list"',
            lambda sock: ['EVENT', sock.subscription, second],
            lambda sock: ['EOSE', sock.subscription],
        ])
        relay_b = FakeSocket([
            lambda sock: ['EVENT', sock.subscription, first_new],
            lambda sock: ['CLOSED', sock.subscription, 'done'],
        ])
        _fake_relays(monkeypatch, {
            'a.relay.org': relay_a, 'b.relay.org': relay_b, 'c.relay.org': OSError('connection reset'),
        })

        with caplog.at_level(logging.WARNING, logger='p2p.directory'):
            records = trio.run(directory.fetch, 9050, ['a.relay.org', 'b.relay.org', 'c.relay.org'], NOW)

        assert sorted((record.onion, record.created_at) for record in records) == [
            (ONION, NOW - 50), (OTHER_ONION, NOW - 5),
        ]
        request = relay_a.sent[0]
        assert request[0] == 'REQ' and len(request[1]) == 16
        assert request[2] == {'kinds': [34101], '#d': ['audiomuse-ai-p2p'], 'since': NOW - DAY, 'limit': 500}
        assert relay_a.sent[-1] == ['CLOSE', relay_a.subscription]
        assert relay_b.sent[-1] == ['CLOSE', relay_b.subscription]
        assert relay_a.subscription != relay_b.subscription
        assert 'Could not read the directory relay c.relay.org: OSError: connection reset' in caplog.text

    def test_a_relay_that_breaks_mid_answer_keeps_what_it_sent(self, monkeypatch, caplog):
        relay = FakeSocket([
            lambda sock: ['EVENT', sock.subscription, _event(created=NOW - 5)],
            '{broken json',
        ])
        _fake_relays(monkeypatch, {'a.relay.org': relay})

        with caplog.at_level(logging.WARNING, logger='p2p.directory'):
            records = trio.run(directory.fetch, 9050, ['a.relay.org'], NOW)

        assert [record.onion for record in records] == [ONION]
        assert 'Could not read the directory relay a.relay.org' in caplog.text

    def test_no_relay_is_no_record(self):
        assert trio.run(directory.fetch, 9050, [], NOW) == []

    def test_a_relay_that_closes_untidily_after_eose_keeps_its_records(self, monkeypatch, caplog):
        relay = FakeSocket([
            lambda sock: ['EVENT', sock.subscription, _event(created=NOW - 5)],
            lambda sock: ['EOSE', sock.subscription],
        ])
        relay.close_error = RuntimeError('Event Pong cannot be sent in state LOCAL_CLOSING')
        _fake_relays(monkeypatch, {'a.relay.org': relay})

        with caplog.at_level(logging.WARNING, logger='p2p.directory'):
            records = trio.run(directory.fetch, 9050, ['a.relay.org'], NOW)

        assert [record.onion for record in records] == [ONION]
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
