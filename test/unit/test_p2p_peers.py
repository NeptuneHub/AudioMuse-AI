# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The AudioMuse-AI servers a member knows and which of them answer (p2p.peers).

The PeerBook keeps no connection: a server is live for LIVE_SECONDS after its
last good exchange and is probed again before that. These tests drive it with
a fake clock and plain onion addresses, with no network and no database.

Main Features:
* learn keys servers by onion address, skips this server's own address and
  anything that is not a v3 onion id, keeps the newest heard time and only
  overwrites spaces or version when they are given
* succeeded makes a server live for 20 minutes (and learns an unknown one);
  failed counts failures and drops a server nobody heard of for 7 days
* candidates offers only live servers whose model spaces match (all local
  spaces when a server's are unknown), at most P2P_MAX_PEERS of them
* to_probe: never tried first (newest heard first), then servers due for a
  refresh after 10 minutes, then failing servers after a pause of 1, 2, 4,
  8, 16 and at most 32 minutes; limit caps the batch
* listed hands out live servers only, at most 32
* cache_payload / load_cache round-trip through JSON, keep the 256 newest
  within 7 days, and load_cache ignores anything malformed
"""

import json

import pytest

from p2p import peers, wire

SPACES = {key: bytes([index + 1]) * 32 for index, key in enumerate(wire.SPACE_KEYS)}
OWN = 'ownserver' + 'o' * 47
BASE32 = 'abcdefghijklmnopqrstuvwxyz234567'


def onion(index):
    digits = []
    value = index
    for _ in range(8):
        digits.append(BASE32[value % 32])
        value //= 32
    return 'peer' + 'x' * 44 + ''.join(digits)


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def _book(max_peers=4, spaces=SPACES, clock=None):
    return peers.PeerBook(OWN, spaces, max_peers, clock=clock or Clock())


def _live(book, count, spaces=None):
    names = [onion(index) for index in range(count)]
    for name in names:
        book.learn(name, spaces=spaces)
        book.succeeded(name)
    return names


class TestLearn:
    def test_a_valid_onion_is_learned_once(self):
        clock = Clock()
        book = _book(clock=clock)

        first = book.learn(onion(1))
        again = book.learn(onion(1))

        assert first is again
        assert list(book.peers) == [onion(1)]
        assert first.heard == clock.now and first.ok is None and first.failures == 0

    @pytest.mark.parametrize('name', [
        OWN, '', None, 'short', onion(1).upper(), onion(1) + '.onion', onion(1)[:-1], onion(1) + 'a',
        'peer' + 'x' * 44 + '0189abcd',
    ])
    def test_its_own_address_and_anything_but_an_onion_id_are_skipped(self, name):
        book = _book()

        assert book.learn(name) is None
        assert book.peers == {}

    def test_the_newest_heard_time_wins(self):
        clock = Clock()
        book = _book(clock=clock)

        book.learn(onion(1), heard=clock.now - 50)
        book.learn(onion(1), heard=clock.now - 500)
        assert book.peers[onion(1)].heard == clock.now - 50
        clock.now += 10
        book.learn(onion(1))
        assert book.peers[onion(1)].heard == clock.now

    def test_spaces_and_version_are_kept_unless_given_again(self):
        book = _book()
        other = dict(SPACES, c=b'\xee' * 32)

        book.learn(onion(1), spaces=SPACES, version='v3.6.3')
        book.learn(onion(1))
        assert (book.peers[onion(1)].spaces, book.peers[onion(1)].version) == (SPACES, 'v3.6.3')
        book.learn(onion(1), spaces=other, version='v3.7.0')
        assert (book.peers[onion(1)].spaces, book.peers[onion(1)].version) == (other, 'v3.7.0')

    def test_the_peer_limit_is_at_least_one(self):
        assert _book(max_peers=0).max_peers == 1
        assert _book(max_peers='3').max_peers == 3


class TestLiveness:
    def test_a_success_keeps_a_server_live_for_twenty_minutes(self):
        clock = Clock()
        book = _book(clock=clock)
        book.learn(onion(1))
        assert book.live_peers() == []

        book.succeeded(onion(1))
        peer = book.peers[onion(1)]
        assert book.is_live(peer) and peer.ok == peer.tried == clock.now
        clock.now += peers.LIVE_SECONDS - 1
        assert book.live_peers() == [peer]
        clock.now += 1
        assert book.live_peers() == []

    def test_a_success_learns_an_unknown_server_and_resets_its_failures(self):
        book = _book()
        book.learn(onion(2))
        book.failed(onion(2))
        book.failed(onion(2))

        book.succeeded(onion(1))
        book.succeeded(onion(2))

        assert book.peers[onion(1)].ok is not None
        assert book.peers[onion(2)].failures == 0

    def test_a_success_for_an_invalid_address_changes_nothing(self):
        book = _book()

        book.succeeded('not-an-onion')
        book.succeeded(OWN)

        assert book.peers == {}

    def test_a_failure_of_an_unknown_server_is_ignored(self):
        book = _book()

        book.failed(onion(1))

        assert book.peers == {}

    def test_a_failure_counts_and_keeps_a_recent_server(self):
        clock = Clock()
        book = _book(clock=clock)
        book.learn(onion(1))

        book.failed(onion(1))
        book.failed(onion(1))

        assert book.peers[onion(1)].failures == 2
        assert book.peers[onion(1)].tried == clock.now

    def test_a_server_nobody_heard_of_for_seven_days_is_dropped_when_it_fails(self):
        clock = Clock()
        book = _book(clock=clock)
        book.learn(onion(1))
        book.learn(onion(2))
        book.succeeded(onion(2))
        clock.now += peers.CACHE_MAX_AGE_SECONDS

        book.failed(onion(1))
        book.failed(onion(2))
        assert set(book.peers) == {onion(1), onion(2)}
        clock.now += 1
        book.failed(onion(1))
        book.failed(onion(2))

        assert book.peers == {}


class TestCandidates:
    def test_only_live_compatible_servers_are_asked(self):
        clock = Clock()
        book = _book(max_peers=10, clock=clock)
        same, other_clap, foreign, unknown = onion(1), onion(2), onion(3), onion(4)
        book.learn(same, spaces=SPACES)
        book.learn(other_clap, spaces=dict(SPACES, c=b'\xee' * 32))
        book.learn(foreign, spaces=dict(SPACES, m=b'\xee' * 32))
        book.learn(unknown)
        for name in (same, other_clap, foreign, unknown):
            book.succeeded(name)
        book.learn(onion(5), spaces=SPACES)

        found = dict(book.candidates(('c', 'l', 'n')))

        assert found == {same: ('c', 'l', 'n'), other_clap: ('l', 'n'), unknown: ('c', 'l', 'n')}

    def test_a_server_whose_record_lacks_an_output_space_is_not_asked_for_it(self):
        book = _book()
        book.learn(onion(1), spaces={key: value for key, value in SPACES.items() if key != 'n'})
        book.succeeded(onion(1))

        assert book.candidates(('c', 'n')) == [(onion(1), ('c',))]

    def test_a_want_without_a_local_space_is_never_asked(self):
        book = _book(spaces=dict(SPACES, l=None))
        _live(book, 1)

        assert book.candidates(('c', 'l', 'n')) == [(onion(0), ('c', 'n'))]

    def test_without_a_local_musicnn_space_nobody_is_asked(self):
        book = _book(spaces=dict(SPACES, m=None))
        _live(book, 3)

        assert book.candidates(('c', 'l', 'n')) == []

    def test_at_most_the_peer_limit_is_asked(self):
        book = _book(max_peers=3)
        names = _live(book, 10)

        found = book.candidates(('c',))

        assert len(found) == 3
        assert {name for name, _keys in found} <= set(names)
        assert all(keys == ('c',) for _name, keys in found)

    def test_a_stale_server_is_not_asked(self):
        clock = Clock()
        book = _book(clock=clock)
        _live(book, 2)
        clock.now += peers.LIVE_SECONDS

        assert book.candidates(('c',)) == []


class TestToProbe:
    def test_never_tried_then_due_for_refresh_then_failing(self):
        clock = Clock()
        book = _book(clock=clock)
        book.learn(onion(1))
        book.succeeded(onion(1))
        book.learn(onion(2))
        book.failed(onion(2))
        clock.now += peers.PROBE_SECONDS
        book.learn(onion(3), heard=clock.now - 100)
        book.learn(onion(4), heard=clock.now - 10)

        assert book.to_probe(10) == [onion(4), onion(3), onion(1), onion(2)]
        assert book.to_probe(2) == [onion(4), onion(3)]
        assert book.to_probe(0) == []

    def test_a_server_that_just_answered_is_not_due(self):
        clock = Clock()
        book = _book(clock=clock)
        book.learn(onion(1))
        book.succeeded(onion(1))

        clock.now += peers.PROBE_SECONDS - 1
        assert book.to_probe(8) == []
        clock.now += 1
        assert book.to_probe(8) == [onion(1)]

    @pytest.mark.parametrize('failures, minutes', [
        (1, 1), (2, 2), (3, 4), (4, 8), (5, 16), (6, 32), (7, 32), (12, 32),
    ])
    def test_a_failing_server_waits_a_doubling_pause_up_to_32_minutes(self, failures, minutes):
        clock = Clock()
        book = _book(clock=clock)
        book.learn(onion(1))
        for _ in range(failures):
            book.failed(onion(1))

        clock.now += minutes * 60 - 1
        assert book.to_probe(8) == []
        clock.now += 1
        assert book.to_probe(8) == [onion(1)]

    def test_among_failing_servers_the_fewest_failures_come_first(self):
        clock = Clock()
        book = _book(clock=clock)
        for name, failures in ((onion(1), 3), (onion(2), 1), (onion(3), 2)):
            book.learn(name)
            for _ in range(failures):
                book.failed(name)
        clock.now += 32 * 60

        assert book.to_probe(8) == [onion(2), onion(3), onion(1)]


class TestListed:
    def test_only_live_servers_are_listed(self):
        clock = Clock()
        book = _book(clock=clock)
        _live(book, 3)
        book.learn(onion(9))

        assert sorted(book.listed()) == sorted(onion(index) for index in range(3))

    def test_at_most_32_servers_are_listed(self):
        book = _book()
        names = _live(book, 40)

        listed = book.listed()

        assert wire.MAX_PEERS_LISTED == 32
        assert len(listed) == 32 and set(listed) <= set(names)
        assert len(book.listed(5)) == 5


class TestCache:
    def test_the_payload_round_trips_through_json(self):
        clock = Clock()
        book = _book(clock=clock)
        book.learn(onion(1), heard=clock.now - 3600)
        book.succeeded(onion(2))
        payload = json.loads(json.dumps(book.cache_payload()))

        fresh = _book(clock=clock)
        fresh.load_cache(payload)

        assert payload == {'v': 1, 'peers': {
            onion(1): {'heard': int(clock.now - 3600), 'ok': None},
            onion(2): {'heard': int(clock.now), 'ok': int(clock.now)},
        }}
        assert fresh.peers[onion(1)].heard == clock.now - 3600 and fresh.peers[onion(1)].ok is None
        assert fresh.peers[onion(2)].ok == clock.now
        assert fresh.live_peers() == [fresh.peers[onion(2)]]

    def test_the_payload_keeps_the_256_newest_within_seven_days(self):
        clock = Clock()
        book = _book(clock=clock)
        for index in range(300):
            book.learn(onion(index), heard=clock.now - 300 + index)
        book.learn(onion(999), heard=clock.now - peers.CACHE_MAX_AGE_SECONDS - 1)

        kept = book.cache_payload()['peers']

        assert len(kept) == peers.CACHE_MAX == 256
        assert set(kept) == {onion(index) for index in range(44, 300)}

    def test_a_server_that_answered_this_week_is_kept_even_if_first_heard_long_ago(self):
        clock = Clock()
        book = _book(clock=clock)
        book.learn(onion(1), heard=clock.now - 30 * 24 * 3600)
        book.succeeded(onion(1))

        assert onion(1) in book.cache_payload()['peers']

    @pytest.mark.parametrize('data', [
        None, [], 'text', {}, {'peers': []}, {'peers': 'x'},
    ])
    def test_a_malformed_cache_loads_nothing(self, data):
        book = _book()

        book.load_cache(data)

        assert book.peers == {}

    def test_malformed_entries_are_skipped_one_by_one(self):
        clock = Clock()
        book = _book(clock=clock)
        now = int(clock.now)

        book.load_cache({'v': 1, 'peers': {
            onion(1): {'heard': now, 'ok': now},
            onion(2): {'heard': 'yesterday'},
            onion(3): 'not a dict',
            onion(4): {'heard': now - peers.CACHE_MAX_AGE_SECONDS - 10, 'ok': None},
            onion(5): {'heard': now, 'ok': 'soon'},
            'not-an-onion': {'heard': now},
            OWN: {'heard': now},
        }})

        assert set(book.peers) == {onion(1), onion(5)}
        assert book.peers[onion(1)].ok == now
        assert book.peers[onion(5)].ok is None
