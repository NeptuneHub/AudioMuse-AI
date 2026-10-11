# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The AudioMuse-AI servers this member knows, and which of them answer.

A member learns other members from the directory, from the PEERS answers of
the members it talks to and from its saved list. It keeps no connection open:
a member counts as live for LIVE_SECONDS after its last successful exchange and
is probed again before that runs out; one that fails is retried after a minute,
then after a pause that doubles up to 32 minutes (a freshly published onion
service is often unreachable for its first seconds). Songs are asked only of live members that use the same MusiCNN
model space (and, when their record says so, the same space for the output),
at most P2P_MAX_PEERS of them, picked at random.

Main Features:
* PeerBook.learn / succeeded / failed: the member table, keyed by onion address
* live_peers / candidates / to_probe / listed: what the lookup, the probe loop
  and a PEERS answer need
* cache_payload / load_cache: the saved list (up to 256 members, 7 days)
"""

import random
import time

from p2p import tor as p2p_tor, wire

LIVE_SECONDS = 20 * 60
PROBE_SECONDS = 10 * 60
RETRY_SECONDS = 60
MAX_BACKOFF_STEPS = 5
CACHE_MAX = 256
CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600


class Peer:
    def __init__(self, onion, heard):
        self.onion = onion
        self.heard = heard
        self.spaces = None
        self.version = None
        self.ok = None
        self.tried = None
        self.failures = 0


class PeerBook:
    def __init__(self, own_onion, spaces, max_peers, clock=None):
        self.own_onion = own_onion
        self.spaces = dict(spaces)
        self.max_peers = max(1, int(max_peers))
        self._clock = clock or time.time
        self.peers = {}

    def learn(self, onion, spaces=None, version=None, heard=None):
        if onion == self.own_onion or not p2p_tor.valid_onion(onion):
            return None
        now = self._clock()
        peer = self.peers.get(onion)
        if peer is None:
            peer = self.peers[onion] = Peer(onion, heard or now)
        else:
            peer.heard = max(peer.heard, heard or now)
        if spaces is not None:
            peer.spaces = dict(spaces)
        if version is not None:
            peer.version = version
        return peer

    def succeeded(self, onion):
        peer = self.peers.get(onion) or self.learn(onion)
        if peer is not None:
            peer.ok = peer.tried = self._clock()
            peer.failures = 0

    def failed(self, onion):
        peer = self.peers.get(onion)
        if peer is None:
            return
        peer.tried = self._clock()
        peer.failures += 1
        if peer.heard < peer.tried - CACHE_MAX_AGE_SECONDS and (peer.ok is None or peer.ok < peer.tried - CACHE_MAX_AGE_SECONDS):
            self.peers.pop(onion, None)

    def is_live(self, peer):
        return peer.ok is not None and self._clock() - peer.ok < LIVE_SECONDS

    def live_peers(self):
        return [peer for peer in self.peers.values() if self.is_live(peer)]

    def usable_wants(self, peer, wants):
        if not self.spaces.get(wire.OUT_MUSICNN):
            return ()
        if peer.spaces is None:
            return tuple(key for key in wants if self.spaces.get(key))
        if peer.spaces.get(wire.OUT_MUSICNN) != self.spaces.get(wire.OUT_MUSICNN):
            return ()
        return tuple(key for key in wants if self.spaces.get(key) and peer.spaces.get(key) == self.spaces.get(key))

    def candidates(self, wants):
        found = []
        for peer in self.live_peers():
            usable = self.usable_wants(peer, wants)
            if usable:
                found.append((peer.onion, usable))
        random.shuffle(found)
        return found[:self.max_peers]

    def to_probe(self, limit):
        now = self._clock()
        due = []
        for peer in self.peers.values():
            if peer.tried is None:
                due.append((0, peer.heard, peer))
            elif peer.failures:
                wait = RETRY_SECONDS * (2 ** min(peer.failures - 1, MAX_BACKOFF_STEPS))
                if now - peer.tried >= wait:
                    due.append((2, -peer.failures, peer))
            elif now - peer.tried >= PROBE_SECONDS:
                due.append((1, peer.ok or 0, peer))
        due.sort(key=lambda item: (item[0], -item[1]))
        return [peer.onion for _rank, _order, peer in due[:max(0, int(limit))]]

    def listed(self, limit=wire.MAX_PEERS_LISTED):
        live = [peer.onion for peer in self.live_peers()]
        random.shuffle(live)
        return live[:limit]

    def cache_payload(self):
        horizon = self._clock() - CACHE_MAX_AGE_SECONDS
        kept = [
            peer for peer in self.peers.values()
            if max(peer.heard, peer.ok or 0) >= horizon
        ]
        kept.sort(key=lambda peer: max(peer.heard, peer.ok or 0), reverse=True)
        return {'v': 1, 'peers': {
            peer.onion: {'heard': int(peer.heard), 'ok': int(peer.ok) if peer.ok else None} for peer in kept[:CACHE_MAX]
        }}

    def load_cache(self, data):
        if not isinstance(data, dict) or not isinstance(data.get('peers'), dict):
            return
        horizon = self._clock() - CACHE_MAX_AGE_SECONDS
        for onion, entry in data['peers'].items():
            if not isinstance(entry, dict) or not isinstance(entry.get('heard'), int):
                continue
            ok = entry.get('ok') if isinstance(entry.get('ok'), int) else None
            if max(entry['heard'], ok or 0) < horizon:
                continue
            peer = self.learn(onion, heard=entry['heard'])
            if peer is not None and ok is not None:
                peer.ok = ok
