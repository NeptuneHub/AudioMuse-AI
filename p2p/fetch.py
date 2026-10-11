# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The requesting half of the node: turn one worker request into peer results.

The node reads the song's own MusiCNN embedding, duration and chromaprint
(read-only) and sends a QUERY with the signature, the duration and the space
ids to every live compatible member at once, each over its own Tor stream. It
stops when every member answered, 300 ms after the first confirmed offer, at
five offers, or at the worker's deadline.
An offer counts only when it is the same recording by the local dedup rule
(simhash.p2p_same_recording: signature, duration, cosine and chromaprint;
with a local chromaprint and CHROMAPRINT_GATE_ENABLED, an offer without the
peer's chromaprint is refused) and its MusiCNN result is v3.0.0 or newer.
The pick is random, then FETCH. Each result is taken as it is when it comes
from the same model space, is v3.0.0 or newer and decodes into the stored
shape; the reply goes back to the worker in NOTIFY-sized parts. The requester
never sends its embedding.

Main Features:
* confirm / offered_keys / pick_order / check_outputs as pure functions
* Fetcher.handle: drops a request past its deadline, merges a second request
  for an item already in flight, always answers the worker (ok, none,
  no_peers, busy or error) on its reply channel
* a FETCH waits for a download slot; every exchange travels through Tor, the
  neural fingerprint included; the peer book learns which members answer
* a result that is unasked, older than v3.0.0, from another model space or
  not decodable is dropped and logged (7003); a malformed frame ends that
  stream and is logged (7003); nothing else happens to the peer
"""

import logging
import math
import random
from typing import NamedTuple

import trio

import config
from error import error_manager
from error.error_dictionary import ERR_P2P_PEER_REJECTED, ERR_P2P_REQUEST_FAILED
from p2p import compare, ipc, wire
from tasks import simhash

logger = logging.getLogger(__name__)

QUERY_SECONDS = 10.0
CONFIRM_GRACE_SECONDS = 0.3
MAX_OFFERS = 5
FETCH_ATTEMPTS = 2
FETCH_MAX_SECONDS = 20.0
REPLY_LATE_SECONDS = 1.0


class Offered(NamedTuple):
    peer: str
    candidate: wire.Candidate
    keys: tuple


class Supplied(NamedTuple):
    peer: str
    versions: dict
    outputs: dict


class Outcome(NamedTuple):
    status: str
    supplied: object


def offered_keys(candidate, wants):
    return tuple(
        key for key in wants
        if key in candidate.versions and wire.version_passes(candidate.versions[key])
    )


def confirm(local, candidate):
    if not wire.version_passes(candidate.versions.get(wire.OUT_MUSICNN)):
        return False
    peer_print = bytes(candidate.chromaprint) if candidate.chromaprint else None
    if config.CHROMAPRINT_GATE_ENABLED and local.chromaprint and not peer_print:
        return False
    return simhash.p2p_same_recording(
        local.embedding, candidate.embedding, local.duration, candidate.duration,
        local.chromaprint, peer_print,
    )


def pick_order(offers, rng):
    ordered = list(offers)
    rng.shuffle(ordered)
    return ordered


def decodable(key, value, codebook_crc):
    if key == wire.OUT_CLAP:
        return compare.clap_ok(value)
    if key == wire.OUT_LYRICS:
        return compare.lyrics_ok(value[0], value[1])
    return compare.neural_ok(value, codebook_crc)


def check_outputs(message, keys, spaces, codebook_crc):
    accepted = {}
    rejected = []
    for key, item in message.items.items():
        if key not in keys:
            rejected.append((key, 'it was not asked for'))
        elif item.space != spaces.get(key):
            rejected.append((key, 'it comes from another model space'))
        elif not wire.version_passes(item.version):
            rejected.append((key, f'its version {item.version!r} is below v3.0.0 or unreadable'))
        elif not decodable(key, item.value, codebook_crc):
            rejected.append((key, 'it cannot be decoded into the stored shape'))
        else:
            accepted[key] = (item.version, item.value)
    return accepted, rejected


def log_malformed(peer, what, exc):
    error_manager.record(
        ERR_P2P_PEER_REJECTED, f"Dropped a malformed {what} from peer {peer}: {exc}",
        logger=logger, level=logging.WARNING,
    )


def encode_outputs(outputs):
    encoded = {}
    for key, value in outputs.items():
        if key == wire.OUT_CLAP:
            encoded[ipc.OUT_CLAP] = ipc.encode_floats(value)
        elif key == wire.OUT_LYRICS:
            encoded[ipc.OUT_LYRICS] = ipc.encode_floats(value[0])
            encoded[ipc.OUT_AXIS] = ipc.encode_floats(value[1])
        elif key == wire.OUT_NEURAL:
            encoded[ipc.OUT_NEURAL] = ipc.encode_bytes(value)
    return encoded


def reply_for(rid, outcome, wants):
    supplied = outcome.supplied
    if outcome.status != ipc.REPLY_OK or supplied is None:
        return ipc.supply_reply(rid, outcome.status)
    outputs = {key: value for key, value in supplied.outputs.items() if key in wants}
    if not outputs:
        return ipc.supply_reply(rid, ipc.REPLY_NONE)
    versions = {wire.OUT_MUSICNN: supplied.versions[wire.OUT_MUSICNN]}
    versions.update({key: supplied.versions[key] for key in outputs})
    return ipc.supply_reply(rid, ipc.REPLY_OK, peer=supplied.peer, versions=versions,
                            outputs=encode_outputs(outputs))


class _InFlight:
    def __init__(self, wants):
        self.wants = frozenset(wants)
        self.done = trio.Event()
        self.outcome = Outcome(ipc.REPLY_NONE, None)


class Fetcher:
    def __init__(self, store, book, spaces, codebook_crc, limits, opener, send_reply, clock=None, rng=None):
        self._store = store
        self._book = book
        self.spaces = dict(spaces)
        self._codebook_crc = codebook_crc
        self._limits = limits
        self._opener = opener
        self._send_reply = send_reply
        self._clock = clock or trio.current_time
        self._rng = rng or random.SystemRandom()
        self._inflight = {}
        self.counters = {'requests': 0, 'supplied': 0, 'none': 0, 'no_peers': 0, 'busy': 0,
                         'errors': 0, 'dropped': 0, 'merged': 0}

    def _count(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1

    async def handle(self, request, deadline):
        rid = request['rid']
        item_id = request['item']
        wants = tuple(key for key in request['wants'] if key in wire.OUTPUT_KEYS)
        self._count('requests')
        if self._clock() >= deadline:
            self._count('dropped')
            logger.warning("Dropped the P2P request %s for %s: it expired before the node reached it", rid, item_id)
            return
        leader = self._inflight.get(item_id)
        if leader is not None and set(wants) <= leader.wants:
            self._count('merged')
            with trio.move_on_at(deadline):
                await leader.done.wait()
            outcome = leader.outcome if leader.done.is_set() else Outcome(ipc.REPLY_NONE, None)
        else:
            entry = _InFlight(wants)
            self._inflight[item_id] = entry
            outcome = Outcome(ipc.REPLY_ERROR, None)
            try:
                outcome = await self._run(request, wants, deadline)
            finally:
                entry.outcome = outcome
                entry.done.set()
                if self._inflight.get(item_id) is entry:
                    self._inflight.pop(item_id, None)
        self._count({ipc.REPLY_OK: 'supplied', ipc.REPLY_NONE: 'none', ipc.REPLY_NO_PEERS: 'no_peers',
                     ipc.REPLY_BUSY: 'busy'}.get(outcome.status, 'errors'))
        if self._clock() > deadline + REPLY_LATE_SECONDS:
            logger.warning("The P2P answer for %s is past the worker's deadline; not sent", item_id)
            return
        await self._send_reply(request['reply'], reply_for(rid, outcome, wants))

    async def _run(self, request, wants, deadline):
        try:
            local = await trio.to_thread.run_sync(
                self._store.local_track, request['item'], request.get('server'), request.get('provider'),
            )
            if local is None:
                logger.warning(
                    "P2P request for %s: no MusiCNN embedding and duration stored yet; nothing to ask peers",
                    request['item'],
                )
                return Outcome(ipc.REPLY_NONE, None)
            return await self.lookup(request['item'], local, wants, deadline)
        except Exception:
            logger.exception("[%s] The P2P lookup for %s failed", ERR_P2P_REQUEST_FAILED, request['item'])
            return Outcome(ipc.REPLY_ERROR, None)

    async def lookup(self, item_id, local, wants, deadline):
        peers = self._book.candidates(wants)
        if not peers:
            return Outcome(ipc.REPLY_NO_PEERS, None)
        signature = simhash.embedding_signature(local.embedding)
        if signature is None:
            return Outcome(ipc.REPLY_NONE, None)
        send, receive = trio.open_memory_channel(len(peers))
        wanted_by_peer = dict(peers)
        offers = []
        answers = {'busy': 0, 'answered': 0, 'unreachable': 0, 'offers': 0}
        async with trio.open_nursery() as nursery:
            async with send:
                for peer, keys in peers:
                    nursery.start_soon(self._ask, peer, keys, local, signature, send.clone())
            grace = math.inf
            while answers['answered'] < len(peers) and answers['offers'] < MAX_OFFERS:
                result = None
                with trio.move_on_at(min(deadline, grace)):
                    try:
                        result = await receive.receive()
                    except trio.EndOfChannel:
                        break
                if result is None:
                    break
                answers['answered'] += 1
                kind, peer, payload = result
                if kind == 'busy':
                    answers['busy'] += 1
                if kind == 'failed':
                    answers['unreachable'] += 1
                if kind != 'offer':
                    continue
                answers['offers'] += 1
                for candidate in payload.candidates:
                    keys = offered_keys(candidate, wanted_by_peer[peer])
                    if keys and confirm(local, candidate):
                        offers.append(Offered(peer, candidate, keys))
                if offers and grace == math.inf:
                    grace = self._clock() + CONFIRM_GRACE_SECONDS
            nursery.cancel_scope.cancel()
        if not offers:
            logger.info(
                "P2P lookup for %s: %d servers asked, %d unreachable through Tor, %d offers, none confirmed",
                item_id, len(peers), answers['unreachable'], answers['offers'],
            )
            if answers['busy'] and not answers['offers']:
                return Outcome(ipc.REPLY_BUSY, None)
            return Outcome(ipc.REPLY_NONE, None)
        for offered in pick_order(offers, self._rng)[:FETCH_ATTEMPTS]:
            if self._clock() >= deadline:
                break
            supplied = await self._fetch(item_id, offered, deadline)
            if supplied is not None:
                logger.info(
                    "P2P lookup for %s: %s supplied by peer %s (versions %s)",
                    item_id, ','.join(sorted(supplied.outputs)), supplied.peer,
                    ', '.join(f"{key}={value}" for key, value in sorted(supplied.versions.items())),
                )
                return Outcome(ipc.REPLY_OK, supplied)
        return Outcome(ipc.REPLY_NONE, None)

    async def _ask(self, peer, keys, local, signature, send):
        async with send:
            result = ('failed', peer, None)
            with trio.move_on_after(QUERY_SECONDS):
                try:
                    async with self._opener(peer) as io:
                        await io.write(wire.encode_query(
                            simhash.signature_to_bytes(signature), local.duration, keys, self.spaces,
                        ))
                        frame_type, message = await wire.read_frame(io)
                        if frame_type == wire.T_OFFER:
                            result = ('offer', peer, message)
                            await io.write(wire.encode_end())
                        elif frame_type == wire.T_BUSY:
                            result = ('busy', peer, None)
                        elif frame_type == wire.T_NO:
                            result = ('no', peer, message)
                        else:
                            result = ('no', peer, None)
                except (wire.UnknownVersion, wire.UnknownType):
                    result = ('no', peer, None)
                except wire.Violation as exc:
                    log_malformed(peer, 'answer to a QUERY', exc)
                    result = ('failed', peer, None)
                except Exception:
                    logger.warning("P2P QUERY to peer %s failed", peer, exc_info=True)
                    result = ('failed', peer, None)
            if result[0] == 'failed':
                self._book.failed(peer)
            else:
                self._book.succeeded(peer)
            try:
                send.send_nowait(result)
            except (trio.WouldBlock, trio.BrokenResourceError, trio.ClosedResourceError):
                pass

    async def _fetch(self, item_id, offered, deadline):
        until = min(deadline, self._clock() + FETCH_MAX_SECONDS)
        with trio.move_on_at(until):
            async with self._limits.downloads:
                try:
                    keys = offered.keys
                    async with self._opener(offered.peer) as io:
                        await io.write(wire.encode_fetch(offered.candidate.token, keys, self.spaces))
                        frame_type, message = await wire.read_frame(io)
                        await io.write(wire.encode_end())
                except (wire.UnknownVersion, wire.UnknownType):
                    return None
                except wire.Violation as exc:
                    log_malformed(offered.peer, 'answer to a FETCH', exc)
                    return None
                except Exception:
                    logger.warning("P2P FETCH from peer %s failed", offered.peer, exc_info=True)
                    return None
            if frame_type != wire.T_OUTPUTS:
                logger.info("P2P FETCH for %s: peer %s answered frame type %d instead of results",
                            item_id, offered.peer, frame_type)
                return None
            accepted, rejected = check_outputs(message, keys, self.spaces, self._codebook_crc)
            for key, reason in rejected:
                error_manager.record(
                    ERR_P2P_PEER_REJECTED,
                    f"Dropped the {key} result of {item_id} from peer {offered.peer}: {reason}",
                    logger=logger, level=logging.WARNING,
                )
            if not accepted:
                return None
            versions = {wire.OUT_MUSICNN: offered.candidate.versions[wire.OUT_MUSICNN]}
            versions.update({key: version for key, (version, _value) in accepted.items()})
            return Supplied(offered.peer, versions, {key: value for key, (_version, value) in accepted.items()})
        logger.info("P2P FETCH for %s from peer %s ran out of time", item_id, offered.peer)
        return None
