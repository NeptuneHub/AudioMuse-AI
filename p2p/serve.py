# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The serving half of the node: answer QUERY, FETCH and PEERS from other servers.

Requests arrive through Tor, so the caller is anonymous: a FETCH token is a
bearer secret (16 random bytes, one use, 60 s) and the QUERY budget is shared
by every caller.

A resident PackedSignatureIndex holds the shareable rows: signature ids with a
known duration whose MusiCNN result passes the version rule. It is built once
(queries get NO until then) and rebuilt hourly; every 60 s the rows created
since are put in a small delta index, so both are immutable once published and
a lookup never sees a half-built index. A candidate is re-read in one
statement before anything leaves the server.

The version rule: every stored output is served, computed here or received
from a peer, when its output_origin stamp names AudioMuse-AI v3.0.0 or newer
(a received output keeps the version of the install that computed it; its
"p" is provenance only). An output with no stamp was analysed before stamps
existed and is served with this install's own APP_VERSION. The only other
checks are the ones needed to decode the bytes: exact size and finite values,
and a neural fingerprint only with the local codebook. The lyrics
instrumental sentinel is never served.

Main Features:
* musicnn_version / output_version / servable_outputs: the version rule as
  pure functions
* Server.answer_query: rate limit, space match, index lookup, up to two
  candidates, one-time 16-byte tokens valid 60 s
* Server.answer_fetch: token, re-read, OUTPUTS; handle_stream answers BUSY
  when the upload slots are full
* Server.handle_stream: the per-stream frame loop; a NO answers an unknown
  version or type, a malformed frame raises Violation and ends the stream; a
  PEERS answer lists the members this server has reached recently, so members
  introduce each other
"""

import datetime
import logging
import secrets
from typing import NamedTuple

import trio

import config
from p2p import compare, wire
from tasks import simhash

logger = logging.getLogger(__name__)

TOKEN_SECONDS = 60.0
MAX_TOKENS = 4096
LOOKUP_CANDIDATES = 4
CATCH_UP_SECONDS = 60.0
REBUILD_SECONDS = 3600.0
CATCH_UP_OVERLAP = datetime.timedelta(seconds=120)
STREAM_IDLE_SECONDS = 10.0
MAX_FRAMES_PER_STREAM = 4


class Token(NamedTuple):
    item_id: str
    peer: str
    expires: float
    keys: tuple


class IndexState(NamedTuple):
    main: object
    delta: object
    newest: object
    known: frozenset
    delta_rows: tuple


def _entry_version(entry):
    if entry is None:
        version = config.APP_VERSION
    elif isinstance(entry, dict):
        version = entry.get('v')
    else:
        return None
    return version if wire.version_passes(version) else None


def musicnn_version(origin):
    return _entry_version((origin or {}).get(wire.OUT_MUSICNN))


def output_version(origin, key):
    return _entry_version((origin or {}).get(key))


def shareable(row):
    return (
        simhash.is_signature_id(row.item_id)
        and row.duration is not None and row.duration > 0
        and _entry_version(row.musicnn_origin) is not None
    )


def servable_outputs(row, keys, spaces, codebook_crc):
    served = {}
    for key in keys:
        if not spaces.get(key):
            continue
        version = output_version(row.origin, key)
        if version is None:
            continue
        if key == wire.OUT_CLAP and compare.clap_ok(row.clap):
            served[key] = (version, compare.as_vector(row.clap, wire.CLAP_DIM))
        elif key == wire.OUT_LYRICS and compare.lyrics_ok(row.lyrics, row.axis):
            served[key] = (version, (
                compare.as_vector(row.lyrics, wire.LYRICS_DIM), compare.as_vector(row.axis, wire.AXIS_DIM),
            ))
        elif key == wire.OUT_NEURAL and row.neural is not None:
            if len(row.neural) <= wire.MAX_NEURAL_BYTES and compare.neural_ok(row.neural, codebook_crc):
                served[key] = (version, bytes(row.neural))
    return served


def _index_of(rows):
    index = simhash.PackedSignatureIndex()
    if rows:
        index.extend(
            [row.item_id for row in rows],
            [simhash.signature_from_canonical_id(row.item_id) for row in rows],
            [row.duration for row in rows],
        )
    return index


class Server:
    def __init__(self, store, spaces, codebook_crc, limits, peers_provider=None, clock=None):
        self._store = store
        self.spaces = dict(spaces)
        self._codebook_crc = codebook_crc
        self._limits = limits
        self._peers_provider = peers_provider
        self._clock = clock or trio.current_time
        self.state = None
        self._building = False
        self._tokens = {}
        self.counters = {'queries': 0, 'offers': 0, 'fetches': 0, 'served_outputs': 0, 'busy': 0}

    def _count(self, key, amount=1):
        self.counters[key] = self.counters.get(key, 0) + amount

    @property
    def size(self):
        state = self.state
        return 0 if state is None else len(state.main) + len(state.delta)

    def _build_blocking(self):
        main = simhash.PackedSignatureIndex()
        after = None
        newest = None
        overlap = {}
        while True:
            page = self._store.shareable_page(after)
            if not page:
                break
            after = page[-1].item_id
            kept = [row for row in page if shareable(row)]
            if kept:
                main.extend(
                    [row.item_id for row in kept],
                    [simhash.signature_from_canonical_id(row.item_id) for row in kept],
                    [row.duration for row in kept],
                )
            for row in kept:
                overlap[row.item_id] = row.created_at
                if newest is None or row.created_at > newest:
                    newest = row.created_at
        known = frozenset(
            item_id for item_id, created in overlap.items()
            if newest is not None and newest - created <= CATCH_UP_OVERLAP
        )
        return IndexState(main, _index_of(()), newest, known, ())

    def _catch_up_blocking(self, state):
        if state.newest is None:
            return self._build_blocking()
        cursor = (state.newest - CATCH_UP_OVERLAP, '')
        known = set(state.known)
        fresh = list(state.delta_rows)
        newest = state.newest
        while True:
            page = self._store.shareable_since(cursor[0], cursor[1])
            if not page:
                break
            for row in page:
                if row.created_at > newest:
                    newest = row.created_at
                if row.item_id not in known and shareable(row):
                    known.add(row.item_id)
                    fresh.append(row)
            cursor = (page[-1].created_at, page[-1].item_id)
        if len(fresh) == len(state.delta_rows):
            return state._replace(newest=newest)
        return IndexState(state.main, _index_of(fresh), newest, frozenset(known), tuple(fresh))

    async def build_index(self):
        if self._building:
            return False
        self._building = True
        try:
            state = await trio.to_thread.run_sync(self._build_blocking)
        finally:
            self._building = False
        self.state = state
        logger.info("P2P serving index built: %d shareable songs", len(state.main))
        return True

    async def catch_up(self):
        state = self.state
        if state is None or self._building:
            return 0
        self._building = True
        try:
            updated = await trio.to_thread.run_sync(self._catch_up_blocking, state)
        finally:
            self._building = False
        added = len(updated.delta_rows) - len(state.delta_rows)
        self.state = updated
        if added > 0:
            logger.info("P2P serving index caught up: %d new shareable songs", added)
        return max(0, added)

    async def index_loop(self, task_status=trio.TASK_STATUS_IGNORED):
        task_status.started()
        while self.state is None:
            try:
                await self.build_index()
            except Exception:
                logger.exception("Building the P2P serving index failed; retrying in %.0f s", CATCH_UP_SECONDS)
                await trio.sleep(CATCH_UP_SECONDS)
        since_rebuild = 0.0
        while True:
            await trio.sleep(CATCH_UP_SECONDS)
            since_rebuild += CATCH_UP_SECONDS
            try:
                if since_rebuild >= REBUILD_SECONDS:
                    since_rebuild = 0.0
                    await self.build_index()
                else:
                    await self.catch_up()
            except Exception:
                logger.exception("Refreshing the P2P serving index failed; keeping the previous one")

    def find(self, signature, duration):
        state = self.state
        if state is None:
            return None
        found = {}
        for index in (state.main, state.delta):
            for item_id, distance in index.find(signature, duration, limit=LOOKUP_CANDIDATES):
                if item_id not in found or distance < found[item_id]:
                    found[item_id] = distance
        return sorted(found.items(), key=lambda item: item[1])[:LOOKUP_CANDIDATES]

    def _prune_tokens(self):
        now = self._clock()
        for token in [token for token, entry in self._tokens.items() if entry.expires <= now]:
            self._tokens.pop(token, None)

    def _issue(self, item_id, peer, keys):
        self._prune_tokens()
        if len(self._tokens) >= MAX_TOKENS:
            return None
        token = secrets.token_bytes(wire.TOKEN_BYTES)
        self._tokens[token] = Token(item_id, str(peer), self._clock() + TOKEN_SECONDS, tuple(keys))
        return token

    def usable_keys(self, requested_spaces, wants):
        return tuple(
            key for key in wants
            if self.spaces.get(key) and requested_spaces.get(key) == self.spaces.get(key)
        )

    async def answer_query(self, peer, query):
        self._count('queries')
        if not self._limits.queries.allow():
            self._count('busy')
            return wire.encode_busy()
        own = self.spaces.get(wire.OUT_MUSICNN)
        if not own or query.spaces.get(wire.OUT_MUSICNN) != own:
            return wire.encode_no(wire.NO_SPACE)
        keys = self.usable_keys(query.spaces, query.wants)
        if not keys:
            return wire.encode_no(wire.NO_SPACE)
        matches = self.find(query.signature, query.duration)
        if matches is None:
            return wire.encode_no(wire.NO_BUILDING)
        if not matches:
            return wire.encode_no(wire.NO_NOT_FOUND)
        rows = await trio.to_thread.run_sync(self._store.serve_rows, [item_id for item_id, _ in matches])
        candidates = []
        for item_id, _distance in matches:
            row = rows.get(item_id)
            if row is None:
                continue
            candidate = self._candidate(peer, row, keys)
            if candidate is None:
                continue
            if candidate is False:
                self._count('busy')
                return wire.encode_busy()
            candidates.append(candidate)
            if len(candidates) >= wire.MAX_CANDIDATES:
                break
        if not candidates:
            return wire.encode_no(wire.NO_OUTPUTS)
        self._count('offers')
        return wire.encode_offer(candidates)

    def _candidate(self, peer, row, keys):
        m_version = musicnn_version(row.origin)
        if m_version is None or not compare.musicnn_ok(row.embedding):
            return None
        served = servable_outputs(row, keys, self.spaces, self._codebook_crc)
        if not served:
            return None
        token = self._issue(row.item_id, peer, served.keys())
        if token is None:
            return False
        versions = {wire.OUT_MUSICNN: m_version}
        versions.update({key: version for key, (version, _value) in served.items()})
        chromaprint = row.chromaprint or b''
        if len(chromaprint) > wire.MAX_CHROMAPRINT_BYTES:
            chromaprint = b''
        embedding = compare.as_vector(row.embedding, wire.MUSICNN_DIM)
        return wire.Candidate(token, embedding, row.duration, versions, chromaprint)

    def take_token(self, peer, token):
        self._prune_tokens()
        entry = self._tokens.get(bytes(token))
        if entry is None or entry.peer != str(peer):
            return None
        return self._tokens.pop(bytes(token))

    async def answer_fetch(self, peer, fetch):
        self._count('fetches')
        entry = self.take_token(peer, fetch.token)
        if entry is None:
            return wire.encode_no(wire.NO_TOKEN)
        keys = tuple(key for key in self.usable_keys(fetch.spaces, fetch.wants) if key in entry.keys)
        if not keys:
            return wire.encode_no(wire.NO_SPACE)
        rows = await trio.to_thread.run_sync(self._store.serve_rows, [entry.item_id])
        row = rows.get(entry.item_id)
        if row is None or musicnn_version(row.origin) is None:
            return wire.encode_no(wire.NO_OUTPUTS)
        served = servable_outputs(row, keys, self.spaces, self._codebook_crc)
        if not served:
            return wire.encode_no(wire.NO_OUTPUTS)
        items = {
            key: wire.Output(self.spaces[key], version, value)
            for key, (version, value) in served.items()
        }
        self._count('served_outputs', len(items))
        logger.info(
            "P2P served %s to an AudioMuse-AI server (versions %s)", ','.join(sorted(items)),
            ', '.join(f"{key}={item.version}" for key, item in sorted(items.items())),
        )
        return wire.encode_outputs(items)


    def answer_peers(self):
        onions = self._peers_provider() if self._peers_provider is not None else []
        return wire.encode_peers([wire.PeerEntry(onion.encode('ascii'), ()) for onion in onions])

    async def _fetch_with_slot(self, io, peer, message):
        try:
            self._limits.uploads.acquire_on_behalf_of_nowait(io)
        except trio.WouldBlock:
            self._count('busy')
            await io.write(wire.encode_busy())
            return
        try:
            await io.write(await self.answer_fetch(peer, message))
        finally:
            self._limits.uploads.release_on_behalf_of(io)

    async def handle_stream(self, io, peer):
        for _ in range(MAX_FRAMES_PER_STREAM):
            frame = None
            with trio.move_on_after(STREAM_IDLE_SECONDS):
                try:
                    frame = await wire.read_frame(io)
                except wire.UnknownVersion:
                    await io.write(wire.encode_no(wire.NO_VERSION))
                    return
                except wire.UnknownType:
                    await io.write(wire.encode_no(wire.NO_NOT_FOUND))
                    return
            if frame is None:
                return
            frame_type, message = frame
            if frame_type == wire.T_QUERY:
                await io.write(await self.answer_query(peer, message))
            elif frame_type == wire.T_FETCH:
                await self._fetch_with_slot(io, peer, message)
            elif frame_type == wire.T_PEERS:
                await io.write(self.answer_peers())
            elif frame_type == wire.T_END:
                return
            else:
                await io.write(wire.encode_no(wire.NO_NOT_FOUND))
                return
