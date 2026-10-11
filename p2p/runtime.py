# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The running P2P node: one trio nursery around Tor and the sharing protocol.

node.py imports this module lazily, only once every gate passed, so the idle
node never loads its libraries. The worker's requests arrive on a taskqueue
Listener thread and are handed to trio through a bounded memory channel;
answers go back as NOTIFY parts on each request's reply channel. The status
row is written every 15 s on the session that holds the node's advisory lock,
so a lost lock or a dropped database ends the process and the supervisor
restarts it. The node never writes catalogue tables.

Every AudioMuse-AI server is a Tor onion service (p2p/tor.py): the node starts
Tor, serves the protocol on a socket bound to 127.0.0.1, publishes it as an
onion service and reaches the other servers through Tor. It opens no port on
the network, needs no public IP address and works behind any NAT; no project
server is involved. Servers find each other through a public directory on
Nostr relays (p2p/directory.py), the PEERS answers of other servers and the
list saved in the database.

Main Features:
* Runtime.run: spaces, Tor (its bootstrap percentage is the status reason),
  the local server, the onion service with the stored key (a new key is
  stored once), then the loops below until SIGTERM/SIGINT (or SIGBREAK); Tor
  is stopped on the way out
* IPC pump: supply requests past their ttl are dropped, a full queue or too
  many active requests answers BUSY
* directory loop: this server's record is published every 3 h and the records
  are read every 10 min while fewer than P2P_MAX_PEERS servers answer, hourly
  after that
* probe loop: every minute (and right after a directory read), a PEERS
  exchange with up to 8 servers that are due: never tried, about to go stale,
  or failing after a pause that doubles; an answer keeps a server live and
  teaches the servers it knows; the list is saved every 10 min
* a malformed frame on a stream is logged (7003) and ends that stream, nothing
  else
"""

import logging
import signal
import time
from contextlib import asynccontextmanager
from functools import partial

import trio

import config
from p2p import available, directory, fetch, ipc, limits, peers as p2p_peers, serve
from p2p import spaces as p2p_spaces, store as p2p_store, tor as p2p_tor, wire
from p2p.errors import P2PError
from taskqueue.listen import Listener

logger = logging.getLogger(__name__)

STATUS_SECONDS = 15.0
LOCK_CHECK_EVERY = 4
STARTUP_GRACE_SECONDS = 120.0
PUBLISH_SECONDS = 3 * 3600.0
READ_FAST_SECONDS = 600.0
READ_SLOW_SECONDS = 3600.0
PROBE_LOOP_SECONDS = 60.0
PROBE_BATCH = 8
PROBE_PARALLEL = 4
PROBE_SECONDS = 60.0
CACHE_SAVE_SECONDS = 600.0
REQUEST_QUEUE = 256
MAX_ACTIVE_REQUESTS = 64
REPLY_MARGIN = 0.25
STREAM_CLOSE_SECONDS = 2.0
FINAL_WRITE_SECONDS = 5.0
ANONYMOUS = 'member'
LIBRARY_LOG_LEVEL = logging.WARNING


def valid_supply(message):
    rid = message.get('rid')
    item = message.get('item')
    reply = message.get('reply')
    wants = message.get('wants')
    ttl = message.get('ttl')
    if not isinstance(rid, str) or not rid or len(rid) > 64:
        return None
    if not isinstance(item, str) or not item:
        return None
    if not isinstance(reply, str) or not reply.startswith(ipc.REPLY_CHANNEL_PREFIX):
        return None
    if not isinstance(wants, list) or not all(isinstance(key, str) for key in wants):
        return None
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not 0 < float(ttl) <= ipc.MAX_REQUEST_TTL_SECONDS:
        return None
    return dict(message, wants=[key for key in wants if key in ipc.SUPPLIABLE_OUTPUTS], ttl=float(ttl))


def _stop_signals():
    names = ('SIGTERM', 'SIGINT', 'SIGBREAK')
    return [getattr(signal, name) for name in names if hasattr(signal, name)]


async def _close_stream(stream):
    with trio.CancelScope(shield=True):
        with trio.move_on_after(STREAM_CLOSE_SECONDS):
            try:
                await stream.aclose()
            except Exception:
                logger.debug("Closing a P2P stream failed", exc_info=True)


class Runtime:
    def __init__(self, store):
        self.store = store
        self.started_at = time.monotonic()
        self.reason = 'starting'
        self.running_reason = 'running'
        self.onion = ''
        self.key = None
        self.spaces = {}
        self.nursery = None
        self.tor = None
        self.book = None
        self.server = None
        self.fetcher = None
        self.limits = None
        self.probe_due = None

    def current_state(self):
        if self.book is None or self.server is None or self.server.state is None or not self.onion:
            return ipc.STATE_STARTING
        if self.book.live_peers():
            return ipc.STATE_CONNECTED
        if time.monotonic() - self.started_at < STARTUP_GRACE_SECONDS:
            return ipc.STATE_STARTING
        return ipc.STATE_NO_PEERS

    def status(self):
        status = {
            'state': self.current_state(),
            'reason': self.reason,
            'peer_id': self.onion,
            'version': config.APP_VERSION,
        }
        if self.book is not None:
            status['peers'] = len(self.book.live_peers())
            status['known'] = len(self.book.peers)
        if self.server is not None:
            status['index'] = self.server.size
            status['served'] = dict(self.server.counters)
        if self.fetcher is not None:
            status['requested'] = dict(self.fetcher.counters)
        if self.limits is not None:
            status['bytes'] = dict(self.limits.counters)
        return status

    def _tor_progress(self, percent):
        self.reason = f'connecting to the Tor network ({percent}%)'

    async def run(self):
        logging.getLogger('stem').setLevel(LIBRARY_LOG_LEVEL)
        self.probe_due = trio.Event()
        await trio.to_thread.run_sync(self.store.write_status, {
            'state': ipc.STATE_STARTING, 'reason': 'starting', 'peer_id': '',
        })
        binary = available.tor_binary()
        if binary is None:
            raise P2PError(f'the Tor program ({config.TOR_BINARY}) is not installed')
        spaces, codebook_crc = await trio.to_thread.run_sync(p2p_spaces.compute_spaces)
        missing = [key for key in wire.SPACE_KEYS if spaces.get(key) is None]
        if spaces.get(wire.OUT_MUSICNN) is None:
            self.running_reason = 'the MusiCNN model file is missing; nothing can be matched'
        elif missing:
            self.running_reason = 'running; results not shared for: ' + ','.join(missing)
        self.spaces = spaces
        try:
            self.key = await trio.to_thread.run_sync(self.store.onion_key)
        except ValueError as exc:
            raise P2PError(f'the stored P2P onion key is unreadable ({exc})') from exc
        cache = await trio.to_thread.run_sync(self.store.peer_cache)
        self.limits = limits.Limits()
        self.book = p2p_peers.PeerBook('', spaces, config.P2P_MAX_PEERS)
        self.server = serve.Server(self.store, spaces, codebook_crc, self.limits, peers_provider=self.book.listed)
        self.fetcher = fetch.Fetcher(
            self.store, self.book, spaces, codebook_crc, self.limits, self._open, self._reply,
        )
        logger.info("P2P node starting Tor (%s) for AudioMuse-AI %s", binary, config.APP_VERSION)
        try:
            async with trio.open_nursery() as nursery:
                self.nursery = nursery
                nursery.start_soon(self._signals)
                nursery.start_soon(self._status_loop)
                await self._come_online(binary, cache)
                await nursery.start(self.server.index_loop)
                nursery.start_soon(self._ipc_loop)
                nursery.start_soon(self._directory_loop)
                nursery.start_soon(self._probe_loop)
                nursery.start_soon(self._cache_loop)
        finally:
            with trio.CancelScope(shield=True):
                await self._shutdown()
        logger.info("P2P node stopped")

    async def _come_online(self, binary, cache):
        self.tor = await p2p_tor.start(binary, self._tor_progress)
        listeners = await self.nursery.start(partial(trio.serve_tcp, self._on_connection, 0, host='127.0.0.1'))
        local_port = listeners[0].socket.getsockname()[1]
        self.reason = 'publishing the onion service'
        onion, key = await p2p_tor.publish(self.tor, local_port, self.key)
        if self.key is None:
            key = await trio.to_thread.run_sync(self.store.save_onion_key, key)
        self.key = key
        self.onion = onion
        self.book.own_onion = onion
        self.book.load_cache(cache)
        self.reason = self.running_reason
        logger.info(
            "P2P node online as %s.onion: reached only through Tor, no port open on the network (%d servers known)",
            onion, len(self.book.peers),
        )

    async def _shutdown(self):
        with trio.move_on_after(FINAL_WRITE_SECONDS):
            try:
                if self.book is not None and self.onion:
                    await trio.to_thread.run_sync(self.store.save_peer_cache, self.book.cache_payload())
                await trio.to_thread.run_sync(self.store.write_status, {
                    'state': ipc.STATE_NOT_RUNNING, 'reason': 'stopped', 'peer_id': self.onion,
                })
            except Exception:
                logger.exception("Could not record the P2P node shutdown")
        if self.tor is not None:
            await trio.to_thread.run_sync(self.tor.close)

    async def _signals(self):
        with trio.open_signal_receiver(*_stop_signals()) as receiver:
            async for signum in receiver:
                logger.info("P2P node received signal %s; stopping", signum)
                self.nursery.cancel_scope.cancel()
                return

    async def _status_loop(self):
        round_number = 0
        while True:
            await trio.to_thread.run_sync(self.store.write_status, self.status())
            round_number += 1
            if round_number % LOCK_CHECK_EVERY == 0:
                alive = await trio.to_thread.run_sync(self.store.lock_alive)
                if not alive:
                    raise p2p_store.LockLost('the P2P node no longer holds its advisory lock')
            await trio.sleep(STATUS_SECONDS)

    async def _cache_loop(self):
        while True:
            await trio.sleep(CACHE_SAVE_SECONDS)
            await trio.to_thread.run_sync(self.store.save_peer_cache, self.book.cache_payload())

    async def _ipc_loop(self):
        send, receive = trio.open_memory_channel(REQUEST_QUEUE)
        token = trio.lowlevel.current_trio_token()

        def enqueue(message):
            try:
                send.send_nowait((message, trio.current_time()))
            except trio.WouldBlock:
                logger.exception("The P2P node request queue is full; answering %s with busy", message.get('rid'))
                request = valid_supply(message) if message.get('op') == ipc.OP_SUPPLY else None
                if request is not None:
                    self.nursery.start_soon(self._send_busy, request)

        def on_notify(_channel, payload):
            message = ipc.parse(payload)
            if message is None:
                logger.warning("Ignored a malformed message on the P2P node channel")
                return
            try:
                trio.from_thread.run_sync(enqueue, message, trio_token=token)
            except trio.RunFinishedError:
                return

        Listener(
            [ipc.NODE_CHANNEL], on_notify, application_name=ipc.NODE_APP_NAME, name='p2p-node-ipc',
        ).start()
        active = trio.CapacityLimiter(MAX_ACTIVE_REQUESTS)
        async with trio.open_nursery() as nursery:
            async for message, received in receive:
                op = message.get('op')
                if op != ipc.OP_SUPPLY:
                    logger.warning("Ignored the unknown P2P node operation %r", op)
                    continue
                request = valid_supply(message)
                if request is None:
                    logger.warning("Ignored a malformed P2P supply request %r", message.get('rid'))
                    continue
                deadline = received + request['ttl'] - REPLY_MARGIN
                nursery.start_soon(self._supply, request, deadline, active)

    async def _supply(self, request, deadline, active):
        try:
            active.acquire_on_behalf_of_nowait(request['rid'])
        except trio.WouldBlock:
            logger.exception("Too many P2P requests at once; answering %s with busy", request['rid'])
            await self._send_busy(request)
            return
        try:
            await self.fetcher.handle(request, deadline)
        finally:
            active.release_on_behalf_of(request['rid'])

    async def _send_busy(self, request):
        await self._reply(request['reply'], ipc.supply_reply(request['rid'], ipc.REPLY_BUSY))

    async def _reply(self, channel, reply):
        try:
            parts = ipc.split_reply(reply)
            await trio.to_thread.run_sync(self.store.notify_all, channel, parts)
        except ValueError:
            logger.exception("Could not send the P2P reply %s on %s", reply.get('rid'), channel)

    @asynccontextmanager
    async def _open(self, onion):
        stream = await p2p_tor.connect_onion(self.tor.socks_port, onion)
        try:
            yield self.limits.meter(p2p_tor.StreamIO(stream))
        finally:
            await _close_stream(stream)

    async def _on_connection(self, stream):
        try:
            await self.server.handle_stream(self.limits.meter(p2p_tor.StreamIO(stream)), ANONYMOUS)
        except wire.Violation as exc:
            fetch.log_malformed('an AudioMuse-AI server', 'request', exc)
        except (limits.StreamClosed, trio.BrokenResourceError, trio.ClosedResourceError):
            logger.debug("A P2P request stream ended early", exc_info=True)
        except Exception:
            logger.exception("Serving a P2P request failed")
        finally:
            await _close_stream(stream)

    async def _directory_loop(self):
        relays = directory.relay_names()
        if not relays:
            logger.info("P2P directory off (P2P_DIRECTORY_RELAYS is empty): only the servers already known are used")
            return
        signing = directory.signing_key(self.key)
        published = None
        while True:
            try:
                if published is None or trio.current_time() - published >= PUBLISH_SECONDS:
                    event = directory.build_event(signing, self.onion, config.APP_VERSION, self.spaces)
                    results = await directory.publish(self.tor.socks_port, relays, event)
                    accepted = [host for host, ok in results.items() if ok]
                    if accepted:
                        published = trio.current_time()
                        logger.info("This server is listed on %d of %d directory relays", len(accepted), len(relays))
                    else:
                        logger.warning("No directory relay accepted this server's record; retrying later")
                records = await directory.fetch(self.tor.socks_port, relays)
                for record in records:
                    self.book.learn(record.onion, record.spaces, record.version, record.created_at)
                logger.info("P2P directory: %d AudioMuse-AI servers listed, %d known", len(records), len(self.book.peers))
                self.probe_due.set()
            except p2p_store.LockLost:
                raise
            except Exception:
                logger.exception("A P2P directory round failed; retrying later")
            fast = len(self.book.live_peers()) < self.book.max_peers
            await trio.sleep(READ_FAST_SECONDS if fast else READ_SLOW_SECONDS)

    async def _probe_loop(self):
        limiter = trio.CapacityLimiter(PROBE_PARALLEL)
        while True:
            due = self.book.to_probe(PROBE_BATCH)
            if due:
                async with trio.open_nursery() as nursery:
                    for onion in due:
                        nursery.start_soon(self._probe, onion, limiter)
            with trio.move_on_after(PROBE_LOOP_SECONDS):
                await self.probe_due.wait()
            self.probe_due = trio.Event()

    async def _probe(self, onion, limiter):
        async with limiter:
            listed = ()
            try:
                with trio.fail_after(PROBE_SECONDS):
                    async with self._open(onion) as io:
                        await io.write(wire.encode_peers([]))
                        frame_type, message = await wire.read_frame(io)
                        await io.write(wire.encode_end())
                if frame_type == wire.T_PEERS:
                    listed = message.peers
            except wire.Violation as exc:
                fetch.log_malformed(onion, 'PEERS answer', exc)
                self.book.failed(onion)
                return
            except Exception as exc:
                self.book.failed(onion)
                logger.info("AudioMuse-AI server %s did not answer through Tor: %s", onion,
                            directory.failure_reason(exc))
                return
            peer = self.book.peers.get(onion)
            was_live = peer is not None and self.book.is_live(peer)
            self.book.succeeded(onion)
            if not was_live:
                logger.info("Reached AudioMuse-AI server %s through Tor (%d servers answer)",
                            onion, len(self.book.live_peers()))
            for entry in listed:
                try:
                    self.book.learn(bytes(entry.peer_id).decode('ascii'))
                except UnicodeDecodeError as exc:
                    fetch.log_malformed(onion, 'PEERS entry', exc)


def main(store):
    trio.run(Runtime(store).run)
