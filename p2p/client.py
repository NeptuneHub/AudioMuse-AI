# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The analysis worker's side of the worker-to-node P2P requests.

One Session per album job. It owns one raw autocommit Postgres connection that
LISTENs on a private reply channel; each ask() sends a supply request to the
node over NOTIFY and waits for the reply parts until a monotonic deadline, so a
reply that arrives later is never read and the worker stays the only writer of
analysis data. A peer result is taken as it is once it passes the AudioMuse-AI
version rule and decodes into the stored shape.

Main Features:
* Session.open returns None when P2P sharing is off, a not-live Session when
  the node status is anything but Connected (the caller analyses locally and
  logs every such track), else a live Session listening on its reply channel
* ask waits with select() on the connection socket, calls the job's cancel
  check every loop, keeps reply parts only for the request it is waiting on
  and returns a Supply (peer, versions, decoded outputs) or None
* per output: version_allowed for every version including MusiCNN's, then the
  p2p.compare decode checks (exact size, finite values, the neural blob header
  against the local codebook); a failing output is dropped and logged (7003)
* two consecutive timeouts set skipping for the rest of the album; database
  connectivity errors and TaskCancelled propagate, anything else is logged
  (7002) and ends as None
"""

import logging
import select
import time
from typing import NamedTuple

from psycopg2 import InterfaceError, OperationalError
from psycopg2 import sql as pgsql

import config
import database
from error import error_manager
from error.error_dictionary import ERR_P2P_PEER_REJECTED, ERR_P2P_REQUEST_FAILED
from p2p import compare, ipc, wire
from taskqueue import TaskCancelled
from taskqueue.sql import notify_p2p
from tasks import neural_fingerprint

logger = logging.getLogger(__name__)

CLIENT_APP_NAME = 'audiomuse-p2p-client'
MAX_CONSECUTIVE_TIMEOUTS = 2
WAIT_SLICE_SECONDS = 0.25
OUTPUT_NAMES = {
    ipc.OUT_MUSICNN: 'MusiCNN',
    ipc.OUT_CLAP: 'CLAP',
    ipc.OUT_LYRICS: 'lyrics',
    ipc.OUT_NEURAL: 'neural fingerprint',
}


class Supply(NamedTuple):
    peer: str
    versions: dict
    outputs: dict


class Session:
    def __init__(self, cancel_check, conn=None, channel=None, state=ipc.STATE_NOT_RUNNING):
        self._cancel_check = cancel_check
        self._conn = conn
        self.channel = channel
        self.state = state
        self.timeouts = 0
        self._assembler = ipc.PartAssembler()
        self._codebook_crc = None

    @classmethod
    def open(cls, cancel_check):
        if not config.P2P_ENABLED:
            return None
        conn = None
        try:
            conn = database.connect_raw(
                application_name=CLIENT_APP_NAME,
                keepalive_idle_seconds=config.QUEUE_KEEPALIVE_IDLE_SECONDS,
                keepalive_interval_seconds=config.QUEUE_KEEPALIVE_INTERVAL_SECONDS,
                keepalive_count=config.QUEUE_KEEPALIVE_COUNT,
            )
            conn.autocommit = True
            with conn.cursor() as cur:
                state = ipc.read_status(cur).get('state') or ipc.STATE_NOT_RUNNING
            if state != ipc.STATE_CONNECTED:
                _close_quietly(conn)
                error_manager.record(
                    ERR_P2P_REQUEST_FAILED,
                    f"P2P sharing is on but the P2P node is '{ipc.status_label(state)}'; "
                    "this album is analyzed locally",
                    logger=logger, level=logging.WARNING,
                )
                return cls(cancel_check, state=state)
            channel = ipc.new_reply_channel()
            with conn.cursor() as cur:
                cur.execute(pgsql.SQL('LISTEN {}').format(pgsql.Identifier(channel)))
        except (OperationalError, InterfaceError):
            _close_quietly(conn)
            raise
        except Exception:
            logger.exception(
                "[%s] Could not open the P2P session; this album is analyzed locally",
                ERR_P2P_REQUEST_FAILED,
            )
            _close_quietly(conn)
            return cls(cancel_check, state=ipc.STATE_ERROR)
        logger.info(
            "P2P sharing: this album has songs without CLAP, lyrics or neural fingerprint results; each of "
            "them is looked up on the other servers first (reply channel %s)", channel,
        )
        return cls(cancel_check, conn=conn, channel=channel, state=state)

    @property
    def live(self):
        return self._conn is not None

    @property
    def skipping(self):
        return self.timeouts >= MAX_CONSECUTIVE_TIMEOUTS

    @property
    def reason(self):
        if self._conn is None:
            return f"the P2P node is '{ipc.status_label(self.state)}'"
        if self.skipping:
            return f"the P2P node did not answer {self.timeouts} requests in a row"
        return "the P2P node is connected"

    def ask(self, item_id, server_id, provider_id, wants):
        wanted = [key for key in ipc.SUPPLIABLE_OUTPUTS if key in set(wants or ())]
        if self._conn is None or self.skipping or not wanted:
            return None
        try:
            return self._ask(item_id, server_id, provider_id, wanted)
        except (OperationalError, InterfaceError, TaskCancelled):
            raise
        except Exception:
            logger.exception(
                "[%s] The P2P request for %s failed; it is analyzed locally",
                ERR_P2P_REQUEST_FAILED, item_id,
            )
            return None

    def close(self):
        conn, self._conn = self._conn, None
        _close_quietly(conn)

    def _ask(self, item_id, server_id, provider_id, wanted):
        timeout = float(config.P2P_LOOKUP_TIMEOUT_SECONDS)
        rid = ipc.new_request_id()
        request = ipc.supply_request(rid, item_id, server_id, provider_id, wanted, self.channel, timeout)
        self._drop_pending()
        self._assembler.expect(rid)
        try:
            with self._conn.cursor() as cur:
                notify_p2p(cur, ipc.NODE_CHANNEL, ipc.dumps(request))
            reply = self._wait(time.monotonic() + timeout)
        finally:
            self._assembler.discard(rid)
        if reply is None:
            self.timeouts += 1
            error_manager.record(
                ERR_P2P_REQUEST_FAILED,
                f"No P2P reply for {item_id} within {timeout:g} s "
                f"({self.timeouts} in a row); it is analyzed locally",
                logger=logger, level=logging.WARNING,
            )
            if self.skipping:
                error_manager.record(
                    ERR_P2P_REQUEST_FAILED,
                    f"The P2P node did not answer {self.timeouts} requests in a row; "
                    "the rest of this album is analyzed locally",
                    logger=logger, level=logging.WARNING,
                )
            return None
        self.timeouts = 0
        return self._accept(reply, item_id, wanted)

    def _drop_pending(self):
        self._conn.poll()
        self._conn.notifies.clear()

    def _wait(self, deadline):
        while True:
            self._cancel_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            readable, _, _ = select.select([self._conn], [], [], min(remaining, WAIT_SLICE_SECONDS))
            if not readable:
                continue
            self._conn.poll()
            if time.monotonic() > deadline:
                self._conn.notifies.clear()
                return None
            while self._conn.notifies:
                note = self._conn.notifies.pop(0)
                if note.channel != self.channel:
                    continue
                reply = self._assembler.add(ipc.parse(note.payload))
                if reply is not None:
                    return reply

    def _accept(self, reply, item_id, wanted):
        status = reply.get('status')
        if status == ipc.REPLY_NONE:
            logger.info("No other server has the results for %s; it is analyzed locally", item_id)
            return None
        if status != ipc.REPLY_OK:
            error_manager.record(
                ERR_P2P_REQUEST_FAILED,
                f"The P2P node answered '{status}' for {item_id}; it is analyzed locally",
                logger=logger, level=logging.WARNING,
            )
            return None
        peer = reply.get('peer')
        versions = reply.get('versions')
        outputs = reply.get('outputs')
        if not isinstance(peer, str) or not peer or not isinstance(versions, dict) or not isinstance(outputs, dict):
            _reject(peer, item_id, 'the reply', 'it is malformed')
            return None
        musicnn_version = versions.get(ipc.OUT_MUSICNN)
        if not wire.version_allowed(musicnn_version):
            _reject(peer, item_id, 'every result', f"its MusiCNN version {musicnn_version!r} is not v3.0.0 or newer")
            return None
        kept = {}
        kept_versions = {ipc.OUT_MUSICNN: musicnn_version}
        for key in wanted:
            if key not in outputs:
                continue
            version = versions.get(key)
            if not wire.version_allowed(version):
                _reject(peer, item_id, OUTPUT_NAMES[key], f"its version {version!r} is not v3.0.0 or newer")
                continue
            decoded = self._decode(key, outputs)
            if decoded is None:
                _reject(peer, item_id, OUTPUT_NAMES[key], 'it does not decode into the stored shape')
                continue
            kept.update(decoded)
            kept_versions[key] = version
        if not kept:
            return None
        return Supply(peer, kept_versions, kept)

    def _decode(self, key, outputs):
        if key == ipc.OUT_CLAP:
            vector = ipc.decode_floats(outputs.get(key), config.CLAP_EMBEDDING_DIMENSION)
            return {key: vector} if vector is not None and compare.clap_ok(vector) else None
        if key == ipc.OUT_LYRICS:
            vector = ipc.decode_floats(outputs.get(key), config.LYRICS_EMBEDDING_DIMENSION)
            axis = ipc.decode_floats(outputs.get(ipc.OUT_AXIS), wire.AXIS_DIM)
            if vector is None or axis is None or not compare.lyrics_ok(vector, axis):
                return None
            return {key: vector, ipc.OUT_AXIS: axis}
        blob = ipc.decode_bytes(outputs.get(key))
        if blob is None or not compare.neural_ok(blob, self._local_codebook_crc()):
            return None
        return {key: blob}

    def _local_codebook_crc(self):
        if self._codebook_crc is None:
            try:
                self._codebook_crc = neural_fingerprint.codebook()[1]
            except (OSError, RuntimeError, ValueError, KeyError):
                logger.exception(
                    "[%s] The local neural fingerprint codebook cannot be read; peer neural "
                    "fingerprints cannot be decoded and are dropped",
                    ERR_P2P_PEER_REJECTED,
                )
                return None
        return self._codebook_crc


def _reject(peer, item_id, what, why):
    error_manager.record(
        ERR_P2P_PEER_REJECTED,
        f"Dropped {what} from peer {peer} for {item_id}: {why}",
        logger=logger, level=logging.WARNING,
    )


def _close_quietly(conn):
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        logger.exception("Could not close the P2P session connection")
