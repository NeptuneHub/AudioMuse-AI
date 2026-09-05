# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one LISTEN/NOTIFY loop, shared by the workers, the control listener and Flask.

Postgres delivers a NOTIFY to every session currently listening on the channel,
across processes and across containers. That broadcast property is what lets one
Flask container reach N worker containers with a cancel or a restart request and
no broker in between.

The connection is dedicated and autocommit, never borrowed from the pool or from
Flask's ``g``: a listening session sits blocked in ``select()`` most of the time,
so sharing it with query traffic would stall both. It is also the session whose
``application_name`` identifies this process in ``pg_stat_activity``, which is
the only worker registry the queue has.

A notification is a wake-up, never data. Payloads are a queue name or a task id,
because Postgres caps a NOTIFY payload at 8000 bytes and because the row in
``task_status`` is already the authority on what to do. Losing a notification
therefore costs latency and nothing else, which is why the poll fallback in the
worker can be as slow as ``QUEUE_POLL_INTERVAL_SECONDS`` without risking a
stuck queue.

Main Features:
* ``Listener`` owns a dedicated autocommit connection and reconnects on drop
* Blocks in ``select()`` so an idle worker costs no queries at all
* Dispatches ``(channel, payload)`` to a callback on the listener thread
* ``on_ready`` runs once per (re)connection, for callers that must re-check
  state they could not be notified about while they were disconnected
* ``on_idle`` runs at every poll timeout that brought nothing to dispatch; the
  worker uses it to make sure the claim connection of the task it is running
  is still alive, because nothing else touches that connection mid-job
"""

import logging
import select
import threading
import time

import config

logger = logging.getLogger(__name__)


class Listener:
    def __init__(self, channels, on_notify, application_name=None, name='listener',
                 on_ready=None, on_idle=None):
        self._channels = tuple(channels)
        self._on_notify = on_notify
        self._application_name = application_name
        self._name = name
        self._on_ready = on_ready
        self._on_idle = on_idle
        self._thread = None
        self._conn = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        return self._thread

    def _connect(self):
        from database import connect_raw

        conn = connect_raw(
            application_name=self._application_name,
            keepalive_idle_seconds=config.QUEUE_KEEPALIVE_IDLE_SECONDS,
            keepalive_interval_seconds=config.QUEUE_KEEPALIVE_INTERVAL_SECONDS,
            keepalive_count=config.QUEUE_KEEPALIVE_COUNT,
        )
        conn.set_session(autocommit=True)
        with conn.cursor() as cur:
            for channel in self._channels:
                cur.execute('LISTEN {}'.format(_safe_channel(channel)))
        return conn

    def _run(self):
        while True:
            try:
                self._conn = self._connect()
                logger.info(
                    "Listening on %s as %s", ', '.join(self._channels), self._application_name
                )
                if self._on_ready is not None:
                    try:
                        self._on_ready(self._conn)
                    except Exception:
                        logger.exception("Ready handler for %s failed", self._name)
                self._pump()
            except Exception:
                logger.exception("Listen connection failed; reconnecting")
                time.sleep(config.QUEUE_RECONNECT_DELAY_SECONDS)
            finally:
                conn, self._conn = self._conn, None
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        logger.debug("Listen connection close failed", exc_info=True)

    def _pump(self):
        conn = self._conn
        self._dispatch_pending(conn)
        while True:
            if select.select([conn], [], [], config.QUEUE_POLL_INTERVAL_SECONDS) == ([], [], []):
                self._idle()
                continue
            conn.poll()
            self._dispatch_pending(conn)

    def _idle(self):
        if self._on_idle is None:
            return
        try:
            self._on_idle()
        except Exception:
            logger.exception("The idle handler of %s failed", self._name)

    def _dispatch_pending(self, conn):
        while conn.notifies:
            notify = conn.notifies.pop(0)
            try:
                self._on_notify(notify.channel, notify.payload)
            except Exception:
                logger.exception("Notification handler for %s failed", notify.channel)


def _safe_channel(channel):
    if not channel.replace('_', '').isalnum():
        raise ValueError("Refusing to LISTEN on a non-identifier channel name")
    return channel
