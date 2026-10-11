# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Entry point of the supervised p2p-node service (python -m p2p.node).

The same supervisor configuration starts this program in every container, so
the node decides for itself whether to run. It declares the worker role before
config is imported (the node must never run Flask's schema bootstrap), then:
outside the Flask container it idles without touching the database; it waits
for the schema Flask creates, reloads the settings from the database, idles
when P2P sharing is off, reports Unavailable (7001) and idles when the node's
Python libraries or the Tor program are missing (p2p/available.py; the noavx2
image), and takes the node advisory lock on a
dedicated connection, retrying every 30 s, so one database never has two
nodes. Only then is p2p.runtime imported, the one lazy import of the package.

Main Features:
* main(): the idle rules above, then runtime.main on the lock session
* an idle node sleeps forever; the wizard save and a restore restart it
* a runtime failure is logged (7005), recorded in the status row and ends the
  process after a pause so the supervisor restarts it without a tight loop
"""

import logging
import os
import sys
import time

import service_roles

service_roles.declare_worker_role(force=True)

import config  # noqa: E402
from app_logging import configure_logging  # noqa: E402
from error import error_manager  # noqa: E402
from error.error_dictionary import ERR_P2P_NODE_FAILED, ERR_P2P_UNAVAILABLE  # noqa: E402
from p2p import available, ipc  # noqa: E402
from p2p import store as p2p_store  # noqa: E402
from p2p.errors import P2PError  # noqa: E402

logger = logging.getLogger(__name__)

FLASK_SERVICE_TYPE = 'flask'
IDLE_SLEEP_SECONDS = 3600
SCHEMA_RETRY_SECONDS = 10
LOCK_RETRY_SECONDS = 30
FAILURE_PAUSE_SECONDS = 60
GENERIC_FAILURE = 'the P2P node stopped with an error; see the logs (7005)'


def idle_forever(reason):
    logger.info("P2P node idle: %s", reason)
    while True:
        time.sleep(IDLE_SLEEP_SECONDS)


def serves_here():
    return os.environ.get(service_roles.SERVICE_TYPE_ENV, '').lower() == FLASK_SERVICE_TYPE


def _close(conn):
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        logger.exception("Could not close a P2P node database connection")


def wait_for_schema():
    announced = False
    while True:
        conn = None
        try:
            conn = p2p_store.connect()
            if p2p_store.schema_ready(conn):
                return
            if not announced:
                logger.info("P2P node waiting for the database schema (app_config, embedding.output_origin)")
                announced = True
        except Exception:
            logger.exception("P2P node could not reach the database; retrying in %d s", SCHEMA_RETRY_SECONDS)
        finally:
            _close(conn)
        time.sleep(SCHEMA_RETRY_SECONDS)


def write_status(state, reason):
    conn = None
    try:
        conn = p2p_store.connect()
        p2p_store.write_status_on(conn, {'state': state, 'reason': reason, 'peer_id': ''})
    except Exception:
        logger.exception("Could not record the P2P node status '%s'", state)
    finally:
        _close(conn)


def acquire_lock():
    waiting = False
    while True:
        conn = None
        try:
            conn = p2p_store.connect()
            if p2p_store.try_lock(conn):
                return conn
            if not waiting:
                logger.warning(
                    "Another P2P node holds the lock on this database; retrying every %d s",
                    LOCK_RETRY_SECONDS,
                )
                waiting = True
        except Exception:
            logger.exception("P2P node could not take its database lock; retrying in %d s", LOCK_RETRY_SECONDS)
        _close(conn)
        time.sleep(LOCK_RETRY_SECONDS)


def failure_reason(exc):
    pending = [exc]
    while pending:
        item = pending.pop()
        if isinstance(item, P2PError):
            return str(item)
        pending.extend(getattr(item, 'exceptions', None) or ())
    return GENERIC_FAILURE


def run_node(lock_conn):
    from p2p import runtime

    store = p2p_store.Store(lock_conn)
    try:
        runtime.main(store)
    finally:
        store.close()


def main():
    configure_logging()
    if not serves_here():
        idle_forever("P2P sharing runs only in the Flask container")
    wait_for_schema()
    config.refresh_config()
    if not config.P2P_ENABLED:
        idle_forever("P2P sharing is off")
    reason = available.unavailable_reason()
    if reason:
        error_manager.record(
            ERR_P2P_UNAVAILABLE, f"P2P sharing is on but {reason}; the node stays idle", logger=logger,
        )
        write_status(ipc.STATE_UNAVAILABLE, reason)
        idle_forever(reason)
    lock_conn = acquire_lock()
    try:
        run_node(lock_conn)
    except Exception as exc:
        logger.exception("[%s] The P2P node stopped with an error", ERR_P2P_NODE_FAILED)
        write_status(ipc.STATE_ERROR, failure_reason(exc))
        _close(lock_conn)
        time.sleep(FAILURE_PAUSE_SECONDS)
        sys.exit(1)
    _close(lock_conn)


if __name__ == '__main__':
    main()
