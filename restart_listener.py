# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Long-running worker-side listener for restart/stop/start control signals.

Subscribes to the Redis restart channel and, on worker containers only, drives
the supervisor actions defined in ``restart_manager`` in response to published
``restart``/``stop``/``start`` messages, reconnecting on failure. Also handles a
``plugin-sync`` signal, pre-installing plugin code and pip dependencies into this
worker's own volume so the apply restart reloads fast.

Main Features:
* Redis pub/sub loop with automatic reconnect and health checks.
* Acts only when ``SERVICE_TYPE`` is ``worker``, ignoring other roles.
* Pre-installs plugin dependencies on ``plugin-sync`` in a background thread.
"""

import logging
import os
import threading
import time

from redis import Redis
import config
from app_logging import configure_logging
from restart_manager import (
    RESTART_CHANNEL,
    restart_supervisor_workers,
    stop_supervisor_workers,
    start_supervisor_workers,
)

logger = logging.getLogger(__name__)
configure_logging()

try:
    from plugin.manager import worker_presync
except Exception:
    worker_presync = None
    logger.exception('plugin.manager import failed; plugin-sync signals will be ignored')


def _dispatch_plugin_sync():
    if worker_presync is None:
        logger.warning('plugin-sync received but the plugin subsystem is unavailable; ignoring')
        return

    def _run():
        try:
            worker_presync()
        except Exception:
            logger.exception('Plugin-sync handling on this worker failed')

    threading.Thread(target=_run, name='plugin-sync', daemon=True).start()


def main():
    redis_url = config.REDIS_URL
    channel = os.environ.get('AUDIO_MUSE_CONFIG_RESTART_CHANNEL', RESTART_CHANNEL)
    logger.info('Starting restart listener on channel: %s', channel)

    while True:
        redis_conn = None
        pubsub = None
        try:
            redis_conn = Redis.from_url(
                redis_url,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
                retry_on_timeout=True,
                decode_responses=True,
            )
            pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(channel)
            logger.info('Subscribed to restart channel. Waiting for restart messages...')

            for message in pubsub.listen():
                if not message:
                    continue
                if message.get('type') != 'message':
                    continue
                payload = message.get('data')
                logger.info('Control listener received signal: %s', payload)
                service_type = os.environ.get('SERVICE_TYPE', '').lower()
                if service_type != 'worker':
                    logger.info('Control signal received, but SERVICE_TYPE is not worker; skipping')
                    continue
                if payload == 'restart':
                    logger.info('Restart signal received, restarting worker processes...')
                    if restart_supervisor_workers():
                        logger.info('Worker restart completed successfully')
                    else:
                        logger.warning('Worker restart failed; will continue listening')
                elif payload == 'stop':
                    logger.info('Stop signal received, stopping worker processes...')
                    if stop_supervisor_workers():
                        logger.info('Worker stop completed successfully')
                    else:
                        logger.warning('Worker stop failed; will continue listening')
                elif payload == 'start':
                    logger.info('Start signal received, starting worker processes...')
                    if start_supervisor_workers():
                        logger.info('Worker start completed successfully')
                    else:
                        logger.warning('Worker start failed; will continue listening')
                elif payload == 'plugin-sync':
                    logger.info('Plugin sync signal received; syncing plugins for this worker...')
                    _dispatch_plugin_sync()
        except Exception:
            logger.exception('Restart listener connection error, retrying in 5 seconds')
            time.sleep(5)
        finally:
            try:
                if pubsub is not None:
                    pubsub.close()
            except Exception:
                pass
            try:
                if redis_conn is not None:
                    redis_conn.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
