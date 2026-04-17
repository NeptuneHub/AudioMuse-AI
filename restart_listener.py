import logging
import os
import time
import traceback

from redis import Redis
import config
from restart_manager import RESTART_CHANNEL, restart_supervisor_workers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s')


def main():
    redis_url = config.REDIS_URL
    channel = os.environ.get('AUDIO_MUSE_CONFIG_RESTART_CHANNEL', RESTART_CHANNEL)
    logger.info('Starting restart listener on channel: %s', channel)

    while True:
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
                logger.info('Restart listener received payload: %s', payload)
                if payload == 'restart':
                    service_type = os.environ.get('SERVICE_TYPE', '').lower()
                    if service_type != 'worker':
                        logger.info('Restart signal received, but SERVICE_TYPE is not worker; skipping worker restart')
                        continue
                    logger.info('Restart signal received, restarting worker processes...')
                    if restart_supervisor_workers():
                        logger.info('Worker restart completed successfully')
                    else:
                        logger.warning('Worker restart failed; will continue listening')
        except Exception:
            logger.error('Restart listener connection error, retrying in 5 seconds')
            traceback.print_exc()
            time.sleep(5)


if __name__ == '__main__':
    main()
