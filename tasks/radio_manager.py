import logging

from .song_alchemy import song_alchemy
from .mediaserver import create_playlist, delete_playlists_by_suffix

logger = logging.getLogger(__name__)

RADIO_PLAYLIST_SUFFIX = '_radio'


def run_radio_playlists():
    """Generate one playlist per enabled radio (anchor + temperature + number of results).

    Runs synchronously, like the sonic fingerprint cron flow: compute all
    playlists first, then delete every existing playlist ending with '_radio',
    then create the new ones on the media server.
    """
    from app_helper import get_alchemy_radios

    radios = [r for r in get_alchemy_radios() if r.get('enabled')]
    logger.info(f"Radio playlist run started for {len(radios)} enabled radios.")

    generated = []
    failed = []
    for radio in radios:
        playlist_name = f"{radio['name']}{RADIO_PLAYLIST_SUFFIX}"
        try:
            outcome = song_alchemy(
                add_items=[{'type': 'anchor', 'id': radio['anchor_id']}],
                n_results=int(radio['n_results']),
                temperature=float(radio['temperature'])
            )
            item_ids = [r['item_id'] for r in (outcome.get('results') or []) if r.get('item_id')]
            if item_ids:
                generated.append((playlist_name, item_ids))
            else:
                failed.append(playlist_name)
                logger.warning(f"Radio '{radio['name']}' produced no results; skipping playlist creation.")
        except Exception:
            failed.append(playlist_name)
            logger.exception(f"Radio '{radio['name']}' failed; skipping playlist creation.")

    try:
        delete_playlists_by_suffix(RADIO_PLAYLIST_SUFFIX)
    except Exception:
        logger.exception(f"Failed to delete old '{RADIO_PLAYLIST_SUFFIX}' playlists; continuing with playlist creation.")

    created = 0
    for playlist_name, item_ids in generated:
        try:
            create_playlist(playlist_name, item_ids)
            created += 1
        except Exception:
            failed.append(playlist_name)
            logger.exception(f"Failed to create playlist '{playlist_name}' on the media server.")

    summary = {
        "message": f"Created {created} radio playlist(s) from {len(radios)} enabled radio(s).",
        "radios_enabled": len(radios),
        "playlists_created": created,
        "failed": failed,
    }
    logger.info(f"Radio playlist run finished: {summary}")
    return summary
