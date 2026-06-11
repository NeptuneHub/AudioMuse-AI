import logging

from .song_alchemy import song_alchemy
from .mediaserver import create_or_replace_playlist, create_playlist

logger = logging.getLogger(__name__)


def run_radio_playlists():
    """Generate one playlist per enabled radio (anchor + temperature + number of results).

    Uses create_or_replace_playlist so the same server-side playlist (and ID,
    where the backend allows) gets reused across runs — avoiding duplicate
    playlists on online-first sync clients (e.g. Symfonium on Navidrome)
    that track playlists by ID.

    Falls back to create_playlist for MPD and other unsupported backends.
    """
    from app_helper import get_alchemy_radios

    radios = [r for r in get_alchemy_radios() if r.get('enabled')]
    logger.info(f"Radio playlist run started for {len(radios)} enabled radios.")

    generated = []
    failed = []
    for radio in radios:
        playlist_name = radio['name']
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

    created = 0
    for playlist_name, item_ids in generated:
        try:
            try:
                create_or_replace_playlist(playlist_name, item_ids)
            except NotImplementedError:
                # MPD or unsupported backend: fall back to plain create.
                create_playlist(playlist_name, item_ids)
            created += 1
            logger.info(f"Radio playlist '{playlist_name}' upserted with {len(item_ids)} tracks.")
        except Exception:
            failed.append(playlist_name)
            logger.exception(f"Failed to upsert playlist '{playlist_name}' on the media server.")

    summary = {
        "message": f"Created {created} radio playlist(s) from {len(radios)} enabled radio(s).",
        "radios_enabled": len(radios),
        "playlists_created": created,
        "failed": failed,
    }
    logger.info(f"Radio playlist run finished: {summary}")
    return summary
