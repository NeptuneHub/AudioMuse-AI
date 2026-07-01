# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Build and refresh the user's "radio" playlists on the media server.

Batch job that regenerates every enabled alchemy radio by running song_alchemy
against each radio's anchor and pushing the result to the media server.

Main Features:
* Generates tracks per radio from its stored anchor, result count, and
  temperature, skipping radios that yield no results.
* Upserts each playlist, falling back to create_playlist when the provider does
  not support create_or_replace_playlist, and returns a created/failed summary.
"""

import logging

from .song_alchemy import song_alchemy
from .mediaserver import create_or_replace_playlist, create_playlist

logger = logging.getLogger(__name__)


def run_radio_playlists():
    from database import get_alchemy_radios

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
                temperature=float(radio['temperature']),
            )
            item_ids = [r['item_id'] for r in (outcome.get('results') or []) if r.get('item_id')]
            if item_ids:
                generated.append((playlist_name, item_ids))
            else:
                failed.append(playlist_name)
                logger.warning(
                    f"Radio '{radio['name']}' produced no results; skipping playlist creation."
                )
        except Exception:
            failed.append(playlist_name)
            logger.exception(f"Radio '{radio['name']}' failed; skipping playlist creation.")

    created = 0
    for playlist_name, item_ids in generated:
        try:
            try:
                create_or_replace_playlist(playlist_name, item_ids)
            except NotImplementedError:
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
