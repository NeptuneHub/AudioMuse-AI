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
* Runs against every enabled media server in the requested scope (all or
  default), isolating failures so one server cannot abort the others.
* Upserts each playlist, falling back to create_playlist when the provider does
  not support create_or_replace_playlist, and returns a created/failed summary.
"""

import logging

from .song_alchemy import song_alchemy
from .mediaserver import create_or_replace_playlist, create_playlist

logger = logging.getLogger(__name__)


def run_radio_playlists(server_scope="all"):
    from database import get_alchemy_radios
    from .mediaserver import context, registry

    radios = [r for r in get_alchemy_radios() if r.get('enabled')]
    servers = registry.servers_for_scope(server_scope)
    logger.info(
        "Radio playlist run started for %d radio(s) across %d server(s).",
        len(radios), len(servers),
    )

    failed = []
    created = 0
    for server in servers:
        server_name = server['name'] if server else 'default server'
        try:
            with context.use_server(
                registry.context_for(server['server_id']) if server else None
            ):
                for radio in radios:
                    playlist_name = radio['name']
                    try:
                        outcome = song_alchemy(
                            add_items=[{'type': 'anchor', 'id': radio['anchor_id']}],
                            n_results=int(radio['n_results']),
                            temperature=float(radio['temperature']),
                        )
                        item_ids = [
                            row['item_id']
                            for row in (outcome.get('results') or [])
                            if row.get('item_id')
                        ]
                        if not item_ids:
                            raise ValueError("no tracks available on this server")
                        try:
                            create_or_replace_playlist(playlist_name, item_ids)
                        except NotImplementedError:
                            create_playlist(playlist_name, item_ids)
                        created += 1
                        logger.info(
                            "Radio playlist '%s' upserted on %s with %d tracks.",
                            playlist_name, server_name, len(item_ids),
                        )
                    except Exception:
                        failed.append(
                            f"{playlist_name}@{server_name}" if len(servers) > 1 else playlist_name
                        )
                        logger.exception(
                            "Radio '%s' failed on %s; skipping.", playlist_name, server_name
                        )
        except Exception:
            logger.exception(
                "Radio playlist run failed on %s; continuing with remaining servers.",
                server_name,
            )

    summary = {
        "message": f"Created {created} server radio playlist(s).",
        "radios_enabled": len(radios),
        "servers_enabled": len(servers),
        "playlists_created": created,
        "failed": failed,
    }
    logger.info(f"Radio playlist run finished: {summary}")
    return summary
