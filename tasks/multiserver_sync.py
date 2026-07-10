# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Match the analyzed library against secondary media servers.

After analysis runs against the default server, this sweep pulls each enabled
secondary server's catalogue and matches it to the local ``score`` rows using
the same tiered matcher as provider migration (MusicBrainz id, then normalised
path, path tail, and metadata). Every confident pairing is stored in
``track_server_map`` so playlists can be translated to that server later. Tracks
that do not match are simply left unmapped.

Main Features:
* ``sweep_server`` matches one secondary server and upserts its id mappings.
* ``sweep_all_secondary_servers`` sweeps every enabled non-default server and is
  safe to re-run (upsert-only) at the end of every analysis.
"""

import logging
import os

import config
from database import connect_raw
from tasks import audio_fingerprint, provider_probe
from tasks.mediaserver import registry
from tasks.provider_migration_matcher import match_tracks

logger = logging.getLogger(__name__)


def _attach_secondary_fingerprints(server, tracks, db):
    """Download and fingerprint each not-yet-mapped track so the fingerprint tier fires.

    Expensive (one download per track); gated by MULTISERVER_SWEEP_FINGERPRINT.
    Already-mapped provider ids are skipped so re-sweeps only fingerprint new tracks.
    """
    from tasks import mediaserver

    server_id = server['server_id']
    cur = db.cursor()
    try:
        cur.execute("SELECT provider_track_id FROM track_server_map WHERE server_id = %s", (server_id,))
        mapped = {str(r[0]) for r in cur.fetchall()}
    finally:
        cur.close()

    bound = mediaserver.for_server(server_id)
    fingerprinted = 0
    for track in tracks:
        pid = track.get('id')
        if not pid or str(pid) in mapped:
            continue
        path = None
        try:
            path = bound.download_track(config.TEMP_DIR, {'Id': pid, 'id': pid})
            if not path:
                continue
            fp = audio_fingerprint.canonical_fingerprint_file(path)
            if fp is not None:
                track['fingerprint'] = fp
                fingerprinted += 1
        except Exception:
            logger.debug("Secondary fingerprint failed for %s on %s", pid, server_id, exc_info=True)
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    if fingerprinted:
        logger.info("Fingerprinted %d tracks on secondary server %s", fingerprinted, server['name'])
    return fingerprinted


def _load_local_rows(conn):
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT item_id, title, author, album, album_artist, file_path, mbid, fingerprint FROM score"
        )
        return [
            {
                'item_id': r[0],
                'title': r[1],
                'author': r[2],
                'album': r[3],
                'album_artist': r[4],
                'file_path': r[5],
                'mbid': r[6],
                'fingerprint': r[7],
            }
            for r in cur.fetchall()
        ]
    finally:
        cur.close()


def sweep_server(server_id, conn=None):
    """Match the local library against one secondary server and store the mapping."""
    own_conn = conn is None
    db = conn or connect_raw()
    try:
        server = registry.get_server(server_id, conn=db)
        if server is None:
            return {'server_id': server_id, 'error': 'unknown server'}
        if server['is_default']:
            return {'server_id': server_id, 'skipped': 'default', 'matched': 0}
        if not server['enabled']:
            return {'server_id': server_id, 'skipped': 'disabled', 'matched': 0}

        stype = server['server_type']
        logger.info("Multi-server sweep: matching library against '%s' (%s)", server['name'], stype)
        target_tracks = provider_probe.fetch_all_tracks(stype, server['creds'])
        if config.MULTISERVER_SWEEP_FINGERPRINT:
            _attach_secondary_fingerprints(server, target_tracks, db)
        old_rows = _load_local_rows(db)
        result = match_tracks(old_rows, target_tracks)
        mapping = {
            item_id: (new_id, result['match_tiers'].get(item_id))
            for item_id, new_id in result['matches'].items()
        }
        written = registry.upsert_track_maps(server_id, mapping, conn=db)
        logger.info(
            "Multi-server sweep for '%s': mapped %d/%d local tracks (target=%d, tiers=%s)",
            server['name'], written, len(old_rows), len(target_tracks), result['tier_counts'],
        )
        return {
            'server_id': server_id,
            'name': server['name'],
            'server_type': stype,
            'target_tracks': len(target_tracks),
            'local_tracks': len(old_rows),
            'matched': written,
            'tier_counts': result['tier_counts'],
        }
    finally:
        if own_conn:
            db.close()


def sweep_all_secondary_servers(conn=None):
    """Sweep every enabled non-default server; never raises for one bad server."""
    own_conn = conn is None
    db = conn or connect_raw()
    try:
        results = []
        for server in registry.list_servers(conn=db):
            if server['is_default'] or not server['enabled']:
                continue
            try:
                results.append(sweep_server(server['server_id'], conn=db))
            except Exception:
                logger.exception(
                    "Multi-server sweep failed for server %s", server['server_id']
                )
                results.append({'server_id': server['server_id'], 'error': 'sweep failed'})
        return results
    finally:
        if own_conn:
            db.close()
