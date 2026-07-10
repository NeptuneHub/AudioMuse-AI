# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Order tracks within a playlist for a smooth sonic flow.

Consumed by playlist-building features (radio, sonic fingerprint, alchemy) to
sequence a finished set of item ids rather than to select them.

Main Features:
* Greedy nearest-neighbour walk over a composite tempo/energy/key distance,
  seeding from a low-energy track (the first quartile of the energy sort).
* Optional energy-arc reshaping (build-up then wind-down) applied only to
  playlists of ten or more tracks; ids with no score row are appended untouched.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CIRCLE_OF_FIFTHS = {
    'C': 0,
    'G': 1,
    'D': 2,
    'A': 3,
    'E': 4,
    'B': 5,
    'F#': 6,
    'GB': 6,
    'DB': 7,
    'C#': 7,
    'AB': 8,
    'G#': 8,
    'EB': 9,
    'D#': 9,
    'BB': 10,
    'A#': 10,
    'F': 11,
}


def _key_distance(
    key1: Optional[str], scale1: Optional[str], key2: Optional[str], scale2: Optional[str]
) -> float:
    if not key1 or not key2:
        return 0.5

    pos1 = CIRCLE_OF_FIFTHS.get(key1.upper().replace(' ', ''), None)
    pos2 = CIRCLE_OF_FIFTHS.get(key2.upper().replace(' ', ''), None)

    if pos1 is None or pos2 is None:
        return 0.5

    raw = abs(pos1 - pos2)
    steps = min(raw, 12 - raw)
    dist = steps / 6.0

    if scale1 and scale2 and scale1.lower() == scale2.lower():
        dist *= 0.8

    return dist


def _composite_distance(
    song_a: Dict, song_b: Dict, w_tempo: float = 0.35, w_energy: float = 0.35, w_key: float = 0.30
) -> float:
    tempo_a = song_a.get('tempo') or 0
    tempo_b = song_b.get('tempo') or 0
    tempo_diff = min(abs(tempo_a - tempo_b) / 80.0, 1.0)

    energy_a = song_a.get('energy') or 0
    energy_b = song_b.get('energy') or 0
    energy_diff = min(abs(energy_a - energy_b) / 0.14, 1.0)

    key_dist = _key_distance(
        song_a.get('key'), song_a.get('scale'), song_b.get('key'), song_b.get('scale')
    )

    return w_tempo * tempo_diff + w_energy * energy_diff + w_key * key_dist


def order_playlist(song_ids: List[str], energy_arc: bool = False) -> List[str]:
    if len(song_ids) <= 2:
        return song_ids

    from tasks.mcp_helper import get_db_connection
    from psycopg2.extras import DictCursor

    db_conn = get_db_connection()
    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            placeholders = ','.join(['%s'] * len(song_ids))
            cur.execute(
                f"""
                SELECT item_id, tempo, energy, key, scale
                FROM public.score
                WHERE item_id IN ({placeholders})
            """,
                song_ids,
            )
            rows = cur.fetchall()
    finally:
        db_conn.close()

    if not rows:
        return song_ids

    song_data = {}
    for r in rows:
        song_data[r['item_id']] = {
            'tempo': r['tempo'] or 0,
            'energy': r['energy'] or 0,
            'key': r['key'] or '',
            'scale': r['scale'] or '',
        }

    orderable_ids = [sid for sid in song_ids if sid in song_data]
    unorderable_ids = [sid for sid in song_ids if sid not in song_data]

    if len(orderable_ids) <= 2:
        return song_ids

    sorted_by_energy = sorted(orderable_ids, key=lambda sid: song_data[sid]['energy'])
    start_idx = len(sorted_by_energy) // 4
    start_id = sorted_by_energy[start_idx]

    remaining = set(orderable_ids)
    remaining.remove(start_id)
    ordered = [start_id]

    current = start_id
    while remaining:
        best_id = None
        best_dist = float('inf')
        for candidate in remaining:
            d = _composite_distance(song_data[current], song_data[candidate])
            if d < best_dist:
                best_dist = d
                best_id = candidate
        ordered.append(best_id)
        remaining.remove(best_id)
        current = best_id

    if energy_arc and len(ordered) >= 10:
        ordered = _apply_energy_arc(ordered, song_data)

    return ordered + unorderable_ids


def _apply_energy_arc(ordered_ids: List[str], song_data: Dict) -> List[str]:
    n = len(ordered_ids)

    by_energy = sorted(ordered_ids, key=lambda sid: song_data[sid]['energy'])

    third = n // 3
    low = by_energy[:third]
    mid = by_energy[third : 2 * third]
    high = by_energy[2 * third :]

    half_low = len(low) // 2
    half_mid = len(mid) // 2

    arc = (
        low[:half_low]
        + mid[:half_mid]
        + high
        + list(reversed(mid[half_mid:]))
        + list(reversed(low[half_low:]))
    )

    return arc
