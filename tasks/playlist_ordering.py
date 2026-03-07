"""
Playlist Ordering Algorithm
Orders songs for smooth transitions using tempo, energy, and key distance.
Uses a greedy nearest-neighbor approach with a composite distance metric.
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Circle of Fifths order for key distance calculation
# Maps key name -> position on the circle (0-11)
CIRCLE_OF_FIFTHS = {
    'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5,
    'F#': 6, 'GB': 6, 'DB': 7, 'C#': 7, 'AB': 8, 'G#': 8,
    'EB': 9, 'D#': 9, 'BB': 10, 'A#': 10, 'F': 11,
}


def _key_distance(key1: Optional[str], scale1: Optional[str],
                  key2: Optional[str], scale2: Optional[str]) -> float:
    """Calculate distance between two keys on the Circle of Fifths (0-1 normalized).

    Same-scale bonus: if both keys share the same scale (major/minor), distance
    is reduced by 20% to encourage keeping scale consistency.
    """
    if not key1 or not key2:
        return 0.5  # neutral when key data is missing

    pos1 = CIRCLE_OF_FIFTHS.get(key1.upper().replace(' ', ''), None)
    pos2 = CIRCLE_OF_FIFTHS.get(key2.upper().replace(' ', ''), None)

    if pos1 is None or pos2 is None:
        return 0.5

    # Shortest distance around the circle (max 6 steps)
    raw = abs(pos1 - pos2)
    steps = min(raw, 12 - raw)  # 0-6
    dist = steps / 6.0  # normalize to 0-1

    # Same-scale bonus
    if scale1 and scale2 and scale1.lower() == scale2.lower():
        dist *= 0.8

    return dist


def _composite_distance(song_a: Dict, song_b: Dict,
                        w_tempo: float = 0.35,
                        w_energy: float = 0.35,
                        w_key: float = 0.30) -> float:
    """Compute composite distance between two songs.

    Args:
        song_a, song_b: Dicts with keys 'tempo', 'energy', 'key', 'scale'
        w_tempo, w_energy, w_key: Weights (should sum to 1.0)
    """
    # Tempo difference, normalized by typical BPM range (80 BPM span)
    tempo_a = song_a.get('tempo') or 0
    tempo_b = song_b.get('tempo') or 0
    tempo_diff = min(abs(tempo_a - tempo_b) / 80.0, 1.0)

    # Energy difference, normalized by energy range (0.14 span for raw 0.01-0.15)
    energy_a = song_a.get('energy') or 0
    energy_b = song_b.get('energy') or 0
    energy_diff = min(abs(energy_a - energy_b) / 0.14, 1.0)

    # Key distance
    key_dist = _key_distance(
        song_a.get('key'), song_a.get('scale'),
        song_b.get('key'), song_b.get('scale')
    )

    return w_tempo * tempo_diff + w_energy * energy_diff + w_key * key_dist


def order_playlist(song_ids: List[str], energy_arc: bool = False) -> List[str]:
    """Order a list of song IDs for smooth listening transitions.

    Uses greedy nearest-neighbor: start from the song at the 25th percentile
    of energy, then greedily pick the nearest unvisited song.

    Args:
        song_ids: List of item_id strings
        energy_arc: If True, shape an energy arc (gentle start -> peak -> cooldown)

    Returns:
        Reordered list of item_id strings
    """
    if len(song_ids) <= 2:
        return song_ids

    from tasks.mcp_server import get_db_connection
    from psycopg2.extras import DictCursor

    # Fetch song attributes
    db_conn = get_db_connection()
    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            placeholders = ','.join(['%s'] * len(song_ids))
            cur.execute(f"""
                SELECT item_id, tempo, energy, key, scale
                FROM public.score
                WHERE item_id IN ({placeholders})
            """, song_ids)
            rows = cur.fetchall()
    finally:
        db_conn.close()

    if not rows:
        return song_ids

    # Build lookup
    song_data = {}
    for r in rows:
        song_data[r['item_id']] = {
            'tempo': r['tempo'] or 0,
            'energy': r['energy'] or 0,
            'key': r['key'] or '',
            'scale': r['scale'] or '',
        }

    # Only order songs we have data for; keep others at the end
    orderable_ids = [sid for sid in song_ids if sid in song_data]
    unorderable_ids = [sid for sid in song_ids if sid not in song_data]

    if len(orderable_ids) <= 2:
        return song_ids

    # Find starting song: 25th percentile energy (gentle start)
    sorted_by_energy = sorted(orderable_ids, key=lambda sid: song_data[sid]['energy'])
    start_idx = len(sorted_by_energy) // 4  # 25th percentile
    start_id = sorted_by_energy[start_idx]

    # Greedy nearest-neighbor
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

    # Optional energy arc: reorder for gentle start -> peak -> cooldown
    if energy_arc and len(ordered) >= 10:
        ordered = _apply_energy_arc(ordered, song_data)

    return ordered + unorderable_ids


def _apply_energy_arc(ordered_ids: List[str], song_data: Dict) -> List[str]:
    """Reshape ordering for an energy arc: build up -> peak at 60-70% -> cool down.

    Split the smooth-ordered list into low/medium/high energy buckets,
    then interleave: low-start -> medium -> high (peak) -> medium -> low-end.
    """
    n = len(ordered_ids)

    # Sort by energy for bucketing
    by_energy = sorted(ordered_ids, key=lambda sid: song_data[sid]['energy'])

    # Split into 3 segments
    third = n // 3
    low = by_energy[:third]
    mid = by_energy[third:2*third]
    high = by_energy[2*third:]

    # Build arc: low-start -> mid-rise -> high-peak -> mid-fall -> low-end
    half_low = len(low) // 2
    half_mid = len(mid) // 2

    arc = (
        low[:half_low] +           # gentle start
        mid[:half_mid] +           # building
        high +                      # peak
        list(reversed(mid[half_mid:])) +  # cooling
        list(reversed(low[half_low:]))    # gentle end
    )

    return arc
