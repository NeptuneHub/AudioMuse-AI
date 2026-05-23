"""Listening History Timeline blueprint.

Fetches top played songs from the media server and enriches them with
local analysis data (mood, energy, tempo) to display an interactive
timeline of listening activity and mood shifts over time.
"""
import json
import logging
import psycopg2
from flask import Blueprint, render_template, jsonify, request
from psycopg2.extras import DictCursor

from app_helper import get_db
from tz_helper import to_local_str

logger = logging.getLogger(__name__)
timeline_bp = Blueprint('timeline_bp', __name__)


@timeline_bp.route('/timeline')
def timeline_page():
    """
    Listening History Timeline page.
    ---
    tags:
      - Timeline
    summary: HTML page showing listening history timeline with mood shifts.
    responses:
      200:
        description: HTML page rendered.
    """
    return render_template('timeline.html', title='AudioMuse-AI - Listening Timeline', active='timeline')


@timeline_bp.route('/timeline/data')
def timeline_data():
    """
    Fetch listening history data for the timeline chart.
    ---
    tags:
      - Timeline
    summary: Returns top played songs enriched with mood/energy analysis data.
    parameters:
      - name: limit
        in: query
        type: integer
        default: 50
        description: Number of top played songs to fetch.
    responses:
      200:
        description: JSON array of listening history entries.
    """
    from tasks.mediaserver import get_top_played_songs

    limit = request.args.get('limit', 50, type=int)
    limit = min(max(limit, 10), 200)  # Clamp between 10-200

    try:
        items = get_top_played_songs(limit)
    except Exception as e:
        logger.error(f"Timeline: failed to fetch top played songs: {e}")
        return jsonify({'error': 'Failed to fetch listening data from media server'}), 500

    if not items:
        return jsonify({'entries': [], 'summary': {}})

    # Collect item IDs for DB lookup
    item_ids = []
    for item in items:
        item_id = item.get('Id') or item.get('id') or item.get('item_id')
        if item_id:
            item_ids.append(str(item_id))

    # Fetch analysis data from local DB
    analysis_map = {}
    if item_ids:
        try:
            db = get_db()
            with db.cursor(cursor_factory=DictCursor) as cur:
                placeholders = ','.join(['%s'] * len(item_ids))
                cur.execute(
                    f"SELECT item_id, title, author, album, tempo, energy, mood_vector, key, scale "
                    f"FROM score WHERE item_id IN ({placeholders})",
                    item_ids
                )
                for row in cur.fetchall():
                    analysis_map[row['item_id']] = dict(row)
        except Exception as e:
            logger.warning(f"Timeline: DB lookup failed: {e}")

    # Build response entries
    entries = []
    for item in items:
        item_id = str(item.get('Id') or item.get('id') or item.get('item_id') or '')
        play_count = (
            item.get('UserData', {}).get('PlayCount')
            or item.get('playCount')
            or item.get('play_count')
            or 0
        )
        last_played = (
            item.get('UserData', {}).get('LastPlayedDate')
            or item.get('lastPlayed')
            or item.get('last_played')
        )
        title = item.get('Name') or item.get('title') or 'Unknown'
        artist = item.get('AlbumArtist') or item.get('artist') or item.get('author') or 'Unknown'
        album = item.get('Album') or item.get('album') or ''

        # Enrich with local analysis
        analysis = analysis_map.get(item_id, {})
        mood_vector_raw = analysis.get('mood_vector')
        mood = None
        if mood_vector_raw:
            try:
                mood = json.loads(mood_vector_raw) if isinstance(mood_vector_raw, str) else mood_vector_raw
            except (json.JSONDecodeError, TypeError):
                pass

        entry = {
            'item_id': item_id,
            'title': title,
            'artist': artist,
            'album': album,
            'play_count': play_count,
            'last_played': last_played,
            'tempo': analysis.get('tempo'),
            'energy': analysis.get('energy'),
            'mood': mood,
            'key': analysis.get('key'),
            'scale': analysis.get('scale'),
        }
        entries.append(entry)

    # Sort by last_played (most recent first), then by play_count
    entries.sort(key=lambda e: (e['last_played'] or '', e['play_count']), reverse=True)

    # Compute summary stats
    energies = [e['energy'] for e in entries if e['energy'] is not None]
    tempos = [e['tempo'] for e in entries if e['tempo'] is not None]
    summary = {
        'total_tracks': len(entries),
        'total_plays': sum(e['play_count'] for e in entries),
        'avg_energy': round(sum(energies) / len(energies), 2) if energies else None,
        'avg_tempo': round(sum(tempos) / len(tempos), 1) if tempos else None,
    }

    return jsonify({'entries': entries, 'summary': summary})
