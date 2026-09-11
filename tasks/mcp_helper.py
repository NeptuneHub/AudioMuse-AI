# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Database context and connection helpers for the MCP / AI-chat server.

Supplies the MCP server and AI-chat features with a cached, high-level summary
of the music library and with a read-only connection to the main database, so
the AI side reads library data without ever being able to write.

Main Features:
* get_library_context: single aggregate snapshot (song/artist counts, year span,
  rating coverage, top genres/moods, scales) cached in process until refreshed.
* get_db_connection: database.connect_raw(read_only=True), the application's one
  connection helper (statement timeout, serial plans, keepalives, an
  application_name) plus the server-side session option
  default_transaction_read_only=on, so every statement on it, autocommit or not,
  runs in a READ ONLY transaction and any write fails in Postgres. psycopg2's
  set_session(readonly=True) is deliberately not used: it only wraps explicit
  transactions in BEGIN READ ONLY and leaves autocommit statements writable. No
  dedicated database role is configured or created for the chat.
"""

import logging
from typing import Dict

from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)

_library_context_cache = None


def get_db_connection():
    from database import connect_raw

    return connect_raw(application_name='audiomuse-ai-chat', read_only=True)


def get_library_context(force_refresh: bool = False) -> Dict:
    global _library_context_cache
    if _library_context_cache is not None and not force_refresh:
        return _library_context_cache

    db_conn = get_db_connection()
    try:
        with db_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                "SELECT COUNT(*) AS cnt, COUNT(DISTINCT author) AS artists FROM public.score"
            )
            row = cur.fetchone()
            total_songs = row['cnt']
            unique_artists = row['artists']

            cur.execute(
                "SELECT MIN(year) AS ymin, MAX(year) AS ymax FROM public.score WHERE year IS NOT NULL AND year > 0"
            )
            yr = cur.fetchone()
            year_min = yr['ymin']
            year_max = yr['ymax']

            cur.execute(
                "SELECT COUNT(*) AS rated FROM public.score WHERE rating IS NOT NULL AND rating > 0"
            )
            rated_count = cur.fetchone()['rated']
            rated_pct = round(100.0 * rated_count / total_songs, 1) if total_songs > 0 else 0

            cur.execute("""
                SELECT split_part(trim(tag), ':', 1) AS name, COUNT(*) AS cnt
                FROM (
                    SELECT unnest(string_to_array(mood_vector, ',')) AS tag
                    FROM public.score
                    WHERE mood_vector IS NOT NULL AND mood_vector != ''
                ) t
                WHERE trim(tag) != ''
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT 15
            """)
            top_genres = [r['name'] for r in cur.fetchall() if r['name']]

            cur.execute(
                "SELECT DISTINCT scale FROM public.score WHERE scale IS NOT NULL AND scale != '' ORDER BY scale"
            )
            scales = [r['scale'] for r in cur.fetchall()]

            cur.execute("""
                SELECT lower(split_part(trim(mood), ':', 1)) AS name, COUNT(*) AS cnt
                FROM (
                    SELECT unnest(string_to_array(other_features, ',')) AS mood
                    FROM public.score
                    WHERE other_features IS NOT NULL AND other_features != ''
                ) t
                WHERE trim(mood) != ''
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT 10
            """)
            top_moods = [r['name'] for r in cur.fetchall() if r['name']]

        ctx = {
            'total_songs': total_songs,
            'unique_artists': unique_artists,
            'top_genres': top_genres,
            'top_moods': top_moods,
            'year_min': year_min,
            'year_max': year_max,
            'has_ratings': rated_count > 0,
            'rated_songs_pct': rated_pct,
            'scales': scales,
        }
        _library_context_cache = ctx
        return ctx
    except Exception as e:
        logger.warning(f"Failed to get library context: {e}")
        return {
            'total_songs': 0,
            'unique_artists': 0,
            'top_genres': [],
            'top_moods': [],
            'year_min': None,
            'year_max': None,
            'has_ratings': False,
            'rated_songs_pct': 0,
            'scales': [],
        }
    finally:
        db_conn.close()
