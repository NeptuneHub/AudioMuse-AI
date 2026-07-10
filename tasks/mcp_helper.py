# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Database context and connection helpers for the MCP / AI-chat server.

Supplies the MCP server and AI-chat features with a cached, high-level summary
of the music library and with a connection scoped to an optional least-privilege
chat DB user. Sits between those callers and Postgres so the AI side reads
library stats without holding admin credentials.

Main Features:
* get_library_context: single aggregate snapshot (song/artist counts, year span,
  rating coverage, top genres/moods, scales) cached in process until refreshed.
* get_db_connection with _ensure_ai_chat_db_user: connect as the configured
  AI_CHAT_DB_USER when set, auto-creating or resetting that role's password, and
  fall back to the primary DATABASE_URL otherwise.
"""

import logging
from typing import Dict
from urllib.parse import quote, urlparse, urlunparse

import psycopg2
from psycopg2 import OperationalError, sql
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)

_library_context_cache = None


def _build_ai_chat_db_url():
    from config import AI_CHAT_DB_USER_NAME, AI_CHAT_DB_USER_PASSWORD, DATABASE_URL

    if not AI_CHAT_DB_USER_NAME:
        return DATABASE_URL
    parsed = urlparse(DATABASE_URL)
    host = parsed.hostname or ''
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunparse(
        (
            parsed.scheme,
            f"{quote(AI_CHAT_DB_USER_NAME, safe='')}:{quote(AI_CHAT_DB_USER_PASSWORD, safe='')}@{host}",
            parsed.path or '',
            parsed.params or '',
            parsed.query or '',
            parsed.fragment or '',
        )
    )


_ai_chat_db_user_configured = False


def _ensure_ai_chat_db_user():
    global _ai_chat_db_user_configured
    if _ai_chat_db_user_configured:
        return
    from config import AI_CHAT_DB_USER_NAME, AI_CHAT_DB_USER_PASSWORD, DATABASE_URL

    if not AI_CHAT_DB_USER_NAME or not AI_CHAT_DB_USER_PASSWORD:
        return
    try:
        with psycopg2.connect(DATABASE_URL) as admin_conn, admin_conn.cursor() as cur:
            cur.execute('SELECT 1 FROM pg_roles WHERE rolname = %s', (AI_CHAT_DB_USER_NAME,))
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL('CREATE USER {} WITH LOGIN PASSWORD %s').format(
                        sql.Identifier(AI_CHAT_DB_USER_NAME)
                    ),
                    [AI_CHAT_DB_USER_PASSWORD],
                )
            else:
                try:
                    psycopg2.connect(_build_ai_chat_db_url()).close()
                except OperationalError:
                    cur.execute(
                        sql.SQL('ALTER USER {} WITH PASSWORD %s').format(
                            sql.Identifier(AI_CHAT_DB_USER_NAME)
                        ),
                        [AI_CHAT_DB_USER_PASSWORD],
                    )
            dbname = admin_conn.get_dsn_parameters().get('dbname')
            if dbname:
                cur.execute(
                    sql.SQL('GRANT CONNECT ON DATABASE {} TO {}').format(
                        sql.Identifier(dbname), sql.Identifier(AI_CHAT_DB_USER_NAME)
                    )
                )
            cur.execute(
                sql.SQL('GRANT USAGE ON SCHEMA public TO {}').format(
                    sql.Identifier(AI_CHAT_DB_USER_NAME)
                )
            )
            cur.execute(
                sql.SQL('GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}').format(
                    sql.Identifier(AI_CHAT_DB_USER_NAME)
                )
            )
            admin_conn.commit()
            _ai_chat_db_user_configured = True
    except Exception as exc:
        logger.warning('AI chat user setup failed: %s', exc)


def get_db_connection():
    from config import AI_CHAT_DB_USER_NAME

    if AI_CHAT_DB_USER_NAME:
        _ensure_ai_chat_db_user()
        return psycopg2.connect(_build_ai_chat_db_url())
    from config import DATABASE_URL

    return psycopg2.connect(DATABASE_URL)


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
