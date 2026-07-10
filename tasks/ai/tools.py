# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""MCP tool schemas and dispatch for the playlist AI.

Defines the tool surface the LLM sees (seed_search, text_match,
knowledge_lookup, search_database) via get_mcp_tools, and dispatches each
emitted call to the grounded implementations in ``tool_impl``. Sits between
``planner`` (which builds the plan) and the real library queries.

Main Features:
* get_mcp_tools builds the schema dynamically, exposing text_match modes only when CLAP/LYRICS are enabled; tool descriptions carry the routing rules (when to use each tool and when to use a sibling instead) so they work as the primary routing signal for small models, with genre/voice/mood enums from the canonical vocab.
* execute_mcp_tool converts normalized energy 0..1 to raw score units before search_database, expands female/male voice spelling variants deterministically, passes exclude_artists/exclude_genres through as hard SQL cuts, and rejects year-only text_match queries (routing them to search_database); all failures return a generic error, never a traceback.
* Array args carry maxItems/uniqueItems caps so small-model structured output cannot loop a value forever; exclusion fields document that excluded names never go in seeds or positive filters.
"""

import logging
import re
from typing import Dict, List, Optional

import config

from tasks.ai.tool_impl import (
    _ai_brainstorm_sync,
    _artist_similarity_api_sync,
    _database_genre_query_sync,
    _fuzzy_match_author_title,
    _lyrics_search_sync,
    _song_alchemy_sync,
    _song_similarity_api_sync,
    _text_search_sync,
)
from tasks.mcp_helper import get_db_connection as _get_db_connection
from tasks.ai.vocab import GENRE_VOCAB

logger = logging.getLogger(__name__)


_YEAR_ONLY_RE = re.compile(
    r"^(songs?\s+(from\s+)?)?(\d{4})\s*(songs?|music|tracks?)?$",
    re.IGNORECASE,
)

VOICE_ENUM = [v for v in config.VOICE_VOCAB if v.endswith('vocalists')]

_VOICE_SPELLING_GROUPS = [
    [v for v in config.VOICE_VOCAB if v.startswith('female')],
    [v for v in config.VOICE_VOCAB if v.startswith('male')],
]


def _expand_voice_spellings(voices) -> Optional[List[str]]:
    if not voices:
        return None
    out = [v for v in voices if isinstance(v, str) and v.strip()]
    lows = {v.strip().lower() for v in out}
    for group in _VOICE_SPELLING_GROUPS:
        if any(g.lower() in lows for g in group):
            for g in group:
                if g.lower() not in lows:
                    out.append(g)
                    lows.add(g.lower())
    return out or None


def _default_text_mode() -> str:
    if config.CLAP_ENABLED:
        return "audio"
    if config.LYRICS_ENABLED:
        return "lyrics"
    return "audio"


def _seed_to_alchemy_item(seed: Dict) -> Optional[Dict]:
    if not isinstance(seed, dict):
        return None
    stype = (seed.get("type") or "").lower()
    if stype == "artist":
        name = (seed.get("name") or seed.get("artist") or seed.get("id") or "").strip()
        if not name:
            return None
        return {"type": "artist", "id": name}
    if stype == "song":
        title = (seed.get("title") or seed.get("song_title") or "").strip()
        artist = (seed.get("artist") or seed.get("song_artist") or "").strip()
        if not title or not artist:
            return None
        return {"type": "song", "id": f"{title} by {artist}"}
    return None


def _dispatch_seed_search(tool_args: Dict, ai_config: Dict) -> Dict:
    seeds = tool_args.get("seeds") or []
    if not seeds:
        return {"songs": [], "message": "seed_search: no seeds provided"}

    blend_mode = (tool_args.get("blend_mode") or "union").lower()
    get_songs = int(tool_args.get("get_songs", 200) or 200)
    subtract = tool_args.get("subtract") or []

    if blend_mode == "alchemy" or (blend_mode == "subtract" and subtract):
        add_items = [it for it in (_seed_to_alchemy_item(s) for s in seeds) if it]
        sub_items = [it for it in (_seed_to_alchemy_item(s) for s in subtract) if it]
        if blend_mode == "alchemy" and len(add_items) < 2:
            blend_mode = "union"
        elif blend_mode == "subtract" and not sub_items:
            return {
                "songs": [],
                "message": "seed_search(subtract): subtract list was empty after validation",
            }
        else:
            return _song_alchemy_sync(add_items, sub_items, get_songs)

    all_songs: List[Dict] = []
    ids_seen: set = set()
    messages: List[str] = []
    per_seed_budget = max(50, get_songs)

    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        stype = (seed.get("type") or "").lower()
        if stype == "song":
            title = (seed.get("title") or seed.get("song_title") or "").strip()
            artist = (seed.get("artist") or seed.get("song_artist") or "").strip()
            if not title or not artist:
                messages.append(f"seed_search: skipping malformed song seed {seed}")
                continue
            res = _song_similarity_api_sync(title, artist, per_seed_budget)
        elif stype == "artist":
            name = (seed.get("name") or seed.get("artist") or seed.get("id") or "").strip()
            if not name:
                messages.append(f"seed_search: skipping malformed artist seed {seed}")
                continue
            res = _artist_similarity_api_sync(name, 15, per_seed_budget)
        else:
            messages.append(f"seed_search: unknown seed type '{stype}', skipping")
            continue

        if res.get("message"):
            messages.append(res["message"])
        for s in res.get("songs", []) or []:
            iid = s.get("item_id")
            if iid and iid not in ids_seen:
                all_songs.append(s)
                ids_seen.add(iid)

    if not all_songs:
        return {
            "songs": [],
            "message": "seed_search(union) found no songs across seeds\n" + "\n".join(messages),
        }
    return {
        "songs": all_songs[: get_songs * len(seeds)],
        "message": f"seed_search(union) collected {len(all_songs)} unique songs across {len(seeds)} seed(s)\n"
        + "\n".join(messages),
    }


def _dispatch_text_match(tool_args: Dict, ai_config: Dict) -> Dict:
    query = (tool_args.get("query") or "").strip()
    if not query:
        return {"songs": [], "message": "text_match: empty query"}

    mode = (tool_args.get("mode") or _default_text_mode()).lower()
    get_songs = int(tool_args.get("get_songs", 200) or 200)

    if _YEAR_ONLY_RE.match(query):
        return {
            "songs": [],
            "message": (
                f"text_match rejected: '{query}' is a metadata query (year). "
                "Use search_database with year_min/year_max instead."
            ),
        }

    if mode == "lyrics":
        return _lyrics_search_sync(query, get_songs)

    return _text_search_sync(
        query,
        tool_args.get("tempo_filter"),
        tool_args.get("energy_filter"),
        get_songs,
    )


_SEARCH_OTHER_FILTER_KEYS = (
    "genres", "moods", "tempo_min", "tempo_max",
    "energy_min", "energy_max", "key", "scale",
    "year_min", "year_max", "min_rating",
    "album", "other_features", "candidate_item_ids",
    "voices", "instrumental",
    "exclude_artists", "exclude_genres",
)


def _scale_energy(raw) -> Optional[float]:
    if raw is None:
        return None
    return config.ENERGY_MIN + float(raw) * (config.ENERGY_MAX - config.ENERGY_MIN)


def _has_other_search_filters(tool_args: Dict) -> bool:
    return any(tool_args.get(k) for k in _SEARCH_OTHER_FILTER_KEYS)


def _retry_artist_substring(do_query, artist_arg):
    logger.info(
        "search_database exact match returned 0 songs; retrying with "
        "whole-word artist match for '%s'",
        artist_arg,
    )
    result = do_query(artist_arg, fuzzy=True)
    songs = result.get("songs", [])
    if songs:
        msg = result.get("message", "")
        result["message"] = (
            f"{msg}\n(artist relaxed to whole-word match: "
            f"author contains the word '{artist_arg}')"
        )
    return result, songs


def _retry_fuzzy_artist(do_query, artist_arg, prev_result):
    try:
        db_conn = _get_db_connection()
        try:
            fuzzy_hit = _fuzzy_match_author_title(db_conn, artist_arg)
        finally:
            db_conn.close()
        if fuzzy_hit and fuzzy_hit.get("author"):
            canonical = fuzzy_hit["author"]
            logger.info(
                "Fuzzy-matched artist '%s' -> '%s' (score %s); re-querying",
                artist_arg, canonical, fuzzy_hit.get("score", "?"),
            )
            result = do_query(canonical, fuzzy=False)
            if result.get("songs", []):
                msg = result.get("message", "")
                result["message"] = (
                    f"{msg}\n(artist fuzzy-matched: "
                    f"'{artist_arg}' -> '{canonical}')"
                )
            return result
    except Exception:
        logger.warning(
            "Fuzzy artist fallback failed for '%s'", artist_arg, exc_info=True
        )
    return prev_result


def _dispatch_search_database(tool_args: Dict) -> Dict:
    energy_min_raw = _scale_energy(tool_args.get("energy_min"))
    energy_max_raw = _scale_energy(tool_args.get("energy_max"))
    artist_arg = tool_args.get("artist")
    album_arg = tool_args.get("album")

    def _do_query(name, fuzzy: bool = False) -> Dict:
        return _database_genre_query_sync(
            tool_args.get("genres"),
            tool_args.get("get_songs", 200),
            tool_args.get("moods"),
            tool_args.get("tempo_min"),
            tool_args.get("tempo_max"),
            energy_min_raw,
            energy_max_raw,
            tool_args.get("key"),
            tool_args.get("scale"),
            tool_args.get("year_min"),
            tool_args.get("year_max"),
            tool_args.get("min_rating"),
            album_arg,
            name,
            other_features=tool_args.get("other_features"),
            candidate_item_ids=tool_args.get("candidate_item_ids"),
            voices=_expand_voice_spellings(tool_args.get("voices")),
            score_threshold=tool_args.get("score_threshold"),
            instrumental=tool_args.get("instrumental"),
            fuzzy_match=fuzzy,
            exclude_artists=tool_args.get("exclude_artists"),
            exclude_genres=tool_args.get("exclude_genres"),
        )

    result = _do_query(artist_arg, fuzzy=False)
    songs = result.get("songs", [])

    if not songs and artist_arg:
        result, songs = _retry_artist_substring(_do_query, artist_arg)

    if not songs and artist_arg and not _has_other_search_filters(tool_args):
        result = _retry_fuzzy_artist(_do_query, artist_arg, result)

    return result


def execute_mcp_tool(tool_name: str, tool_args: Dict, ai_config: Dict) -> Dict:
    try:
        if tool_name == "seed_search":
            return _dispatch_seed_search(tool_args, ai_config)

        if tool_name == "text_match":
            return _dispatch_text_match(tool_args, ai_config)

        if tool_name == "knowledge_lookup":
            request = tool_args.get("user_request") or tool_args.get("query") or ""
            return _ai_brainstorm_sync(request, ai_config, tool_args.get("get_songs", 200))

        if tool_name == "search_database":
            return _dispatch_search_database(tool_args)

        return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.exception("Error executing MCP tool")
        return {"error": f"Tool execution error: {str(e)}"}


def get_mcp_tools() -> List[Dict]:
    from config import CLAP_ENABLED, LYRICS_ENABLED

    text_match_modes = ["audio"] if CLAP_ENABLED else []
    if LYRICS_ENABLED:
        text_match_modes.append("lyrics")

    tools: List[Dict] = [
        {
            "name": "seed_search",
            "description": (
                "Find songs that sound SIMILAR to named seed songs or artists. "
                "Use when the user asks for something LIKE a name they gave: "
                "'similar to X', 'songs like X', 'sounds like X', 'X meets Y', 'X but not Y'. "
                "An artist seed returns that artist plus related artists, so for an artist's "
                "OWN songs ('songs by X', 'play X') use search_database with artist instead. "
                "Seeds can mix songs and artists in one call."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seeds": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "description": (
                            "Seed songs and/or artists the results should resemble. "
                            "Only names the user wants MORE of; never an excluded name."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["song", "artist"]},
                                "title": {
                                    "type": "string",
                                    "description": "Song title (when type='song'), e.g. 'Hotel California'",
                                },
                                "artist": {
                                    "type": "string",
                                    "description": "Artist of that song (when type='song'), e.g. 'Eagles'",
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Artist name (when type='artist'), e.g. 'Nina Simone'",
                                },
                            },
                            "required": ["type"],
                        },
                    },
                    "blend_mode": {
                        "type": "string",
                        "enum": ["union", "alchemy", "subtract"],
                        "default": "union",
                        "description": (
                            "union (default): songs similar to each seed, merged. "
                            "alchemy: one blended flavor of 2+ seeds ('X meets Y'). "
                            "subtract: like the seeds minus the 'subtract' items ('X but not Y')."
                        ),
                    },
                    "subtract": {
                        "type": "array",
                        "maxItems": 5,
                        "description": "Items whose flavor to remove (only with blend_mode='subtract'). Same shape as seeds.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["song", "artist"]},
                                "title": {"type": "string"},
                                "artist": {"type": "string"},
                                "name": {"type": "string"},
                            },
                            "required": ["type"],
                        },
                    },
                },
                "required": ["seeds"],
            },
        }
    ]

    if text_match_modes:
        mode_desc_parts = []
        if "audio" in text_match_modes:
            mode_desc_parts.append(
                "mode 'audio': how the music SOUNDS (instruments, texture, atmosphere), "
                "e.g. 'calm solo piano', 'dark heavy synth bass'"
            )
        if "lyrics" in text_match_modes:
            mode_desc_parts.append(
                "mode 'lyrics': what the words are ABOUT (topic, story, scenario), "
                "e.g. 'songs about heartbreak', 'summer road trip'"
            )
        mode_desc = ". ".join(mode_desc_parts)

        tools.append(
            {
                "name": "text_match",
                "description": (
                    f"Find songs from a free-text description. {mode_desc}. "
                    "Use for sound or topic descriptions that plain metadata cannot express. "
                    "Write the query in English; when the request is in another language, "
                    "translate it (the match only understands English). "
                    "For genre, era, tempo, energy, vocals, artist or album use search_database; "
                    "for 'similar to <name>' use seed_search."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Short vivid description of the sound or the lyrical topic. "
                                "ALWAYS in English: when the user wrote another language, "
                                "translate the description (the underlying embeddings only "
                                "understand English)."
                            ),
                        },
                        "mode": {
                            "type": "string",
                            "enum": text_match_modes,
                            "default": text_match_modes[0],
                            "description": "'audio' for how it sounds, 'lyrics' for what the words are about.",
                        },
                    },
                    "required": ["query"],
                },
            }
        )

    tools.append(
        {
            "name": "knowledge_lookup",
            "description": (
                "Answer popularity, cultural or historical requests that need world knowledge: "
                "'best rap of the 90s', '#1 hits of 1985', 'festival anthems', 'Grammy winners 2020', "
                "'one hit wonders'. Turns the request into a grounded search recipe (filters, "
                "sound descriptions, well-known artists) run against this library; it never "
                "invents song titles. When the request only names plain metadata or a specific "
                "artist, use search_database instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The user's popularity/cultural request in their own words.",
                    },
                },
                "required": ["user_request"],
            },
        }
    )

    tools.append(
        {
            "name": "search_database",
            "description": (
                "Filter the library by exact metadata. The tool for an artist's OWN songs "
                "(artist), an album (album), and for genre, vocal type, mood, release year or "
                "decade, tempo BPM, energy, key, scale, rating and instrumental. "
                "Also the ONLY tool for exclusions: 'no X', 'without X', 'except X' go in "
                "exclude_artists/exclude_genres, never in the positive fields. "
                "Fill only the fields the user asked for. Works alone for pure metadata "
                "requests, or alongside seed_search/text_match to constrain their results."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(GENRE_VOCAB)},
                        "maxItems": 5,
                        "uniqueItems": True,
                        "description": "Music genres the user WANTS, e.g. ['jazz'] or ['rock', 'blues'].",
                    },
                    "voices": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(VOICE_ENUM)},
                        "maxItems": 2,
                        "uniqueItems": True,
                        "description": (
                            "Vocal type: 'female vocalists' for any female-voice request, "
                            "'male vocalists' for any male-voice request."
                        ),
                    },
                    "moods": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(config.OTHER_FEATURE_LABELS)},
                        "maxItems": 3,
                        "uniqueItems": True,
                        "description": "How it feels, e.g. ['sad'] or ['danceable', 'party'].",
                    },
                    "tempo_min": {"type": "number", "description": "Min BPM 40-200, e.g. 120"},
                    "tempo_max": {"type": "number", "description": "Max BPM 40-200, e.g. 90"},
                    "energy_min": {
                        "type": "number",
                        "description": "Min energy from 0.0 (calm) to 1.0 (intense), e.g. 0.7",
                    },
                    "energy_max": {
                        "type": "number",
                        "description": "Max energy from 0.0 (calm) to 1.0 (intense), e.g. 0.35",
                    },
                    "key": {
                        "type": "string",
                        "description": "Musical key note name, e.g. 'C', 'F#', 'Eb'",
                    },
                    "scale": {"type": "string", "enum": ["major", "minor"]},
                    "year_min": {
                        "type": "integer",
                        "description": "Earliest release year, e.g. 1990 for '90s'",
                    },
                    "year_max": {
                        "type": "integer",
                        "description": "Latest release year, e.g. 1999 for '90s'",
                    },
                    "min_rating": {"type": "integer", "description": "Minimum user rating 1-5"},
                    "album": {"type": "string", "description": "Album name, e.g. 'Abbey Road'"},
                    "artist": {
                        "type": "string",
                        "description": (
                            "Exact artist name for that artist's OWN songs, e.g. 'Eric Clapton'"
                        ),
                    },
                    "instrumental": {
                        "type": "boolean",
                        "description": "true = only instrumental tracks, false = only tracks with vocals",
                    },
                    "exclude_artists": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                        "uniqueItems": True,
                        "description": (
                            "Artists the user does NOT want ('no 50 Cent' -> ['50 Cent']). "
                            "Hard-removed from the results; never put these in artist or seeds."
                        ),
                    },
                    "exclude_genres": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(GENRE_VOCAB)},
                        "maxItems": 5,
                        "uniqueItems": True,
                        "description": (
                            "Genres the user does NOT want ('no rap' -> ['Hip-Hop']). "
                            "Hard-removed from the results; never put these in genres."
                        ),
                    },
                },
            },
        }
    )

    return tools
