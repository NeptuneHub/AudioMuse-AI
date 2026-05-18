"""MCP tool definitions and dispatcher.

This module owns:
* ``get_mcp_tools()`` -- canonical MCP tool definitions (JSON Schema) used by
  every AI provider when calling with tools.
* ``execute_mcp_tool(...)`` -- dispatcher that runs the actual tool body
  (delegating to the ``_*_sync`` helpers in ``tasks.mcp_tool_impl``).

This module is purely MCP plumbing -- no DB queries, no AI calls.
"""
import logging
import re
from typing import Dict, List

import config

from tasks.mcp_tool_impl import (
    _ai_brainstorm_sync,
    _artist_similarity_api_sync,
    _database_genre_query_sync,
    _song_alchemy_sync,
    _song_similarity_api_sync,
    _text_search_sync,
)

logger = logging.getLogger(__name__)


def execute_mcp_tool(tool_name: str, tool_args: Dict, ai_config: Dict) -> Dict:
    """Execute an MCP tool. Returns the tool's result dict."""
    try:
        if tool_name == "artist_similarity":
            return _artist_similarity_api_sync(
                tool_args["artist"], 15, tool_args.get("get_songs", 200)
            )
        elif tool_name == "text_search":
            desc = tool_args.get("description", "")
            # Reject metadata-only queries that CLAP can't handle meaningfully.
            if re.match(
                r"^(songs?\s+(from\s+)?)?(\d{4})\s*(songs?|music|tracks?)?$",
                desc.strip(),
                re.IGNORECASE,
            ):
                return {
                    "songs": [],
                    "message": (
                        f"text_search rejected: '{desc}' is a metadata query (year), not "
                        "an audio description. Use search_database with year_min/year_max instead."
                    ),
                }
            return _text_search_sync(
                desc,
                tool_args.get("tempo_filter"),
                tool_args.get("energy_filter"),
                tool_args.get("get_songs", 200),
            )
        elif tool_name == "song_similarity":
            return _song_similarity_api_sync(
                tool_args["song_title"],
                tool_args["song_artist"],
                tool_args.get("get_songs", 200),
            )
        elif tool_name == "song_alchemy":
            add_items = tool_args.get("add_items", [])
            subtract_items = tool_args.get("subtract_items", [])

            def normalize_items(items):
                if not items:
                    return []
                normalized = []
                for item in items:
                    if isinstance(item, str):
                        normalized.append({"type": "artist", "id": item})
                    elif isinstance(item, dict):
                        normalized.append(item)
                return normalized

            return _song_alchemy_sync(
                normalize_items(add_items),
                normalize_items(subtract_items),
                tool_args.get("get_songs", 200),
            )
        elif tool_name == "search_database":
            # Convert normalized energy (0-1) to raw energy scale.
            energy_min_raw = None
            energy_max_raw = None
            e_min = tool_args.get("energy_min")
            e_max = tool_args.get("energy_max")
            if e_min is not None:
                e_min = float(e_min)
                energy_min_raw = config.ENERGY_MIN + e_min * (
                    config.ENERGY_MAX - config.ENERGY_MIN
                )
            if e_max is not None:
                e_max = float(e_max)
                energy_max_raw = config.ENERGY_MIN + e_max * (
                    config.ENERGY_MAX - config.ENERGY_MIN
                )

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
                tool_args.get("album"),
                tool_args.get("artist"),
            )
        elif tool_name == "ai_brainstorm":
            return _ai_brainstorm_sync(
                tool_args["user_request"], ai_config, tool_args.get("get_songs", 200)
            )
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        # SECURITY: do not interpolate user-controlled `tool_name` into the log
        # message (CodeQL clear-text-logging). The exception traceback already
        # captures the offending call site; the returned error string is for
        # the API caller, not the log.
        logger.exception("Error executing MCP tool")
        return {"error": f"Tool execution error: {str(e)}"}


def get_mcp_tools() -> List[Dict]:
    """Return the list of available MCP tools (6 CORE TOOLS).

    Tool list is filtered by ``config.CLAP_ENABLED`` -- ``text_search`` is only
    exposed when CLAP is available.
    """
    from config import CLAP_ENABLED

    tools = [
        {
            "name": "song_similarity",
            "description": "\U0001f947 PRIORITY #1: MOST SPECIFIC - Find songs similar to a specific song (requires exact title+artist). \u2705 USE when user mentions a SPECIFIC SONG TITLE.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "song_title": {"type": "string", "description": "Song title"},
                    "song_artist": {"type": "string", "description": "Artist name"},
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 200,
                    },
                },
                "required": ["song_title", "song_artist"],
            },
        }
    ]

    if CLAP_ENABLED:
        tools.append(
            {
                "name": "text_search",
                "description": "\U0001f948 PRIORITY #2: HIGH PRIORITY - Natural language search using CLAP. \u2705 USE for: INSTRUMENTS (piano, guitar, ukulele), SOUND DESCRIPTIONS (romantic, dreamy, chill vibes), DESCRIPTIVE QUERIES ('energetic workout'). Supports optional tempo/energy filters for hybrid search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Natural language description (e.g., 'piano music', 'romantic pop', 'ukulele songs', 'energetic guitar rock')",
                        },
                        "tempo_filter": {
                            "type": "string",
                            "enum": ["slow", "medium", "fast"],
                            "description": "Optional: Filter CLAP results by tempo (hybrid mode)",
                        },
                        "energy_filter": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Optional: Filter CLAP results by energy (hybrid mode)",
                        },
                        "get_songs": {
                            "type": "integer",
                            "description": "Number of songs",
                            "default": 200,
                        },
                    },
                    "required": ["description"],
                },
            }
        )

    tools.extend(
        [
            {
                "name": "artist_similarity",
                "description": f"\U0001f949 PRIORITY #{'5' if CLAP_ENABLED else '4'}: Find songs BY an artist AND similar artists. \u2705 USE for: 'songs by/from/like Artist X' including the artist's own songs (call once per artist). \u274c DON'T USE for: 'sounds LIKE multiple artists blended' (use song_alchemy).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "artist": {"type": "string", "description": "Artist name"},
                        "get_songs": {
                            "type": "integer",
                            "description": "Number of songs",
                            "default": 200,
                        },
                    },
                    "required": ["artist"],
                },
            },
            {
                "name": "song_alchemy",
                "description": f"\U0001f3c5 PRIORITY #{'6' if CLAP_ENABLED else '5'}: VECTOR ARITHMETIC - Blend or subtract MULTIPLE artists/songs. REQUIRES 2+ items. Keywords: 'meets', 'combined', 'blend', 'mix of', 'but not', 'without'. \u2705 BEST for: 'play like A + B' ('play like Iron Maiden, Metallica, Deep Purple'), 'like X but NOT Y', 'Artist A meets Artist B', 'mix of A and B'. \u274c DON'T USE for: single artist (use artist_similarity), genre/mood (use search_database). Examples: 'play like Iron Maiden + Metallica + Deep Purple' = add all 3; 'Beatles but not ballads' = add Beatles, subtract ballads.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "add_items": {
                            "type": "array",
                            "description": "Items to ADD (blend into result). Each item: {type: 'song' or 'artist', id: 'artist_name' or 'song_title by artist'}",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["song", "artist"],
                                        "description": "Item type: 'song' or 'artist'",
                                    },
                                    "id": {
                                        "type": "string",
                                        "description": "For artist: 'Artist Name'; For song: 'Song Title by Artist Name'",
                                    },
                                },
                                "required": ["type", "id"],
                            },
                        },
                        "subtract_items": {
                            "type": "array",
                            "description": "Items to SUBTRACT (remove from result). Same format as add_items.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["song", "artist"],
                                        "description": "Item type: 'song' or 'artist'",
                                    },
                                    "id": {
                                        "type": "string",
                                        "description": "For artist: 'Artist Name'; For song: 'Song Title by Artist Name'",
                                    },
                                },
                                "required": ["type", "id"],
                            },
                        },
                        "get_songs": {
                            "type": "integer",
                            "description": "Number of songs",
                            "default": 200,
                        },
                    },
                    "required": ["add_items"],
                },
            },
            {
                "name": "ai_brainstorm",
                "description": f"\U0001f3c5 PRIORITY #{'7' if CLAP_ENABLED else '6'}: AI world knowledge - Use ONLY when other tools CAN'T work. \u2705 USE for: named events (Grammy, Billboard, festivals), cultural knowledge (trending, viral, classic hits), historical significance (best of decade, iconic albums), songs NOT in library. \u274c DON'T USE for: artist's own songs (use artist_similarity), 'sounds like' (use song_alchemy), genre/mood (use search_database), instruments/moods (use text_search if available).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_request": {"type": "string", "description": "User's request"},
                        "get_songs": {
                            "type": "integer",
                            "description": "Number of songs",
                            "default": 200,
                        },
                    },
                    "required": ["user_request"],
                },
            },
            {
                "name": "search_database",
                "description": f"\U0001f396\ufe0f PRIORITY #{'8' if CLAP_ENABLED else '7'}: MOST GENERAL (last resort) - Search by genre/mood/tempo/energy/year/rating/scale filters. \u2705 USE for: genre/mood/tempo combinations when NO specific artists/songs mentioned AND text_search not available/suitable. \u274c DON'T USE if you can use other more specific tools. COMBINE all filters in ONE call! Use 1-3 SPECIFIC genres (not 'rock' which matches everything).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "genres": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Genres (rock, pop, metal, jazz, etc.)",
                        },
                        "moods": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Moods (danceable, aggressive, happy, party, relaxed, sad)",
                        },
                        "tempo_min": {"type": "number", "description": "Min BPM (40-200)"},
                        "tempo_max": {"type": "number", "description": "Max BPM (40-200)"},
                        "energy_min": {
                            "type": "number",
                            "description": "Min energy 0.0 (calm) to 1.0 (intense)",
                        },
                        "energy_max": {
                            "type": "number",
                            "description": "Max energy 0.0 (calm) to 1.0 (intense)",
                        },
                        "key": {
                            "type": "string",
                            "description": "Musical key (C, D, E, F, G, A, B with # or b)",
                        },
                        "scale": {
                            "type": "string",
                            "enum": ["major", "minor"],
                            "description": "Musical scale: major or minor",
                        },
                        "year_min": {
                            "type": "integer",
                            "description": "Earliest release year (e.g. 1990)",
                        },
                        "year_max": {
                            "type": "integer",
                            "description": "Latest release year (e.g. 1999)",
                        },
                        "min_rating": {
                            "type": "integer",
                            "description": "Minimum user rating 1-5",
                        },
                        "album": {
                            "type": "string",
                            "description": "Album name to filter by (e.g. 'Abbey Road', 'Thriller')",
                        },
                        "artist": {
                            "type": "string",
                            "description": "Artist name - returns ONLY songs BY this artist (e.g. 'Madonna', 'Blink-182')",
                        },
                        "get_songs": {
                            "type": "integer",
                            "description": "Number of songs",
                            "default": 200,
                        },
                    },
                },
            },
        ]
    )

    return tools
