"""MCP tool dispatcher.

This module owns:
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
    _lyrics_search_sync,
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
                tool_args.get("item_ids"),
            )
        elif tool_name == "ai_brainstorm":
            return _ai_brainstorm_sync(
                tool_args["user_request"], ai_config, tool_args.get("get_songs", 200)
            )
        elif tool_name == "lyrics_search":
            return _lyrics_search_sync(
                tool_args.get("description", ""),
                tool_args.get("get_songs", 200),
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
