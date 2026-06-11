"""MCP tool definitions and dispatcher.

This module owns:
* ``get_mcp_tools()`` -- canonical MCP tool definitions (JSON Schema) used by
  every AI provider when calling with tools.
* ``execute_mcp_tool(...)`` -- dispatcher that runs the actual tool body
  (delegating to the ``_*_sync`` helpers in ``tasks.ai.tool_impl``).

Surface: 4 LLM-facing tools that collapse the previous 7 into shapes a small
self-hosted model (qwen2.5:7b-9b on Ollama) can pick reliably:

* ``seed_search``     -- replaces song_similarity / artist_similarity / song_alchemy
* ``text_match``      -- replaces text_search / lyrics_search
* ``knowledge_lookup`` -- renamed ai_brainstorm
* ``search_database`` -- unchanged metadata filter

This module is purely MCP plumbing -- no DB queries, no AI calls.
"""
import logging
import re
from typing import Dict, List, Optional

import config

from tasks.ai.tool_impl import (
    _ai_brainstorm_sync,
    _artist_similarity_api_sync,
    _database_genre_query_sync,
    _lyrics_search_sync,
    _song_alchemy_sync,
    _song_similarity_api_sync,
    _text_search_sync,
)

logger = logging.getLogger(__name__)


_YEAR_ONLY_RE = re.compile(
    r"^(songs?\s+(from\s+)?)?(\d{4})\s*(songs?|music|tracks?)?$",
    re.IGNORECASE,
)


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
        "songs": all_songs[:get_songs * len(seeds)],
        "message": f"seed_search(union) collected {len(all_songs)} unique songs across {len(seeds)} seed(s)\n" + "\n".join(messages),
    }


def _dispatch_text_match(tool_args: Dict, ai_config: Dict) -> Dict:
    query = (tool_args.get("query") or "").strip()
    if not query:
        return {"songs": [], "message": "text_match: empty query"}

    mode = (tool_args.get("mode") or "audio").lower()
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


def execute_mcp_tool(tool_name: str, tool_args: Dict, ai_config: Dict) -> Dict:
    """Execute an MCP tool. Returns the tool's result dict."""
    try:
        if tool_name == "seed_search":
            return _dispatch_seed_search(tool_args, ai_config)

        if tool_name == "text_match":
            return _dispatch_text_match(tool_args, ai_config)

        if tool_name == "knowledge_lookup":
            request = tool_args.get("user_request") or tool_args.get("query") or ""
            return _ai_brainstorm_sync(request, ai_config, tool_args.get("get_songs", 200))

        if tool_name == "search_database":
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
                other_features=tool_args.get("other_features"),
                candidate_item_ids=tool_args.get("candidate_item_ids"),
                voices=tool_args.get("voices"),
                score_threshold=tool_args.get("score_threshold"),
                instrumental=tool_args.get("instrumental"),
            )

        return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.exception("Error executing MCP tool")
        return {"error": f"Tool execution error: {str(e)}"}


def get_mcp_tools() -> List[Dict]:
    """Return the LLM-facing tool list. Gated by CLAP_ENABLED / LYRICS_ENABLED."""
    from config import CLAP_ENABLED, LYRICS_ENABLED

    text_match_modes = ["audio"] if CLAP_ENABLED else []
    if LYRICS_ENABLED:
        text_match_modes.append("lyrics")

    tools: List[Dict] = [
        {
            "name": "seed_search",
            "description": (
                "Find songs from one or more SEED songs/artists. Use this for: "
                "'similar to X', 'songs like A and B', 'sounds like X meets Y', 'X but not Y'. "
                "Supports multiple songs and/or artists as seeds in a single call. "
                "Use blend_mode='union' (default) for 'similar to A and similar to B'; "
                "use blend_mode='alchemy' for vector-blend ('A meets B', requires 2+ seeds); "
                "use blend_mode='subtract' to remove a flavor ('A but not Y', requires 'subtract')."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "seeds": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Seed songs and/or artists.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["song", "artist"]},
                                "title": {"type": "string", "description": "Song title (when type='song')"},
                                "artist": {"type": "string", "description": "Artist name (when type='song')"},
                                "name": {"type": "string", "description": "Artist name (when type='artist')"},
                            },
                            "required": ["type"],
                        },
                    },
                    "blend_mode": {
                        "type": "string",
                        "enum": ["union", "alchemy", "subtract"],
                        "default": "union",
                        "description": "union (default): similar to each seed, results merged. alchemy: vector blend (needs 2+ seeds). subtract: remove items in 'subtract'.",
                    },
                    "subtract": {
                        "type": "array",
                        "description": "Items to subtract (only with blend_mode='subtract'). Same shape as seeds.",
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
                    "get_songs": {"type": "integer", "default": 200},
                },
                "required": ["seeds"],
            },
        }
    ]

    if text_match_modes:
        mode_desc_parts = []
        if "audio" in text_match_modes:
            mode_desc_parts.append("'audio' (default): match sound/instruments/textures. Include 'instrumental' in the query to find instrumental-sounding tracks ('calm instrumental piano', 'epic orchestral instrumental').")
        if "lyrics" in text_match_modes:
            mode_desc_parts.append("'lyrics': match lyrical themes ('songs about heartbreak', 'lyrics about freedom').")
        mode_desc = ". ".join(mode_desc_parts)

        tools.append(
            {
                "name": "text_match",
                "description": (
                    "Semantic text search. "
                    f"{mode_desc}. "
                    "DO NOT use for year/genre/mood metadata (use search_database). "
                    "Year-only queries like '2024 songs' are rejected."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text description of the audio or lyrical theme.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": text_match_modes,
                            "default": text_match_modes[0],
                            "description": "'audio' for sound/instruments, 'lyrics' for lyrical themes.",
                        },
                        "tempo_filter": {
                            "type": "string",
                            "enum": ["slow", "medium", "fast"],
                            "description": "Optional tempo filter (audio mode only).",
                        },
                        "energy_filter": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Optional energy filter (audio mode only).",
                        },
                        "get_songs": {"type": "integer", "default": 200},
                    },
                    "required": ["query"],
                },
            }
        )

    tools.append(
        {
            "name": "knowledge_lookup",
            "description": (
                "World-knowledge fallback. USE ONLY when the library can't surface the answer "
                "via seed_search or search_database. Good for: 'Grammy winners 2020', "
                "'#1 hits of 1985', 'songs sampled by Daft Punk', 'best festival anthems'. "
                "Returns AI-suggested songs that are then matched against the library."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The user's cultural / historical query in their own words.",
                    },
                    "get_songs": {"type": "integer", "default": 200},
                },
                "required": ["user_request"],
            },
        }
    )

    tools.append(
        {
            "name": "search_database",
            "description": (
                "Filter the library by metadata. Use when the user names genres, vocals, "
                "year/decade, tempo, energy, scale, key, rating, album, artist, or instrumental. "
                "For instrumental tracks, set instrumental=true (queries musicnn score). "
                "For non-instrumental, set instrumental=false. "
                "Can stand alone OR refine a seed_search/text_match/knowledge_lookup pool."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(config.STRATIFIED_GENRES)},
                        "description": "Music genres. Queries mood_vector with score > 0.5.",
                    },
                    "voices": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["female vocalists", "female vocalist", "male vocalists"],
                        },
                        "description": (
                            "Vocal type. For 'female voice'/'woman singer' use BOTH "
                            "['female vocalists','female vocalist'] (catalog has both spellings). "
                            "For 'male voice' use ['male vocalists']. Queries mood_vector > 0.5."
                        ),
                    },
                    "moods": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(config.OTHER_FEATURE_LABELS)},
                        "description": (
                            "Real moods. ONLY these 6 are valid: "
                            "danceable, aggressive, happy, party, relaxed, sad. "
                            "Queries other_features > 0.5. Never put genres or vocals here."
                        ),
                    },
                    "tempo_min": {"type": "number", "description": "Min BPM (40-200)"},
                    "tempo_max": {"type": "number", "description": "Max BPM (40-200)"},
                    "energy_min": {"type": "number", "description": "Min energy 0.0 (calm) to 1.0 (intense)"},
                    "energy_max": {"type": "number", "description": "Max energy 0.0 (calm) to 1.0 (intense)"},
                    "key": {"type": "string", "description": "Musical key (C, D, E, F, G, A, B with # or b)"},
                    "scale": {"type": "string", "enum": ["major", "minor"]},
                    "year_min": {"type": "integer", "description": "Earliest release year (e.g. 1990)"},
                    "year_max": {"type": "integer", "description": "Latest release year (e.g. 1999)"},
                    "min_rating": {"type": "integer", "description": "Minimum user rating 1-5"},
                    "album": {"type": "string", "description": "Album name to filter by"},
                    "artist": {"type": "string", "description": "Single artist name (use seed_search for multiple)"},
                    "instrumental": {
                        "type": "boolean",
                        "description": "true = only instrumental tracks. false = only tracks with vocals.",
                    },
                    "get_songs": {"type": "integer", "default": 200},
                },
            },
        }
    )

    return tools
