#!/usr/bin/env python3
"""
AudioMuse-AI - Instant Playlist Tool-Calling Performance Test

Benchmarks how well different AI models select the correct MCP tool
when given a natural language playlist request.  Mirrors test_ai_naming.py
structure but tests tool selection instead of text generation.

Sends the unified system prompt + user query to each model, parses the
tool call response, and scores: JSON valid, correct tool, valid args,
pre-execution valid.

Usage:
  python testing_suite/test_instant_playlist.py
  python testing_suite/test_instant_playlist.py --config path/to/config.yaml
  python testing_suite/test_instant_playlist.py --runs 5
  python testing_suite/test_instant_playlist.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import requests
import yaml


# ---------------------------------------------------------------------------
# Valid tool names (authoritative list)
# ---------------------------------------------------------------------------
VALID_TOOL_NAMES = [
    "song_similarity",
    "text_search",
    "artist_similarity",
    "song_alchemy",
    "ai_brainstorm",
    "search_database",
]

# search_database filter keys checked during pre-execution validation
SEARCH_DB_FILTER_KEYS = [
    "genres", "moods", "tempo_min", "tempo_max", "energy_min", "energy_max",
    "key", "scale", "year_min", "year_max", "min_rating",
]


# ---------------------------------------------------------------------------
# Tool definitions (inlined from ai_mcp_client.py:674-904)
# ---------------------------------------------------------------------------
def get_tool_definitions(clap_enabled: bool) -> list[dict]:
    """Return the 6 MCP tool definitions. Mirrors get_mcp_tools()."""
    tools = [
        {
            "name": "song_similarity",
            "description": "PRIORITY #1: MOST SPECIFIC - Find songs similar to a specific song (requires exact title+artist). USE when user mentions a SPECIFIC SONG TITLE.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "song_title": {
                        "type": "string",
                        "description": "Song title"
                    },
                    "song_artist": {
                        "type": "string",
                        "description": "Artist name"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 100
                    }
                },
                "required": ["song_title", "song_artist"]
            }
        }
    ]

    if clap_enabled:
        tools.append({
            "name": "text_search",
            "description": "PRIORITY #2: HIGH PRIORITY - Natural language search using CLAP. USE for: INSTRUMENTS (piano, guitar, ukulele), SOUND DESCRIPTIONS (romantic, dreamy, chill vibes), DESCRIPTIVE QUERIES ('energetic workout'). Supports optional tempo/energy filters for hybrid search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description (e.g., 'piano music', 'romantic pop', 'ukulele songs', 'energetic guitar rock')"
                    },
                    "tempo_filter": {
                        "type": "string",
                        "enum": ["slow", "medium", "fast"],
                        "description": "Optional: Filter CLAP results by tempo (hybrid mode)"
                    },
                    "energy_filter": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Optional: Filter CLAP results by energy (hybrid mode)"
                    },
                    "get_songs": {
                        "type": "integer",
                        "description": "Number of songs",
                        "default": 100
                    }
                },
                "required": ["description"]
            }
        })

    p = '3' if clap_enabled else '2'
    tools.append({
        "name": "artist_similarity",
        "description": f"PRIORITY #{p}: Find songs BY an artist AND similar artists. USE for: 'songs by/from/like Artist X' including the artist's own songs (call once per artist). DON'T USE for: 'sounds LIKE multiple artists blended' (use song_alchemy).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artist": {
                    "type": "string",
                    "description": "Artist name"
                },
                "get_songs": {
                    "type": "integer",
                    "description": "Number of songs",
                    "default": 100
                }
            },
            "required": ["artist"]
        }
    })

    p2 = '4' if clap_enabled else '3'
    tools.append({
        "name": "song_alchemy",
        "description": f"PRIORITY #{p2}: VECTOR ARITHMETIC - Blend or subtract MULTIPLE artists/songs. REQUIRES 2+ items. Keywords: 'meets', 'combined', 'blend', 'mix of', 'but not', 'without'. BEST for: 'play like A + B' ('play like Iron Maiden, Metallica, Deep Purple'), 'like X but NOT Y', 'Artist A meets Artist B', 'mix of A and B'. DON'T USE for: single artist (use artist_similarity), genre/mood (use search_database). Examples: 'play like Iron Maiden + Metallica + Deep Purple' = add all 3; 'Beatles but not ballads' = add Beatles, subtract ballads.",
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
                                "description": "Item type: 'song' or 'artist'"
                            },
                            "id": {
                                "type": "string",
                                "description": "For artist: 'Artist Name'; For song: 'Song Title by Artist Name'"
                            }
                        },
                        "required": ["type", "id"]
                    }
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
                                "description": "Item type: 'song' or 'artist'"
                            },
                            "id": {
                                "type": "string",
                                "description": "For artist: 'Artist Name'; For song: 'Song Title by Artist Name'"
                            }
                        },
                        "required": ["type", "id"]
                    }
                },
                "get_songs": {
                    "type": "integer",
                    "description": "Number of songs",
                    "default": 100
                }
            },
            "required": ["add_items"]
        }
    })

    p3 = '5' if clap_enabled else '4'
    tools.append({
        "name": "ai_brainstorm",
        "description": f"PRIORITY #{p3}: AI world knowledge - Use ONLY when other tools CAN'T work. USE for: named events (Grammy, Billboard, festivals), cultural knowledge (trending, viral, classic hits), historical significance (best of decade, iconic albums), songs NOT in library. DON'T USE for: artist's own songs (use artist_similarity), 'sounds like' (use song_alchemy), genre/mood (use search_database), instruments/moods (use text_search if available).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_request": {
                    "type": "string",
                    "description": "User's request"
                },
                "get_songs": {
                    "type": "integer",
                    "description": "Number of songs",
                    "default": 100
                }
            },
            "required": ["user_request"]
        }
    })

    p4 = '6' if clap_enabled else '5'
    tools.append({
        "name": "search_database",
        "description": f"PRIORITY #{p4}: MOST GENERAL (last resort) - Search by genre/mood/tempo/energy/year/rating/scale filters. USE for: genre/mood/tempo combinations when NO specific artists/songs mentioned AND text_search not available/suitable. DON'T USE if you can use other more specific tools. COMBINE all filters in ONE call!",
        "inputSchema": {
            "type": "object",
            "properties": {
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Genres (rock, pop, metal, jazz, etc.)"
                },
                "moods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Moods (danceable, aggressive, happy, party, relaxed, sad)"
                },
                "tempo_min": {
                    "type": "number",
                    "description": "Min BPM (40-200)"
                },
                "tempo_max": {
                    "type": "number",
                    "description": "Max BPM (40-200)"
                },
                "energy_min": {
                    "type": "number",
                    "description": "Min energy 0.0 (calm) to 1.0 (intense)"
                },
                "energy_max": {
                    "type": "number",
                    "description": "Max energy 0.0 (calm) to 1.0 (intense)"
                },
                "key": {
                    "type": "string",
                    "description": "Musical key (C, D, E, F, G, A, B with # or b)"
                },
                "scale": {
                    "type": "string",
                    "enum": ["major", "minor"],
                    "description": "Musical scale: major or minor"
                },
                "year_min": {
                    "type": "integer",
                    "description": "Earliest release year (e.g. 1990)"
                },
                "year_max": {
                    "type": "integer",
                    "description": "Latest release year (e.g. 1999)"
                },
                "min_rating": {
                    "type": "integer",
                    "description": "Minimum user rating 1-5"
                },
                "get_songs": {
                    "type": "integer",
                    "description": "Number of songs",
                    "default": 100
                }
            }
        }
    })

    return tools


# ---------------------------------------------------------------------------
# System prompt builder (inlined from ai_mcp_client.py:13-83)
# ---------------------------------------------------------------------------
_FALLBACK_GENRES = "rock, pop, metal, jazz, electronic, dance, alternative, indie, punk, blues, hard rock, heavy metal, hip-hop, funk, country, soul"
_FALLBACK_MOODS = "danceable, aggressive, happy, party, relaxed, sad"


def _get_dynamic_genres(library_context: dict | None) -> str:
    """Return genre list from library context, falling back to defaults."""
    if library_context and library_context.get('top_genres'):
        return ', '.join(library_context['top_genres'][:15])
    return _FALLBACK_GENRES


def _get_dynamic_moods(library_context: dict | None) -> str:
    """Return mood list from library context, falling back to defaults."""
    if library_context and library_context.get('top_moods'):
        return ', '.join(library_context['top_moods'][:10])
    return _FALLBACK_MOODS


def build_system_prompt(tools: list[dict], library_context: dict | None = None) -> str:
    """Build the unified system prompt used by ALL AI providers."""
    tool_names = [t['name'] for t in tools]
    has_text_search = 'text_search' in tool_names

    # Build library context section
    lib_section = ""
    if library_context and library_context.get('total_songs', 0) > 0:
        ctx = library_context
        year_range = ''
        if ctx.get('year_min') and ctx.get('year_max'):
            year_range = f"\n- Year range: {ctx['year_min']}-{ctx['year_max']}"
        rating_info = ''
        if ctx.get('has_ratings'):
            rating_info = f"\n- {ctx['rated_songs_pct']}% of songs have ratings (0-5 scale)"
        scale_info = ''
        if ctx.get('scales'):
            scale_info = f"\n- Scales available: {', '.join(ctx['scales'])}"

        lib_section = f"""
=== USER'S MUSIC LIBRARY ===
- {ctx['total_songs']} songs from {ctx['unique_artists']} artists{year_range}{rating_info}{scale_info}
"""

    # Build tool decision tree
    decision_tree = []
    decision_tree.append("1. Specific song+artist mentioned? -> song_similarity")
    if has_text_search:
        decision_tree.append("2. Instruments (piano, guitar, ukulele) or SOUND DESCRIPTIONS (romantic, dreamy, chill vibes)? -> text_search")
        decision_tree.append("3. 'songs by/from/like [ARTIST]'? -> artist_similarity (returns artist's own + similar)")
        decision_tree.append("4. MULTIPLE artists blended ('A meets B', 'A + B', 'like A and B combined') OR negation ('X but not Y', 'X without Y')? -> song_alchemy (REQUIRES 2+ items)")
        decision_tree.append("5. Songs NOT in library, trending, award winners (Grammy, Billboard), cultural knowledge? -> ai_brainstorm")
        decision_tree.append("6. Genre/mood/tempo/energy/year/rating filters? -> search_database (last resort)")
    else:
        decision_tree.append("2. 'songs by/from/like [ARTIST]'? -> artist_similarity (returns artist's own + similar)")
        decision_tree.append("3. MULTIPLE artists blended ('A meets B', 'A + B', 'like A and B combined') OR negation ('X but not Y', 'X without Y')? -> song_alchemy (REQUIRES 2+ items)")
        decision_tree.append("4. Songs NOT in library, trending, award winners (Grammy, Billboard), cultural knowledge? -> ai_brainstorm")
        decision_tree.append("5. Genre/mood/tempo/energy/year/rating filters? -> search_database (last resort)")

    decision_text = '\n'.join(decision_tree)

    prompt = f"""You are an expert music playlist curator. Analyze the user's request and call the appropriate tools to build a playlist of 100 songs.
{lib_section}
=== TOOL SELECTION (most specific -> most general) ===
{decision_text}

=== RULES ===
1. Call one or more tools - each returns songs with item_id, title, and artist
2. song_similarity REQUIRES both title AND artist - never leave empty
3. artist_similarity returns the artist's OWN songs + songs from SIMILAR artists
4. search_database: COMBINE all filters in ONE call. Use for genre/mood/tempo/energy/year/rating
5. For multiple artists: call artist_similarity once per artist, or use song_alchemy to blend
6. Prefer tool calls over text explanations
7. For complex requests, call MULTIPLE tools in ONE turn for better coverage:
   - "relaxing piano jazz" -> text_search("relaxing piano") + search_database(genres=["jazz"])
   - "energetic songs by Metallica and AC/DC" -> artist_similarity("Metallica") + artist_similarity("AC/DC")
8. When a query has BOTH a genre AND a mood from the MOODS list, prefer search_database over text_search:
   - "sad jazz" -> search_database(genres=["jazz"], moods=["sad"])  NOT text_search
   - But "dreamy atmospheric" -> text_search (no specific genre, sound description)

=== VALID search_database VALUES ===
GENRES: {_get_dynamic_genres(library_context)}
MOODS: {_get_dynamic_moods(library_context)}
TEMPO: 40-200 BPM
ENERGY: 0.0 (calm) to 1.0 (intense) - use 0.0-0.35 for low, 0.35-0.65 for medium, 0.65-1.0 for high
SCALE: major, minor
YEAR: year_min/year_max (e.g., 1990-1999 for 90s). For decade requests (80s, 90s), prefer year filters over genres.
RATING: min_rating 1-5 (user's personal ratings)"""

    return prompt


# ---------------------------------------------------------------------------
# Ollama prompt builder (inlined from ai_mcp_client.py:426-466)
# ---------------------------------------------------------------------------
def build_ollama_prompt(user_query: str, tools: list[dict],
                        library_context: dict | None = None) -> str:
    """Build the full Ollama prompt with JSON output instructions."""
    has_text_search = 'text_search' in [t['name'] for t in tools]

    # Build tool parameter descriptions
    tools_list = []
    for tool in tools:
        props = tool['inputSchema'].get('properties', {})
        params_desc = ", ".join([f"{k} ({v.get('type')})" for k, v in props.items()])
        tools_list.append(f"- {tool['name']}: {params_desc}")
    tools_text = "\n".join(tools_list)

    system_prompt = build_system_prompt(tools, library_context)

    # Build examples
    examples = []
    examples.append('"Similar to By the Way by Red Hot Chili Peppers"\n{{"tool_calls": [{{"name": "song_similarity", "arguments": {{"song_title": "By the Way", "song_artist": "Red Hot Chili Peppers", "get_songs": 100}}}}]}}')
    if has_text_search:
        examples.append('"calm piano song"\n{{"tool_calls": [{{"name": "text_search", "arguments": {{"description": "calm piano", "get_songs": 100}}}}]}}')
    examples.append('"songs like blink-182"\n{{"tool_calls": [{{"name": "artist_similarity", "arguments": {{"artist": "blink-182", "get_songs": 100}}}}]}}')
    examples.append('"blink-182 songs"\n{{"tool_calls": [{{"name": "artist_similarity", "arguments": {{"artist": "blink-182", "get_songs": 100}}}}]}}')
    examples.append('"energetic rock"\n{{"tool_calls": [{{"name": "search_database", "arguments": {{"genres": ["rock"], "energy_min": 0.65, "get_songs": 100}}}}]}}')
    examples_text = "\n\n".join(examples)

    prompt = f"""{system_prompt}

=== TOOL PARAMETERS ===
{tools_text}

=== OUTPUT FORMAT (CRITICAL) ===
Return ONLY a valid JSON object with this EXACT format:
{{
  "tool_calls": [
    {{"name": "tool_name", "arguments": {{"param": "value"}}}}
  ]
}}

=== EXAMPLES ===
{examples_text}

Now analyze this request and return ONLY the JSON:
Request: "{user_query}"
"""
    return prompt


# ---------------------------------------------------------------------------
# OpenAI-format payload builder (inlined from ai_mcp_client.py:257-301)
# ---------------------------------------------------------------------------
def build_openai_payload(user_query: str, tools: list[dict], model_id: str,
                         library_context: dict | None = None) -> dict:
    """Build the OpenAI/OpenRouter chat-completion payload with tools."""
    functions = []
    for tool in tools:
        functions.append({
            "type": "function",
            "function": {
                "name": tool['name'],
                "description": tool['description'],
                "parameters": tool['inputSchema']
            }
        })

    system_prompt = build_system_prompt(tools, library_context)

    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "tools": functions,
        "tool_choice": "auto"
    }


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------
def call_ollama_model(model_cfg: dict, user_query: str, tools: list[dict],
                      library_context: dict | None, timeout: int) -> dict:
    """Call an Ollama model and return parsed tool calls."""
    url = model_cfg["url"]
    model_id = model_cfg["model_id"]

    prompt = build_ollama_prompt(user_query, tools, library_context)

    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time

        if 'response' not in result:
            return {"error": "Invalid Ollama response", "elapsed": elapsed, "raw_response": str(result)[:500]}

        response_text = result['response']
        cleaned = response_text.strip()

        # Remove markdown code blocks if present
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        cleaned = cleaned.strip()

        # Strip think tags
        for tag in ["</think>", "[/INST]", "[/THOUGHT]"]:
            if tag in cleaned:
                cleaned = cleaned.split(tag, 1)[-1].strip()

        parsed = json.loads(cleaned)

        # Extract tool_calls from various response shapes
        if isinstance(parsed, dict) and 'tool_calls' in parsed:
            tool_calls = parsed['tool_calls']
        elif isinstance(parsed, list):
            tool_calls = parsed
        elif isinstance(parsed, dict) and 'name' in parsed:
            tool_calls = [parsed]
        else:
            return {"error": "Missing 'tool_calls' field", "elapsed": elapsed,
                    "raw_response": cleaned[:500]}

        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        # Validate structure
        valid_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict) and 'name' in tc:
                if 'arguments' not in tc:
                    tc['arguments'] = {}
                valid_calls.append(tc)

        if not valid_calls:
            return {"error": "No valid tool calls found", "elapsed": elapsed,
                    "raw_response": cleaned[:500]}

        return {"tool_calls": valid_calls, "elapsed": elapsed, "raw_response": response_text}

    except json.JSONDecodeError as e:
        elapsed = time.time() - start_time
        return {"error": f"JSON parse error: {e}", "elapsed": elapsed,
                "raw_response": response_text[:500] if 'response_text' in locals() else ""}
    except requests.exceptions.ConnectionError:
        elapsed = time.time() - start_time
        return {"error": "Connection refused", "elapsed": elapsed, "raw_response": ""}
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {"error": f"Timeout after {timeout}s", "elapsed": elapsed, "raw_response": ""}
    except requests.exceptions.HTTPError as e:
        elapsed = time.time() - start_time
        detail = ""
        try:
            detail = e.response.text[:200]
        except Exception:
            pass
        return {"error": f"HTTP {e.response.status_code}: {detail}", "elapsed": elapsed, "raw_response": ""}
    except Exception as e:
        elapsed = time.time() - start_time
        return {"error": str(e), "elapsed": elapsed, "raw_response": ""}


def call_openai_model(model_cfg: dict, user_query: str, tools: list[dict],
                      library_context: dict | None, timeout: int) -> dict:
    """Call an OpenAI-compatible API and return parsed tool calls."""
    url = model_cfg["url"]
    model_id = model_cfg["model_id"]
    api_key = model_cfg.get("api_key", "")

    payload = build_openai_payload(user_query, tools, model_id, library_context)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if "openrouter" in url.lower():
        headers["HTTP-Referer"] = "https://github.com/NeptuneHub/AudioMuse-AI"
        headers["X-Title"] = "AudioMuse-AI"

    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time

        raw_response = json.dumps(result, indent=2)

        tool_calls = []
        if 'choices' in result and result['choices']:
            message = result['choices'][0].get('message', {})
            if 'tool_calls' in message:
                for tc in message['tool_calls']:
                    if tc.get('type') == 'function':
                        try:
                            args = json.loads(tc['function']['arguments'])
                        except (json.JSONDecodeError, KeyError):
                            args = {}
                        tool_calls.append({
                            "name": tc['function']['name'],
                            "arguments": args
                        })

        if not tool_calls:
            text_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            return {"error": "No tool calls returned", "elapsed": elapsed,
                    "raw_response": text_response[:500] if text_response else raw_response[:500]}

        return {"tool_calls": tool_calls, "elapsed": elapsed, "raw_response": raw_response}

    except requests.exceptions.ConnectionError:
        elapsed = time.time() - start_time
        return {"error": "Connection refused", "elapsed": elapsed, "raw_response": ""}
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {"error": f"Timeout after {timeout}s", "elapsed": elapsed, "raw_response": ""}
    except requests.exceptions.HTTPError as e:
        elapsed = time.time() - start_time
        detail = ""
        try:
            detail = e.response.text[:200]
        except Exception:
            pass
        return {"error": f"HTTP {e.response.status_code}: {detail}", "elapsed": elapsed, "raw_response": ""}
    except Exception as e:
        elapsed = time.time() - start_time
        return {"error": str(e), "elapsed": elapsed, "raw_response": ""}


def call_model(model_cfg: dict, user_query: str, tools: list[dict],
               library_context: dict | None, timeout: int) -> dict:
    """Dispatch to the correct API caller based on provider config."""
    url = model_cfg.get("url", "")
    api_key = model_cfg.get("api_key", "")
    is_openai_format = (
        bool(api_key) or
        "openai" in url.lower() or
        "openrouter" in url.lower()
    )

    if is_openai_format:
        return call_openai_model(model_cfg, user_query, tools, library_context, timeout)
    else:
        return call_ollama_model(model_cfg, user_query, tools, library_context, timeout)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _score_args_quality(selected_tool: str, selected_args: dict, expected_args: dict) -> float:
    """Score argument quality against expected_args from YAML config.

    Returns a float 0.0-1.0 representing how well the arguments match expectations.
    Uses case-insensitive matching to be tolerant of formatting variations.
    """
    if not expected_args:
        return 1.0  # No expected args defined = full marks

    if not isinstance(selected_args, dict):
        return 0.0

    checks = []

    if selected_tool == "song_similarity":
        # Check song_title and song_artist
        if 'song_title' in expected_args:
            actual = (selected_args.get('song_title') or '').lower()
            expected = expected_args['song_title'].lower()
            checks.append(expected in actual or actual in expected)
        if 'song_artist' in expected_args:
            actual = (selected_args.get('song_artist') or '').lower()
            expected = expected_args['song_artist'].lower()
            checks.append(expected in actual or actual in expected)

    elif selected_tool == "search_database":
        # Check genres
        if 'genres' in expected_args:
            actual_genres = [g.lower() for g in (selected_args.get('genres') or [])]
            for exp_genre in expected_args['genres']:
                checks.append(any(exp_genre.lower() in ag for ag in actual_genres))
        # Check moods
        if 'moods' in expected_args:
            actual_moods = [m.lower() for m in (selected_args.get('moods') or [])]
            for exp_mood in expected_args['moods']:
                checks.append(any(exp_mood.lower() in am for am in actual_moods))
        # Check scale
        if 'scale' in expected_args:
            checks.append((selected_args.get('scale') or '').lower() == expected_args['scale'].lower())
        # Check min_rating
        if 'min_rating' in expected_args:
            checks.append(selected_args.get('min_rating') == expected_args['min_rating'])

    elif selected_tool == "artist_similarity":
        if 'artist' in expected_args:
            actual = (selected_args.get('artist') or '').lower()
            expected = expected_args['artist'].lower()
            checks.append(expected in actual or actual in expected)

    elif selected_tool == "song_alchemy":
        # Check that expected artists appear in add_items
        if 'add_items_artists' in expected_args:
            add_items = selected_args.get('add_items') or []
            actual_ids = []
            for item in add_items:
                if isinstance(item, dict):
                    actual_ids.append((item.get('id') or '').lower())
                elif isinstance(item, str):
                    actual_ids.append(item.lower())
            for exp_artist in expected_args['add_items_artists']:
                checks.append(any(exp_artist.lower() in aid for aid in actual_ids))

    if not checks:
        return 1.0  # No checkable fields for this tool = full marks
    return sum(1 for c in checks if c) / len(checks)


def validate_result(result: dict, test_query: dict, tools: list[dict]) -> dict:
    """
    Validate a single model response against the expected outcome.

    Returns a dict with:
        json_valid, correct_tool, valid_args, pre_exec_valid,
        args_quality, composite_score, selected_tool, selected_args,
        all_tools_called
    """
    tool_names = [t['name'] for t in tools]
    expected_tool = test_query['expected_tool']
    acceptable_tools = test_query.get('acceptable_tools', [expected_tool])
    expected_args = test_query.get('expected_args', {})

    # Default: everything fails
    validation = {
        "json_valid": False,
        "correct_tool": False,
        "valid_args": False,
        "pre_exec_valid": False,
        "args_quality": 0.0,
        "composite_score": 0.0,
        "selected_tool": "",
        "selected_args": {},
        "all_tools_called": [],
    }

    if 'error' in result or 'tool_calls' not in result:
        return validation

    tc_list = result.get('tool_calls', [])
    if not tc_list or not isinstance(tc_list, list):
        return validation

    # JSON is valid if we got parseable tool calls
    validation["json_valid"] = True

    # Record all tools called
    all_called = [tc.get('name', '') for tc in tc_list]
    validation["all_tools_called"] = all_called

    # Check if expected tool appears ANYWHERE in tool calls (not just first)
    matched_tc = None
    for tc in tc_list:
        if tc.get('name') == expected_tool:
            matched_tc = tc
            break

    # If ideal tool not found, check acceptable_tools
    if matched_tc is None:
        for tc in tc_list:
            if tc.get('name') in acceptable_tools:
                matched_tc = tc
                break

    # Use matched tool call for scoring, fall back to first
    scoring_tc = matched_tc if matched_tc else tc_list[0]
    selected_tool = scoring_tc.get('name', '')
    selected_args = scoring_tc.get('arguments', {})

    validation["selected_tool"] = selected_tool
    validation["selected_args"] = selected_args

    # Correct tool? Check against acceptable_tools list
    validation["correct_tool"] = (selected_tool in acceptable_tools)

    # Valid args? (check required args and types for the SELECTED tool)
    validation["valid_args"] = _check_args_valid(selected_tool, selected_args, tool_names)

    # Pre-execution valid? (mirrors app_chat.py:448-475)
    validation["pre_exec_valid"] = _check_pre_exec_valid(selected_tool, selected_args)

    # Args quality scoring (0.0-1.0) using expected_args from YAML
    if validation["correct_tool"] and expected_args:
        validation["args_quality"] = _score_args_quality(selected_tool, selected_args, expected_args)
    elif validation["correct_tool"]:
        validation["args_quality"] = 1.0  # Correct tool, no expected_args to check

    # Composite score: tool_correct (50) + args_quality (25) + pre_exec_valid (15) + json_valid (10)
    score = 0.0
    if validation["json_valid"]:
        score += 10.0
    if validation["correct_tool"]:
        score += 50.0
    if validation["correct_tool"]:
        score += 25.0 * validation["args_quality"]
    if validation["pre_exec_valid"]:
        score += 15.0
    validation["composite_score"] = round(score, 1)

    return validation


def _check_args_valid(tool_name: str, args: dict, available_tools: list[str]) -> bool:
    """Check that required arguments are present and of correct type."""
    if not isinstance(args, dict):
        return False
    if tool_name not in available_tools and tool_name not in VALID_TOOL_NAMES:
        return False

    if tool_name == "song_similarity":
        return (isinstance(args.get('song_title'), str) and len(args['song_title']) > 0 and
                isinstance(args.get('song_artist'), str) and len(args['song_artist']) > 0)

    elif tool_name == "text_search":
        return isinstance(args.get('description'), str) and len(args['description']) > 0

    elif tool_name == "artist_similarity":
        return isinstance(args.get('artist'), str) and len(args['artist']) > 0

    elif tool_name == "song_alchemy":
        add_items = args.get('add_items', [])
        if not isinstance(add_items, list) or len(add_items) == 0:
            return False
        # Accept both structured dicts and simple strings
        for item in add_items:
            if isinstance(item, dict):
                if not item.get('type') or not item.get('id'):
                    return False
            elif not isinstance(item, str):
                return False
        return True

    elif tool_name == "ai_brainstorm":
        return isinstance(args.get('user_request'), str) and len(args['user_request']) > 0

    elif tool_name == "search_database":
        # search_database has no required args (but pre_exec checks for filters)
        return True

    return False


def _check_pre_exec_valid(tool_name: str, args: dict) -> bool:
    """Mirror pre-execution validation from app_chat.py."""
    if not isinstance(args, dict):
        return False
    if tool_name == "song_similarity":
        title = args.get('song_title', '')
        artist = args.get('song_artist', '')
        if isinstance(title, str) and isinstance(artist, str):
            return bool(title.strip()) and bool(artist.strip())
        return False

    elif tool_name == "search_database":
        # At least one filter must be present
        return any(args.get(k) for k in SEARCH_DB_FILTER_KEYS)

    # All other tools pass pre-execution validation if they have valid args
    return _check_args_valid(tool_name, args, VALID_TOOL_NAMES)


# ---------------------------------------------------------------------------
# Config helpers (mirrors test_ai_naming.py)
# ---------------------------------------------------------------------------
def apply_defaults(config: dict) -> None:
    """Merge provider defaults (url, api_key) into each model entry."""
    defaults = config.get("defaults", {})
    for model in config.get("models", []):
        provider = model.get("provider", "")
        provider_defaults = defaults.get(provider, {})
        for key, value in provider_defaults.items():
            if key not in model:
                model[key] = value

    # Allow environment variable override for API keys
    env_api_key = os.environ.get('OPENROUTER_API_KEY')
    if env_api_key:
        for model in config.get("models", []):
            if model.get("provider") == "openrouter":
                model["api_key"] = env_api_key


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------
def run_tests(config: dict, dry_run: bool = False) -> tuple[dict, list[dict]]:
    """
    Execute the full test suite.

    Returns:
        (results_dict, test_queries)
        results_dict keys are model names, values have 'runs' list and metadata.
    """
    tc = config["test_config"]
    models = [m for m in config["models"] if m.get("enabled", False)]
    clap_enabled = tc.get("clap_enabled", True)
    library_context = config.get("library_context")

    if not models:
        print("ERROR: No models enabled in configuration.")
        sys.exit(1)

    num_runs = tc["num_runs_per_model"]
    timeout = tc.get("timeout_per_request", 120)

    # Build tool definitions
    tools = get_tool_definitions(clap_enabled)
    tool_names_available = [t['name'] for t in tools]

    # Filter test queries
    all_queries = config.get("test_queries", [])
    test_queries = []
    for q in all_queries:
        if q.get("skip_if_clap_disabled") and not clap_enabled:
            continue
        test_queries.append(q)

    if not test_queries:
        print("ERROR: No test queries after filtering.")
        sys.exit(1)

    num_queries = len(test_queries)

    if dry_run:
        print("=== DRY RUN MODE ===")
        print(f"Would test {len(models)} model(s), {num_queries} queries, {num_runs} run(s) each")
        print(f"CLAP enabled: {clap_enabled}")
        print(f"Tools available: {', '.join(tool_names_available)}")
        print(f"Library context: {'yes' if library_context else 'no'}\n")

        for mi, m in enumerate(models):
            print(f"  Model {mi + 1}: {m['name']} ({m['provider']}) - {m['model_id']}")

        print(f"\n--- System Prompt Preview ---")
        sys_prompt = build_system_prompt(tools, library_context)
        print(sys_prompt[:800])
        print("...\n")

        # Show Ollama prompt for first query
        print(f"--- Ollama Prompt Preview (query 1: \"{test_queries[0]['query']}\") ---")
        ollama_prompt = build_ollama_prompt(test_queries[0]['query'], tools, library_context)
        print(ollama_prompt[:1200])
        print("...\n")

        # Show OpenAI payload for first query
        print(f"--- OpenAI Payload Preview (query 1) ---")
        openai_payload = build_openai_payload(test_queries[0]['query'], tools, "example-model", library_context)
        # Show just messages, not the full tool defs
        print(json.dumps(openai_payload['messages'], indent=2)[:600])
        print(f"  ... + {len(openai_payload['tools'])} tool definitions\n")

        print(f"--- Test Queries ({num_queries}) ---")
        for qi, q in enumerate(test_queries):
            print(f"  {qi + 1:2d}. [{q['category']}] \"{q['query']}\" -> {q['expected_tool']}")

        return {}, test_queries

    # Run tests
    results = {}
    total_models = len(models)
    connection_failures = set()

    for mi, model in enumerate(models):
        model_name = model["name"]
        print(f"[{mi + 1}/{total_models}] Testing: {model_name} ({model['provider']})")

        results[model_name] = {
            "provider": model["provider"],
            "model_id": model["model_id"],
            "url": model["url"],
            "runs": [],
        }

        # Skip if previous connection to same URL failed
        if model["url"] in connection_failures:
            print(f"  Skipping (connection to {model['url']} already failed)\n")
            for qi in range(num_queries):
                for ri in range(num_runs):
                    results[model_name]["runs"].append({
                        "query_index": qi,
                        "query": test_queries[qi]["query"],
                        "expected_tool": test_queries[qi]["expected_tool"],
                        "category": test_queries[qi]["category"],
                        "run_index": ri,
                        "json_valid": False,
                        "correct_tool": False,
                        "valid_args": False,
                        "pre_exec_valid": False,
                        "args_quality": 0.0,
                        "composite_score": 0.0,
                        "selected_tool": "",
                        "selected_args": {},
                        "all_tools_called": [],
                        "raw_response": "",
                        "elapsed": 0,
                        "error": "Skipped (connection failed)",
                    })
            continue

        model_correct = 0
        model_total = 0
        model_times = []
        abort_model = False

        for qi, tq in enumerate(test_queries):
            for ri in range(num_runs):
                if abort_model:
                    results[model_name]["runs"].append({
                        "query_index": qi,
                        "query": tq["query"],
                        "expected_tool": tq["expected_tool"],
                        "category": tq["category"],
                        "run_index": ri,
                        "json_valid": False,
                        "correct_tool": False,
                        "valid_args": False,
                        "pre_exec_valid": False,
                        "args_quality": 0.0,
                        "composite_score": 0.0,
                        "selected_tool": "",
                        "selected_args": {},
                        "all_tools_called": [],
                        "raw_response": "",
                        "elapsed": 0,
                        "error": "Skipped (connection failed)",
                    })
                    continue

                model_total += 1
                status_prefix = f"  Q{qi + 1} Run {ri + 1}/{num_runs}:"

                api_result = call_model(model, tq["query"], tools, library_context, timeout)
                validation = validate_result(api_result, tq, tools)

                elapsed = api_result.get("elapsed", 0)
                error = api_result.get("error")
                raw_resp = api_result.get("raw_response", "")

                run_result = {
                    "query_index": qi,
                    "query": tq["query"],
                    "expected_tool": tq["expected_tool"],
                    "category": tq["category"],
                    "run_index": ri,
                    "json_valid": validation["json_valid"],
                    "correct_tool": validation["correct_tool"],
                    "valid_args": validation["valid_args"],
                    "pre_exec_valid": validation["pre_exec_valid"],
                    "args_quality": validation["args_quality"],
                    "composite_score": validation["composite_score"],
                    "selected_tool": validation["selected_tool"],
                    "selected_args": validation["selected_args"],
                    "all_tools_called": validation["all_tools_called"],
                    "raw_response": raw_resp if isinstance(raw_resp, str) else str(raw_resp),
                    "elapsed": elapsed,
                    "error": error,
                }
                results[model_name]["runs"].append(run_result)

                if error == "Connection refused":
                    print(f"{status_prefix} FAIL  (connection refused)")
                    connection_failures.add(model["url"])
                    abort_model = True
                    continue

                if error:
                    print(f"{status_prefix} ERR   {elapsed:.1f}s  {error}")
                elif validation["correct_tool"]:
                    model_correct += 1
                    model_times.append(elapsed)
                    args_ok = "args OK" if validation["valid_args"] else "args INVALID"
                    print(f"{status_prefix} OK    {elapsed:.1f}s  {validation['selected_tool']} ({args_ok})")
                else:
                    model_times.append(elapsed)
                    print(f"{status_prefix} WRONG {elapsed:.1f}s  got={validation['selected_tool']}  expected={tq['expected_tool']}")

        # Model summary
        if model_total > 0 and not abort_model:
            avg_t = sum(model_times) / len(model_times) if model_times else 0
            rate = model_correct / model_total * 100
            print(f"  Result: {model_correct}/{model_total} correct ({rate:.1f}%), avg {avg_t:.1f}s\n")
        elif abort_model:
            print(f"  Result: Aborted (connection failed)\n")

    return results, test_queries


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_summary_table(results: dict, timestamp: str) -> str:
    """Generate the ASCII summary table."""
    lines = []
    lines.append("=" * 105)
    lines.append(f" RESULTS - Instant Playlist Tool-Calling Test ({timestamp})")
    lines.append("=" * 105)
    lines.append(f" {'Model':<22} {'Total':>5}  {'JSON OK':>7}  {'Tool OK':>7}  {'Args OK':>7}  {'Rate':>6}  {'Score':>6}  {'Avg Time':>8}")
    lines.append("-" * 105)

    for model_name, model_data in results.items():
        all_runs = model_data["runs"]
        total = len(all_runs)
        json_ok = sum(1 for r in all_runs if r["json_valid"])
        tool_ok = sum(1 for r in all_runs if r["correct_tool"])
        args_ok = sum(1 for r in all_runs if r["correct_tool"] and r["valid_args"])
        rate = (tool_ok / total * 100) if total > 0 else 0
        avg_composite = sum(r.get("composite_score", 0) for r in all_runs) / total if total > 0 else 0
        times = [r["elapsed"] for r in all_runs if r["error"] is None]
        avg_t = sum(times) / len(times) if times else 0

        lines.append(
            f" {model_name:<22} {total:>5}  {json_ok:>7}  {tool_ok:>7}  {args_ok:>7}  {rate:>5.1f}%  {avg_composite:>5.1f}  {avg_t:>7.1f}s"
        )

    lines.append("-" * 105)
    return "\n".join(lines)


def generate_query_detail_table(results: dict, test_queries: list[dict], num_runs: int) -> str:
    """Generate per-query detail table."""
    lines = []
    model_names = list(results.keys())

    # Group queries by category
    categories = {}
    for qi, tq in enumerate(test_queries):
        cat = tq["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((qi, tq))

    for cat, queries in categories.items():
        lines.append(f"\n=== Category: {cat} ===")
        for qi, tq in queries:
            lines.append(f"\n  Q{qi + 1}: \"{tq['query']}\" -> expected: {tq['expected_tool']}")
            for model_name in model_names:
                runs = [r for r in results[model_name]["runs"] if r["query_index"] == qi]
                correct = sum(1 for r in runs if r["correct_tool"])
                total = len(runs)
                tools_selected = [r["selected_tool"] or "(none)" for r in runs]
                tools_str = ", ".join(tools_selected[:5])
                lines.append(f"    {model_name:<22} {correct}/{total}  [{tools_str}]")

    return "\n".join(lines)


def generate_html_report(results: dict, test_queries: list[dict],
                         num_runs: int, timestamp: str, config: dict,
                         save_raw: bool, system_prompt: str) -> str:
    """Generate a self-contained HTML report."""
    model_names = list(results.keys())
    num_queries = len(test_queries)

    # Build summary rows
    summary_rows = ""
    for model_name, model_data in results.items():
        all_runs = model_data["runs"]
        total = len(all_runs)
        json_ok = sum(1 for r in all_runs if r["json_valid"])
        tool_ok = sum(1 for r in all_runs if r["correct_tool"])
        args_ok = sum(1 for r in all_runs if r["correct_tool"] and r["valid_args"])
        pre_exec = sum(1 for r in all_runs if r["correct_tool"] and r["pre_exec_valid"])
        errors = sum(1 for r in all_runs if r["error"])
        rate = (tool_ok / total * 100) if total > 0 else 0
        avg_composite = sum(r.get("composite_score", 0) for r in all_runs) / total if total > 0 else 0
        times = [r["elapsed"] for r in all_runs if r["error"] is None]
        avg_t = sum(times) / len(times) if times else 0
        min_t = min(times) if times else 0
        max_t = max(times) if times else 0

        rate_class = "pass" if rate >= 80 else ("warn" if rate >= 50 else "fail")
        score_class = "pass" if avg_composite >= 75 else ("warn" if avg_composite >= 50 else "fail")
        provider = model_data.get("provider", "")

        summary_rows += f"""<tr>
            <td>{model_name}</td><td>{provider}</td>
            <td>{total}</td><td>{json_ok}</td><td>{tool_ok}</td><td>{args_ok}</td><td>{pre_exec}</td>
            <td>{errors}</td>
            <td class="{rate_class}">{rate:.1f}%</td>
            <td class="{score_class}">{avg_composite:.1f}</td>
            <td>{avg_t:.2f}s</td><td>{min_t:.2f}s</td><td>{max_t:.2f}s</td>
        </tr>\n"""

    # Build category breakdown
    categories = {}
    for qi, tq in enumerate(test_queries):
        cat = tq["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(qi)

    category_rows = ""
    for cat, query_indices in categories.items():
        for model_name, model_data in results.items():
            cat_runs = [r for r in model_data["runs"] if r["query_index"] in query_indices]
            cat_total = len(cat_runs)
            cat_correct = sum(1 for r in cat_runs if r["correct_tool"])
            cat_rate = (cat_correct / cat_total * 100) if cat_total > 0 else 0
            cat_composite = sum(r.get("composite_score", 0) for r in cat_runs) / cat_total if cat_total > 0 else 0
            cat_class = "pass" if cat_rate >= 80 else ("warn" if cat_rate >= 50 else "fail")
            score_class = "pass" if cat_composite >= 75 else ("warn" if cat_composite >= 50 else "fail")
            category_rows += f"""<tr>
                <td>{cat}</td><td>{model_name}</td>
                <td>{cat_correct}/{cat_total}</td>
                <td class="{cat_class}">{cat_rate:.0f}%</td>
                <td class="{score_class}">{cat_composite:.1f}</td>
            </tr>\n"""

    # Calculate per-query difficulty based on aggregate success rate
    query_difficulty = {}
    for qi, tq in enumerate(test_queries):
        all_query_runs = []
        for model_data in results.values():
            all_query_runs.extend(r for r in model_data["runs"] if r["query_index"] == qi)
        total_runs = len(all_query_runs)
        correct_runs = sum(1 for r in all_query_runs if r["correct_tool"])
        success_rate = (correct_runs / total_runs * 100) if total_runs > 0 else 0
        if success_rate >= 90:
            difficulty = "Easy"
            diff_class = "pass"
        elif success_rate >= 70:
            difficulty = "Medium"
            diff_class = "warn"
        elif success_rate >= 50:
            difficulty = "Hard"
            diff_class = "fail"
        else:
            difficulty = "Very Hard"
            diff_class = "fail"
        query_difficulty[qi] = {"label": difficulty, "class": diff_class, "rate": success_rate}

    # Build query difficulty summary table
    difficulty_rows = ""
    sorted_queries = sorted(query_difficulty.items(), key=lambda x: x[1]["rate"])
    for qi, diff in sorted_queries:
        tq = test_queries[qi]
        acceptable = tq.get('acceptable_tools')
        accept_str = f' <span class="meta">(also accepts: {", ".join(t for t in acceptable if t != tq["expected_tool"])})</span>' if acceptable and len(acceptable) > 1 else ""
        difficulty_rows += f"""<tr>
            <td>Q{qi + 1}</td><td>{tq['query']}</td><td>{tq['category']}</td>
            <td>{tq['expected_tool']}{accept_str}</td>
            <td class="{diff['class']}">{diff['rate']:.0f}%</td>
            <td class="{diff['class']}">{diff['label']}</td>
        </tr>\n"""

    # Build per-query detail sections
    query_sections = ""
    for qi, tq in enumerate(test_queries):
        diff = query_difficulty[qi]
        detail_rows = ""
        for model_name in model_names:
            runs = [r for r in results[model_name]["runs"] if r["query_index"] == qi]
            correct_count = sum(1 for r in runs if r["correct_tool"])
            total_count = len(runs)
            times = [r["elapsed"] for r in runs if r["error"] is None]
            avg_t = sum(times) / len(times) if times else 0
            avg_score = sum(r.get("composite_score", 0) for r in runs) / total_count if total_count else 0
            rate = (correct_count / total_count * 100) if total_count else 0
            rate_class = "pass" if rate >= 80 else ("warn" if rate >= 50 else "fail")

            runs_html = ""
            for ri, r in enumerate(runs):
                error = r.get("error")
                selected = r.get("selected_tool", "")
                all_tools = r.get("all_tools_called", [])
                args_str = json.dumps(r.get("selected_args", {}), indent=1)
                raw = str(r.get("raw_response", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                score = r.get("composite_score", 0)
                multi_tool_str = f' [{", ".join(all_tools)}]' if len(all_tools) > 1 else ""

                if error:
                    runs_html += f'<div class="name-entry error">Run {ri + 1}: <em>{error}</em> ({r["elapsed"]:.1f}s)</div>\n'
                elif r["correct_tool"]:
                    args_ok_str = "args OK" if r["valid_args"] else "args INVALID"
                    aq = r.get("args_quality", 0)
                    aq_str = f" aq={aq:.0%}" if aq < 1.0 else ""
                    raw_detail = f' <details class="inline-raw"><summary>raw</summary><pre>{raw[:1000]}</pre></details>' if save_raw and raw else ""
                    runs_html += f'<div class="name-entry pass">Run {ri + 1}: {selected} ({args_ok_str}{aq_str}) score={score:.0f}{multi_tool_str} ({r["elapsed"]:.1f}s){raw_detail}</div>\n'
                else:
                    raw_detail = f' <details class="inline-raw"><summary>raw</summary><pre>{raw[:1000]}</pre></details>' if save_raw and raw else ""
                    runs_html += f'<div class="name-entry fail">Run {ri + 1}: {selected or "(none)"} (expected {tq["expected_tool"]}) score={score:.0f}{multi_tool_str} ({r["elapsed"]:.1f}s){raw_detail}</div>\n'

            detail_rows += f"""<tr>
                <td><strong>{model_name}</strong><br><span class="meta">{results[model_name].get('provider', '')}</span></td>
                <td class="{rate_class}">{correct_count}/{total_count} ({rate:.0f}%)</td>
                <td>{avg_score:.1f}</td>
                <td>{avg_t:.2f}s</td>
                <td class="names-cell">{runs_html}</td>
            </tr>\n"""

        acceptable = tq.get('acceptable_tools')
        accept_note = f' <span class="meta">(also accepts: {", ".join(t for t in acceptable if t != tq["expected_tool"])})</span>' if acceptable and len(acceptable) > 1 else ""

        query_sections += f"""
        <details>
            <summary><h3>Q{qi + 1} [{tq['category']}]: "{tq['query']}" &rarr; {tq['expected_tool']}{accept_note} <span class="{diff['class']}" style="font-size:0.8em; padding:2px 6px; border-radius:3px;">{diff['label']}</span></h3></summary>
            <table>
                <thead><tr>
                    <th>Model</th><th>Correct</th><th>Avg Score</th><th>Avg Time</th>
                    <th>Results</th>
                </tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </details>
        """

    # Escape system prompt for HTML
    sys_prompt_escaped = system_prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Sanitize config for display (hide API keys)
    display_config = json.loads(json.dumps(config, default=str))
    if "defaults" in display_config:
        for provider in display_config["defaults"]:
            if "api_key" in display_config["defaults"][provider]:
                display_config["defaults"][provider]["api_key"] = "***hidden***"
    for m in display_config.get("models", []):
        if "api_key" in m:
            m["api_key"] = "***hidden***"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Instant Playlist Tool-Calling Test - {timestamp}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 2rem; background: #f8f9fa; color: #212529; }}
    h1 {{ color: #2563eb; }}
    h2 {{ margin-top: 2rem; border-bottom: 2px solid #dee2e6; padding-bottom: 0.5rem; }}
    h3 {{ margin: 0; display: inline; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff; }}
    th, td {{ border: 1px solid #dee2e6; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #e9ecef; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .pass {{ background: #d4edda; color: #155724; font-weight: bold; }}
    .fail {{ background: #f8d7da; color: #721c24; font-weight: bold; }}
    .warn {{ background: #fff3cd; color: #856404; font-weight: bold; }}
    .error {{ background: #f8d7da; color: #721c24; }}
    .names-cell {{ padding: 0.25rem 0.5rem; }}
    .name-entry {{ padding: 0.3rem 0.5rem; margin: 0.2rem 0; border-radius: 3px; font-size: 0.9rem; }}
    .name-entry.pass {{ background: #d4edda; color: #155724; font-weight: normal; }}
    .name-entry.fail {{ background: #fff3cd; color: #856404; font-weight: normal; }}
    .name-entry.error {{ background: #f8d7da; color: #721c24; font-weight: normal; }}
    .meta {{ font-size: 0.8rem; color: #6c757d; }}
    .inline-raw {{ display: inline; margin-left: 0.5rem; }}
    .inline-raw summary {{ display: inline; background: none; padding: 0; font-size: 0.75rem;
                           color: #6c757d; text-decoration: underline; cursor: pointer; }}
    .inline-raw pre {{ margin-top: 0.25rem; }}
    details {{ margin: 1rem 0; }}
    summary {{ cursor: pointer; padding: 0.5rem; background: #e9ecef; border-radius: 4px; }}
    summary:hover {{ background: #dee2e6; }}
    .config {{ background: #fff; padding: 1rem; border: 1px solid #dee2e6;
               border-radius: 4px; font-family: monospace; font-size: 0.85rem; }}
    .prompt-box {{ white-space: pre-wrap; max-width: 100%; background: #fff; padding: 1rem;
                   border: 1px solid #dee2e6; border-radius: 4px; font-size: 0.9rem;
                   font-family: monospace; }}
    pre {{ white-space: pre-wrap; word-break: break-all; max-width: 600px;
           font-size: 0.8rem; background: #f1f3f5; padding: 0.5rem; border-radius: 4px; }}
    .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #dee2e6;
               color: #6c757d; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>Instant Playlist Tool-Calling Test</h1>
<p><strong>Date:</strong> {timestamp} &nbsp;|&nbsp;
   <strong>Runs per model:</strong> {num_runs} &nbsp;|&nbsp;
   <strong>Queries:</strong> {num_queries} &nbsp;|&nbsp;
   <strong>CLAP enabled:</strong> {config['test_config'].get('clap_enabled', True)}</p>

<h2>Summary</h2>
<table>
    <thead><tr>
        <th>Model</th><th>Provider</th><th>Total</th><th>JSON OK</th><th>Tool OK</th>
        <th>Args OK</th><th>Pre-Exec OK</th><th>Errors</th>
        <th>Tool Rate</th><th>Composite</th><th>Avg Time</th><th>Min Time</th><th>Max Time</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
</table>

<h2>Query Difficulty</h2>
<p>Difficulty is auto-calculated from aggregate success rate across all models. Queries below 50% may need prompt improvement or reclassification.</p>
<table>
    <thead><tr>
        <th>Query</th><th>Text</th><th>Category</th><th>Expected Tool</th><th>Success Rate</th><th>Difficulty</th>
    </tr></thead>
    <tbody>{difficulty_rows}</tbody>
</table>

<h2>Category Breakdown</h2>
<table>
    <thead><tr>
        <th>Category</th><th>Model</th><th>Correct</th><th>Rate</th><th>Avg Score</th>
    </tr></thead>
    <tbody>{category_rows}</tbody>
</table>

<h2>System Prompt Used</h2>
<details>
    <summary>Show system prompt</summary>
    <pre class="prompt-box">{sys_prompt_escaped}</pre>
</details>

<h2>Per-Query Details</h2>
{query_sections}

<h2>Test Configuration</h2>
<div class="config"><pre>{json.dumps(display_config, indent=2, default=str)}</pre></div>

<div class="footer">
    Generated by AudioMuse-AI Testing Suite
</div>
</body>
</html>"""
    return html


def generate_json_report(results: dict, test_queries: list[dict],
                         timestamp: str, config: dict) -> dict:
    """Generate the full JSON report."""
    # Group queries by category for per-category stats
    categories = {}
    for qi, tq in enumerate(test_queries):
        cat = tq["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(qi)

    # Calculate per-query difficulty
    query_stats = []
    for qi, tq in enumerate(test_queries):
        all_query_runs = []
        for model_data in results.values():
            all_query_runs.extend(r for r in model_data["runs"] if r["query_index"] == qi)
        total_runs = len(all_query_runs)
        correct_runs = sum(1 for r in all_query_runs if r["correct_tool"])
        success_rate = round(correct_runs / total_runs * 100, 1) if total_runs > 0 else 0
        if success_rate >= 90:
            difficulty = "easy"
        elif success_rate >= 70:
            difficulty = "medium"
        elif success_rate >= 50:
            difficulty = "hard"
        else:
            difficulty = "very_hard"
        query_stats.append({
            "index": qi, "query": tq["query"], "expected_tool": tq["expected_tool"],
            "acceptable_tools": tq.get("acceptable_tools", [tq["expected_tool"]]),
            "category": tq["category"],
            "success_rate": success_rate,
            "difficulty": difficulty,
        })

    report = {
        "timestamp": timestamp,
        "test_type": "instant_playlist_tool_calling",
        "config": {
            "clap_enabled": config["test_config"].get("clap_enabled", True),
            "num_runs_per_model": config["test_config"]["num_runs_per_model"],
            "timeout": config["test_config"].get("timeout_per_request", 120),
            "num_queries": len(test_queries),
        },
        "queries": query_stats,
        "models": {},
    }

    for model_name, model_data in results.items():
        all_runs = model_data["runs"]
        total = len(all_runs)
        json_ok = sum(1 for r in all_runs if r["json_valid"])
        tool_ok = sum(1 for r in all_runs if r["correct_tool"])
        args_ok = sum(1 for r in all_runs if r["correct_tool"] and r["valid_args"])
        pre_exec = sum(1 for r in all_runs if r["correct_tool"] and r["pre_exec_valid"])
        errors = sum(1 for r in all_runs if r["error"])
        times = [r["elapsed"] for r in all_runs if r["error"] is None]

        avg_composite = sum(r.get("composite_score", 0) for r in all_runs) / total if total > 0 else 0

        # Per-category breakdown
        per_category = {}
        for cat, query_indices in categories.items():
            cat_runs = [r for r in all_runs if r["query_index"] in query_indices]
            cat_total = len(cat_runs)
            cat_correct = sum(1 for r in cat_runs if r["correct_tool"])
            cat_composite = sum(r.get("composite_score", 0) for r in cat_runs) / cat_total if cat_total > 0 else 0
            per_category[cat] = {
                "total": cat_total,
                "correct": cat_correct,
                "rate": round(cat_correct / cat_total * 100, 1) if cat_total > 0 else 0,
                "avg_composite": round(cat_composite, 1),
            }

        report["models"][model_name] = {
            "provider": model_data.get("provider", ""),
            "model_id": model_data.get("model_id", ""),
            "url": model_data.get("url", ""),
            "summary": {
                "total_tests": total,
                "json_valid": json_ok,
                "tool_correct": tool_ok,
                "args_valid": args_ok,
                "pre_exec_valid": pre_exec,
                "errors": errors,
                "tool_rate": round(tool_ok / total * 100, 1) if total > 0 else 0,
                "avg_composite": round(avg_composite, 1),
                "avg_time": round(sum(times) / len(times), 3) if times else 0,
                "min_time": round(min(times), 3) if times else 0,
                "max_time": round(max(times), 3) if times else 0,
            },
            "per_category": per_category,
            "runs": [
                {
                    "query_index": r["query_index"],
                    "query": r["query"],
                    "expected_tool": r["expected_tool"],
                    "category": r["category"],
                    "run_index": r["run_index"],
                    "json_valid": r["json_valid"],
                    "correct_tool": r["correct_tool"],
                    "valid_args": r["valid_args"],
                    "pre_exec_valid": r["pre_exec_valid"],
                    "args_quality": r.get("args_quality", 0),
                    "composite_score": r.get("composite_score", 0),
                    "selected_tool": r["selected_tool"],
                    "selected_args": r["selected_args"],
                    "all_tools_called": r.get("all_tools_called", []),
                    "elapsed": round(r["elapsed"], 3),
                    "error": r.get("error"),
                }
                for r in all_runs
            ],
        }

    return report


# ---------------------------------------------------------------------------
# Save reports
# ---------------------------------------------------------------------------
def save_reports(results: dict, test_queries: list[dict], config: dict,
                 num_runs: int, output_dir: str, save_raw: bool,
                 system_prompt: str):
    """Save TXT, HTML, and JSON reports to disk."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Console / TXT summary
    summary = generate_summary_table(results, timestamp)
    query_detail = generate_query_detail_table(results, test_queries, num_runs)
    full_txt = summary + "\n" + query_detail + "\n"

    print("\n" + full_txt)

    txt_path = os.path.join(output_dir, f"instant_playlist_{file_ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_txt)
    print(f"TXT report saved: {txt_path}")

    # HTML report
    html = generate_html_report(results, test_queries, num_runs, timestamp,
                                config, save_raw, system_prompt)
    html_path = os.path.join(output_dir, f"instant_playlist_{file_ts}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {html_path}")

    # JSON report
    json_data = generate_json_report(results, test_queries, timestamp, config)
    json_path = os.path.join(output_dir, f"instant_playlist_{file_ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"JSON report saved: {json_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AudioMuse-AI - Instant Playlist Tool-Calling Performance Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _default_cfg = "testing_suite/instant_playlist_test_config.yaml"
    if not os.path.exists(_default_cfg):
        _default_cfg = "testing_suite/instant_playlist_test_config.example.yaml"
    parser.add_argument("--config", "-c", type=str,
                        default=_default_cfg,
                        help="Path to YAML config file (default: instant_playlist_test_config.yaml)")
    parser.add_argument("--runs", "-n", type=int, default=None,
                        help="Override num_runs_per_model from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and show config, but don't call any APIs")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print(f"Usage: python testing_suite/test_instant_playlist.py --config path/to/config.yaml")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Merge provider defaults into each model entry
    apply_defaults(config)

    # Apply CLI overrides
    if args.runs is not None:
        config["test_config"]["num_runs_per_model"] = args.runs

    num_runs = config["test_config"]["num_runs_per_model"]
    clap_enabled = config["test_config"].get("clap_enabled", True)
    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("directory", "testing_suite/reports/instant_playlist")
    save_raw = output_cfg.get("save_raw_responses", True)

    # Build tools and system prompt for display
    tools = get_tool_definitions(clap_enabled)
    library_context = config.get("library_context")
    system_prompt = build_system_prompt(tools, library_context)

    # Filter queries for count
    all_queries = config.get("test_queries", [])
    active_queries = [q for q in all_queries
                      if not (q.get("skip_if_clap_disabled") and not clap_enabled)]

    print("=" * 60)
    print(" AudioMuse-AI - Instant Playlist Tool-Calling Test")
    print("=" * 60)

    enabled = [m for m in config["models"] if m.get("enabled", False)]
    print(f" Models:      {len(enabled)} enabled")
    print(f" Queries:     {len(active_queries)}")
    print(f" Runs/model:  {num_runs}")
    print(f" CLAP:        {'enabled' if clap_enabled else 'disabled'}")
    print(f" Tools:       {', '.join(t['name'] for t in tools)}")
    print("=" * 60 + "\n")

    # Run tests
    results, test_queries = run_tests(config, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete. No API calls were made.")
        return

    if not results:
        print("No results to report.")
        return

    # Generate and save reports
    save_reports(results, test_queries, config, num_runs, output_dir, save_raw,
                 system_prompt)


if __name__ == "__main__":
    main()
