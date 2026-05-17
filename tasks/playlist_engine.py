"""
Core instant playlist building logic — extracted from app_chat.py.
Routes user queries via JointBERT and executes tools directly, with AI brainstorm as fallback.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from config import (
    FILTER_OVERFETCH_FACTOR,
    JOINBERT_CONFIDENCE_THRESHOLD,
    MAX_SONGS_PER_ARTIST_PLAYLIST,
    PLAYLIST_ENERGY_ARC,
)
from tasks.joinbert_client import route_query, sanitize_user_input
from tasks.mcp_tools import execute_mcp_tool
from tasks.playlist_ordering import order_playlist

logger = logging.getLogger(__name__)


def build_instant_playlist(user_input: str, ai_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an instant playlist from user text via JointBERT routing.

    Flow:
    1. Sanitize user input
    2. Route via JointBERT (get tool calls + confidence)
    3. If confidence < threshold OR only ai_brainstorm → fallback to AI brainstorm
    4. Otherwise → execute all tools directly, aggregate songs, apply diversity, order

    Returns:
        {
            "songs": [{"item_id": ..., "title": ..., "artist": ...}, ...],
            "message": str (detailed log),
            "ai_used": bool,
            "original_request": str,
            "executed_query": str,
        }
    """
    log_messages: List[str] = []

    clean_text = sanitize_user_input(user_input)
    if not clean_text:
        return {
            "songs": [],
            "message": "❌ Input is empty or contains no valid characters.",
            "ai_used": False,
            "original_request": user_input,
            "executed_query": "invalid_input",
        }

    # --- Route via JointBERT ---
    tool_calls, confidence = route_query(clean_text)
    log_messages.append(f"🎵 Query: '{clean_text}'")
    log_messages.append(f"📍 JointBERT confidence: {confidence:.2%}")

    # --- Decide: JointBERT tools or AI brainstorm fallback ---
    only_brainstorm = len(tool_calls) == 1 and tool_calls[0][0] == "ai_brainstorm"
    if confidence < JOINBERT_CONFIDENCE_THRESHOLD or only_brainstorm:
        log_messages.append(f"⚠️  Low confidence or brainstorm-only → fallback to AI brainstorm")
        return _build_ai_brainstorm_playlist(clean_text, ai_config, log_messages)

    # --- Execute JointBERT tools directly ---
    return _build_tool_playlist(tool_calls, ai_config, log_messages)


def _build_tool_playlist(
    tool_calls: List[Tuple[str, Dict]], ai_config: Dict, log_messages: List[str]
) -> Dict[str, Any]:
    """
    Execute tool calls directly and aggregate results.
    When search_database appears alongside other tools, it acts as a filter on their results.
    """
    all_songs: List[Dict] = []
    song_ids_seen: Set[str] = set()
    song_keys_seen: Set[Tuple[str, str]] = set()
    song_sources: Dict[str, int] = {}  # item_id → tool call index
    tools_used_history: List[Dict] = []
    tool_execution_summary: List[str] = []

    PRIMARY_TOOLS = {"lyrics_search", "song_similarity", "artist_similarity", "text_search"}
    has_db_filter = any(name == "search_database" for name, _ in tool_calls)
    has_primary = any(name in PRIMARY_TOOLS for name, _ in tool_calls)
    filter_mode = has_db_filter and has_primary

    primary_candidate_ids = []
    tool_call_counter = 0

    for tool_name, tool_args in tool_calls:
        tool_args = dict(tool_args)

        if filter_mode and tool_name in PRIMARY_TOOLS:
            tool_args["get_songs"] = 200 * FILTER_OVERFETCH_FACTOR

            log_messages.append(f"\n🔧 Tool {tool_call_counter + 1}: {tool_name} (over-fetch for filtering)")
            try:
                log_messages.append(f"   Arguments: {json.dumps(tool_args, indent=6)}")
            except (TypeError, ValueError):
                log_messages.append(f"   Arguments: {str(tool_args)}")

            tool_result = execute_mcp_tool(tool_name, tool_args, ai_config)

            if "error" in tool_result:
                log_messages.append(f"   ❌ Error: {tool_result['error']}")
                tools_used_history.append({
                    "name": tool_name,
                    "args": tool_args,
                    "songs": 0,
                    "error": True,
                    "call_index": tool_call_counter,
                })
                tool_call_counter += 1
                continue

            songs = tool_result.get("songs", [])
            log_messages.append(f"   ✅ Retrieved {len(songs)} songs")

            if tool_result.get("message"):
                for line in tool_result["message"].split("\n"):
                    if line.strip():
                        log_messages.append(f"   {line}")

            primary_candidate_ids.extend(s["item_id"] for s in songs)
            tools_used_history.append({
                "name": tool_name,
                "args": tool_args,
                "songs": len(songs),
                "call_index": tool_call_counter,
            })
            args_summary = _summarize_tool_args(tool_name, tool_args)
            tool_summary = f"{tool_name}({args_summary}, +{len(songs)})" if args_summary else f"{tool_name}(+{len(songs)})"
            tool_execution_summary.append(tool_summary)
            tool_call_counter += 1
            continue

        if filter_mode and tool_name == "search_database":
            tool_args["item_ids"] = primary_candidate_ids
            tool_args["get_songs"] = 200

            log_messages.append(f"\n🔧 Tool {tool_call_counter + 1}: {tool_name} (filter mode)")
            log_messages.append(f"   🔍 Filter mode: filtering {len(primary_candidate_ids)} candidates")
            try:
                log_messages.append(f"   Arguments: {json.dumps({k: v for k, v in tool_args.items() if k != 'item_ids'}, indent=6)} (+ {len(primary_candidate_ids)} item_ids)")
            except (TypeError, ValueError):
                log_messages.append(f"   Arguments: {str(tool_args)}")

            tool_result = execute_mcp_tool(tool_name, tool_args, ai_config)

            if "error" in tool_result:
                log_messages.append(f"   ❌ Error: {tool_result['error']}")
                tools_used_history.append({
                    "name": tool_name,
                    "args": tool_args,
                    "songs": 0,
                    "error": True,
                    "call_index": tool_call_counter,
                })
                tool_call_counter += 1
                continue

            songs = tool_result.get("songs", [])
            log_messages.append(f"   ✅ Retrieved {len(songs)} songs")
            log_messages.append(f"   ✅ Filter kept {len(songs)} / {len(primary_candidate_ids)} songs")

            if tool_result.get("message"):
                for line in tool_result["message"].split("\n"):
                    if line.strip():
                        log_messages.append(f"   {line}")

            new_songs = 0
            for song in songs:
                song_key = (song.get("title", "").strip().lower(), song.get("artist", "").strip().lower())
                if song["item_id"] not in song_ids_seen and song_key not in song_keys_seen:
                    all_songs.append(song)
                    song_ids_seen.add(song["item_id"])
                    song_keys_seen.add(song_key)
                    song_sources[song["item_id"]] = tool_call_counter
                    new_songs += 1

            log_messages.append(f"   📊 Added {new_songs} new unique songs")
            tools_used_history.append({
                "name": tool_name,
                "args": tool_args,
                "songs": new_songs,
                "call_index": tool_call_counter,
            })

            args_summary = _summarize_tool_args(tool_name, tool_args)
            tool_summary = f"{tool_name}({args_summary}, +{new_songs})" if args_summary else f"{tool_name}(+{new_songs})"
            tool_execution_summary.append(tool_summary)
            tool_call_counter += 1
            continue

        tool_args["get_songs"] = 200

        log_messages.append(f"\n🔧 Tool {tool_call_counter + 1}: {tool_name}")
        try:
            log_messages.append(f"   Arguments: {json.dumps(tool_args, indent=6)}")
        except (TypeError, ValueError):
            log_messages.append(f"   Arguments: {str(tool_args)}")

        tool_result = execute_mcp_tool(tool_name, tool_args, ai_config)

        if "error" in tool_result:
            log_messages.append(f"   ❌ Error: {tool_result['error']}")
            tools_used_history.append({
                "name": tool_name,
                "args": tool_args,
                "songs": 0,
                "error": True,
                "call_index": tool_call_counter,
            })
            tool_call_counter += 1
            continue

        songs = tool_result.get("songs", [])
        log_messages.append(f"   ✅ Retrieved {len(songs)} songs")

        if tool_result.get("message"):
            for line in tool_result["message"].split("\n"):
                if line.strip():
                    log_messages.append(f"   {line}")

        new_songs = 0
        for song in songs:
            song_key = (song.get("title", "").strip().lower(), song.get("artist", "").strip().lower())
            if song["item_id"] not in song_ids_seen and song_key not in song_keys_seen:
                all_songs.append(song)
                song_ids_seen.add(song["item_id"])
                song_keys_seen.add(song_key)
                song_sources[song["item_id"]] = tool_call_counter
                new_songs += 1

        log_messages.append(f"   📊 Added {new_songs} new unique songs")
        tools_used_history.append({
            "name": tool_name,
            "args": tool_args,
            "songs": new_songs,
            "call_index": tool_call_counter,
        })

        args_summary = _summarize_tool_args(tool_name, tool_args)
        tool_summary = f"{tool_name}({args_summary}, +{new_songs})" if args_summary else f"{tool_name}(+{new_songs})"
        tool_execution_summary.append(tool_summary)
        tool_call_counter += 1

    if not all_songs:
        log_messages.append("\n❌ No songs found. Falling back to AI brainstorm.")
        return _build_ai_brainstorm_playlist(user_input="", ai_config=ai_config, log_messages=log_messages)

    # --- Post-processing ---
    target_song_count = 100

    # Phase 1: Artist Diversity Cap
    max_per_artist = MAX_SONGS_PER_ARTIST_PLAYLIST
    diversified_pool, diversity_overflow = _apply_artist_diversity(all_songs, max_per_artist)
    diversity_removed = len(all_songs) - len(diversified_pool)
    if diversity_removed > 0:
        log_messages.append(f"\n🎨 Artist diversity: removed {diversity_removed} songs (max {max_per_artist}/artist)")

    # Phase 2: Proportional sampling
    if len(diversified_pool) <= target_song_count:
        final_songs = list(diversified_pool)
        # Progressive cap relaxation if needed
        if len(final_songs) < target_song_count and diversity_overflow:
            final_songs, relaxed_cap = _backfill_with_progressive_relaxation(
                final_songs, diversity_overflow, max_per_artist, target_song_count
            )
            if relaxed_cap > max_per_artist:
                log_messages.append(f"   Progressive relaxation: {max_per_artist} → {relaxed_cap}/artist")
    else:
        final_songs = _proportional_sample(diversified_pool, song_sources, target_song_count)

    log_messages.append(f"\n📊 Pool: {len(all_songs)} collected → {len(diversified_pool)} after diversity → {len(final_songs)} final")

    # Phase 3: Ordering
    try:
        song_id_list = [s["item_id"] for s in final_songs]
        ordered_ids = order_playlist(song_id_list, energy_arc=PLAYLIST_ENERGY_ARC)
        id_to_song = {s["item_id"]: s for s in final_songs}
        final_songs = [id_to_song[sid] for sid in ordered_ids if sid in id_to_song]
        log_messages.append(f"\n🎵 Playlist ordered for smooth transitions")
    except Exception as e:
        logger.warning(f"Playlist ordering failed: {e}")
        log_messages.append(f"\n⚠️  Playlist ordering skipped: {str(e)[:100]}")

    executed_query = f"JointBERT ({len(tools_used_history)} tools): {' → '.join(tool_execution_summary)}"
    log_messages.append(f"\n✅ SUCCESS! Generated {len(final_songs)}-song playlist")

    return {
        "songs": final_songs,
        "message": "\n".join(log_messages),
        "ai_used": False,
        "executed_query": executed_query,
    }


def _build_ai_brainstorm_playlist(
    user_input: str, ai_config: Dict, log_messages: List[str]
) -> Dict[str, Any]:
    """
    Fallback: use AI brainstorming directly.
    """
    if not log_messages:
        log_messages.append(f"🎵 Query: '{user_input}'")

    log_messages.append(f"\n🧠 Using AI brainstorm mode (confidence too low or user intent unclear)")

    result = execute_mcp_tool("ai_brainstorm", {"user_request": user_input, "get_songs": 100}, ai_config)

    if "error" in result:
        log_messages.append(f"\n❌ Error: {result['error']}")
        return {
            "songs": [],
            "message": "\n".join(log_messages),
            "ai_used": True,
            "executed_query": "ai_brainstorm_failed",
        }

    songs = result.get("songs", [])
    log_messages.append(f"✅ AI generated {len(songs)} songs")

    # Order
    try:
        song_id_list = [s["item_id"] for s in songs]
        ordered_ids = order_playlist(song_id_list, energy_arc=PLAYLIST_ENERGY_ARC)
        id_to_song = {s["item_id"]: s for s in songs}
        songs = [id_to_song[sid] for sid in ordered_ids if sid in id_to_song]
    except Exception as e:
        logger.warning(f"Playlist ordering failed: {e}")

    return {
        "songs": songs,
        "message": "\n".join(log_messages),
        "ai_used": True,
        "executed_query": "ai_brainstorm",
    }


def _apply_artist_diversity(
    songs: List[Dict], max_per_artist: int
) -> Tuple[List[Dict], List[Dict]]:
    """Separate songs into diverse pool and overflow."""
    artist_counts = {}
    diverse_pool = []
    overflow = []

    for song in songs:
        artist = song.get("artist", "Unknown")
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        if artist_counts[artist] <= max_per_artist:
            diverse_pool.append(song)
        else:
            overflow.append(song)

    return diverse_pool, overflow


def _backfill_with_progressive_relaxation(
    current_list: List[Dict], overflow: List[Dict], initial_cap: int, target: int
) -> Tuple[List[Dict], int]:
    """Progressively raise per-artist cap to backfill from overflow."""
    result = list(current_list)
    current_cap = initial_cap

    while len(result) < target and overflow:
        current_cap += 1
        artist_counts = {}
        for s in result:
            a = s.get("artist", "Unknown")
            artist_counts[a] = artist_counts.get(a, 0) + 1

        still_overflow = []
        for song in overflow:
            if len(result) >= target:
                still_overflow.append(song)
                continue
            artist = song.get("artist", "Unknown")
            if artist_counts.get(artist, 0) < current_cap:
                result.append(song)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
            else:
                still_overflow.append(song)

        overflow = still_overflow
        if len(still_overflow) == len(overflow):
            break  # No progress

    return result, current_cap


def _proportional_sample(
    diverse_pool: List[Dict], song_sources: Dict[str, int], target: int
) -> List[Dict]:
    """Sample proportionally by tool call source."""
    songs_by_source = {}
    for song in diverse_pool:
        source = song_sources.get(song["item_id"], -1)
        if source not in songs_by_source:
            songs_by_source[source] = []
        songs_by_source[source].append(song)

    result = []
    total = len(diverse_pool)
    for source, songs in sorted(songs_by_source.items()):
        proportion = len(songs) / total
        allocated = max(1, int(proportion * target))
        result.extend(songs[:allocated])

    # Backfill if short
    if len(result) < target:
        selected_ids = {s["item_id"] for s in result}
        remaining = [s for s in diverse_pool if s["item_id"] not in selected_ids]
        needed = target - len(result)
        result.extend(remaining[:needed])

    return result[:target]


def _summarize_tool_args(tool_name: str, args: Dict[str, Any]) -> str:
    """Create a readable summary of tool arguments."""
    summary = []

    if tool_name == "search_database":
        for key in ["artist", "genres", "moods", "album", "key", "scale"]:
            if key in args and args[key]:
                summary.append(f"{key}='{args[key]}'")
        for key in ["tempo_min", "energy_min", "year_min"]:
            if key in args or key.replace("_min", "_max") in args:
                min_v = args.get(key, "")
                max_v = args.get(key.replace("_min", "_max"), "")
                if min_v or max_v:
                    summary.append(f"{key.replace('_min', '')}={min_v}..{max_v}")
        if "min_rating" in args:
            summary.append(f"rating={args['min_rating']}")

    elif tool_name in ["artist_similarity", "artist_hits"]:
        artist = args.get("artist") or args.get("artist_name")
        if artist:
            summary.append(f"artist='{artist}'")

    elif tool_name == "song_similarity":
        if args.get("song_title"):
            summary.append(f"song='{args['song_title']}'")
        if args.get("song_artist"):
            summary.append(f"artist='{args['song_artist']}'")

    elif tool_name == "ai_brainstorm":
        if "user_request" in args:
            req = args["user_request"][:30]
            summary.append(f"req='{req}...'")

    elif tool_name == "text_search":
        if "description" in args:
            desc = args["description"][:30]
            summary.append(f"desc='{desc}...'")

    elif tool_name == "lyrics_search":
        if "description" in args:
            desc = args["description"][:30]
            summary.append(f"desc='{desc}...'")

    return ", ".join(summary)
