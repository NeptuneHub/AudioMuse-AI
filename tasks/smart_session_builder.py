"""Smart Listening Sessions request handling and preview skeleton."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


SMART_SESSION_MIN_LENGTH = 5
SMART_SESSION_DEFAULT_LENGTH = 25
SMART_SESSION_MAX_LENGTH = 100
SMART_SESSION_DEFAULT_MAX_PER_ARTIST = 2
SMART_SESSION_MAX_ANCHORS = 5


class SmartSessionValidationError(ValueError):
    """Raised when a Smart Listening Sessions request is invalid."""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _unique_clean_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise SmartSessionValidationError("List value must be an array or string.")

    result = []
    seen = set()
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def clamp_session_length(value: Any) -> int:
    if value in (None, ""):
        return SMART_SESSION_DEFAULT_LENGTH
    try:
        length = int(value)
    except (TypeError, ValueError):
        raise SmartSessionValidationError("Session length must be a number.")
    return max(SMART_SESSION_MIN_LENGTH, min(SMART_SESSION_MAX_LENGTH, length))


def clamp_max_per_artist(value: Any) -> int:
    if value in (None, ""):
        return SMART_SESSION_DEFAULT_MAX_PER_ARTIST
    try:
        max_per_artist = int(value)
    except (TypeError, ValueError):
        raise SmartSessionValidationError("Max per artist must be a number.")
    return max(1, min(10, max_per_artist))


def normalize_anchor(anchor: Any) -> Dict[str, Any]:
    if not isinstance(anchor, dict):
        raise SmartSessionValidationError("Each anchor must be an object.")

    anchor_type = _clean_text(anchor.get("type") or "song").casefold()
    if anchor_type != "song":
        raise SmartSessionValidationError("Only song anchors are supported in the first Smart Sessions version.")

    item_id = _clean_text(anchor.get("item_id"))
    if not item_id:
        raise SmartSessionValidationError("Song anchors require an item_id.")

    try:
        weight = float(anchor.get("weight", 1.0))
    except (TypeError, ValueError):
        raise SmartSessionValidationError("Anchor weight must be a number.")
    weight = max(0.0, min(1.0, weight))

    return {
        "type": "song",
        "item_id": item_id,
        "weight": weight,
    }


def normalize_avoid_rules(data: Any) -> Dict[str, List[str]]:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SmartSessionValidationError("Avoid rules must be an object.")
    return {
        "artists": _unique_clean_list(data.get("artists")),
        "terms": _unique_clean_list(data.get("terms")),
    }


def validate_preview_request(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise SmartSessionValidationError("Request body must be a JSON object.")

    prompt = _clean_text(data.get("prompt"))
    anchors_raw = data.get("anchors") or []
    if not isinstance(anchors_raw, list):
        raise SmartSessionValidationError("Anchors must be an array.")
    if len(anchors_raw) > SMART_SESSION_MAX_ANCHORS:
        raise SmartSessionValidationError(f"A session can use at most {SMART_SESSION_MAX_ANCHORS} anchors.")

    anchors = [normalize_anchor(anchor) for anchor in anchors_raw]
    if not prompt and not anchors:
        raise SmartSessionValidationError("Provide a prompt or at least one song anchor.")

    curve = _clean_text(data.get("curve") or "steady").casefold()
    valid_curves = {"steady", "calm_to_intense", "intense_to_calm", "near_anchor_then_explore"}
    if curve not in valid_curves:
        raise SmartSessionValidationError("Unsupported session curve.")

    return {
        "prompt": prompt,
        "length": clamp_session_length(data.get("length")),
        "curve": curve,
        "anchors": anchors,
        "avoid": normalize_avoid_rules(data.get("avoid")),
        "max_per_artist": clamp_max_per_artist(data.get("max_per_artist")),
        "include_explanations": bool(data.get("include_explanations", True)),
    }


def validate_export_request(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise SmartSessionValidationError("Request body must be a JSON object.")

    playlist_name = _clean_text(data.get("playlist_name"))
    if not playlist_name:
        raise SmartSessionValidationError("Missing playlist_name.")

    track_ids_raw = data.get("track_ids") or []
    if not isinstance(track_ids_raw, list):
        raise SmartSessionValidationError("track_ids must be an array.")

    track_ids = []
    seen = set()
    for value in track_ids_raw:
        item_id = _clean_text(value)
        if item_id and item_id not in seen:
            seen.add(item_id)
            track_ids.append(item_id)

    if not track_ids:
        raise SmartSessionValidationError("At least one track ID is required.")

    return {
        "playlist_name": playlist_name,
        "track_ids": track_ids,
    }


def get_smart_session_capabilities() -> Dict[str, Any]:
    from config import CLAP_ENABLED, LYRICS_ENABLED

    clap_cache_loaded = False
    clap_song_count = 0
    try:
        from tasks.clap_text_search import get_cache_stats
        clap_stats = get_cache_stats()
        clap_cache_loaded = bool(clap_stats.get("loaded"))
        clap_song_count = int(clap_stats.get("song_count") or 0)
    except Exception:
        clap_cache_loaded = False

    sem_grove_available = False
    sem_grove_song_count = 0
    try:
        from tasks.sem_grove_manager import get_sem_grove_stats
        sem_grove_stats = get_sem_grove_stats()
        sem_grove_available = bool(sem_grove_stats.get("loaded"))
        sem_grove_song_count = int(sem_grove_stats.get("song_count") or 0)
    except Exception:
        sem_grove_available = False

    return {
        "clap_enabled": bool(CLAP_ENABLED),
        "clap_cache_loaded": clap_cache_loaded,
        "clap_song_count": clap_song_count,
        "sem_grove_available": sem_grove_available,
        "sem_grove_song_count": sem_grove_song_count,
        "lyrics_enabled": bool(LYRICS_ENABLED),
        "min_length": SMART_SESSION_MIN_LENGTH,
        "max_length": SMART_SESSION_MAX_LENGTH,
        "default_length": SMART_SESSION_DEFAULT_LENGTH,
        "default_max_per_artist": SMART_SESSION_DEFAULT_MAX_PER_ARTIST,
        "supported_curves": ["steady", "calm_to_intense", "intense_to_calm", "near_anchor_then_explore"],
    }


def build_smart_session_preview(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    request_data = validate_preview_request(data)
    prompt_name = request_data["prompt"][:40].strip() or "Anchored Session"
    playlist_name = f"Smart Session - {prompt_name}"

    return {
        "session_id": None,
        "playlist_name": playlist_name,
        "tracks": [],
        "warnings": [
            "Smart Listening Sessions request validation is ready; candidate ranking is scheduled for Day 2 and Day 3."
        ],
        "request": request_data,
    }


def export_smart_session_playlist(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = validate_export_request(data)
    from tasks.voyager_manager import create_playlist_from_ids

    playlist_id = create_playlist_from_ids(payload["playlist_name"], payload["track_ids"])
    return {
        "message": f"Playlist '{payload['playlist_name']}' created successfully!",
        "playlist_id": playlist_id,
    }