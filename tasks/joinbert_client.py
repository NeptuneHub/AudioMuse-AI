"""
Production runtime wrapper for JointBERT inference with mood normalization.
Sanitizes user input, provides singleton access to trained model, and normalizes
moods against MOOD_LABELS and OTHER_FEATURE_LABELS from config.
"""
import re
import sys
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher
import shutil

HERE = Path(__file__).parent
JOINBERT_DIR = (HERE.parent / "joinbert").resolve()
sys.path.insert(0, str(JOINBERT_DIR))

# Clean stale HuggingFace lock files on module import (app startup)
# This prevents "PermissionError on .locks" when containers restart with persistent cache
_hf_cache = Path("/app/.cache/huggingface")
if _hf_cache.exists():
    _locks_dir = _hf_cache / ".locks"
    if _locks_dir.exists():
        try:
            shutil.rmtree(_locks_dir)
            print("[joinbert_client] Cleaned stale HuggingFace lock files")
        except Exception as e:
            print(f"[joinbert_client] Warning: Could not clean HF locks: {e}")

try:
    from inference import Router
except ImportError:
    Router = None


_router: Optional[Router] = None


def sanitize_user_input(text: str) -> str:
    """
    Sanitize user input for JointBERT inference.
    - Strip NUL bytes and control characters
    - Keep only printable chars, newlines, tabs, spaces
    - Truncate to 512 chars
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Remove NUL bytes and control characters (except \n, \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Keep only printable + \n, \t, space
    text = re.sub(r'[^\x20-\x7e\n\t]', '', text)

    # Truncate to 512
    text = text[:512]

    # Strip leading/trailing whitespace
    return text.strip()


def get_router() -> Optional[Router]:
    """
    Lazy-load the JointBERT router singleton.
    Returns None if model files are missing or cannot be loaded.
    """
    global _router
    if _router is None:
        try:
            _router = Router()
        except Exception as e:
            print(f"[joinbert_client] Failed to load Router: {e}")
            return None
    return _router


def _normalize_mood(mood: str, label_set: list) -> tuple[str | None, float]:
    """Fuzzy match mood to closest label in label_set. Returns (best_match, score) or (None, 0) if score < 0.5."""
    mood_lower = mood.lower()

    best_match, best_score = max(
        ((label, SequenceMatcher(None, mood_lower, label.lower()).ratio()) for label in label_set),
        key=lambda x: x[1]
    )
    if best_score < 0.5:
        return None, best_score
    return best_match, best_score


def _dispatch_production(text: str, intents: list, entities: list) -> list:
    """
    Production dispatch with smart mood/genre matching.
    - Matches each entity against both MOOD_LABELS (genres+voice) and OTHER_FEATURE_LABELS (moods)
    - Uses the better match (higher score), ignoring poor matches (score < 0.5)
    """
    from config import MOOD_LABELS, OTHER_FEATURE_LABELS

    mood_set_lower = {m.lower() for m in OTHER_FEATURE_LABELS}
    genre_set_lower = {g.lower() for g in MOOD_LABELS}

    ents: dict[str, list[str]] = {}
    for e in entities:
        e_type = e["type"]
        e_value = e["value"]
        e_value_lower = e_value.lower()

        if e_type == "mood" or e_type == "genre":
            if e_value_lower in mood_set_lower:
                ents.setdefault("mood", []).append(e_value)
            elif e_value_lower in genre_set_lower:
                ents.setdefault("genre", []).append(e_value)
            else:
                mood_match, mood_score = _normalize_mood(e_value, OTHER_FEATURE_LABELS)
                genre_match, genre_score = _normalize_mood(e_value, MOOD_LABELS)

                if mood_score >= genre_score and mood_match:
                    ents.setdefault("mood", []).append(mood_match)
                elif genre_match:
                    ents.setdefault("genre", []).append(genre_match)
        else:
            ents.setdefault(e_type, []).append(e_value)

    intent_names = {name for name, _ in intents}
    calls: list[tuple[str, dict]] = []

    n_song_artist_pairs = 0
    if "song_similarity" in intent_names and "song" in ents and "artist" in ents:
        n_song_artist_pairs = min(len(ents["song"]), len(ents["artist"]))
        for i in range(n_song_artist_pairs):
            calls.append(("song_similarity", {
                "song_title": ents["song"][i],
                "song_artist": ents["artist"][i],
            }))

    if "artist_similarity" in intent_names and "artist" in ents:
        for a in ents["artist"][n_song_artist_pairs:]:
            calls.append(("artist_similarity", {"artist": a}))

    if "text_search" in intent_names:
        desc = ents.get("description", [None])[0]
        if desc is None:
            desc = text
        args = {"description": desc}
        if "tempo" in ents:
            t = _tempo_filter_token(ents["tempo"][0])
            if t: args["tempo_filter"] = t
        if "energy" in ents:
            e = _energy_filter_token(ents["energy"][0])
            if e: args["energy_filter"] = e
        calls.append(("text_search", args))

    if "song_alchemy" in intent_names:
        add_items, sub_items = [], []
        for a in ents.get("add_artist", []):
            add_items.append({"type": "artist", "id": a})
        if not add_items and len(ents.get("artist", [])) >= 2:
            for a in ents["artist"]:
                add_items.append({"type": "artist", "id": a})
        for a in ents.get("subtract_artist", []):
            sub_items.append({"type": "artist", "id": a})
        for g in ents.get("subtract_genre", []):
            sub_items.append({"type": "genre", "id": g})
        if add_items:
            args = {"add_items": add_items}
            if sub_items: args["subtract_items"] = sub_items
            calls.append(("song_alchemy", args))

    if "ai_brainstorm" in intent_names:
        calls.append(("ai_brainstorm", {"user_request": text}))

    if "lyrics_search" in intent_names:
        topics = ents.get("lyrics_query", [])
        query = " and ".join(topics) if topics else text
        calls.append(("lyrics_search", {"query": query}))

    if "search_database" in intent_names:
        args: dict = {}
        if "genre" in ents: args["genres"] = [g.lower() for g in ents["genre"]]
        if "mood" in ents: args["moods"] = [m.lower() for m in ents["mood"]]
        if "key" in ents: args["key"] = ents["key"][0]
        if "scale" in ents: args["scale"] = ents["scale"][0].lower()
        if "album" in ents: args["album"] = ents["album"][0]
        if "year" in ents:
            years = []
            for y in ents["year"]:
                m = re.search(r"\d{4}", y)
                if m: years.append(int(m.group()))
            if years:
                args["year_min"] = min(years)
                args["year_max"] = max(years)
        if "time_range" in ents:
            args.update(_normalize_time_range(ents["time_range"][0]))
        if "tempo" in ents: args.update(_normalize_tempo(ents["tempo"][0]))
        if "energy" in ents: args.update(_normalize_energy(ents["energy"][0]))
        if "rating" in ents: args.update(_normalize_rating(ents["rating"][0]))
        if "artist" in ents and "artist_similarity" not in intent_names:
            args["artist"] = ents["artist"][0]
        if args:
            calls.append(("search_database", args))

    return calls


def _tempo_filter_token(value: str) -> str | None:
    v = value.lower()
    bpm = re.search(r"(\d{2,3})", v)
    if "bpm" in v and bpm:
        n = int(bpm.group(1))
        if n < 90:  return "slow"
        if n < 130: return "medium"
        return "fast"
    if "slow" in v: return "slow"
    if "fast" in v: return "fast"
    if "medium" in v or "mid-tempo" in v: return "medium"
    return None


def _energy_filter_token(value: str) -> str | None:
    v = value.lower()
    if "low" in v or "calm" in v: return "low"
    if "high" in v: return "high"
    if "medium" in v: return "medium"
    return None


def _normalize_time_range(value: str) -> dict:
    from datetime import datetime
    v = value.lower()
    now = datetime.now().year
    m = re.search(r"last\s+(\d+)\s+year", v)
    if m:
        return {"year_min": now - int(m.group(1)), "year_max": now}
    decade_map = {
        "90s": (1990, 1999), "80s": (1980, 1989), "70s": (1970, 1979),
        "60s": (1960, 1969), "50s": (1950, 1959), "2000s": (2000, 2009),
        "2010s": (2010, 2019), "2020s": (2020, 2029),
    }
    for tag, (lo, hi) in decade_map.items():
        if tag in v:
            return {"year_min": lo, "year_max": hi}
    if "last year" in v: return {"year_min": now - 1, "year_max": now - 1}
    if "this year" in v: return {"year_min": now, "year_max": now}
    if "recent" in v:    return {"year_min": now - 3, "year_max": now}
    if "mozart era" in v: return {"year_min": 1750, "year_max": 1820}
    return {}


def _normalize_rating(value: str) -> dict:
    digit = re.search(r"\d", value)
    if digit:
        return {"min_rating": int(digit.group())}
    if "favorite" in value.lower() or "favourite" in value.lower():
        return {"min_rating": 4}
    return {}


def _normalize_tempo(value: str) -> dict:
    v = value.lower()
    bpm = re.search(r"(\d{2,3})\s*bpm", v)
    if bpm:
        n = int(bpm.group(1))
        return {"tempo_min": max(40, n - 10), "tempo_max": min(220, n + 10)}
    if "slow" in v:                            return {"tempo_min": 40,  "tempo_max": 90}
    if "medium" in v or "mid-tempo" in v:      return {"tempo_min": 90,  "tempo_max": 130}
    if "fast" in v:                            return {"tempo_min": 130, "tempo_max": 200}
    return {}


def _normalize_energy(value: str) -> dict:
    v = value.lower()
    if "low" in v or "calm" in v: return {"energy_min": 0.0,  "energy_max": 0.35}
    if "medium" in v:             return {"energy_min": 0.35, "energy_max": 0.7}
    if "high" in v:               return {"energy_min": 0.7,  "energy_max": 1.0}
    return {}


def route_query(text: str) -> tuple[list, float]:
    """
    Route a user query through JointBERT with production mood normalization.

    Args:
        text: User query (will be sanitized)

    Returns:
        (tool_calls, max_confidence) where:
        - tool_calls: list of (tool_name, tool_args) tuples with normalized moods
        - max_confidence: max probability across all intents (0.0 if no intents)
    """
    router = get_router()
    if router is None:
        return [], 0.0

    clean_text = sanitize_user_input(text)
    if not clean_text:
        return [], 0.0

    try:
        intents, entities, intent_probs = router.predict(clean_text)
        max_confidence = float(max(intent_probs)) if len(intent_probs) > 0 else 0.0
        tool_calls = _dispatch_production(clean_text, intents, entities)
        return tool_calls, max_confidence
    except Exception as e:
        print(f"[joinbert_client] JointBERT prediction failed: {e}")
        return [], 0.0
