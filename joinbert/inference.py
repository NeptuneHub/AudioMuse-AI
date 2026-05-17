"""
Load the exported ONNX JointBERT and demo it on natural-language music queries.

For each demo query we print:
    - the input text
    - intent probabilities (which tools to fire, multi-label)
    - decoded entities (BIO spans)
    - the final dispatcher output: [(tool_name, tool_args), ...]
      ready to be passed to AudioMuse-AI's existing `execute_mcp_tool()`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
LABELS_PATH = HERE / "labels.json"
# Prefer the best-val-loss checkpoint; fall back to legacy / last for backwards
# compatibility with checkouts trained before the best/last split.
_CANDIDATES = [
    HERE / "joint_bert_best.onnx",
    HERE / "joint_bert_last.onnx",
    HERE / "joint_bert.onnx",
]
ONNX_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

INTENT_THRESHOLD = 0.5


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------
class Router:
    def __init__(self):
        meta = json.loads(LABELS_PATH.read_text())
        self.intent_labels: list[str] = meta["intents"]
        self.slot_labels: list[str] = meta["slots"]
        self.max_len: int = meta["max_len"]
        self.tokenizer = AutoTokenizer.from_pretrained(meta["model_name"])
        self.session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    def predict(self, text: str):
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids = np.array([enc["input_ids"]], dtype=np.int64)
        attention_mask = np.array([enc["attention_mask"]], dtype=np.int64)
        intent_logits, slot_logits = self.session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        intent_probs = sigmoid(intent_logits[0])
        slot_ids = slot_logits[0].argmax(-1)

        intents = sorted(
            [(self.intent_labels[i], float(p)) for i, p in enumerate(intent_probs) if p >= INTENT_THRESHOLD],
            key=lambda kv: -kv[1],
        )
        entities = self._decode_entities(text, enc["offset_mapping"], slot_ids)
        return intents, entities, intent_probs

    def _decode_entities(self, text, offsets, slot_ids) -> list[dict]:
        spans, current = [], None
        for i, (a, b) in enumerate(offsets):
            if a == b == 0:
                if current:
                    spans.append(current)
                    current = None
                continue
            lbl = self.slot_labels[slot_ids[i]]
            if lbl == "O":
                if current:
                    spans.append(current)
                    current = None
            elif lbl.startswith("B-"):
                if current:
                    spans.append(current)
                current = {"type": lbl[2:], "start": a, "end": b, "value": text[a:b]}
            elif lbl.startswith("I-") and current and current["type"] == lbl[2:]:
                current["end"] = b
                current["value"] = text[current["start"] : b]
            else:
                if current:
                    spans.append(current)
                current = None
        if current:
            spans.append(current)
        return spans


# ----------------------------------------------------------------------
# Deterministic value normalizers (pure Python, no model)
# ----------------------------------------------------------------------
def normalize_time_range(value: str) -> dict:
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


def normalize_rating(value: str) -> dict:
    digit = re.search(r"\d", value)
    if digit:
        return {"min_rating": int(digit.group())}
    if "favorite" in value.lower() or "favourite" in value.lower():
        return {"min_rating": 4}
    return {}


def normalize_tempo(value: str) -> dict:
    v = value.lower()
    bpm = re.search(r"(\d{2,3})\s*bpm", v)
    if bpm:
        n = int(bpm.group(1))
        return {"tempo_min": max(40, n - 10), "tempo_max": min(220, n + 10)}
    if "slow" in v:                            return {"tempo_min": 40,  "tempo_max": 90}
    if "medium" in v or "mid-tempo" in v:      return {"tempo_min": 90,  "tempo_max": 130}
    if "fast" in v:                            return {"tempo_min": 130, "tempo_max": 200}
    return {}


def normalize_energy(value: str) -> dict:
    v = value.lower()
    if "low" in v or "calm" in v: return {"energy_min": 0.0,  "energy_max": 0.35}
    if "medium" in v:             return {"energy_min": 0.35, "energy_max": 0.7}
    if "high" in v:               return {"energy_min": 0.7,  "energy_max": 1.0}
    return {}


def tempo_filter_token(value: str) -> str | None:
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


def energy_filter_token(value: str) -> str | None:
    v = value.lower()
    if "low" in v or "calm" in v: return "low"
    if "high" in v: return "high"
    if "medium" in v: return "medium"
    return None


# ----------------------------------------------------------------------
# Dispatcher: intents + entities -> concrete tool calls
# ----------------------------------------------------------------------
def dispatch(text: str, intents: list[tuple[str, float]], entities: list[dict]) -> list[tuple[str, dict]]:
    ents: dict[str, list[str]] = {}
    for e in entities:
        ents.setdefault(e["type"], []).append(e["value"])
    intent_names = {name for name, _ in intents}
    calls: list[tuple[str, dict]] = []

    # 1) song_similarity ------------------------------------------------
    # Pair songs with artists by order: song[i] + artist[i]. Multi-song
    # queries ("songs like S1 by A1 and S2 by A2") produce one call per pair.
    n_song_artist_pairs = 0
    if "song_similarity" in intent_names and "song" in ents and "artist" in ents:
        n_song_artist_pairs = min(len(ents["song"]), len(ents["artist"]))
        for i in range(n_song_artist_pairs):
            calls.append(("song_similarity", {
                "song_title": ents["song"][i],
                "song_artist": ents["artist"][i],
            }))

    # 2) artist_similarity (one call per artist) ------------------------
    # If song_similarity also fired, the first N artists were already consumed
    # as the songs' owners — only use the trailing artists here.
    if "artist_similarity" in intent_names and "artist" in ents:
        for a in ents["artist"][n_song_artist_pairs:]:
            calls.append(("artist_similarity", {"artist": a}))

    # 3) text_search ----------------------------------------------------
    if "text_search" in intent_names:
        desc = ents.get("description", [None])[0]
        if desc is None:
            desc = text  # fallback: use the whole query as the CLAP description
        args = {"description": desc}
        if "tempo" in ents:
            t = tempo_filter_token(ents["tempo"][0])
            if t: args["tempo_filter"] = t
        if "energy" in ents:
            e = energy_filter_token(ents["energy"][0])
            if e: args["energy_filter"] = e
        calls.append(("text_search", args))

    # 4) song_alchemy ---------------------------------------------------
    if "song_alchemy" in intent_names:
        add_items, sub_items = [], []
        for a in ents.get("add_artist", []):
            add_items.append({"type": "artist", "id": a})
        # If alchemy fires but model used generic 'artist' for multiple seeds,
        # promote them to add_items.
        if not add_items and len(ents.get("artist", [])) >= 2:
            for a in ents["artist"]:
                add_items.append({"type": "artist", "id": a})
        for a in ents.get("subtract_artist", []):
            sub_items.append({"type": "artist", "id": a})
        for g in ents.get("subtract_genre", []):
            sub_items.append({"type": "artist", "id": g})
        if add_items:
            args = {"add_items": add_items}
            if sub_items: args["subtract_items"] = sub_items
            calls.append(("song_alchemy", args))

    # 5) ai_brainstorm --------------------------------------------------
    if "ai_brainstorm" in intent_names:
        calls.append(("ai_brainstorm", {"user_request": text}))

    # 5b) lyrics_search ------------------------------------------------
    if "lyrics_search" in intent_names:
        # Prefer the extracted lyric topic(s); fall back to the raw text.
        topics = ents.get("lyrics_query", [])
        query = " and ".join(topics) if topics else text
        calls.append(("lyrics_search", {"query": query}))

    # 6) search_database (catch-all for genre/mood/tempo/etc.) ----------
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
            args.update(normalize_time_range(ents["time_range"][0]))
        if "tempo" in ents: args.update(normalize_tempo(ents["tempo"][0]))
        if "energy" in ents: args.update(normalize_energy(ents["energy"][0]))
        if "rating" in ents: args.update(normalize_rating(ents["rating"][0]))
        # When BOTH artist_similarity and search_database fire, the artist already
        # went to artist_similarity. Only add it here for pure search queries.
        if "artist" in ents and "artist_similarity" not in intent_names:
            args["artist"] = ents["artist"][0]
        if args:
            calls.append(("search_database", args))

    return calls


# ----------------------------------------------------------------------
# Demo queries: 5+ per tool, plus a multi-tool block
# ----------------------------------------------------------------------
DEMO_QUERIES: dict[str, list[str]] = {
    "song_similarity": [
        "songs like Bohemian Rhapsody by Queen",
        "find tracks similar to Hotel California by Eagles",
        "more like Stairway to Heaven by Led Zeppelin",
        "play things like Smells Like Teen Spirit by Nirvana",
        "give me music in the style of Imagine by John Lennon",
    ],
    "artist_similarity": [
        "songs by Beyoncé",
        "play me The Beatles",
        "tracks from Pink Floyd",
        "more music from Radiohead",
        "artists similar to Daft Punk",
        "play me Red Hot Chili Peppers, Blink 182 and Sum 41",
        "songs similar to Coldplay and Keane",
    ],
    "song_similarity (multi-pair)": [
        "songs like Bohemian Rhapsody by Queen and Hotel California by Eagles",
        "tracks similar to Creep by Radiohead, Wonderwall by Oasis and Karma Police by Radiohead",
    ],
    "song_similarity + artist_similarity combo": [
        "songs like Bohemian Rhapsody by Queen and more by David Bowie",
        "tracks similar to Hotel California by Eagles plus more Fleetwood Mac",
    ],
    "text_search": [
        "piano music with rain sounds",
        "energetic workout music with heavy bass",
        "chill lofi beats for studying",
        "romantic dreamy guitar tracks",
        "ambient electronic with synthesizers",
    ],
    "song_alchemy": [
        "mix of Beatles and Rolling Stones",
        "Iron Maiden meets Metallica meets Deep Purple",
        "blend Daft Punk and The Chemical Brothers",
        "Pink Floyd but not ballads",
        "combine Radiohead with Muse",
    ],
    "ai_brainstorm": [
        "Grammy winners from 2023",
        "viral TikTok hits",
        "best of Coachella 2024",
        "Christmas classics",
        "songs from the Woodstock festival",
    ],
    "lyrics_search": [
        "songs about heartbreak",
        "lyrics about growing up",
        "tracks that mention the rain",
        "songs whose lyrics talk about loneliness",
        "music with political lyrics",
    ],
    "search_database": [
        "pop songs from the 90s",
        "fast metal tracks",
        "happy danceable songs",
        "indie rock from 2010 onwards",
        "low energy ambient music",
    ],
    "multi_tool (intent combinations)": [
        "songs like Beyoncé but more chill",
        "pop songs like Taylor Swift",
        "piano music similar to Ludovico Einaudi",
        "Radiohead tracks rated 5 stars",
        "Christmas classics from Mariah Carey",
        "Bob Dylan songs about war",
        "90s rock songs about loneliness",
    ],
}


def print_prediction(router: Router, text: str) -> None:
    intents, entities, probs = router.predict(text)
    print(f'\nQUERY: "{text}"')
    # Always show full intent probability vector so the user can see the routing
    prob_str = "  ".join(f"{name}={probs[i]:.2f}" for i, name in enumerate(router.intent_labels))
    print(f"  intent probs: {prob_str}")
    if intents:
        fired = ", ".join(f"{n} ({p:.2f})" for n, p in intents)
        print(f"  -> FIRED:    {fired}")
    else:
        print(f"  -> FIRED:    (none above threshold {INTENT_THRESHOLD})")
    if entities:
        ent_str = ", ".join(f'{e["type"]}="{e["value"]}"' for e in entities)
        print(f"  -> entities: {ent_str}")
    else:
        print("  -> entities: (none)")
    calls = dispatch(text, intents, entities)
    if calls:
        print("  -> tool calls:")
        for name, args in calls:
            print(f"     • {name}({json.dumps(args, ensure_ascii=False)})")
    else:
        print("  -> tool calls: (none — would fall back to default search or LLM)")


def main() -> None:
    if not ONNX_PATH.exists() or not LABELS_PATH.exists():
        raise SystemExit(
            f"[inference] missing {ONNX_PATH.name} or {LABELS_PATH.name}. Run train.py first (or run main.py)."
        )
    router = Router()
    print(f"[inference] model loaded: {ONNX_PATH.name} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"[inference] intents: {router.intent_labels}")
    print(f"[inference] threshold for firing a tool: {INTENT_THRESHOLD}")
    print()

    for section, queries in DEMO_QUERIES.items():
        print(f"\n{'=' * 70}\n{section.upper()}\n{'=' * 70}")
        for q in queries:
            print_prediction(router, q)


if __name__ == "__main__":
    main()
