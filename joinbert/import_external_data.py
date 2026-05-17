"""
Import open-source human-written MIR queries and append them to
training_data.json (80%) and validation_data.json (20%).

Sources
-------
1. SNIPS NLU benchmark — PlayMusic intent (~2k crowdsourced queries, joint
   intent + slot already annotated). Apache-2.0.
       https://github.com/sonos/nlu-benchmark

2. CPCD — Conversational Playlist Curation Dataset (Google Research, 917
   human-to-human music discovery dialogues, multi-turn, with per-turn track
   metadata). CC-BY 4.0.
       https://github.com/google-research-datasets/cpcd

3. CPCD-intent (Doh et al., ISMIR 2024) — utterance-level intent + musical
   attribute annotations layered on top of CPCD. We *cross-join* CPCD-intent
   with CPCD raw on dialog_id, so the intent labels come from Doh and the
   span-taggable artist/track strings come from CPCD's track metadata.

Output
------
Both training_data.json and validation_data.json are read, the new examples
are appended in-place, and a one-line entry per source is added to
`_provenance_notes`. Re-running with `--force` re-imports; otherwise sources
already listed in `_imported_sources` are skipped.

Run
---
    python import_external_data.py             # all three sources
    python import_external_data.py --dry-run   # print stats only
    python import_external_data.py --force     # ignore _imported_sources guard
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "training_data.json"
VAL_PATH = HERE / "validation_data.json"

SEED = 2026
VAL_FRACTION = 0.20

SNIPS_PLAYMUSIC_URL = (
    "https://raw.githubusercontent.com/sonos/nlu-benchmark/master/"
    "2017-06-custom-intent-engines/PlayMusic/train_PlayMusic_full.json"
)
CPCD_JSONL_URLS = [
    "https://raw.githubusercontent.com/google-research-datasets/cpcd/main/"
    "data/cpcd_v1.dialogs.dev.jsonl",
    "https://raw.githubusercontent.com/google-research-datasets/cpcd/main/"
    "data/cpcd_v1.dialogs.test.jsonl",
]
CPCD_INTENT_REPO = "seungheondoh/cpcd-intent"
CPCD_INTENT_FILE = "data/train-00000-of-00001.parquet"

USER_INTENTS = {
    "song_similarity", "artist_similarity", "text_search", "song_alchemy",
    "ai_brainstorm", "search_database", "lyrics_search",
}
USER_SLOTS = {
    "add_artist", "album", "artist", "description", "energy", "event", "genre",
    "key", "lyrics_query", "mood", "rating", "scale", "song", "subtract_artist",
    "subtract_genre", "tempo", "time_range", "year",
}

# Lightweight in-text vocabs for CPCD substring matching. Kept small and
# unambiguous on purpose: each phrase must be specific enough that a random
# substring hit in a music conversation is almost certainly the slot value.
GENRE_TERMS = [
    "bossa nova", "trip hop", "trip-hop", "hip hop", "hip-hop", "drum and bass",
    "death metal", "black metal", "thrash metal", "post-rock", "post rock",
    "post-punk", "post punk", "dream pop", "synthwave", "shoegaze", "grunge",
    "krautrock", "bluegrass", "afrobeat", "reggae", "ska", "punk", "metal",
    "blues", "jazz", "soul", "funk", "disco", "country", "folk", "indie",
    "alternative", "classical", "ambient", "techno", "house", "trance",
    "dubstep", "trap", "lo-fi", "lofi", "garage", "edm", "rap", "r&b", "rnb",
    "rock", "pop", "gospel", "salsa", "tango", "samba",
]
MOOD_TERMS = [
    "melancholic", "melancholy", "uplifting", "nostalgic", "aggressive",
    "energetic", "romantic", "relaxing", "soothing", "intense", "peaceful",
    "cheerful", "dreamy", "groovy", "smooth", "moody", "happy", "sad", "dark",
    "chill", "calm", "epic", "intimate",
]
TEMPO_TERMS = ["uptempo", "downtempo", "mid-tempo", "midtempo", "danceable",
               "slow", "fast"]
ENERGY_TERMS = ["high energy", "high-energy", "low energy", "low-energy",
                "mellow", "high", "low"]

# A "decade-ish" expression -> time_range; a 4-digit year -> year.
YEAR_RE = re.compile(r"\b(19\d{2}|20[0-2]\d)\b")
DECADE_RE = re.compile(
    r"\b(?:(?:19|20)\d0s|'?\d0s|seventies|eighties|nineties|sixties|fifties|"
    r"2000s|2010s|2020s)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _download_text(url: str) -> str:
    print(f"  -> GET {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # SNIPS PlayMusic ships a few latin-1 / cp1252 encoded characters.
        return data.decode("latin-1")


def _emit(text: str, spans: list[dict], intents: list[str]) -> dict | None:
    """Build a Rasa-style inline-annotated example. Returns None if invalid."""
    if not intents or not all(i in USER_INTENTS for i in intents):
        return None
    spans = sorted(spans, key=lambda s: s["start"])
    # Drop overlapping spans (keep the first), drop empty/whitespace spans.
    clean: list[dict] = []
    last_end = -1
    for s in spans:
        if s["type"] not in USER_SLOTS:
            continue
        if s["start"] < last_end or s["start"] >= s["end"]:
            continue
        if not text[s["start"]:s["end"]].strip():
            continue
        clean.append(s)
        last_end = s["end"]
    out, cursor = [], 0
    for s in clean:
        out.append(text[cursor:s["start"]])
        out.append(f"[{text[s['start']:s['end']]}]({s['type']})")
        cursor = s["end"]
    out.append(text[cursor:])
    annotated = "".join(out).strip()
    if not annotated:
        return None
    return {"text": annotated, "intents": sorted(set(intents))}


def _find_spans(text: str, needle: str, slot_type: str) -> list[dict]:
    """Case-insensitive whole-substring search; returns all non-overlapping hits."""
    if not needle or len(needle) < 2:
        return []
    spans: list[dict] = []
    low_t = text.lower()
    low_n = needle.lower().strip()
    start = 0
    while True:
        idx = low_t.find(low_n, start)
        if idx < 0:
            break
        # Word-ish boundary check so "rap" doesn't match "rapid".
        left_ok = idx == 0 or not text[idx - 1].isalnum()
        end = idx + len(low_n)
        right_ok = end == len(text) or not text[end].isalnum()
        if left_ok and right_ok:
            spans.append({"start": idx, "end": end, "type": slot_type})
        start = idx + 1
    return spans


def _regex_spans(text: str, pattern: re.Pattern[str], slot_type: str) -> list[dict]:
    return [
        {"start": m.start(), "end": m.end(), "type": slot_type}
        for m in pattern.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Source 1 — SNIPS PlayMusic
# ---------------------------------------------------------------------------

# Mapping documented in validation_data.json's _provenance_notes (preserved here
# so import & val are consistent).
SNIPS_SLOT_MAP = {
    "track": "song",
    "artist": "artist",
    "album": "album",
    "genre": "genre",
    "year": "year",
    "sort": "rating",
    # Dropped on purpose: service (not a tool we expose), music_item
    # (we don't distinguish song/symphony/melody), playlist / playlist_owner
    # (no playlist tool).
}


def _snips_intent_from_slots(slot_types: set[str]) -> str | None:
    if "playlist" in slot_types or "playlist_owner" in slot_types:
        return None  # AddToPlaylist-style; we don't expose that tool
    if "track" in slot_types:
        return "song_similarity"
    if "album" in slot_types:
        return "search_database"
    if "artist" in slot_types and not slot_types & {"genre", "year"}:
        return "artist_similarity"
    if slot_types & {"genre", "year"}:
        return "search_database"
    if "artist" in slot_types:
        return "artist_similarity"
    return None


def convert_snips() -> list[dict]:
    raw = json.loads(_download_text(SNIPS_PLAYMUSIC_URL))
    rows = raw.get("PlayMusic", [])
    out: list[dict] = []
    skipped_no_intent = 0
    for row in rows:
        text_parts: list[str] = []
        spans: list[dict] = []
        slot_types: set[str] = set()
        cursor = 0
        skip = False
        for seg in row.get("data", []):
            chunk = seg["text"]
            ent = seg.get("entity")
            text_parts.append(chunk)
            if ent:
                if ent in ("playlist", "playlist_owner"):
                    skip = True
                    break
                mapped = SNIPS_SLOT_MAP.get(ent)
                if mapped == "year" and not YEAR_RE.search(chunk):
                    # SNIPS "year" is overloaded: decade words like "sixties"
                    # belong to our time_range, real 4-digit years to year.
                    mapped = "time_range"
                if mapped:
                    spans.append({
                        "start": cursor, "end": cursor + len(chunk), "type": mapped,
                    })
                    slot_types.add(ent)
            cursor += len(chunk)
        if skip:
            continue
        intent = _snips_intent_from_slots(slot_types)
        if intent is None:
            skipped_no_intent += 1
            continue
        ex = _emit("".join(text_parts), spans, [intent])
        if ex:
            out.append(ex)
    print(f"  SNIPS: kept {len(out)} examples, dropped {skipped_no_intent} (no mappable intent)")
    return out


# ---------------------------------------------------------------------------
# Source 2 + 3 — CPCD + Doh's intent annotations (joined)
# ---------------------------------------------------------------------------

# Doh intent -> our tool. Only intents that should *fire a tool* are mapped;
# greetings / accept-reject / wizard-side intents are dropped.
DOH_USER_INTENTS = {"initial_query", "positive_filter", "continue"}


def _doh_intent_to_tool(doh_intents: list[str], attrs: list[str]) -> list[str]:
    """Pick the tool(s) implied by Doh's intent labels + attribute mix.

    Intentionally conservative: Doh's `theme` attribute does NOT map to our
    lyrics_query slot (theme is high-level, lyrics_query is a literal lyric
    topic phrase), so we don't fire lyrics_search here. Doh's mood-only
    queries also don't map to our text_search (which expects descriptive
    multi-word sound paragraphs).
    """
    if not any(i in DOH_USER_INTENTS for i in doh_intents):
        return []
    attrs_s = set(attrs)
    tools: list[str] = []
    has_artist = bool(attrs_s & {"artist", "similar_artist"})
    has_track = bool(attrs_s & {"track", "similar_track"})
    has_genre = "genre" in attrs_s
    has_year_or_album = bool(attrs_s & {"year", "album"})

    if has_track:
        tools.append("song_similarity")
    if has_genre or has_year_or_album or "mood" in attrs_s or "tempo" in attrs_s:
        tools.append("search_database")
    if has_artist and not tools:
        tools.append("artist_similarity")
    return sorted(set(t for t in tools if t in USER_INTENTS))


def _load_cpcd_raw() -> dict[str, dict]:
    """dialog_id -> {tracks: {trackId: {title, artist, ...}}, ...}"""
    out: dict[str, dict] = {}
    for url in CPCD_JSONL_URLS:
        text = _download_text(url)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["id"]] = d
    print(f"  CPCD raw: loaded {len(out)} dialogs")
    return out


def _load_cpcd_intent() -> list[dict]:
    """Returns list of {unique_id, dialog_id, role, content, intent, music_attribute}."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("[import] missing huggingface_hub — pip install huggingface-hub") from e
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        raise SystemExit(
            "[import] missing pyarrow — pip install pyarrow\n"
            "         (only needed for the CPCD-intent source; ~5MB)"
        )
    path = hf_hub_download(
        repo_id=CPCD_INTENT_REPO, filename=CPCD_INTENT_FILE, repo_type="dataset"
    )
    table = pq.read_table(path)
    rows = table.to_pylist()
    print(f"  CPCD-intent: loaded {len(rows)} annotated utterances")
    return rows


def convert_cpcd() -> list[dict]:
    cpcd_raw = _load_cpcd_raw()
    rows = _load_cpcd_intent()

    out: list[dict] = []
    no_tool = no_span = wrong_role = ok = 0
    for row in rows:
        if row.get("role") != "user":
            wrong_role += 1
            continue
        intents = row.get("intent") or []
        attrs = row.get("music_attribute") or []
        if isinstance(intents, str):
            intents = [intents]
        if isinstance(attrs, str):
            attrs = [attrs]
        tools = _doh_intent_to_tool(list(intents), list(attrs))
        if not tools:
            no_tool += 1
            continue

        text = (row.get("content") or "").strip()
        if not text:
            continue

        dialog = cpcd_raw.get(row.get("dialog_id", ""))
        spans: list[dict] = []

        # Span-tag artists & track titles using this dialog's track metadata.
        if dialog:
            tracks = dialog.get("tracks") or {}
            seen_artists: set[str] = set()
            seen_titles: set[str] = set()
            for tinfo in tracks.values() if isinstance(tracks, dict) else tracks:
                if not isinstance(tinfo, dict):
                    continue
                a = (tinfo.get("artist") or "").strip()
                title = (tinfo.get("title") or tinfo.get("name") or "").strip()
                if a and a.lower() not in seen_artists:
                    seen_artists.add(a.lower())
                    spans.extend(_find_spans(text, a, "artist"))
                if title and title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    spans.extend(_find_spans(text, title, "song"))

        # Domain-vocabulary substring tags.
        for g in GENRE_TERMS:
            spans.extend(_find_spans(text, g, "genre"))
        for m in MOOD_TERMS:
            spans.extend(_find_spans(text, m, "mood"))
        for t in TEMPO_TERMS:
            spans.extend(_find_spans(text, t, "tempo"))
        for e in ENERGY_TERMS:
            spans.extend(_find_spans(text, e, "energy"))
        spans.extend(_regex_spans(text, YEAR_RE, "year"))
        spans.extend(_regex_spans(text, DECADE_RE, "time_range"))

        if not spans:
            # No spans found: training on this would push the model to predict
            # all-O for utterances that semantically contain entities. Skip.
            no_span += 1
            continue

        ex = _emit(text, spans, tools)
        if ex:
            out.append(ex)
            ok += 1

    print(
        f"  CPCD+Doh: kept {ok} examples "
        f"(skipped: wrong_role={wrong_role}, no_mapped_tool={no_tool}, no_spans={no_span})"
    )
    return out


# ---------------------------------------------------------------------------
# Split + append
# ---------------------------------------------------------------------------


def _stratified_split(
    examples: list[dict], rng: random.Random, val_fraction: float
) -> tuple[list[dict], list[dict]]:
    """80/20 split, but force every entity type seen in val to also appear in train."""
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]

    val_types: set[str] = set()
    for ex in val:
        for m in re.finditer(r"\]\(([^)]+)\)", ex["text"]):
            val_types.add(m.group(1))
    train_types: set[str] = set()
    for ex in train:
        for m in re.finditer(r"\]\(([^)]+)\)", ex["text"]):
            train_types.add(m.group(1))

    # Move one example per orphan-in-val type from val -> train.
    orphans = val_types - train_types
    if orphans:
        kept_val: list[dict] = []
        for ex in val:
            ex_types = {m.group(1) for m in re.finditer(r"\]\(([^)]+)\)", ex["text"])}
            if ex_types & orphans:
                train.append(ex)
                orphans -= ex_types
            else:
                kept_val.append(ex)
        val = kept_val
    return train, val


def _intent_histogram(examples: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for ex in examples:
        for i in ex["intents"]:
            c[i] += 1
    return dict(sorted(c.items()))


def _load_jsonfile(path: Path) -> dict:
    if not path.exists():
        return {"examples": []}
    return json.loads(path.read_text())


def _append_to_file(
    path: Path, new_examples: list[dict], source: str, note: str
) -> None:
    blob = _load_jsonfile(path)
    blob.setdefault("examples", [])
    blob.setdefault("_provenance_notes", [])
    blob.setdefault("_imported_sources", [])
    blob["examples"].extend(new_examples)
    blob["_provenance_notes"].append(note)
    if source not in blob["_imported_sources"]:
        blob["_imported_sources"].append(source)
    path.write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n")


def _already_imported(source: str) -> bool:
    for path in (TRAIN_PATH, VAL_PATH):
        if path.exists():
            blob = json.loads(path.read_text())
            if source in blob.get("_imported_sources", []):
                return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats only; don't write to disk.")
    ap.add_argument("--force", action="store_true",
                    help="Re-import sources even if listed in _imported_sources.")
    ap.add_argument("--skip-snips", action="store_true")
    ap.add_argument("--skip-cpcd", action="store_true")
    args = ap.parse_args()

    sources: list[tuple[str, callable, str]] = []
    if not args.skip_snips:
        sources.append((
            "snips_playmusic_v2017-06",
            convert_snips,
            "Imported SNIPS NLU benchmark / PlayMusic intent (Apache-2.0). "
            "Slot map: track->song, sort->rating, artist/album/genre/year preserved; "
            "service/music_item/playlist dropped.",
        ))
    if not args.skip_cpcd:
        sources.append((
            "cpcd_v1_with_doh_intent",
            convert_cpcd,
            "Imported CPCD (Google Research, CC-BY 4.0) joined with Doh et al. "
            "ISMIR'24 intent annotations (seungheondoh/cpcd-intent). Intent labels "
            "from Doh; artist/song spans matched against CPCD's per-dialog track "
            "metadata; genre/mood/tempo/energy/year/time_range tagged via domain "
            "vocabulary substring match.",
        ))

    rng = random.Random(SEED)
    total_train = total_val = 0

    for source, converter, note in sources:
        print(f"\n[{source}]")
        if _already_imported(source) and not args.force:
            print(f"  already in _imported_sources — skip (use --force to re-import)")
            continue

        examples = converter()
        if not examples:
            print(f"  no examples produced — skip")
            continue

        train_split, val_split = _stratified_split(examples, rng, VAL_FRACTION)
        print(f"  split: {len(train_split)} train + {len(val_split)} val")
        print(f"  train intents: {_intent_histogram(train_split)}")
        print(f"  val   intents: {_intent_histogram(val_split)}")

        if args.dry_run:
            print(f"  --dry-run: not writing")
            for ex in train_split[:3] + val_split[:2]:
                print(f"    sample: {ex}")
            continue

        _append_to_file(
            TRAIN_PATH, train_split, source,
            f"{note} ({len(train_split)} examples added to training set, seed={SEED}).",
        )
        _append_to_file(
            VAL_PATH, val_split, source,
            f"{note} ({len(val_split)} examples added to validation set, seed={SEED}).",
        )
        total_train += len(train_split)
        total_val += len(val_split)

    print(
        f"\n[done] appended {total_train} training + {total_val} validation examples"
        f"{' (dry-run)' if args.dry_run else ''}"
    )
    if not args.dry_run:
        print(f"       wrote: {TRAIN_PATH.name}, {VAL_PATH.name}")
        print(f"       next: re-run `python train.py` to retrain on the expanded set")


if __name__ == "__main__":
    sys.exit(main())
