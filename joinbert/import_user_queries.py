"""
Append a curated set of real user-style queries to training_data.json (80%)
and validation_data.json (20%).

Covers patterns that the synthetic templates and the SNIPS/CPCD imports don't
hit, observed in actual usage:

    * song_similarity using "ARTIST - SONG" (dash separator, not "by")
    * multi-artist artist_similarity ("Songs from X and Y", "songs that sound
      like X, Y and Z")
    * "top songs of X" / "best of X" routed to artist_similarity
    * numeric BPM tempo ("120 BPM")
    * hyphenated tempo descriptors (fast-paced, mid-tempo, slow-paced)
    * vocal style descriptors (raspy, falsetto, breathy, autotuned, ...)
    * short multi-attribute text_search descriptions (2-5 word combos)

Idempotent: once 'user_curated_v1' is in _imported_sources of either file,
re-running is a no-op (use --force to re-import).

Run:
    python import_user_queries.py
    python import_user_queries.py --dry-run
    python import_user_queries.py --force
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "training_data.json"
VAL_PATH = HERE / "validation_data.json"

SOURCE_ID = "user_curated_v1"
SEED = 4242
VAL_FRACTION = 0.20

# ---------------------------------------------------------------------------
# The actual examples. Mostly user-provided; a few extra variations added per
# pattern so the model sees the shape more than once.
# ---------------------------------------------------------------------------

# (1) song_similarity with "ARTIST - SONG" dash separator
DASH_SONG_SIM = [
    "Similar song to [Red Hot Chili Peppers](artist) - [By The Way](song)",
    "songs like [Pink Floyd](artist) - [Wish You Were Here](song)",
    "find me tracks similar to [Queen](artist) - [Bohemian Rhapsody](song)",
    "more like [Radiohead](artist) - [Karma Police](song)",
    "give me songs like [The Beatles](artist) - [Hey Jude](song)",
    "[Nirvana](artist) - [Smells Like Teen Spirit](song) and similar",
    "tracks in the style of [Daft Punk](artist) - [Get Lucky](song)",
]

# (2) Multi-artist artist_similarity
MULTI_ARTIST_SIM = [
    "Songs from [Blink-182](artist) and [Green Day](artist)",
    "Songs that sound like [Iron Maiden](artist), [Metallica](artist) and [Deep Purple](artist)",
    "music from [The Beatles](artist) and [The Rolling Stones](artist)",
    "play me [Radiohead](artist), [Muse](artist) and [Coldplay](artist)",
    "tracks by [Daft Punk](artist) and [The Chemical Brothers](artist)",
    "stuff like [Phoebe Bridgers](artist) and [Mitski](artist)",
    "songs from [Tame Impala](artist) and [MGMT](artist)",
    "artists similar to [Wu-Tang Clan](artist) and [Nas](artist)",
]

# (3) "Top songs of X" / "Best of X" -> artist_similarity (with a rating
# tag where the user explicitly says "top"/"best")
TOP_OF_ARTIST = [
    "Give me the [top](rating) songs of [Madonna](artist)",
    "[top](rating) tracks by [The Beatles](artist)",
    "[best](rating) of [Queen](artist)",
    "the [greatest hits](rating) of [Michael Jackson](artist)",
    "[top rated](rating) [David Bowie](artist) songs",
    "play me the [best](rating) [Pink Floyd](artist)",
]

# (4) Numeric BPM tempo searches
BPM_SEARCH = [
    "[Danceable](mood) [happy](mood) songs with [120 BPM](tempo)",
    "[energetic](mood) tracks at [140 BPM](tempo)",
    "[chill](mood) songs around [80 BPM](tempo)",
    "[uplifting](mood) [pop](genre) at [128 BPM](tempo)",
    "give me [techno](genre) at [130 BPM](tempo)",
    "[romantic](mood) ballads around [70 BPM](tempo)",
    "[fast](tempo) [punk](genre) above [180 BPM](tempo)",
    "[hip hop](genre) tracks at [90 BPM](tempo)",
]

# (5) Short multi-attribute descriptions -> text_search.
# Whole phrase tagged as `description`, mirroring the existing pattern
# ("[hypnotic motorik krautrock with steady drums](description)" etc.).
SHORT_DESCRIPTIONS = [
    "[female vocal romantic trap](description)",
    "[synth indie pop raspy](description)",
    "[sad hard rock male vocal](description)",
    "[funk falsetto energetic](description)",
    "[groovy sax blues](description)",
    "[classical relaxed piano](description)",
    "[belting jazz happy](description)",
    "[tabla afrobeat fast-paced](description)",
    "[harmonized vocals slow-paced electronica](description)",
    "[autotuned gospel excited](description)",
    "[breathy aggressive house](description)",
    "[smooth folk mid-tempo](description)",
    "[deep voice r&b dark](description)",
    "[punk guitar angry](description)",
    "[metal choir dreamy](description)",
    "[chant reggae trumpet](description)",
    "[high-pitched brass hip-hop](description)",
    "[disco whispered drum machine](description)",
    "[happy whispered indie pop](description)",
    "[synth energetic raspy](description)",
    "[rock slow-paced cello](description)",
    "[falsetto jazz excited](description)",
    "[r&b male vocal romantic](description)",
    "[harmonized vocals dark trap](description)",
    "[smooth blues sax](description)",
    "[high-pitched fast-paced soul](description)",
    "[female vocal sad hip-hop](description)",
    "[congas aggressive soul](description)",
    "[mid-tempo afrobeat autotuned](description)",
    "[belting funk groovy](description)",
    "[angry alternative breathy](description)",
    "[gospel choir steelpan](description)",
    "[viola relaxed folk](description)",
    "[dreamy rhodes metal](description)",
    "[acoustic guitar country chant](description)",
    "[deep voice orchestra reggae](description)",
    "[fast-paced synth progressive rock](description)",
    "[hard rock raspy romantic](description)",
    "[fast-paced electric guitar progressive rock](description)",
    "[hard rock aggressive breathy](description)",
    "[rock high-pitched energetic](description)",
    "[autotuned energetic hip-hop](description)",
    "[raspy fast-paced blues](description)",
    "[belting electronica energetic](description)",
    "[whispered indie pop aggressive](description)",
    "[harmonized vocals aggressive synth](description)",
    "[orchestra whispered romantic](description)",
    "[belting mid-tempo progressive rock](description)",
    "[autotuned pop mid-tempo](description)",
    "[pop energetic synthesizer](description)",
]


def _build_examples() -> list[dict]:
    out: list[dict] = []
    for text in DASH_SONG_SIM:
        out.append({"text": text, "intents": ["song_similarity"]})
    for text in MULTI_ARTIST_SIM:
        out.append({"text": text, "intents": ["artist_similarity"]})
    for text in TOP_OF_ARTIST:
        out.append({"text": text, "intents": ["artist_similarity"]})
    for text in BPM_SEARCH:
        out.append({"text": text, "intents": ["search_database"]})
    for text in SHORT_DESCRIPTIONS:
        out.append({"text": text, "intents": ["text_search"]})
    return out


# ---------------------------------------------------------------------------
# Split + append (mirrors import_external_data.py's helpers)
# ---------------------------------------------------------------------------


def _stratified_split(
    examples: list[dict], rng: random.Random, val_fraction: float
) -> tuple[list[dict], list[dict]]:
    """80/20 split, forcing every entity type seen in val to also exist in train."""
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]

    def types_in(exs):
        t: set[str] = set()
        for ex in exs:
            for m in re.finditer(r"\]\(([^)]+)\)", ex["text"]):
                t.add(m.group(1))
        return t

    orphans = types_in(val) - types_in(train)
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


def _append_to_file(path: Path, new_examples: list[dict], note: str) -> None:
    blob = json.loads(path.read_text())
    blob.setdefault("examples", [])
    blob.setdefault("_provenance_notes", [])
    blob.setdefault("_imported_sources", [])
    blob["examples"].extend(new_examples)
    blob["_provenance_notes"].append(note)
    if SOURCE_ID not in blob["_imported_sources"]:
        blob["_imported_sources"].append(SOURCE_ID)
    path.write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n")


def _already_imported() -> bool:
    for path in (TRAIN_PATH, VAL_PATH):
        if path.exists():
            blob = json.loads(path.read_text())
            if SOURCE_ID in blob.get("_imported_sources", []):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if _already_imported() and not args.force:
        print(f"[{SOURCE_ID}] already imported — use --force to re-import")
        return

    examples = _build_examples()
    print(f"[{SOURCE_ID}] built {len(examples)} curated examples")

    rng = random.Random(SEED)
    train_split, val_split = _stratified_split(examples, rng, VAL_FRACTION)
    print(f"  split: {len(train_split)} train + {len(val_split)} val")
    print(f"  train intents: {_intent_histogram(train_split)}")
    print(f"  val   intents: {_intent_histogram(val_split)}")

    if args.dry_run:
        print("  --dry-run: not writing")
        for ex in train_split[:3] + val_split[:2]:
            print(f"    sample: {ex}")
        return

    note = (
        "Imported user_curated_v1: real-user-style queries covering gaps in "
        "synthetic+SNIPS+CPCD — dash song-similarity, multi-artist, "
        "top-songs-of-X, numeric BPM tempo, hyphenated tempo descriptors, "
        "vocal-style words, short multi-attribute descriptions."
    )
    _append_to_file(
        TRAIN_PATH, train_split,
        f"{note} ({len(train_split)} examples added to training set, seed={SEED}).",
    )
    _append_to_file(
        VAL_PATH, val_split,
        f"{note} ({len(val_split)} examples added to validation set, seed={SEED}).",
    )
    print(f"[done] appended {len(train_split)} training + {len(val_split)} validation examples")
    print(f"       wrote: {TRAIN_PATH.name}, {VAL_PATH.name}")


if __name__ == "__main__":
    sys.exit(main())
