"""
Augment training_data.json (80%) and validation_data.json (20%) with ~1500
paraphrased variants of ~500 sampled entries.

Strategies (each variant uses one or two, picked deterministically per entry):
    1. carrier-phrase substitution — swap "songs by X" → "play me X" /
       "music from X" / "anything by X"   (highest signal)
    2. synonym substitution         — swap mood/tempo/energy/genre values for
       near-synonyms                       (vocab enrichment)
    3. word-order variation         — for [description](description) slots,
       shuffle the words inside the span   (cheap diversity)
    4. casing & punctuation         — lowercase, add "plz", drop caps,
       drop terminal punctuation, optional "?" — mimics real-user typing

Sampling: stratified across sources via index ranges:
    - all 79 user_curated_v1 entries (highest leverage)
    - sample 300 from the SNIPS+CPCD real-human slice
    - sample 120 from the synthetic-template slice
Target: ~500 entries → up to 3 variants each → ~1500 new examples.

Idempotent via _imported_sources tag "augmented_v1". --force to re-import.
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

SOURCE_ID = "augmented_v1"
SEED = 7777
VAL_FRACTION = 0.20
N_VARIANTS = 3

# Counts mirror the import order in training_data.json:
#   [0 : N_SYNTHETIC)           — generate_data.py output
#   [N_SYNTHETIC : N_SYNTHETIC+N_REAL) — SNIPS + CPCD imports
#   [N_SYNTHETIC+N_REAL : end)  — user_curated_v1 imports
N_SYNTHETIC_TRAIN = 5330
N_REAL_TRAIN = 2127
N_CURATED_TRAIN = 63
SAMPLE_SYNTHETIC = 120
SAMPLE_REAL = 300
SAMPLE_CURATED = 79

INLINE_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# ---------------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------------

# Carrier-phrase rewrites. Keyed by intent → list of (pattern, replacements).
# Each pattern matches a prefix/structural fragment in the entry text; the
# replacements are substituted in place. Patterns are tried in order; the
# first that matches wins.
CARRIER_REWRITES = {
    "artist_similarity": [
        (re.compile(r"^\s*songs\s+by\s+", re.IGNORECASE),
         ["play me ", "music from ", "tracks by ", "anything by ",
          "give me ", "more from "]),
        (re.compile(r"^\s*play\s+me\s+", re.IGNORECASE),
         ["songs by ", "music from ", "give me ", "I want to hear ",
          "I'd like ", "queue up "]),
        (re.compile(r"^\s*music\s+from\s+", re.IGNORECASE),
         ["songs by ", "play me ", "tracks by ", "I want to hear "]),
        (re.compile(r"^\s*tracks\s+(?:by|from)\s+", re.IGNORECASE),
         ["songs by ", "play me ", "music from ", "anything by "]),
        (re.compile(r"^\s*more\s+(?:music\s+)?from\s+", re.IGNORECASE),
         ["more by ", "give me more ", "songs by ", "anything by "]),
        (re.compile(r"^\s*give\s+me\s+", re.IGNORECASE),
         ["play me ", "I want ", "queue ", "let me hear "]),
        (re.compile(r"^\s*i\s+want\s+(?:to\s+hear\s+)?", re.IGNORECASE),
         ["play me ", "give me ", "let me hear ", "queue "]),
        (re.compile(r"^\s*anything\s+by\s+", re.IGNORECASE),
         ["songs by ", "music from ", "tracks by ", "give me "]),
    ],
    "song_similarity": [
        (re.compile(r"^\s*songs\s+like\s+", re.IGNORECASE),
         ["tracks like ", "music like ", "more like ", "songs similar to ",
          "find me songs like ", "play me tracks like ", "give me things like "]),
        (re.compile(r"^\s*tracks?\s+(?:similar\s+to|like)\s+", re.IGNORECASE),
         ["songs like ", "music like ", "more like ", "give me tracks like "]),
        (re.compile(r"^\s*more\s+(?:music\s+)?like\s+", re.IGNORECASE),
         ["songs like ", "tracks like ", "give me more like ", "I want more like "]),
        (re.compile(r"^\s*find\s+me\s+", re.IGNORECASE),
         ["give me ", "play me ", "I want ", "queue up "]),
        (re.compile(r"^\s*similar\s+song\s+to\s+", re.IGNORECASE),
         ["songs like ", "tracks like ", "more like ", "give me things like "]),
        (re.compile(r"^\s*give\s+me\s+songs\s+like\s+", re.IGNORECASE),
         ["songs like ", "tracks like ", "more like ", "play me things like "]),
    ],
    "text_search": [
        (re.compile(r"^\s*find\s+me\s+", re.IGNORECASE),
         ["give me ", "I want ", "play me ", "I'm looking for ", "looking for "]),
        (re.compile(r"^\s*give\s+me\s+", re.IGNORECASE),
         ["find me ", "I want ", "play me ", "looking for "]),
        (re.compile(r"^\s*i'?m\s+looking\s+for\s+", re.IGNORECASE),
         ["give me ", "find me ", "I want ", "play me ", "looking for "]),
        (re.compile(r"^\s*tracks?\s+(?:featuring|with)\s+", re.IGNORECASE),
         ["songs with ", "music with ", "find me ", "give me "]),
        (re.compile(r"^\s*play\s+", re.IGNORECASE),
         ["I want ", "give me ", "find me "]),
    ],
    "lyrics_search": [
        (re.compile(r"^\s*songs\s+about\s+", re.IGNORECASE),
         ["tracks about ", "music about ", "find me songs about ",
          "play me songs about ", "give me tracks about "]),
        (re.compile(r"^\s*tracks\s+(?:about|with\s+lyrics\s+about)\s+", re.IGNORECASE),
         ["songs about ", "music about ", "lyrics about ",
          "find me tracks about "]),
        (re.compile(r"^\s*lyrics\s+(?:about|that\s+mention)\s+", re.IGNORECASE),
         ["songs about ", "tracks about ", "find lyrics that talk about ",
          "music about "]),
        (re.compile(r"^\s*music\s+about\s+", re.IGNORECASE),
         ["songs about ", "tracks about ", "find me songs about "]),
    ],
    "search_database": [
        (re.compile(r"^\s*give\s+me\s+", re.IGNORECASE),
         ["find me ", "play me ", "I want ", "show me ", "queue up "]),
        (re.compile(r"^\s*find\s+me\s+", re.IGNORECASE),
         ["give me ", "play me ", "I want ", "show me "]),
        (re.compile(r"^\s*play\s+(?:me\s+)?", re.IGNORECASE),
         ["give me ", "find me ", "I want ", "queue up "]),
    ],
    "song_alchemy": [
        (re.compile(r"^\s*mix\s+of\s+", re.IGNORECASE),
         ["blend ", "combination of ", "cross between ", "fusion of "]),
        (re.compile(r"^\s*blend\s+", re.IGNORECASE),
         ["mix of ", "combine ", "cross ", "fuse "]),
        (re.compile(r"^\s*combine\s+", re.IGNORECASE),
         ["blend ", "mix ", "cross "]),
    ],
    "ai_brainstorm": [
        (re.compile(r"^\s*best\s+of\s+", re.IGNORECASE),
         ["the greatest ", "top of ", "the highlights of ", "essential "]),
        (re.compile(r"^\s*give\s+me\s+", re.IGNORECASE),
         ["play me ", "find me ", "I want ", "show me "]),
    ],
}

# Synonym substitutions, applied only when the original word/phrase appears
# unambiguously and outside an entity annotation conflict. Conservative —
# meanings are kept close.
SYNONYMS_BY_SLOT = {
    "mood": {
        "happy": ["cheerful", "upbeat", "joyful"],
        "sad": ["melancholic", "downcast", "gloomy"],
        "chill": ["mellow", "laid-back", "relaxed"],
        "energetic": ["high-energy", "lively", "pumped"],
        "dreamy": ["ethereal", "hazy", "wistful"],
        "romantic": ["tender", "loving", "amorous"],
        "dark": ["brooding", "somber", "ominous"],
        "intense": ["fierce", "powerful", "heavy"],
        "uplifting": ["inspiring", "feel-good", "soaring"],
        "moody": ["atmospheric", "introspective", "broody"],
        "danceable": ["groovy", "bouncy", "rhythmic"],
        "aggressive": ["forceful", "hard-hitting", "fierce"],
        "nostalgic": ["wistful", "sentimental", "reminiscent"],
        "epic": ["grand", "cinematic", "sweeping"],
        "relaxed": ["mellow", "easy", "calm"],
        "soothing": ["calming", "tranquil", "peaceful"],
    },
    "tempo": {
        "slow": ["downtempo", "slow-paced", "laid-back"],
        "fast": ["uptempo", "fast-paced", "quick"],
        "medium": ["mid-tempo", "moderate", "midpaced"],
    },
    "energy": {
        "high": ["intense", "punchy", "driving"],
        "low": ["mellow", "gentle", "subdued"],
        "calm": ["serene", "tranquil", "peaceful"],
        "intense": ["fierce", "high-energy", "punchy"],
        "mellow": ["chilled", "soft", "easy"],
    },
    "genre": {
        "hip hop": ["hip-hop", "rap"],
        "hip-hop": ["hip hop", "rap"],
        "rap": ["hip hop", "hip-hop"],
        "rnb": ["r&b", "R&B"],
        "r&b": ["rnb", "R'n'B"],
        "rock": ["rock music"],
        "pop": ["pop music"],
        "edm": ["electronic dance music", "EDM"],
        "lo-fi": ["lofi"],
        "lofi": ["lo-fi"],
        "trip-hop": ["trip hop"],
        "trip hop": ["trip-hop"],
    },
    "rating": {
        "top": ["best", "greatest"],
        "best": ["top", "greatest"],
        "top rated": ["highest rated", "best rated"],
        "favorites": ["faves", "favourites"],
        "5 stars": ["five-star", "top rated"],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intents(ex: dict) -> list[str]:
    return ex.get("intents") or []


def _replace_in_annotated(text: str, old_inner: str, new_inner: str,
                          slot_type: str) -> str:
    """Replace `[old_inner](slot_type)` with `[new_inner](slot_type)` once."""
    pattern = re.compile(r"\[" + re.escape(old_inner) + r"\]\(" + re.escape(slot_type) + r"\)")
    return pattern.sub(f"[{new_inner}]({slot_type})", text, count=1)


def variant_carrier(text: str, intents: list[str], rng: random.Random) -> str | None:
    """Apply a carrier-phrase rewrite if any matches."""
    candidates = []
    for intent in intents:
        for pat, replacements in CARRIER_REWRITES.get(intent, []):
            if pat.search(text):
                candidates.append((pat, replacements))
    if not candidates:
        return None
    pat, replacements = rng.choice(candidates)
    new_prefix = rng.choice(replacements)
    return pat.sub(new_prefix, text, count=1)


def variant_synonym(text: str, rng: random.Random) -> str | None:
    """Swap one slot value for a near-synonym from SYNONYMS_BY_SLOT."""
    matches = list(INLINE_RE.finditer(text))
    rng.shuffle(matches)
    for m in matches:
        value, slot = m.group(1), m.group(2)
        syns = SYNONYMS_BY_SLOT.get(slot, {}).get(value.lower())
        if not syns:
            continue
        new = rng.choice(syns)
        return _replace_in_annotated(text, value, new, slot)
    return None


def variant_word_order(text: str, rng: random.Random) -> str | None:
    """For [description](description) spans, shuffle inner word order."""
    m = re.search(r"\[([^\]]+)\]\(description\)", text)
    if not m:
        return None
    inner = m.group(1).strip()
    words = inner.split()
    if len(words) < 3:
        return None
    new_words = words[:]
    for _ in range(5):
        rng.shuffle(new_words)
        if new_words != words:
            break
    if new_words == words:
        return None
    return _replace_in_annotated(text, inner, " ".join(new_words), "description")


def variant_casing_punct(text: str, rng: random.Random) -> str | None:
    """Apply a lightweight casing/punctuation tweak. Always returns a string
    (this is the fallback when richer strategies fail), but caller still
    de-duplicates against the original."""
    choices = []

    # lowercase EVERYTHING outside slot markers
    def lower_outside_brackets(s: str) -> str:
        out, depth = [], 0
        for ch in s:
            if ch == "[":
                depth += 1
                out.append(ch)
            elif ch == "]":
                depth -= 1
                out.append(ch)
            elif ch == "(":
                depth += 1
                out.append(ch)
            elif ch == ")":
                depth -= 1
                out.append(ch)
            elif depth == 0:
                out.append(ch.lower())
            else:
                out.append(ch)
        return "".join(out)

    lo = lower_outside_brackets(text)
    if lo != text:
        choices.append(lo)

    # Append "plz" if not already there
    if "plz" not in text.lower() and not text.endswith("?"):
        choices.append(text.rstrip(".!? ") + " plz")

    # Strip terminal punctuation
    stripped = text.rstrip(".!?")
    if stripped != text:
        choices.append(stripped)

    # Add trailing "?" if the entry looks question-like ("can you...", "what...")
    if re.match(r"^\s*(can you|what|where|how|do you have)", text, re.IGNORECASE) and not text.endswith("?"):
        choices.append(text.rstrip(".!? ") + "?")

    # Convert "I am" -> "I'm" or vice versa
    if "I am " in text:
        choices.append(text.replace("I am ", "I'm ", 1))
    elif "I'm " in text:
        choices.append(text.replace("I'm ", "I am ", 1))

    return rng.choice(choices) if choices else None


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------


def make_variants(ex: dict, rng: random.Random, n: int = N_VARIANTS) -> list[dict]:
    """Try multiple strategies, return up to `n` unique variants."""
    text = ex["text"]
    intents = _intents(ex)
    seen = {text}
    out: list[dict] = []

    # Strategies in priority order; each may return None.
    strategies = [
        lambda t, r: variant_carrier(t, intents, r),
        lambda t, r: variant_synonym(t, r),
        lambda t, r: variant_word_order(t, r),
        lambda t, r: variant_casing_punct(t, r),
    ]

    # Try up to N_VARIANTS * 3 attempts, mixing single + composed strategies.
    attempts = 0
    while len(out) < n and attempts < n * 4:
        attempts += 1
        base = text
        # 50% chance to compose two strategies for more variety
        n_strats = 2 if rng.random() < 0.5 else 1
        rng.shuffle(strategies)
        for s in strategies[:n_strats]:
            r = s(base, rng)
            if r is not None:
                base = r
        if base not in seen:
            seen.add(base)
            out.append({"text": base, "intents": list(intents)})
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_stratified(examples: list[dict], rng: random.Random) -> list[dict]:
    """Pull from index ranges that correspond to known sources."""
    syn_end = N_SYNTHETIC_TRAIN
    real_end = N_SYNTHETIC_TRAIN + N_REAL_TRAIN

    synthetic = examples[:syn_end]
    real = examples[syn_end:real_end]
    curated = examples[real_end:real_end + N_CURATED_TRAIN]

    def take(pool: list[dict], n: int) -> list[dict]:
        if len(pool) <= n:
            return pool[:]
        return rng.sample(pool, n)

    sampled = take(curated, SAMPLE_CURATED) + take(real, SAMPLE_REAL) + take(synthetic, SAMPLE_SYNTHETIC)
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Append (mirrors import_external_data.py / import_user_queries.py)
# ---------------------------------------------------------------------------


def _stratified_split(examples: list[dict], rng: random.Random,
                      val_fraction: float) -> tuple[list[dict], list[dict]]:
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]

    def types_in(exs):
        t: set[str] = set()
        for ex in exs:
            for m in INLINE_RE.finditer(ex["text"]):
                t.add(m.group(2))
        return t

    orphans = types_in(val) - types_in(train)
    if orphans:
        kept_val: list[dict] = []
        for ex in val:
            ex_types = {m.group(2) for m in INLINE_RE.finditer(ex["text"])}
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

    rng = random.Random(SEED)
    train_examples = json.loads(TRAIN_PATH.read_text())["examples"]
    print(f"[{SOURCE_ID}] loaded {len(train_examples)} source training examples")

    sampled = sample_stratified(train_examples, rng)
    print(f"[{SOURCE_ID}] sampled {len(sampled)} entries for augmentation")

    variants: list[dict] = []
    seen_texts = {ex["text"] for ex in train_examples}
    n_skipped_no_variants = 0
    for ex in sampled:
        v = make_variants(ex, rng, n=N_VARIANTS)
        # de-dupe against existing training set + previous variants from this run
        for nv in v:
            if nv["text"] in seen_texts:
                continue
            seen_texts.add(nv["text"])
            variants.append(nv)
        if not v:
            n_skipped_no_variants += 1

    print(f"[{SOURCE_ID}] generated {len(variants)} unique variants"
          f" (avg {len(variants) / max(1, len(sampled)):.2f}/entry, "
          f"{n_skipped_no_variants} entries produced 0 variants)")

    train_split, val_split = _stratified_split(variants, rng, VAL_FRACTION)
    print(f"  split: {len(train_split)} train + {len(val_split)} val")
    print(f"  train intents: {_intent_histogram(train_split)}")
    print(f"  val   intents: {_intent_histogram(val_split)}")

    if args.dry_run:
        print("  --dry-run: not writing")
        for ex in train_split[:4] + val_split[:2]:
            print(f"    sample: {ex}")
        return

    note = (
        "Augmented from existing training data with carrier-phrase, synonym, "
        "word-order, and casing/punctuation paraphrases."
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


if __name__ == "__main__":
    sys.exit(main())
