"""
Append the curated AI-generated examples (curated_ai_v2.json) back to training/validation data
after regeneration.

This is a companion script to export_ai_examples.py. It restores the 2000 AI examples that
were extracted before running the full regeneration pipeline.

Idempotent: once 'curated_ai_v2' is in _imported_sources, re-running is a no-op (use --force).

Run:
    python import_ai_examples.py
    python import_ai_examples.py --dry-run
    python import_ai_examples.py --force
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "training_data.json"
VAL_PATH = HERE / "validation_data.json"
CURATED_PATH = HERE / "curated_ai_v2.json"

SOURCE_ID = "curated_ai_v2"
SEED = 5555  # arbitrary seed for reproducibility
VAL_FRACTION = 0.20  # split 80/20 train/val


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

    if not CURATED_PATH.exists():
        print(f"ERROR: {CURATED_PATH.name} not found. Run export_ai_examples.py first.")
        sys.exit(1)

    if _already_imported() and not args.force:
        print(f"[{SOURCE_ID}] already imported — use --force to re-import")
        return

    # Load the curated AI examples
    curated_blob = json.loads(CURATED_PATH.read_text())
    all_examples = curated_blob.get("examples", [])
    print(f"[{SOURCE_ID}] loaded {len(all_examples)} curated AI examples")

    # Split 80/20 train/val
    rng = random.Random(SEED)
    shuffled = all_examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * VAL_FRACTION)))
    val_split = shuffled[:n_val]
    train_split = shuffled[n_val:]

    print(f"  split: {len(train_split)} train + {len(val_split)} val")

    if args.dry_run:
        print("  --dry-run: not writing")
        for ex in train_split[:2] + val_split[:1]:
            print(f"    sample: {ex}")
        return

    note = (
        "Imported curated_ai_v2: 2000 AI-generated examples preserved from previous "
        "training run, restored after full regeneration pipeline."
    )
    _append_to_file(
        TRAIN_PATH,
        train_split,
        f"{note} ({len(train_split)} examples added to training set, seed={SEED}).",
    )
    _append_to_file(
        VAL_PATH,
        val_split,
        f"{note} ({len(val_split)} examples added to validation set, seed={SEED}).",
    )
    print(f"[done] appended {len(train_split)} training + {len(val_split)} validation examples")
    print(f"       wrote: {TRAIN_PATH.name}, {VAL_PATH.name}")


if __name__ == "__main__":
    sys.exit(main())
