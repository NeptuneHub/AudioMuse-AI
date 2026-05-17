"""
Extract the ~2000 AI-generated examples from training_data.json before regenerating.
Run once, then commit joinbert/curated_ai_v2.json alongside the script.

Background: training_data.json contains examples from multiple sources appended in order:
- generate_data.py synthetic: indices 0-5329 (5330 examples)
- SNIPS import: indices 5330-6734 (1405 examples)
- CPCD import: indices 6735-7456 (722 examples)
- user_curated_v1: indices 7457-7519 (63 examples)
- AI-generated (no source marker): indices 7520-9519 (2000 examples) ← PRESERVE THESE
- augmented_v1: indices 9520-10410 (891 examples)

This script extracts the AI-generated block and saves it standalone.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "training_data.json"
OUT = HERE / "curated_ai_v2.json"

# Documented source sizes (must match actual import scripts + generate_data.py)
N_SYNTHETIC = 5330  # generate_data.py
N_SNIPS = 1405  # import_external_data.py (SNIPS)
N_CPCD = 722  # import_external_data.py (CPCD)
N_USER_CURATED = 63  # import_user_queries.py

START = N_SYNTHETIC + N_SNIPS + N_CPCD + N_USER_CURATED  # = 7520
END = START + 2000  # = 9520

print(f"Loading {TRAIN.name}...")
blob = json.loads(TRAIN.read_text())
examples = blob["examples"]

print(f"Total examples in file: {len(examples)}")
print(f"Extracting AI-generated examples from indices [{START}:{END}]...")

ai_examples = examples[START:END]
print(f"  Extracted {len(ai_examples)} examples")

# Show samples for verification
print(f"\nSample [0] (should be AI-generated, not augmented style):")
print(f"  {ai_examples[0]}")
print(f"\nSample [-1]:")
print(f"  {ai_examples[-1]}")

# Count by intent
intent_count = {}
for ex in ai_examples:
    for intent in ex.get("intents", []):
        intent_count[intent] = intent_count.get(intent, 0) + 1

print(f"\nIntent distribution in AI examples:")
for intent, count in sorted(intent_count.items(), key=lambda x: -x[1]):
    print(f"  {intent}: {count}")

# Save standalone
out_data = {"examples": ai_examples, "_source": "curated_ai_v2", "_count": len(ai_examples)}
OUT.write_text(json.dumps(out_data, indent=2, ensure_ascii=False) + "\n")
print(f"\n✓ Saved → {OUT}")
print(f"  Next: git add {OUT.name}")
