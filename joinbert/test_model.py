"""
Test the enhanced JointBERT model with verification queries from the training plan.

Tests three main categories:
1. Multi-song format support (artist - song)
2. Top songs of X routing to ai_braystorm
3. Vocal style support in search_database
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inference import Router

# Verification queries from the plan
VERIFICATION_QUERIES = [
    {
        "query": "Similar song to Red Hot Chili Peppers - By The Way and ed sheeran 2step",
        "expected_intent": "song_similarity",
        "test_name": "Multi-song dash format (RHCP - By The Way + Ed Sheeran - 2step)",
        "critical": True,
    },
    {
        "query": "Give me the top songs of Madonna",
        "expected_intent": "ai_brainstorm",
        "test_name": "Top songs routing to ai_braystorm",
        "critical": True,
    },
    {
        "query": "Pop song with female vocalist from 2026",
        "expected_intent": "search_database",
        "test_name": "Female vocalist in search_database",
        "critical": True,
    },
    {
        "query": "Give me song similar to blink182, redhot chili peppers and iron maiden",
        "expected_intent": "artist_similarity",
        "test_name": "Multiple artists (3-way concatenation)",
        "critical": True,
    },
    {
        "query": "Similar to RHCP-By The Way, Ed Sheeran-2step, and Iron Maiden-Run to the Hills",
        "expected_intent": "song_similarity",
        "test_name": "Multiple songs with 3-way concatenation",
        "critical": True,
    },
    {
        "query": "top radio songs of madonna",
        "expected_intent": "ai_brainstorm",
        "test_name": "Radio hits routing to ai_braystorm",
        "critical": True,
    },
    {
        "query": "top radio song of 2026",
        "expected_intent": "ai_brainstorm",
        "test_name": "Radio song with year to ai_braystorm",
        "critical": True,
    },
    {
        "query": "top pop radio song of the last year",
        "expected_intent": "ai_brainstorm",
        "test_name": "Pop radio song with time range to ai_braystorm",
        "critical": True,
    },
    {
        "query": "songs by Madonna",
        "expected_intent": "artist_similarity",
        "test_name": "Pure artist similarity (prevent regression)",
        "critical": True,
    },
    {
        "query": "songs like Hotel California by Eagles",
        "expected_intent": "song_similarity",
        "test_name": "Song similarity (prevent regression)",
        "critical": True,
    },
    {
        "query": "energetic dance from 90s",
        "expected_intent": "search_database",
        "test_name": "Search with mood and decade",
        "critical": False,
    },
    {
        "query": "songs about loneliness",
        "expected_intent": "lyrics_search",
        "test_name": "Lyrics search routing",
        "critical": False,
    },
]


def main() -> None:
    try:
        router = Router()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print(f"✓ Model loaded successfully")
    print(f"  Intents: {router.intent_labels}")
    print(f"  Slots: {len(router.slot_labels)} entity types")
    print()

    passed = 0
    failed = 0
    critical_failed = False

    for test in VERIFICATION_QUERIES:
        query = test["query"]
        expected = test["expected_intent"]
        name = test["test_name"]
        is_critical = test["critical"]

        try:
            intents, entities, intent_probs = router.predict(query)
            top_intent = intents[0][0] if intents else "NONE"
            top_conf = intents[0][1] if intents else 0.0

            passed_test = top_intent == expected
            status = "✓ PASS" if passed_test else "✗ FAIL"

            if not passed_test and is_critical:
                critical_failed = True

            result_str = f"{status} | {name}"
            result_str += f"\n       Query: '{query}'"
            result_str += f"\n       Expected: {expected}, Got: {top_intent} (confidence: {top_conf:.3f})"

            if entities:
                entity_str = ", ".join(
                    f"{e['type']}='{e['value']}'" for e in entities
                )
                result_str += f"\n       Entities: {entity_str}"

            if len(intents) > 1:
                other_intents = ", ".join(
                    f"{name}({conf:.3f})" for name, conf in intents[1:4]
                )
                result_str += f"\n       Alt intents: {other_intents}"

            print(result_str)
            print()

            if passed_test:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"✗ ERROR | {name}")
            print(f"       Query: '{query}'")
            print(f"       Error: {e}")
            print()
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")

    if critical_failed:
        print()
        print("⚠️  CRITICAL FAILURES DETECTED - Fix these before deployment")
        sys.exit(1)
    elif failed > 0:
        print()
        print("⚠️  Some non-critical tests failed - Review before deployment")
        sys.exit(0)
    else:
        print()
        print("✓ All tests passed! Model ready for deployment.")
        sys.exit(0)


if __name__ == "__main__":
    main()
