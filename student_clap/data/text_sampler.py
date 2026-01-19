import json
import random
from pathlib import Path

def sample_text_queries(json_path, categories, n_samples=2000):
    import itertools

    with open(json_path, 'r') as f:
        data = json.load(f)

    # All category names (fixed order)
    all_cats = ["Genre_Style", "Instrumentation_Vocal", "Emotion_Mood", "Voice_Type"]

    # Generate all non-empty combinations of categories, all possible orders
    templates = []
    for r in range(1, len(all_cats)+1):
        for combo in itertools.permutations(all_cats, r):
            templates.append(list(combo))

    # Remove duplicates (permutations can repeat for r=1)
    templates = [list(x) for x in set(tuple(t) for t in templates)]

    # If n_samples is 0, default to 50,000
    if n_samples == 0:
        n_samples = 50000

    selected = []
    case_choices = ['asis', 'upper', 'lower']
    for _ in range(n_samples):
        template = random.choice(templates)
        try:
            query = []
            for cat in template:
                if cat in data and data[cat]:
                    term = random.choice(data[cat])
                else:
                    term = cat  # fallback to category name if missing
                # Randomly apply case to each term
                case = random.choice(case_choices)
                if case == 'upper':
                    term = term.upper()
                elif case == 'lower':
                    term = term.lower()
                # else as is
                query.append(term)
            q = " ".join(query)
            selected.append(q)
        except Exception:
            continue  # skip if any error
    print(f"[text_sampler] Generated {len(selected)} queries (requested {n_samples})")
    return selected
