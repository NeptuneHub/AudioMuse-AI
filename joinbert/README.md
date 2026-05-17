# JointBERT for AudioMuse-AI — proof of concept

A tiny, deterministic NLU model that replaces the LLM "tool router" in
[app_chat.py](../app_chat.py). Given a free-text query, it predicts in **one
forward pass**:

1. **Which AudioMuse-AI tool(s) to fire** (multi-label, sigmoid head)
2. **Which spans of the query are which entity** (BIO-tagged token classification)

A deterministic Python dispatcher then maps those entities onto each tool's
input schema (genre, year_min/max, song_title+song_artist, add_items, ...) ready
for `tasks/mcp_tools.py:execute_mcp_tool()`.

The 7 tools covered are the 6 from `get_mcp_tools()`
(`song_similarity`, `text_search`, `artist_similarity`, `song_alchemy`,
`ai_brainstorm`, `search_database`) plus `lyrics_search` — the lyrics
free-text endpoint that lives in [app_lyrics.py](../app_lyrics.py) but is not
yet wired into the chat. The router exposes it as a first-class tool so the
dispatcher can route lyric-topic queries straight to
`/api/lyrics/search/text` instead of hoping the LLM picks the right thing.

---

## Files

| File                    | Role                                                            |
|-------------------------|-----------------------------------------------------------------|
| `main.py`               | Orchestrator — runs train, then inference. No logic.            |
| `generate_data.py`      | Templated generator. Vocab + intent templates → training_data.json |
| `training_data.json`    | ~5000 inline-annotated queries (Rasa-style `[value](type)`)     |
| `validation_data.json`  | 81 held-out queries: SNIPS-adapted (real human) + hand-written  |
| `train.py`              | PyTorch fine-tune of DistilBERT + intent/slot heads → ONNX      |
| `inference.py`          | Loads `joint_bert.onnx`, runs 5 demo queries per tool           |
| `requirements.txt`      | Pip dependencies (pinned to match the main project exactly)     |
| `labels.json`           | (Created by train.py) intent + slot vocab + tokenizer name      |
| `joint_bert.pt`         | (Created by train.py) PyTorch weights                           |
| `joint_bert.onnx`       | (Created by train.py) ONNX export (~250 MB)                     |

---

## Quick start (macOS)

```bash
# 1. Move into the folder
cd joinbert

# 2. Create + activate a Python 3.11 venv (use python3.10 / 3.11 / 3.12)
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip then install deps
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the whole pipeline (train → ONNX export → inference demo)
python main.py
```

That's it. On an Apple Silicon Mac mini, training takes **~5–15 minutes**
(MPS backend is detected automatically). Inference is ~10–30 ms per query
on CPU.

If you want to run steps individually:

```bash
python generate_data.py --n 5200   # regenerate training_data.json (optional)
python train.py                    # writes labels.json, joint_bert.pt, joint_bert.onnx
python inference.py                # uses the artifacts written above
```

`training_data.json` is committed pre-generated, so a fresh checkout can skip
straight to `train.py`. Re-run `generate_data.py` after changing vocab,
templates, or the intent share in that file. The `--seed` flag makes the
generator deterministic.

---

## What you'll see

`train.py` prints per-epoch metrics on both the training set and the
held-out validation set:

```
[train] device = mps
[train] loaded 5075 training examples from training_data.json
[train] loaded 81 validation examples from validation_data.json
[train] intents  (7): ['ai_brainstorm', 'artist_similarity', 'lyrics_search', ...]
[train] slots    (37): ['O', 'B-add_artist', 'I-add_artist', ...]
[train] training for 12 epochs (batch=16, lr=5e-05)
     epoch | train_loss |   val_loss | val_intent_F1 | val_slot_acc
  -------- | ---------- | ---------- | ------------- | ------------
  01/12    |     1.8412 |     0.9203 |        0.6210 |       0.8514
  02/12    |     0.6203 |     0.3214 |        0.8541 |       0.9421
  ...
```

What to watch for:
- **`train_loss` and `val_loss` both fall** → model is learning to generalize ✅
- **`train_loss` falls but `val_loss` plateaus** → overfitting (memorizing the
  generator's templates/vocab) ⚠️
- **`train_loss` falls but `val_loss` rises** → severe overfit, lower `EPOCHS`
  in `train.py` ❌
- **`val_intent_F1`** = multi-label F1 across all (example, tool) cells. 1.0 =
  every tool decision correct on held-out queries.
- **`val_slot_acc`** = per-token accuracy on non-ignored positions. 1.0 =
  every BIO tag correct.

`inference.py` prints, for every demo query, the full intent-probability
vector, the fired tools, the decoded entities, and the final tool-call list
that would be sent to `execute_mcp_tool()`:

```
QUERY: "songs like Beyoncé but more chill"
  intent probs: ai_brainstorm=0.04  artist_similarity=0.91  search_database=0.08  song_alchemy=0.02  song_similarity=0.05  text_search=0.78
  -> FIRED:    artist_similarity (0.91), text_search (0.78)
  -> entities: artist="Beyoncé", description="chill"
  -> tool calls:
     • artist_similarity({"artist": "Beyoncé"})
     • text_search({"description": "chill"})
```

---

## How to extend it

- **Add a new tool**: add 2-3 templates + (if needed) a small vocab list in
  `generate_data.py`, write a `gen_<tool>()` helper, bump its share in
  `SHARE`, regenerate + retrain. ~50 lines total.
- **Add a new entity type**: just use it in `[value](new_type)` inside a
  template; the slot vocabulary is built automatically from the data and the
  dispatcher in `inference.py` can be updated to consume it.
- **Use a smaller / faster backbone**: change `MODEL_NAME` in `train.py` to
  `sentence-transformers/all-MiniLM-L6-v2` (~80 MB ONNX, ~3 ms inference).
- **Drop into AudioMuse-AI**: import `Router` + `dispatch` from
  `inference.py` and call before the LLM loop in `app_chat.py`. Fall back to
  the LLM only when `max(intent_probs) < threshold`.

---

## Validation set provenance

`validation_data.json` (81 queries, held out from training) is designed
specifically to detect memorization. Entries fall into three groups:

- **SNIPS-adapted (~25 entries)**: real crowdsourced human queries from the
  [Snips/Sonos NLU benchmark](https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines/PlayMusic)
  (Apache-2.0), mapped onto our intent + entity schema (SNIPS `track` → our
  `song`; SNIPS `sort=best/top` → our `rating`; SNIPS `service` and
  `music_item` dropped — we don't expose those tools).
- **Hand-written with out-of-vocab values (~55 entries)**: artists, songs,
  lyric topics, and descriptions that **do not appear** in
  `generate_data.py`'s vocabulary lists. If the model still scores well here,
  it has learned the *structural patterns* of each intent rather than
  memorizing the generator's vocab.
- **Casual phrasings**: lowercase, "plz", abbreviations, sentence fragments
  the templates never produce.

Related academic context: Jin Ha Lee's analysis of real-world music queries
(*JASIST 2010*) and recent work like [The Language of Sound Search](https://arxiv.org/abs/2410.08324)
both confirm that real human music queries have wider phrasing diversity
than any template grammar — which is exactly why a held-out, hand-crafted
val set is more honest than a random split of templated data.

## Notes & caveats

- The 5k examples are synthetic (template-driven). That's enough to learn the
  routing and slot schema reliably for the demo. For production, sample real
  user queries from logs and append them to `training_data.json` (the file
  format is just a list of `{text, intents}` dicts — generator-produced and
  hand-written entries mix freely).
- ONNX file size is ~250 MB with DistilBERT. Switch to MiniLM-L6 in
  `train.py` for ~80 MB and 3× faster inference with marginal accuracy loss
  on this task.
- The dispatcher in `inference.py` is intentionally explicit and
  deterministic — no magic. Edit it to suit any tool-arg conventions
  specific to your environment.
