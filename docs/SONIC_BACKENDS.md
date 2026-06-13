# Sonic Backends

AudioMuse-AI's per-track *similarity embedding* (the vector stored in
`embedding` and queried via the Voyager index for "find similar songs")
is produced by a swappable backend. Two are shipped:

| Backend  | Embedding dim | Source                                    | Tag head        | Default? |
| -------- | ------------- | ----------------------------------------- | --------------- | -------- |
| musicnn  | 200           | Legacy MusiCNN ONNX (Essentia lineage)    | MusiCNN-prediction | yes      |
| mert     | 768 / 1024    | `m-a-p/MERT-v1-95M` / `-330M` via HF      | MusiCNN-prediction | no       |

Selection is controlled by the `SONIC_BACKEND` environment variable.
`mood_vector` / mood-tagging output is currently produced by the same
MusiCNN-prediction ONNX in both backends, so `score.mood_vector` rows
stay schema-compatible across backend changes.

## When to switch

`mert` produces a richer, self-supervised embedding trained on millions
of hours of music. Open-MIR benchmarks consistently show it
out-performing CNN-on-mel-spectrogram models like MusiCNN on genre
tagging, mood, instrument and similarity tasks. If you find the default
"Instant Mix" / "similar tracks" recommendations underwhelming on
non-Western or instrumental material — exactly where MusiCNN is weakest
— switching to MERT is the highest-leverage change you can make.

The trade-offs:

* **Compute**. MERT-95M runs comfortably on a modern desktop CPU
  (~1–3 s/track on an i9-14900K). MERT-330M is meaningfully better but
  effectively requires a GPU.
* **Image size**. Pulling in `torch` adds ~700 MB.
* **Switching is destructive**. Embedding dimensions differ (200 vs
  768/1024), so the Voyager index and the `embedding` table must be
  rebuilt from scratch — exactly like the v2.0.0 lyrics-model swap.

## Switching backends

1. Drop the audio embedding tables (Voyager picks up the new
   dimensionality from `EMBEDDING_DIMENSION` automatically):

   ```bash
   docker compose exec -e PGPASSWORD=audiomusepassword postgres \
     psql -U audiomuse -d audiomusedb \
     -c "TRUNCATE embedding; DELETE FROM voyager_index_data;"
   ```

2. Rebuild the image with MERT deps and set the backend env vars:

   ```yaml
   # compose snippet
   audiomuse-flask:
     build:
       context: /path/to/AudioMuse-AI
       args:
         INSTALL_MERT: "1"
     environment:
       SONIC_BACKEND: "mert"
       MERT_MODEL_ID: "m-a-p/MERT-v1-95M"   # or -330M
       MERT_DEVICE: "auto"                  # cuda if available, else cpu
       MERT_LAYER: "-1"                     # last hidden layer; tune if desired
   ```

3. Restart the stack and run a full analysis (the Admin → Analysis
   page). Voyager rebuild happens automatically at the end of the run.

## Configuration reference

| Variable             | Default               | Notes                                             |
| -------------------- | --------------------- | ------------------------------------------------- |
| `SONIC_BACKEND`      | `musicnn`             | One of `musicnn`, `mert`                          |
| `MERT_MODEL_ID`      | `m-a-p/MERT-v1-95M`   | HF model id; `-330M` for the larger checkpoint    |
| `MERT_LAYER`         | `-1`                  | Hidden-layer index to mean-pool. `-1` = last      |
| `MERT_DEVICE`        | `auto`                | `auto` / `cpu` / `cuda`                           |
| `MERT_TARGET_SR`     | `24000`               | MERT's training sample rate                       |
| `MERT_HF_CACHE_DIR`  | *(HF default)*        | Override `$HF_HOME` for the MERT download         |

## Writing a new backend

Implement `tasks/sonic_backends/base.SonicBackend` and `register()` an
instance in your module. The minimum surface is:

```python
from tasks.sonic_backends import SonicBackend, register

class MyBackend(SonicBackend):
    name = "mybackend"
    embedding_dim = 1024
    target_sr = 16000

    def load_sessions(self): ...
    def cleanup_sessions(self, sessions, context=""): ...
    def analyze(self, audio, sr, sessions, *, file_basename, mood_labels):
        return track_embedding, {label: float_score, ...}

register(MyBackend())
```

Add the embedding dimension to `_BACKEND_EMBEDDING_DIMS` in `config.py`
so the Voyager builder picks up the correct size, then point your
deployment at `SONIC_BACKEND=mybackend`.
