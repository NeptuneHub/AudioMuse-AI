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
* **Switching is non-destructive.** Each backend stores its embeddings
  and Voyager index under its own namespace (composite
  `(item_id, backend)` PK on `embedding`; namespaced Voyager
  `INDEX_NAME`), so flipping `SONIC_BACKEND` preserves the outgoing
  backend's data — you can switch back later without re-analyzing.
  When you no longer want the outgoing backend's data, drop it from
  the Admin → Cleaning → Sonic Backend Storage panel.

## Switching backends

1. Rebuild the image with MERT deps and set the backend env vars:

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

2. Restart the stack and run a full analysis (Admin → Analysis). Tracks
   that already have the previous backend's embedding stay tagged as
   "needs analysis under the active backend" until their new row is
   written. Voyager rebuild for the active backend happens
   automatically at the end of the run.

3. Once you're happy with the new backend, go to Admin → Cleaning →
   **Sonic Backend Storage**, find the previous backend in the table
   and click "Clear this backend" to free the disk space. The active
   backend is read-only there — switching `SONIC_BACKEND` is the only
   way to free the active backend's data.

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
