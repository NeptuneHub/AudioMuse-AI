# Student CLAP: Lightweight Audio Encoder

This is a standalone project that try to do a distillation process of LAION CLAP:
- https://github.com/LAION-AI/CLAP

by using as teacher the pretrained model: 
- music_audioset_epoch_15_esc_90.14.pt

and following the tinyCLAP distillation approch for the AUDIO part:
- https://github.com/fpaissan/tinyCLAP

but using the efficentat model (mn10_as):
https://github.com/fschmid56/EfficientAT


It also try to distill a text model, in `config.yaml` you can decide if train audio, text or both:

```yaml
distillation:
	audio_enabled: true   # Enable/disable audio (songs) distillation
	text_enabled: true    # Enable/disable text distillation
```

## Quick Start

With this command you will create the virtual env with all the dependencies and start the training:

```bash
# Setup and install
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run training
python3 train_real.py --config config.yaml
```

## Useful command

You can check how the average cosine similarity (training and validation) is going for each epoch with this one line command:
```
python3 - <<'PY'
import glob, torch
for f in sorted(glob.glob('student_clap/checkpoints/checkpoint_epoch_*.pth')):
    ckpt = torch.load(f, map_location='cpu')
    m = ckpt.get('train_metrics', {})
    val_mse = ckpt.get('val_mse', ckpt.get('last_val_mse', ckpt.get('best_val_mse','N/A')))
    val_cos = ckpt.get('val_cosine_sim', ckpt.get('last_val_cosine', ckpt.get('best_val_cosine','N/A')))
    print(f"{f}: train_cos={m.get('avg_cosine_sim')}, train_mse={m.get('avg_mse')}, val_mse={val_mse}, val_cos={val_cos}, lr={m.get('learning_rate')}")
PY
```

You can check the million of parameter used for your input configuration with this command:
```
PYTHONPATH=.. python -c "import yaml; from student_clap.models.student_onnx_model import StudentCLAPAudio; config=yaml.safe_load(open('config.yaml')); m=StudentCLAPAudio(config); print(m.count_parameters())"
```

To check instead which configuration of input you used for a checkpoint you can use this command:
```
PYTHONPATH=.. python -c "import torch; m=torch.load('student_clap/checkpoints/CHECKPOINT-NAME-HERE.pth', map_location='cpu'); print({k: v for k, v in m['config']['model'].items() if k.startswith('efficientat_') or k=='efficientat_model'})"
```

To force the algorithm to read LR from config.yaml after a stop, instead of reading from the scheduler:
```
python3 - <<'PY'
import torch, glob
for p in glob.glob('student_clap/checkpoints/checkpoint_epoch_*.pth'):
    ckpt = torch.load(p, map_location='cpu')
    ckpt.pop('optimizer_state_dict', None)
    ckpt.pop('scheduler_state_dict', None)
    torch.save(ckpt, p)
    print("Stripped optimizer/scheduler from", p)
PY
```

To force the algorithm to read the weight decay value from config.yaml after a stop, instead of reading from schedule:
```
python3 - <<'PY'
import torch, glob, shutil
paths = glob.glob('student_clap/checkpoints/checkpoint_epoch_*.pth') + ['student_clap/checkpoints/latest.pth']
for p in paths:
    try:
        shutil.copy(p, p + '.bak')
        ckpt = torch.load(p, map_location='cpu')
        changed = False
        for k in ('scheduler_state_dict','optimizer_state_dict'):
            if ckpt.pop(k, None) is not None:
                changed = True
        if changed:
            torch.save(ckpt, p)
            print('Cleaned', p)
        else:
            print('No optimizer/scheduler state in', p)
    except Exception as e:
        print('Skipped', p, ':', e)
PY
```
Reset logit scale to initial value:
```
python3 -c "
import torch
ckpt = torch.load('student_clap/checkpoints/latest.pth', map_location='cpu')
ckpt['model_state_dict']['logit_scale'] = torch.tensor(2.6592)  # Reset to init
torch.save(ckpt, 'student_clap/checkpoints/latest.pth')
print('Reset logit_scale to 2.6592')
"
```

Check the cosine and val cosine also in subfolder:
```
find student_clap/checkpoints -name "checkpoint_epoch_*.pth" | sort -V | python3 -c '
import torch, sys
for line in sys.stdin:
    f = line.strip()
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        m = ckpt.get("train_metrics", {})
        avg = m.get("avg_cosine_sim", "null")
        lr = m.get("learning_rate", "null")
        val = ckpt.get("last_val_cosine", ckpt.get("val_cosine_sim", ckpt.get("best_val_cosine", "null")))
        print(f"{f}: cosine={avg}, val_cosine={val}, lr={lr}")
    except Exception as e:
        print(f"{f}: ERROR - {e}")
'
```

## Training

## Training - Songs

**Architecture:**
- EfficientAT MobileNet (Transformer-to-CNN distillation backbone)
- Pretrained variants available (e.g. `dymn10_as` / `mn10_as`) - default: `dymn10_as`
- n_mels=128, embedding_dim=512 (configurable)
- Projection: Residual (embed1+embed2), bias=False, dropout configurable
- Model size: depends on width multiplier (e.g. `mn10_as` ≈ 4.9M params)

**Strategy:**
- Stage 1: Distillation from CLAP teacher (epochs 1-100), train all layers
- Stage 2: Encoder frozen, finetune projection head only (epochs 101-110)
- Learning rate warmup (epoch 1: LR linearly increases from 0 to target)
- LR scheduler: ReduceLROnPlateau (monitors validation cosine similarity, reduces LR on plateau). Scheduler parameters can be controlled in `config.yaml` under `training.lr_scheduler` (e.g., `patience`, `factor`, `threshold`).
- Spectrogram augmentation (random gain, all epochs)

**Segmentation:** 10-sec segments, 50% overlap, process 10 segments/batch → train on "both" (individuals + averaged)

**Loss:** Negative cosine similarity

**Loss scaling options (configurable in `config.yaml` under `training`):**

- `loss_temperature` (float, default 1.0): static temperature applied to the cosine similarity (cosine / temperature).
- `use_logit_scale` (bool, default false): if true, a learnable `logit_scale` parameter (stored as `model.logit_scale`) is used and applied as `cosine * exp(logit_scale)`.
- `init_logit_scale` (float, default 1.0): initial value for the learnable logit_scale.
- `normalize_embeddings` (bool, default true): if true, the student and teacher embeddings are L2‑normalized before computing the MSE; set to `false` to compute MSE on raw (unnormalized) embeddings.
- `use_teacher_embedding_cache` (bool, default true): if true, teacher segment/avg embeddings are read from the mel cache when available. Set to `false` to force recomputation of teacher embeddings from the (augmented) mel at training time — note this keeps the mel spectrogram cache in use but disables caching of teacher embeddings, and ensures augmentations/mixup are applied identically to teacher and student inputs.

## Training - Text

**Architecture:**
- Lightweight Transformer encoder (configurable in config.yaml under `model_text`)
- Parameters: embedding_dim, hidden_dim, num_layers, nhead (all configurable)
- Embedding layer > TransformerEncoder > Linear projection > L2 normalization
- Model size: typically ~1M params (depends on config)

**Strategy:**
- Distillation from CLAP teacher text encoder using text queries and teacher embeddings
- Text queries are sampled from categories such as Genre/Style, Instrumentation/Vocal, and Emotion/Mood
- The queries are generated using the `sample_text_queries` utility, which samples from a JSON file of possible text prompts (see `paths.text_json` in config.yaml)
- For each batch, the student text model is trained to match the teacher's embedding for the sampled queries using negative cosine similarity loss
- Learning rate scheduler and optimizer as in the audio model

**Loss:** Negative cosine similarity


## License
### Code and Models
All source code in this repository and the resulting trained model weights are licensed under the AGPL-3.0 License like all the project.

### Training Data
The distillation process utilized a curated dataset of songs:

- [Free Music Archive](https://freemusicarchive.org/): Songs used are under various Creative Commons licenses. Detailed attribution and specific license types for each track can be found in [student_clap/models/FMA_SONGS_LICENSE.md](student_clap/models/FMA_SONGS_LICENSE.md)

- [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset): Songs used are under various Creative Commons licenses. Detailed attribution and specific license types for each track can be found in [student_clap/models/MTG_JAMENDO_SONGS_LICENSE.md](student_clap/models/MTG_JAMENDO_SONGS_LICENSE.md)

- [WIKIMEDIA](https://commons.wikimedia.org/wiki/): Songs used are under various Creative Commons licenses. Detailed attribution and specific license types for each track can be found in [student_clap/models/WIKIMEDIA_SONGS_LICENSE.md](student_clap/models/WIKIMEDIA_SONGS_LICENSE.md)

- [CCMIXTER](https://ccmixter.org/): Songs used are under various Creative Commons licenses. Detailed attribution and specific license types for each track can be found in [student_clap/models/CCMIXTER_SONGS_LICENSE.md](student_clap/models/CCMIXTER_SONGS_LICENSE.md)

- [Incompetech](https://ccmixter.org/): Songs used are under various Creative Commons licenses. Detailed attribution and specific license types for each track can be found in [student_clap/models/INCOMPETECH_SONGS_LICENSE.md](student_clap/models/INCOMPETECH_SONGS_LICENSE.md)

- [Musopen](https://musopen.org/): Songs used are under various Creative Commons licenses. Detailed attribution and specific license types for each track can be found in [student_clap/models/MUSOPEN_SONGS_LICENSE.md](student_clap/models/MUSOPEN_SONGS_LICENSE.md

- 
- Public Domain: Additional tracks were sourced from CC0 1.0 sources.