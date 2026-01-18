# Student CLAP: Lightweight Audio Encoder

This is a standalone project that try to do a distillation process of LAION CLAP:
- https://github.com/LAION-AI/CLAP

by using as teacher the pretrained model: 
- music_audioset_epoch_15_esc_90.14.pt

and following the tinyCLAP distillation approch for the AUDIO part:
- https://github.com/fpaissan/tinyCLAP

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
You can check how the avarage cosine similarity is going for each epoch with this one line command:
```
for f in student_clap/checkpoints/checkpoint_epoch_*.pth; do echo -n "$f: "; python3 -c "import torch; m=torch.load('$f', map_location='cpu')['train_metrics']; print(f\"cosine={m['avg_cosine_sim']}, lr={m['learning_rate']}\")"; done
```

You can check the million of parameter used for your input configuration with this command:
```
PYTHONPATH=.. python -c "import yaml; from student_clap.models.student_onnx_model import StudentCLAPAudio; config=yaml.safe_load(open('config.yaml')); m=StudentCLAPAudio(config); print(m.count_parameters())"
```

To check instead which configuration of input you used for a checkpoint you can use this command:
```
PYTHONPATH=.. python -c "import torch; m=torch.load('student_clap/checkpoints/CHECKPOINT-NAME-HERE.pth', map_location='cpu'); print({k: v for k, v in m['config']['model'].items() if k.startswith('phinet_')})"
```

## Training

## Training - Songs

**Architecture:**
- Custom PhiNet (micromind.PhiNet + BatchNorm2d + Conv2d projection)
- Parameters: alpha, beta, t0, N (configurable in config.yaml)
- n_mels=128, embedding_dim=512 (configurable)
- Projection: Residual (embed1+embed2), bias=False, dropout configurable
- Model size: 2-3M params (depends on config)

**Strategy:**
- Stage 1: Distillation from CLAP teacher (epochs 1-100), train all layers
- Stage 2: Encoder frozen, finetune projection head only (epochs 101-110)
- Learning rate warmup (epoch 1: LR linearly increases from 0 to target)
- LR scheduler: ReduceLROnPlateau (monitors val loss, reduces LR on plateau)
- Spectrogram augmentation (random gain, all epochs)

**Segmentation:** 10-sec segments, 50% overlap, process 10 segments/batch → train on "both" (individuals + averaged)

**Loss:** Negative cosine similarity

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
The distillation process utilized a curated dataset of 2000+ songs:

- [Free Music Archive](https://freemusicarchive.org/): Songs used are under various Creative Commons licenses. Detailed attribution and specific license types for each track can be found in [student_clap/models/FMA_SONGS_LICENSE.md](student_clap/models/FMA_SONGS_LICENSE.md)

- Public Domain: Additional tracks were sourced from CC0 1.0 sources.