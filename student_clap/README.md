# Student CLAP: Lightweight Audio Encoder

## Quick Start

```bash
# Setup and install
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run training
python3 train_real.py --config config.yaml
```

## Training

**Architecture:** PhiNet mobile-optimized:
- PhiNet: 5 PhiBlocks with inverted residuals and SE modules
- Channels: [16, 24, 32, 64, 96]
- Transformer: 2 layers for temporal modeling
- Projection: 384 -> 256 -> 512 dimensional embedding space
- Model size: 1.9M params, 7.4MB

**Strategy:** Two-stage distillation from CLAP teacher
- Stage 1 (15 epochs): Full model, lr=0.003
- Stage 2 (5 epochs): Projection only, lr=0.001

**Segmentation:** 10-sec segments, 50% overlap → train on individuals + averaged ("both" strategy)

**Batch:** 2 songs × gradient_accumulation=8 (effective=16), grad_clip=5.0, weight_decay=0.0

**Loss:** Negative cosine similarity

## License
This is a standalone project that try to do a distillation process of LAION CLAP:
- https://github.com/LAION-AI/CLAP

by using the pretrained model: 
- music_audioset_epoch_15_esc_90.14.pt

As dataset it use a subset of the MTG jamendo dataset:
- https://github.com/MTG/mtg-jamendo-dataset 

that comes from:
- https://www.jamendo.com/

And a dataset from:
- https://freemusicarchive.org/

All the songs are under Creative Commons license, more details can be found here:
- [student_clap/models/FMA_SONGS_2247_LICENSE.md](student_clap/models/FMA_SONGS_2247_LICENSE.md)
- [student_clap/models/JAMENDO_SONGS_LICENSE.md](student_clap/models/JAMENDO_SONGS_LICENSE.md)

The distillation approch follow the tinyCLAP approch:
- https://github.com/fpaissan/tinyCLAP