# Student CLAP: Lightweight Audio Encoder

This is a standalone project that try to do a distillation process of LAION CLAP:
- https://github.com/LAION-AI/CLAP

by using as teacher the pretrained model: 
- music_audioset_epoch_15_esc_90.14.pt

and following the tinyCLAP distillation approch:
- https://github.com/fpaissan/tinyCLAP

## Quick Start

With this command you will create the virtual env with all the dependencies and start the training:

```bash
# Setup and install
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run training
python3 train_real.py --config config.yaml
```

You can check how the avarage cosine similarity is going for each epoch with this one line command:

```
for f in student_clap/checkpoints/checkpoint_epoch_*.pth; do echo -n "$f: "; python3 -c "import torch; m=torch.load('$f', map_location='cpu')['train_metrics']; print(f\"cosine={m['avg_cosine_sim']}, lr={m['learning_rate']}\")"; done
```

You can check the million of parameter used for your configuration with this command:
```
PYTHONPATH=.. python -c "import yaml; from student_clap.models.student_onnx_model import StudentCLAPAudio; config=yaml.safe_load(open('config.yaml')); m=StudentCLAPAudio(config); print(m.count_parameters())"
```

## Training

**Architecture:**
- Custom PhiNet (micromind.PhiNet + BatchNorm2d + Conv2d projection)
- Parameters: alpha=1.5, beta=0.75, t0=4, N=7
- n_mels=128, embedding_dim=512
- Projection: Residual (embed1+embed2), bias=False, dropout=0.5
- Model size: 2.37M params

**Strategy:** Distillation from CLAP teacher (100 epochs, LR=0.012, ReduceLROnPlateau)

**Segmentation:** 10-sec segments, 50% overlap, process 10 segments/batch → train on "both" (individuals + averaged)

**Batch:** 1 song × gradient_accumulation=8 (effective=8), grad_clip=1.0, weight_decay=0.01

**Loss:** Negative cosine similarity


## License

As dataset we used some songs from FMA:
- https://freemusicarchive.org/

All the songs are under Creative Commons license, more details can be found here:
- [student_clap/models/FMA_SONGS_2247_LICENSE.md](student_clap/models/FMA_SONGS_2247_LICENSE.md)