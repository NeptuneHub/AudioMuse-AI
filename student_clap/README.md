# Student CLAP: Lightweight Audio Encoder

## Quick Start

```bash
# Setup and install
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run training
python3 train_real.py --config config.yaml
```

A standalone project for training a lightweight CLAP audio encoder using knowledge distillation from the existing large CLAP music model (268MB) to create a compressed student model (20-40MB) with identical 512-dimensional embeddings.