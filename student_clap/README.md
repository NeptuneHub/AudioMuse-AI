# Student CLAP: Lightweight Audio Encoder

## Quick Start

```bash
# Setup and install
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run training
python3 train_real.py --config config.yaml
```

A standalone project for training a lightweight CLAP audio encoder using knowledge distillation from the existing large CLAP model (268MB) to create a compressed student model (20-40MB) with identical 512-dimensional embeddings.

## Overview

This project implements a **teacher-student knowledge distillation** framework to compress the CLAP audio encoder while maintaining:
- ✅ Same 512-dimensional embedding space
- ✅ Drop-in compatibility with existing CLAP text encoder
- ✅ 5-10x smaller model size (268MB → 20-40MB)
- ✅ 2-5x faster inference
- ✅ No retraining of text encoder needed

### Architecture Inspiration

Based on Microsoft Research's [TinyCLAP](https://github.com/microsoft/tinyclap) project, which demonstrates successful compression of CLAP models through:
- Efficient CNN/Transformer hybrid architecture
- Knowledge distillation from large CLAP models
- Specialized optimization for music (vs. general audio)

## Motivation

The current CLAP audio encoder (~268MB) is effective but computationally heavy for inference. This project creates a lightweight alternative that:
- Learns from existing averaged embeddings in the database (no teacher inference needed)
- Maintains exact segmentation strategy as production CLAP (10s segments, 5s hop)
- Preserves drop-in compatibility with existing text search functionality

## Key Architecture Principles

### Teacher Model (Frozen)
- **CLAP audio encoder** (~268MB) → 512-dim embeddings
- Already stored as averaged embeddings in `clap_embedding` table
- Used only as supervision signal (no inference during training)

### Student Model (To be trained)
- **Lightweight audio encoder** (~20-40MB ONNX model)
- **Output**: 512-dim L2-normalized embeddings
- **Input**: Raw audio at 48kHz (same as teacher)
- **Architecture**: TinyCLAP-inspired CNN/Transformer hybrid

### Text Encoder (Unchanged)
- Keep existing `clap_text_model.onnx` as-is
- Student maintains 512-dim compatibility
- No changes to text search functionality

## Critical: Segmentation Strategy

The student model **MUST** replicate the exact segmentation logic used by the teacher during production analysis. This is critical for embedding compatibility.

### Teacher Segmentation (from `tasks/clap_analyzer.py`)
```python
SAMPLE_RATE = 48000
SEGMENT_LENGTH = 480000  # 10 seconds at 48kHz
HOP_LENGTH = 240000      # 5 seconds (50% overlap)

# For songs > 10s: create overlapping segments
for start in range(0, total_length - SEGMENT_LENGTH + 1, HOP_LENGTH):
    segment = audio_data[start:start + SEGMENT_LENGTH]
    segments.append(segment)

# Add final segment if needed
last_start = len(segments) * HOP_LENGTH
if last_start < total_length:
    last_segment = audio_data[-SEGMENT_LENGTH:]
    segments.append(last_segment)

# For songs < 10s: pad to 10 seconds
if total_length <= SEGMENT_LENGTH:
    padded = np.pad(audio_data, (0, SEGMENT_LENGTH - total_length), mode='constant')

# Average all segment embeddings
avg_embedding = np.mean(segment_embeddings, axis=0)
```

### Student Training (Must Match)
1. Process audio with **same segmentation** (10s segments, 5s hop, 50% overlap)
2. Generate embeddings for **each segment independently**
3. **Average all segment embeddings** (same as teacher)
4. Learn to match the **averaged teacher embedding**

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database (existing AudioMuse-AI database)
- Jellyfin/Emby media server (existing setup)
- GPU with CUDA support recommended (10GB+ VRAM)
- 32GB+ RAM
- 50-100GB storage for audio cache

### Setup

1. Navigate to the `student_clap/` directory:
```bash
cd student_clap/
```

2. Create and activate a Python virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your actual credentials
nano .env  # or use your preferred editor
```

Edit your `.env` file with your actual values:
```bash
# Database Configuration
DB_HOST=your_postgres_host
DB_NAME=audiomuse
DB_USER=your_postgres_user
DB_PASSWORD=your_postgres_password

# Jellyfin Configuration
JELLYFIN_URL=http://your_jellyfin_url:8096
JELLYFIN_USER_ID=your_jellyfin_user_id
JELLYFIN_TOKEN=your_jellyfin_token
```

5. Verify configuration:
```bash
# Test database connection
python3 -c "import psycopg2; print('Database connection OK')"

# Test Jellyfin connection (optional)
curl -H "X-Emby-Token: YOUR_TOKEN" http://your_jellyfin_url:8096/System/Info
```

## Configuration

The configuration uses a combination of `config.yaml` and environment variables loaded from a `.env` file.

### Environment Variables (.env file)

The `.env` file contains your sensitive credentials and connection details:

```bash
# Database Configuration
DB_HOST=your_postgres_host
DB_NAME=audiomuse
DB_USER=your_postgres_user  
DB_PASSWORD=your_postgres_password

# Jellyfin Configuration
JELLYFIN_URL=http://your_jellyfin_url:8096
JELLYFIN_USER_ID=your_jellyfin_user_id
JELLYFIN_TOKEN=your_jellyfin_token
```

### Configuration File (config.yaml)

The `config.yaml` file references these environment variables and contains model/training settings:

### Database Connection
```yaml
database:
  host: ${DB_HOST}
  port: 5432
  database: ${DB_NAME}
  user: ${DB_USER}
  password: ${DB_PASSWORD}
```

### Jellyfin Connection
```yaml
jellyfin:
  url: ${JELLYFIN_URL}
  user_id: ${JELLYFIN_USER_ID}
  token: ${JELLYFIN_TOKEN}
```

### Audio Processing (Match CLAP)
```yaml
audio:
  sample_rate: 48000
  segment_length: 480000      # 10 seconds
  hop_length: 240000          # 5 seconds (50% overlap)
  n_mels: 128                 # For student model
  n_fft: 1024
  hop_length_stft: 480
  fmin: 0
  fmax: 14000
```

### Model Architecture
```yaml
model:
  architecture: "tinyclap"    # TinyCLAP-inspired
  embedding_dim: 512          # MUST be 512 for compatibility
  cnn_channels: [32, 64, 128]
  transformer_layers: 2
  attention_heads: 4
  hidden_dim: 256
```

### Training Parameters
```yaml
training:
  batch_size: 16
  learning_rate: 1e-4
  epochs: 100
  warmup_epochs: 5
  optimizer: "adam"
  weight_decay: 1e-5
  grad_clip: 1.0
  
  # Loss weights
  mse_weight: 0.6
  cosine_weight: 0.4
  
  # Checkpointing
  save_every: 5
  early_stopping_patience: 10
```

## Data Preparation

### 1. Download and Cache Audio

First, download all audio files from Jellyfin to avoid repeated downloads during training:

```bash
bash scripts/prepare_dataset.sh
```

This script will:
- Connect to your PostgreSQL database
- Fetch all songs with CLAP embeddings
- Download audio files from Jellyfin
- Cache them locally in `./cache/audio/`
- Display dataset statistics

### 2. Verify Dataset

Check that audio files and embeddings are properly loaded:

```bash
python3 data/database_loader.py --verify
```

## Training

### Start Training

Ensure your virtual environment is activated:

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

Run the main training script:

```bash
python3 train.py --config config.yaml
```

Or with custom config:

```bash
python3 train.py --config config.local.yaml
```

### Training Process

The training loop:
1. **Loads teacher embeddings** from `clap_embedding` table (512-dim, already averaged)
2. **Downloads audio** from Jellyfin (with caching)
3. **Segments audio** (10s windows, 5s hop, matching teacher)
4. **Processes each segment** through student model → 512-dim embeddings
5. **Averages student segment embeddings** (matching teacher's averaging)
6. **Computes losses** between averaged embeddings
7. **Backpropagates** using ONNX training API

### Monitor Training

Training logs are saved to `./logs/`. Monitor progress:

```bash
tail -f logs/training.log
```

Checkpoints are saved to `./checkpoints/` every 5 epochs (configurable).

### Resume Training

To resume from a checkpoint:

```bash
python3 train.py --config config.yaml --resume checkpoints/epoch_50.onnx
```

## Evaluation

### Validate Student Model

Evaluate the trained student model against teacher embeddings:

```bash
bash scripts/evaluate_model.sh checkpoints/epoch_100.onnx
```

This will compute:
- **MSE** between student and teacher embeddings
- **Cosine similarity** (target: > 0.85)
- **L2 distance** (lower is better)
- **Embedding distribution** comparison

### Metrics

Expected performance:
- Cosine similarity: > 0.85
- Model size: 20-40MB (vs. 268MB teacher)
- Inference speed: 2-5x faster
- Retrieval accuracy: Within 5% of teacher

## Export for Production

### Export Inference Model

Convert the trained model to an optimized inference ONNX model:

```bash
python3 export/export_inference.py \
    --checkpoint checkpoints/epoch_100.onnx \
    --output models/student_clap_audio.onnx
```

This will:
1. Remove training-only operators
2. Optimize for inference (constant folding, operator fusion)
3. Validate output format (512-dim embeddings)
4. Save as `student_clap_audio.onnx`

### Verify Inference Model

Test the exported model:

```bash
python3 export/export_inference.py \
    --checkpoint models/student_clap_audio.onnx \
    --test \
    --test-audio /path/to/test.mp3
```

## Deployment

### Drop-in Replacement

To use the student model in production:

1. **Backup existing model**:
```bash
cd /path/to/AudioMuse-AI
cp models/clap_audio_model.onnx models/clap_audio_model.onnx.backup
```

2. **Replace with student model**:
```bash
cp student_clap/models/student_clap_audio.onnx models/clap_audio_model.onnx
```

3. **Restart AudioMuse-AI services**:
```bash
# Restart worker containers that use CLAP audio model
docker-compose restart worker
```

4. **Verify compatibility**:
- Text search should work identically
- Embeddings remain 512-dimensional
- Performance should improve (faster inference, less memory)

### Rollback

If issues occur, restore the original model:

```bash
cd /path/to/AudioMuse-AI
cp models/clap_audio_model.onnx.backup models/clap_audio_model.onnx
docker-compose restart worker
```

## Project Structure

```
student_clap/
├── README.md                        # This file
├── requirements.txt                 # Training dependencies
├── config.yaml                      # Configuration template
├── .env.example                     # Environment variables template
├── .env                             # Your actual environment variables (git-ignored)
├── .gitignore                       # Git ignore file
├── train.py                         # Main training entry point
├── venv/                            # Python virtual environment (git-ignored)
│
├── models/
│   ├── tinyclap_audio.py           # Lightweight audio encoder architecture
│   ├── build_student_onnx.py       # Build student ONNX graph
│   └── model_utils.py               # Shared utilities
│
├── data/
│   ├── database_loader.py           # Load CLAP embeddings from PostgreSQL
│   ├── jellyfin_downloader.py       # Download audio with caching
│   └── dataset.py                   # Training dataset with segmentation
│
├── preprocessing/
│   ├── audio_segmentation.py        # 10s/5s segmentation logic
│   ├── mel_spectrogram.py           # CLAP-compatible mel-spec
│   └── augmentation.py              # Optional: pitch/time augmentation
│
├── training/
│   ├── trainer.py                   # ONNX training loop
│   ├── losses.py                    # MSE + cosine similarity losses
│   └── evaluation.py                # Validation metrics
│
├── export/
│   └── export_inference.py          # Export to inference ONNX
│
└── scripts/
    ├── prepare_dataset.sh            # Download and cache all audio
    └── evaluate_model.sh             # Test student vs teacher
```

## Technical Details

### TinyCLAP-Inspired Architecture

**Input**: Mel-spectrogram (128 mel bands, variable time frames) from 10s audio
- Sample rate: 48kHz
- Segment length: 480,000 samples (10 seconds)
- Mel bands: 128
- n_fft: 1024, hop_length: 480

**Architecture**:
1. **Efficient CNN Stem**:
   - Conv2D (32 filters, 3×3, stride=2) → BatchNorm → GELU → MaxPool2D
   - Conv2D (64 filters, 3×3, stride=2) → BatchNorm → GELU → MaxPool2D
   - Conv2D (128 filters, 3×3, stride=2) → BatchNorm → GELU

2. **Lightweight Transformer** (2 layers):
   - Flatten spatial dimensions to sequence
   - 2-layer Transformer encoder (4 attention heads, 256 hidden dim)
   - Multi-head self-attention

3. **Projection Head**:
   - Global average pooling over sequence
   - Dense (512) → BatchNorm → GELU
   - Dense (512) → L2 Normalization

**Output**: 512-dim L2-normalized embedding

### Loss Functions

Training uses a weighted combination of:
1. **MSE Loss** (60%): Minimizes Euclidean distance between embeddings
2. **Cosine Similarity Loss** (40%): Ensures directional alignment

```python
mse_loss = MSE(student_embedding, teacher_embedding)
cosine_loss = 1 - CosineSimilarity(student_embedding, teacher_embedding)
total_loss = 0.6 * mse_loss + 0.4 * cosine_loss
```

### ONNX Training

This project uses PyTorch for training with ONNX export for inference:
- Models built and trained using PyTorch
- Export to ONNX format for production deployment
- Compatible with existing ONNX inference pipelines

## Troubleshooting

### Issue: Database connection fails

**Solution**: Verify your database credentials in `config.yaml`:
```bash
psql -h your_host -U your_user -d your_database
```

### Issue: Jellyfin download fails

**Solution**: Check Jellyfin connectivity:
```bash
curl -H "X-Emby-Token: YOUR_TOKEN" http://your_jellyfin_url:8096/System/Info
```

### Issue: Out of memory during training

**Solution**: Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Issue: CUDA out of memory

**Solution**: 
1. Reduce batch size
2. Use CPU-only training (slower but works):
```bash
CUDA_VISIBLE_DEVICES=-1 python3 train.py --config config.yaml
```

### Issue: Student embeddings don't match teacher

**Solution**: 
1. Verify segmentation logic matches teacher exactly
2. Check mel-spectrogram parameters match CLAP preprocessing
3. Ensure embeddings are L2-normalized
4. Increase training epochs or adjust loss weights

### Issue: Audio cache fills disk

**Solution**: Clear cache periodically:
```bash
rm -rf cache/audio/*
```

## References

- **TinyCLAP**: https://github.com/microsoft/tinyclap
  - Microsoft Research project for CLAP compression
  - Paper: "TinyCLAP: Distilling Contrastive Language-Audio Pretraining"

- **CLAP**: https://github.com/LAION-AI/CLAP
  - Original CLAP implementation
  - Paper: "Large-scale Contrastive Language-Audio Pretraining"

- **Knowledge Distillation**: "Distilling the Knowledge in a Neural Network" (Hinton et al.)

- **AudioMuse-AI CLAP Implementation**: `tasks/clap_analyzer.py`
  - Production segmentation logic (lines 584-630)
  - Mel-spectrogram computation (lines 512-559)

## Notes

- This is a **compression project**, not new functionality
- Focuses on **audio encoder only** (text encoder unchanged)
- Uses **existing teacher embeddings** (no teacher inference needed)
- **Music-specialized**: Optimized for music domain (vs. general audio)
- **Segmentation critical**: Must match teacher's 10s/5s overlap exactly
- Consider **data augmentation** for better generalization

## License

This project follows the same license as AudioMuse-AI (see main repository).

## Contributing

Contributions are welcome! Please follow the AudioMuse-AI contribution guidelines.

## Support

For issues or questions:
1. Check this README's Troubleshooting section
2. Review training logs in `logs/`
3. Open an issue in the main AudioMuse-AI repository
