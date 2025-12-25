# Student Training System

A standalone knowledge distillation training system for creating lightweight music-text matching models. The student models learn from both MusiCNN and CLAP teacher embeddings to create unified, efficient embedding models suitable for deployment.

## Overview

This training system implements a teacher-student knowledge distillation framework with the following components:

### Architecture

#### Music Encoder (Student Model)
- **Input**: Raw audio at 48kHz → Mel-spectrogram (128 mel bands)
- **Teachers**: 
  - MusiCNN embeddings (200-dim, from `embedding` table)
  - CLAP audio embeddings (512-dim, from `clap_embedding` table)
- **Architecture**: 3-layer CNN → BatchNorm → ReLU → MaxPool → GlobalAvgPool → Dense(256) → L2 Norm
- **Output**: 256-dim unified music embedding
- **Target size**: ~5MB ONNX model

#### Text Encoder (Student Model)
- **Input**: Text descriptions (tokenized)
- **Teacher**: CLAP text embeddings
- **Architecture**: Embedding → Mean Pooling → Dense(256) → L2 Norm
- **Output**: 256-dim text embedding (matching music encoder dimensionality)
- **Target size**: ~8MB ONNX model

### Training Strategy

Multi-task learning with three loss components:

1. **MusiCNN Distillation Loss (30%)**: MSE between student music embedding and MusiCNN teacher
2. **CLAP Distillation Loss (40%)**: MSE between student music embedding and CLAP audio teacher
3. **Contrastive Loss (30%)**: InfoNCE loss for music-text matching

## Prerequisites

### 1. Database Access
- PostgreSQL database with AudioMuse-AI schema
- Required tables:
  - `score`: Song metadata (tempo, energy, key, scale, mood_vector, other_features)
  - `embedding`: MusiCNN embeddings (200-dim BYTEA)
  - `clap_embedding`: CLAP embeddings (512-dim BYTEA)

### 2. Jellyfin Media Server
- Access to Jellyfin instance with music library
- Valid user ID and API token
- Network access to download audio files

### 3. OpenAI API
- OpenAI API key for GPT-4 or GPT-3.5-turbo
- Used for generating diverse text descriptions

### 4. Hardware Requirements
- **Recommended**: GPU with CUDA support (8GB+ VRAM)
- **Minimum**: CPU with 16GB+ RAM
- **Storage**: ~100GB for audio cache + model checkpoints

## Installation

### 1. Install Dependencies

```bash
cd student/
pip install -r requirements.txt
```

### 2. Configure Environment

Create environment variables or edit `config.yaml`:

```bash
# Database
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_USER="audiomuse"
export POSTGRES_PASSWORD="your_password"
export POSTGRES_DB="audiomusedb"

# Jellyfin
export JELLYFIN_URL="http://your_jellyfin:8096"
export JELLYFIN_USER_ID="your_user_id"
export JELLYFIN_TOKEN="your_token"

# OpenAI
export OPENAI_API_KEY="your_openai_api_key"
```

### 3. Verify Setup

```bash
# Test database connection
python -c "from data.database_reader import DatabaseReader; import yaml; config = yaml.safe_load(open('config.yaml')); db = DatabaseReader(config['database']); print(f'Songs: {len(db.get_songs_with_embeddings())}')"

# Test Jellyfin connection
python -c "from data.jellyfin_client import JellyfinClient; import yaml; config = yaml.safe_load(open('config.yaml')); client = JellyfinClient(config['jellyfin']); print('Jellyfin OK')"
```

## Configuration

The `config.yaml` file contains all training parameters. Key sections:

### Database Configuration
```yaml
database:
  host: "localhost"
  port: 5432
  user: "audiomuse"
  password: "audiomusepassword"
  dbname: "audiomusedb"
```

### Model Architecture
```yaml
model:
  music_encoder:
    conv_filters: [32, 64, 128]
    kernel_sizes: [[3, 3], [3, 3], [3, 3]]
    pool_sizes: [[2, 2], [2, 2], [2, 2]]
    embedding_dim: 256
    dropout: 0.1
  
  text_encoder:
    vocab_size: 30000
    embedding_dim: 256
    max_seq_length: 128
```

### Training Parameters
```yaml
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001
  loss_weights:
    musicnn_distillation: 0.3
    clap_distillation: 0.4
    contrastive: 0.3
```

### Audio Processing
```yaml
audio:
  sample_rate: 48000  # CLAP's native rate
  duration: 30.0      # Seconds (0 for full song)
  n_mels: 128
  n_fft: 2048
  hop_length: 512
```

### Text Generation
```yaml
text_generation:
  descriptions_per_song: 5
openai:
  model: "gpt-4"
  temperature: 0.7
```

## Training

### Full Training Pipeline

Run the complete training pipeline:

```bash
cd student/
python train.py
```

This executes:
1. **Data Preparation**: Load songs, download audio, generate text descriptions
2. **Model Building**: Create ONNX model graphs
3. **Training**: Train student models with knowledge distillation
4. **Export**: Export inference-ready ONNX models

### Stage-by-Stage Training

Skip specific stages:

```bash
# Skip data preparation (if already done)
python train.py --skip-data-prep

# Skip training (only build models)
python train.py --skip-training

# Only export models
python train.py --export-only
```

### Custom Configuration

```bash
python train.py --config my_config.yaml
```

## Data Pipeline

### 1. Database Integration

```python
from data.database_reader import DatabaseReader

db_reader = DatabaseReader(config['database'])
songs = db_reader.get_songs_with_embeddings()
musicnn_emb, clap_emb = db_reader.get_batch_embeddings(item_ids)
```

### 2. Audio Download

```python
from data.jellyfin_client import JellyfinClient

client = JellyfinClient(config['jellyfin'])
audio_files = client.batch_download(item_ids, output_dir)
```

### 3. Text Generation

```python
from data.text_generator import TextGenerator

generator = TextGenerator(config['openai'])
descriptions = generator.batch_generate(
    songs_metadata,
    anchor_contexts,
    num_descriptions=5,
    cache_file='cache/descriptions.json'
)
```

### 4. Audio Processing

```python
from preprocessing.audio_processor import AudioProcessor
from preprocessing.feature_extractor import FeatureExtractor

audio_proc = AudioProcessor(config['audio'])
feature_ext = FeatureExtractor(config['audio'])

audio = audio_proc.process_audio_file(file_path)
mel_spec = feature_ext.extract_features(audio)
```

## Model Building

### Build Music Encoder

```python
from models.build_music_encoder import build_music_encoder

model = build_music_encoder(config['model'], output_path='music_encoder.onnx')
```

### Build Text Encoder

```python
from models.build_text_encoder import build_text_encoder

model = build_text_encoder(config['model'], output_path='text_encoder.onnx')
```

## Evaluation

### Training Metrics

The trainer logs:
- **Loss Components**: MusiCNN distillation, CLAP distillation, contrastive
- **Validation Loss**: Early stopping based on validation performance
- **Retrieval Accuracy**: Music-to-text and text-to-music matching

### Embedding Quality

Evaluate embedding similarity:

```python
from training.distillation_loss import mse_loss
from training.contrastive_loss import compute_accuracy, compute_recall_at_k

# Distillation quality
musicnn_loss = mse_loss(student_embeddings, teacher_embeddings)

# Retrieval quality
acc = compute_accuracy(music_embeddings, text_embeddings)
recall = compute_recall_at_k(music_embeddings, text_embeddings, k_values=[1, 5, 10])
```

## Inference

### Using Exported Models

The exported ONNX models can be used with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

# Load models
music_session = ort.InferenceSession('exported_models/student_music_encoder.onnx')
text_session = ort.InferenceSession('exported_models/student_text_encoder.onnx')

# Music inference
mel_spec = extract_mel_spectrogram(audio_file)  # Shape: (1, 1, 128, time)
music_emb = music_session.run(None, {'mel_spectrogram': mel_spec})[0]

# Text inference
text_ids = tokenize(text)  # Shape: (1, seq_len)
text_emb = text_session.run(None, {'input_ids': text_ids})[0]

# Compute similarity
similarity = np.dot(music_emb, text_emb.T)
```

### Integration with AudioMuse-AI

Replace existing CLAP models:

1. Copy exported models to AudioMuse-AI model directory:
   ```bash
   cp exported_models/student_music_encoder.onnx /app/model/
   cp exported_models/student_text_encoder.onnx /app/model/
   ```

2. Update AudioMuse-AI config to use student models

3. Student models provide faster inference with smaller memory footprint

## Troubleshooting

### Database Connection Issues

```bash
# Test PostgreSQL connection
psql -h localhost -U audiomuse -d audiomusedb -c "SELECT COUNT(*) FROM score;"

# Check embeddings
psql -h localhost -U audiomuse -d audiomusedb -c "SELECT COUNT(*) FROM embedding WHERE embedding IS NOT NULL;"
```

### Jellyfin Download Failures

- Verify token validity: Check token hasn't expired
- Check network connectivity: Ping Jellyfin server
- Verify user permissions: Ensure user has download rights
- Check disk space: Audio cache can be large

### OpenAI API Issues

- Rate limiting: Reduce batch size, add delays
- API key: Verify key is valid
- Model availability: Check GPT-4 access
- Cost management: Monitor API usage

### Memory Issues

- Reduce batch size in `config.yaml`
- Enable audio caching to avoid re-downloading
- Process songs in smaller chunks
- Use CPU if GPU memory is insufficient

### ONNX Runtime Issues

```bash
# Install CPU version if GPU issues
pip uninstall onnxruntime-training
pip install onnxruntime-training

# Check CUDA availability
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Directory Structure

```
student/
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── train.py                    # Main training script
├── README.md                   # This file
├── models/
│   ├── build_music_encoder.py # Music encoder builder
│   ├── build_text_encoder.py  # Text encoder builder
│   └── student_utils.py        # Model building utilities
├── data/
│   ├── database_reader.py     # PostgreSQL data loader
│   ├── jellyfin_client.py     # Jellyfin audio downloader
│   ├── text_generator.py      # OpenAI text generation
│   └── clap_anchor_search.py  # CLAP-based similar song search
├── training/
│   ├── distillation_loss.py   # Knowledge distillation losses
│   ├── contrastive_loss.py    # InfoNCE contrastive loss
│   └── trainer.py              # Training loop
├── preprocessing/
│   ├── audio_processor.py     # Audio loading (48kHz)
│   └── feature_extractor.py   # Mel-spectrogram extraction
├── export/
│   └── export_onnx.py          # Export to inference ONNX
├── cache/                      # Cached data (not in repo)
│   ├── audio/                  # Downloaded audio files
│   └── text/                   # Generated descriptions
├── checkpoints/                # Training checkpoints (not in repo)
└── exported_models/            # Final inference models (not in repo)
```

## Performance Expectations

### Training Time
- **Small dataset** (1K songs): ~2-4 hours on GPU
- **Medium dataset** (10K songs): ~1-2 days on GPU
- **Large dataset** (100K songs): ~1-2 weeks on GPU

### Model Performance
- **Target similarity**: Cosine similarity > 0.7 with teacher embeddings
- **Retrieval**: Recall@10 > 50% for music-text matching
- **Inference speed**: ~10ms per audio file (30s) on GPU
- **Model size**: Music encoder ~5MB, Text encoder ~8MB

## Notes

### ONNX Runtime Training

This implementation provides a complete framework for the training pipeline. Full ONNX Runtime Training integration requires:

1. **Training Graphs**: Convert inference graphs to training graphs with gradient computation
2. **Optimizers**: Implement Adam optimizer in ONNX
3. **Backward Pass**: Add gradient computation nodes
4. **Training API**: Use ONNX Runtime Training API for parameter updates

The current implementation focuses on:
- Complete data pipeline
- Model architecture definition
- Loss function implementation
- Training framework structure
- Export functionality

For production use, enhance with full ONNX Runtime Training integration or consider PyTorch training with ONNX export.

### Caching

- **Audio cache**: Persists across runs to avoid re-downloading
- **Text cache**: Saves OpenAI API calls and costs
- Clear caches manually if needed: `rm -rf cache/`

### Cost Considerations

- **OpenAI API**: ~$0.01-0.02 per song for 5 descriptions (GPT-3.5-turbo)
- **Storage**: ~50-100MB per hour of audio
- **GPU compute**: Recommended for training, but CPU works for small datasets

## License

This training system is part of AudioMuse-AI. See main project LICENSE.

## Support

For issues and questions:
1. Check troubleshooting section
2. Review training logs: `student/training.log`
3. Open issue on GitHub with logs and configuration
