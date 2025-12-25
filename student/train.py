"""
Main training script for student models.
Orchestrates the entire training pipeline.
"""

import logging
import os
import sys
import yaml
import argparse
from pathlib import Path

# Add student directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.database_reader import DatabaseReader
from data.jellyfin_client import JellyfinClient
from data.clap_anchor_search import CLAPAnchorSearch
from data.text_generator import TextGenerator
from preprocessing.audio_processor import AudioProcessor
from preprocessing.feature_extractor import FeatureExtractor
from models.build_music_encoder import build_music_encoder
from models.build_text_encoder import build_text_encoder
from training.trainer import ONNXTrainer, create_train_val_split
from export.export_onnx import export_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('student/training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "student/config.yaml") -> dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    
    # Support environment variable overrides
    config_path = os.environ.get('STUDENT_CONFIG_PATH', config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if 'POSTGRES_HOST' in os.environ:
        config['database']['host'] = os.environ['POSTGRES_HOST']
    if 'POSTGRES_PORT' in os.environ:
        config['database']['port'] = int(os.environ['POSTGRES_PORT'])
    if 'POSTGRES_USER' in os.environ:
        config['database']['user'] = os.environ['POSTGRES_USER']
    if 'POSTGRES_PASSWORD' in os.environ:
        config['database']['password'] = os.environ['POSTGRES_PASSWORD']
    if 'POSTGRES_DB' in os.environ:
        config['database']['dbname'] = os.environ['POSTGRES_DB']
    
    if 'JELLYFIN_URL' in os.environ:
        config['jellyfin']['url'] = os.environ['JELLYFIN_URL']
    if 'JELLYFIN_USER_ID' in os.environ:
        config['jellyfin']['user_id'] = os.environ['JELLYFIN_USER_ID']
    if 'JELLYFIN_TOKEN' in os.environ:
        config['jellyfin']['token'] = os.environ['JELLYFIN_TOKEN']
    
    if 'OPENAI_API_KEY' in os.environ:
        config['openai']['api_key'] = os.environ['OPENAI_API_KEY']
    
    return config


def prepare_data(config: dict):
    """
    Prepare training data: load from database, download audio, generate text.
    
    Returns:
        Dictionary with prepared data
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("=" * 80)
    
    # Initialize components
    db_reader = DatabaseReader(config['database'])
    jellyfin_client = JellyfinClient(config['jellyfin'])
    clap_anchor_search = CLAPAnchorSearch(config['clap_anchor_search'])
    text_generator = TextGenerator(config['openai'])
    
    # Get songs with embeddings
    logger.info("Fetching songs with embeddings from database...")
    songs = db_reader.get_songs_with_embeddings()
    
    if not songs:
        logger.error("No songs found with both MusiCNN and CLAP embeddings!")
        return None
    
    logger.info(f"Found {len(songs)} songs with complete embeddings")
    
    # Get embeddings
    item_ids = [song['item_id'] for song in songs]
    logger.info("Fetching teacher embeddings...")
    musicnn_embeddings, clap_embeddings = db_reader.get_batch_embeddings(item_ids)
    
    # Set up CLAP anchor search
    clap_anchor_search.set_embeddings(clap_embeddings)
    
    # Create song metadata dictionary
    song_metadata = {song['item_id']: song for song in songs}
    
    # Generate text descriptions
    logger.info("Generating text descriptions...")
    text_cache_file = os.path.join(config['cache']['text_cache_dir'], 'descriptions.json')
    
    # Generate anchor contexts
    anchor_contexts = []
    for song in songs:
        item_id = song['item_id']
        clap_emb = clap_embeddings.get(item_id)
        
        if clap_emb is not None:
            similar_songs = clap_anchor_search.find_similar_songs(
                clap_emb, song_metadata, exclude_item_id=item_id
            )
            context = clap_anchor_search.extract_context_from_similar(similar_songs)
        else:
            context = {
                'common_moods': [],
                'mood_keywords': [],
                'tempo_range': (None, None),
                'energy_range': (None, None)
            }
        
        anchor_contexts.append(context)
    
    # Generate descriptions
    descriptions = text_generator.batch_generate(
        songs,
        anchor_contexts,
        num_descriptions=config['text_generation']['descriptions_per_song'],
        cache_file=text_cache_file
    )
    
    logger.info(f"Generated descriptions for {len(descriptions)} songs")
    
    # Download audio files (with caching)
    logger.info("Downloading audio files from Jellyfin...")
    audio_cache_dir = config['cache']['audio_cache_dir']
    
    audio_files = jellyfin_client.batch_download(
        item_ids,
        audio_cache_dir,
        force_download=False
    )
    
    logger.info(f"Downloaded {len(audio_files)} audio files")
    
    # Close database connection
    db_reader.close()
    
    return {
        'songs': songs,
        'musicnn_embeddings': musicnn_embeddings,
        'clap_embeddings': clap_embeddings,
        'descriptions': descriptions,
        'audio_files': audio_files
    }


def build_models(config: dict):
    """Build ONNX models."""
    logger.info("=" * 80)
    logger.info("STAGE 2: MODEL BUILDING")
    logger.info("=" * 80)
    
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build music encoder
    logger.info("Building music encoder...")
    music_encoder_path = os.path.join(checkpoint_dir, 'music_encoder_initial.onnx')
    music_encoder = build_music_encoder(config['model'], music_encoder_path)
    logger.info(f"Music encoder saved to {music_encoder_path}")
    
    # Build text encoder
    logger.info("Building text encoder...")
    text_encoder_path = os.path.join(checkpoint_dir, 'text_encoder_initial.onnx')
    text_encoder = build_text_encoder(config['model'], text_encoder_path)
    logger.info(f"Text encoder saved to {text_encoder_path}")
    
    return {
        'music_encoder': music_encoder,
        'text_encoder': text_encoder
    }


def train_models(config: dict, data: dict):
    """Train the student models."""
    logger.info("=" * 80)
    logger.info("STAGE 3: TRAINING")
    logger.info("=" * 80)
    
    # Note: This is a framework implementation
    # Full ONNX Runtime Training integration would happen here
    
    logger.info("Initializing trainer...")
    trainer = ONNXTrainer(config)
    
    # Prepare training data
    # In a full implementation, this would create batches with:
    # - Audio features (mel-spectrograms)
    # - Text inputs (tokenized descriptions)
    # - Teacher embeddings (MusiCNN and CLAP)
    
    logger.info("Preparing training dataset...")
    
    # Create simple training examples dictionary
    training_examples = []
    for song in data['songs']:
        item_id = song['item_id']
        
        if (item_id in data['musicnn_embeddings'] and 
            item_id in data['clap_embeddings'] and
            item_id in data['descriptions'] and
            item_id in data['audio_files']):
            
            training_examples.append({
                'item_id': item_id,
                'song': song,
                'audio_file': data['audio_files'][item_id],
                'descriptions': data['descriptions'][item_id],
                'musicnn_embedding': data['musicnn_embeddings'][item_id],
                'clap_embedding': data['clap_embeddings'][item_id]
            })
    
    logger.info(f"Created {len(training_examples)} training examples")
    
    # Split into train/val
    train_data, val_data = create_train_val_split(
        training_examples,
        train_ratio=config['training']['train_split']
    )
    
    # Train
    logger.info("Starting training loop...")
    logger.warning("NOTE: This is a framework implementation. Full ONNX Runtime Training "
                  "integration requires additional setup with training graphs and optimizers.")
    
    history = trainer.train(train_data, val_data)
    
    logger.info("Training completed")
    return history


def export_trained_models(config: dict):
    """Export trained models to inference format."""
    logger.info("=" * 80)
    logger.info("STAGE 4: MODEL EXPORT")
    logger.info("=" * 80)
    
    checkpoint_dir = config['training']['checkpoint_dir']
    output_dir = config['export']['output_dir']
    
    exported_models = export_models(config, checkpoint_dir, output_dir)
    
    if exported_models:
        logger.info("Successfully exported models:")
        for model_type, path in exported_models.items():
            logger.info(f"  {model_type}: {path}")
    
    return exported_models


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train student models for AudioMuse-AI')
    parser.add_argument('--config', type=str, default='student/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-data-prep', action='store_true',
                       help='Skip data preparation stage')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training stage')
    parser.add_argument('--export-only', action='store_true',
                       help='Only export models')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("STUDENT TRAINING SYSTEM - AudioMuse-AI")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(args.config)
    logger.info("Configuration loaded successfully")
    
    try:
        if args.export_only:
            # Only export existing models
            exported_models = export_trained_models(config)
            logger.info("Export completed")
            return
        
        # Stage 1: Data Preparation
        if not args.skip_data_prep:
            data = prepare_data(config)
            if data is None:
                logger.error("Data preparation failed")
                return
        else:
            logger.info("Skipping data preparation")
            data = None
        
        # Stage 2: Model Building
        models = build_models(config)
        
        # Stage 3: Training
        if not args.skip_training and data is not None:
            history = train_models(config, data)
        else:
            logger.info("Skipping training")
        
        # Stage 4: Export
        exported_models = export_trained_models(config)
        
        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
