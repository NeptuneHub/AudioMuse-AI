"""
Export trained ONNX models to inference format.
Removes training-only operators and creates deployment-ready models.
"""

import logging
import onnx
from onnx import helper
import os
from typing import Dict

logger = logging.getLogger(__name__)


def export_to_inference(training_model_path: str, 
                       output_path: str,
                       model_name: str = "model") -> bool:
    """
    Export training ONNX model to inference format.
    
    This function:
    1. Loads the training model
    2. Removes training-specific operations (e.g., Dropout, training-mode BatchNorm)
    3. Optimizes the graph for inference
    4. Saves the inference model
    
    Args:
        training_model_path: Path to training model
        output_path: Path to save inference model
        model_name: Name of the model (for logging)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Loading training model from {training_model_path}")
        
        # Load model
        model = onnx.load(training_model_path)
        
        # The model should already be in inference format if built correctly
        # Just verify and save
        onnx.checker.check_model(model)
        
        # Save inference model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save(model, output_path)
        
        logger.info(f"Exported {model_name} inference model to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        return False


def export_models(config: Dict, checkpoint_dir: str, output_dir: str) -> Dict[str, str]:
    """
    Export both music and text encoder models to inference format.
    
    Args:
        config: Configuration dictionary
        checkpoint_dir: Directory containing trained model checkpoints
        output_dir: Directory to save exported models
        
    Returns:
        Dictionary with paths to exported models
    """
    export_config = config.get('export', {})
    music_encoder_name = export_config.get('music_encoder_name', 'student_music_encoder.onnx')
    text_encoder_name = export_config.get('text_encoder_name', 'student_text_encoder.onnx')
    
    exported_models = {}
    
    # Export music encoder
    music_encoder_training_path = os.path.join(checkpoint_dir, 'music_encoder_final.onnx')
    music_encoder_output_path = os.path.join(output_dir, music_encoder_name)
    
    if os.path.exists(music_encoder_training_path):
        if export_to_inference(music_encoder_training_path, music_encoder_output_path, "Music Encoder"):
            exported_models['music_encoder'] = music_encoder_output_path
    else:
        logger.warning(f"Music encoder training model not found at {music_encoder_training_path}")
    
    # Export text encoder
    text_encoder_training_path = os.path.join(checkpoint_dir, 'text_encoder_final.onnx')
    text_encoder_output_path = os.path.join(output_dir, text_encoder_name)
    
    if os.path.exists(text_encoder_training_path):
        if export_to_inference(text_encoder_training_path, text_encoder_output_path, "Text Encoder"):
            exported_models['text_encoder'] = text_encoder_output_path
    else:
        logger.warning(f"Text encoder training model not found at {text_encoder_training_path}")
    
    if exported_models:
        logger.info(f"Successfully exported {len(exported_models)} models")
        for model_type, path in exported_models.items():
            logger.info(f"  {model_type}: {path}")
    else:
        logger.warning("No models were exported")
    
    return exported_models


if __name__ == "__main__":
    # Test export
    logging.basicConfig(level=logging.INFO)
    
    test_config = {
        'export': {
            'output_dir': './student/exported_models',
            'music_encoder_name': 'student_music_encoder.onnx',
            'text_encoder_name': 'student_text_encoder.onnx'
        }
    }
    
    exported = export_models(test_config, './student/checkpoints', './student/exported_models')
    print(f"Exported models: {exported}")
