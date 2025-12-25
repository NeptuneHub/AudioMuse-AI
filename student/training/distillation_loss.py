"""
Distillation loss functions for knowledge transfer from teacher models.
Implements MSE loss between student and teacher embeddings.
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


def mse_loss(student_embeddings: np.ndarray, teacher_embeddings: np.ndarray) -> float:
    """
    Compute Mean Squared Error loss between student and teacher embeddings.
    
    Args:
        student_embeddings: Student model embeddings (batch_size, embedding_dim)
        teacher_embeddings: Teacher model embeddings (batch_size, embedding_dim)
        
    Returns:
        MSE loss value
    """
    if student_embeddings.shape != teacher_embeddings.shape:
        raise ValueError(f"Shape mismatch: student {student_embeddings.shape} vs teacher {teacher_embeddings.shape}")
    
    # MSE = mean((student - teacher)^2)
    diff = student_embeddings - teacher_embeddings
    mse = np.mean(diff ** 2)
    
    return float(mse)


def distillation_loss_musicnn(student_music_embeddings: np.ndarray,
                              musicnn_embeddings: np.ndarray,
                              projection_weight: np.ndarray = None) -> float:
    """
    Compute distillation loss from MusiCNN teacher (200-dim) to student (256-dim).
    
    Args:
        student_music_embeddings: Student embeddings (batch_size, 256)
        musicnn_embeddings: MusiCNN teacher embeddings (batch_size, 200)
        projection_weight: Optional projection matrix to align dimensions (256, 200)
                          If None, will project student down to 200-dim
        
    Returns:
        Distillation loss value
    """
    batch_size = student_music_embeddings.shape[0]
    
    # Project student embeddings to match MusiCNN dimension
    if projection_weight is None:
        # Simple linear projection: take first 200 dimensions
        student_projected = student_music_embeddings[:, :200]
    else:
        # Use learned projection
        student_projected = np.matmul(student_music_embeddings, projection_weight)
    
    # Normalize both embeddings
    student_norm = student_projected / (np.linalg.norm(student_projected, axis=1, keepdims=True) + 1e-8)
    teacher_norm = musicnn_embeddings / (np.linalg.norm(musicnn_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # MSE loss
    loss = mse_loss(student_norm, teacher_norm)
    
    return loss


def distillation_loss_clap(student_music_embeddings: np.ndarray,
                          clap_embeddings: np.ndarray,
                          projection_weight: np.ndarray = None) -> float:
    """
    Compute distillation loss from CLAP teacher (512-dim) to student (256-dim).
    
    Args:
        student_music_embeddings: Student embeddings (batch_size, 256)
        clap_embeddings: CLAP teacher embeddings (batch_size, 512)
        projection_weight: Optional projection matrix to align dimensions (256, 512)
                          If None, will project CLAP down to 256-dim
        
    Returns:
        Distillation loss value
    """
    batch_size = student_music_embeddings.shape[0]
    
    # Project to match dimensions
    if projection_weight is None:
        # Simple projection: take first 256 dimensions of CLAP
        clap_projected = clap_embeddings[:, :256]
    else:
        # Project student up to CLAP dimension
        student_projected = np.matmul(student_music_embeddings, projection_weight)
        clap_projected = clap_embeddings
        student_music_embeddings = student_projected
    
    # Normalize both embeddings
    student_norm = student_music_embeddings / (np.linalg.norm(student_music_embeddings, axis=1, keepdims=True) + 1e-8)
    teacher_norm = clap_projected / (np.linalg.norm(clap_projected, axis=1, keepdims=True) + 1e-8)
    
    # MSE loss
    loss = mse_loss(student_norm, teacher_norm)
    
    return loss


def combined_distillation_loss(student_music_embeddings: np.ndarray,
                               musicnn_embeddings: np.ndarray,
                               clap_embeddings: np.ndarray,
                               weights: Dict[str, float]) -> Dict[str, float]:
    """
    Compute combined distillation loss from both teachers.
    
    Args:
        student_music_embeddings: Student embeddings (batch_size, 256)
        musicnn_embeddings: MusiCNN embeddings (batch_size, 200)
        clap_embeddings: CLAP embeddings (batch_size, 512)
        weights: Loss weights dictionary with keys:
                'musicnn_distillation', 'clap_distillation'
        
    Returns:
        Dictionary with individual and total losses
    """
    # Compute individual losses
    musicnn_loss = distillation_loss_musicnn(student_music_embeddings, musicnn_embeddings)
    clap_loss = distillation_loss_clap(student_music_embeddings, clap_embeddings)
    
    # Weighted combination
    total_loss = (
        weights.get('musicnn_distillation', 0.3) * musicnn_loss +
        weights.get('clap_distillation', 0.4) * clap_loss
    )
    
    return {
        'musicnn_loss': musicnn_loss,
        'clap_loss': clap_loss,
        'distillation_loss': total_loss
    }
