"""
Contrastive loss (InfoNCE) for music-text matching.
Implements the contrastive learning objective for aligning music and text embeddings.
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


def infonce_loss(music_embeddings: np.ndarray,
                text_embeddings: np.ndarray,
                temperature: float = 0.07) -> Dict[str, float]:
    """
    Compute InfoNCE (Normalized Temperature-scaled Cross Entropy) loss.
    
    This is the contrastive loss used in CLIP and CLAP for music-text matching.
    For each music-text pair in the batch, the music should be most similar to its
    corresponding text compared to all other texts in the batch, and vice versa.
    
    Args:
        music_embeddings: Music embeddings (batch_size, embedding_dim)
        text_embeddings: Text embeddings (batch_size, embedding_dim)
        temperature: Temperature parameter for scaling similarities
        
    Returns:
        Dictionary with music-to-text, text-to-music, and total contrastive losses
    """
    batch_size = music_embeddings.shape[0]
    
    # Normalize embeddings
    music_norm = music_embeddings / (np.linalg.norm(music_embeddings, axis=1, keepdims=True) + 1e-8)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix (batch_size, batch_size)
    # similarity[i, j] = cosine_similarity(music[i], text[j])
    similarity_matrix = np.matmul(music_norm, text_norm.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = np.arange(batch_size)
    
    # Music-to-text loss (each music should match its text)
    # For each row i, we want similarity[i, i] to be highest
    music_to_text_logits = similarity_matrix  # (batch_size, batch_size)
    
    # Compute log-softmax along text dimension (axis=1)
    max_logits = np.max(music_to_text_logits, axis=1, keepdims=True)
    exp_logits = np.exp(music_to_text_logits - max_logits)
    log_softmax = music_to_text_logits - max_logits - np.log(np.sum(exp_logits, axis=1, keepdims=True))
    
    # Negative log likelihood for correct pairs
    music_to_text_loss = -np.mean(log_softmax[np.arange(batch_size), labels])
    
    # Text-to-music loss (each text should match its music)
    # For each column j, we want similarity[j, j] to be highest
    text_to_music_logits = similarity_matrix.T  # (batch_size, batch_size)
    
    # Compute log-softmax along music dimension (axis=1)
    max_logits = np.max(text_to_music_logits, axis=1, keepdims=True)
    exp_logits = np.exp(text_to_music_logits - max_logits)
    log_softmax = text_to_music_logits - max_logits - np.log(np.sum(exp_logits, axis=1, keepdims=True))
    
    # Negative log likelihood for correct pairs
    text_to_music_loss = -np.mean(log_softmax[np.arange(batch_size), labels])
    
    # Total contrastive loss (average of both directions)
    total_loss = (music_to_text_loss + text_to_music_loss) / 2.0
    
    return {
        'music_to_text_loss': float(music_to_text_loss),
        'text_to_music_loss': float(text_to_music_loss),
        'contrastive_loss': float(total_loss)
    }


def clip_loss(music_embeddings: np.ndarray,
             text_embeddings: np.ndarray,
             temperature: float = 0.07) -> float:
    """
    Compute CLIP-style contrastive loss (alias for InfoNCE).
    
    Args:
        music_embeddings: Music embeddings (batch_size, embedding_dim)
        text_embeddings: Text embeddings (batch_size, embedding_dim)
        temperature: Temperature parameter
        
    Returns:
        Contrastive loss value
    """
    result = infonce_loss(music_embeddings, text_embeddings, temperature)
    return result['contrastive_loss']


def compute_accuracy(music_embeddings: np.ndarray,
                    text_embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute retrieval accuracy metrics.
    
    For each music embedding, checks if its corresponding text is the most similar.
    For each text embedding, checks if its corresponding music is the most similar.
    
    Args:
        music_embeddings: Music embeddings (batch_size, embedding_dim)
        text_embeddings: Text embeddings (batch_size, embedding_dim)
        
    Returns:
        Dictionary with music-to-text and text-to-music accuracy
    """
    batch_size = music_embeddings.shape[0]
    
    # Normalize embeddings
    music_norm = music_embeddings / (np.linalg.norm(music_embeddings, axis=1, keepdims=True) + 1e-8)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarity_matrix = np.matmul(music_norm, text_norm.T)
    
    # Music-to-text: for each music, is the corresponding text the most similar?
    music_to_text_predictions = np.argmax(similarity_matrix, axis=1)
    music_to_text_accuracy = np.mean(music_to_text_predictions == np.arange(batch_size))
    
    # Text-to-music: for each text, is the corresponding music the most similar?
    text_to_music_predictions = np.argmax(similarity_matrix, axis=0)
    text_to_music_accuracy = np.mean(text_to_music_predictions == np.arange(batch_size))
    
    return {
        'music_to_text_acc': float(music_to_text_accuracy),
        'text_to_music_acc': float(text_to_music_accuracy),
        'mean_acc': float((music_to_text_accuracy + text_to_music_accuracy) / 2.0)
    }


def compute_recall_at_k(music_embeddings: np.ndarray,
                       text_embeddings: np.ndarray,
                       k_values: list = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute Recall@K metrics for music-text retrieval.
    
    Args:
        music_embeddings: Music embeddings (batch_size, embedding_dim)
        text_embeddings: Text embeddings (batch_size, embedding_dim)
        k_values: List of K values to compute recall for
        
    Returns:
        Dictionary with Recall@K metrics
    """
    batch_size = music_embeddings.shape[0]
    
    # Normalize embeddings
    music_norm = music_embeddings / (np.linalg.norm(music_embeddings, axis=1, keepdims=True) + 1e-8)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarity_matrix = np.matmul(music_norm, text_norm.T)
    
    results = {}
    
    for k in k_values:
        if k > batch_size:
            continue
        
        # Music-to-text Recall@K
        # For each music, check if corresponding text is in top-K
        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]
        music_to_text_recall = np.mean([i in top_k_indices[i] for i in range(batch_size)])
        
        # Text-to-music Recall@K
        top_k_indices = np.argsort(similarity_matrix.T, axis=1)[:, -k:]
        text_to_music_recall = np.mean([i in top_k_indices[i] for i in range(batch_size)])
        
        results[f'recall@{k}_m2t'] = float(music_to_text_recall)
        results[f'recall@{k}_t2m'] = float(text_to_music_recall)
        results[f'recall@{k}_mean'] = float((music_to_text_recall + text_to_music_recall) / 2.0)
    
    return results
