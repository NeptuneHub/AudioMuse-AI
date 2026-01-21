"""
Evaluation Metrics for Student CLAP Training

Validates student model performance against teacher embeddings.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def evaluate_embeddings(student_embeddings: np.ndarray,
                       teacher_embeddings: np.ndarray) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        student_embeddings: Student embeddings, shape (n_samples, 512)
        teacher_embeddings: Teacher embeddings, shape (n_samples, 512)
        
    Returns:
        Dict with evaluation metrics
    """
    if len(student_embeddings) != len(teacher_embeddings):
        raise ValueError(f"Size mismatch: {len(student_embeddings)} vs {len(teacher_embeddings)}")
    
    # Normalize embeddings
    student_norm = student_embeddings / (np.linalg.norm(student_embeddings, axis=1, keepdims=True) + 1e-8)
    teacher_norm = teacher_embeddings / (np.linalg.norm(teacher_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # 1. MSE
    mse = np.mean((student_norm - teacher_norm) ** 2)
    
    # 2. Cosine similarity
    cosine_sims = np.sum(student_norm * teacher_norm, axis=1)
    mean_cosine = np.mean(cosine_sims)
    std_cosine = np.std(cosine_sims)
    min_cosine = np.min(cosine_sims)
    max_cosine = np.max(cosine_sims)
    
    # 3. L2 distance
    l2_distances = np.linalg.norm(student_norm - teacher_norm, axis=1)
    mean_l2 = np.mean(l2_distances)
    std_l2 = np.std(l2_distances)
    
    # 4. Embedding norms
    student_norms = np.linalg.norm(student_embeddings, axis=1)
    teacher_norms = np.linalg.norm(teacher_embeddings, axis=1)
    
    metrics = {
        'mse': float(mse),
        'cosine_similarity': {
            'mean': float(mean_cosine),
            'std': float(std_cosine),
            'min': float(min_cosine),
            'max': float(max_cosine)
        },
        'l2_distance': {
            'mean': float(mean_l2),
            'std': float(std_l2)
        },
        'embedding_norms': {
            'student_mean': float(np.mean(student_norms)),
            'teacher_mean': float(np.mean(teacher_norms))
        },
        'num_samples': len(student_embeddings)
    }
    
    return metrics


def evaluate_retrieval(student_embeddings: np.ndarray,
                       teacher_embeddings: np.ndarray,
                       k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Evaluate retrieval accuracy using teacher as ground truth.
    
    For each student embedding, find nearest neighbors in student space
    and compare to nearest neighbors in teacher space.
    
    Args:
        student_embeddings: Student embeddings, shape (n_samples, 512)
        teacher_embeddings: Teacher embeddings, shape (n_samples, 512)
        k_values: List of k values for recall@k
        
    Returns:
        Dict with retrieval metrics
    """
    n_samples = len(student_embeddings)
    
    # Normalize
    student_norm = student_embeddings / (np.linalg.norm(student_embeddings, axis=1, keepdims=True) + 1e-8)
    teacher_norm = teacher_embeddings / (np.linalg.norm(teacher_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrices
    student_sim = cosine_similarity(student_norm)
    teacher_sim = cosine_similarity(teacher_norm)
    
    # Set diagonal to -inf (exclude self)
    np.fill_diagonal(student_sim, -np.inf)
    np.fill_diagonal(teacher_sim, -np.inf)
    
    # Compute recall@k for each k
    recall_at_k = {}
    
    for k in k_values:
        if k >= n_samples:
            logger.warning(f"k={k} >= n_samples={n_samples}, skipping")
            continue
        
        # Get top-k for each sample
        student_topk = np.argsort(-student_sim, axis=1)[:, :k]
        teacher_topk = np.argsort(-teacher_sim, axis=1)[:, :k]
        
        # Compute overlap
        overlaps = []
        for i in range(n_samples):
            student_set = set(student_topk[i])
            teacher_set = set(teacher_topk[i])
            overlap = len(student_set & teacher_set) / k
            overlaps.append(overlap)
        
        recall_at_k[f'recall@{k}'] = float(np.mean(overlaps))
    
    return recall_at_k


def print_evaluation_report(metrics: Dict, title: str = "Evaluation Results"):
    """
    Print formatted evaluation report (logged via logger.info so it appears in training logs).
    
    Args:
        metrics: Metrics dict from evaluate_embeddings()
        title: Report title
    """
    logger.info("\n" + "=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    
    logger.info(f"\nNumber of samples: {metrics['num_samples']}")
    
    logger.info(f"\nMSE Loss: {metrics['mse']:.6f}")
    
    logger.info(f"\nCosine Similarity:")
    logger.info(f"  Mean: {metrics['cosine_similarity']['mean']:.4f}")
    logger.info(f"  Std:  {metrics['cosine_similarity']['std']:.4f}")
    logger.info(f"  Min:  {metrics['cosine_similarity']['min']:.4f}")
    logger.info(f"  Max:  {metrics['cosine_similarity']['max']:.4f}")
    
    logger.info(f"\nL2 Distance:")
    logger.info(f"  Mean: {metrics['l2_distance']['mean']:.4f}")
    logger.info(f"  Std:  {metrics['l2_distance']['std']:.4f}")
    
    logger.info(f"\nEmbedding Norms:")
    logger.info(f"  Student: {metrics['embedding_norms']['student_mean']:.4f}")
    logger.info(f"  Teacher: {metrics['embedding_norms']['teacher_mean']:.4f}")
    
    # Performance assessment
    logger.info(f"\nPerformance Assessment:")
    cosine_mean = metrics['cosine_similarity']['mean']
    if cosine_mean > 0.9:
        logger.info(f"  ✓ Excellent (cosine > 0.9)")
    elif cosine_mean > 0.85:
        logger.info(f"  ✓ Good (cosine > 0.85)")
    elif cosine_mean > 0.8:
        logger.info(f"  ⚠ Acceptable (cosine > 0.8)")
    else:
        logger.info(f"  ✗ Poor (cosine < 0.8)")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    """Test evaluation metrics."""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Test evaluation metrics')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to test')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Noise level for student embeddings')
    args = parser.parse_args()
    
    print(f"Testing evaluation metrics with {args.num_samples} samples...")
    
    # Create synthetic embeddings
    n_samples = args.num_samples
    embedding_dim = 512
    
    # Teacher embeddings (normalized)
    teacher = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    teacher = teacher / np.linalg.norm(teacher, axis=1, keepdims=True)
    
    # Student embeddings (teacher + noise)
    noise_level = args.noise
    student = teacher + np.random.randn(n_samples, embedding_dim).astype(np.float32) * noise_level
    student = student / np.linalg.norm(student, axis=1, keepdims=True)
    
    print(f"  Noise level: {noise_level}")
    
    # Evaluate
    print("\n1. Computing embedding metrics...")
    metrics = evaluate_embeddings(student, teacher)
    print_evaluation_report(metrics, f"Evaluation (noise={noise_level})")
    
    # Test retrieval
    print("\n2. Computing retrieval metrics...")
    if n_samples >= 10:
        retrieval_metrics = evaluate_retrieval(student, teacher, k_values=[1, 5, 10])
        print("\nRetrieval Metrics:")
        for key, value in retrieval_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Test with perfect match
    print("\n3. Testing perfect match...")
    perfect_metrics = evaluate_embeddings(teacher, teacher)
    print_evaluation_report(perfect_metrics, "Perfect Match Test")
    
    # Test with high noise
    print("\n4. Testing high noise (0.5)...")
    high_noise_student = teacher + np.random.randn(n_samples, embedding_dim).astype(np.float32) * 0.5
    high_noise_student = high_noise_student / np.linalg.norm(high_noise_student, axis=1, keepdims=True)
    high_noise_metrics = evaluate_embeddings(high_noise_student, teacher)
    print_evaluation_report(high_noise_metrics, "High Noise Test (0.5)")
    
    print("\n✓ Evaluation metrics test complete")
