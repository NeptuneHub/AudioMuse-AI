import numpy as np
import importlib

import tasks.clap_analyzer as clap
import config


def test_get_label_text_embeddings_and_probability(monkeypatch):
    # Create two orthogonal basis vectors in CLAP dimension
    dim = config.CLAP_EMBEDDING_DIMENSION
    v1 = np.zeros(dim, dtype=np.float32)
    v2 = np.zeros(dim, dtype=np.float32)
    v1[0] = 1.0
    v2[1] = 1.0

    # Monkeypatch get_text_embeddings_batch to return [v1, v2]
    monkeypatch.setattr(clap, 'get_text_embeddings_batch', lambda labels: np.vstack([v1, v2]))

    labels = ['label1', 'label2']
    emb_mat = clap.get_label_text_embeddings(labels)
    assert emb_mat.shape == (2, dim)
    # normalized rows: norms == 1
    norms = np.linalg.norm(emb_mat, axis=1)
    assert np.allclose(norms, 1.0)

    # Audio vector identical to v1
    audio_vec = v1.copy()
    audio_vec = audio_vec / (np.linalg.norm(audio_vec) + 1e-12)

    sims = emb_mat @ audio_vec
    probs = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)

    # First label should match perfectly -> prob ~1.0, second label orthogonal -> prob ~0.5
    assert abs(probs[0] - 1.0) < 1e-6
    assert abs(probs[1] - 0.5) < 1e-6
