import numpy as np
from student_clap.data.clap_embedder import CLAPEmbedder


class DummySession:
    def run(self, *args, **kwargs):
        # Return a fake embedding of shape (1, 512)
        return [np.zeros((1, 512), dtype=np.float32)]


def test_compute_embeddings_from_mel_resamples():
    # Create a dummy CLAPEmbedder without initializing ONNX session
    emb = CLAPEmbedder.__new__(CLAPEmbedder)
    emb.session = DummySession()

    # Create fake mel segments with student mel bins (128) and time frames 10
    mel_segments = np.random.randn(2, 1, 128, 10).astype(np.float32)

    avg_emb, seg_embs = emb.compute_embeddings_from_mel(mel_segments)

    assert avg_emb is not None
    assert seg_embs is not None
    assert len(seg_embs) == 2
    assert avg_emb.shape == (512,)
    assert seg_embs[0].shape == (512,)
