import numpy as np
from student_clap.data.clap_embedder import CLAPEmbedder


class DummySession:
    def run(self, *args, **kwargs):
        # Return a fake embedding of shape (batch, 512) matching input batch size
        batch_size = args[1]['mel_spectrogram'].shape[0] if args else 1
        return [np.zeros((batch_size, 512), dtype=np.float32)]


def test_compute_embeddings_from_mel_resamples():
    # Create a dummy CLAPEmbedder without initializing ONNX session
    emb = CLAPEmbedder.__new__(CLAPEmbedder)
    emb.session = DummySession()
    emb._backend = 'onnx'
    emb.segment_batch_size = 5

    # Create fake mel segments with student mel bins (128) and time frames 10
    mel_segments = np.random.randn(2, 1, 128, 10).astype(np.float32)

    avg_emb, seg_embs = emb.compute_embeddings_from_mel(mel_segments)

    assert avg_emb is not None
    assert seg_embs is not None
    assert len(seg_embs) == 2
    assert avg_emb.shape == (512,)
    assert seg_embs[0].shape == (512,)


def test_compute_embeddings_from_mel_teacher_format():
    """Teacher mel is (n_mels=64, time) — resample should be no-op."""
    emb = CLAPEmbedder.__new__(CLAPEmbedder)
    emb.session = DummySession()
    emb._backend = 'onnx'
    emb.segment_batch_size = 5

    # Teacher mel: 64 mel bins, 1501 time frames (matching compute_mel_spectrogram output)
    mel_segments = np.random.randn(3, 1, 64, 1501).astype(np.float32)

    avg_emb, seg_embs = emb.compute_embeddings_from_mel(mel_segments)

    assert avg_emb is not None
    assert seg_embs is not None
    assert len(seg_embs) == 3
    assert avg_emb.shape == (512,)
    assert seg_embs[0].shape == (512,)
