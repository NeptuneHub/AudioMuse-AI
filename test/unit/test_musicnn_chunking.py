# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""MusiCNN chunked embedding inference (MUSICNN_BATCH_SIZE).

Verifies that _run_musicnn_models never sends more than MUSICNN_BATCH_SIZE
patches to the embedding model in one inference call, that every patch is
still processed exactly once and in order, and that the pooled outputs are
identical to the single whole-track batch (MUSICNN_BATCH_SIZE=0).

Main Features:
* Asserts no embedding inference call exceeds MUSICNN_BATCH_SIZE patches.
* Checks every patch is processed exactly once and in original order.
* Confirms chunked output matches the whole-track batch (MUSICNN_BATCH_SIZE=0).
"""

import numpy as np

from tasks.analysis import song


def _fake_inference(calls):
    def fake(session, feed_dict, output_name, model_path, label, name):
        (batch,) = feed_dict.values()
        calls.append((label, batch.shape[0]))
        if label == 'embedding':
            out = np.repeat(batch[:, :1, 0], 200, axis=1).astype(np.float32)
        else:
            out = batch[:, :50].astype(np.float32)
        return out, session
    return fake


def _run(monkeypatch, batch_size, patches):
    calls = []
    monkeypatch.setattr(song, 'MUSICNN_BATCH_SIZE', batch_size)
    monkeypatch.setattr(song, 'run_inference_with_oom_fallback', _fake_inference(calls))
    embedding, moods = song._run_musicnn_models(
        patches, [f'mood{i}' for i in range(50)],
        {'embedding': 'unused', 'prediction': 'unused'},
        {'embedding': object(), 'prediction': object()},
        'test-track',
    )
    return embedding, moods, calls


def test_embedding_calls_never_exceed_batch_size(monkeypatch):
    patches = np.arange(10 * 187 * 96, dtype=np.float32).reshape(10, 187, 96)
    _, _, calls = _run(monkeypatch, 3, patches)
    embedding_calls = [n for label, n in calls if label == 'embedding']
    assert embedding_calls == [3, 3, 3, 1]
    assert [n for label, n in calls if label == 'prediction'] == [10]


def test_chunked_output_matches_whole_track_batch(monkeypatch):
    patches = np.random.default_rng(42).random((7, 187, 96)).astype(np.float32)
    chunked_emb, chunked_moods, _ = _run(monkeypatch, 3, patches)
    full_emb, full_moods, calls = _run(monkeypatch, 0, patches.copy())
    assert [n for label, n in calls if label == 'embedding'] == [7]
    np.testing.assert_array_equal(chunked_emb, full_emb)
    assert chunked_moods == full_moods


def test_every_patch_processed_once_and_in_order(monkeypatch):
    patches = np.zeros((5, 187, 96), dtype=np.float32)
    patches[:, 0, 0] = np.arange(5)
    embedding, _, _ = _run(monkeypatch, 2, patches)
    np.testing.assert_allclose(embedding, np.full(200, np.arange(5).mean()))
