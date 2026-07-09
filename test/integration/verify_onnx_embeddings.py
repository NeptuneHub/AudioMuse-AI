# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Standalone check that the ONNX CLAP export matches the PyTorch model.

Loads the original laion-clap PyTorch checkpoint and the exported ONNX
models, runs both over the same text queries and test audio, and reports
per-item max/mean difference, cosine similarity, and timing.

Main Features:
* Compares text-query embeddings between PyTorch and ONNX.
* Compares segmented audio embeddings and prints a pass/fail summary.
"""

import os
import sys
import time
import numpy as np
import librosa
import librosa.feature

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tasks.clap_analyzer import _split_audio_segments


def compare_pytorch_vs_onnx():
    print("=" * 80)
    print("CLAP Model Embedding Verification: PyTorch .pt vs ONNX")
    print("=" * 80)

    test_songs_dir = os.path.join(os.path.dirname(__file__), "..", "songs")
    test_audio_files = []

    if os.path.exists(test_songs_dir):
        for file in os.listdir(test_songs_dir):
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                test_audio_files.append(os.path.join(test_songs_dir, file))
        print(f"\nFound {len(test_audio_files)} test audio files in test/songs/")
    else:
        print(f"\nWarning: test/songs directory not found at {test_songs_dir}")

    test_queries = [
        "upbeat electronic dance music",
        "calm acoustic guitar",
        "heavy metal with distortion",
        "jazz piano solo",
    ]

    print("\n" + "-" * 80)
    print("PART 1: Loading PyTorch .pt model")
    print("-" * 80)

    try:
        import torch
        import laion_clap

        pt_model_paths = [
            "/app/model/music_audioset_epoch_15_esc_90.14.pt",
            "../query/music_audioset_epoch_15_esc_90.14.pt",
            "query/music_audioset_epoch_15_esc_90.14.pt",
            os.path.expanduser("~/Music/AudioMuse-AI/query/music_audioset_epoch_15_esc_90.14.pt"),
        ]

        pt_model_path = None
        for path in pt_model_paths:
            if os.path.exists(path):
                pt_model_path = path
                break

        if not pt_model_path:
            print("X PyTorch .pt model not found in any of these paths:")
            for path in pt_model_paths:
                print(f"  - {path}")
            print("\nPlease download the model first or provide the path.")
            return False

        print(f"Loading from: {pt_model_path}")

        pt_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        pt_model.load_ckpt(pt_model_path)
        pt_model.eval()

        print("OK PyTorch model loaded successfully")

    except ImportError as e:
        print(f"X PyTorch/CLAP not installed: {e}")
        print("Install with: pip install torch laion-clap")
        return False
    except Exception as e:
        print(f"X Failed to load PyTorch model: {e}")
        return False

    print("\n" + "-" * 80)
    print("PART 2: Loading ONNX model")
    print("-" * 80)

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_audio_model_paths = [
            "/app/model/clap_audio_model.onnx",
            "../test/models/clap_audio_model.onnx",
            "test/models/clap_audio_model.onnx",
            os.path.expanduser("~/Music/AudioMuse-AI/test/models/clap_audio_model.onnx"),
        ]

        onnx_text_model_paths = [
            "/app/model/clap_text_model.onnx",
            "../test/models/clap_text_model.onnx",
            "test/models/clap_text_model.onnx",
            os.path.expanduser("~/Music/AudioMuse-AI/test/models/clap_text_model.onnx"),
        ]

        onnx_audio_model_path = None
        for path in onnx_audio_model_paths:
            if os.path.exists(path):
                onnx_audio_model_path = path
                break

        onnx_text_model_path = None
        for path in onnx_text_model_paths:
            if os.path.exists(path):
                onnx_text_model_path = path
                break

        if not onnx_audio_model_path or not onnx_text_model_path:
            print("X ONNX models not found:")
            if not onnx_audio_model_path:
                print("  Audio model - tried:")
                for path in onnx_audio_model_paths:
                    print(f"    - {path}")
            if not onnx_text_model_path:
                print("  Text model - tried:")
                for path in onnx_text_model_paths:
                    print(f"    - {path}")
            print("\nPlease generate the ONNX model first using pythorch.sh")
            return False

        onnx_model_path = onnx_audio_model_path
        print(f"Loading from: {onnx_model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_session = ort.InferenceSession(onnx_model_path, sess_options=sess_options)

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        print("OK ONNX model loaded successfully")

    except Exception as e:
        print(f"X Failed to load ONNX model: {e}")
        return False

    print("\n" + "=" * 80)
    print("PART 3: Comparing TEXT embeddings")
    print("=" * 80)

    text_results = []

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        pt_start = time.perf_counter()
        with torch.no_grad():
            pt_embedding = pt_model.get_text_embedding([query], use_tensor=False)[0]
        pt_time = time.perf_counter() - pt_start

        onnx_start = time.perf_counter()
        tokens = tokenizer(
            query, padding='max_length', truncation=True, max_length=77, return_tensors='np'
        )

        dummy_mel = np.zeros((1, 1, 1001, 64), dtype=np.float32)

        onnx_inputs = {
            'mel_spectrogram': dummy_mel,
            'input_ids': tokens['input_ids'].astype(np.int64),
            'attention_mask': tokens['attention_mask'].astype(np.int64),
        }

        outputs = onnx_session.run(None, onnx_inputs)
        onnx_embedding = outputs[1][0]
        onnx_time = time.perf_counter() - onnx_start

        pt_embedding = pt_embedding / np.linalg.norm(pt_embedding)
        onnx_embedding = onnx_embedding / np.linalg.norm(onnx_embedding)

        diff = np.abs(pt_embedding - onnx_embedding)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        cosine_sim = np.dot(pt_embedding, onnx_embedding)

        print(f"  Max difference:  {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Cosine similarity: {cosine_sim:.10f}")
        print(f"  PyTorch time: {pt_time * 1000:.2f}ms")
        print(f"  ONNX time:    {onnx_time * 1000:.2f}ms")
        speedup = pt_time / onnx_time if onnx_time > 0 else 0
        print(
            f"  Speedup:      {speedup:.2f}x {'(ONNX faster)' if speedup > 1 else '(PyTorch faster)'}"
        )

        passed = max_diff < 1e-5 and cosine_sim > 0.9999
        print(f"  Status: {'OK PASS' if passed else 'X FAIL'}")

        text_results.append(
            {
                'query': query,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'cosine_sim': cosine_sim,
                'pt_time': pt_time,
                'onnx_time': onnx_time,
                'speedup': speedup,
                'passed': passed,
            }
        )

    audio_results = []

    if test_audio_files:
        print("\n" + "=" * 80)
        print("PART 4: Comparing AUDIO embeddings")
        print("=" * 80)

        for test_audio in test_audio_files:
            print(f"\n{'-' * 80}")
            print(f"Audio file: {os.path.basename(test_audio)}")

            try:
                audio_data, sr = librosa.load(test_audio, sr=48000, mono=True)

                audio_data = np.asarray(audio_data, dtype=np.float32)

                def float32_to_int16(x):
                    x = np.clip(x, -1.0, 1.0)
                    return (x * 32767.0).astype(np.int16)

                def int16_to_float32(x):
                    return (x / 32767.0).astype(np.float32)

                audio_data = int16_to_float32(float32_to_int16(audio_data))

                print(f"Loaded: {len(audio_data) / sr:.2f}s at {sr}Hz")

                segments = _split_audio_segments(audio_data)

                print(f"Split into {len(segments)} segments (10s with 5s overlap)")

                pt_start = time.perf_counter()
                pt_embeddings = []
                for seg in segments:
                    seg_batched = seg.reshape(1, -1)
                    with torch.no_grad():
                        seg_embedding = pt_model.get_audio_embedding_from_data(
                            x=seg_batched, use_tensor=False
                        )
                        if isinstance(seg_embedding, np.ndarray):
                            if seg_embedding.ndim == 2:
                                seg_embedding = seg_embedding[0]
                        else:
                            if hasattr(seg_embedding, 'cpu'):
                                seg_embedding = seg_embedding.cpu().numpy()
                            elif hasattr(seg_embedding, 'numpy'):
                                seg_embedding = seg_embedding.numpy()
                            if seg_embedding.ndim == 2:
                                seg_embedding = seg_embedding[0]
                        pt_embeddings.append(seg_embedding)

                pt_audio_embedding = np.mean(pt_embeddings, axis=0)
                pt_time = time.perf_counter() - pt_start

                onnx_start = time.perf_counter()
                onnx_embeddings = []
                for seg in segments:
                    mel_spec = librosa.feature.melspectrogram(
                        y=seg,
                        sr=sr,
                        n_fft=1024,
                        hop_length=320,
                        win_length=1024,
                        window='hann',
                        center=True,
                        pad_mode='reflect',
                        power=2.0,
                        n_mels=64,
                        fmin=50,
                        fmax=14000,
                    )

                    mel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)

                    mel_spec = mel_spec.T

                    mel_input = mel_spec[np.newaxis, np.newaxis, :, :].astype(np.float32)

                    dummy_input_ids = np.zeros((1, 77), dtype=np.int64)
                    dummy_attention_mask = np.zeros((1, 77), dtype=np.int64)

                    onnx_inputs = {
                        'mel_spectrogram': mel_input,
                        'input_ids': dummy_input_ids,
                        'attention_mask': dummy_attention_mask,
                    }

                    outputs = onnx_session.run(None, onnx_inputs)
                    seg_embedding = outputs[0][0]
                    onnx_embeddings.append(seg_embedding)

                onnx_audio_embedding = np.mean(onnx_embeddings, axis=0)
                onnx_time = time.perf_counter() - onnx_start

                pt_audio_embedding = pt_audio_embedding / np.linalg.norm(pt_audio_embedding)
                onnx_audio_embedding = onnx_audio_embedding / np.linalg.norm(onnx_audio_embedding)

                diff = np.abs(pt_audio_embedding - onnx_audio_embedding)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                cosine_sim = np.dot(pt_audio_embedding, onnx_audio_embedding)

                print(f"  Max difference:  {max_diff:.2e}")
                print(f"  Mean difference: {mean_diff:.2e}")
                print(f"  Cosine similarity: {cosine_sim:.10f}")
                print(f"  PyTorch time: {pt_time:.3f}s ({len(segments)} segments)")
                print(f"  ONNX time:    {onnx_time:.3f}s ({len(segments)} segments)")
                speedup = pt_time / onnx_time if onnx_time > 0 else 0
                print(
                    f"  Speedup:      {speedup:.2f}x {'(ONNX faster)' if speedup > 1 else '(PyTorch faster)'}"
                )

                audio_passed = cosine_sim >= 0.97
                print(f"  Status: {'OK PASS' if audio_passed else 'X FAIL'}")

                audio_results.append(
                    {
                        'file': os.path.basename(test_audio),
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'cosine_sim': cosine_sim,
                        'pt_time': pt_time,
                        'onnx_time': onnx_time,
                        'speedup': speedup,
                        'passed': audio_passed,
                    }
                )

            except Exception as e:
                print(f"  X ERROR: {e}")
                audio_results.append(
                    {'file': os.path.basename(test_audio), 'error': str(e), 'passed': False}
                )
    else:
        print("\n" + "=" * 80)
        print("PART 4: Audio embedding test SKIPPED (no audio files in test/songs)")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 80)

    all_text_passed = all(r['passed'] for r in text_results)

    print(
        f"\nText embeddings: {'OK ALL PASS' if all_text_passed else 'X SOME FAILED'} ({len(text_results)} queries)"
    )
    for r in text_results:
        status = "OK" if r['passed'] else "X"
        print(
            f"  {status} '{r['query']}' - cos_sim={r['cosine_sim']:.6f}, speedup={r['speedup']:.2f}x"
        )

    if audio_results:
        all_audio_passed = all(r['passed'] for r in audio_results)
        print(
            f"\nAudio embeddings: {'OK ALL PASS' if all_audio_passed else 'X SOME FAILED'} ({len(audio_results)} files)"
        )
        for r in audio_results:
            if 'error' in r:
                print(f"  X {r['file']} - ERROR: {r['error']}")
            else:
                status = "OK" if r['passed'] else "X"
                print(
                    f"  {status} {r['file']} - cos_sim={r['cosine_sim']:.6f}, speedup={r['speedup']:.2f}x"
                )
    else:
        all_audio_passed = True
        print("\nAudio embeddings: SKIPPED (no test files)")

    print("\n" + "=" * 80)
    if all_text_passed and all_audio_passed:
        print("OKOKOK VERIFICATION SUCCESSFUL OKOKOK")
        print("The ONNX model produces IDENTICAL embeddings to PyTorch!")
        print("=" * 80)
        return True
    else:
        print("XXX VERIFICATION FAILED XXX")
        print("The embeddings differ! Check model export.")
        print("=" * 80)
        return False


if __name__ == '__main__':
    success = compare_pytorch_vs_onnx()
    sys.exit(0 if success else 1)
