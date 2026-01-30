#!/usr/bin/env python3
"""Simple ONNX speed/throughput comparator

Measures average inference time for a teacher ONNX (default: model/clap_audio_model.onnx)
and a student ONNX (default: student_clap/checkpoints/back-nvidia/model_epoch_5.onnx).

Usage examples (from repo root):

# Using venv inside student_clap (activate first if present):
# source student_clap/venv/bin/activate
# python student_clap/tools/onnx_speed_test.py

# Or with explicit files / device:
# python student_clap/tools/onnx_speed_test.py --teacher model/clap_audio_model.onnx \
#   --student student_clap/checkpoints/back-nvidia/model_epoch_5.onnx --device cpu --iters 200

"""

import argparse
import time
import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path


def get_provider_name(device: str):
    device = device.lower()
    if device in ("cpu", "cpu_execution_provider"):
        return "CPUExecutionProvider"
    if device in ("cuda", "cuda_execution_provider", "gpu"):
        return "CUDAExecutionProvider"
    raise ValueError(f"Unknown device: {device}")


def make_input(shape):
    return np.random.randn(*shape).astype(np.float32)


def run_session(session, input_name, input_data, warmup=10, iters=200):
    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: input_data})

    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        session.run(None, {input_name: input_data})
    end = time.perf_counter()

    total = end - start
    avg = total / iters
    return total, avg


def main():
    p = argparse.ArgumentParser(description="ONNX Teacher vs Student speed test")
    p.add_argument("--teacher", default="model/clap_audio_model.onnx", help="Teacher ONNX path")
    p.add_argument("--student", default="student_clap/checkpoints/back-nvidia/model_epoch_5.onnx", help="Student ONNX path")
    p.add_argument("--device", default="cpu", help="Execution device: cpu or cuda")
    p.add_argument("--shape", default="1,1,128,1000", help="Input shape (comma-separated ints), e.g. 1,1,128,1000")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p.add_argument("--iters", type=int, default=200, help="Measured iterations")
    p.add_argument("--compare-output", action="store_true", help="Compare outputs between teacher and student for one run")
    p.add_argument("--use-cache-song", action="store_true", help="Pick a random mel from cache and compute song-level times (10s segments, 50% overlap)")
    p.add_argument("--song-duration", type=float, default=None, help="Override song duration in seconds (if provided, used instead of DB audio length)")
    args = p.parse_args()

    teacher_path = Path(args.teacher)
    student_path = Path(args.student)

    if not teacher_path.exists():
        print(f"ERROR: teacher ONNX not found: {teacher_path}")
        sys.exit(2)
    if not student_path.exists():
        print(f"ERROR: student ONNX not found: {student_path}")
        sys.exit(2)

    provider = get_provider_name(args.device)
    available = ort.get_available_providers()
    if provider not in available:
        print(f"WARNING: requested provider {provider} not available. Available: {available}. Falling back to CPU.")
        provider = "CPUExecutionProvider"

    print(f"Provider used: {provider}")
    print(f"Teacher: {teacher_path}")
    print(f"Student: {student_path}")

    input_shape = tuple(int(x) for x in args.shape.split(",") if x)
    print(f"Input shape: {input_shape}")

    # Load sessions
    teacher_sess = ort.InferenceSession(str(teacher_path), providers=[provider])
    student_sess = ort.InferenceSession(str(student_path), providers=[provider])

    # Determine input names
    teacher_in = teacher_sess.get_inputs()[0].name
    student_in = student_sess.get_inputs()[0].name

    print(f"Teacher input name: {teacher_in}")
    print(f"Student input name: {student_in}")

    # Make input array
    inp = make_input(input_shape)

    print("Warming up and measuring teacher model...")
    try:
        t_total, t_avg = run_session(teacher_sess, teacher_in, inp, warmup=args.warmup, iters=args.iters)
        print(f"Teacher total: {t_total:.4f}s over {args.iters} iters, avg: {t_avg*1000:.3f} ms / run")
    except Exception as e:
        print(f"Could not run generic teacher measurement with provided --shape={args.shape}: {e}")
        t_total, t_avg = None, None

    print("Warming up and measuring student model...")
    try:
        s_total, s_avg = run_session(student_sess, student_in, inp, warmup=args.warmup, iters=args.iters)
        print(f"Student total: {s_total:.4f}s over {args.iters} iters, avg: {s_avg*1000:.3f} ms / run")
    except Exception as e:
        print(f"Could not run generic student measurement with provided --shape={args.shape}: {e}")
        s_total, s_avg = None, None

    print("\nRESULTS:")
    if t_avg is None or s_avg is None:
        print("  Could not compute simple per-inference speedup because one of the generic measurements failed (incompatible --shape).")
        if t_avg is not None:
            print(f"  Teacher avg: {t_avg*1000:.3f} ms")
        if s_avg is not None:
            print(f"  Student avg: {s_avg*1000:.3f} ms")
    else:
        speedup = t_avg / s_avg if s_avg > 0 else float('inf')
        print(f"  Student is {speedup:.2f}x faster than Teacher (higher is faster)")
        print(f"  Teacher avg: {t_avg*1000:.3f} ms | Student avg: {s_avg*1000:.3f} ms")

    # Optionally compare outputs for a single run
    if args.compare_output:
        to = teacher_sess.run(None, {teacher_in: inp})[0]
        so = student_sess.run(None, {student_in: inp})[0]
        # If outputs have different shapes, try to reduce or warn
        try:
            if to.shape == so.shape:
                diff = np.abs(to - so).max()
                print(f"Max absolute difference between teacher & student: {diff:.6e}")
            else:
                print(f"Warning: output shapes differ (teacher={to.shape}, student={so.shape}), skipping numeric comparison")
        except Exception as e:
            print(f"Could not compare outputs: {e}")

    # If requested, compute song-level totals using cached mel spectrograms
    if args.use_cache_song:
        try:
            import yaml
            # Ensure repo root is on sys.path so we can import student_clap package
            repo_root = Path(__file__).resolve().parents[2]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from student_clap.data.mel_cache import MelSpectrogramCache
            import random

            # Read config.yaml to find mel_cache and sample_rate
            cfg_path = Path("student_clap/config.yaml")
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text())
                mel_cache_path = cfg['paths'].get('mel_cache') or cfg.get('audio', {}).get('mel_cache')
                sample_rate = cfg.get('audio', {}).get('sample_rate', 48000)
            else:
                mel_cache_path = "/Volumes/audiomuse/student_clap_cache/mel_spectrograms.db"
                sample_rate = 48000

            mel_cache = MelSpectrogramCache(mel_cache_path)
            item_ids = mel_cache.get_cached_item_ids()
            if not item_ids:
                print("No cached songs found in mel cache. Abort.")
            else:
                item_id = random.choice(item_ids)
                mel_and_len = mel_cache.get_with_audio_length(item_id)
                if mel_and_len is None:
                    print(f"Could not load mel for selected item {item_id}")
                else:
                    full_mel, audio_length_samples = mel_and_len
                    song_seconds = args.song_duration if args.song_duration is not None else audio_length_samples / sample_rate

                    # Extract overlapped 10s segments using same defaults as dataset
                    segment_length = 480000
                    hop_length = 240000
                    hop_length_stft = 480
                    segment_time_frames = segment_length // hop_length_stft

                    # Determine teacher/student model native time frames from inputs
                    teacher_time_frames = None
                    student_time_frames = None
                    try:
                        t_shape = teacher_sess.get_inputs()[0].shape
                        # shape like [1, 1, 128, 64] or ['batch', 1, 128, 64]
                        teacher_time_frames = int(t_shape[-1]) if isinstance(t_shape[-1], int) else None
                    except Exception:
                        teacher_time_frames = None
                    try:
                        s_shape = student_sess.get_inputs()[0].shape
                        student_time_frames = int(s_shape[-1]) if isinstance(s_shape[-1], int) else None
                    except Exception:
                        student_time_frames = None

                    # Fallbacks: common known values
                    if teacher_time_frames is None:
                        teacher_time_frames = 64
                    if student_time_frames is None:
                        student_time_frames = segment_time_frames

                    # Measure per-inference time for teacher & student with native shapes
                    t_inp_shape = (1, 1, input_shape[2], teacher_time_frames) if len(input_shape) >= 4 else (1,1,128,teacher_time_frames)
                    s_inp_shape = (1, 1, input_shape[2], student_time_frames) if len(input_shape) >= 4 else (1,1,128,student_time_frames)

                    t_dummy = make_input(t_inp_shape)
                    s_dummy = make_input(s_inp_shape)

                    print(f"Measured teacher per-inference avg (shape={t_inp_shape}) ...")
                    t_total2, t_avg2 = run_session(teacher_sess, teacher_in, t_dummy, warmup=args.warmup, iters=args.iters)
                    print(f"Measured student per-inference avg (shape={s_inp_shape}) ...")
                    s_total2, s_avg2 = run_session(student_sess, student_in, s_dummy, warmup=args.warmup, iters=args.iters)

                    # Compute segments count using cache's extraction behavior
                    segments = mel_cache.extract_overlapped_segments(full_mel, audio_length_samples,
                                                                    segment_length=segment_length, hop_length=hop_length,
                                                                    sample_rate=sample_rate, hop_length_stft=hop_length_stft)
                    num_segments = segments.shape[0]

                    # For each segment, teacher needs ceil(segment_time_frames / teacher_time_frames) inferences
                    teacher_inferences_per_segment = int(np.ceil(segment_time_frames / teacher_time_frames))
                    teacher_total_time = teacher_inferences_per_segment * t_avg2 * num_segments

                    # For student, typically 1 inference per segment (segment_time_frames == student_time_frames)
                    student_inferences_per_segment = int(np.ceil(segment_time_frames / student_time_frames))
                    student_total_time = student_inferences_per_segment * s_avg2 * num_segments

                    # Print requested output lines
                    print("\nSONG COMPUTATION RESULTS:")
                    print(f"TEACHER: {teacher_total_time:.2f} SECOND COMPUTATION")
                    print(f"STUDENT: {student_total_time:.2f} SECOND COMPUTATION")
                    print(f"SONG LENGTH: {song_seconds:.2f} SECOND")

        except Exception as e:
            print(f"Could not compute song-level times from cache: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
