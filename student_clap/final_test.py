#!/usr/bin/env python3
"""
Student CLAP Final Evaluation Script

Compares teacher vs student audio embeddings against teacher text embeddings
for all songs in the test directory. Shows cosine similarity for each
text query / song combination.

Usage:
    python final_test.py
    python final_test.py --songs-dir ../test/songs
    python final_test.py --student-model ./model/audiomuseai_clap_auido.onnx
"""

import os
import sys
import argparse
import numpy as np
import librosa
import onnxruntime as ort

# ── Text queries to evaluate (easy to extend) ──────────────────────────
TEXT_QUERIES = [
    "Calm Piano song",
    "Energetic POP song",
    "Love Rock Song",
    "Happy Pop song",
    "POP song with Female vocalist",
    "Instrumental song",
    "Female Vocalist",
    "Male Vocalist",
    "Ukulele POP song",
    "Jazz Sax song",
    "Distorted Electric Guitar",
    "Drum and Bass beat",
    "Heavy Metal song",
    "Ambient song"
]

# ── Audio constants ─────────────────────────────────────────────────────
SAMPLE_RATE = 48000
SEGMENT_LENGTH = 480000   # 10 seconds
HOP_LENGTH = 240000       # 5 seconds (50 % overlap)

# Teacher mel-spec params (HTSAT-base)
TEACHER_N_FFT = 1024
TEACHER_HOP = 320
TEACHER_N_MELS = 64
TEACHER_FMIN = 50
TEACHER_FMAX = 14000

# Student mel-spec params (EfficientAT)
STUDENT_N_FFT = 2048
STUDENT_HOP = 480
STUDENT_N_MELS = 128
STUDENT_FMIN = 0
STUDENT_FMAX = 14000


# ── Helpers ─────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_audio(path: str) -> np.ndarray:
    """Load and quantize audio to match CLAP preprocessing."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767.0).astype(np.int16)
    audio = (audio / 32767.0).astype(np.float32)
    return audio


def segment_audio(audio: np.ndarray):
    """Split audio into overlapping 10-second segments."""
    segments = []
    total = len(audio)
    if total <= SEGMENT_LENGTH:
        segments.append(np.pad(audio, (0, SEGMENT_LENGTH - total)))
    else:
        for start in range(0, total - SEGMENT_LENGTH + 1, HOP_LENGTH):
            segments.append(audio[start:start + SEGMENT_LENGTH])
        last_start = len(segments) * HOP_LENGTH
        if last_start < total:
            segments.append(audio[-SEGMENT_LENGTH:])
    return segments


# ── Teacher mel spectrogram (HTSAT-base) ────────────────────────────────

def teacher_mel(audio_segment: np.ndarray) -> np.ndarray:
    """Compute teacher mel-spec: returns (1, 1, time, 64)."""
    mel = librosa.feature.melspectrogram(
        y=audio_segment, sr=SAMPLE_RATE,
        n_fft=TEACHER_N_FFT, hop_length=TEACHER_HOP, win_length=TEACHER_N_FFT,
        window="hann", center=True, pad_mode="reflect", power=2.0,
        n_mels=TEACHER_N_MELS, fmin=TEACHER_FMIN, fmax=TEACHER_FMAX,
    )
    mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    mel = mel.T  # (time, 64)
    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,T,64)


# ── Student mel spectrogram (EfficientAT) ──────────────────────────────

def student_mel(audio_segment: np.ndarray) -> np.ndarray:
    """Compute student mel-spec: returns (1, 1, 128, time).

    Must match the training preprocessing in
    student_clap/preprocessing/mel_spectrogram.py which uses
    librosa.power_to_db (dB scale), NOT np.log.
    """
    mel = librosa.feature.melspectrogram(
        y=audio_segment, sr=SAMPLE_RATE,
        n_fft=STUDENT_N_FFT, hop_length=STUDENT_HOP, win_length=STUDENT_N_FFT,
        window="hann", center=True, pad_mode="reflect", power=2.0,
        n_mels=STUDENT_N_MELS, fmin=STUDENT_FMIN, fmax=STUDENT_FMAX,
    )
    mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    # shape: (128, time) -> (1, 1, 128, time)
    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


# ── ONNX session loader ────────────────────────────────────────────────

def load_onnx(path: str) -> ort.InferenceSession:
    """Load an ONNX model, fixing external data references if needed."""
    import onnx

    model = onnx.load(path, load_external_data=False)

    # If the model uses external data, rewrite references to the actual .data
    # file sitting next to the .onnx file (handles renames after export).
    # torch.onnx.export names it "<name>.onnx.data", so check both patterns.
    data_file = path + ".data"  # e.g. model_epoch_1.onnx.data
    if not os.path.exists(data_file):
        data_file = os.path.splitext(path)[0] + ".data"  # e.g. model_epoch_1.data
    if os.path.exists(data_file):
        # Use absolute path so ONNXRuntime can find the external data even when the
        # session is created from serialized bytes (cwd might be different).
        data_path = os.path.abspath(data_file)
        for tensor in model.graph.initializer:
            if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
                for entry in tensor.external_data:
                    if entry.key == "location":
                        entry.value = data_path

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=opts,
        providers=providers,
    )


# ── Embedding functions ─────────────────────────────────────────────────

def get_teacher_audio_embedding(session: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    """Average teacher audio embedding over segments."""
    segments = segment_audio(audio)
    embeddings = []
    for seg in segments:
        mel = teacher_mel(seg)
        out = session.run(None, {"mel_spectrogram": mel})[0]
        embeddings.append(out[0])
    avg = np.mean(embeddings, axis=0)
    return avg / (np.linalg.norm(avg) + 1e-9)


def get_student_audio_embedding(session: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    """Average student audio embedding over segments."""
    segments = segment_audio(audio)
    embeddings = []
    for seg in segments:
        mel = student_mel(seg)
        out = session.run(None, {"mel_spectrogram": mel})[0]
        embeddings.append(out[0])
    avg = np.mean(embeddings, axis=0)
    return avg / (np.linalg.norm(avg) + 1e-9)


def get_teacher_text_embeddings(session: ort.InferenceSession, texts: list) -> np.ndarray:
    """Get teacher text embeddings for a list of queries."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    encoded = tokenizer(
        texts, max_length=77, padding="max_length",
        truncation=True, return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    out = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / (norms + 1e-9)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Student CLAP final evaluation")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--songs-dir", default=os.path.join(script_dir, "..", "test", "songs"))
    parser.add_argument("--student-model", default=os.path.join(script_dir, "models", "model_epoch_3.onnx"))
    parser.add_argument("--teacher-audio-model", default=os.path.join(script_dir, "..", "model", "clap_audio_model.onnx"))
    parser.add_argument("--teacher-text-model", default=os.path.join(script_dir, "..", "model", "clap_text_model.onnx"))
    args = parser.parse_args()

    # Resolve paths
    songs_dir = os.path.abspath(args.songs_dir)
    student_path = os.path.abspath(args.student_model)
    teacher_audio_path = os.path.abspath(args.teacher_audio_model)
    teacher_text_path = os.path.abspath(args.teacher_text_model)

    # Validate
    for label, p in [("Songs dir", songs_dir), ("Student model", student_path),
                      ("Teacher audio model", teacher_audio_path), ("Teacher text model", teacher_text_path)]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    # Discover songs
    exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    songs = sorted([f for f in os.listdir(songs_dir) if os.path.splitext(f)[1].lower() in exts])
    if not songs:
        print(f"No audio files found in {songs_dir}")
        sys.exit(1)

    print("=" * 80)
    print("  STUDENT CLAP — FINAL EVALUATION")
    print("=" * 80)
    print(f"  Songs directory : {songs_dir}")
    print(f"  Student model   : {student_path}")
    print(f"  Teacher audio   : {teacher_audio_path}")
    print(f"  Teacher text    : {teacher_text_path}")
    print(f"  Songs found     : {len(songs)}")
    print(f"  Text queries    : {len(TEXT_QUERIES)}")
    print("=" * 80)

    # Load models
    print("\nLoading models...")
    teacher_audio_sess = load_onnx(teacher_audio_path)
    student_audio_sess = load_onnx(student_path)
    teacher_text_sess = load_onnx(teacher_text_path)
    print("  All models loaded.\n")

    # Compute text embeddings (once)
    print("Computing teacher text embeddings...")
    text_embeddings = get_teacher_text_embeddings(teacher_text_sess, TEXT_QUERIES)
    for i, q in enumerate(TEXT_QUERIES):
        print(f"  [{i+1}] \"{q}\"  (norm={np.linalg.norm(text_embeddings[i]):.4f})")
    print()

    # Per-song results: list of dicts
    all_results = []

    for song_idx, song_file in enumerate(songs):
        song_path = os.path.join(songs_dir, song_file)
        song_name = os.path.splitext(song_file)[0]
        print("-" * 80)
        print(f"  Song {song_idx+1}/{len(songs)}: {song_name}")
        print("-" * 80)

        # Load audio
        audio = load_audio(song_path)
        duration = len(audio) / SAMPLE_RATE
        segments = segment_audio(audio)
        print(f"  Duration: {duration:.1f}s | Segments: {len(segments)}")

        # Teacher audio embedding
        print("  Computing teacher audio embedding...", end=" ", flush=True)
        teacher_emb = get_teacher_audio_embedding(teacher_audio_sess, audio)
        print("done")

        # Student audio embedding
        print("  Computing student audio embedding...", end=" ", flush=True)
        student_emb = get_student_audio_embedding(student_audio_sess, audio)
        print("done")

        # Teacher-Student audio cosine
        ts_cos = cosine_similarity(teacher_emb, student_emb)
        print(f"\n  Teacher vs Student audio cosine: {ts_cos:.4f}")

        # Text-vs-audio cosine similarities
        song_result = {"name": song_name, "teacher_student_cos": ts_cos, "queries": {}}
        print(f"\n  {'Text Query':<30s}  {'Teacher':>9s}  {'Student':>9s}  {'Delta':>9s}")
        print(f"  {'─'*30}  {'─'*9}  {'─'*9}  {'─'*9}")

        for qi, query in enumerate(TEXT_QUERIES):
            t_cos = cosine_similarity(text_embeddings[qi], teacher_emb)
            s_cos = cosine_similarity(text_embeddings[qi], student_emb)
            delta = s_cos - t_cos
            song_result["queries"][query] = {"teacher": t_cos, "student": s_cos, "delta": delta}
            print(f"  {query:<30s}  {t_cos:>+9.4f}  {s_cos:>+9.4f}  {delta:>+9.4f}")

        all_results.append(song_result)
        print()

    # ── Final recap ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL RECAP")
    print("=" * 80)

    # Teacher vs Student audio similarity
    ts_values = [r["teacher_student_cos"] for r in all_results]
    print(f"\n  Teacher vs Student audio cosine similarity:")
    for r in all_results:
        print(f"    {r['name'][:50]:<50s}  {r['teacher_student_cos']:>+.4f}")
    print(f"    {'MEAN':<50s}  {np.mean(ts_values):>+.4f}")
    print(f"    {'MIN':<50s}  {np.min(ts_values):>+.4f}")
    print(f"    {'MAX':<50s}  {np.max(ts_values):>+.4f}")

    # Per-query summary
    print(f"\n  Text query cosine similarities (mean across songs):")
    print(f"\n  {'Query':<30s}  {'Teacher':>9s}  {'Student':>9s}  {'Delta':>9s}")
    print(f"  {'─'*30}  {'─'*9}  {'─'*9}  {'─'*9}")

    all_teacher_vals = []
    all_student_vals = []
    for query in TEXT_QUERIES:
        t_vals = [r["queries"][query]["teacher"] for r in all_results]
        s_vals = [r["queries"][query]["student"] for r in all_results]
        d_vals = [r["queries"][query]["delta"] for r in all_results]
        all_teacher_vals.extend(t_vals)
        all_student_vals.extend(s_vals)
        print(f"  {query:<30s}  {np.mean(t_vals):>+9.4f}  {np.mean(s_vals):>+9.4f}  {np.mean(d_vals):>+9.4f}")

    print(f"  {'─'*30}  {'─'*9}  {'─'*9}  {'─'*9}")
    overall_delta = np.mean(all_student_vals) - np.mean(all_teacher_vals)
    print(f"  {'OVERALL MEAN':<30s}  {np.mean(all_teacher_vals):>+9.4f}  {np.mean(all_student_vals):>+9.4f}  {overall_delta:>+9.4f}")

    # ── MIR ranking preservation report using R@k and mAP@10 ─────────────
    print("\n  MIR RANKING METRICS: R@1, R@5, mAP@10 (teacher top-10 as relevance)")

    songs_list = [r["name"] for r in all_results]
    num_songs = len(songs_list)

    query_metrics = {}
    sum_r1 = 0.0
    sum_r5 = 0.0
    sum_ap10 = 0.0

    for query in TEXT_QUERIES:
        t_pairs = [(r["name"], r["queries"][query]["teacher"]) for r in all_results]
        s_pairs = [(r["name"], r["queries"][query]["student"]) for r in all_results]
        t_sorted = [s for s, _ in sorted(t_pairs, key=lambda x: x[1], reverse=True)]
        s_sorted = [s for s, _ in sorted(s_pairs, key=lambda x: x[1], reverse=True)]

        # R@1 (accuracy on top-1)
        r1 = 1.0 if (t_sorted and s_sorted and t_sorted[0] == s_sorted[0]) else 0.0

        # R@5 (fraction of teacher top-5 present in student top-5)
        k5 = min(5, num_songs)
        t_top5 = set(t_sorted[:k5])
        s_top5 = set(s_sorted[:k5])
        overlap5 = len(t_top5 & s_top5)
        r5 = overlap5 / k5 if k5 > 0 else 0.0

        # mAP@10: average precision where relevant = teacher top-10
        k10 = min(10, num_songs)
        t_top10_list = t_sorted[:k10]
        s_top10_list = s_sorted[:k10]

        def calculate_ap(teacher_top_k, student_top_k):
            score = 0.0
            num_hits = 0.0
            for i, p in enumerate(student_top_k):
                # check if student item is relevant (in teacher top-k)
                if p in teacher_top_k:
                    num_hits += 1.0
                    precision_at_i = num_hits / (i + 1.0)
                    score += precision_at_i
            # divide by number of relevant items (teacher top-k)
            return score / float(len(teacher_top_k)) if len(teacher_top_k) > 0 else 0.0

        ap10 = calculate_ap(t_top10_list, s_top10_list)

        query_metrics[query] = {"r1": r1, "r5": r5, "ap10": ap10, "overlap5": overlap5}
        sum_r1 += r1
        sum_r5 += r5
        sum_ap10 += ap10

    # Print per-query table
    print("\n  {:<30s}  {:^7s}  {:^12s}  {:^8s}".format("Query", "R@1", "R@5", "mAP@10"))
    print("  " + "-"*30 + "  " + "-"*7 + "  " + "-"*12 + "  " + "-"*8)
    for q in TEXT_QUERIES:
        qm = query_metrics[q]
        r1_str = f"{int(qm['r1'])}/1"
        r5_str = f"{qm['overlap5']}/{min(5, num_songs)} ({qm['r5']*100:.1f}%)"
        ap10_str = f"{qm['ap10']:.3f}"
        short_q = (q[:27] + '...') if len(q) > 30 else q
        print(f"  {short_q:<30s}  {r1_str:^7s}  {r5_str:^12s}  {ap10_str:^8s}")

    # Summary statistics
    n = len(TEXT_QUERIES)
    mean_r1 = sum_r1 / n if n else 0.0
    mean_r5 = sum_r5 / n if n else 0.0
    mean_map10 = sum_ap10 / n if n else 0.0

    print("\n  SUMMARY:")
    print(f"    Mean R@1 (accuracy) : {mean_r1*100:.1f}% ({int(sum_r1)}/{n})")
    print(f"    Mean R@5            : {mean_r5*100:.1f}% (mean overlap {mean_r5*5:.2f}/5)")
    print(f"    mAP@10 (mean)       : {mean_map10:.3f}")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
