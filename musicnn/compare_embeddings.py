"""
Compare embeddings from three MSD-MusiCNN sources:

  1. Original musicnn Python library  (TF model, direct session inference)
  2. Our newly exported ONNX model    (musicnn_embedding.onnx)
  3. Project ONNX model               (test/models/msd-musicnn-1.onnx)

For each song in test/songs/ the script:
  - Computes mel-spectrogram patches (shared preprocessing)
  - Runs each model to get per-patch 200-dim embeddings
  - Averages patches → one embedding per song per model
  - Reports pairwise cosine similarity & cosine distance

Usage:
    python compare_embeddings.py
"""

import os
import sys
import time
import glob
import numpy as np
import librosa

# ─── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_SONGS   = os.path.join(PROJECT_ROOT, 'test', 'songs')
PROJECT_ONNX = os.path.join(PROJECT_ROOT, 'test', 'models', 'msd-musicnn-1.onnx')
OUR_ONNX     = os.path.join(SCRIPT_DIR, 'musicnn_embedding.onnx')

# ─── Preprocessing parameters ──────────────────────────────────────────
# Matches the project pipeline (tasks/analysis.py) AND the musicnn model
SR         = 16000
N_FFT      = 512
HOP_LENGTH = 256
N_MELS     = 96
FRAME_SIZE = 187   # time-frames per patch (≈ 3 s of audio)


def preprocess_audio(file_path):
    """
    Load audio → mel spectrogram → log-compress → split into
    non-overlapping (187, 96) patches.

    Returns
    -------
    patches : np.ndarray, shape (num_patches, 187, 96), dtype float32
    """
    audio, sr = librosa.load(file_path, sr=SR, mono=True)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window='hann',
        center=False,
        power=2.0,
        norm='slaney',
        htk=False,
    )
    log_mel = np.log10(1 + 10000 * mel)   # (96, T)
    log_mel = log_mel.T                    # (T, 96)

    patches = []
    for i in range(0, log_mel.shape[0] - FRAME_SIZE + 1, FRAME_SIZE):
        patches.append(log_mel[i : i + FRAME_SIZE])

    if not patches:
        # Audio shorter than one patch → zero-pad
        padded = np.zeros((FRAME_SIZE, N_MELS), dtype=np.float32)
        padded[: log_mel.shape[0]] = log_mel
        patches.append(padded)

    return np.array(patches, dtype=np.float32)


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ─── Model loaders ─────────────────────────────────────────────────────

def load_musicnn_tf():
    """
    Load the original TF model shipped with the musicnn library.
    Returns (infer_fn, session).
    infer_fn(patches) → embeddings  (numpy array, shape (N, 200))
    """
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    import musicnn
    musicnn_pkg = os.path.dirname(musicnn.__file__)

    model_dir = os.path.join(musicnn_pkg, 'MSD_musicnn')
    if not os.path.isdir(model_dir):
        model_dir = os.path.join(musicnn_pkg, 'trained_models', 'MSD_musicnn')

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"MSD_musicnn not found in {musicnn_pkg}")

    metas = [f for f in os.listdir(model_dir) if f.endswith('.meta')]
    if not metas:
        raise FileNotFoundError(f"No .meta in {model_dir}")

    meta_file   = os.path.join(model_dir, metas[0])
    ckpt_prefix = meta_file[: -len('.meta')]

    tf.compat.v1.reset_default_graph()

    # Strip _output_shapes to work around TF2 incompatibility with
    # the gradient nodes saved in the original musicnn checkpoint.
    from tensorflow.core.protobuf import meta_graph_pb2

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    with open(meta_file, 'rb') as f:
        meta_graph_def.ParseFromString(f.read())

    for node in meta_graph_def.graph_def.node:
        if '_output_shapes' in node.attr:
            del node.attr['_output_shapes']

    sess = tf.compat.v1.Session()
    tf.compat.v1.train.import_meta_graph(meta_graph_def)
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, ckpt_prefix)

    INPUT   = 'model/Placeholder:0'
    LABELS  = 'model/Placeholder_1:0'    # (batch, 50) — unused, must be fed
    ISTRAIN = 'model/Placeholder_2:0'    # bool — is_training flag
    EMBED   = 'model/dense/BiasAdd:0'

    def infer(patches):
        dummy_labels = np.zeros((patches.shape[0], 50), dtype=np.float32)
        return sess.run(EMBED, feed_dict={
            INPUT: patches, ISTRAIN: False, LABELS: dummy_labels
        })

    return infer, sess


def load_onnx_model(onnx_path):
    """
    Load an ONNX model via onnxruntime.
    Returns infer_fn(patches) → embeddings  (numpy array, shape (N, 200))
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, opts)

    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def infer(patches):
        return session.run([output_name], {input_name: patches})[0]

    return infer


# ─── Main comparison ───────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  MSD-MusiCNN Embedding Comparison")
    print("=" * 90)

    # ── Check prerequisites ─────────────────────────────────────────────
    errors = []
    if not os.path.isdir(TEST_SONGS):
        errors.append(f"test/songs not found: {TEST_SONGS}")
    if not os.path.isfile(PROJECT_ONNX):
        errors.append(f"Project ONNX not found: {PROJECT_ONNX}")
    if not os.path.isfile(OUR_ONNX):
        errors.append(
            f"Our ONNX not found: {OUR_ONNX}\n"
            "  → Run  python export_onnx.py  first."
        )
    if errors:
        for e in errors:
            print(f"✗ {e}")
        sys.exit(1)

    # ── Discover songs ──────────────────────────────────────────────────
    exts = ('*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a')
    songs = sorted(
        p for ext in exts for p in glob.glob(os.path.join(TEST_SONGS, ext))
    )
    print(f"\nFound {len(songs)} song(s) in test/songs/")
    if not songs:
        print("✗ No audio files found.")
        sys.exit(1)

    # ── Load all three models ───────────────────────────────────────────
    print("\n--- Loading models ---")

    print("  [1/3] musicnn TF model …", end=" ", flush=True)
    tf_infer, tf_sess = load_musicnn_tf()
    print("✓")

    print(f"  [2/3] Our ONNX ({os.path.basename(OUR_ONNX)}) …", end=" ", flush=True)
    our_infer = load_onnx_model(OUR_ONNX)
    print("✓")

    print(f"  [3/3] Project ONNX ({os.path.basename(PROJECT_ONNX)}) …", end=" ", flush=True)
    proj_infer = load_onnx_model(PROJECT_ONNX)
    print("✓")

    # ── Process each song ───────────────────────────────────────────────
    all_results = []

    for song_path in songs:
        name = os.path.basename(song_path)
        print(f"\n{'─' * 90}")
        print(f"Song: {name}")

        patches = preprocess_audio(song_path)
        print(f"  Patches: {patches.shape[0]}  (shape per patch: {patches.shape[1:]})")

        # --- Inference ---
        t0 = time.perf_counter()
        emb_tf = tf_infer(patches)
        dt_tf  = time.perf_counter() - t0

        t0 = time.perf_counter()
        emb_our = our_infer(patches)
        dt_our  = time.perf_counter() - t0

        t0 = time.perf_counter()
        emb_proj = proj_infer(patches)
        dt_proj  = time.perf_counter() - t0

        # --- Aggregate: mean embedding across patches ---
        mean_tf   = np.mean(emb_tf,   axis=0)
        mean_our  = np.mean(emb_our,  axis=0)
        mean_proj = np.mean(emb_proj, axis=0)

        # --- Pairwise cosine similarity ---
        cos_tf_our   = cosine_similarity(mean_tf,  mean_our)
        cos_tf_proj  = cosine_similarity(mean_tf,  mean_proj)
        cos_our_proj = cosine_similarity(mean_our, mean_proj)

        # --- Per-patch max absolute difference ---
        maxd_tf_our   = float(np.max(np.abs(emb_tf  - emb_our)))
        maxd_tf_proj  = float(np.max(np.abs(emb_tf  - emb_proj)))
        maxd_our_proj = float(np.max(np.abs(emb_our - emb_proj)))

        print(f"\n  Pairwise cosine similarity / (1 − cos = cosine distance):")
        print(f"    musicnn TF  ↔  Our ONNX     : {cos_tf_our:.10f}  "
              f"dist={1 - cos_tf_our:.2e}   max_abs_diff={maxd_tf_our:.2e}")
        print(f"    musicnn TF  ↔  Project ONNX  : {cos_tf_proj:.10f}  "
              f"dist={1 - cos_tf_proj:.2e}   max_abs_diff={maxd_tf_proj:.2e}")
        print(f"    Our ONNX    ↔  Project ONNX  : {cos_our_proj:.10f}  "
              f"dist={1 - cos_our_proj:.2e}   max_abs_diff={maxd_our_proj:.2e}")

        print(f"\n  Inference time ({patches.shape[0]} patches):")
        print(f"    musicnn TF:   {dt_tf  * 1000:7.1f} ms")
        print(f"    Our ONNX:     {dt_our * 1000:7.1f} ms")
        print(f"    Project ONNX: {dt_proj * 1000:7.1f} ms")

        all_results.append({
            'song':         name,
            'cos_tf_our':   cos_tf_our,
            'cos_tf_proj':  cos_tf_proj,
            'cos_our_proj': cos_our_proj,
        })

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")

    header = f"{'Song':<55} {'TF↔Our':>10} {'TF↔Proj':>10} {'Our↔Proj':>10}"
    print(f"\n{header}")
    print("-" * 90)

    for r in all_results:
        print(
            f"{r['song']:<55} "
            f"{r['cos_tf_our']:>10.6f} "
            f"{r['cos_tf_proj']:>10.6f} "
            f"{r['cos_our_proj']:>10.6f}"
        )

    avg_tf_our   = np.mean([r['cos_tf_our']   for r in all_results])
    avg_tf_proj  = np.mean([r['cos_tf_proj']  for r in all_results])
    avg_our_proj = np.mean([r['cos_our_proj'] for r in all_results])

    print("-" * 90)
    print(
        f"{'AVERAGE':<55} "
        f"{avg_tf_our:>10.6f} "
        f"{avg_tf_proj:>10.6f} "
        f"{avg_our_proj:>10.6f}"
    )

    # ── Pass / Fail ─────────────────────────────────────────────────────
    THRESHOLD = 0.99
    all_pass = all(
        r['cos_tf_our']   >= THRESHOLD and
        r['cos_tf_proj']  >= THRESHOLD and
        r['cos_our_proj'] >= THRESHOLD
        for r in all_results
    )

    print(f"\nThreshold: cosine similarity ≥ {THRESHOLD}")
    if all_pass:
        print("✓ ALL PASS — embeddings are consistent across all three models!")
    else:
        print("✗ SOME FAILED — details:")
        for r in all_results:
            fails = []
            if r['cos_tf_our']   < THRESHOLD: fails.append(f"TF↔Our ({r['cos_tf_our']:.6f})")
            if r['cos_tf_proj']  < THRESHOLD: fails.append(f"TF↔Proj ({r['cos_tf_proj']:.6f})")
            if r['cos_our_proj'] < THRESHOLD: fails.append(f"Our↔Proj ({r['cos_our_proj']:.6f})")
            if fails:
                print(f"  ✗ {r['song']}: {', '.join(fails)}")

    # ── Cleanup ─────────────────────────────────────────────────────────
    tf_sess.close()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
