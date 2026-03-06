"""
Export the MSD-MusiCNN **classification head** from jordipons/musicnn to ONNX.

Standalone model:  embedding (batch, 200)  →  50-dim raw logits (batch, 50)
Architecture:      ReLU → BatchNorm(batch_normalization_10) → MatMul + BiasAdd

(Dropout is a no-op at inference time, so we skip it.)

This model is meant to be used together with musicnn_embedding.onnx:
  1. musicnn_embedding.onnx   :  mel (batch,187,96)  →  embedding (batch,200)
  2. musicnn_prediction.onnx  :  embedding (batch,200)  →  logits (batch,50)

Usage:
    python export_prediction_onnx.py
"""

import os
import sys
import glob
import numpy as np


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_SONGS   = os.path.join(PROJECT_ROOT, 'test', 'songs')

OUR_EMBED_ONNX  = os.path.join(SCRIPT_DIR, 'musicnn_embedding.onnx')
PROJ_EMBED_ONNX = os.path.join(PROJECT_ROOT, 'test', 'models', 'msd-musicnn-1.onnx')
PROJ_PRED_ONNX  = os.path.join(PROJECT_ROOT, 'model', 'msd-msd-musicnn-1.onnx')
if not os.path.isfile(PROJ_PRED_ONNX):
    PROJ_PRED_ONNX = os.path.join(PROJECT_ROOT, 'test', 'models', 'msd-msd-musicnn-1.onnx')

SR, N_FFT, HOP_LENGTH, N_MELS, FRAME_SIZE = 16000, 512, 256, 96, 187


def preprocess_audio(file_path):
    import librosa
    audio, sr = librosa.load(file_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, window='hann', center=False,
        power=2.0, norm='slaney', htk=False,
    )
    log_mel = np.log10(1 + 10000 * mel).T
    patches = []
    for i in range(0, log_mel.shape[0] - FRAME_SIZE + 1, FRAME_SIZE):
        patches.append(log_mel[i : i + FRAME_SIZE])
    if not patches:
        padded = np.zeros((FRAME_SIZE, N_MELS), dtype=np.float32)
        padded[: log_mel.shape[0]] = log_mel
        patches.append(padded)
    return np.array(patches, dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def cosine_similarity(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def export_prediction_to_onnx(output_path):
    """
    Extract dense_1 kernel+bias and batch_normalization_10 params from the
    musicnn TF checkpoint and build a standalone ONNX model:
      (batch,200) → ReLU → BatchNorm → dense_1 → (batch,50)
    """
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import onnxruntime as ort
    from tensorflow.core.protobuf import meta_graph_pb2

    # ── 1. Locate checkpoint ────────────────────────────────────────────
    import musicnn
    musicnn_pkg = os.path.dirname(musicnn.__file__)
    model_dir = os.path.join(musicnn_pkg, 'MSD_musicnn')
    if not os.path.isdir(model_dir):
        model_dir = os.path.join(musicnn_pkg, 'trained_models', 'MSD_musicnn')
    if not os.path.isdir(model_dir):
        print(f"✗ Cannot find MSD_musicnn in {musicnn_pkg}")
        sys.exit(1)

    metas = [f for f in os.listdir(model_dir) if f.endswith('.meta')]
    meta_file   = os.path.join(model_dir, metas[0])
    ckpt_prefix = meta_file[: -len('.meta')]
    print(f"Checkpoint : {ckpt_prefix}")

    # ── 2. Load TF model and extract weights ────────────────────────────
    print("\nLoading TF model …")
    mg = meta_graph_pb2.MetaGraphDef()
    with open(meta_file, 'rb') as f:
        mg.ParseFromString(f.read())
    for node in mg.graph_def.node:
        if '_output_shapes' in node.attr:
            del node.attr['_output_shapes']

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.train.import_meta_graph(mg)
        tf.compat.v1.train.Saver().restore(sess, ckpt_prefix)

        kernel = sess.run('dense_1/kernel:0')   # (200, 50)
        bias   = sess.run('dense_1/bias:0')     # (50,)
        bn_gamma = sess.run('batch_normalization_10/gamma:0')          # (200,)
        bn_beta  = sess.run('batch_normalization_10/beta:0')           # (200,)
        bn_mean  = sess.run('batch_normalization_10/moving_mean:0')    # (200,)
        bn_var   = sess.run('batch_normalization_10/moving_variance:0')  # (200,)
        print(f"  kernel: {kernel.shape}  bias: {bias.shape}")
        print(f"  BN10 gamma: {bn_gamma.shape}  beta: {bn_beta.shape}  "
              f"mean: {bn_mean.shape}  var: {bn_var.shape}")

        # Get reference: full TF pipeline for verification
        rng   = np.random.RandomState(42)
        dummy_mel = rng.randn(2, 187, 96).astype(np.float32)

        tf_embed, tf_pred = sess.run(
            ['model/dense/BiasAdd:0', 'model/dense_1/BiasAdd:0'],
            feed_dict={
                'model/Placeholder:0':   dummy_mel,
                'model/Placeholder_2:0': False,
                'model/Placeholder_1:0': np.zeros((2, 50), dtype=np.float32),
            },
        )
        print(f"  TF ref: embed {tf_embed.shape} → pred {tf_pred.shape}")

    # ── 3. Build standalone ONNX ────────────────────────────────────────
    # Architecture: (batch,200) → ReLU → BatchNorm → MatMul + BiasAdd → (batch,50)
    print("\nBuilding standalone ONNX classification head …")

    X = helper.make_tensor_value_info(
        'serving_default_model_Placeholder:0', TensorProto.FLOAT, [None, 200],
    )
    Y = helper.make_tensor_value_info(
        'PartitionedCall:0', TensorProto.FLOAT, [None, 50],
    )

    # Initializers
    W     = numpy_helper.from_array(kernel.astype(np.float32), name='dense_1_kernel')
    B     = numpy_helper.from_array(bias.astype(np.float32),   name='dense_1_bias')
    scale = numpy_helper.from_array(bn_gamma.astype(np.float32), name='bn_scale')
    bn_b  = numpy_helper.from_array(bn_beta.astype(np.float32),  name='bn_bias')
    mean  = numpy_helper.from_array(bn_mean.astype(np.float32),  name='bn_mean')
    var   = numpy_helper.from_array(bn_var.astype(np.float32),   name='bn_var')

    # Nodes
    relu_node = helper.make_node(
        'Relu', ['serving_default_model_Placeholder:0'], ['relu_out'],
    )
    bn_node = helper.make_node(
        'BatchNormalization',
        inputs=['relu_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
        outputs=['bn_out'],
        epsilon=0.001,  # TF default
    )
    matmul_node = helper.make_node('MatMul', ['bn_out', 'dense_1_kernel'], ['matmul_out'])
    add_node    = helper.make_node('Add',    ['matmul_out', 'dense_1_bias'], ['PartitionedCall:0'])

    graph = helper.make_graph(
        [relu_node, bn_node, matmul_node, add_node],
        'musicnn_prediction',
        [X], [Y],
        initializer=[W, B, scale, bn_b, mean, var],
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model_proto.ir_version = 7

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✓ Saved: {output_path}  ({size_kb:.1f} KB)")

    # ── 4. Verify vs TF reference ──────────────────────────────────────
    print("\nVerifying …")
    ort_sess = ort.InferenceSession(output_path)
    for i in ort_sess.get_inputs():
        print(f"  Input:  {i.name}  shape={i.shape}")
    for o in ort_sess.get_outputs():
        print(f"  Output: {o.name}  shape={o.shape}")

    # Feed the raw embedding (BiasAdd output) — our model applies ReLU+BN internally
    onnx_pred = ort_sess.run(None, {
        'serving_default_model_Placeholder:0': tf_embed,
    })[0]

    max_diff = float(np.max(np.abs(tf_pred - onnx_pred)))
    cos_sim  = float(
        np.dot(tf_pred.flatten(), onnx_pred.flatten())
        / (np.linalg.norm(tf_pred) * np.linalg.norm(onnx_pred))
    )
    print(f"\n  vs TF full pipeline: max_diff={max_diff:.2e}  cos_sim={cos_sim:.10f}")
    ok = max_diff < 1e-4
    print(f"  {'✓ Export verified!' if ok else '✗ Difference too large!'}")
    return output_path


def compare_on_audio():
    """Compare our embed+pred pipeline vs project's embed+pred pipeline on real audio."""
    import onnxruntime as ort

    our_pred_path = os.path.join(SCRIPT_DIR, 'musicnn_prediction.onnx')

    missing = []
    for label, path in [
        ("musicnn_prediction.onnx", our_pred_path),
        ("musicnn_embedding.onnx",  OUR_EMBED_ONNX),
        ("project embedding",       PROJ_EMBED_ONNX),
        ("project prediction",      PROJ_PRED_ONNX),
    ]:
        if not os.path.isfile(path):
            missing.append(f"  {label}: {path}")
    if missing:
        print("\n✗ Missing:\n" + "\n".join(missing))
        return

    songs = sorted(
        p for ext in ('*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a')
        for p in glob.glob(os.path.join(TEST_SONGS, ext))
    )
    if not songs:
        print("✗ No audio files in test/songs/")
        return

    # Our pipeline: embed → pred (raw logits → sigmoid)
    our_embed_sess = ort.InferenceSession(OUR_EMBED_ONNX)
    our_pred_sess  = ort.InferenceSession(our_pred_path)
    our_embed_in   = our_embed_sess.get_inputs()[0].name
    our_pred_in    = our_pred_sess.get_inputs()[0].name

    # Project pipeline: embed → pred (sigmoid built in)
    proj_embed_sess = ort.InferenceSession(PROJ_EMBED_ONNX)
    proj_pred_sess  = ort.InferenceSession(PROJ_PRED_ONNX)
    proj_embed_in   = proj_embed_sess.get_inputs()[0].name
    proj_pred_in    = proj_pred_sess.get_inputs()[0].name

    print("\n" + "=" * 90)
    print("  COMPARISON: musicnn head vs Essentia head (both using same embeddings)")
    print("=" * 90)
    print()
    print("  musicnn   : embedding → ReLU → BatchNorm → dense_1 → logits → sigmoid")
    print("  Essentia  : embedding → ReLU → BatchNorm → dense_1 → Sigmoid")
    print("  (Same architecture, different trained weights)\n")

    all_results = []
    for song_path in songs:
        name = os.path.basename(song_path)
        print(f"{'─' * 90}")
        print(f"Song: {name}")

        patches = preprocess_audio(song_path)
        print(f"  Patches: {patches.shape[0]}")

        # Our pipeline
        our_emb    = our_embed_sess.run(None, {our_embed_in: patches})[0]
        our_logits = our_pred_sess.run(None, {our_pred_in: our_emb})[0]
        our_probs  = sigmoid(np.mean(our_logits, axis=0))

        # Project pipeline
        proj_emb   = proj_embed_sess.run(None, {proj_embed_in: patches})[0]
        proj_probs = np.mean(proj_pred_sess.run(None, {proj_pred_in: proj_emb})[0], axis=0)

        cos  = cosine_similarity(our_probs, proj_probs)
        maxd = float(np.max(np.abs(our_probs - proj_probs)))

        top5_ours = np.argsort(our_probs)[::-1][:5]
        top5_proj = np.argsort(proj_probs)[::-1][:5]
        overlap   = len(set(top5_ours) & set(top5_proj))

        print(f"  Cosine similarity (probs): {cos:.6f}")
        print(f"  Max abs diff (probs):      {maxd:.4f}")
        print(f"  Top-5 tag overlap:         {overlap}/5")

        all_results.append({'song': name, 'cos': cos, 'overlap': overlap})

    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    for r in all_results:
        print(f"  {r['song']:<55} cos={r['cos']:.6f}  top5={r['overlap']}/5")
    avg_cos = np.mean([r['cos'] for r in all_results])
    print(f"\n  Average cosine similarity: {avg_cos:.6f}")
    print("  (Same architecture, different trained weights — some difference expected)")


if __name__ == '__main__':
    dest = os.path.join(SCRIPT_DIR, 'musicnn_prediction.onnx')
    export_prediction_to_onnx(dest)
    compare_on_audio()
