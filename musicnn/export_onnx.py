"""
Export the MSD-MusiCNN embedding model from jordipons/musicnn to ONNX.

Exports ONLY the embedding (penultimate layer, 200-dim),
NOT the 50-class classification head.

Model input:  (batch, 187, 96)  float32 — mel spectrogram patches
Model output: (batch, 200)      float32 — embedding vectors

Usage:
    python export_onnx.py
"""

import os
import sys
import numpy as np


def export_musicnn_to_onnx(output_path):
    """
    1. Load TF checkpoint from the musicnn library
    2. Freeze graph (variables → constants)
    3. Replace is_training placeholder with constant False
    4. Convert to ONNX via tf2onnx
    5. Verify the exported model
    """
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    import tf2onnx
    import onnx
    import onnxruntime as ort

    # ── 1. Locate musicnn TF checkpoint ─────────────────────────────────
    import musicnn
    musicnn_pkg = os.path.dirname(musicnn.__file__)

    # The wheel ships checkpoints as bare dotfiles (.meta, .index, .data-*)
    # directly inside  musicnn/MSD_musicnn/  (no "trained_models" subdir,
    # no filename prefix).
    model_dir = os.path.join(musicnn_pkg, 'MSD_musicnn')
    if not os.path.isdir(model_dir):
        # Fallback: some forks put them under trained_models/
        model_dir = os.path.join(musicnn_pkg, 'trained_models', 'MSD_musicnn')

    if not os.path.isdir(model_dir):
        print(f"✗ Cannot find MSD_musicnn directory in {musicnn_pkg}")
        sys.exit(1)

    # Discover the .meta file (could be ".meta" or "something.ckpt.meta")
    meta_candidates = [f for f in os.listdir(model_dir) if f.endswith('.meta')]
    if not meta_candidates:
        print(f"✗ No .meta files in {model_dir}")
        print(f"  Contents: {os.listdir(model_dir)}")
        sys.exit(1)

    meta_file   = os.path.join(model_dir, meta_candidates[0])
    # Checkpoint prefix = meta path without the ".meta" suffix
    ckpt_prefix = meta_file[: -len('.meta')]

    print(f"Checkpoint : {ckpt_prefix}")
    print(f"Meta graph : {meta_file}")

    # Tensor names inside the musicnn TF graph
    INPUT_TENSOR   = 'model/Placeholder:0'       # (batch, 187, 96) mel patches
    LABELS_TENSOR  = 'model/Placeholder_1:0'     # (batch, 50) labels — unused
    ISTRAIN_TENSOR = 'model/Placeholder_2:0'     # bool scalar — is_training
    EMBED_TENSOR   = 'model/dense/BiasAdd:0'     # (batch, 200) embedding

    # ── 2. Load model & get reference output ────────────────────────────
    print("\nLoading TensorFlow model …")
    tf.compat.v1.reset_default_graph()

    # The .meta graph saved by the original musicnn training code contains
    # gradient nodes (batch_normalization/cond/FusedBatchNorm_1_grad) whose
    # _output_shapes attributes are invalid in TF2.  We strip them before
    # importing so import_meta_graph succeeds.
    from tensorflow.core.protobuf import meta_graph_pb2

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    with open(meta_file, 'rb') as f:
        meta_graph_def.ParseFromString(f.read())

    for node in meta_graph_def.graph_def.node:
        if '_output_shapes' in node.attr:
            del node.attr['_output_shapes']

    with tf.compat.v1.Session() as sess:
        tf.compat.v1.train.import_meta_graph(meta_graph_def)
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, ckpt_prefix)

        # Reference inference with a deterministic dummy input
        rng   = np.random.RandomState(42)
        dummy = rng.randn(1, 187, 96).astype(np.float32)
        tf_out = sess.run(
            EMBED_TENSOR,
            feed_dict={
                INPUT_TENSOR: dummy,
                ISTRAIN_TENSOR: False,
                LABELS_TENSOR: np.zeros((1, 50), dtype=np.float32),
            },
        )
        print(f"  TF reference: {dummy.shape} → {tf_out.shape}")

        # Freeze: convert trainable variables to constants
        embed_node = EMBED_TENSOR.split(':')[0]      # 'model/dense/BiasAdd'
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, [embed_node],
        )
        print(f"  Frozen graph: {len(frozen.node)} nodes")

    # ── 3. Replace unused placeholders with constants ─────────────────
    #   - model/Placeholder_1  (labels, shape None×50) → zeros
    #   - model/Placeholder_2  (is_training, bool)     → False
    print("\nPatching graph: removing unused placeholders …")
    from tensorflow.core.framework import graph_pb2, types_pb2, tensor_pb2

    is_train_node_name = ISTRAIN_TENSOR.split(':')[0]   # 'model/Placeholder_2'
    labels_node_name   = LABELS_TENSOR.split(':')[0]    # 'model/Placeholder_1'
    patched  = graph_pb2.GraphDef()
    replaced_count = 0

    for node in frozen.node:
        if node.name == is_train_node_name:
            c = patched.node.add()
            c.name = node.name
            c.op   = 'Const'
            c.attr['dtype'].type               = types_pb2.DT_BOOL
            c.attr['value'].tensor.dtype        = types_pb2.DT_BOOL
            c.attr['value'].tensor.bool_val.append(False)
            replaced_count += 1
        elif node.name == labels_node_name:
            # The labels placeholder is never consumed in the embedding
            # subgraph, but freeze keeps it.  Replace with a harmless const.
            c = patched.node.add()
            c.name = node.name
            c.op   = 'Const'
            c.attr['dtype'].type               = types_pb2.DT_FLOAT
            c.attr['value'].tensor.dtype        = types_pb2.DT_FLOAT
            from tensorflow.core.framework import tensor_shape_pb2
            dim = c.attr['value'].tensor.tensor_shape.dim.add()
            dim.size = 0
            replaced_count += 1
        else:
            patched.node.add().CopyFrom(node)

    print(f"  Replaced {replaced_count} placeholder(s)")

    # ── 4. Convert to ONNX ──────────────────────────────────────────────
    print("\nConverting to ONNX (opset 13) …")
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.import_graph_def(patched, name='')

    # Use tf2onnx.convert.from_graph_def (works with newer tf2onnx versions)
    model_proto, _ = tf2onnx.convert.from_graph_def(
        patched,
        input_names=[INPUT_TENSOR],
        output_names=[EMBED_TENSOR],
        opset=13,
    )

    onnx.save(model_proto, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Saved: {output_path}  ({size_mb:.1f} MB)")

    # ── 5. Verify ONNX vs original TF ──────────────────────────────────
    print("\nVerifying ONNX model …")
    ort_sess = ort.InferenceSession(output_path)

    for inp in ort_sess.get_inputs():
        print(f"  Input:  {inp.name}  shape={inp.shape}  type={inp.type}")
    for out in ort_sess.get_outputs():
        print(f"  Output: {out.name}  shape={out.shape}  type={out.type}")

    onnx_input_name = ort_sess.get_inputs()[0].name
    onnx_out = ort_sess.run(None, {onnx_input_name: dummy})[0]

    max_diff = float(np.max(np.abs(tf_out - onnx_out)))
    cos_sim  = float(
        np.dot(tf_out.flatten(), onnx_out.flatten())
        / (np.linalg.norm(tf_out) * np.linalg.norm(onnx_out))
    )

    print(f"\n  Max abs diff (TF vs ONNX): {max_diff:.2e}")
    print(f"  Cosine similarity:         {cos_sim:.10f}")
    ok = max_diff < 1e-4
    print(f"  {'✓ Export verified!' if ok else '✗ Difference too large!'}")

    return output_path


if __name__ == '__main__':
    dest = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'musicnn_embedding.onnx',
    )
    export_musicnn_to_onnx(dest)
