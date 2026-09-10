# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Train the product-quantisation codebook of the neural fingerprint.

One-time build tool that reads fingerprint vectors from a database whose
tracks were analysed with the neural fingerprint stage, trains the codebook
that ``tasks/neural_fingerprint.py`` uses to store every 128-number vector as
32 bytes, and writes it as ``neural_fingerprint_pq.npz`` next to the model.
Companion to ``export_neural_fingerprint_to_onnx.py``. The codebook is part
of the stored data: every blob carries its checksum, so it is trained once and
kept, never regenerated for a library that already holds fingerprints.

Main Features:
* Samples up to --sample-rows vectors evenly across the stored tracks, from
  both the int8 legacy blobs and the code blobs (decoded through the codebook
  in use, when retraining from an already quantised library is all there is).
* Runs the same k-means as the runtime module (256 centroids per slice of four
  numbers) and reports the held-out reconstruction cosine.
* Writes the npz with the codebook and its checksum, the id the blobs carry.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)


def default_output() -> str:
    return os.path.join(_REPO_ROOT, 'model', 'neural_fingerprint_pq.npz')


def sample_vectors(dsn: str, sample_rows: int, seed: int):
    import numpy as np
    import psycopg2
    from tasks import neural_fingerprint as nf

    rng = np.random.default_rng(seed)
    conn = psycopg2.connect(dsn, connect_timeout=30)
    conn.set_session(readonly=True)
    try:
        cur = conn.cursor()
        cur.execute('SELECT count(*) FROM embedding WHERE neural_fingerprint IS NOT NULL')
        tracks = int(cur.fetchone()[0])
        if tracks == 0:
            raise SystemExit('No neural fingerprints are stored in that database; run the analysis first.')
        per_track = max(8, sample_rows // tracks)
        cur = conn.cursor(name='codebook_sample')
        cur.itersize = 200
        cur.execute('SELECT neural_fingerprint FROM embedding WHERE neural_fingerprint IS NOT NULL')
        pieces, legacy, quantised = [], 0, 0
        for (blob,) in cur:
            raw = bytes(blob)
            if nf.is_legacy_blob(raw):
                vectors = nf.decode_legacy_blob(raw)
                legacy += 1
            else:
                vectors = nf.decode_blob_f32(raw)
                quantised += 1
            if vectors is None or vectors.shape[0] == 0:
                continue
            take = min(per_track, vectors.shape[0])
            pieces.append(vectors[np.sort(rng.choice(vectors.shape[0], take, replace=False))])
        cur.close()
    finally:
        conn.close()
    print(f'{tracks} tracks ({legacy} int8 legacy, {quantised} already quantised), {per_track} rows each', flush=True)
    return np.concatenate(pieces, axis=0) if pieces else np.zeros((0, nf.DIM), np.float32)


def train_neural_fingerprint_codebook(dsn: str, output_path: str, sample_rows: int, iterations: int, seed: int) -> None:
    import numpy as np
    from tasks import neural_fingerprint as nf

    vectors = sample_vectors(dsn, sample_rows, seed)
    if vectors.shape[0] < 4 * nf.PQ_CENTROIDS:
        raise SystemExit(f'Only {vectors.shape[0]} vectors sampled; at least {4 * nf.PQ_CENTROIDS} are needed.')
    rng = np.random.default_rng(seed)
    order = rng.permutation(vectors.shape[0])
    held = max(nf.PQ_CENTROIDS, vectors.shape[0] // 10)
    train, test = vectors[order[held:]], vectors[order[:held]]
    print(f'Training on {train.shape[0]} vectors, {iterations} iterations per slice...', flush=True)
    book = nf.train_codebook(train, iterations=iterations, seed=seed)
    recon = nf.decode_codes(nf.encode_codes(test, book), book)
    cosine = float(np.einsum('ij,ij->i', recon, test).mean())
    book_id = nf.codebook_id(book)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(output_path, codebook=book, codebook_id=np.uint32(book_id))
    size_kb = os.path.getsize(output_path) / 1024
    print(
        f'Wrote {output_path} ({size_kb:.0f} KB, codebook id {book_id:08x}); '
        f'held-out reconstruction cosine {cosine:.4f} on {test.shape[0]} vectors',
        flush=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', required=True, help='Postgres DSN of a database with neural fingerprints (read only).')
    parser.add_argument('--output', default=default_output(), help='Destination .npz path (default: model/ in the repository).')
    parser.add_argument('--sample-rows', type=int, default=300000, help='Vectors to sample across the stored tracks.')
    parser.add_argument('--iterations', type=int, default=30, help='k-means iterations per slice.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    train_neural_fingerprint_codebook(args.dsn, args.output, args.sample_rows, args.iterations, args.seed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
