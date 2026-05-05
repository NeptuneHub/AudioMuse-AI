import onnx
import numpy as np
import psycopg2
import json
from onnx import numpy_helper
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': '192.168.3.208',
    'port': 5432,
    'user': 'audiomuse',
    'password': 'audiomusepassword',
    'dbname': 'audiomusedb',
}

MOOD_ONNX = {
    'happy':      ('mood_happy', 0),
    'sad':        ('mood_sad', 1),
    'aggressive': ('mood_aggressive', 0),
    'party':      ('mood_party', 1),
    'relaxed':    ('mood_relaxed', 1),
    'danceable':  ('danceability', 0),
}
MOODS = list(MOOD_ONNX.keys())

MSD_LABELS = [
    'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists',
    'dance', '00s', 'alternative rock', 'jazz', 'beautiful', 'metal', 'chillout',
    'male vocalists', 'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
    '80s', 'folk', '90s', 'chill', 'instrumental', 'punk', 'oldies', 'blues',
    'hard rock', 'ambient', 'acoustic', 'experimental', 'female vocalist', 'guitar',
    'Hip-Hop', '70s', 'party', 'country', 'easy listening', 'sexy', 'catchy', 'funk',
    'electro', 'heavy metal', 'Progressive rock', '60s', 'rnb', 'indie pop', 'sad',
    'House', 'happy'
]

USE_MUSICNN = False
USE_CLAP = True

SCORE_THRESHOLD = 0.90
CLAP_THRESHOLD = 0.65
MAX_SONGS_PER_MOOD = 500
GMM_K_RANGE = range(2, 100)
OUTPUT_FILE = 'mood_centroids_real_080_clap_test.json'

MOOD_CLAP_LABEL = {
    'happy': 'happy',
    'sad': 'sad',
    'aggressive': 'aggressive',
    'party': 'party',
    'relaxed': 'relaxed',
    'danceable': 'danceable',
}


def load_mood_weights(name):
    model = onnx.load(f'model/{name}-msd-musicnn-1.onnx')
    w = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    return (w['dense/kernel/read:0'].copy(), w['dense/bias/read:0'].copy(),
            w['dense_1/kernel/read:0'].copy(), w['dense_1/bias/read:0'].copy())


def load_prediction_weights():
    model = onnx.load('model/musicnn_prediction.onnx')
    w = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    return {
        'W': w['dense_1_kernel'].copy(),
        'b': w['dense_1_bias'].copy(),
        'bn_scale': w['bn_scale'].copy(),
        'bn_bias': w['bn_bias'].copy(),
        'bn_mean': w['bn_mean'].copy(),
        'bn_var': w['bn_var'].copy(),
    }


def forward_mood_batch(X, W1, b1, W2, b2, positive_idx):
    h = X @ W1 + b1
    h = np.maximum(h, 0)
    logits = h @ W2 + b2
    exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return probs[:, positive_idx]


def forward_mood_single(x, W1, b1, W2, b2, positive_idx):
    h = x @ W1 + b1
    h = np.maximum(h, 0)
    logits = h @ W2 + b2
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()
    return probs[positive_idx]


def predict_msd_tags(x, pred_w):
    h = np.maximum(x, 0)
    h = (h - pred_w['bn_mean']) / np.sqrt(pred_w['bn_var'] + 1e-5)
    h = h * pred_w['bn_scale'] + pred_w['bn_bias']
    logits = h @ pred_w['W'] + pred_w['b']
    return 1.0 / (1.0 + np.exp(-logits))


def load_all_embeddings():
    print("Loading all embeddings from DB...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT item_id, embedding FROM embedding")

    item_ids = []
    embeddings = []
    for item_id, emb_bytes in cur:
        vec = np.frombuffer(bytes(emb_bytes), dtype=np.float32)
        if vec.shape[0] == 200:
            item_ids.append(item_id)
            embeddings.append(vec)

    cur.close()
    conn.close()

    X = np.stack(embeddings)
    print(f"  Loaded {X.shape[0]} embeddings, shape {X.shape}\n")
    return item_ids, X


def load_clap_scores():
    print("Loading CLAP scores from DB (other_features)...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT item_id, other_features FROM score WHERE other_features IS NOT NULL")

    clap_scores = {}
    for item_id, feat_str in cur:
        feats = {}
        for pair in feat_str.split(','):
            parts = pair.split(':')
            if len(parts) == 2:
                feats[parts[0].strip()] = float(parts[1])
        clap_scores[item_id] = feats

    cur.close()
    conn.close()
    print(f"  Loaded CLAP scores for {len(clap_scores)} songs\n")
    return clap_scores


def fit_gmm_bic(X, k_range=GMM_K_RANGE):
    X64 = X.astype(np.float64)
    best_bic = np.inf
    best_gmm = None
    best_k = -1
    bic_values = {}

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='diag',
                               n_init=3, max_iter=300, reg_covar=1e-5,
                               random_state=42)
        gmm.fit(X64)
        bic = gmm.bic(X64)
        bic_values[k] = bic
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    return best_gmm, best_k, bic_values


def main():
    item_ids, all_X = load_all_embeddings()

    clap_scores = load_clap_scores() if USE_CLAP else {}
    id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    pred_w = load_prediction_weights()
    print("Loaded MSD prediction weights")

    mood_weights = {}
    for mood, (onnx_name, pi) in MOOD_ONNX.items():
        mood_weights[mood] = (*load_mood_weights(onnx_name), pi)
    print(f"Loaded {len(MOODS)} mood ONNX models\n")

    results = {}

    for mood in MOODS:
        onnx_name, pi = MOOD_ONNX[mood]
        W1, b1, W2, b2 = mood_weights[mood][:4]

        print("=" * 70)
        print(f"  {mood.upper()}: Processing {len(all_X)} songs "
              f"(MusiCNN={'ON' if USE_MUSICNN else 'OFF'}, CLAP={'ON' if USE_CLAP else 'OFF'})")
        print("=" * 70)

        scores = forward_mood_batch(all_X, W1, b1, W2, b2, pi)

        if USE_MUSICNN:
            mask = scores >= SCORE_THRESHOLD
            above_idx = np.where(mask)[0]
            n_musicnn = len(above_idx)
            print(f"  Stage 1 — MusiCNN > {SCORE_THRESHOLD}: {n_musicnn} songs "
                  f"({100*n_musicnn/len(all_X):.1f}% of library)")
        else:
            above_idx = np.arange(len(all_X))
            print(f"  Stage 1 — MusiCNN OFF: keeping all {len(above_idx)} songs")

        if USE_CLAP:
            clap_label = MOOD_CLAP_LABEL[mood]
            clap_pass = []
            for idx in above_idx:
                iid = item_ids[idx]
                if iid in clap_scores:
                    clap_val = clap_scores[iid].get(clap_label, 0.0)
                    if clap_val >= CLAP_THRESHOLD:
                        clap_pass.append(idx)
            clap_pass = np.array(clap_pass)
            n_pass = len(clap_pass)
            print(f"  Stage 2 — CLAP '{clap_label}' > {CLAP_THRESHOLD}: {n_pass} songs remain")
        else:
            clap_pass = above_idx
            n_pass = len(clap_pass)
            print(f"  Stage 2 — CLAP OFF: keeping {n_pass} songs")

        if n_pass > MAX_SONGS_PER_MOOD:
            top_idx = clap_pass[np.argsort(scores[clap_pass])[-MAX_SONGS_PER_MOOD:]]
            sel_X = all_X[top_idx]
            sel_scores = scores[top_idx]
            print(f"  Capped to top {MAX_SONGS_PER_MOOD} by MusiCNN score")
        else:
            sel_X = all_X[clap_pass] if n_pass > 0 else np.array([])
            sel_scores = scores[clap_pass] if n_pass > 0 else np.array([])

        if len(sel_X) > 0:
            print(f"  Final: {len(sel_X)} songs, "
                  f"score range: {sel_scores.min():.4f} – {sel_scores.max():.4f}")

        if len(sel_X) < 10:
            print(f"  Too few songs, skipping {mood}")
            continue

        print(f"  Fitting GMM (K={GMM_K_RANGE.start}..{GMM_K_RANGE.stop-1}) "
              f"on {len(sel_X)} songs...")
        best_gmm, best_k, bic_values = fit_gmm_bic(sel_X)
        print(f"  Best K by BIC: {best_k}")

        sorted_bic = sorted(bic_values.items(), key=lambda x: x[1])
        print(f"  Top-5 K by BIC:")
        for k, bic in sorted_bic[:5]:
            print(f"    K={k:2d}  BIC={bic:,.0f}")

        labels = best_gmm.predict(sel_X.astype(np.float64))
        centroids = best_gmm.means_

        print(f"\n  Cluster sizes:")
        for c in range(best_k):
            cmask = labels == c
            count = cmask.sum()
            mean_score = sel_scores[cmask].mean()
            print(f"    Cluster {c:2d}: {count:5d} songs, "
                  f"mean {mood} score: {mean_score:.4f}")

        print(f"\n  Centroid MSD tag profiles (top-8 tags):")
        print(f"  {'#':>3}  {'songs':>5}  {'mood_score':>10}  top MSD tags")
        print(f"  {'─'*75}")

        centroid_data = []
        for c in range(best_k):
            cmask = labels == c
            count = cmask.sum()
            cscore = forward_mood_single(centroids[c], W1, b1, W2, b2, pi)

            tag_probs = predict_msd_tags(centroids[c], pred_w)
            top8_idx = np.argsort(tag_probs)[-8:][::-1]
            tag_str = ", ".join(
                f"{MSD_LABELS[i]}({tag_probs[i]:.2f})" for i in top8_idx
            )
            print(f"  {c:3d}  {count:5d}  {cscore:10.4f}  {tag_str}")

            centroid_data.append({
                'cluster_id': int(c),
                'n_songs': int(count),
                'mood_score': float(cscore),
                'centroid': centroids[c].tolist(),
                'top_tags': {MSD_LABELS[i]: float(tag_probs[i]) for i in top8_idx},
            })

        print(f"\n  Cross-mood scores of {mood} centroids:")
        header = f"  {'#':>3} |" + "".join(
            f"  {m:>10}" for m in MOODS
        )
        print(header)
        for c in range(best_k):
            row = f"  {c:3d} |"
            for m2 in MOODS:
                W1b, b1b, W2b, b2b = mood_weights[m2][:4]
                pi2 = mood_weights[m2][4]
                s = forward_mood_single(centroids[c], W1b, b1b, W2b, b2b, pi2)
                row += f"  {s:10.3f}"
            print(row)

        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        normed = centroids / (norms + 1e-12)
        cos_matrix = normed @ normed.T
        np.fill_diagonal(cos_matrix, 0)
        upper = cos_matrix[np.triu_indices(best_k, k=1)]
        if len(upper) > 0:
            print(f"\n  Pairwise cosine between centroids: "
                  f"min={upper.min():.3f} mean={upper.mean():.3f} max={upper.max():.3f}")

        results[mood] = {
            'best_k': best_k,
            'bic_values': {str(k): float(v) for k, v in bic_values.items()},
            'centroids': centroid_data,
        }
        print()

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mood in MOODS:
        if mood not in results:
            continue
        r = results[mood]
        n_songs = sum(c['n_songs'] for c in r['centroids'])
        print(f"  {mood:15s}: K={r['best_k']:2d} clusters, {n_songs:,d} songs")


if __name__ == '__main__':
    main()
