"""
Export GBDT model (70 trees, depth 3) and full 80-feature extraction parameters to JSON
and generate a standalone JavaScript inference engine.
"""

import os
import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from src.features_mfcc import get_mel_filterbank
import pywt
from scipy.fftpack import dct

# 1. Load dataset features and labels
features_path = r"data/features/enhanced_80.npy"
labels_path = r"data/features/labels.npy"

X = np.load(features_path)
y = np.load(labels_path)
classes = ["AS", "MR", "MS", "MVP", "N"]

print(f"Training GBDT on {X.shape[0]} samples with 80 features...")
gb = GradientBoostingClassifier(n_estimators=70, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X, y)
train_acc = gb.score(X, y) * 100.0
print(f"GBDT Training Accuracy: {train_acc:.2f}%")

# 2. Extract trees in compact flat format
# For each stage m in 0..69, for each class k in 0..4:
trees_export = []
for m in range(gb.n_estimators_):
    stage_trees = []
    for k in range(len(classes)):
        t = gb.estimators_[m, k].tree_
        # Export as parallel arrays or array of nodes
        n_nodes = t.node_count
        tree_obj = {
            "f": t.feature.tolist(),
            "th": [round(float(v), 5) for v in t.threshold],
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "v": [round(float(v[0, 0]), 5) for v in t.value]
        }
        stage_trees.append(tree_obj)
    trees_export.append(stage_trees)

# 3. Export Filterbanks and Constants
# Mel filterbank: 26 filters x 257 bins
fbank = get_mel_filterbank(num_filters=26, n_fft=512, sample_rate=8000, low_freq_hz=20.0, high_freq_hz=4000.0)
# Round to 5 decimals to save space
fbank_list = [[round(float(val), 5) for val in row] for row in fbank]

# DCT-II matrix: 19 x 26
N = 26
dct_mat = np.zeros((19, N), dtype=np.float32)
for k in range(19):
    for n in range(N):
        factor = np.sqrt(1.0 / (4.0 * N)) if k == 0 else np.sqrt(1.0 / (2.0 * N))
        dct_mat[k, n] = 2.0 * factor * np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * N))
dct_list = [[round(float(val), 5) for val in row] for row in dct_mat]

# DWT db4 filters
w = pywt.Wavelet("db4")
dec_lo = [round(float(v), 8) for v in w.dec_lo]
dec_hi = [round(float(v), 8) for v in w.dec_hi]

# Class priors / init
priors = gb.init_.class_prior_
init_raw = np.log(priors).tolist()

model_payload = {
    "classes": classes,
    "init_raw": [round(float(v), 5) for v in init_raw],
    "learning_rate": float(gb.learning_rate),
    "n_estimators": int(gb.n_estimators_),
    "trees": trees_export,
    "fbank": fbank_list,
    "dct": dct_list,
    "dwt_h": dec_lo,
    "dwt_g": dec_hi
}

# Save to docs and root
for target_dir in ["docs", "."]:
    out_path = os.path.join(target_dir, "model_weights.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f)
    print(f"Saved {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

# Also save as inline JS export so app.js can load it with zero CORS/file:// issues
js_model_code = f"const PCG_MODEL_DATA = {json.dumps(model_payload)};\nif (typeof module !== 'undefined') module.exports = PCG_MODEL_DATA;\n"
for target_dir in ["docs", "."]:
    out_js = os.path.join(target_dir, "model_data.js")
    with open(out_js, "w", encoding="utf-8") as f:
        f.write(js_model_code)
    print(f"Saved {out_js} ({os.path.getsize(out_js) / 1024:.1f} KB)")
