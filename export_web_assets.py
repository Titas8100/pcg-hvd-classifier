"""
Export sample audio files for all 5 classes and export trained model weights (MLP + StandardScaler)
to JSON for fast in-browser client-side inference on GitHub Pages.
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

docs_dir = r"f:\ML-PCG-HVD\docs"
samples_dir = os.path.join(docs_dir, "samples")
os.makedirs(samples_dir, exist_ok=True)

# 1. Copy sample audio files for the 5 classes
raw_dir = r"f:\ML-PCG-HVD\data\raw"
samples_map = {
    "N": ("normal_sample.wav", "New_N_001.wav"),
    "AS": ("aortic_stenosis_sample.wav", "New_AS_001.wav"),
    "MR": ("mitral_regurgitation_sample.wav", "New_MR_001.wav"),
    "MS": ("mitral_stenosis_sample.wav", "New_MS_001.wav"),
    "MVP": ("mitral_valve_prolapse_sample.wav", "New_MVP_001.wav")
}

for label, (target_name, src_name) in samples_map.items():
    src_file = os.path.join(raw_dir, label, src_name)
    dst_file = os.path.join(samples_dir, target_name)
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"Copied sample {label} -> {target_name} ({os.path.getsize(dst_file)} bytes)")
    else:
        print(f"Warning: {src_file} not found!")

# 2. Train and export MLP model on the 80-dimensional Enhanced Features
features_path = r"f:\ML-PCG-HVD\data\features\enhanced_80.npy"
labels_path = r"f:\ML-PCG-HVD\data\features\labels.npy"

if os.path.exists(features_path) and os.path.exists(labels_path):
    X = np.load(features_path)
    y = np.load(labels_path)
    print(f"Loaded feature matrix: {X.shape}, labels: {y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train high-accuracy MLP with 2 hidden layers (128, 64)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        max_iter=600,
        random_state=42
    )
    mlp.fit(X_scaled, y)
    train_acc = mlp.score(X_scaled, y) * 100.0
    print(f"Trained in-browser MLP model. Training Accuracy: {train_acc:.2f}%")

    # Serialize weights, biases, and scaler parameters
    model_data = {
        "classes": ["AS", "MR", "MS", "MVP", "N"],
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "layers": [
            {
                "weights": mlp.coefs_[0].tolist(),  # (80, 128)
                "biases": mlp.intercepts_[0].tolist()  # (128,)
            },
            {
                "weights": mlp.coefs_[1].tolist(),  # (128, 64)
                "biases": mlp.intercepts_[1].tolist()  # (64,)
            },
            {
                "weights": mlp.coefs_[2].tolist(),  # (64, 5)
                "biases": mlp.intercepts_[2].tolist()  # (5,)
            }
        ]
    }

    model_json_path = os.path.join(docs_dir, "model_weights.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f)
    print(f"Saved in-browser model weights to {model_json_path} ({os.path.getsize(model_json_path) / 1024:.1f} KB)")
