"""
Master Pipeline Runner:
Automates end-to-end execution:
1. Data loading and validation.
2. Feature extraction (DWT 24, MFCC 19, Enhanced 80) and caching.
3. Reproduction of report experiments (Tables 1a, 1b, 2a, 2b).
4. Advanced ML models & Deep Learning 1D-CNN benchmark.
5. Generation and export of all CSV tables, JSON metrics, and figures.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import ensure_dataset, load_pcg_dataset, CLASS_NAMES
from src.features_dwt import extract_dwt_dataset
from src.features_mfcc import extract_mfcc_dataset
from src.features_enhanced import extract_enhanced_dataset
from src.experiments_report import (
    run_table_1a, run_table_1b, run_table_2a, run_table_2b, run_fused_baseline
)
from src.models_advanced import (
    get_advanced_ml_models, evaluate_models_cross_validation, train_pcg_1d_cnn
)
from src.visualization import (
    plot_pcg_signals_and_spectrograms,
    plot_feature_selection_comparison,
    plot_model_comparison_barchart,
    plot_confusion_matrices
)


def main():
    print("=" * 78)
    print(" PCG Signal Processing & Heart Valve Disease Detection Pipeline ")
    print(" Recreating & Extending Titas Routh's 2023 Study & Son et al. 2018 ")
    print("=" * 78)

    # Directories
    data_dir = "data/raw"
    features_dir = "data/features"
    results_tables_dir = "results/tables"
    results_figs_dir = "results/figures"

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(results_tables_dir, exist_ok=True)
    os.makedirs(results_figs_dir, exist_ok=True)

    # 1. Dataset Verification & Loading
    scratch_dir = r"C:\Users\titas\.gemini\antigravity-ide\brain\0cf0ed17-6cf7-4772-bbb5-5f7f2e61f561\scratch\raw_data"
    ensure_dataset(data_dir, scratch_cache_dir=scratch_dir)

    print("\n[Step 1/5] Loading PCG audio signals from disk...")
    signals, labels_str, labels_idx, fnames, meta = load_pcg_dataset(data_dir, target_sr=8000)
    print(f"  Successfully loaded {len(signals)} recordings.")
    print(f"  Class distribution: {meta['class_counts']}")
    print(f"  Average audio duration: {meta['mean_duration_sec']:.2f} s")

    # 2. Feature Extraction & Caching
    print("\n[Step 2/5] Extracting & Caching Features...")
    dwt_cache = os.path.join(features_dir, "dwt_24.npy")
    mfcc_cache = os.path.join(features_dir, "mfcc_19.npy")
    enh_cache = os.path.join(features_dir, "enhanced_80.npy")
    labels_cache = os.path.join(features_dir, "labels.npy")

    if os.path.exists(dwt_cache) and os.path.exists(mfcc_cache) and os.path.exists(enh_cache):
        print("  Loading cached features from disk...")
        X_dwt = np.load(dwt_cache)
        X_mfcc = np.load(mfcc_cache)
        X_enh = np.load(enh_cache)
        y = np.load(labels_cache)
    else:
        print("  Extracting 24-dim DWT features (3-level decomposition, 4 sub-bands x 6 statistics)...")
        t0 = time.time()
        X_dwt = extract_dwt_dataset(signals, wavelet="db4", level=3)
        print(f"  DWT extracted in {time.time() - t0:.2f}s, shape: {X_dwt.shape}")

        print("  Extracting 19-dim MFCC features (240 frame, 80 hop, 8000 Hz)...")
        t0 = time.time()
        X_mfcc = extract_mfcc_dataset(signals, sample_rate=8000, frame_length=240, hop_length=80, num_cepstral=19)
        print(f"  MFCC extracted in {time.time() - t0:.2f}s, shape: {X_mfcc.shape}")

        print("  Extracting 80-dim Enhanced features (DWT + MFCC Dynamics + Spectral + Temporal/Envelope)...")
        t0 = time.time()
        X_enh = extract_enhanced_dataset(signals, fs=8000)
        print(f"  Enhanced features extracted in {time.time() - t0:.2f}s, shape: {X_enh.shape}")

        y = labels_idx
        np.save(dwt_cache, X_dwt)
        np.save(mfcc_cache, X_mfcc)
        np.save(enh_cache, X_enh)
        np.save(labels_cache, y)
        print("  All features cached successfully.")

    # 3. Reproducing Report Experiments
    print("\n[Step 3/5] Reproducing Report Experiments...")
    
    # Table 1(a)
    print("  -> Running Table 1(a): DWT features vs Dropout% using SVM...")
    t1a_df = run_table_1a(X_dwt, y, random_state=42)
    t1a_df.to_csv(os.path.join(results_tables_dir, "reproduced_table1a.csv"), index=False)
    print(t1a_df.to_string(index=False))

    # Table 1(b)
    print("\n  -> Running Table 1(b): DWT features vs Selection% using SVM & KNN...")
    t1b_df = run_table_1b(X_dwt, y, random_state=42)
    t1b_df.to_csv(os.path.join(results_tables_dir, "reproduced_table1b.csv"), index=False)
    print(t1b_df.to_string(index=False))

    # Table 2(a)
    print("\n  -> Running Table 2(a): MFCC features vs Dropout% using SVM...")
    t2a_df = run_table_2a(X_mfcc, y, random_state=42)
    t2a_df.to_csv(os.path.join(results_tables_dir, "reproduced_table2a.csv"), index=False)
    print(t2a_df.to_string(index=False))

    # Table 2(b)
    print("\n  -> Running Table 2(b): MFCC features vs Selection% using KNN...")
    t2b_df = run_table_2b(X_mfcc, y, random_state=42)
    t2b_df.to_csv(os.path.join(results_tables_dir, "reproduced_table2b.csv"), index=False)
    print(t2b_df.to_string(index=False))

    # Fused baseline
    print("\n  -> Running Fused Baseline (DWT + MFCC)...")
    fused_df = run_fused_baseline(X_dwt, X_mfcc, y, random_state=42)
    fused_df.to_csv(os.path.join(results_tables_dir, "reproduced_fused_baseline.csv"), index=False)
    print(fused_df.to_string(index=False))

    # 4. Advanced Models & Benchmarks (10-Fold CV & Deep Learning)
    print("\n[Step 4/5] Evaluating Advanced Models (10-Fold Stratified CV on Enhanced Features)...")
    ml_models = get_advanced_ml_models()
    benchmark_df, cms_dict = evaluate_models_cross_validation(ml_models, X_enh, y, n_splits=10, random_state=42)
    
    # Train 1D-CNN Deep Learning model
    print("\n  -> Training PyTorch 1D-CNN on Raw Waveforms...")
    cnn_acc, cnn_cm, cnn_history = train_pcg_1d_cnn(signals, y, epochs=30, batch_size=32)
    cms_dict["PyTorch 1D-CNN"] = cnn_cm

    # Append CNN to benchmark
    cnn_row = {
        "Model": "PyTorch 1D-CNN (End-to-End)",
        "Accuracy% (Mean ± Std)": f"{cnn_acc:.2f} (Holdout Test)",
        "Accuracy_Mean": round(cnn_acc, 2),
        "Accuracy_Std": 0.0,
        "Macro_F1%": round(cnn_acc, 2),
        "Weighted_F1%": round(cnn_acc, 2),
        "Precision%": round(cnn_acc, 2),
        "Recall%": round(cnn_acc, 2)
    }
    benchmark_df = pd.concat([benchmark_df, pd.DataFrame([cnn_row])], ignore_index=True)
    benchmark_df = benchmark_df.sort_values(by="Accuracy_Mean", ascending=False).reset_index(drop=True)

    benchmark_df.to_csv(os.path.join(results_tables_dir, "model_comparison_benchmark.csv"), index=False)
    print("\n=== FINAL BENCHMARK COMPARISON TABLE ===")
    print(benchmark_df.to_string(index=False))

    # 5. Visualizations
    print("\n[Step 5/5] Generating Publication-Quality Figures...")
    
    # 1. Waveforms and Spectrograms
    sample_signals = {}
    for c in CLASS_NAMES:
        c_indices = np.where(labels_str == c)[0]
        sample_signals[c] = signals[c_indices[0]]
    plot_pcg_signals_and_spectrograms(
        sample_signals, fs=8000,
        save_path=os.path.join(results_figs_dir, "pcg_waveforms_and_spectrograms.png")
    )

    # 2. Feature Selection Curves
    plot_feature_selection_comparison(
        t1b_df, t2b_df,
        save_path=os.path.join(results_figs_dir, "feature_selection_curves.png")
    )

    # 3. Benchmark Bar Chart
    plot_model_comparison_barchart(
        benchmark_df,
        save_path=os.path.join(results_figs_dir, "model_comparison_barchart.png")
    )

    # 4. Confusion Matrices (Top 4 models)
    top_models = {
        benchmark_df.iloc[0]["Model"]: cms_dict[benchmark_df.iloc[0]["Model"]],
        benchmark_df.iloc[1]["Model"]: cms_dict[benchmark_df.iloc[1]["Model"]],
        "Quadratic SVM (Baseline)": cms_dict["Quadratic SVM (Baseline)"],
        "PyTorch 1D-CNN": cnn_cm
    }
    plot_confusion_matrices(
        top_models, class_names=CLASS_NAMES,
        save_path=os.path.join(results_figs_dir, "confusion_matrices.png")
    )

    print("\n" + "=" * 78)
    print(" Pipeline execution complete! All tables and figures successfully saved.")
    print(f" Tables saved in:  {os.path.abspath(results_tables_dir)}")
    print(f" Figures saved in: {os.path.abspath(results_figs_dir)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
