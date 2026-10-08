"""
Visualization utilities for PCG signals, feature distributions,
confusion matrices, and experimental benchmark comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Optional, Tuple

CLASS_NAMES = ['AS', 'MR', 'MS', 'MVP', 'N']
CLASS_FULL_NAMES = {
    'AS': 'Aortic Stenosis',
    'MR': 'Mitral Regurgitation',
    'MS': 'Mitral Stenosis',
    'MVP': 'Mitral Valve Prolapse',
    'N': 'Normal / Healthy'
}

# Color palette
PALETTE = {
    'AS': '#d9534f',
    'MR': '#f0ad4e',
    'MS': '#5bc0de',
    'MVP': '#9c27b0',
    'N': '#5cb85c'
}


def plot_pcg_signals_and_spectrograms(
    signals_dict: Dict[str, np.ndarray],
    fs: int = 8000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots representative PCG waveforms and corresponding spectrograms for the 5 classes.
    """
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex='col')
    fig.suptitle("Phonocardiogram (PCG) Waveforms & Spectrograms by Cardiac Condition", fontsize=16, fontweight='bold')

    for i, c in enumerate(CLASS_NAMES):
        sig = signals_dict[c]
        time_axis = np.arange(len(sig)) / fs
        
        # 1. Waveform
        ax_wave = axes[i, 0]
        ax_wave.plot(time_axis, sig, color=PALETTE[c], alpha=0.9, lw=0.8)
        ax_wave.set_ylabel(f"{c}\n({CLASS_FULL_NAMES[c]})", fontsize=10, fontweight='bold')
        ax_wave.grid(True, alpha=0.3)
        ax_wave.set_ylim(-1.1, 1.1)
        if i == 0:
            ax_wave.set_title("Time-Domain PCG Waveform", fontsize=12, fontweight='bold')
        if i == 4:
            ax_wave.set_xlabel("Time (seconds)", fontsize=11)

        # 2. Spectrogram
        ax_spec = axes[i, 1]
        ax_spec.specgram(sig, Fs=fs, NFFT=256, noverlap=128, cmap='viridis')
        ax_spec.set_ylim(0, 1500)  # Diagnostic heart sound energy is mostly < 1.5 kHz
        if i == 0:
            ax_spec.set_title("Time-Frequency Spectrogram (0–1500 Hz)", fontsize=12, fontweight='bold')
        if i == 4:
            ax_spec.set_xlabel("Time (seconds)", fontsize=11)
        ax_spec.set_ylabel("Freq (Hz)", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved waveform & spectrogram figure to '{save_path}'.")
    return fig


def plot_feature_selection_comparison(
    table_1b_df: pd.DataFrame,
    table_2b_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots Feature Selection % vs Accuracy % curves comparing:
    1. Table 1(b) DWT (Report vs Python for SVM & KNN)
    2. Table 2(b) MFCC (Report vs Python for KNN)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: DWT Feature Selection (Table 1b)
    ax1 = axes[0]
    ax1.plot(table_1b_df["Selection%"], table_1b_df["Report_SVM_Acc%"], 's--', color='#1f77b4', label="Report SVM", lw=1.8)
    ax1.plot(table_1b_df["Selection%"], table_1b_df["Python_SVM_Acc%"], 's-', color='#004c8c', label="Python SVM (Reproduced)", lw=2.2)
    ax1.plot(table_1b_df["Selection%"], table_1b_df["Report_KNN_Acc%"], 'o--', color='#ff7f0e', label="Report KNN", lw=1.8)
    ax1.plot(table_1b_df["Selection%"], table_1b_df["Python_KNN_Acc%"], 'o-', color='#b35400', label="Python KNN (Reproduced)", lw=2.2)
    ax1.set_title("Table 1(b): DWT Features vs Selection %", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Feature Selection (%)", fontsize=11)
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax1.set_xticks(range(10, 101, 10))
    ax1.set_ylim(65, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Right: MFCC Feature Selection (Table 2b)
    ax2 = axes[1]
    ax2.plot(table_2b_df["Selection%"], table_2b_df["Report_KNN_Acc%"], 'o--', color='#2ca02c', label="Report KNN (MFCC)", lw=1.8)
    ax2.plot(table_2b_df["Selection%"], table_2b_df["Python_KNN_Acc%"], 'o-', color='#137313', label="Python KNN (MFCC Reproduced)", lw=2.2)
    ax2.set_title("Table 2(b): MFCC Features vs Selection %", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Feature Selection (%)", fontsize=11)
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax2.set_xticks(range(10, 101, 10))
    ax2.set_ylim(65, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved feature selection comparison figure to '{save_path}'.")
    return fig


def plot_model_comparison_barchart(
    benchmark_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a horizontal bar chart comparing Accuracy and Macro-F1 across all evaluated models.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(benchmark_df))
    height = 0.38

    bars_acc = ax.barh(y_pos - height/2, benchmark_df["Accuracy_Mean"], height=height, label="Accuracy (%)", color="#2b5c8f")
    bars_f1 = ax.barh(y_pos + height/2, benchmark_df["Macro_F1%"], height=height, label="Macro F1 (%)", color="#48a9a6")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(benchmark_df["Model"], fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Performance Metric (%)", fontsize=12)
    ax.set_title("10-Fold Stratified Cross-Validation Benchmark Comparison", fontsize=14, fontweight='bold')
    ax.set_xlim(50, 103)
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=11)

    # Add numeric labels to bars
    for bar in bars_acc:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', ha='left', fontsize=9, fontweight='bold', color="#1b3b5a")
    for bar in bars_f1:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', ha='left', fontsize=9, color="#1e5c5a")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved benchmark barchart to '{save_path}'.")
    return fig


def plot_confusion_matrices(
    cms_dict: Dict[str, np.ndarray],
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots 2x2 or 3x2 grid of confusion matrices for key benchmark models.
    """
    num_models = len(cms_dict)
    cols = 2
    rows = int(np.ceil(num_models / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (mname, cm) in enumerate(cms_dict.items()):
        ax = axes[idx]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
        
        # Annotations: Count (Percent%)
        annot = np.empty_like(cm).astype(str)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f"{cm[r, c]}\n({cm_norm[r, c]:.1f}%)"

        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(mname, fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved confusion matrices figure to '{save_path}'.")
    return fig
