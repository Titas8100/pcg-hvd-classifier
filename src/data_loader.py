"""
Dataset acquisition, extraction, and PCG audio ingestion module.
Supports both local execution and Google Colab environments.
"""

import os
import shutil
import urllib.request
import urllib.parse
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
from typing import Tuple, List, Dict, Optional

CLASS_NAMES = ['AS', 'MR', 'MS', 'MVP', 'N']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

GITHUB_ARCHIVES = {
    'AS': 'AS_New_3주기.rar',
    'MR': 'MR_New_3주기.rar',
    'MS': 'MS_New_3주기.rar',
    'MVP': 'MVP_New_3주기.rar',
    'N': 'N_New_3주기.rar'
}

RAW_GITHUB_BASE = (
    "https://raw.githubusercontent.com/yaseen21khan/"
    "Classification-of-Heart-Sound-Signal-Using-Multiple-Features-/master/"
)


def ensure_dataset(data_dir: str = "data/raw", scratch_cache_dir: Optional[str] = None) -> str:
    """
    Ensures that all 1,000 PCG .wav files (200 per class) are present in data_dir.
    Downloads and extracts from GitHub if missing.
    """
    os.makedirs(data_dir, exist_ok=True)
    all_complete = True

    for c in CLASS_NAMES:
        class_path = os.path.join(data_dir, c)
        if not os.path.exists(class_path):
            all_complete = False
            break
        wavs = [f for f in os.listdir(class_path) if f.lower().endswith(".wav")]
        if len(wavs) < 200:
            all_complete = False
            break

    if all_complete:
        print(f"[Dataset] All 5 classes (200 files each) present in '{data_dir}'.")
        return data_dir

    # Check scratch cache directory if available
    if scratch_cache_dir and os.path.exists(scratch_cache_dir):
        print(f"[Dataset] Checking scratch cache at '{scratch_cache_dir}'...")
        can_copy = True
        for c in CLASS_NAMES:
            c_dir = os.path.join(scratch_cache_dir, c)
            if not os.path.exists(c_dir) or len([f for f in os.listdir(c_dir) if f.endswith(".wav")]) < 200:
                can_copy = False
                break
        if can_copy:
            for c in CLASS_NAMES:
                src_c = os.path.join(scratch_cache_dir, c)
                dst_c = os.path.join(data_dir, c)
                os.makedirs(dst_c, exist_ok=True)
                for f in os.listdir(src_c):
                    if f.endswith(".wav"):
                        shutil.copy2(os.path.join(src_c, f), os.path.join(dst_c, f))
            print(f"[Dataset] Successfully copied dataset from scratch cache to '{data_dir}'.")
            return data_dir

    # Otherwise download from GitHub
    print("[Dataset] Downloading archives from GitHub repository...")
    temp_dir = os.path.join(data_dir, "_temp_download")
    os.makedirs(temp_dir, exist_ok=True)

    for label, arc_name in GITHUB_ARCHIVES.items():
        dst_class_dir = os.path.join(data_dir, label)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        current_wavs = [f for f in os.listdir(dst_class_dir) if f.lower().endswith('.wav')]
        if len(current_wavs) >= 200:
            continue

        url = RAW_GITHUB_BASE + urllib.parse.quote(arc_name)
        rar_path = os.path.join(temp_dir, f"{label}.rar")
        if not os.path.exists(rar_path) or os.path.getsize(rar_path) < 1000000:
            print(f"  Downloading {label} archive from {url}...")
            urllib.request.urlretrieve(url, rar_path)

        # Extraction strategy:
        extracted = False
        # 1. Try unrar command
        for unrar_cmd in ["unrar", "UnRAR", "7z", "7za"]:
            try:
                cmd = [unrar_cmd, "e", "-y", rar_path, dst_class_dir + os.sep]
                ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if ret.returncode == 0:
                    extracted = True
                    break
            except FileNotFoundError:
                continue

        if not extracted:
            # Try python rarfile or patool if available
            try:
                import patoolib
                patoolib.extract_archive(rar_path, outdir=dst_class_dir)
                extracted = True
            except Exception:
                pass

        if not extracted:
            raise RuntimeError(
                f"Failed to extract {rar_path}. Please install 'unrar' or '7zip' to extract RAR archives."
            )

    # Clean up temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[Dataset] Dataset download & extraction complete in '{data_dir}'.")
    return data_dir


def load_pcg_dataset(
    data_dir: str = "data/raw",
    target_sr: int = 8000
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str], Dict[str, any]]:
    """
    Loads all PCG audio files from data_dir.
    
    Returns:
        signals: List of 1D numpy arrays (normalized float32 in [-1, 1]).
        labels_str: Array of string class names ('AS', 'MR', 'MS', 'MVP', 'N').
        labels_idx: Array of integer labels (0 to 4).
        file_names: List of source file basenames.
        metadata: Summary metadata dictionary.
    """
    signals = []
    labels_str = []
    labels_idx = []
    file_names = []
    durations = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        wav_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(".wav")])
        for fname in wav_files:
            fpath = os.path.join(class_dir, fname)
            sr, raw_audio = wavfile.read(fpath)

            # Convert to float32 in [-1.0, 1.0]
            if raw_audio.dtype == np.int16:
                audio = raw_audio.astype(np.float32) / 32768.0
            elif raw_audio.dtype == np.int32:
                audio = raw_audio.astype(np.float32) / 2147483648.0
            elif raw_audio.dtype == np.uint8:
                audio = (raw_audio.astype(np.float32) - 128.0) / 128.0
            else:
                audio = raw_audio.astype(np.float32)

            # Mono channel conversion
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Resample if not matching target sampling rate
            if sr != target_sr:
                from scipy.signal import resample_poly
                import math
                gcd = math.gcd(target_sr, sr)
                audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)

            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Normalize peak amplitude to prevent clipping
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                audio = audio / peak

            signals.append(audio)
            labels_str.append(class_name)
            labels_idx.append(CLASS_TO_IDX[class_name])
            file_names.append(fname)
            durations.append(len(audio) / target_sr)

    labels_str = np.array(labels_str)
    labels_idx = np.array(labels_idx, dtype=np.int64)

    metadata = {
        "num_samples": len(signals),
        "sampling_rate": target_sr,
        "classes": CLASS_NAMES,
        "class_counts": {c: int(np.sum(labels_str == c)) for c in CLASS_NAMES},
        "mean_duration_sec": float(np.mean(durations)),
        "min_duration_sec": float(np.min(durations)),
        "max_duration_sec": float(np.max(durations)),
    }

    return signals, labels_str, labels_idx, file_names, metadata
