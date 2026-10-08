"""
Discrete Wavelet Transform (DWT) feature extraction module.
Faithfully implements:
1. 3-level DWT decomposition with 24-dimensional statistical feature vectors (Report Sec 4.1).
2. Multi-scale Wavelet Transform feature extraction (getmswtfeat from Reference 24 / Son et al. 2018).
"""

import numpy as np
import pywt
from scipy import stats
from typing import List, Optional, Tuple


def compute_subband_statistics(coeffs: np.ndarray) -> np.ndarray:
    """
    Computes 6 descriptive statistical features for a given sub-band coefficient array:
    1. Mean
    2. Standard Deviation
    3. Variance
    4. Skewness
    5. Kurtosis
    6. Shannon Entropy
    """
    n = len(coeffs)
    if n == 0:
        return np.zeros(6, dtype=np.float32)

    mean_val = np.mean(coeffs)
    std_val = np.std(coeffs)
    var_val = np.var(coeffs)
    
    # Skewness and Kurtosis
    skew_val = stats.skew(coeffs) if std_val > 1e-8 else 0.0
    kurt_val = stats.kurtosis(coeffs) if std_val > 1e-8 else 0.0

    # Shannon Entropy of energy distribution
    energy = coeffs ** 2
    total_energy = np.sum(energy)
    if total_energy > 1e-12:
        prob = energy / total_energy
        prob = prob[prob > 1e-12]
        entropy_val = -np.sum(prob * np.log2(prob)) / np.log2(len(coeffs))
    else:
        entropy_val = 0.0

    return np.array([mean_val, std_val, var_val, skew_val, kurt_val, entropy_val], dtype=np.float32)


def extract_dwt_24(signal: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    Extracts the 24-dimensional DWT feature vector as described in Section 4.1 of the report:
    - 3-level decomposition: cA3, cD3, cD2, cD1 (4 sub-bands)
    - 6 statistics per sub-band -> 4 * 6 = 24 features.
    """
    # Multilevel 1D wavelet decomposition: returns [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    # coeffs[0] = cA3, coeffs[1] = cD3, coeffs[2] = cD2, coeffs[3] = cD1
    
    features = []
    for subband in coeffs:
        subband_feats = compute_subband_statistics(subband)
        features.extend(subband_feats)
        
    return np.array(features, dtype=np.float32)


def extract_dwt_dataset(
    signals: List[np.ndarray],
    wavelet: str = "db4",
    level: int = 3
) -> np.ndarray:
    """
    Extracts 24-dimensional DWT features for an entire dataset of PCG signals.
    Returns:
        X_dwt: 2D numpy array of shape (N, 24).
    """
    feature_matrix = []
    for sig in signals:
        feat = extract_dwt_24(sig, wavelet=wavelet, level=level)
        feature_matrix.append(feat)
    return np.vstack(feature_matrix).astype(np.float32)


def get_dwt_feature_names() -> List[str]:
    """Returns human-readable names for the 24 DWT features."""
    subbands = ["cA3", "cD3", "cD2", "cD1"]
    metrics = ["mean", "std", "var", "skewness", "kurtosis", "shannon_entropy"]
    names = []
    for sb in subbands:
        for m in metrics:
            names.append(f"DWT_{sb}_{m}")
    return names


def extract_getmswtfeat(
    signal: np.ndarray,
    fs: int = 8000,
    winsize: Optional[int] = None,
    wininc: Optional[int] = None,
    wavelet: str = "db4",
    j_level: int = 5
) -> np.ndarray:
    """
    Python implementation of Dr. Rami Khushaba's getmswtfeat algorithm
    used in Reference 24 (Son et al. 2018):
    Sliding window wavelet feature extraction (Energy, Variance, Waveform Length, Entropy).
    """
    if winsize is None:
        winsize = fs // 2  # 4000 samples (0.5 s)
    if wininc is None:
        wininc = fs // 8   # 1000 samples (0.125 s)

    n_samples = len(signal)
    if n_samples < winsize:
        signal = np.pad(signal, (0, winsize - n_samples))
        n_samples = len(signal)

    num_wins = max(1, (n_samples - winsize) // wininc + 1)
    
    # We aggregate window features across the 3 cardiac cycles to get a fixed-length summary
    window_features = []
    for w in range(num_wins):
        st = w * wininc
        en = st + winsize
        win_sig = signal[st:en]
        
        # Detrend
        win_sig = win_sig - np.polyval(np.polyfit(np.arange(len(win_sig)), win_sig, 1), np.arange(len(win_sig)))
        
        # Multi-level DWT
        coeffs = pywt.wavedec(win_sig, wavelet=wavelet, level=j_level)
        # Sub-band lengths
        feats_win = []
        for c in coeffs:
            c_sq = c ** 2
            tot_e = np.sum(c_sq)
            pct_e = (c_sq / (tot_e + 1e-12)) * 100.0
            
            e_mean = np.mean(pct_e)
            v_var = np.var(pct_e)
            wl = np.sum(np.diff(pct_e) ** 2) if len(pct_e) > 1 else 0.0
            
            # Entropy
            p = pct_e / (np.sum(pct_e) + 1e-12)
            p = p[p > 1e-12]
            ent = -np.sum(p * np.log(p)) / len(pct_e) if len(pct_e) > 0 else 0.0
            
            feats_win.extend([np.log1p(max(0, e_mean)), np.log1p(max(0, v_var)), np.log1p(max(0, wl)), ent])
        window_features.append(feats_win)
        
    window_features = np.array(window_features, dtype=np.float32)
    # Aggregate mean across windows
    return np.mean(window_features, axis=0)
