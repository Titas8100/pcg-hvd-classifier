"""
Enhanced clinical, spectral, and temporal feature extraction module.
Engineers additional discriminative features to improve upon the baseline report models:
1. Spectral Centroid, Bandwidth, Rolloff, and Flatness (capturing murmur frequency signatures).
2. Temporal Energy, Zero-Crossing Rate, Crest Factor, and Envelope Dynamics.
3. MFCC Frame Variance (Delta dynamics).
4. Feature fusion pipeline combining DWT (24), MFCC (19), and Enhanced (31) into a 74-dim vector.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy import stats
from typing import List, Tuple, Dict
from .features_dwt import extract_dwt_24
from .features_mfcc import extract_mfcc_frames


def compute_spectral_features(signal: np.ndarray, fs: int = 8000) -> np.ndarray:
    """
    Computes spectral shape descriptors:
    - Spectral Centroid (mean & std across STFT frames)
    - Spectral Bandwidth / Spread
    - Spectral Rolloff at 85% and 95%
    - Spectral Flatness
    """
    # STFT
    nperseg = 256
    noverlap = 128
    f, t, Zxx = sp_signal.stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx) + 1e-12
    power = mag ** 2
    
    # 1. Spectral Centroid
    norm_power = power / np.sum(power, axis=0, keepdims=True)
    f_col = f[:, np.newaxis]
    centroid = np.sum(f_col * norm_power, axis=0)
    
    # 2. Spectral Bandwidth / Spread
    spread = np.sqrt(np.sum(((f_col - centroid) ** 2) * norm_power, axis=0))
    
    # 3. Spectral Rolloff (85% and 95%)
    cum_power = np.cumsum(power, axis=0)
    total_power = cum_power[-1:, :]
    
    rolloff_85 = np.zeros(power.shape[1])
    rolloff_95 = np.zeros(power.shape[1])
    for j in range(power.shape[1]):
        idx85 = np.where(cum_power[:, j] >= 0.85 * total_power[0, j])[0]
        rolloff_85[j] = f[idx85[0]] if len(idx85) > 0 else f[-1]
        idx95 = np.where(cum_power[:, j] >= 0.95 * total_power[0, j])[0]
        rolloff_95[j] = f[idx95[0]] if len(idx95) > 0 else f[-1]

    # 4. Spectral Flatness (Geometric Mean / Arithmetic Mean of power)
    geom_mean = np.exp(np.mean(np.log(power + 1e-12), axis=0))
    arith_mean = np.mean(power, axis=0)
    flatness = geom_mean / (arith_mean + 1e-12)

    return np.array([
        np.mean(centroid), np.std(centroid),
        np.mean(spread), np.std(spread),
        np.mean(rolloff_85), np.std(rolloff_85),
        np.mean(rolloff_95), np.std(rolloff_95),
        np.mean(flatness), np.std(flatness)
    ], dtype=np.float32)


def compute_temporal_envelope_features(signal: np.ndarray, fs: int = 8000) -> np.ndarray:
    """
    Computes temporal amplitude, energy, zero-crossing, and Hilbert envelope statistics:
    - Zero-crossing rate (mean & std)
    - RMS Energy (mean & std)
    - Crest Factor (peak-to-RMS ratio)
    - Hilbert Amplitude Envelope statistics (mean, std, skewness, kurtosis)
    """
    # Zero Crossing Rate
    zcr = np.abs(np.diff(np.sign(signal))) / 2.0
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # RMS Energy
    rms_val = np.sqrt(np.mean(signal ** 2))
    peak_val = np.max(np.abs(signal))
    crest_factor = peak_val / (rms_val + 1e-8)

    # Hilbert Analytic Envelope
    analytic_signal = sp_signal.hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    env_mean = np.mean(amplitude_envelope)
    env_std = np.std(amplitude_envelope)
    env_skew = stats.skew(amplitude_envelope)
    env_kurt = stats.kurtosis(amplitude_envelope)

    return np.array([
        zcr_mean, zcr_std,
        rms_val, crest_factor,
        env_mean, env_std, env_skew, env_kurt
    ], dtype=np.float32)


def extract_enhanced_features_single(signal: np.ndarray, fs: int = 8000) -> np.ndarray:
    """
    Extracts the combined 74-dimensional feature vector for a single PCG signal:
    - 24 DWT features (3-level subband statistics)
    - 19 MFCC mean coefficients
    - 19 MFCC std coefficients (capturing spectral dynamics)
    - 10 Spectral shape descriptors (centroid, spread, rolloff, flatness)
    - 8 Temporal & Hilbert envelope descriptors
    Total: 24 + 19 + 19 + 10 + 8 = 80 features.
    """
    # 1. DWT (24)
    dwt_feats = extract_dwt_24(signal, wavelet="db4", level=3)
    
    # 2. MFCC (19 mean + 19 std = 38)
    mfcc_frames = extract_mfcc_frames(signal, sample_rate=fs, frame_length=240, hop_length=80, num_cepstral=19)
    mfcc_mean = np.mean(mfcc_frames, axis=0)
    mfcc_std = np.std(mfcc_frames, axis=0)
    
    # 3. Spectral Descriptors (10)
    spec_feats = compute_spectral_features(signal, fs=fs)
    
    # 4. Temporal & Envelope Descriptors (8)
    temp_feats = compute_temporal_envelope_features(signal, fs=fs)

    return np.concatenate([dwt_feats, mfcc_mean, mfcc_std, spec_feats, temp_feats]).astype(np.float32)


def extract_enhanced_dataset(signals: List[np.ndarray], fs: int = 8000) -> np.ndarray:
    """
    Extracts the 80-dimensional enhanced feature matrix for the entire dataset.
    Returns:
        X_enhanced: 2D numpy array of shape (N, 80).
    """
    feature_list = [extract_enhanced_features_single(sig, fs=fs) for sig in signals]
    return np.vstack(feature_list).astype(np.float32)


def get_enhanced_feature_names() -> List[str]:
    """Returns human-readable names for all 80 enhanced features."""
    names = []
    # 24 DWT
    subbands = ["cA3", "cD3", "cD2", "cD1"]
    metrics = ["mean", "std", "var", "skewness", "kurtosis", "entropy"]
    for sb in subbands:
        for m in metrics:
            names.append(f"DWT_{sb}_{m}")
            
    # 19 MFCC mean
    for i in range(19):
        names.append(f"MFCC_Mean_{i+1}")
        
    # 19 MFCC std
    for i in range(19):
        names.append(f"MFCC_Std_{i+1}")
        
    # 10 Spectral
    names.extend([
        "Spec_Centroid_Mean", "Spec_Centroid_Std",
        "Spec_Spread_Mean", "Spec_Spread_Std",
        "Spec_Rolloff85_Mean", "Spec_Rolloff85_Std",
        "Spec_Rolloff95_Mean", "Spec_Rolloff95_Std",
        "Spec_Flatness_Mean", "Spec_Flatness_Std"
    ])
    
    # 8 Temporal/Envelope
    names.extend([
        "Temp_ZCR_Mean", "Temp_ZCR_Std",
        "Temp_RMS_Energy", "Temp_Crest_Factor",
        "Env_Hilbert_Mean", "Env_Hilbert_Std",
        "Env_Hilbert_Skewness", "Env_Hilbert_Kurtosis"
    ])
    return names
