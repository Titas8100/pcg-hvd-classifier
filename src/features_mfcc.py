"""
Mel-Frequency Cepstral Coefficients (MFCC) feature extraction module.
Faithfully implements the specifications from Section 4.2 of the report:
- Sampling frequency: 8,000 Hz
- Frame length: 240 samples (30 ms)
- Step size (hop): 80 samples (10 ms)
- Window: Hamming window
- Log Mel-filterbank with Discrete Cosine Transform (DCT)
- 19 cepstral coefficients per signal
"""

import numpy as np
from scipy.fftpack import dct
from typing import List, Tuple, Optional


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Converts Hz frequency to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Converts Mel scale to Hz frequency."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def get_mel_filterbank(
    num_filters: int = 26,
    n_fft: int = 512,
    sample_rate: int = 8000,
    low_freq_hz: float = 20.0,
    high_freq_hz: Optional[float] = None
) -> np.ndarray:
    """
    Creates triangular Mel filterbank matrix of shape (num_filters, n_fft // 2 + 1).
    """
    if high_freq_hz is None:
        high_freq_hz = sample_rate / 2.0

    low_mel = hz_to_mel(low_freq_hz)
    high_mel = hz_to_mel(high_freq_hz)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    fbank = np.zeros((num_filters, n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank


def extract_mfcc_frames(
    signal: np.ndarray,
    sample_rate: int = 8000,
    frame_length: int = 240,
    hop_length: int = 80,
    num_cepstral: int = 19,
    num_filters: int = 26,
    n_fft: int = 512
) -> np.ndarray:
    """
    Extracts frame-wise MFCCs.
    Returns:
        mfcc: 2D array of shape (num_frames, num_cepstral).
    """
    sig_len = len(signal)
    if sig_len < frame_length:
        signal = np.pad(signal, (0, frame_length - sig_len))
        sig_len = len(signal)

    # Framing
    num_frames = 1 + int(np.floor((sig_len - frame_length) / hop_length))
    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(np.arange(0, num_frames * hop_length, hop_length), (frame_length, 1)).T
    )
    frames = signal[indices.astype(np.int32, copy=False)]

    # Hamming window
    window = np.hamming(frame_length)
    frames *= window

    # Magnitude Spectrum via FFT
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    # Filterbank energies
    fbank = get_mel_filterbank(num_filters, n_fft, sample_rate)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_filter_banks = np.log(filter_banks)

    # Discrete Cosine Transform (DCT Type-II)
    mfcc = dct(log_filter_banks, type=2, axis=1, norm='ortho')[:, :num_cepstral]
    return mfcc.astype(np.float32)


def extract_mfcc_19(
    signal: np.ndarray,
    sample_rate: int = 8000,
    frame_length: int = 240,
    hop_length: int = 80,
    num_cepstral: int = 19
) -> np.ndarray:
    """
    Extracts the 19-dimensional MFCC feature vector per PCG recording
    by averaging across all frames, as described in Section 4.2 of the report.
    """
    mfcc_frames = extract_mfcc_frames(
        signal,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        num_cepstral=num_cepstral
    )
    # Mean coefficient per signal
    return np.mean(mfcc_frames, axis=0)


def extract_mfcc_dataset(
    signals: List[np.ndarray],
    sample_rate: int = 8000,
    frame_length: int = 240,
    hop_length: int = 80,
    num_cepstral: int = 19
) -> np.ndarray:
    """
    Extracts 19-dimensional MFCC features for an entire dataset of PCG signals.
    Returns:
        X_mfcc: 2D numpy array of shape (N, 19).
    """
    features = []
    for sig in signals:
        feat = extract_mfcc_19(
            sig,
            sample_rate=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            num_cepstral=num_cepstral
        )
        features.append(feat)
    return np.vstack(features).astype(np.float32)


def get_mfcc_feature_names(num_cepstral: int = 19) -> List[str]:
    """Returns human-readable names for the MFCC features."""
    return [f"MFCC_{i+1}" for i in range(num_cepstral)]


def extract_mfcc_480(
    signal: np.ndarray,
    sample_rate: int = 8000,
    target_frames: int = 30,
    num_cepstral: int = 16
) -> np.ndarray:
    """
    Extracts 480-dimensional flattened MFCC features (30 frames x 16 coefficients)
    used in Reference 24 (Son et al. 2018) for deep neural network training.
    """
    mfcc_frames = extract_mfcc_frames(
        signal,
        sample_rate=sample_rate,
        frame_length=240,
        hop_length=80,
        num_cepstral=num_cepstral
    )
    if len(mfcc_frames) >= target_frames:
        mfcc_sub = mfcc_frames[:target_frames, :]
    else:
        pad_len = target_frames - len(mfcc_frames)
        mfcc_sub = np.pad(mfcc_frames, ((0, pad_len), (0, 0)), mode='edge')
    return mfcc_sub.flatten().astype(np.float32)
