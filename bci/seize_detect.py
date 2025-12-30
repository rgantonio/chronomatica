"""
Seizure pipeline demo (CHB-MIT):
ADC -> (FFT, XCOR, BBF) -> SVM
Stops at SVM (no threshold/gate).

Install:
  pip install mne numpy scipy scikit-learn
"""

from __future__ import annotations
import numpy as np
import mne
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# -----------------------------
# Feature extraction blocks
# -----------------------------

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpower_from_psd(
    freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float
) -> np.ndarray:
    """Integrate PSD between [f_lo, f_hi] for each channel.
    Args:
        freqs: shape (n_freqs,)
        psd: shape (n_channels, n_freqs)
    Returns:
        bandpowers: shape (n_channels,)
    """
    # Boolean mask for desired band
    idx = (freqs >= f_lo) & (freqs <= f_hi)
    # trapzal integration over frequency axis
    return np.trapz(psd[:, idx], freqs[idx], axis=1)


def feat_fft_psd_welch(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD per-channel (this is the 'FFT/PSD' heavy part in this pipeline).
    Args:
        x: shape (n_channels, n_samples)
        fs: sampling frequency
    Returns:
        freqs: shape (n_freqs,)
        psd: shape (n_channels, n_freqs)
    """
    freqs, psd = welch(x, fs=fs, nperseg=min(512, x.shape[1]), axis=1)
    # For protecting divide by 0
    psd = np.maximum(psd, 1e-12)
    return freqs, psd


def feat_fft_and_bbf(x: np.ndarray, fs: float) -> np.ndarray:
    """
    FFT + BBF block:
    - Welch PSD per channel
    - Return: [log_total_power, log_bandpowers...] per channel concatenated

    Args:
        x: shape (n_channels, n_samples)
        fs: sampling frequency
    Returns:
        feats: shape (n_channels * (1 + n_bands),)
    """
    # Do the welch filter
    # x: shape (n_channels, n_samples)
    freqs, psd = feat_fft_psd_welch(x, fs=fs)

    # Total power under psd curve
    total = np.trapz(psd, freqs, axis=1)  # per-channel total power

    # This is n_channels long
    # Meaning each channel has a total power value
    feats = [np.log(total)]

    # This is simply bin band features
    for f_lo, f_hi in EEG_BANDS.values():
        # Mask and integrate to extract band power
        bp = bandpower_from_psd(freqs, psd, f_lo, f_hi)
        feats.append(np.log(np.maximum(bp, 1e-12)))

    # shape: (n_channels * (1 + n_bands),)
    # the 1 comes from total power
    # the n_bands comes from each EEG band
    # Multiplied to each channel
    return np.concatenate(feats, axis=0)


def feat_xcor(
    x: np.ndarray,
    max_lag_s: float,
    fs: float,
    pairs: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    XCOR block:
    For selected channel pairs, compute normalized cross-correlation peak within +/- max_lag.
    Args:
        x: shape (n_channels, n_samples)
        max_lag_s: maximum lag in seconds
        fs: sampling frequency
        pairs: list of (i,j) channel index pairs to compute XCOR for. If None, use default small set.
    Returns:
        feats: shape (n_pairs,)
    """
    # Get shape
    n_ch, n = x.shape
    # Get max lag in samples
    max_lag = int(max_lag_s * fs)

    if pairs is None:
        # default: a small set of pairs to limit feature size (first 6 channels)
        m = min(n_ch, 6)
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]

    feats = []
    for i, j in pairs:
        xi = x[i] - x[i].mean()
        xj = x[j] - x[j].mean()

        denom = np.linalg.norm(xi) * np.linalg.norm(xj) + 1e-12
        # full correlation via FFT is faster, but direct is fine for small windows
        corr = np.correlate(xi, xj, mode="full") / denom
        mid = len(corr) // 2
        lo = max(0, mid - max_lag)
        hi = min(len(corr), mid + max_lag + 1)
        peak = np.max(np.abs(corr[lo:hi]))
        feats.append(peak)

    return np.array(feats, dtype=np.float32)


def extract_features(window: np.ndarray, fs: float) -> np.ndarray:
    """ADC output window -> (FFT, XCOR, BBF) -> concatenated feature vector.
    Args:
        window: shape (n_channels, n_samples)
        fs: sampling frequency
    Returns:
        feats: shape (n_features,)
    """
    f1 = feat_fft_and_bbf(window, fs)
    f2 = feat_xcor(window, max_lag_s=0.25, fs=fs)
    return np.concatenate([f1, f2], axis=0)


# -----------------------------
# Windowing + labeling
# -----------------------------


def is_in_seizure(
    t0: float, t1: float, seizure_intervals: list[tuple[float, float]]
) -> int:
    """Label window as seizure if it overlaps any seizure interval.
    Args:
        t0: window start time (s)
        t1: window end time (s)
        seizure_intervals: list of (start_s, end_s) seizure intervals
    Returns:
        label: 1 if seizure present, 0 otherwise
    """
    for s0, s1 in seizure_intervals:
        if (t0 < s1) and (t1 > s0):  # overlap
            return 1
    return 0


def build_dataset_from_edf(
    edf_path: str,
    seizure_intervals: list[tuple[float, float]],
    win_s: float = 2.0,
    hop_s: float = 1.0,
    pick_n_ch: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load EDF (ADC), slice into windows, extract features, label by seizure intervals.

    Args:
        edf_path (str): Path to EDF file
        seizure_intervals (list of (float, float)): List of (start_s, end_s) seizure intervals
        win_s (float): Window size in seconds
        hop_s (float): Hop size in seconds
        pick_n_ch (int): Number of channels to pick from EDF (first N)
    Returns:
        X (np.ndarray): Feature matrix, shape (n_windows, n_features)
        y (np.ndarray): Labels, shape (n_windows,)
    """

    # Reads EDF file into MNE Raw object
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    # Pick only a subset of channels for speed
    raw.pick(raw.ch_names[: min(pick_n_ch, len(raw.ch_names))])

    # Extract sampling frequency in Hz
    fs = float(raw.info["sfreq"])
    # Extract data array and shape is in (n_ch, n_samples)
    data = raw.get_data()

    # Convert window/hop from seconds to samples
    # Number of samples per window
    win = int(win_s * fs)
    # Stride between windows in samples
    hop = int(hop_s * fs)
    # Total number of samples
    n_samples = data.shape[1]

    # Initialize data
    X, y = [], []

    # Slide windows across entire recording
    # Note, that the stride overlaps the windows
    for start in range(0, n_samples - win + 1, hop):
        end = start + win
        # Get start and end time
        t0, t1 = start / fs, end / fs
        # Note that data is in shape (n_ch, n_samples)
        window = data[:, start:end]
        # Extract features from this window
        feats = extract_features(window, fs)
        # Check if a seizure is present in this window
        label = is_in_seizure(t0, t1, seizure_intervals)
        # Collect features and label
        X.append(feats)
        y.append(label)

    return np.vstack(X), np.array(y, dtype=np.int64)


def svm_predict_proba(clf, X):
    """
    Predict probability estimates for the positive class using the SVM classifier.
    Args:
        clf: Trained SVM classifier pipeline
        X: Feature matrix, shape (n_samples, n_features)
    Returns:
        probs: Probability estimates for the positive class, shape (n_samples,)
    """
    return clf.predict_proba(X)


# -----------------------------
# Main: train SVM and output scores
# -----------------------------


def main():
    # 1) Point to ONE EDF file from CHB-MIT
    edf_path = r"../data/chb_mit_seizures/chb01_03.edf"

    # 2) For the same file, provide seizure start/end times (seconds)
    #    You can find these in the CHB-MIT summary text files or *.edf.seizures annotations.
    seizure_intervals = [
        # (start_s, end_s),
        (130.0, 212.0),
    ]

    X, y = build_dataset_from_edf(edf_path, seizure_intervals, win_s=2.0, hop_s=1.0)

    # Avoid degenerate training if you didn't fill seizure_intervals yet
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Labels are all one class. Fill seizure_intervals for this EDF file."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True)),
        ]
    )

    clf.fit(X_train, y_train)

    # ---- STOP HERE (SVM output) ----
    # decision scores (higher => more seizure-like)
    scores = svm_predict_proba(clf, X_test)[:, 1]
    # scores = clf.predict_proba(X_test)[:, 1]
    preds = (scores >= 0.5).astype(int)  # not your THR yet; just for reporting

    print("scores shape:", scores.shape)
    print(classification_report(y_test, preds, digits=3))
    print("ROC-AUC:", roc_auc_score(y_test, scores))

    # If you want to inspect raw SVM outputs for the pipeline analysis:
    print("Example SVM scores:", scores[:10])


if __name__ == "__main__":
    main()
