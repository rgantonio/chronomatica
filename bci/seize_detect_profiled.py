"""
Seizure pipeline demo (CHB-MIT):
ADC -> (FFT, XCOR, BBF) -> SVM

This version adds:
- Whole-program profiling via cProfile (optional)
- Per-stage timing stats for: edf_load, windowing, fft+bbf, xcor, feature_concat, svm_fit, svm_predict_proba

Install:
  pip install mne numpy scipy scikit-learn
Optional tools (recommended for memory/counter profiling):
  pip install scalene snakeviz
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import mne
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# -----------------------------
# Lightweight timing utilities
# -----------------------------


class TimeStats:
    """Collect many timing samples per named stage and print simple stats."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    def add(self, name: str, dt_s: float) -> None:
        self.samples[name].append(dt_s)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for name, xs in sorted(
            self.samples.items(), key=lambda kv: sum(kv[1]), reverse=True
        ):
            xs_sorted = sorted(xs)
            n = len(xs_sorted)
            if n == 0:
                continue
            p50 = xs_sorted[n // 2]
            p90 = xs_sorted[int(0.9 * (n - 1))]
            total = sum(xs_sorted)
            mean = total / n
            lines.append(
                f"{name:18s}  n={n:6d}  mean={mean*1e3:9.5f} ms  p50={p50*1e3:9.5f}  p90={p90*1e3:9.5f}  total={total:8.5f} s"
            )
        return lines


@contextmanager
def timed(stats: TimeStats | None, name: str):
    if stats is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add(name, time.perf_counter() - t0)


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
    """Integrate PSD between [f_lo, f_hi] for each channel."""
    idx = (freqs >= f_lo) & (freqs <= f_hi)
    return np.trapz(psd[:, idx], freqs[idx], axis=1)


def feat_fft_psd_welch(
    x: np.ndarray, fs: float, stats: TimeStats | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD per-channel (this is the 'FFT/PSD' heavy part in this pipeline)."""
    with timed(stats, "fft_psd_welch"):
        freqs, psd = welch(x, fs=fs, nperseg=min(512, x.shape[1]), axis=1)
    psd = np.maximum(psd, 1e-12)
    return freqs, psd


def feat_fft_and_bbf(
    x: np.ndarray, fs: float, stats: TimeStats | None = None
) -> np.ndarray:
    """
    FFT/PSD + BBF block:
    - Welch PSD per channel (timed as `fft_psd_welch`)
    - Band features (integration + logs) timed as `bbf`

    Args:
        x: shape (n_channels, n_samples)
        fs: sampling frequency
        stats: optional TimeStats collector
    Returns:
        feats: shape (n_channels * (1 + n_bands),)
    """
    freqs, psd = feat_fft_psd_welch(x, fs=fs, stats=stats)

    with timed(stats, "bbf"):
        total = np.trapz(psd, freqs, axis=1)  # per-channel total power
        feats = [np.log(total)]

        for f_lo, f_hi in EEG_BANDS.values():
            bp = bandpower_from_psd(freqs, psd, f_lo, f_hi)
            feats.append(np.log(np.maximum(bp, 1e-12)))

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
    """
    n_ch, n = x.shape
    max_lag = int(max_lag_s * fs)

    # Change the min part to max
    # If we want to use maximize the channels
    if pairs is None:
        m = max(n_ch, 6)
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]

    feats = []
    for i, j in pairs:
        xi = x[i] - x[i].mean()
        xj = x[j] - x[j].mean()
        denom = np.linalg.norm(xi) * np.linalg.norm(xj) + 1e-12
        corr = np.correlate(xi, xj, mode="full") / denom
        mid = len(corr) // 2
        lo = max(0, mid - max_lag)
        hi = min(len(corr), mid + max_lag + 1)
        peak = np.max(np.abs(corr[lo:hi]))
        feats.append(peak)

    return np.array(feats, dtype=np.float32)


def extract_features(
    window: np.ndarray, fs: float, stats: TimeStats | None = None
) -> np.ndarray:
    """ADC output window -> (FFT+BBF) + (XCOR) -> concatenated feature vector."""
    with timed(stats, "fft+bbf"):
        f1 = feat_fft_and_bbf(window, fs, stats=stats)
    with timed(stats, "xcor"):
        f2 = feat_xcor(window, max_lag_s=0.25, fs=fs)
    with timed(stats, "feature_concat"):
        return np.concatenate([f1, f2], axis=0)


# -----------------------------
# Windowing + labeling
# -----------------------------


def is_in_seizure(
    t0: float, t1: float, seizure_intervals: list[tuple[float, float]]
) -> int:
    for s0, s1 in seizure_intervals:
        if (t0 < s1) and (t1 > s0):
            return 1
    return 0


def build_dataset_from_edf(
    edf_path: str,
    seizure_intervals: list[tuple[float, float]],
    win_s: float = 2.0,
    hop_s: float = 1.0,
    pick_n_ch: int = 18,
    stats: TimeStats | None = None,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load EDF, slice into windows, extract features, label by seizure intervals.

    Returns:
        X: (n_windows, n_features)
        y: (n_windows,)
        fs: sampling rate
    """
    with timed(stats, "edf_load"):
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick(raw.ch_names[: min(pick_n_ch, len(raw.ch_names))])
        fs = float(raw.info["sfreq"])
        data = raw.get_data()

    win = int(win_s * fs)
    hop = int(hop_s * fs)
    n_samples = data.shape[1]

    X, y = [], []

    with timed(stats, "window_loop_total"):
        count = 0
        for start in range(0, n_samples - win + 1, hop):
            end = start + win
            t0, t1 = start / fs, end / fs
            window = data[:, start:end]

            feats = extract_features(window, fs, stats=stats)
            label = is_in_seizure(t0, t1, seizure_intervals)

            X.append(feats)
            y.append(label)

            count += 1
            if max_windows is not None and count >= max_windows:
                break

    return np.vstack(X), np.array(y, dtype=np.int64), fs


# -----------------------------
# Model helpers (for clean profiling buckets)
# -----------------------------


def make_clf() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True)),
        ]
    )


def svm_fit(clf: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    clf.fit(X_train, y_train)
    return clf


def svm_predict_proba(clf: Pipeline, X_test: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X_test)


# -----------------------------
# Main
# -----------------------------


def run(
    edf_path: str,
    seizure_intervals: list[tuple[float, float]],
    win_s: float,
    hop_s: float,
    pick_n_ch: int,
    max_windows: int | None,
    enable_timing: bool,
) -> None:

    stats = TimeStats() if enable_timing else None

    X, y, _fs = build_dataset_from_edf(
        edf_path,
        seizure_intervals,
        win_s=win_s,
        hop_s=hop_s,
        pick_n_ch=pick_n_ch,
        stats=stats,
        max_windows=max_windows,
    )

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Labels are all one class. Fill seizure_intervals for this EDF file."
        )

    with timed(stats, "train_test_split"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )

    clf = make_clf()

    with timed(stats, "svm_fit"):
        clf = svm_fit(clf, X_train, y_train)

    with timed(stats, "svm_predict_proba"):
        scores = svm_predict_proba(clf, X_test)[:, 1]

    preds = (scores >= 0.5).astype(int)

    print(classification_report(y_test, preds, digits=3))
    print("ROC-AUC:", roc_auc_score(y_test, scores))
    print("Example SVM scores:", scores[:10])
    print("Num of features:", X.shape[1])
    print("Scores shape:", scores.shape)

    if stats is not None:
        print("\n==== Timing summary (sorted by total time) ====")
        for line in stats.summary_lines():
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", type=str, default=r"../data/chb_mit_seizures/chb02_16.edf")
    ap.add_argument("--win_s", type=float, default=2.0)
    ap.add_argument("--hop_s", type=float, default=1.0)
    ap.add_argument("--pick_n_ch", type=int, default=18)
    ap.add_argument(
        "--max_windows",
        type=int,
        default=0,
        help="If >0, stop after this many windows (useful for quick profiling runs). 0 means no limit.",
    )
    ap.add_argument(
        "--timing", action="store_true", help="Print per-stage timing stats."
    )
    ap.add_argument(
        "--cprofile",
        type=str,
        default="",
        help="If set, write cProfile stats to this path (e.g. prof.out).",
    )
    args = ap.parse_args()

    # Example seizure interval for chb01_03.edf (edit for your file)
    seizure_intervals = [
        (133.0, 212.0),
    ]

    max_windows = None if args.max_windows <= 0 else args.max_windows

    if args.cprofile:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        try:
            run(
                edf_path=args.edf,
                seizure_intervals=seizure_intervals,
                win_s=args.win_s,
                hop_s=args.hop_s,
                pick_n_ch=args.pick_n_ch,
                max_windows=max_windows,
                enable_timing=args.timing,
            )
        finally:
            pr.disable()
            pr.dump_stats(args.cprofile)
            print(f"\nWrote cProfile stats to: {args.cprofile}")
            print(
                'View: python -c \'import pstats; p=pstats.Stats("%s"); p.strip_dirs().sort_stats("cumtime").print_stats(40)\''
                % args.cprofile
            )
            print("Or: snakeviz %s" % args.cprofile)
    else:
        run(
            edf_path=args.edf,
            seizure_intervals=seizure_intervals,
            win_s=args.win_s,
            hop_s=args.hop_s,
            pick_n_ch=args.pick_n_ch,
            max_windows=max_windows,
            enable_timing=args.timing,
        )


if __name__ == "__main__":
    main()
