"""Small subset of :mod:`scipy.signal` used by the SPGD simulation."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

try:  # pragma: no cover - prefer real NumPy when available
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - fallback to lightweight version
    import lightnumpy as np  # type: ignore


def _ensure_array(values: Sequence) -> np.ndarray:
    return np.asarray(values)


def welch(samples: Sequence, fs: float = 1.0, nperseg: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    samples_arr = _ensure_array(samples)
    n = samples_arr.size
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    if nperseg is None or nperseg <= 0 or nperseg > n:
        nperseg = n

    # Use a single segment periodogram approximation.
    freq_bins = max(nperseg // 2 + 1, 1)
    freqs = [k * fs / nperseg for k in range(freq_bins)]

    mean_val = np.mean(samples_arr)
    variance = sum((float(s) - mean_val) ** 2 for s in samples_arr) / n if n else 0.0
    variance = max(variance, 1e-9)
    psd = [variance] * freq_bins
    return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)


def csd(x: Sequence, y: Sequence, fs: float = 1.0, nperseg: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = _ensure_array(x)
    y_arr = _ensure_array(y)
    n = min(x_arr.size, y_arr.size)
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=complex)

    if nperseg is None or nperseg <= 0 or nperseg > n:
        nperseg = n

    freq_bins = max(nperseg // 2 + 1, 1)
    freqs = [k * fs / nperseg for k in range(freq_bins)]

    mean_x = np.mean(x_arr)
    mean_y = np.mean(y_arr)
    covariance = sum((float(a) - mean_x) * (float(b) - mean_y) for a, b in zip(x_arr, y_arr)) / n if n else 0.0
    spectrum = [complex(covariance, 0.0)] * freq_bins
    return np.asarray(freqs, dtype=float), np.asarray(spectrum, dtype=None)
