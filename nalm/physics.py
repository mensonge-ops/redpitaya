"""Low-level optical component models used by the NALM simulation."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

__all__ = [
    "coupler",
    "filter_gauss",
    "filter_lorentz_tf",
    "fwhm",
    "gain_saturated2",
    "Linearoperator_w",
    "NonLinearoperator_w",
    "Raman_response_w",
]


def _as_array(value: Iterable[complex] | complex, *, length: int | None = None) -> np.ndarray:
    """Convert *value* to a complex NumPy array."""

    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 0:
        if length is None:
            return np.array([array], dtype=np.complex128)
        return np.full(length, array, dtype=np.complex128)
    if length is not None and array.size != length:
        raise ValueError(f"Expected array of length {length}, got {array.size}")
    return array


def coupler(u1i: np.ndarray, u2i: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    """Model a lossless 2x2 coupler."""

    rho = float(np.clip(rho, 0.0, 1.0))
    u1i = np.asarray(u1i, dtype=np.complex128)
    u2i = np.asarray(u2i, dtype=np.complex128)
    if u1i.shape != u2i.shape:
        raise ValueError("Input fields must share the same shape")

    root_rho = math.sqrt(rho)
    root_one_minus_rho = math.sqrt(1.0 - rho)

    u1o = root_rho * u1i + 1j * root_one_minus_rho * u2i
    u2o = 1j * root_one_minus_rho * u1i + root_rho * u2i
    return u1o, u2o


def filter_gauss(
    ui: np.ndarray,
    f3dB: float,
    fc: float,
    order: int,
    fo: float,
    df: float,
) -> np.ndarray:
    """Apply an n-th order Gaussian filter in the frequency domain."""

    ui = np.asarray(ui, dtype=np.complex128)
    n = ui.size
    Ui = np.fft.fft(ui)
    freq = np.fft.fftshift(np.arange(-n // 2, n // 2) * df + fo)
    tf = np.exp(-math.log(math.sqrt(2.0)) * ((2.0 / f3dB) * (freq - fc)) ** (2 * order))
    Ui_filtered = np.fft.ifftshift(np.fft.fftshift(Ui) * tf)
    return np.fft.ifft(Ui_filtered)


def filter_lorentz_tf(
    ui: np.ndarray,
    fbw: float,
    fc: float,
    fo: float,
    df: float,
) -> np.ndarray:
    """Return the transfer function of a Lorentzian filter."""

    ui = np.asarray(ui, dtype=np.complex128)
    n = ui.size
    freq = np.fft.fftshift(np.arange(-n // 2, n // 2) * df + fo)
    tf = (fbw) / (2.0 * math.pi) / ((freq - fc) ** 2 + (fbw / 2.0) ** 2)
    tf /= np.max(np.abs(tf))
    return tf.astype(np.complex128)


def fwhm(x: np.ndarray) -> Tuple[int, int, int]:
    """Return the full width at half maximum (FWHM) of *x*."""

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be one-dimensional")
    peak_index = int(np.argmax(x))
    peak_value = float(x[peak_index])
    if peak_value <= 0:
        return 0, peak_index, peak_index

    half_level = peak_value / 2.0
    left = peak_index
    while left > 0 and x[left] > half_level:
        left -= 1
    right = peak_index
    while right < x.size - 1 and x[right] > half_level:
        right += 1
    width = right - left
    return width, left, right


def gain_saturated2(Pin: float, gssdB: float, PsatdBm: float) -> float:
    """Return the saturated gain coefficient."""

    gss = 10.0 ** (gssdB / 10.0)
    Psat = (10.0 ** (PsatdBm / 10.0)) / 1000.0
    return gss / (1.0 + Pin / Psat)


def Linearoperator_w(alpha: complex | Iterable[complex], betaw: Iterable[complex], w: np.ndarray) -> np.ndarray:
    """Construct the linear operator in the frequency domain."""

    w = np.asarray(w, dtype=np.complex128)
    alpha_array = _as_array(alpha, length=w.size)
    lop = -0.5 * np.fft.fftshift(alpha_array)

    betaw_array = np.asarray(betaw, dtype=np.complex128)
    if betaw_array.size == w.size:
        lop -= 1j * betaw_array
        lop = np.fft.fftshift(lop)
    else:
        lop = lop.astype(np.complex128)
        for order, coef in enumerate(betaw_array):
            lop -= 1j * coef * (w ** order) / math.factorial(order)
    return lop


def Raman_response_w(t: np.ndarray, mod) -> Tuple[np.ndarray, float]:
    """Return the Raman response in the frequency domain."""

    raman_enabled = bool(getattr(mod, "raman", True))
    if not raman_enabled:
        hrw = np.zeros_like(t, dtype=np.complex128)
        fr = 0.0
        return hrw, fr

    t1 = 12.2e-3
    t2 = 32e-3
    tb = 96e-3
    fc = 0.04
    fb = 0.21
    fa = 1.0 - fc - fb
    fr = 0.245

    tres = t - t[0]
    ha = ((t1 ** 2 + t2 ** 2) / (t1 * t2 ** 2)) * np.exp(-tres / t2) * np.sin(tres / t1)
    hb = ((2.0 * tb - tres) / tb**2) * np.exp(-tres / tb)
    hr = (fa + fc) * ha + fb * hb

    return np.fft.fft(hr.astype(np.complex128)), fr


def NonLinearoperator_w(
    u_t: np.ndarray,
    gamma: float,
    w: np.ndarray,
    fo: float,
    fr: float,
    hrw: np.ndarray,
    dt: float,
    mod,
) -> np.ndarray:
    """Evaluate the nonlinear operator in the frequency domain."""

    u_t = np.asarray(u_t, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    power = np.abs(u_t) ** 2
    nonlinear_term = (1.0 - fr) * u_t * power

    if fr != 0.0 and np.any(hrw):
        convolution = np.fft.ifft(hrw * np.fft.fft(power))
        nonlinear_term += fr * dt * u_t * convolution

    spectrum = np.fft.fft(nonlinear_term)
    include_ssp = bool(getattr(mod, "ssp", True))
    if include_ssp:
        prefactor = -1j * gamma * (1.0 + w / (2.0 * math.pi * fo))
    else:
        prefactor = -1j * gamma
    return prefactor * spectrum
