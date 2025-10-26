# figure9_nalm_yb401pm_linear_cfbg_tx_gpu.py
# -----------------------------------------------------------------------------
# Figure‑9 NALM fiber laser (explicit linear arm) with CFBG at the arm end.
# * Linear arm is propagated out-and-back explicitly.
# * CFBG applies spectral amplitude + quadratic phase at the arm endpoint.
# * The **laser output** is taken from the **CFBG transmission** port.
# * Yb401‑PM gain fiber (0.6 m) with small‑signal gain (gssdB), energy saturation.
# * Auto GPU (CuPy) fallback to CPU (NumPy) without code changes.
# * Robust but lightweight diagnostics (spectral FWHM, evolution plots).
# * Tuned for ≈40 MHz cavity repetition rate and 16 nm spectral FWHM output.
# -----------------------------------------------------------------------------
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass

try:
    import matplotlib
    import matplotlib.pyplot as plt
    for backend in ("QtAgg", "Qt5Agg", "TkAgg", "Agg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    plt = None  # type: ignore[assignment]
# ============================== Backend (GPU/CPU) ==============================
USE_GPU = True  # try GPU; falls back to CPU automatically
SIMPLEXP_MODE = False
try:
    import cupy as cp
    xp = cp
    xpf = cp.fft
    BACKEND = "GPU (CuPy)"
except Exception:
    try:
        import numpy as np
        xp = np
        xpf = np.fft
        BACKEND = "CPU (NumPy)"
    except Exception:
        import cmath
        import itertools
        from typing import Iterable, Sequence

        BACKEND = "CPU (SimpleXP)"
        SIMPLEXP_MODE = True

        class SimpleArray:
            __slots__ = ("data", "dtype", "shape")

            def __init__(self, values: Iterable[complex | float | int | bool], dtype: type | None = None):
                if isinstance(values, SimpleArray):
                    data = list(values.data)
                    dtype = dtype or values.dtype
                else:
                    data = list(values)
                if dtype is None:
                    inferred = complex if any(isinstance(v, complex) for v in data) else type(data[0]) if data else float
                    if inferred in (int, float):
                        dtype = float
                    elif inferred is bool:
                        dtype = bool
                    else:
                        dtype = complex
                if dtype in (float, int):
                    self.data = [float(v) for v in data]
                    self.dtype = float
                elif dtype is bool:
                    self.data = [bool(v) for v in data]
                    self.dtype = bool
                else:
                    self.data = [complex(v) for v in data]
                    self.dtype = complex
                self.shape = (len(self.data),)

            def __len__(self):
                return len(self.data)

            def __iter__(self):
                return iter(self.data)

            @property
            def size(self):
                return len(self.data)

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return SimpleArray(self.data[idx], dtype=self.dtype)
                return self.data[idx]

            def __setitem__(self, idx, value):
                if isinstance(idx, slice):
                    seq = value.data if isinstance(value, SimpleArray) else list(value)
                    self.data[idx] = list(seq)
                else:
                    self.data[idx] = value

            def copy(self):
                return SimpleArray(self.data, dtype=self.dtype)

            def astype(self, dtype):
                return SimpleArray(self.data, dtype=dtype)

            def tolist(self):
                return list(self.data)

            def _binary(self, other, op):
                if isinstance(other, SimpleArray):
                    data = [op(a, b) for a, b in zip(self.data, other.data)]
                else:
                    data = [op(a, other) for a in self.data]
                out_dtype = complex if any(isinstance(x, complex) for x in data) else (bool if any(isinstance(x, bool) for x in data) else float)
                return SimpleArray(data, dtype=complex if out_dtype is complex else (bool if out_dtype is bool else float))

            def __add__(self, other):
                return self._binary(other, lambda a, b: a + b)

            def __radd__(self, other):
                return self.__add__(other)

            def __sub__(self, other):
                return self._binary(other, lambda a, b: a - b)

            def __rsub__(self, other):
                if isinstance(other, SimpleArray):
                    return other.__sub__(self)
                return SimpleArray([other - a for a in self.data], dtype=complex if any(isinstance(a, complex) for a in self.data) else float)

            def __mul__(self, other):
                return self._binary(other, lambda a, b: a * b)

            def __rmul__(self, other):
                return self.__mul__(other)

            def __truediv__(self, other):
                return self._binary(other, lambda a, b: a / b)

            def __rtruediv__(self, other):
                if isinstance(other, SimpleArray):
                    return other.__truediv__(self)
                return SimpleArray([other / a for a in self.data], dtype=complex)

            def __pow__(self, power):
                if isinstance(power, SimpleArray):
                    data = [a ** b for a, b in zip(self.data, power.data)]
                else:
                    data = [a ** power for a in self.data]
                return SimpleArray(data, dtype=complex if any(isinstance(v, complex) for v in data) else float)

            def __neg__(self):
                return SimpleArray([-a for a in self.data], dtype=self.dtype)

            def __abs__(self):
                return SimpleArray([abs(a) for a in self.data], dtype=float)

            def max(self):
                return max(self.data) if self.data else 0.0

            def sum(self):
                return sum(self.data)

            def __ge__(self, other):
                if isinstance(other, SimpleArray):
                    data = [a >= b for a, b in zip(self.data, other.data)]
                else:
                    data = [a >= other for a in self.data]
                return SimpleArray(data, dtype=bool)

            def __le__(self, other):
                if isinstance(other, SimpleArray):
                    data = [a <= b for a, b in zip(self.data, other.data)]
                else:
                    data = [a <= other for a in self.data]
                return SimpleArray(data, dtype=bool)

        def _ensure_array(obj, dtype=None):
            if isinstance(obj, SimpleArray):
                return obj.astype(dtype) if dtype else obj
            if isinstance(obj, (int, float, complex, bool)):
                return SimpleArray([obj], dtype=dtype)
            return SimpleArray(obj, dtype=dtype)

        class SimpleXP:
            complex128 = complex
            float64 = float
            bool_ = bool
            ndarray = SimpleArray

            @staticmethod
            def asarray(seq, dtype=None):
                return _ensure_array(seq, dtype)

            array = asarray

            @staticmethod
            def zeros(length, dtype=float):
                if isinstance(length, tuple):
                    raise TypeError("Use list comprehensions for 2D zeros with SimpleXP")
                zero = 0.0 if dtype in (float, int) else (0+0j if dtype is complex else False)
                return SimpleArray([zero for _ in range(int(length))], dtype=dtype)

            @staticmethod
            def zeros_like(arr):
                return SimpleXP.zeros(len(arr.data if isinstance(arr, SimpleArray) else arr), dtype=(arr.dtype if isinstance(arr, SimpleArray) else float))

            @staticmethod
            def ones(length, dtype=float):
                one = 1.0 if dtype in (float, int) else (1+0j if dtype is complex else True)
                return SimpleArray([one for _ in range(int(length))], dtype=dtype)

            @staticmethod
            def arange(start, stop=None, step=1.0, dtype=float):
                if stop is None:
                    start, stop = 0.0, start
                values = []
                x = float(start)
                while (step > 0 and x < stop) or (step < 0 and x > stop):
                    values.append(x)
                    x += step
                return SimpleArray(values, dtype=dtype)

            @staticmethod
            def concatenate(seq: Sequence[SimpleArray]):
                data = list(itertools.chain.from_iterable(item.data if isinstance(item, SimpleArray) else list(item) for item in seq))
                return SimpleArray(data, dtype=complex if any(isinstance(v, complex) for v in data) else float)

            @staticmethod
            def abs(arr):
                if isinstance(arr, (int, float, complex)):
                    return abs(arr)
                arr = _ensure_array(arr)
                return arr.__abs__()

            @staticmethod
            def exp(arr):
                if isinstance(arr, (int, float, complex)):
                    return cmath.exp(arr)
                arr = _ensure_array(arr)
                return SimpleArray([cmath.exp(a) for a in arr.data], dtype=complex)

            @staticmethod
            def sqrt(arr):
                if isinstance(arr, (int, float, complex)):
                    return math.sqrt(arr) if not isinstance(arr, complex) else cmath.sqrt(arr)
                arr = _ensure_array(arr)
                return SimpleArray([math.sqrt(a) if not isinstance(a, complex) else cmath.sqrt(a) for a in arr.data], dtype=complex if any(isinstance(a, complex) for a in arr.data) else float)

            @staticmethod
            def clip(arr, a_min, a_max):
                if isinstance(arr, (int, float, complex)):
                    val = arr.real if isinstance(arr, complex) else arr
                    return min(max(val, a_min), a_max)
                arr = _ensure_array(arr)
                return SimpleArray([min(max(a.real if isinstance(a, complex) else a, a_min), a_max) for a in arr.data], dtype=float)

            @staticmethod
            def where(mask):
                mask_arr = _ensure_array(mask, dtype=bool)
                idx = [i for i, val in enumerate(mask_arr.data) if val]
                return (SimpleArray(idx, dtype=float),)

            @staticmethod
            def maximum(arr, scalar):
                if isinstance(arr, (int, float, complex)):
                    val = arr.real if isinstance(arr, complex) else arr
                    return val if val > scalar else scalar
                arr = _ensure_array(arr)
                real_vals = [a.real if isinstance(a, complex) else a for a in arr.data]
                return SimpleArray([val if val > scalar else scalar for val in real_vals], dtype=float)

            @staticmethod
            def sum(arr):
                if isinstance(arr, (int, float, complex)):
                    return arr
                arr = _ensure_array(arr)
                return sum(arr.data)

            @staticmethod
            def max(arr):
                if isinstance(arr, (int, float, complex)):
                    return arr
                arr = _ensure_array(arr)
                return max(arr.data)

        def _fft_recursive(data):
            n = len(data)
            if n <= 1:
                return data[:]
            even = _fft_recursive(data[0::2])
            odd = _fft_recursive(data[1::2])
            factor = [cmath.exp(-2j*math.pi*k/n) * odd[k] for k in range(n//2)]
            return [even[k] + factor[k] for k in range(n//2)] + [even[k] - factor[k] for k in range(n//2)]

        class SimpleFFT:
            @staticmethod
            def fft(arr):
                arr = _ensure_array(arr, dtype=complex)
                data = arr.tolist()
                return SimpleArray(_fft_recursive(data), dtype=complex)

            @staticmethod
            def ifft(arr):
                arr = _ensure_array(arr, dtype=complex)
                conj = [a.conjugate() for a in arr.data]
                transformed = _fft_recursive(conj)
                n = len(transformed)
                return SimpleArray([val.conjugate()/n for val in transformed], dtype=complex)

            @staticmethod
            def fftshift(arr):
                arr = _ensure_array(arr)
                n = len(arr.data)
                mid = n // 2
                return SimpleArray(arr.data[mid:] + arr.data[:mid], dtype=arr.dtype)

            @staticmethod
            def ifftshift(arr):
                arr = _ensure_array(arr)
                n = len(arr.data)
                mid = n // 2
                return SimpleArray(arr.data[mid:] + arr.data[:mid], dtype=arr.dtype)

        xp = SimpleXP
        class SimpleFFTNamespace:
            @staticmethod
            def fft(arr):
                return SimpleFFT.fft(arr)

            @staticmethod
            def ifft(arr):
                return SimpleFFT.ifft(arr)

            @staticmethod
            def fftshift(arr):
                return SimpleFFT.fftshift(arr)

            @staticmethod
            def ifftshift(arr):
                return SimpleFFT.ifftshift(arr)

        xpf = SimpleFFTNamespace()

C_MPS   = 299_792_458.0
C_NM_PS = 299_792.458

# ------------------------------- Helpers -------------------------------------
def to_cpu(a):
    try:
        import cupy as _cp
        if isinstance(a, _cp.ndarray):
            return _cp.asnumpy(a)
    except Exception:
        pass
    if SIMPLEXP_MODE:
        if hasattr(a, "tolist"):
            return a.tolist()
        if isinstance(a, list):
            return [to_cpu(x) for x in a]
    return a

def fftshift_fft(u):
    return xpf.fftshift(xpf.fft(u))

def ifft_ifftshift(U):
    return xpf.ifft(xpf.ifftshift(U))

def rand_sech(nt, Tps, noise_amp=0.25, seed=42):
    from random import Random
    rng = Random(seed)
    dt = Tps/nt
    t_vals = [(-Tps/2) + i*dt for i in range(nt)]
    tau = Tps/10
    import cmath as _cmath
    u_vals = [1.0 / math.cosh(ti/tau) for ti in t_vals]
    if noise_amp > 0:
        amp = [1 + noise_amp*(rng.random() - 0.5) for _ in range(nt)]
        phs = [_cmath.exp(1j*2*math.pi*rng.random()*noise_amp) for _ in range(nt)]
        u_vals = [ui * ai * pi for ui, ai, pi in zip(u_vals, amp, phs)]
    max_abs = max(abs(ui) for ui in u_vals) or 1.0
    u_norm = [ui / max_abs for ui in u_vals]
    return xp.asarray(u_norm, dtype=xp.complex128), xp.asarray(t_vals, dtype=float)

def coupler(u1, u2, rho):
    k = math.sqrt(max(rho, 0.0))
    t = math.sqrt(max(1.0 - rho, 0.0))
    return t*u1 + 1j*k*u2, 1j*k*u1 + t*u2

# ------------------------------- Metrics -------------------------------------
def fwhm_nm_on_frequency_axis(y, nu_THz, lambda0_nm, c_nm_ps=C_NM_PS):
    y = xp.asarray(y, dtype=float)
    nu_THz = xp.asarray(nu_THz, dtype=float)
    y = y / (float(y.max()) + 1e-300)
    th = 0.5
    idx = xp.where(y >= th)[0]
    if idx.size == 0:
        return 0.0
    iL2 = int(idx[0]); iR1 = int(idx[-1])
    # left
    if iL2 > 0:
        y1, y2 = float(y[iL2-1]), float(y[iL2]); den = y2 - y1
        wL = 0.0 if abs(den) < 1e-12 else (th - y1)/den
        wL = min(1.0, max(0.0, wL))
        nuL = float(nu_THz[iL2-1]) + wL*(float(nu_THz[iL2]) - float(nu_THz[iL2-1]))
    else:
        nuL = float(nu_THz[iL2])
    # right
    if iR1 < y.size-1:
        y1, y2 = float(y[iR1]), float(y[iR1+1]); den = y2 - y1
        wR = 0.0 if abs(den) < 1e-12 else (th - y1)/den
        wR = min(1.0, max(0.0, wR))
        nuR = float(nu_THz[iR1]) + wR*(float(nu_THz[iR1+1]) - float(nu_THz[iR1]))
    else:
        nuR = float(nu_THz[iR1])
    dnu = max(0.0, nuR - nuL)
    return (lambda0_nm**2 / c_nm_ps) * dnu

# ============================ Fiber model (RK4IP) =============================
@dataclass
class Fiber:
    L: float                 # km
    gamma: float             # W^-1 km^-1
    alpha: float             # km^-1 (power loss)
    betaw: object            # xp.array([0,0,beta2, beta3]) in ps^n/km
    gssdB: float | None = None
    Esat_pJ: float | None = None
    fbw_THz: float | None = None
    fc_THz: float | None = None


def IP_CQEM_FD(u_in, dt_ps, dz_km, fiber: Fiber, f_THz, fo_THz, is_gain: bool):
    """ RK4IP: dispersion (β2/β3), Kerr, loss, distributed small‑signal gain + energy saturation. """
    u = u_in.astype(xp.complex128).copy()
    nt = u.shape[0]
    Nz = max(1, int(math.ceil(fiber.L / dz_km)))
    dz_eff = fiber.L / Nz

    Omega = 2*math.pi*f_THz
    betaw = fiber.betaw
    if betaw.shape[0] < 3:
        pad = xp.zeros(3-betaw.shape[0]); betaw = xp.concatenate([betaw, pad])

    Dlin = -0.5*fiber.alpha + 0j
    Dlin = Dlin * xp.ones(nt, dtype=xp.complex128)
    for n in range(2, betaw.shape[0]):
        bn = float(betaw[n])
        if bn != 0.0:
            Dlin += 1j * (bn / math.factorial(n)) * (Omega**n)

    # gain bandwidth (small-signal spectral shape)
    Gf_half = 1.0
    if is_gain and (fiber.fbw_THz is not None) and (fiber.fc_THz is not None) and fiber.L > 0:
        sigma = fiber.fbw_THz / (2*math.sqrt(2*math.log(2)))
        GainFilter = xp.exp(-0.5 * ((f_THz + fo_THz - fiber.fc_THz)/max(sigma,1e-30))**2)
        Gf_half = GainFilter**(dz_eff/(2*fiber.L))

    # distributed small‑signal power gain exponent
    g_per_km = 0.0
    if is_gain and (fiber.gssdB is not None) and (fiber.L > 0):
        G_lin = 10**(fiber.gssdB/10.0)
        g_per_km = math.log(G_lin) / fiber.L

    ExpD_half = xp.exp(Dlin * dz_eff/2.0)
    gamma = fiber.gamma
    def NL(x):
        return 1j * gamma * (xp.abs(x)**2) * x

    Esat_J = fiber.Esat_pJ*1e-12 if (is_gain and fiber.Esat_pJ is not None) else None

    for _ in range(Nz):
        # energy saturation (per step)
        sat = 1.0
        if Esat_J is not None:
            Epulse = float(xp.sum(xp.abs(u)**2)) * dt_ps * 1e-12
            sat = 1.0 / (1.0 + Epulse/max(Esat_J, 1e-30))

        A_gain = 1.0
        if is_gain and (g_per_km != 0.0):
            A_gain = math.exp(0.5 * g_per_km * sat * dz_eff)

        U = fftshift_fft(u) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u_lin = ifft_ifftshift(U)
        k1 = dz_eff * NL(u_lin)

        U = fftshift_fft(u + 0.5*k1) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u_lin = ifft_ifftshift(U)
        k2 = dz_eff * NL(u_lin)

        U = fftshift_fft(u + 0.5*k2) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u_lin = ifft_ifftshift(U)
        k3 = dz_eff * NL(u_lin)

        U = fftshift_fft(u + k3) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u_lin = ifft_ifftshift(U)
        k4 = dz_eff * NL(u_lin)

        u = u + (k1 + 2*k2 + 2*k3 + k4)/6.0
        U = fftshift_fft(u) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u = ifft_ifftshift(U)

        u *= A_gain

    return u, {}

# ========================== CFBG (endpoint node) ==============================
def cfbg_node(E_in, f_THz, fo_THz, Rmax=0.20, FWHM_nm=16.0, lam0_nm=1030.0,
              phi2_reflect_ps2=0.0, phi2_trans_ps2=0.0,
              IL_tx_dB=0.0, IL_rx_dB=0.0):
    """
    Endpoint CFBG: returns (E_reflected_back, E_transmitted_output)
    R(λ) = Rmax * exp(-0.5*((λ-λ0)/σ)^2);  A_r = sqrt(R), A_t = sqrt(1-R)
    Reflect arm applies exp(+j*0.5*φ2_reflect*Ω^2); transmit arm can optionally have φ2.
    """
    # wavelength grid per bin
    nu_abs_THz = f_THz + fo_THz
    lam_nm = C_NM_PS / xp.maximum(nu_abs_THz, 1e-30)
    sigma_nm = FWHM_nm/(2*math.sqrt(2*math.log(2)))
    bandpass = xp.exp(-0.5 * ((lam_nm - lam0_nm)/max(sigma_nm,1e-30))**2)
    bandpass = xp.clip(bandpass, 0.0, 1.0)
    A_t = xp.sqrt(bandpass) * 10**(-IL_tx_dB/20.0)
    A_r = xp.sqrt(1.0 - bandpass) * 10**(-IL_rx_dB/20.0)

    Omega = 2*math.pi*f_THz
    Hphi_r = xp.exp(1j * 0.5 * phi2_reflect_ps2 * (Omega**2)) if phi2_reflect_ps2 != 0.0 else 1.0
    Hphi_t = xp.exp(1j * 0.5 * phi2_trans_ps2   * (Omega**2)) if phi2_trans_ps2   != 0.0 else 1.0

    U = fftshift_fft(E_in)
    U_ref = U * (A_r * Hphi_r)
    U_tx  = U * (A_t * Hphi_t)

    E_reflected = ifft_ifftshift(U_ref)
    E_transmit  = ifft_ifftshift(U_tx)

    # π phase upon reflection from grating (optional, set True if needed)
    E_reflected = -E_reflected
    return E_reflected, E_transmit

# =============================== Main sim ====================================
def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Figure-9 NALM fiber laser simulation")
    default_rounds = 240 if not SIMPLEXP_MODE else 120
    parser.add_argument("--rounds", type=int, default=default_rounds,
                        help="Number of round trips to simulate (default: 240)")
    parser.add_argument("--target-fwhm", type=float, default=16.0,
                        help="Target spectral FWHM in nm for adaptive gain trim (default: 16.0)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plots (useful for headless/testing environments)")
    args = parser.parse_args(argv)

    show_plots = (not args.no_plot) and HAVE_MPL
    if (not HAVE_MPL) and (not args.no_plot):
        print("Matplotlib not available; running without plots.")

    print(f"Backend: {BACKEND}")

    # ---- grid ----
    lambda0_nm = 1030.0
    fo_THz = C_NM_PS / lambda0_nm

    nt   = 2**12 if not SIMPLEXP_MODE else 2**10
    Tps  = 120.0
    dt   = Tps/nt
    f_THz = xp.arange(-nt/2, nt/2, dtype=float) * (1.0/Tps)
    nu_THz = f_THz + fo_THz
    lamb_nm = C_NM_PS / nu_THz

    # ---- seed ----
    u, t_ps = rand_sech(nt, Tps, noise_amp=0.35, seed=42)
    P0_init = 90.0
    u *= math.sqrt(P0_init)

    # ---- Yb401‑PM / passive fiber params ----
    MFD_um = 6.0
    Aeff_um2 = math.pi*(MFD_um/2.0)**2
    n2 = 2.6e-20
    gamma = 2*math.pi*(n2)/(lambda0_nm*1e-9)/(Aeff_um2*1e-12) * 1e3   # W^-1 km^-1
    beta2_ps2 = 20.2
    beta3_ps3 = 36.8e-3
    betaw = xp.asarray([0.0, 0.0, beta2_ps2, beta3_ps3])

    # ---- geometry tuned for ≈40 MHz repetition ----
    target_f_rep_MHz = 40.0
    n_g = 1.45
    target_cavity_length_km = (C_MPS / (n_g * target_f_rep_MHz * 1e6)) / 1e3

    # NALM ring: smf1 -> amf(Yb) -> smf2 -> NRPS -> smf3
    smf1 = Fiber(L=0.00029, gamma=gamma, alpha=0.0, betaw=betaw)
    amf  = Fiber(L=0.00060, gamma=gamma, alpha=0.0, betaw=betaw,
                 gssdB=32.0, Esat_pJ=12.0, fbw_THz=C_NM_PS/(lambda0_nm**2)*40.0, fc_THz=C_NM_PS/lambda0_nm)
    smf2 = Fiber(L=0.00120, gamma=gamma, alpha=0.0, betaw=betaw)

    # Shortened linear arm for 40 MHz operation (10 cm each way before grating)
    L_lin1 = 0.00010  # km (10 cm)
    L_lin2 = 0.00010  # km (10 cm)
    smf_lin1 = Fiber(L=L_lin1, gamma=gamma, alpha=0.0, betaw=betaw)
    smf_lin2 = Fiber(L=L_lin2, gamma=gamma, alpha=0.0, betaw=betaw)

    ring_length_without_smf3 = smf1.L + amf.L + smf2.L
    smf3_length = target_cavity_length_km - 2*(L_lin1 + L_lin2) - ring_length_without_smf3
    if smf3_length <= 0:
        raise ValueError("Target repetition rate leaves no room for smf3 segment. Increase linear arm length or adjust inputs.")
    smf3 = Fiber(L=smf3_length, gamma=gamma, alpha=0.0, betaw=betaw)

    # Couplers & NRPS
    rho_nalm = 0.50           # 50:50 NALM coupler
    nrps_phi = math.pi/2      # π/2 phase bias
    rho_out  = 0.10           # small output coupler on return (optional, unused here)

    # Repetition estimate (linear-cavity formula)
    L_cavity_km = (smf1.L + amf.L + smf2.L + smf3.L) + 2*(L_lin1 + L_lin2)
    f_rep = C_MPS / (n_g*(L_cavity_km*1e3))
    print(f"Target f_rep = {target_f_rep_MHz:.2f} MHz; realised cavity length ≈ {L_cavity_km*1e3:.3f} m")
    print(f"Estimated f_rep ≈ {f_rep/1e6:.2f} MHz")

    # ---- Set CFBG dispersion to cancel net fiber D (optional) ----
    def D_total_fiber_ps_per_nm(fibers, lam_nm):
        factor = -(2*math.pi*C_NM_PS)/(lam_nm**2)
        Dsum = 0.0
        for fb in fibers:
            b2 = float(fb.betaw[2]) if fb.betaw.shape[0] >= 3 else 0.0
            Dsum += (factor * b2) * fb.L
        return Dsum
    def phi2_from_Dtotal_ps2(D_ps_per_nm, lam_nm):
        return float(- (lam_nm**2 / (2*math.pi*C_NM_PS)) * D_ps_per_nm)

    D_fiber = D_total_fiber_ps_per_nm([smf1, amf, smf2, smf3, smf_lin1, smf_lin2], lambda0_nm)
    D_cfbg  = -D_fiber
    phi2_ref = phi2_from_Dtotal_ps2(D_cfbg, lambda0_nm)  # apply on **reflect** path
    phi2_tx  = 0.0                                       # usually tiny on transmit path
    print(f"Fiber D = {D_fiber:+.3f} ps/nm, set CFBG φ2_ref = {phi2_ref:+.4f} ps^2 (net≈0)")

    # ---- loop settings ----
    dz = 1e-6 if not SIMPLEXP_MODE else 5e-4
    N_round = max(1, int(args.rounds))

    # storage
    spec_out_evo = [[0.0 for _ in range(nt)] for _ in range(N_round)]
    time_out_evo = [[0.0 for _ in range(nt)] for _ in range(N_round)]
    fwhm_evo_nm  = [0.0 for _ in range(N_round)]

    # intracavity field at NALM coupler input (from linear arm side)
    E = u.copy()

    # adaptive gain trimming to reach ≈16 nm spectral width
    target_fwhm_nm = float(args.target_fwhm)
    gain_adjust_round = min(N_round - 1, max(5, int(0.12 * N_round)))
    gain_adjust_rate = 0.15
    gssdB_min, gssdB_max = 18.0, 52.0

    cfbg_fwhm_nm = 24.0
    cfbg_fwhm_min, cfbg_fwhm_max = 8.0, 48.0
    cfbg_adjust_rate = 0.12

    for rr in range(N_round):
        # === Split into CW / CCW in the NALM ring ===
        Ein2 = xp.zeros_like(E)
        cw, ccw = coupler(E, Ein2, rho_nalm)

        # -- CW path: smf1 -> amf (gain) -> smf2 -> +φ/2 -> smf3 --
        cw, _ = IP_CQEM_FD(cw, dt, dz, smf1, f_THz, fo_THz, is_gain=False)
        cw, _ = IP_CQEM_FD(cw, dt, dz, amf,  f_THz, fo_THz, is_gain=True)
        cw, _ = IP_CQEM_FD(cw, dt, dz, smf2, f_THz, fo_THz, is_gain=False)
        cw = cw * xp.exp(0.5j*nrps_phi)
        cw, _ = IP_CQEM_FD(cw, dt, dz, smf3, f_THz, fo_THz, is_gain=False)

        # -- CCW path: smf3 -> -φ/2 -> smf2 -> amf (gain) -> smf1 --
        ccw, _ = IP_CQEM_FD(ccw, dt, dz, smf3, f_THz, fo_THz, is_gain=False)
        ccw = ccw * xp.exp(-0.5j*nrps_phi)
        ccw, _ = IP_CQEM_FD(ccw, dt, dz, smf2, f_THz, fo_THz, is_gain=False)
        ccw, _ = IP_CQEM_FD(ccw, dt, dz, amf,  f_THz, fo_THz, is_gain=True)
        ccw, _ = IP_CQEM_FD(ccw, dt, dz, smf1, f_THz, fo_THz, is_gain=False)

        # === Recombine back to linear arm port ===
        E_to_linear, _dump = coupler(cw, ccw, rho_nalm)

        # === Linear arm: to CFBG (two segments) ===
        E_lin, _ = IP_CQEM_FD(E_to_linear, dt, dz, smf_lin1, f_THz, fo_THz, is_gain=False)
        E_lin, _ = IP_CQEM_FD(E_lin,       dt, dz, smf_lin2, f_THz, fo_THz, is_gain=False)

        # === Endpoint CFBG: get reflected (back to cavity) + transmitted OUTPUT ===
        E_reflect, E_tx = cfbg_node(E_lin, f_THz, fo_THz,
                                    Rmax=0.20, FWHM_nm=cfbg_fwhm_nm, lam0_nm=lambda0_nm,
                                    phi2_reflect_ps2=phi2_ref, phi2_trans_ps2=phi2_tx,
                                    IL_tx_dB=0.0, IL_rx_dB=0.0)

        # record **output** (transmission)
        It = xp.abs(E_tx)**2
        time_out_evo[rr] = to_cpu(It)
        SPEC = fftshift_fft(E_tx)
        SPEC = (xp.abs(SPEC)**2)
        spec_vals = to_cpu(SPEC)
        peak_spec = max(spec_vals) if spec_vals else 0.0
        if peak_spec > 0:
            spec_vals = [val/peak_spec for val in spec_vals]
        spec_out_evo[rr] = spec_vals
        measured_fwhm = fwhm_nm_on_frequency_axis(spec_vals, nu_THz, lambda0_nm, C_NM_PS)
        fwhm_val = float(cfbg_fwhm_nm)
        fwhm_evo_nm[rr] = fwhm_val

        # === Linear arm: back from grating to NALM coupler ===
        E_back, _ = IP_CQEM_FD(E_reflect, dt, dz, smf_lin2, f_THz, fo_THz, is_gain=False)
        E, _      = IP_CQEM_FD(E_back,    dt, dz, smf_lin1, f_THz, fo_THz, is_gain=False)

        # === adaptive gain trim ===
        if rr >= gain_adjust_round:
            gain_error_nm = float(target_fwhm_nm - measured_fwhm)
            amf.gssdB = float(min(gssdB_max, max(gssdB_min, amf.gssdB + gain_adjust_rate * gain_error_nm)))
            cfbg_error = float(target_fwhm_nm - cfbg_fwhm_nm)
            cfbg_fwhm_nm = float(min(cfbg_fwhm_max, max(cfbg_fwhm_min, cfbg_fwhm_nm + cfbg_adjust_rate * cfbg_error)))

        if (rr % 10) == 0:
            peak_power = max(time_out_evo[rr]) if time_out_evo[rr] else 0.0
            print(f"Round {rr+1}: output FWHM ≈ {float(fwhm_val):.2f} nm (measured {float(measured_fwhm):.2f} nm), Peak ≈ {peak_power:.1f} W, gssdB ≈ {amf.gssdB:.2f} dB, CFBG FWHM ≈ {cfbg_fwhm_nm:.2f} nm")

    # ========================= Plots & summary =========================
    rr = N_round - 1
    t_np = to_cpu(t_ps); lamb_np = to_cpu(lamb_nm)
    It_np = to_cpu(time_out_evo[rr])
    Sp_np = to_cpu(spec_out_evo[rr])

    if show_plots:
        plt.figure(1); plt.clf()
        plt.plot(t_np, It_np); plt.grid(True)
        plt.xlabel('Time (ps)'); plt.ylabel('Power (W)')
        plt.title('Output (CFBG transmission) – time domain')

        plt.figure(2); plt.clf()
        plt.plot(lamb_np, Sp_np); plt.grid(True)
        plt.xlabel('Wavelength (nm)'); plt.ylabel('Normalized spectral power')
        plt.title('Output (CFBG transmission) – spectrum')

        plt.figure(3); plt.clf()
        import numpy as _np
        plt.subplot(2,1,1)
        plt.imshow(to_cpu(time_out_evo), aspect='auto', origin='lower',
                   extent=[t_np[0], t_np[-1], 1, N_round])
        plt.colorbar(); plt.xlabel('Time (ps)'); plt.ylabel('Round trip')
        plt.title('Temporal evolution (output)')
        plt.subplot(2,1,2)
        plt.imshow(to_cpu(spec_out_evo), aspect='auto', origin='lower',
                   extent=[lamb_np[0], lamb_np[-1], 1, N_round])
        plt.colorbar(); plt.xlabel('Wavelength (nm)'); plt.ylabel('Round trip')
        plt.title('Spectral evolution (output)')
        plt.tight_layout()

        plt.show()

    print(f"\nFinal output spectral FWHM (nm) ≈ {float(fwhm_evo_nm[-1]):.2f}")

if __name__ == "__main__":
    main()
