#!/usr/bin/env python3
"""Numerical simulation of a nonlinear amplifying loop mirror (NALM).

This module ports the reference MATLAB implementation to Python and adapts it
for the MMTools/Red Pitaya environment.  The code follows the interaction
picture combined with a conserved quantity error metric (IP-CQEM) to solve the
Generalized Nonlinear Schrödinger Equation (GNLSE) in optical fibres.  The
simulation reproduces the full cavity used in the original script – a NALM
formed by several passive and active fibre segments, intracavity couplers, and a
spectral filter.

The implementation keeps the structure of the MATLAB prototype while adding a
Pythonic interface, dataclasses for the fibre segments, optional plotting, and
extensive diagnostics that can be consumed programmatically.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# The simulation frequently runs on headless servers.  Use a non-interactive
# backend so figures can be exported without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FiberSegment:
    """Parameters describing a single fibre or gain segment."""

    L: float
    alpha: float
    betaw: np.ndarray
    gamma: float
    raman: bool = True
    ssp: bool = True
    gssdB: Optional[float] = None
    PsatdBm: Optional[float] = None
    fbw: Optional[float] = None
    fc: Optional[float] = None
    name: str = "segment"

    def __post_init__(self) -> None:
        self.betaw = np.array(self.betaw, dtype=float)

    def clone(self, **kwargs: float) -> "FiberSegment":
        params = {
            "L": self.L,
            "alpha": self.alpha,
            "betaw": self.betaw.copy(),
            "gamma": self.gamma,
            "raman": self.raman,
            "ssp": self.ssp,
            "gssdB": self.gssdB,
            "PsatdBm": self.PsatdBm,
            "fbw": self.fbw,
            "fc": self.fc,
            "name": self.name,
        }
        params.update(kwargs)
        return FiberSegment(**params)


@dataclass
class PropagationTrace:
    """Diagnostics collected along a propagation segment."""

    z: np.ndarray
    ufft: np.ndarray
    u: np.ndarray


@dataclass
class SimulationResult:
    """Container holding the high-level results of a cavity simulation."""

    time: np.ndarray
    frequency: np.ndarray
    wavelength: np.ndarray
    input_field: np.ndarray
    output_field: np.ndarray
    spectrum_history: np.ndarray
    field_history: np.ndarray
    segments: Dict[str, PropagationTrace]
    combined_trace: PropagationTrace
    key_positions: Dict[str, int]
    key_distances: Dict[str, float]
    diagnostics: Dict[str, float]


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------


def coupler(u1i: np.ndarray, u2i: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    """Directional coupler used in the NALM.

    Parameters
    ----------
    u1i, u2i:
        Input complex envelopes at both ports.  Scalars are promoted to arrays.
    rho:
        Power coupling ratio in [0, 1].

    Returns
    -------
    Tuple containing the output envelopes ``(u1o, u2o)``.
    """

    rho = float(np.clip(rho, 0.0, 1.0))
    u1 = np.asarray(u1i, dtype=np.complex128)
    u2 = np.asarray(u2i, dtype=np.complex128)
    if u2.shape != u1.shape:
        u2 = np.broadcast_to(u2, u1.shape)

    sqrt_rho = math.sqrt(rho)
    sqrt_comp = math.sqrt(max(1.0 - rho, 0.0))
    u1o = sqrt_rho * u1 + 1j * sqrt_comp * u2
    u2o = 1j * sqrt_comp * u1 + sqrt_rho * u2
    return u1o, u2o


def filter_gauss(
    ui: np.ndarray,
    f3db: float,
    fc: float,
    order: int,
    fo: float,
    df: float,
) -> np.ndarray:
    """Apply an *n*-order Gaussian spectral filter."""

    Ui = np.fft.fft(ui)
    N = Ui.size
    freq = np.fft.fftshift(np.arange(-N // 2, N // 2) * df + fo)
    transfer = np.exp(-math.log(math.sqrt(2.0)) * ((2.0 / f3db) * (freq - fc)) ** (2 * order))
    return np.fft.ifft(Ui * transfer)


def filter_lorentz_tf(
    ui: np.ndarray,
    fbw: float,
    fc: float,
    fo: float,
    df: float,
) -> np.ndarray:
    """Return the transfer function of a Lorentzian filter."""

    N = ui.size
    freq = np.arange(-N // 2, N // 2) * df + fo
    tf = (fbw / (2.0 * math.pi)) / ((freq - fc) ** 2 + (fbw / 2.0) ** 2)
    tf /= np.max(tf)
    return tf


def fwhm(x: np.ndarray) -> Tuple[int, int, int]:
    """Full width at half maximum for a positive envelope."""

    arr = np.asarray(x)
    peak_idx = int(np.argmax(arr))
    peak_value = arr[peak_idx]
    half = peak_value / 2.0

    left_indices = np.where(arr[: peak_idx + 1] <= half)[0]
    if left_indices.size:
        left = left_indices[-1]
    else:
        left = 0

    right_indices = np.where(arr[peak_idx:] <= half)[0]
    if right_indices.size:
        right = peak_idx + right_indices[0]
    else:
        right = arr.size - 1

    width = max(right - left, 1)
    return width, left, right


def gain_saturated2(Pin: float, gssdB: float, PsatdBm: float) -> float:
    """Return the saturated gain for a given input power."""

    gss = 10 ** (gssdB / 10.0)
    Psat = 10 ** (PsatdBm / 10.0) / 1000.0
    return gss / (1.0 + Pin / Psat)


# ---------------------------------------------------------------------------
# Operators for the GNLSE solver
# ---------------------------------------------------------------------------


def Raman_response_w(t: np.ndarray, segment: FiberSegment) -> Tuple[np.ndarray, float]:
    """Raman response in the frequency domain."""

    if not segment.raman:
        return np.zeros_like(t, dtype=np.complex128), 0.0

    t1 = 12.2e-3
    t2 = 32e-3
    tb = 96e-3
    fc = 0.04
    fb = 0.21
    fa = 1.0 - fc - fb
    fr = 0.245

    tres = t - t[0]
    ha = ((t1 ** 2 + t2 ** 2) / (t1 * t2 ** 2)) * np.exp(-tres / t2) * np.sin(tres / t1)
    hb = ((2 * tb - tres) / tb**2) * np.exp(-tres / tb)
    hr = (fa + fc) * ha + fb * hb
    return np.fft.fft(hr), fr


def NonLinearoperator_w(
    u_t: np.ndarray,
    gamma: float,
    w: np.ndarray,
    fo: float,
    fr: float,
    hrw: np.ndarray,
    dt: float,
    segment: FiberSegment,
) -> np.ndarray:
    """Frequency-domain nonlinear operator for the GNLSE."""

    envelope = np.abs(u_t) ** 2
    if fr:
        convolution = np.fft.ifft(hrw * np.fft.fft(envelope))
        response = (1.0 - fr) * u_t * envelope + fr * dt * u_t * convolution
    else:
        response = u_t * envelope

    ssp_factor = 1.0
    if segment.ssp:
        ssp_factor = 1.0 + w / (2.0 * math.pi * fo)

    return -1j * gamma * ssp_factor * np.fft.fft(response)


def Linearoperator_w(alpha: np.ndarray, betaw: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear operator in the frequency domain."""

    if np.isscalar(alpha):
        lop = np.full_like(w, -float(alpha) / 2.0, dtype=np.complex128)
        alpha_vec = np.full_like(w, float(alpha), dtype=float)
    else:
        alpha_vec = np.asarray(alpha, dtype=float)
        lop = -np.fft.fftshift(alpha_vec) / 2.0
        lop = lop.astype(np.complex128)

    if betaw.size == w.size:
        lop -= 1j * betaw
        lop = np.fft.fftshift(lop)
    else:
        for order, coef in enumerate(betaw):
            lop -= 1j * coef * (w ** order) / math.factorial(order)

    return lop


# ---------------------------------------------------------------------------
# IP-CQEM solver
# ---------------------------------------------------------------------------


def IP_CQEM_FD(
    u0: np.ndarray,
    dt: float,
    dz: float,
    segment: FiberSegment,
    fo: float,
    tol: float,
    collect_trace: bool,
    quiet: bool,
) -> Tuple[np.ndarray, int, PropagationTrace]:
    """Propagate a pulse through a fibre segment using IP-CQEM."""

    nt = u0.size
    w = np.fft.fftshift(2.0 * math.pi * np.arange(-nt // 2, nt // 2) / (dt * nt))
    t = np.arange(-nt // 2, nt // 2) * dt
    hrw, fr = Raman_response_w(t, segment)

    ufft = np.fft.fft(u0)
    u1 = u0.astype(np.complex128, copy=True)
    propagated = 0.0
    nf = 1

    if segment.gssdB is not None:
        gain_w = filter_lorentz_tf(u1, segment.fbw or 0.0, segment.fc or 0.0, fo, 1.0 / (dt * nt))
        alpha_base = segment.alpha
    else:
        gain_w = None
        alpha_base = segment.alpha

    trace_z: List[float] = []
    trace_ufft: List[np.ndarray] = []
    trace_u: List[np.ndarray] = []

    while propagated < segment.L - 1e-15:
        step = min(dz, segment.L - propagated)

        if segment.gssdB is not None and gain_w is not None:
            Pin0 = float(np.sum(u1 * np.conj(u1)).real / nt)
            gain = gain_saturated2(Pin0, segment.gssdB, segment.PsatdBm or 0.0)
            alpha_vec = alpha_base - gain * gain_w
        else:
            alpha_vec = alpha_base

        LOP = Linearoperator_w(alpha_vec, segment.betaw, w)

        denom = w + 2.0 * math.pi * fo
        denom[np.isclose(denom, 0.0)] = np.finfo(float).tiny

        photon_n = np.sum((np.abs(ufft) ** 2) / denom)
        if np.isscalar(alpha_vec):
            alpha_fft = np.full_like(w, float(alpha_vec), dtype=float)
        else:
            alpha_fft = np.asarray(alpha_vec, dtype=float)
        photon_n_z = np.sum(np.exp(-step * np.fft.fftshift(alpha_fft)) * (np.abs(ufft) ** 2) / denom)

        halfstep = np.exp(LOP * step / 2.0)
        uip = halfstep * ufft
        k1 = halfstep * step * NonLinearoperator_w(u1, segment.gamma, w, fo, fr, hrw, dt, segment)

        uhalf2 = np.fft.ifft(uip + 0.5 * k1)
        k2 = step * NonLinearoperator_w(uhalf2, segment.gamma, w, fo, fr, hrw, dt, segment)

        uhalf3 = np.fft.ifft(uip + 0.5 * k2)
        k3 = step * NonLinearoperator_w(uhalf3, segment.gamma, w, fo, fr, hrw, dt, segment)

        uhalf4 = np.fft.ifft(halfstep * (uip + k3))
        k4 = step * NonLinearoperator_w(uhalf4, segment.gamma, w, fo, fr, hrw, dt, segment)

        uaux = halfstep * (uip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0) + k4 / 6.0
        error = abs(np.sum((np.abs(uaux) ** 2) / denom) - photon_n_z) / abs(photon_n_z)

        if error > 2.0 * tol:
            dz = step / 2.0
            continue

        propagated += step
        ufft = uaux
        u1 = np.fft.ifft(ufft)

        if error > tol:
            dz = step / (2.0 ** 0.2)
        elif error < 0.5 * tol:
            dz = step * (2.0 ** 0.2)
        else:
            dz = step

        if collect_trace:
            trace_z.append(propagated)
            trace_ufft.append(np.abs(np.fft.fftshift(ufft)))
            trace_u.append(np.abs(u1))

        nf += 16

        if not quiet:
            print(f"    z = {propagated:.5f} km (step {step:.5f} km, error {error:.2e})")

    if collect_trace and trace_ufft:
        trace = PropagationTrace(
            z=np.array(trace_z, dtype=float),
            ufft=np.vstack(trace_ufft),
            u=np.vstack(trace_u),
        )
    else:
        trace = PropagationTrace(
            z=np.empty((0,), dtype=float),
            ufft=np.empty((0, nt), dtype=float),
            u=np.empty((0, nt), dtype=float),
        )

    return u1, nf, trace


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------


def simulate_nalm(
    trips: int = 25,
    output_dir: Optional[Path] = None,
    plot: bool = True,
    quiet: bool = False,
) -> SimulationResult:
    """Run a full NALM cavity simulation."""

    c = 299_792.458
    N2 = 1.0
    tfwhm = 50.0
    lamda_pulse = 1550.0
    fo = c / lamda_pulse

    smf1_Aeff = 42.6488
    smf1_n2 = 30.0
    gamma = 2.0 * math.pi * smf1_n2 / lamda_pulse / smf1_Aeff * 1e4

    smf1 = FiberSegment(
        L=0.0015,
        alpha=0.0,
        betaw=np.array([0.0, 0.0, 17.8777, 39.4876e-6]),
        gamma=gamma,
        raman=False,
        ssp=False,
        name="SMF1",
    )

    amf1 = smf1.clone(
        L=0.00015,
        gssdB=20.0,
        PsatdBm=41.0,
        fbw=c / (lamda_pulse**2) * 100.0,
        fc=c / lamda_pulse,
        name="AMF1",
    )

    smf2 = smf1.clone(L=0.002 - smf1.L - amf1.L, name="SMF2")
    smf3 = smf1.clone(L=0.0005, name="SMF3")
    smf4 = smf1.clone(name="SMF4")
    smf5 = smf1.clone(L=0.0115, name="SMF5")

    amf2 = amf1.clone(L=0.004, gssdB=35.0, name="AMF2")

    rho = 0.45
    rho_out = 0.35

    nt = 2**11
    time_window = 70.0
    dt = time_window / nt
    t = np.linspace(-time_window / 2.0, time_window / 2.0, nt, endpoint=False)

    df = 1.0 / (nt * dt)
    f = np.arange(-nt // 2, nt // 2) * df
    wavelength = c / (f + fo)

    dz = 1e-5
    tol = 2e-4

    P_peak = 2.0 * N2 * abs(smf1.betaw[2]) / smf1.gamma / tfwhm**2
    u0 = np.sqrt(P_peak) / np.cosh(t / tfwhm)
    rng = np.random.default_rng(0)
    noise_scale = math.sqrt(10 ** (25.0 / 10.0))
    u0 = u0 * rng.normal(0.0, noise_scale, size=nt)

    peak_power = float(np.max(np.abs(u0) ** 2))
    pulse_energy = float(dt * np.sum(np.abs(u0) ** 2))

    if not quiet:
        print("----------------------------------------------")
        print(f"Input peak power (W) = {peak_power:5.2f}")
        print(f"Input pulse energy (pJ) = {pulse_energy:5.2f}")
        print("Starting IP-CQEM propagation...")

    spec_history: List[np.ndarray] = []
    field_history: List[np.ndarray] = []

    u = u0.astype(np.complex128, copy=True)
    last_traces: Dict[str, PropagationTrace] = {}
    last_ut = np.zeros_like(u)
    last_uf = np.zeros_like(u)
    last_uout = np.zeros_like(u)

    start_time = time.perf_counter()

    for trip in range(1, trips + 1):
        if not quiet:
            print(f"Trip {trip}/{trips}")

        u, _, trace_smf4 = IP_CQEM_FD(u, dt, dz, smf4, fo, tol, True, quiet)
        u, _, trace_amf2 = IP_CQEM_FD(u, dt, dz, amf2, fo, tol, True, quiet)
        u, _, trace_smf5 = IP_CQEM_FD(u, dt, dz, smf5, fo, tol, True, quiet)

        uf, ub = coupler(u, np.zeros_like(u), rho)

        ufo, _, _ = IP_CQEM_FD(uf, dt, dz, smf1, fo, tol, True, quiet)
        ufo, _, _ = IP_CQEM_FD(ufo, dt, dz, amf1, fo, tol, True, quiet)
        ufo, _, _ = IP_CQEM_FD(ufo, dt, dz, smf2, fo, tol, True, quiet)

        ubo, _, _ = IP_CQEM_FD(ub, dt, dz, smf2, fo, tol, True, quiet)
        ubo, _, _ = IP_CQEM_FD(ubo, dt, dz, amf1, fo, tol, True, quiet)
        ubo, _, _ = IP_CQEM_FD(ubo, dt, dz, smf1, fo, tol, True, quiet)

        ur, ut = coupler(ubo, ufo, rho)
        u = ut

        u, _, trace_smf3 = IP_CQEM_FD(u, dt, dz, smf3, fo, tol, True, quiet)
        u, uout = coupler(u, np.zeros_like(u), rho_out)

        u_f = filter_gauss(u, c / (lamda_pulse**2) * 30.0, c / lamda_pulse, 1, fo, df)
        u = u_f

        spec = np.fft.fftshift(np.abs(np.fft.fft(uout)) ** 2)
        spec_norm = spec / (wavelength**2)
        spec_norm /= np.max(spec_norm)

        spec_history.append(spec_norm)
        field_history.append(uout.copy())

        last_traces = {
            "SMF4": trace_smf4,
            "AMF2": trace_amf2,
            "SMF5": trace_smf5,
            "SMF3": trace_smf3,
        }
        last_ut = ut.copy()
        last_uout = uout.copy()
        last_uf = u_f.copy()

    elapsed = time.perf_counter() - start_time

    if not quiet:
        print(f"Simulation finished in {elapsed:5.2f} s")

    spec_history_arr = np.vstack(spec_history)
    field_history_arr = np.vstack(field_history)

    ut_fft = np.tile(np.abs(np.fft.fftshift(np.fft.fft(last_ut))), (20, 1))
    uf_fft = np.tile(np.abs(np.fft.fftshift(np.fft.fft(last_uf))), (20, 1))
    uout_fft = np.tile(np.abs(np.fft.fftshift(np.fft.fft(last_uout))), (20, 1))

    combined_ufft = np.vstack([
        last_traces["SMF4"].ufft,
        last_traces["AMF2"].ufft,
        last_traces["SMF5"].ufft,
        ut_fft,
        last_traces["SMF3"].ufft,
        uout_fft,
        uf_fft,
    ])

    combined_u = np.vstack([
        last_traces["SMF4"].u,
        last_traces["AMF2"].u,
        last_traces["SMF5"].u,
        np.tile(np.abs(last_ut), (20, 1)),
        last_traces["SMF3"].u,
        np.tile(np.abs(last_uout), (20, 1)),
        np.tile(np.abs(last_uf), (20, 1)),
    ])

    segments_len = [
        last_traces["SMF4"].ufft.shape[0],
        last_traces["AMF2"].ufft.shape[0],
        last_traces["SMF5"].ufft.shape[0],
        20,
        last_traces["SMF3"].ufft.shape[0],
        20,
        20,
    ]
    cum_segments = np.concatenate([[0], np.cumsum(segments_len)])

    key_labels = [
        "SMF4入口",
        "SMF4中间",
        "SMF4出口",
        "AMF2入口",
        "AMF2中间",
        "AMF2出口",
        "SMF5入口",
        "SMF5中间",
        "SMF5出口",
        "NALM透射光",
        "SMF3入口",
        "SMF3中间",
        "SMF3出口",
        "输出耦合点",
        "滤波器输出",
    ]

    key_indices: List[int] = []

    def append_positions(segment_index: int, has_middle: bool = True) -> None:
        start = cum_segments[segment_index]
        length = segments_len[segment_index]
        if length == 0:
            key_indices.extend([start] * (3 if has_middle else 1))
            return
        entry = start
        exit_pos = start + length - 1
        if has_middle:
            mid = start + max(int(round(length / 2.0)) - 1, 0)
            key_indices.extend([entry, mid, exit_pos])
        else:
            mid = start + length // 2
            key_indices.append(mid)

    append_positions(0)
    append_positions(1)
    append_positions(2)
    append_positions(3, has_middle=False)
    append_positions(4)
    append_positions(5, has_middle=False)
    append_positions(6, has_middle=False)

    z_vector = np.zeros(cum_segments[-1], dtype=float)
    current_z = 0.0

    if segments_len[0]:
        z_vector[cum_segments[0] : cum_segments[1]] = np.linspace(0.0, smf4.L, segments_len[0])
        current_z = smf4.L
    if segments_len[1]:
        z_vector[cum_segments[1] : cum_segments[2]] = np.linspace(current_z, current_z + amf2.L, segments_len[1])
        current_z += amf2.L
    if segments_len[2]:
        z_vector[cum_segments[2] : cum_segments[3]] = np.linspace(current_z, current_z + smf5.L, segments_len[2])
        current_z += smf5.L

    delta = 1e-9
    if segments_len[3]:
        z_vector[cum_segments[3] : cum_segments[4]] = current_z + np.arange(segments_len[3]) * delta
        current_z += segments_len[3] * delta
    if segments_len[4]:
        z_vector[cum_segments[4] : cum_segments[5]] = np.linspace(current_z, current_z + smf3.L, segments_len[4])
        current_z += smf3.L
    if segments_len[5]:
        z_vector[cum_segments[5] : cum_segments[6]] = current_z + np.arange(segments_len[5]) * delta
        current_z += segments_len[5] * delta
    if segments_len[6]:
        z_vector[cum_segments[6] : cum_segments[7]] = current_z + np.arange(segments_len[6]) * delta
        current_z += segments_len[6] * delta

    for idx in range(1, z_vector.size):
        if z_vector[idx] <= z_vector[idx - 1]:
            z_vector[idx] = z_vector[idx - 1] + 1e-12

    key_positions = {label: min(max(idx, 0), z_vector.size - 1) for label, idx in zip(key_labels, key_indices)}
    key_distances = {label: z_vector[pos] for label, pos in key_positions.items()}

    combined_trace = PropagationTrace(
        z=z_vector,
        ufft=combined_ufft,
        u=combined_u,
    )

    diagnostics = {
        "input_peak_power": peak_power,
        "input_pulse_energy": pulse_energy,
        "simulation_time_s": elapsed,
    }

    result = SimulationResult(
        time=t,
        frequency=f,
        wavelength=wavelength,
        input_field=u0,
        output_field=last_uout,
        spectrum_history=spec_history_arr,
        field_history=field_history_arr,
        segments=last_traces,
        combined_trace=combined_trace,
        key_positions=key_positions,
        key_distances=key_distances,
        diagnostics=diagnostics,
    )

    if plot:
        output_dir = Path(output_dir or "figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        _render_plots(result, output_dir)

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _render_plots(result: SimulationResult, output_dir: Path) -> None:
    """Generate the diagnostic plots for a simulation run."""

    t = result.time
    wavelength = result.wavelength

    fig, ax = plt.subplots()
    ax.plot(t, np.abs(result.input_field) ** 2, "b.-", label="输入脉冲")
    ax.plot(t, np.abs(result.output_field) ** 2, "r.-", label="输出脉冲")
    ax.set(xlabel="时间 (ps)", ylabel="|u(z,t)|^2 (W)", title="初始与输出脉冲形状")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_domain.png", dpi=200)
    plt.close(fig)

    spec = np.fft.fftshift(np.abs(np.fft.fft(result.output_field)) ** 2)
    spec_norm = spec / (wavelength**2)
    spec_norm /= np.max(spec_norm)

    fig, ax = plt.subplots()
    ax.plot(wavelength, spec_norm, "r.-")
    ax.set(xlabel="波长 (nm)", ylabel="归一化光谱 (a.u.)", title="输出光谱")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "output_spectrum.png", dpi=200)
    plt.close(fig)

    trips = result.field_history.shape[0]
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(t, np.arange(1, trips + 1), np.abs(result.field_history) ** 2, shading="auto")
    ax.set(xlabel="时间 (ps)", ylabel="循环次数", title="输出光场演化")
    fig.colorbar(mesh, ax=ax, label="|u(z,t)|^2 (W)")
    fig.tight_layout()
    fig.savefig(output_dir / "pulse_evolution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        wavelength,
        np.arange(1, trips + 1),
        result.spectrum_history,
        shading="auto",
    )
    ax.set(xlabel="波长 (nm)", ylabel="循环次数", title="输出光谱演化")
    fig.colorbar(mesh, ax=ax, label="归一化光谱 (a.u.)")
    fig.tight_layout()
    fig.savefig(output_dir / "spectrum_evolution.png", dpi=200)
    plt.close(fig)

    Eout = np.abs(result.output_field) ** 2
    phase_out = np.unwrap(np.angle(result.output_field))
    chirp = -np.diff(phase_out) / (2.0 * math.pi * (t[1] - t[0]))
    width, left_idx, right_idx = fwhm(Eout)
    t_chirp = t[:-1]

    fig, ax1 = plt.subplots()
    ax1.plot(t, Eout, label="强度")
    ax1.set(xlabel="时间 (ps)", ylabel="|u(z,t)|^2 (W)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t_chirp, chirp, "r--", label="啁啾")
    ax2.set_ylabel("啁啾 (THz)")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    ax1.set_title(f"输出脉冲强度与啁啾 (脉宽 {width * (t[1] - t[0]):.2f} ps)")
    fig.tight_layout()
    fig.savefig(output_dir / "chirp.png", dpi=200)
    plt.close(fig)

    combined = result.combined_trace
    spec = np.abs(combined.ufft) ** 2 / (wavelength[np.newaxis, :] ** 2)
    spec /= np.max(spec)

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)

    ax_main = fig.add_subplot(gs[:, 0])
    mesh = ax_main.pcolormesh(wavelength, combined.z, spec, shading="auto")
    ax_main.set(xlabel="波长 (nm)", ylabel="传播距离 (km)", title="腔内光谱演化")
    fig.colorbar(mesh, ax=ax_main, label="归一化强度")

    ax_contour = fig.add_subplot(gs[0, 1])
    contour = ax_contour.contourf(combined.z, wavelength, spec.T, levels=100)
    ax_contour.set(xlabel="传播距离 (km)", ylabel="波长 (nm)", title="光谱强度等高线图")
    fig.colorbar(contour, ax=ax_contour)

    ax_lines = fig.add_subplot(gs[1, 1])
    colors = plt.cm.jet(np.linspace(0, 1, len(result.key_positions)))
    for color, (label, idx) in zip(colors, result.key_positions.items()):
        spectrum = spec[idx]
        ax_lines.plot(wavelength, spectrum, color=color, label=f"{label} ({result.key_distances[label]:.4f} km)")
    ax_lines.set(xlabel="波长 (nm)", ylabel="归一化强度", title="关键位置光谱对比")
    ax_lines.legend(loc="best")
    ax_lines.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "combined_spectrum.png", dpi=200)
    plt.close(fig)

    intensity = np.abs(combined.u) ** 2
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)

    ax_main = fig.add_subplot(gs[:, 0])
    mesh = ax_main.pcolormesh(t, combined.z, intensity, shading="auto")
    ax_main.set(xlabel="时间 (ps)", ylabel="传播距离 (km)", title="腔内脉冲强度演化")
    fig.colorbar(mesh, ax=ax_main, label="强度 (W)")

    ax_contour = fig.add_subplot(gs[0, 1])
    contour = ax_contour.contourf(combined.z, t, intensity.T, levels=100)
    ax_contour.set(xlabel="传播距离 (km)", ylabel="时间 (ps)", title="脉冲强度等高线图")
    fig.colorbar(contour, ax=ax_contour)

    ax_lines = fig.add_subplot(gs[1, 1])
    colors = plt.cm.jet(np.linspace(0, 1, len(result.key_positions)))
    for color, (label, idx) in zip(colors, result.key_positions.items()):
        field = combined.u[idx]
        intensity_profile = field**2
        ax_lines.plot(t, intensity_profile, color=color, label=label)
    ax_lines.set(xlabel="时间 (ps)", ylabel="强度 (W)", title="关键位置脉冲对比")
    ax_lines.legend(loc="best")
    ax_lines.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "combined_time_domain.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NALM cavity simulation")
    parser.add_argument("--trips", type=int, default=25, help="Number of cavity round trips")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for plots")
    parser.add_argument("--no-plots", action="store_true", help="Disable figure generation")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    simulate_nalm(trips=args.trips, output_dir=args.output_dir, plot=not args.no_plots, quiet=args.quiet)


if __name__ == "__main__":
    main()
