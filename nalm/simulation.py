"""High-level orchestration for the NALM cavity simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .physics import (
    coupler,
    filter_gauss,
    fwhm,
)
from .propagation import IP_CQEM_FD, PropagationRecord

__all__ = [
    "FiberSegment",
    "GaussianFilter",
    "SimulationConfig",
    "SimulationResults",
    "KeyProfile",
    "simulate_nalm",
]


@dataclass
class FiberSegment:
    length: float
    alpha: float
    betaw: Iterable[float]
    gamma: float
    raman: bool = True
    ssp: bool = True
    gssdB: float | None = None
    PsatdBm: float | None = None
    fc: float | None = None
    fbw: float | None = None
    name: str = "segment"

    def to_namespace(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            L=self.length,
            alpha=self.alpha,
            betaw=np.asarray(self.betaw, dtype=np.complex128),
            gamma=self.gamma,
            raman=int(self.raman),
            ssp=int(self.ssp),
            gssdB=self.gssdB,
            PsatdBm=self.PsatdBm,
            fc=self.fc,
            fbw=self.fbw,
        )


@dataclass
class GaussianFilter:
    fc: float
    f3dB: float
    order: int
    name: str = "filter"


@dataclass
class SimulationConfig:
    nt: int = 2**11
    time_window: float = 70.0
    tfwhm: float = 50.0
    lamda_pulse: float = 1550.0
    N2: float = 1.0
    dz: float = 1e-5
    tol: float = 2e-4
    rho: float = 0.45
    rho_out: float = 0.35
    trips: int = 25
    noise_power_dbw: float = 25.0
    noise_seed: int = 0
    progress: bool = False


@dataclass
class KeyProfile:
    intensity: np.ndarray
    chirp: np.ndarray
    chirp_time: np.ndarray


@dataclass
class SimulationResults:
    config: SimulationConfig
    time: np.ndarray
    frequency: np.ndarray
    wavelength: np.ndarray
    dt: float
    df: float
    fo: float
    input_field: np.ndarray
    output_field: np.ndarray
    filtered_field: np.ndarray
    transmitted_field: np.ndarray
    spectral_evolution: np.ndarray
    temporal_evolution: np.ndarray
    combined_spectral_map: np.ndarray
    combined_temporal_map: np.ndarray
    z_axis: np.ndarray
    key_labels: List[str]
    key_positions: np.ndarray
    key_distances: np.ndarray
    key_profiles: Dict[str, KeyProfile]
    phase_out: np.ndarray
    chirp: np.ndarray
    output_intensity: np.ndarray
    width_samples: int
    width_time: float
    left_index: int
    right_index: int
    runtime: float
    segments: List[Dict[str, float]]
    final_spectrum: np.ndarray
    final_spectrum_norm: np.ndarray


@dataclass
class _SegmentInfo:
    name: str
    length: float
    points: int


def _create_default_segments(config: SimulationConfig) -> Tuple[FiberSegment, ...]:
    c = 299792.458
    Aeff = 42.6488
    n2 = 30.0
    gamma = 2 * np.pi * n2 / config.lamda_pulse / Aeff * 1e4
    betaw = np.array([0.0, 0.0, 17.8777, 39.4876e-6])

    smf1 = FiberSegment(
        length=0.0015,
        alpha=0.0,
        betaw=betaw,
        gamma=gamma,
        raman=False,
        ssp=False,
        name="SMF1",
    )

    amf1 = replace(
        smf1,
        length=0.00015,
        gssdB=20.0,
        PsatdBm=41.0,
        fc=c / config.lamda_pulse,
        fbw=c / (config.lamda_pulse**2) * 100.0,
        name="AMF1",
    )

    smf2 = replace(
        smf1,
        length=0.002 - smf1.length - amf1.length,
        name="SMF2",
    )

    smf3 = replace(smf1, length=0.0005, name="SMF3")
    smf4 = replace(smf1, name="SMF4")
    smf5 = replace(smf1, length=0.0115, name="SMF5")
    amf2 = replace(amf1, length=0.004, gssdB=35.0, name="AMF2")

    return smf1, amf1, smf2, smf3, smf4, smf5, amf2


def _create_filter(config: SimulationConfig) -> GaussianFilter:
    c = 299792.458
    lamda_c = config.lamda_pulse
    landa_bw = 30.0
    fc = c / lamda_c
    f3dB = c / (lamda_c**2) * landa_bw
    return GaussianFilter(fc=fc, f3dB=f3dB, order=1, name="Filter")


def _prepare_grid(config: SimulationConfig, fo: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    nt = config.nt
    dt = config.time_window / nt
    t = np.linspace(-config.time_window / 2.0, config.time_window / 2.0 - dt, nt)
    df = 1.0 / (nt * dt)
    f = np.linspace(-nt / 2, nt / 2 - 1, nt) * df
    c = 299792.458
    wavelength = c / (f + fo)
    return t, f, wavelength, dt, df


def _white_gaussian_noise(nt: int, power_dbw: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    variance = 10 ** (power_dbw / 10.0)
    return rng.normal(scale=np.sqrt(variance), size=nt)


def _build_distance_profile(segments: List[_SegmentInfo]) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    delta = 1e-9
    z_values = []
    labels = []
    positions = []
    current = 0.0
    cumulative = 0

    for seg in segments:
        if seg.points == 0:
            continue
        start = cumulative
        end = cumulative + seg.points
        if seg.length > 0:
            z_seg = np.linspace(current, current + seg.length, seg.points)
            current += seg.length
            z_values.append(z_seg)
            labels.extend([f"{seg.name}入口", f"{seg.name}中间", f"{seg.name}出口"])
            mid = start + max(seg.points // 2, 0)
            positions.extend([start, mid, end - 1])
        else:
            z_seg = current + np.arange(seg.points) * delta
            current += seg.points * delta
            z_values.append(z_seg)
            labels.append(seg.name)
            positions.append(start + seg.points // 2)
        cumulative = end

    if not z_values:
        return np.array([]), [], np.array([], dtype=int), np.array([])

    z_axis = np.concatenate(z_values)
    positions = np.asarray([min(max(pos, 0), z_axis.size - 1) for pos in positions], dtype=int)
    distances = z_axis[positions]
    return z_axis, labels, positions, distances


def simulate_nalm(config: SimulationConfig = SimulationConfig()) -> SimulationResults:
    start_time = perf_counter()

    if config.trips < 1:
        raise ValueError("config.trips must be a positive integer")

    smf1, amf1, smf2, smf3, smf4, smf5, amf2 = _create_default_segments(config)
    gaussian_filter = _create_filter(config)

    c = 299792.458
    fo = c / config.lamda_pulse

    t, f, wavelength, dt, df = _prepare_grid(config, fo)

    P_peak = 2 * config.N2 * abs(smf1.betaw[2]) / (smf1.gamma * config.tfwhm**2)
    u0 = np.sqrt(P_peak) / np.cosh(t / config.tfwhm)
    noise = _white_gaussian_noise(config.nt, config.noise_power_dbw, config.noise_seed)
    u0 = noise * u0

    spectral_history = []
    field_history = []

    u = u0.copy()
    final_smf4: PropagationRecord | None = None
    final_amf2: PropagationRecord | None = None
    final_smf5: PropagationRecord | None = None
    final_smf3: PropagationRecord | None = None

    for trip in range(config.trips):
        if config.progress:
            print(f"Trip {trip + 1}/{config.trips}")

        u, _, final_smf4 = IP_CQEM_FD(u, dt, config.dz, smf4.to_namespace(), fo, config.tol, True, True)
        u, _, final_amf2 = IP_CQEM_FD(u, dt, config.dz, amf2.to_namespace(), fo, config.tol, True, True)
        u, _, final_smf5 = IP_CQEM_FD(u, dt, config.dz, smf5.to_namespace(), fo, config.tol, True, True)

        uf, ub = coupler(u, np.zeros_like(u), config.rho)

        ufo, _, _ = IP_CQEM_FD(uf, dt, config.dz, smf1.to_namespace(), fo, config.tol, False, True)
        ufo, _, _ = IP_CQEM_FD(ufo, dt, config.dz, amf1.to_namespace(), fo, config.tol, False, True)
        ufo, _, _ = IP_CQEM_FD(ufo, dt, config.dz, smf2.to_namespace(), fo, config.tol, False, True)

        ubo, _, _ = IP_CQEM_FD(ub, dt, config.dz, smf2.to_namespace(), fo, config.tol, False, True)
        ubo, _, _ = IP_CQEM_FD(ubo, dt, config.dz, amf1.to_namespace(), fo, config.tol, False, True)
        ubo, _, _ = IP_CQEM_FD(ubo, dt, config.dz, smf1.to_namespace(), fo, config.tol, False, True)

        ur, ut = coupler(ubo, ufo, config.rho)
        u = ut

        u, _, final_smf3 = IP_CQEM_FD(u, dt, config.dz, smf3.to_namespace(), fo, config.tol, True, True)
        u, uout = coupler(u, np.zeros_like(u), config.rho_out)

        u_filtered = filter_gauss(u, gaussian_filter.f3dB, gaussian_filter.fc, gaussian_filter.order, fo, df)
        u = u_filtered

        spectrum = np.fft.fftshift(np.abs(np.fft.fft(uout)) ** 2)
        spectrum_norm = spectrum / (wavelength**2)
        spectrum_norm /= np.max(spectrum_norm)

        spectral_history.append(spectrum_norm)
        field_history.append(uout.copy())

    runtime = perf_counter() - start_time

    spectral_evolution = np.asarray(spectral_history)
    temporal_evolution = np.asarray(field_history)

    if any(record is None for record in (final_smf4, final_amf2, final_smf5, final_smf3)):
        raise RuntimeError("Propagation records are missing; ensure at least one trip is simulated.")

    ut_fft = np.tile(np.abs(np.fft.fftshift(np.fft.fft(ut))), (20, 1))
    u_filtered_fft = np.tile(np.abs(np.fft.fftshift(np.fft.fft(u_filtered))), (20, 1))
    uout_fft = np.tile(np.abs(np.fft.fftshift(np.fft.fft(uout))), (20, 1))

    combined_spectral = np.vstack(
        [
            final_smf4.spectra,
            final_amf2.spectra,
            final_smf5.spectra,
            ut_fft,
            final_smf3.spectra,
            uout_fft,
            u_filtered_fft,
        ]
    )

    combined_fields = np.vstack(
        [
            final_smf4.fields,
            final_amf2.fields,
            final_smf5.fields,
            np.tile(ut, (20, 1)),
            final_smf3.fields,
            np.tile(uout, (20, 1)),
            np.tile(u_filtered, (20, 1)),
        ]
    )

    segments = [
        _SegmentInfo(name="SMF4", length=smf4.length, points=final_smf4.spectra.shape[0]),
        _SegmentInfo(name="AMF2", length=amf2.length, points=final_amf2.spectra.shape[0]),
        _SegmentInfo(name="SMF5", length=smf5.length, points=final_smf5.spectra.shape[0]),
        _SegmentInfo(name="NALM透射光", length=0.0, points=ut_fft.shape[0]),
        _SegmentInfo(name="SMF3", length=smf3.length, points=final_smf3.spectra.shape[0]),
        _SegmentInfo(name="输出耦合点", length=0.0, points=uout_fft.shape[0]),
        _SegmentInfo(name="滤波器输出", length=0.0, points=u_filtered_fft.shape[0]),
    ]

    z_axis, key_labels, key_positions, key_distances = _build_distance_profile(segments)

    spectra_power = np.abs(combined_spectral) ** 2
    combined_spectral_norm = spectra_power / (wavelength**2)
    combined_spectral_norm /= np.max(combined_spectral_norm)

    intensity_map = np.abs(combined_fields) ** 2

    key_profiles: Dict[str, KeyProfile] = {}
    for label, idx in zip(key_labels, key_positions):
        field = combined_fields[idx]
        intensity = np.abs(field) ** 2
        phase = np.unwrap(np.angle(field))
        chirp = -np.diff(phase) / (2.0 * np.pi * dt)
        key_profiles[label] = KeyProfile(intensity=intensity, chirp=chirp, chirp_time=t[:-1])

    phase_out = np.unwrap(np.angle(uout))
    chirp_out = -np.diff(phase_out) / (2.0 * np.pi * dt)
    output_intensity = np.abs(uout) ** 2
    width_samples, left_idx, right_idx = fwhm(output_intensity)

    final_spectrum = spectrum
    final_spectrum_norm = spectrum_norm

    results = SimulationResults(
        config=config,
        time=t,
        frequency=f,
        wavelength=wavelength,
        dt=dt,
        df=df,
        fo=fo,
        input_field=u0,
        output_field=uout,
        filtered_field=u_filtered,
        transmitted_field=ut,
        spectral_evolution=spectral_evolution,
        temporal_evolution=temporal_evolution,
        combined_spectral_map=combined_spectral_norm,
        combined_temporal_map=intensity_map,
        z_axis=z_axis,
        key_labels=key_labels,
        key_positions=key_positions,
        key_distances=key_distances,
        key_profiles=key_profiles,
        phase_out=phase_out,
        chirp=chirp_out,
        output_intensity=output_intensity,
        width_samples=width_samples,
        width_time=width_samples * dt,
        left_index=left_idx,
        right_index=right_idx,
        runtime=runtime,
        segments=[seg.__dict__ for seg in segments],
        final_spectrum=final_spectrum,
        final_spectrum_norm=final_spectrum_norm,
    )

    return results
