"""Simulation framework for a 1030 nm NALM 9-shaped cavity fiber laser.

This module implements:
    * An RK4 in the interaction picture (RK4IP) solver for the nonlinear Schr\"odinger equation.
    * Four-order Runge-Kutta solver for the coupled rate equations in the Yb-doped gain fiber.
    * Component models for passive fiber sections, Yb-doped gain fiber, nonlinear amplifying loop mirror (NALM),
      and chirped fiber Bragg grating (CFBG).
    * Utilities for building a 1030 nm NALM cavity, evolving the pulse, estimating lock modes, and evaluating
      phase/intensity noise metrics.

The entry point is :func:`build_default_nalm_laser` which returns a configured :class:`NALMLaser` instance.

The code is written for research prototyping and is intentionally modular so that individual components
can be swapped or augmented as better physical data become available.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from simple_numeric import (
    abs_list,
    abs_squared,
    add,
    angle,
    arange,
    array_like,
    clip,
    complex_exp,
    fft,
    ifft,
    fftshift,
    fftfreq,
    linspace,
    mean,
    mul,
    rfft,
    rfftfreq,
    scale,
    trapz,
    unwrap,
    weighted_average,
    zeros,
)


# ---------------------------------------------------------------------------
# Physical constants and utility helpers
# ---------------------------------------------------------------------------
C = 299_792_458.0  # speed of light in vacuum [m/s]
HBAR = 1.054_571_817e-34


ArrayLike = List[float]
FieldArray = List[complex]


def angular_frequency_center(wavelength_nm: float) -> float:
    """Return the angular frequency for the provided vacuum wavelength."""
    wavelength_m = wavelength_nm * 1e-9
    return 2.0 * math.pi * C / wavelength_m


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert a full width at half maximum to the standard deviation of a Gaussian."""
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


# ---------------------------------------------------------------------------
# Numerical solvers
# ---------------------------------------------------------------------------

def rk4_step(func: Callable[[float, float], float], t: float, y: float, dt: float) -> float:
    """Generic explicit fourth-order Runge-Kutta step for scalar states."""
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4ip_nlse(
    field: FieldArray,
    dz: float,
    linear_operator: FieldArray,
    nonlinear_gamma: float,
    dispersion_phase: FieldArray,
) -> FieldArray:
    """Advance the field envelope over *dz* using the RK4 interaction picture algorithm.

    Parameters
    ----------
    field:
        Current complex field envelope in the time domain.
    dz:
        Propagation step length [m].
    linear_operator:
        Frequency-domain representation of the linear effects (dispersion + loss).
    nonlinear_gamma:
        Nonlinear coefficient [1/(W·m)].
    dispersion_phase:
        Pre-computed phase term exp(linear_operator * dz) used for the interaction picture rotation.
    """
    def nl_term(a: FieldArray) -> FieldArray:
        return [1j * nonlinear_gamma * abs(val) ** 2 * val for val in a]

    a_ip = ifft(mul(fft(field), dispersion_phase))

    k1 = scale(nl_term(a_ip), dz)
    k2_input = ifft(mul(fft(add(a_ip, scale(k1, 0.5))), dispersion_phase))
    k2 = scale(nl_term(k2_input), dz)
    k3_input = ifft(mul(fft(add(a_ip, scale(k2, 0.5))), dispersion_phase))
    k3 = scale(nl_term(k3_input), dz)
    k4_input = ifft(mul(fft(add(a_ip, k3)), dispersion_phase))
    k4 = scale(nl_term(k4_input), dz)

    field_ip = [a_ip[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0 for i in range(len(field))]
    return ifft(mul(fft(field_ip), dispersion_phase))


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------


def loss_to_alpha(loss_dB: float, length: float) -> float:
    """Convert a lumped loss in dB to the distributed attenuation coefficient alpha [1/m]."""
    return (loss_dB / 4.343) / max(length, 1e-9)


@dataclass
class FiberSegment:
    """Passive single-mode fiber segment."""

    length: float  # [m]
    beta2: float  # [ps^2/km]
    gamma: float  # [1/(W·km)]
    loss_dB: float = 0.0
    steps: Optional[int] = None

    def _num_steps(self, dz: float) -> Tuple[int, float]:
        if self.steps is not None and self.steps > 0:
            num_steps = self.steps
        else:
            num_steps = int(max(1, round(self.length / dz)))
        dz_eff = self.length / num_steps
        return num_steps, dz_eff

    def propagate(self, field: FieldArray, dz: float, dt: float) -> FieldArray:
        num_steps, dz_eff = self._num_steps(dz)

        freqs = fftfreq(len(field), dt)
        omega = [2.0 * math.pi * f for f in freqs]
        beta2_si = self.beta2 * 1e-24 / 1e3  # convert ps^2/km -> s^2/m
        alpha = loss_to_alpha(self.loss_dB, self.length)
        linear_operator = [(-0.5 * alpha) + 0.5j * beta2_si * (w**2) for w in omega]
        dispersion_phase = [cmath.exp(val * dz_eff) for val in linear_operator]
        gamma_si = self.gamma / 1e3  # 1/(W·km) -> 1/(W·m)

        out = list(field)
        for _ in range(num_steps):
            out = rk4ip_nlse(out, dz_eff, linear_operator, gamma_si, dispersion_phase)
        return out


@dataclass
class GainFiber(FiberSegment):
    """Ytterbium-doped gain fiber using a two-level rate equation model."""

    sigma_emission: float = 3.58e-25  # [m^2]
    sigma_absorption: float = 2.4e-24  # [m^2]
    core_area: float = 30e-12  # [m^2]
    upper_lifetime: float = 0.9e-3  # [s]
    pump_wavelength_nm: float = 975.0
    signal_wavelength_nm: float = 1030.0
    pump_power_W: float = 0.5

    def propagate(self, field: FieldArray, dz: float, dt: float) -> FieldArray:
        pump_freq = angular_frequency_center(self.pump_wavelength_nm)
        signal_freq = angular_frequency_center(self.signal_wavelength_nm)

        def rate_equations(_: float, n2: float) -> float:
            pump_intensity = self.pump_power_W / self.core_area
            signal_intensity = mean(abs_squared(field)) / self.core_area

            wp = pump_intensity * self.sigma_absorption / (HBAR * pump_freq)
            ws = signal_intensity * (self.sigma_emission + self.sigma_absorption) / (HBAR * signal_freq)

            dn2_dt = wp * (1.0 - n2) - ws * n2 - n2 / self.upper_lifetime
            return dn2_dt

        n2 = 0.0
        dz_steps, dz_eff = self._num_steps(dz)

        freqs = fftfreq(len(field), dt)
        omega = [2.0 * math.pi * f for f in freqs]
        beta2_si = self.beta2 * 1e-24 / 1e3
        alpha = loss_to_alpha(self.loss_dB, self.length)
        linear_operator = [(-0.5 * alpha) + 0.5j * beta2_si * (w**2) for w in omega]
        dispersion_phase = [cmath.exp(val * dz_eff) for val in linear_operator]
        gamma_si = self.gamma / 1e3

        out = list(field)
        for step in range(dz_steps):
            z = step * dz_eff
            n2 = rk4_step(rate_equations, z, n2, dz_eff)
            small_signal_gain = (self.sigma_emission * n2 - self.sigma_absorption * (1 - n2))
            gain_per_m = small_signal_gain
            gain_factor = math.exp(gain_per_m * dz_eff)
            out = [val * gain_factor for val in out]
            out = rk4ip_nlse(out, dz_eff, linear_operator, gamma_si, dispersion_phase)
        return out


@dataclass
class CFBG:
    """Chirped fiber Bragg grating modeled in the spectral domain."""

    dispersion_ps_per_nm: float
    fwhm_nm: float
    center_wavelength_nm: float = 1030.0

    def apply(self, field: FieldArray, wavelength_nm: List[float]) -> FieldArray:
        sigma_nm = fwhm_to_sigma(self.fwhm_nm)
        result = []
        for amp, lam in zip(field, wavelength_nm):
            delta = lam - self.center_wavelength_nm
            gaussian = math.exp(-0.5 * (delta / sigma_nm) ** 2)
            dispersion_phase = cmath.exp(1j * 0.5 * self.dispersion_ps_per_nm * (delta**2))
            result.append(amp * gaussian * dispersion_phase)
        return result


@dataclass
class LumpedLoss:
    """Simple lumped loss element applied uniformly in the time domain."""

    loss_dB: float

    def propagate(self, field: FieldArray, dz: float, dt: float) -> FieldArray:
        factor = 10 ** (-0.5 * self.loss_dB / 10.0)
        return [val * factor for val in field]


@dataclass
class NALMComponent:
    """Nonlinear amplifying loop mirror component."""

    coupler_ratio: float
    loop_segments: List[FiberSegment]
    reference_segments: List[FiberSegment]
    bias_phase: float = 0.0
    nonlinear_phase_coeff: float = 2.0
    loop_loss_dB: float = 0.0

    def propagate(self, field: FieldArray, dz: float, dt: float) -> FieldArray:
        k = clip(self.coupler_ratio, 1e-6, 1 - 1e-6)
        sqrt_k = math.sqrt(k)
        sqrt_1mk = math.sqrt(1.0 - k)

        amp_loop = [1j * sqrt_k * val for val in field]
        amp_ref = [sqrt_1mk * val for val in field]

        loop_field = list(amp_loop)
        for segment in self.loop_segments:
            step = segment.length / 20.0 if segment.length else dz
            loop_field = segment.propagate(loop_field, dz=max(step, 1e-3), dt=dt)

        ref_field = list(amp_ref)
        for segment in self.reference_segments:
            step = segment.length / 20.0 if segment.length else dz
            ref_field = segment.propagate(ref_field, dz=max(step, 1e-3), dt=dt)

        energy_loop = trapz(abs_squared(loop_field))
        energy_ref = trapz(abs_squared(ref_field))
        delta_phi = self.bias_phase + self.nonlinear_phase_coeff * (energy_loop - energy_ref)
        phase_factor = cmath.exp(1j * delta_phi)
        loop_field = [val * phase_factor for val in loop_field]

        if self.loop_loss_dB:
            loss_factor = 10 ** (-0.5 * self.loop_loss_dB / 10.0)
            loop_field = [val * loss_factor for val in loop_field]

        return [
            sqrt_1mk * ref_field[i] + 1j * sqrt_k * loop_field[i]
            for i in range(len(field))
        ]


ElementType = Union[FiberSegment, GainFiber, LumpedLoss, NALMComponent]


@dataclass
class NALMLaser:
    """Representation of a NALM cavity."""

    dt: float
    time_window: float
    segments: List[ElementType]
    cfbgs: List[Tuple[CFBG, int]] = field(default_factory=list)  # (CFBG, index after which applied)
    output_coupling: float = 0.1

    def round_trip(self, field: FieldArray) -> Tuple[FieldArray, Optional[FieldArray]]:
        out = list(field)
        extracted = None
        for idx, segment in enumerate(self.segments):
            if isinstance(segment, NALMComponent):
                out = segment.propagate(out, dz=self._segment_step(segment), dt=self.dt)
            else:
                out = segment.propagate(out, dz=self._segment_step(segment), dt=self.dt)
            for cfbg, insert_idx in self.cfbgs:
                if insert_idx == idx:
                    wavelength = self._wavelength_grid(len(out))
                    out_freq = fft(out)
                    filtered = cfbg.apply(out_freq, wavelength)
                    out = ifft(filtered)

        if self.output_coupling:
            transmission = math.sqrt(max(0.0, 1.0 - self.output_coupling))
            coupling = math.sqrt(max(0.0, self.output_coupling))
            extracted = [coupling * val for val in out]
            out = [transmission * val for val in out]
        return out, extracted

    def _segment_step(self, segment: ElementType) -> float:
        if isinstance(segment, (FiberSegment, GainFiber)):
            if segment.steps:
                return segment.length / segment.steps
            return segment.length / 20.0
        return 1.0

    def _wavelength_grid(self, n: int) -> List[float]:
        freq = fftfreq(n, self.dt)
        omega = [2.0 * math.pi * f for f in freq]
        lambda_center = 1030e-9
        freq_center = C / lambda_center
        freq_total = [freq_center + w / (2.0 * math.pi) for w in omega]
        wavelength = [C / f for f in freq_total]
        return [lam * 1e9 for lam in wavelength]

    def evolve(
        self,
        seed: FieldArray,
        num_round_trips: int,
        diagnostics: bool = False,
        return_history: bool = False,
    ) -> Tuple[FieldArray, Dict[str, List]]:
        field = list(seed)
        diag: Dict[str, List] = {"energy": [], "pulse_width_ps": []}
        if return_history:
            diag["field_history"] = []
            diag["spectrum_history"] = []
            diag["extracted_history"] = []
        for _ in range(num_round_trips):
            field, extracted = self.round_trip(field)
            if diagnostics:
                intensities = abs_squared(field)
                energy = trapz(intensities)
                time = [idx * self.dt for idx in arange(len(field))]
                mean_t = weighted_average(time, intensities) if intensities else 0.0
                variance = weighted_average([(t - mean_t) ** 2 for t in time], intensities) if intensities else 0.0
                rms = math.sqrt(max(variance, 0.0))
                diag["energy"].append(energy)
                diag["pulse_width_ps"].append(rms * 1e12)
            if return_history:
                diag["field_history"].append(list(field))
                diag["spectrum_history"].append(fftshift(fft(field)))
                diag["extracted_history"].append(None if extracted is None else list(extracted))
        return field, diag

    def phase_noise(self, field: FieldArray, sampling_rate: float) -> Tuple[List[float], List[float]]:
        phase = unwrap(angle(field))
        phase_mean = mean(phase)
        phase_centered = [p - phase_mean for p in phase]
        spectrum = rfft([complex(p, 0.0) for p in phase_centered])
        freqs = rfftfreq(len(phase_centered), 1.0 / sampling_rate)
        psd = [abs(val) ** 2 / len(phase_centered) for val in spectrum]
        return freqs, psd

    def intensity_noise(self, field: FieldArray, sampling_rate: float) -> Tuple[List[float], List[float]]:
        intensity = abs_squared(field)
        mean_intensity = mean(intensity)
        intensity_centered = [val - mean_intensity for val in intensity]
        spectrum = rfft([complex(v, 0.0) for v in intensity_centered])
        freqs = rfftfreq(len(intensity_centered), 1.0 / sampling_rate)
        psd = [abs(val) ** 2 / len(intensity_centered) for val in spectrum]
        return freqs, psd


# ---------------------------------------------------------------------------
# Configuration utilities
# ---------------------------------------------------------------------------


def build_default_nalm_laser(
    time_window_ps: float = 50.0,
    num_samples: int = 4096,
    pump_power_W: float = 0.5,
    total_dispersion_ps2: float = 0.02,
    passive_loss_dB: float = 1.0,
    coupler_ratio: float = 0.55,
    bias_phase: float = 0.1,
    nonlinear_phase_coeff: float = 0.4,
    output_coupling: float = 0.1,
    cfbg_fwhm_nm: float = 16.0,
    cfbg_dispersion_ps_per_nm: float = 0.2,
) -> Tuple[NALMLaser, FieldArray]:
    """Create a default NALM laser configuration around 1030 nm.

    Parameters
    ----------
    time_window_ps:
        Simulation window in picoseconds.
    num_samples:
        Number of temporal samples for the field envelope.
    pump_power_W:
        Pump power coupled into the gain fiber.
    total_dispersion_ps2:
        Net second-order dispersion target for the cavity.
    passive_loss_dB:
        Lumped loss for passive elements.
    """
    dt = (time_window_ps * 1e-12) / num_samples
    seed_time = linspace(-0.5 * time_window_ps, 0.5 * time_window_ps, num_samples)
    seed_field = [
        math.exp(-(t**2) / (2.0 * (2.5**2))) * cmath.exp(1j * 2 * math.pi * t / 5.0)
        for t in seed_time
    ]

    beta2_target = total_dispersion_ps2 * 1e3  # convert to ps^2/km for convenience

    loop_gain = GainFiber(
        length=0.8,
        beta2=beta2_target,
        gamma=2.5,
        loss_dB=0.3,
        pump_power_W=min(pump_power_W, 1.0),
    )
    loop_passive = FiberSegment(length=3.2, beta2=beta2_target, gamma=1.5, loss_dB=passive_loss_dB * 0.5)
    reference_passive = FiberSegment(length=3.2, beta2=beta2_target, gamma=1.3, loss_dB=passive_loss_dB * 0.5)

    nalm = NALMComponent(
        coupler_ratio=coupler_ratio,
        loop_segments=[loop_gain, loop_passive],
        reference_segments=[reference_passive],
        bias_phase=bias_phase,
        nonlinear_phase_coeff=nonlinear_phase_coeff,
        loop_loss_dB=0.5,
    )

    delivery_fiber = FiberSegment(length=4.0, beta2=beta2_target, gamma=1.3, loss_dB=passive_loss_dB * 0.5)
    stretcher_fiber = FiberSegment(length=2.0, beta2=-beta2_target, gamma=1.1, loss_dB=0.2)

    segments: List[ElementType] = [nalm, delivery_fiber, LumpedLoss(passive_loss_dB * 0.3), stretcher_fiber]

    cfbgs = [(CFBG(dispersion_ps_per_nm=cfbg_dispersion_ps_per_nm, fwhm_nm=cfbg_fwhm_nm), 1)]

    laser = NALMLaser(
        dt=dt,
        time_window=time_window_ps * 1e-12,
        segments=segments,
        cfbgs=cfbgs,
        output_coupling=output_coupling,
    )
    return laser, seed_field


def sweep_parameter(
    laser: NALMLaser,
    seed: FieldArray,
    parameter_updates: Iterable[Dict[str, float]],
    num_round_trips: int = 200,
) -> List[Dict[str, Dict[str, Union[float, Dict[str, List]]]]]:
    """Sweep laser parameters and collect diagnostic metrics."""
    results = []
    for updates in parameter_updates:
        for attr, value in updates.items():
            if hasattr(laser, attr):
                setattr(laser, attr, value)
            else:
                # Allow addressing nested attributes like segments[0].coupler_ratio
                parts = attr.replace("]", "").split("[")
                target = laser
                success = False
                for idx, part in enumerate(parts):
                    if "." in part:
                        subparts = part.split(".")
                    else:
                        subparts = [part]
                    for sub in subparts:
                        if not sub:
                            continue
                        if sub.isdigit():
                            target = target[int(sub)]
                        elif sub.startswith("segment"):
                            # allow segment0 alias
                            target = laser.segments[int(sub.replace("segment", ""))]
                        elif hasattr(target, sub):
                            if idx == len(parts) - 1 and sub == subparts[-1]:
                                setattr(target, sub, value)
                                success = True
                            else:
                                target = getattr(target, sub)
                        else:
                            target = None
                            break
                    if target is None:
                        break
                if not success:
                    raise AttributeError(f"Unknown parameter path '{attr}'")
        field, diag = laser.evolve(seed, num_round_trips=num_round_trips, diagnostics=True)
        avg_power = trapz(abs_squared(field)) / laser.time_window
        results.append({"parameters": updates, "diagnostics": diag, "average_power": avg_power})
    return results


def estimate_locking_state(energy_trace: ArrayLike, tolerance: float = 1e-3) -> bool:
    """Check if the pulse energy converges within a tolerance across the last few round trips."""
    energy = list(energy_trace)
    if len(energy) < 5:
        return False
    recent = energy[-5:]
    mean_recent = sum(recent) / len(recent)
    if mean_recent == 0:
        return False
    return all(abs(val - mean_recent) < tolerance * mean_recent for val in recent)


__all__ = [
    "CFBG",
    "FiberSegment",
    "GainFiber",
    "LumpedLoss",
    "NALMLaser",
    "NALMComponent",
    "angular_frequency_center",
    "build_default_nalm_laser",
    "estimate_locking_state",
    "sweep_parameter",
]
