"""Simulation toolkit for a 1030 nm NALM nine-shaped cavity fiber laser.

This module implements a time-domain model based on the nonlinear Schrödinger
equation (NLSE) that is solved with a fourth-order Runge–Kutta in the
interaction picture (RK4IP) algorithm. The population dynamics of the
Yb-doped gain fiber is described by a set of rate equations integrated with a
standard fourth-order Runge–Kutta (RK4) method. The code is designed to study
how cavity parameters such as pump power, distributed loss, dispersion,
gain-fiber placement and average output power influence the mode-locked state,
as well as the residual phase and intensity noise.

The implementation focuses on flexibility and transparency. Every cavity
section is represented by a Python class with a ``propagate`` method, so the
sequence of optical components can easily be reconfigured. A typical
configuration for the 1030 nm Yb-doped NALM cavity is provided through the
``build_default_cavity`` helper.

Example
-------
```
from simulations.nalm_simulation import (
    SimulationConfig,
    build_default_cavity,
    run_parameter_sweep,
)

config = SimulationConfig()
config.pump_powers = [0.6, 0.8, 1.0]  # in watts
results = run_parameter_sweep(config, build_default_cavity(config))
```

The returned ``results`` dictionary contains detailed round-trip traces for
power, phase, and diagnostic spectra that can be analysed or plotted with
Matplotlib.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# Fundamental constants
# =============================================================================
HBAR = 1.054_571_817e-34  # Planck constant over 2pi (J*s)
PLANCK = 6.626_070_15e-34  # Planck constant (J*s)
LIGHT_SPEED = 299_792_458.0  # speed of light in vacuum (m/s)


# =============================================================================
# Utility dataclasses
# =============================================================================
@dataclass
class TemporalGrid:
    """Uniformly sampled time/frequency grid used for NLSE integration."""

    n_points: int
    time_window: float  # total simulation window (s)

    def __post_init__(self) -> None:
        if self.n_points & (self.n_points - 1) != 0:
            raise ValueError("n_points must be a power of two for FFT efficiency")
        self.dt = self.time_window / self.n_points
        self.time = np.linspace(-self.time_window / 2, self.time_window / 2, self.n_points, endpoint=False)
        self.freq = np.fft.fftfreq(self.n_points, d=self.dt)
        self.omega = 2 * np.pi * self.freq


@dataclass
class SimulationConfig:
    """Top-level configuration for the cavity simulation."""

    wavelength: float = 1030e-9
    pump_wavelength: float = 975e-9
    cavity_length: float = 6.5  # m, effective total length
    n_roundtrips: int = 800
    steady_state_start: int = 400
    sample_roundtrips: int = 200
    n_time_points: int = 2 ** 12
    time_window: float = 20e-12
    gamma: float = 3.0  # W^-1 km^-1, typical nonlinear coefficient
    beta2: float = -25e-27  # s^2/m, average cavity dispersion per meter
    passive_loss_db: float = 0.1  # dB per round-trip
    pump_powers: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0)
    seed_noise_level: float = 1e-6
    cfbg_dispersion: float = 0.2e-12 / 1e-9  # ps/nm -> s^2/m approx
    cfbg_bandwidth: float = 16e-9
    output_coupling_ratio: float = 0.3

    def create_grid(self) -> TemporalGrid:
        return TemporalGrid(self.n_time_points, self.time_window)

    @property
    def passive_loss_linear(self) -> float:
        return 10 ** (-self.passive_loss_db / 20.0)

    @property
    def angular_frequency(self) -> float:
        return 2 * np.pi * LIGHT_SPEED / self.wavelength


@dataclass
class GainFiberParameters:
    """Physical constants for the Yb-doped gain fiber."""

    length: float = 1.2  # meters
    core_area: float = 50e-12  # m^2 (roughly 8 um mode-field diameter)
    lifetime: float = 0.8e-3  # seconds
    sigma_abs: float = 2.8e-24  # m^2 absorption cross-section @ pump
    sigma_em: float = 3.5e-24  # m^2 emission cross-section @ signal
    n_total: float = 1.2e25  # m^-3 total ion concentration
    pump_absorption_db_per_m: float = 600.0
    overlap_factor: float = 0.86

    def pump_absorption_linear(self) -> float:
        return self.pump_absorption_db_per_m * math.log(10) / 10.0


@dataclass
class GainDynamicsState:
    """Holds the current inversion level for the gain fiber."""

    inversion: float


@dataclass
class RoundTripDiagnostics:
    """Stores diagnostics collected during the simulation."""

    average_power: List[float] = field(default_factory=list)
    peak_power: List[float] = field(default_factory=list)
    pulse_energy: List[float] = field(default_factory=list)
    centroid_phase: List[float] = field(default_factory=list)


# =============================================================================
# Numerical integrators
# =============================================================================
def rk4ip_step(field: np.ndarray, dz: float, linear_operator: np.ndarray, gamma: float) -> np.ndarray:
    """Propagate one spatial step using the RK4IP algorithm.

    Parameters
    ----------
    field:
        Complex envelope of the optical field in the time domain.
    dz:
        Spatial step length (m).
    linear_operator:
        Frequency-domain linear operator (complex array) applied through
        multiplication. Typical form: 0.5*(gain - loss) + 0.5j*beta2*omega^2.
    gamma:
        Nonlinear coefficient (W^-1 m^-1).
    """

    def nonlinear_term(a: np.ndarray) -> np.ndarray:
        return 1j * gamma * np.abs(a) ** 2 * a

    exp_half = np.exp(linear_operator * dz / 2.0)
    exp_full = np.exp(linear_operator * dz)

    a0 = np.fft.ifft(np.fft.fft(field) * exp_half)
    k1 = nonlinear_term(a0)

    a1 = np.fft.ifft(np.fft.fft(field + 0.5 * dz * k1) * exp_half)
    k2 = nonlinear_term(a1)

    a2 = np.fft.ifft(np.fft.fft(field + 0.5 * dz * k2) * exp_half)
    k3 = nonlinear_term(a2)

    a3 = np.fft.ifft(np.fft.fft(field + dz * k3) * exp_half)
    k4 = nonlinear_term(a3)

    nonlinear_update = (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    field_updated = np.fft.ifft(np.fft.fft(field + nonlinear_update) * exp_half)
    return np.fft.ifft(np.fft.fft(field_updated) * exp_half)


def rk4_step(y: float, dt: float, derivative) -> float:
    """Classical fourth-order Runge-Kutta helper for scalar ODEs."""

    k1 = derivative(y)
    k2 = derivative(y + 0.5 * dt * k1)
    k3 = derivative(y + 0.5 * dt * k2)
    k4 = derivative(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# =============================================================================
# Gain dynamics
# =============================================================================
class GainFiberModel:
    """Two-level gain model for an Yb-doped fiber segment."""

    def __init__(self, params: GainFiberParameters, config: SimulationConfig):
        self.params = params
        self.config = config
        inversion_init = 0.1 * params.n_total
        self.state = GainDynamicsState(inversion=inversion_init)

    def pump_rate(self, pump_power: float) -> float:
        intensity = pump_power / (self.params.core_area * self.params.overlap_factor)
        photon_energy = PLANCK * LIGHT_SPEED / self.config.pump_wavelength
        return self.params.sigma_abs * intensity / photon_energy

    def signal_rate(self, signal_power: float) -> float:
        if signal_power <= 0.0:
            return 0.0
        intensity = signal_power / (self.params.core_area * self.params.overlap_factor)
        photon_energy = PLANCK * LIGHT_SPEED / self.config.wavelength
        return self.params.sigma_em * intensity / photon_energy

    def rate_equation(self, inversion: float, pump_power: float, signal_power: float) -> float:
        p = self.params
        N_total = p.n_total
        pump_absorption = self.pump_rate(pump_power)
        stimulated = self.signal_rate(signal_power)
        spontaneous = inversion / p.lifetime
        return pump_absorption * (N_total - inversion) - stimulated * inversion - spontaneous

    def update_inversion(self, pump_power: float, signal_power: float, dt: float) -> None:
        def deriv(inv: float) -> float:
            return self.rate_equation(inv, pump_power, signal_power)

        self.state.inversion = np.clip(rk4_step(self.state.inversion, dt, deriv), 0.0, self.params.n_total)

    def small_signal_gain(self) -> float:
        p = self.params
        inv = self.state.inversion
        absorption = p.sigma_abs * (p.n_total - inv)
        emission = p.sigma_em * inv
        return emission - absorption

    def saturation_energy(self) -> float:
        p = self.params
        photon_energy = PLANCK * LIGHT_SPEED / self.config.wavelength
        return photon_energy * p.core_area / (p.sigma_em + p.sigma_abs)


# =============================================================================
# Cavity sections
# =============================================================================
class CavitySection:
    """Abstract base class for a cavity element."""

    def propagate(self, field: np.ndarray, grid: TemporalGrid, model_state: "SimulationState") -> Tuple[np.ndarray, Dict[str, float]]:
        raise NotImplementedError


class PassiveFiberSection(CavitySection):
    def __init__(self, length: float, beta2: float, gamma: float, loss_db: float = 0.0):
        self.length = length
        self.beta2 = beta2
        self.gamma = gamma / 1000.0  # convert W^-1 km^-1 to W^-1 m^-1
        self.loss_db = loss_db

    def propagate(self, field: np.ndarray, grid: TemporalGrid, model_state: "SimulationState") -> Tuple[np.ndarray, Dict[str, float]]:
        loss_linear = 10 ** (-self.loss_db * self.length / 20.0)
        linear_operator = 0.5 * np.log(loss_linear) + 0.5j * self.beta2 * grid.omega ** 2
        dz = self.length / 10
        updated = field.copy()
        for _ in range(10):
            updated = rk4ip_step(updated, dz, linear_operator, self.gamma)
        return updated, {"section": "passive_fiber", "length": self.length}


class GainFiberSection(CavitySection):
    def __init__(self, gain_model: GainFiberModel, beta2: float, gamma: float, loss_db: float = 0.0):
        self.gain_model = gain_model
        self.beta2 = beta2
        self.gamma = gamma / 1000.0
        self.loss_db = loss_db

    def propagate(self, field: np.ndarray, grid: TemporalGrid, model_state: "SimulationState") -> Tuple[np.ndarray, Dict[str, float]]:
        gain_coeff = self.gain_model.small_signal_gain()
        loss_linear = 10 ** (-self.loss_db * self.gain_model.params.length / 20.0)
        linear_gain = 0.5 * gain_coeff * self.gain_model.params.length + 0.5 * math.log(loss_linear)
        linear_operator = linear_gain + 0.5j * self.beta2 * grid.omega ** 2
        dz = self.gain_model.params.length / 10
        updated = field.copy()
        for _ in range(10):
            updated = rk4ip_step(updated, dz, linear_operator / self.gain_model.params.length, self.gamma)
        return updated, {"section": "gain_fiber", "gain": gain_coeff}


class CFBGSection(CavitySection):
    """Implements a dispersion compensating fiber Bragg grating as spectral filter."""

    def __init__(self, dispersion: float, bandwidth: float, center_wavelength: float):
        self.dispersion = dispersion
        self.bandwidth = bandwidth
        self.center_wavelength = center_wavelength

    def propagate(self, field: np.ndarray, grid: TemporalGrid, model_state: "SimulationState") -> Tuple[np.ndarray, Dict[str, float]]:
        omega0 = 2 * np.pi * LIGHT_SPEED / self.center_wavelength
        phase = -0.5j * self.dispersion * (grid.omega - omega0) ** 2
        sigma = self.bandwidth / (2 * math.sqrt(2 * math.log(2)))
        gaussian = np.exp(-((grid.freq * self.center_wavelength / LIGHT_SPEED) ** 2) / (2 * sigma ** 2))
        spectral = np.fft.fft(field) * np.exp(phase) * gaussian
        return np.fft.ifft(spectral), {"section": "cfbg"}


class OutputCoupler(CavitySection):
    def __init__(self, coupling_ratio: float):
        self.coupling_ratio = coupling_ratio

    def propagate(self, field: np.ndarray, grid: TemporalGrid, model_state: "SimulationState") -> Tuple[np.ndarray, Dict[str, float]]:
        transmitted = math.sqrt(1 - self.coupling_ratio) * field
        leaked = math.sqrt(self.coupling_ratio) * field
        model_state.last_output_field = leaked
        return transmitted, {"section": "output_coupler"}


class NALMSection(CavitySection):
    """Nonlinear amplifying loop mirror modeled via bidirectional propagation."""

    def __init__(self, coupler_ratio: float, loop_sections_cw: Sequence[CavitySection], loop_sections_ccw: Optional[Sequence[CavitySection]] = None):
        self.coupler_ratio = coupler_ratio
        self.loop_sections_cw = list(loop_sections_cw)
        self.loop_sections_ccw = list(loop_sections_ccw) if loop_sections_ccw is not None else list(loop_sections_cw)

    def propagate(self, field: np.ndarray, grid: TemporalGrid, model_state: "SimulationState") -> Tuple[np.ndarray, Dict[str, float]]:
        t_coeff = math.sqrt(self.coupler_ratio)
        r_coeff = 1j * math.sqrt(1 - self.coupler_ratio)
        cw = t_coeff * field
        ccw = r_coeff * field

        for section in self.loop_sections_cw:
            cw, _ = section.propagate(cw, grid, model_state)
        for section in self.loop_sections_ccw:
            ccw, _ = section.propagate(ccw, grid, model_state)

        # Recombine at the coupler
        out_field = t_coeff * cw + r_coeff * ccw
        return out_field, {"section": "nalm"}


# =============================================================================
# Simulation state and helpers
# =============================================================================
@dataclass
class SimulationState:
    gain_model: GainFiberModel
    last_output_field: Optional[np.ndarray] = None
    diagnostics: RoundTripDiagnostics = field(default_factory=RoundTripDiagnostics)

    def record_roundtrip(self, field: np.ndarray, grid: TemporalGrid) -> None:
        intensity = np.abs(field) ** 2
        avg_power = intensity.mean()
        peak_power = intensity.max()
        pulse_energy = intensity.sum() * grid.dt
        centroid_phase = cmath.phase(np.vdot(field, np.ones_like(field)))
        self.diagnostics.average_power.append(float(avg_power))
        self.diagnostics.peak_power.append(float(peak_power))
        self.diagnostics.pulse_energy.append(float(pulse_energy))
        self.diagnostics.centroid_phase.append(float(centroid_phase))


class CavityModel:
    def __init__(self, sections: Sequence[CavitySection]):
        self.sections = list(sections)

    def propagate(self, field: np.ndarray, grid: TemporalGrid, state: SimulationState) -> np.ndarray:
        updated = field
        for section in self.sections:
            updated, _ = section.propagate(updated, grid, state)
        return updated


# =============================================================================
# High-level simulation routines
# =============================================================================
def initialize_field(grid: TemporalGrid, noise_level: float) -> np.ndarray:
    noise = (np.random.randn(grid.n_points) + 1j * np.random.randn(grid.n_points)) * noise_level
    return noise


def compute_noise_spectra(sequence: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(sequence)
    window = np.hanning(n)
    spectrum = np.fft.rfft((sequence - np.mean(sequence)) * window)
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = (np.abs(spectrum) ** 2) * (2.0 / (np.sum(window ** 2) * n))
    return freqs, psd


def analyze_noise(state: SimulationState, config: SimulationConfig) -> Dict[str, np.ndarray]:
    diag = state.diagnostics
    roundtrip_time = config.time_window
    intensity_seq = np.array(diag.average_power[config.steady_state_start :])
    phase_seq = np.unwrap(np.array(diag.centroid_phase[config.steady_state_start :]))
    freqs_i, rin_psd = compute_noise_spectra(intensity_seq, roundtrip_time)
    freqs_p, phase_psd = compute_noise_spectra(phase_seq, roundtrip_time)
    return {
        "rin_frequency": freqs_i,
        "rin_psd": rin_psd,
        "phase_frequency": freqs_p,
        "phase_psd": phase_psd,
    }


def simulate_cavity(config: SimulationConfig, cavity: CavityModel, pump_power: float) -> Dict[str, np.ndarray]:
    grid = config.create_grid()
    gain_model = GainFiberModel(GainFiberParameters(), config)
    state = SimulationState(gain_model=gain_model)
    field = initialize_field(grid, config.seed_noise_level)

    roundtrip_time = config.cavity_length / LIGHT_SPEED * config.time_window / (grid.time_window)
    # Integrate over n_roundtrips
    for idx in range(config.n_roundtrips):
        avg_power = np.mean(np.abs(field) ** 2)
        state.gain_model.update_inversion(pump_power, avg_power, roundtrip_time)
        field = cavity.propagate(field, grid, state)
        field *= config.passive_loss_linear
        state.record_roundtrip(field, grid)

    noise = analyze_noise(state, config)
    output_field = state.last_output_field if state.last_output_field is not None else field
    spectrum = np.fft.fftshift(np.fft.fft(output_field))
    freqs = np.fft.fftshift(grid.freq)

    return {
        "time": grid.time,
        "field": field,
        "output_field": output_field,
        "spectrum": spectrum,
        "frequency": freqs,
        "diagnostics": state.diagnostics,
        "noise": noise,
    }


def run_parameter_sweep(config: SimulationConfig, cavity: CavityModel) -> Dict[float, Dict[str, np.ndarray]]:
    results = {}
    for pump in config.pump_powers:
        results[pump] = simulate_cavity(config, cavity, pump)
    return results


# =============================================================================
# Default cavity constructor
# =============================================================================
def build_default_cavity(config: SimulationConfig) -> CavityModel:
    gain_params = GainFiberParameters(length=1.2)
    gain_model = GainFiberModel(gain_params, config)
    gain_section = GainFiberSection(gain_model, beta2=config.beta2, gamma=config.gamma, loss_db=0.3)

    passive_before = PassiveFiberSection(length=1.5, beta2=config.beta2, gamma=config.gamma)
    passive_after = PassiveFiberSection(length=2.5, beta2=config.beta2, gamma=config.gamma)

    nalm_loop = [
        PassiveFiberSection(length=0.4, beta2=config.beta2, gamma=config.gamma, loss_db=0.1),
        PassiveFiberSection(length=0.4, beta2=config.beta2, gamma=config.gamma, loss_db=0.1),
    ]

    nalm = NALMSection(coupler_ratio=0.5, loop_sections_cw=nalm_loop)
    cfbg = CFBGSection(config.cfbg_dispersion, config.cfbg_bandwidth, config.wavelength)
    output = OutputCoupler(config.output_coupling_ratio)

    sections = [gain_section, passive_before, nalm, cfbg, passive_after, output]
    return CavityModel(sections)


__all__ = [
    "SimulationConfig",
    "GainFiberParameters",
    "GainFiberModel",
    "PassiveFiberSection",
    "GainFiberSection",
    "CFBGSection",
    "OutputCoupler",
    "NALMSection",
    "CavityModel",
    "build_default_cavity",
    "run_parameter_sweep",
]
