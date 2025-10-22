"""Foundational building blocks for simulating NALM fiber lasers.

This module provides utilities for creating temporal grids, electric field
containers and the optical components that appear in the sequential simulation
stages described in :mod:`nalmsim.simulation`.  The implementation intentionally
favours clarity over raw performance so the individual pieces can be compared
with reference implementations from the literature when the user reproduces the
step-by-step workflow required for the project.

The numerical model closely follows the split-step Fourier method (SSFM) for the
nonlinear Schrödinger equation with additional lumped elements that implement
saturable gain, saturable absorption, spectral filtering and output coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

ArrayLike = np.ndarray


@dataclass(slots=True)
class TemporalGrid:
    """Uniform temporal grid used throughout the cavity simulation.

    Parameters
    ----------
    points:
        Number of temporal samples in the simulation window (must be a power of
        two for efficient FFT operations).
    window:
        Total simulated time window in seconds.  The sampling period is derived
        as ``dt = window / points``.
    """

    points: int
    window: float
    time: ArrayLike = field(init=False)
    angular_frequency: ArrayLike = field(init=False)

    def __post_init__(self) -> None:  # pragma: no cover - light-weight setter
        if self.points <= 0:
            raise ValueError("points must be positive")
        if self.window <= 0.0:
            raise ValueError("window must be positive")
        dt = self.dt
        t0 = -0.5 * self.window
        self.time = t0 + dt * np.arange(self.points)
        freq = np.fft.fftfreq(self.points, d=dt)
        self.angular_frequency = 2.0 * np.pi * freq

    @property
    def dt(self) -> float:
        return self.window / float(self.points)


@dataclass(slots=True)
class Pulse:
    """Complex envelope sampled on a :class:`TemporalGrid`."""

    grid: TemporalGrid
    field: ArrayLike

    def copy(self) -> "Pulse":
        return Pulse(self.grid, np.array(self.field, dtype=np.complex128, copy=True))

    def ensure_shape(self) -> None:
        if self.field.shape != (self.grid.points,):
            raise ValueError(
                "Field size does not match grid: "
                f"expected {self.grid.points}, received {self.field.shape!r}"
            )

    # --- Diagnostic utilities -------------------------------------------------
    def energy(self) -> float:
        return float(np.real(np.sum(np.abs(self.field) ** 2) * self.grid.dt))

    def peak_power(self) -> float:
        return float(np.max(np.abs(self.field) ** 2))

    def rms_width(self) -> float:
        intensity = np.abs(self.field) ** 2
        energy = np.sum(intensity) * self.grid.dt
        if energy == 0.0:
            return 0.0
        mean_t = np.sum(self.grid.time * intensity) * self.grid.dt / energy
        variance = np.sum(((self.grid.time - mean_t) ** 2) * intensity) * self.grid.dt / energy
        return float(np.sqrt(max(variance, 0.0)))

    def spectrum(self) -> ArrayLike:
        return np.fft.fftshift(np.fft.fft(self.field))

    def spectral_intensity(self) -> ArrayLike:
        spec = self.spectrum()
        return np.abs(spec) ** 2

    def fwhm_duration(self) -> float:
        return _fwhm(self.grid.time, np.abs(self.field) ** 2)

    def fwhm_spectral_width(self) -> float:
        freq = np.fft.fftshift(self.grid.angular_frequency) / (2 * np.pi)
        return _fwhm(freq, self.spectral_intensity())

    def time_bandwidth_product(self) -> float:
        dt_fwhm = self.fwhm_duration()
        df_fwhm = self.fwhm_spectral_width()
        if dt_fwhm <= 0.0 or df_fwhm <= 0.0:
            return 0.0
        return float(dt_fwhm * df_fwhm)


def _fwhm(axis: ArrayLike, intensity: ArrayLike) -> float:
    if intensity.size == 0:
        return 0.0
    i_max = float(np.max(intensity))
    if not np.isfinite(i_max) or i_max <= 0.0:
        return 0.0
    half = 0.5 * i_max
    above = np.nonzero(intensity >= half)[0]
    if above.size < 2:
        return 0.0
    left = above[0]
    right = above[-1]
    if left == right:
        return 0.0
    return float(abs(axis[right] - axis[left]))


class Component:
    """Base class for cavity elements."""

    def propagate(self, pulse: Pulse) -> Pulse:
        raise NotImplementedError


@dataclass(slots=True)
class FiberSegment(Component):
    """Dispersive nonlinear fiber using the split-step Fourier method."""

    length: float
    beta2: float
    beta3: float = 0.0
    gamma: float = 0.0
    loss: float = 0.0
    steps: int = 40

    def propagate(self, pulse: Pulse) -> Pulse:
        pulse.ensure_shape()
        steps = max(self.steps, 1)
        dz = self.length / steps
        omega = pulse.grid.angular_frequency
        linear_coeff = (
            (-self.loss / 2.0)
            + 0.5j * self.beta2 * omega**2
            - (1.0 / 6.0) * self.beta3 * omega**3
        )
        linear_half = np.exp(linear_coeff * dz / 2.0)
        field = np.array(pulse.field, dtype=np.complex128, copy=True)
        for _ in range(steps):
            field = np.fft.ifft(linear_half * np.fft.fft(field))
            if self.gamma != 0.0:
                phase = np.exp(1j * self.gamma * np.abs(field) ** 2 * dz)
                field *= phase
            field = np.fft.ifft(linear_half * np.fft.fft(field))
        return Pulse(pulse.grid, field)


@dataclass(slots=True)
class GainFiber(FiberSegment):
    """Ytterbium-doped fiber with saturable gain and finite bandwidth."""

    small_signal_gain: float = 4.0  # Nepers of gain per pass
    saturation_energy: float = 80e-9
    bandwidth_fwhm: float = 40e12  # Hz
    gain_clamp: Optional[tuple[float, float]] = None

    def propagate(self, pulse: Pulse) -> Pulse:
        base = super().propagate(pulse)
        energy = base.energy()
        if self.saturation_energy <= 0.0:
            net_gain = self.small_signal_gain
        else:
            net_gain = self.small_signal_gain / (1.0 + energy / self.saturation_energy)
        if self.gain_clamp is not None:
            g_min, g_max = self.gain_clamp
            net_gain = float(np.clip(net_gain, g_min, g_max))
        spectrum = np.fft.fft(base.field)
        if self.bandwidth_fwhm > 0.0:
            sigma = self.bandwidth_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            freq = pulse.grid.angular_frequency / (2.0 * np.pi)
            spectral_gain = np.exp(-0.5 * (freq / sigma) ** 2)
            spectrum *= spectral_gain
        amplified = np.fft.ifft(spectrum) * np.exp(net_gain)
        return Pulse(pulse.grid, amplified)


@dataclass(slots=True)
class BandpassFilter(Component):
    """Gaussian amplitude filter in the frequency domain."""

    bandwidth_fwhm: float
    order: float = 2.0

    def propagate(self, pulse: Pulse) -> Pulse:
        freq = np.abs(pulse.grid.angular_frequency / (2.0 * np.pi))
        sigma = self.bandwidth_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        response = np.exp(-0.5 * (freq / sigma) ** self.order)
        filtered = np.fft.ifft(np.fft.fft(pulse.field) * response)
        return Pulse(pulse.grid, filtered)


@dataclass(slots=True)
class SaturableAbsorber(Component):
    """Simple intensity-dependent transmission element."""

    modulation_depth: float
    saturation_power: float
    nonsaturable_loss: float = 0.0

    def propagate(self, pulse: Pulse) -> Pulse:
        intensity = np.abs(pulse.field) ** 2
        if self.saturation_power <= 0.0:
            transmission = 1.0 - self.modulation_depth - self.nonsaturable_loss
        else:
            transmission = 1.0 - self.modulation_depth * np.exp(-intensity / self.saturation_power)
            transmission -= self.nonsaturable_loss
        transmission = np.clip(transmission, 0.0, 1.0)
        return Pulse(pulse.grid, pulse.field * np.sqrt(transmission))


@dataclass(slots=True)
class OutputCoupler(Component):
    """Extract a fraction of the circulating power."""

    ratio: float

    def propagate(self, pulse: Pulse) -> tuple[Pulse, Pulse]:
        if not 0.0 < self.ratio < 1.0:
            raise ValueError("Output coupling ratio must be between 0 and 1")
        transmitted = Pulse(pulse.grid, np.sqrt(self.ratio) * pulse.field)
        intracavity = Pulse(pulse.grid, np.sqrt(1.0 - self.ratio) * pulse.field)
        return intracavity, transmitted


@dataclass(slots=True)
class NALM(Component):
    """Nonlinear amplifying loop mirror."""

    coupling_ratio: float
    cw_path: Sequence[Component]
    ccw_path: Sequence[Component]

    def propagate(self, pulse: Pulse) -> Pulse:
        if not 0.0 < self.coupling_ratio < 1.0:
            raise ValueError("Coupling ratio must be between 0 and 1")
        sqrt_k = np.sqrt(self.coupling_ratio)
        sqrt_t = np.sqrt(1.0 - self.coupling_ratio)
        cw = Pulse(pulse.grid, sqrt_k * pulse.field)
        ccw = Pulse(pulse.grid, 1j * sqrt_t * pulse.field)
        for component in self.cw_path:
            cw = component.propagate(cw)
        for component in self.ccw_path:
            ccw = component.propagate(ccw)
        combined_field = sqrt_t * cw.field + 1j * sqrt_k * ccw.field
        return Pulse(pulse.grid, combined_field)


def cascaded_propagation(pulse: Pulse, components: Iterable[Component]) -> Pulse:
    result = pulse
    for component in components:
        result = component.propagate(result)
    return result
