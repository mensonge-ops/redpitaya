"""Core optical components used in the NALM fiber laser simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FiberSegment:
    """Dispersive nonlinear fiber modeled with the split-step Fourier method.

    Parameters
    ----------
    length : float
        Physical length of the fiber in meters.
    beta2 : float
        Group velocity dispersion coefficient (s^2 / m).  Negative values
        correspond to anomalous dispersion.
    gamma : float
        Nonlinear coefficient (1 / (W * m)).
    loss : float, optional
        Power attenuation coefficient (1 / m).  Zero denotes a lossless fiber.
    n_steps : int, optional
        Number of integration steps used for a single propagation call.
    """

    length: float
    beta2: float
    gamma: float
    loss: float = 0.0
    n_steps: int = 20

    def propagate(self, field: np.ndarray, dt: float, *, n_steps: Optional[int] = None) -> np.ndarray:
        """Propagate ``field`` through the fiber segment.

        Parameters
        ----------
        field:
            Complex electric field samples in the time domain (sqrt(W)).
        dt:
            Temporal sampling period in seconds.
        n_steps:
            Optional override for the number of integration steps.
        """
        steps = n_steps or self.n_steps
        if steps <= 0:
            raise ValueError("Number of integration steps must be positive")

        dz = self.length / steps
        omega = 2 * np.pi * np.fft.fftfreq(field.size, d=dt)
        # Half-step linear operator (dispersion + distributed loss).
        dispersion = np.exp(-0.5j * self.beta2 * omega**2 * dz)
        attenuation = np.exp(-0.5 * self.loss * dz)
        linear_half_step = attenuation * dispersion

        propagated = np.array(field, dtype=np.complex128, copy=True)
        for _ in range(steps):
            propagated = np.fft.ifft(linear_half_step * np.fft.fft(propagated))
            nonlinear_phase = np.exp(1j * self.gamma * np.abs(propagated) ** 2 * dz)
            propagated *= nonlinear_phase
            propagated = np.fft.ifft(linear_half_step * np.fft.fft(propagated))
        return propagated


@dataclass
class GainFiber:
    """Saturable gain medium.

    The model assumes lumped gain with saturation described by::

        g = g0 / (1 + E / E_sat) + bias

    where ``E`` is the pulse energy in Joules.
    """

    small_signal_gain: float
    saturation_energy: float
    bias: float = 0.0
    min_gain: Optional[float] = None
    max_gain: Optional[float] = None

    def __post_init__(self) -> None:
        self._last_gain = 0.0

    @property
    def last_gain(self) -> float:
        return getattr(self, '_last_gain', 0.0)

    def apply(self, field: np.ndarray, dt: float, *, extra_bias: float = 0.0) -> np.ndarray:
        """Apply saturable gain to ``field``.

        Parameters
        ----------
        field:
            Complex electric field samples (sqrt(W)).
        dt:
            Temporal sampling period in seconds.
        extra_bias:
            Additional bias added to the gain exponent.  Positive values
            increase the net gain while negative values reduce it.
        """
        energy = pulse_energy(field, dt)
        if self.saturation_energy <= 0.0:
            gain = self.small_signal_gain + self.bias + extra_bias
        else:
            gain = self.small_signal_gain / (1.0 + energy / self.saturation_energy)
            gain += self.bias + extra_bias

        if self.min_gain is not None:
            gain = max(gain, self.min_gain)
        if self.max_gain is not None:
            gain = min(gain, self.max_gain)

        self._last_gain = gain
        return field * np.exp(gain)


@dataclass
class SpectralFilter:
    """Gaussian spectral filter applied in the frequency domain."""

    bandwidth: float
    order: float = 2.0

    def apply(self, field: np.ndarray, dt: float) -> np.ndarray:
        omega = 2 * np.pi * np.fft.fftfreq(field.size, d=dt)
        response = np.exp(-0.5 * (np.abs(omega) / self.bandwidth) ** self.order)
        return np.fft.ifft(response * np.fft.fft(field))


@dataclass
class NALMState:
    """Diagnostic information returned by :class:`NALM`."""

    transmission: float
    cw_energy: float
    ccw_energy: float
    cw_phase_shift: float
    ccw_phase_shift: float


@dataclass
class NALM:
    """Nonlinear Amplifying Loop Mirror (NALM).

    The NALM is described by a 2x2 fiber coupler with coupling ratio ``kappa``
    and a loop containing a nonlinear fiber and optionally a saturable gain
    segment.  The component acts as an intensity-dependent transmission filter
    when used in reflection configuration inside a fiber laser cavity.
    """

    fiber: FiberSegment
    coupling_ratio: float = 0.5
    gain: Optional[GainFiber] = None
    cw_gain_bias: float = 0.0
    ccw_gain_bias: float = 0.0
    cw_phase_bias: float = 0.0
    ccw_phase_bias: float = 0.0
    n_steps: Optional[int] = None

    def apply(self, field: np.ndarray, dt: float) -> tuple[np.ndarray, NALMState]:
        if not 0.0 < self.coupling_ratio < 1.0:
            raise ValueError("Coupling ratio must lie between 0 and 1")

        sqrt_k = np.sqrt(self.coupling_ratio)
        sqrt_t = np.sqrt(1.0 - self.coupling_ratio)

        cw = sqrt_k * field
        ccw = 1j * sqrt_t * field

        cw, cw_phase = self._propagate_loop(cw, dt, self.cw_gain_bias, self.cw_phase_bias)
        ccw, ccw_phase = self._propagate_loop(ccw, dt, self.ccw_gain_bias, self.ccw_phase_bias)

        transmitted = sqrt_t * cw + 1j * sqrt_k * ccw
        _ = 1j * sqrt_k * cw + sqrt_t * ccw  # reflected port (unused)

        input_energy = pulse_energy(field, dt)
        output_energy = pulse_energy(transmitted, dt)
        transmission = output_energy / input_energy if input_energy > 0 else 0.0

        state = NALMState(
            transmission=transmission,
            cw_energy=pulse_energy(cw, dt),
            ccw_energy=pulse_energy(ccw, dt),
            cw_phase_shift=cw_phase,
            ccw_phase_shift=ccw_phase,
        )
        return transmitted, state

    def _propagate_loop(
        self,
        field: np.ndarray,
        dt: float,
        gain_bias: float,
        phase_bias: float,
    ) -> tuple[np.ndarray, float]:
        propagated = np.array(field, dtype=np.complex128, copy=True)
        if self.gain is not None:
            propagated = self.gain.apply(propagated, dt, extra_bias=gain_bias)
        propagated = self.fiber.propagate(propagated, dt, n_steps=self.n_steps)

        mean_power = np.mean(np.abs(propagated) ** 2)
        nonlinear_phase = self.fiber.gamma * self.fiber.length * mean_power
        propagated *= np.exp(1j * phase_bias)
        return propagated, nonlinear_phase + phase_bias


def pulse_energy(field: np.ndarray, dt: float) -> float:
    r"""Return the pulse energy ``E = \int |A(t)|^2 dt`` in Joules."""

    return float(np.real(np.sum(np.abs(field) ** 2) * dt))
