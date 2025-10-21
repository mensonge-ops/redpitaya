"""Complete simulation of a NALM-based mode-locked fiber laser."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .components import FiberSegment, GainFiber, NALM, NALMState, SpectralFilter, pulse_energy


@dataclass
class SimulationHistoryEntry:
    """Per-round-trip diagnostics collected during the simulation."""

    round_trip: int
    intracavity_energy: float
    output_energy: float
    nalm_transmission: float
    cw_energy: float
    ccw_energy: float
    cw_phase_shift: float
    ccw_phase_shift: float
    gain: float


@dataclass
class SimulationResult:
    """Container for the time-domain traces and metrics returned by ``run``."""

    field_history: np.ndarray
    output_history: np.ndarray
    history: List[SimulationHistoryEntry]
    time_axis: np.ndarray
    frequency_axis: np.ndarray


class NALMFiberLaserSimulation:
    """Iteratively solve the map of a NALM fiber laser cavity.

    The configuration closely follows the implementation from the
    `NALM-fiber-laser <https://github.com/wenzengrun/NALM-fiber-laser>`_
    reference project.  The goal is not to perfectly reproduce a specific
    experimental setup but to capture the main ingredients of passive
    mode-locking with a Nonlinear Amplifying Loop Mirror (NALM).

    The cavity considered here contains::

        - a lumped saturable gain stage representing the active fiber,
        - a segment of dispersive nonlinear fiber,
        - a Gaussian spectral filter,
        - a NALM acting as an effective saturable absorber,
        - an output coupler that extracts a fraction of the circulating power.

    Users can tweak the component parameters after instantiation if they wish
    to replicate a different design.
    """

    def __init__(
        self,
        *,
        time_window: float = 50e-12,
        num_samples: int = 2048,
        main_fiber: Optional[FiberSegment] = None,
        gain: Optional[GainFiber] = None,
        nalm: Optional[NALM] = None,
        spectral_filter: Optional[SpectralFilter] = None,
        output_coupling: float = 0.1,
        store_every: int = 10,
    ) -> None:
        self.time_window = time_window
        self.num_samples = num_samples
        self.dt = time_window / num_samples
        self.time_axis = np.linspace(-time_window / 2, time_window / 2, num_samples, endpoint=False)
        self.frequency_axis = 2 * np.pi * np.fft.fftfreq(num_samples, d=self.dt)

        self.main_fiber = main_fiber or FiberSegment(
            length=8.0,
            beta2=-22e-27,
            gamma=1.3e-3,
            loss=0.0,
            n_steps=30,
        )
        self.gain = gain or GainFiber(
            small_signal_gain=2.5,
            saturation_energy=4e-9,
            bias=0.0,
            min_gain=-1.5,
            max_gain=3.0,
        )
        loop_fiber = FiberSegment(
            length=2.5,
            beta2=-22e-27,
            gamma=1.7e-3,
            loss=0.0,
            n_steps=20,
        )
        loop_gain = GainFiber(
            small_signal_gain=1.2,
            saturation_energy=3e-9,
            bias=-0.1,
            min_gain=-2.0,
            max_gain=2.0,
        )
        self.nalm = nalm or NALM(
            fiber=loop_fiber,
            coupling_ratio=0.55,
            gain=loop_gain,
            cw_gain_bias=0.05,
            ccw_gain_bias=-0.02,
            cw_phase_bias=0.0,
            ccw_phase_bias=0.0,
            n_steps=15,
        )
        self.spectral_filter = spectral_filter or SpectralFilter(
            bandwidth=2.5e12,  # roughly 0.4 nm at 1550 nm
            order=2.0,
        )
        if not 0.0 < output_coupling < 1.0:
            raise ValueError("Output coupling must lie between 0 and 1")
        self.output_coupling = output_coupling
        self.store_every = max(store_every, 1)

    def _initial_field(self, seed: Optional[int]) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise_amplitude = 1e-6
        real_noise = rng.normal(scale=noise_amplitude, size=self.num_samples)
        imag_noise = rng.normal(scale=noise_amplitude, size=self.num_samples)
        return real_noise + 1j * imag_noise

    def run(
        self,
        num_round_trips: int,
        *,
        seed: Optional[int] = None,
        pump_bias: float = 0.0,
    ) -> SimulationResult:
        """Run the cavity simulation for ``num_round_trips`` iterations."""

        field = self._initial_field(seed)
        stored_fields = []
        stored_outputs = []
        history: List[SimulationHistoryEntry] = []

        dt = self.dt
        sqrt_transmission = np.sqrt(self.output_coupling)
        sqrt_reflection = np.sqrt(1.0 - self.output_coupling)

        for round_trip in range(num_round_trips):
            field = self.gain.apply(field, dt, extra_bias=pump_bias)
            gain_value = self.gain.last_gain

            field = self.main_fiber.propagate(field, dt)
            if self.spectral_filter is not None:
                field = self.spectral_filter.apply(field, dt)

            field, nalm_state = self.nalm.apply(field, dt)

            output_field = sqrt_transmission * field
            field = sqrt_reflection * field

            intracavity_energy = pulse_energy(field, dt)
            output_energy = pulse_energy(output_field, dt)

            history.append(
                SimulationHistoryEntry(
                    round_trip=round_trip,
                    intracavity_energy=intracavity_energy,
                    output_energy=output_energy,
                    nalm_transmission=nalm_state.transmission,
                    cw_energy=nalm_state.cw_energy,
                    ccw_energy=nalm_state.ccw_energy,
                    cw_phase_shift=nalm_state.cw_phase_shift,
                    ccw_phase_shift=nalm_state.ccw_phase_shift,
                    gain=gain_value,
                )
            )

            if round_trip % self.store_every == 0 or round_trip == num_round_trips - 1:
                stored_fields.append(field.copy())
                stored_outputs.append(output_field.copy())

        field_history = np.stack(stored_fields, axis=0)
        output_history = np.stack(stored_outputs, axis=0)

        return SimulationResult(
            field_history=field_history,
            output_history=output_history,
            history=history,
            time_axis=self.time_axis,
            frequency_axis=self.frequency_axis,
        )

    def spectrum(self, field: np.ndarray) -> np.ndarray:
        """Return the power spectral density of ``field``."""

        return np.abs(np.fft.fftshift(np.fft.fft(field))) ** 2
