"""Complete simulation of a NALM-based mode-locked fiber laser."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    pump_bias: float
    peak_power: float
    pulse_duration: float
    spectral_width: float
    time_bandwidth_product: float
    pulse_contrast: float


@dataclass
class ModeLockingReport:
    """Summary of the mode-lock assessment for the latest simulation."""

    energy_window: int
    energy_tolerance: float
    contrast_threshold: float
    tbp_threshold: float
    peak_power_threshold: float
    mean_energy: float
    energy_std: float
    mean_peak_power: float
    mean_pulse_duration: float
    mean_spectral_width: float
    mean_time_bandwidth_product: float
    mean_contrast: float
    representative_peak_power: float
    representative_time_bandwidth_product: float
    representative_contrast: float
    met_energy_stability: bool
    met_contrast: bool
    met_tbp: bool
    met_peak_power: bool


@dataclass
class SimulationResult:
    """Container for the time-domain traces and metrics returned by a simulation."""

    field_history: np.ndarray
    output_history: np.ndarray
    history: List[SimulationHistoryEntry]
    stored_round_trips: np.ndarray
    time_axis: np.ndarray
    frequency_axis: np.ndarray
    mode_locked: Optional[bool]
    mode_lock_report: Optional[ModeLockingReport]


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
        sequence_span: int = 8,
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
        self.sequence_span = max(int(sequence_span), 1)

    def _resolve_target_energy(self, target_energy: Optional[float]) -> float:
        if target_energy is not None and target_energy > 0.0:
            return float(target_energy)

        saturation_energy = getattr(self.gain, "saturation_energy", 0.0)
        if saturation_energy and saturation_energy > 0.0:
            return 0.6 * float(saturation_energy)

        return 1e-9

    @staticmethod
    def _apply_pump_control(
        pump_bias: float,
        intracavity_energy: float,
        *,
        adaptive_pump: bool,
        target_energy: float,
        pump_adjustment: float,
        pump_min: float,
        pump_max: float,
        smoothing: float,
        error_state: float,
    ) -> tuple[float, float]:
        if not adaptive_pump:
            return pump_bias, error_state

        if target_energy <= 0.0 or pump_adjustment <= 0.0:
            return pump_bias, error_state

        error = (intracavity_energy - target_energy) / target_energy
        blended_error = (1.0 - smoothing) * error + smoothing * error_state
        new_bias = pump_bias - pump_adjustment * blended_error
        new_bias = float(np.clip(new_bias, pump_min, pump_max))
        return new_bias, blended_error

    def _pulse_diagnostics(self, field: np.ndarray) -> tuple[float, float, float, float, float]:
        """Return key pulse metrics derived from the intracavity field.

        The computed quantities follow the standard definitions commonly used
        to identify mode-locking in passively mode-locked lasers (see, e.g.,
        Haus, *IEEE J. Sel. Top. Quantum Electron.* **2**, 1996; Dudley & Taylor,
        *Nat. Photonics* **3**, 2009).  They include:

        - peak power,
        - RMS pulse duration,
        - RMS spectral width (in Hz),
        - time-bandwidth product (TBP),
        - pulse contrast (peak power over average power).
        """

        intensity = np.abs(field) ** 2
        peak_power = float(np.max(intensity))
        dt = self.dt
        energy = float(np.sum(intensity) * dt)

        if not np.isfinite(energy) or energy <= 0.0:
            return peak_power, 0.0, 0.0, 0.0, 0.0

        mean_power = energy / (self.time_window)

        shift = (self.num_samples // 2) - int(np.argmax(intensity))
        intensity_centered = np.roll(intensity, shift)
        field_centered = np.roll(field, shift)

        weights_time = intensity_centered * dt / energy
        mean_time = float(np.sum(self.time_axis * weights_time))
        variance_time = float(np.sum((self.time_axis - mean_time) ** 2 * weights_time))
        variance_time = max(variance_time, 0.0)
        rms_duration = float(np.sqrt(variance_time))

        spectrum = np.fft.fftshift(np.abs(np.fft.fft(field_centered)) ** 2)
        freq_axis = np.fft.fftshift(self.frequency_axis)
        domega = float(np.abs(freq_axis[1] - freq_axis[0])) if freq_axis.size > 1 else 0.0
        spectral_energy = float(np.sum(spectrum) * domega)

        if not np.isfinite(spectral_energy) or spectral_energy <= 0.0:
            rms_bandwidth_hz = 0.0
        else:
            weights_freq = spectrum * domega / spectral_energy
            mean_freq = float(np.sum(freq_axis * weights_freq))
            variance_freq = float(np.sum((freq_axis - mean_freq) ** 2 * weights_freq))
            variance_freq = max(variance_freq, 0.0)
            rms_bandwidth_hz = float(np.sqrt(variance_freq)) / (2.0 * np.pi)

        tbp = rms_duration * rms_bandwidth_hz
        contrast = peak_power / (mean_power + 1e-18)

        return peak_power, rms_duration, rms_bandwidth_hz, tbp, contrast

    @staticmethod
    def _finite_array(values: List[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return np.empty(0, dtype=float)
        return array[np.isfinite(array)]

    def _assess_mode_locking(
        self,
        history: List[SimulationHistoryEntry],
        *,
        energy_window: int,
        energy_tolerance: float,
        contrast_threshold: float,
        tbp_threshold: float,
        peak_power_threshold: float,
    ) -> Optional[ModeLockingReport]:
        if len(history) < energy_window:
            return None

        window = history[-energy_window:]
        energies = np.array([entry.intracavity_energy for entry in window], dtype=float)
        if np.any(~np.isfinite(energies)):
            return None

        mean_energy = float(np.mean(energies))
        if mean_energy <= 0.0:
            return None

        std_denom = np.std(energies, ddof=1 if energies.size > 1 else 0)
        energy_std = float(std_denom / mean_energy)

        peak_powers = self._finite_array([entry.peak_power for entry in window])
        durations = self._finite_array([entry.pulse_duration for entry in window])
        spectral_widths = self._finite_array([entry.spectral_width for entry in window])
        tbps = self._finite_array([entry.time_bandwidth_product for entry in window])
        contrasts = self._finite_array([entry.pulse_contrast for entry in window])

        mean_peak_power = float(np.mean(peak_powers)) if peak_powers.size else 0.0
        representative_peak_power = (
            float(np.percentile(peak_powers, 10.0)) if peak_powers.size else 0.0
        )
        mean_duration = float(np.mean(durations)) if durations.size else 0.0
        mean_spectral_width = (
            float(np.mean(spectral_widths)) if spectral_widths.size else 0.0
        )
        mean_tbp = float(np.mean(tbps)) if tbps.size else 0.0
        representative_tbp = (
            float(np.percentile(tbps, 90.0)) if tbps.size else float("inf")
        )
        mean_contrast = float(np.mean(contrasts)) if contrasts.size else 0.0
        representative_contrast = (
            float(np.percentile(contrasts, 10.0)) if contrasts.size else 0.0
        )

        met_energy = energy_std < energy_tolerance
        met_contrast = representative_contrast > contrast_threshold
        met_tbp = representative_tbp < tbp_threshold
        met_peak = representative_peak_power > peak_power_threshold

        return ModeLockingReport(
            energy_window=energy_window,
            energy_tolerance=energy_tolerance,
            contrast_threshold=contrast_threshold,
            tbp_threshold=tbp_threshold,
            peak_power_threshold=peak_power_threshold,
            mean_energy=mean_energy,
            energy_std=energy_std,
            mean_peak_power=mean_peak_power,
            mean_pulse_duration=mean_duration,
            mean_spectral_width=mean_spectral_width,
            mean_time_bandwidth_product=mean_tbp,
            mean_contrast=mean_contrast,
            representative_peak_power=representative_peak_power,
            representative_time_bandwidth_product=representative_tbp,
            representative_contrast=representative_contrast,
            met_energy_stability=met_energy,
            met_contrast=met_contrast,
            met_tbp=met_tbp,
            met_peak_power=met_peak,
        )

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
        adaptive_pump: bool = False,
        target_energy: Optional[float] = None,
        pump_adjustment: float = 0.1,
        pump_min: float = -0.5,
        pump_max: float = 2.0,
        pump_smoothing: float = 0.75,
    ) -> SimulationResult:
        """Run the cavity simulation for ``num_round_trips`` iterations.

        When ``adaptive_pump`` is enabled, a simple feedback controller adjusts
        the effective pump bias after each round-trip to drive the intracavity
        energy towards ``target_energy``.
        """

        field = self._initial_field(seed)
        stored_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        recent_buffer: deque[Tuple[int, np.ndarray, np.ndarray]] = deque(
            maxlen=self.sequence_span
        )
        history: List[SimulationHistoryEntry] = []

        dt = self.dt
        sqrt_transmission = np.sqrt(self.output_coupling)
        sqrt_reflection = np.sqrt(1.0 - self.output_coupling)

        target_energy_value = self._resolve_target_energy(target_energy)
        smoothing = float(np.clip(pump_smoothing, 0.0, 0.999))
        pump_minimum, pump_maximum = sorted((float(pump_min), float(pump_max)))
        if pump_maximum == pump_minimum:
            pump_maximum = pump_minimum + 1e-9
        pump_bias_current = float(pump_bias)
        error_state = 0.0

        for round_trip in range(num_round_trips):
            field = self.gain.apply(field, dt, extra_bias=pump_bias_current)
            gain_value = self.gain.last_gain

            field = self.main_fiber.propagate(field, dt)
            if self.spectral_filter is not None:
                field = self.spectral_filter.apply(field, dt)

            field, nalm_state = self.nalm.apply(field, dt)

            output_field = sqrt_transmission * field
            field = sqrt_reflection * field

            intracavity_energy = pulse_energy(field, dt)
            output_energy = pulse_energy(output_field, dt)
            peak_power, pulse_duration, spectral_width, tbp, contrast = self._pulse_diagnostics(field)

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
                    pump_bias=pump_bias_current,
                    peak_power=peak_power,
                    pulse_duration=pulse_duration,
                    spectral_width=spectral_width,
                    time_bandwidth_product=tbp,
                    pulse_contrast=contrast,
                )
            )

            snapshot_field = field.copy()
            snapshot_output = output_field.copy()
            recent_buffer.append((round_trip, snapshot_field, snapshot_output))

            if round_trip % self.store_every == 0 or round_trip == num_round_trips - 1:
                stored_map.setdefault(round_trip, (snapshot_field, snapshot_output))

            pump_bias_current, error_state = self._apply_pump_control(
                pump_bias_current,
                intracavity_energy,
                adaptive_pump=adaptive_pump,
                target_energy=target_energy_value,
                pump_adjustment=pump_adjustment,
                pump_min=pump_minimum,
                pump_max=pump_maximum,
                smoothing=smoothing,
                error_state=error_state,
            )

        for rt, snap_field, snap_output in recent_buffer:
            stored_map.setdefault(rt, (snap_field, snap_output))

        if stored_map:
            ordered = sorted(stored_map.items())
            stored_round_trips = np.array([rt for rt, _ in ordered], dtype=int)
            field_history = np.stack([snap[0] for _, snap in ordered], axis=0)
            output_history = np.stack([snap[1] for _, snap in ordered], axis=0)
        else:
            stored_round_trips = np.empty(0, dtype=int)
            field_history = np.empty((0, self.num_samples), dtype=np.complex128)
            output_history = np.empty((0, self.num_samples), dtype=np.complex128)

        return SimulationResult(
            field_history=field_history,
            output_history=output_history,
            history=history,
            stored_round_trips=np.array(stored_round_trips, dtype=int),
            time_axis=self.time_axis,
            frequency_axis=self.frequency_axis,
            mode_locked=None,
            mode_lock_report=None,
        )

    def spectrum(self, field: np.ndarray) -> np.ndarray:
        """Return the power spectral density of ``field``."""

        return np.abs(np.fft.fftshift(np.fft.fft(field))) ** 2

    def run_until_mode_locked(
        self,
        max_round_trips: int,
        *,
        seed: Optional[int] = None,
        pump_bias: float = 0.0,
        min_round_trips: int = 100,
        energy_window: int = 50,
        relative_tolerance: float = 5e-3,
        adaptive_pump: bool = True,
        target_energy: Optional[float] = None,
        pump_adjustment: float = 0.1,
        pump_min: float = -0.5,
        pump_max: float = 2.0,
        pump_smoothing: float = 0.75,
        contrast_threshold: float = 15.0,
        tbp_threshold: float = 0.65,
        peak_power_threshold: float = 80.0,
    ) -> SimulationResult:
        """Run the simulation until the intracavity energy converges.

        The method keeps iterating the cavity map until the relative standard
        deviation of the intracavity energy over the latest ``energy_window``
        round-trips falls below ``relative_tolerance`` *and* the additional
        pulse quality criteria from ultrafast laser literature are satisfied:

        - the pulse contrast (peak power over average power) must exceed
          ``contrast_threshold`` to ensure a well-defined pulse train,
        - the time-bandwidth product must remain below ``tbp_threshold`` to
          indicate transform-limited behaviour,
        - the peak power must exceed ``peak_power_threshold`` to avoid spurious
          noise-like pulses.

        If convergence is not achieved before ``max_round_trips`` iterations,
        the method returns the full history and flags the run as not
        mode-locked.  With ``adaptive_pump`` enabled (the default), the pump
        bias is gently adjusted every round-trip to steer the energy towards
        ``target_energy``.
        """

        if energy_window <= 1:
            raise ValueError("energy_window must be greater than 1 to assess convergence")

        field = self._initial_field(seed)
        stored_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        recent_buffer: deque[Tuple[int, np.ndarray, np.ndarray]] = deque(
            maxlen=self.sequence_span
        )
        history: List[SimulationHistoryEntry] = []

        dt = self.dt
        sqrt_transmission = np.sqrt(self.output_coupling)
        sqrt_reflection = np.sqrt(1.0 - self.output_coupling)

        locked = False
        mode_lock_report: Optional[ModeLockingReport] = None

        target_energy_value = self._resolve_target_energy(target_energy)
        smoothing = float(np.clip(pump_smoothing, 0.0, 0.999))
        pump_minimum, pump_maximum = sorted((float(pump_min), float(pump_max)))
        if pump_maximum == pump_minimum:
            pump_maximum = pump_minimum + 1e-9
        pump_bias_current = float(pump_bias)
        error_state = 0.0

        for round_trip in range(max_round_trips):
            field = self.gain.apply(field, dt, extra_bias=pump_bias_current)
            gain_value = self.gain.last_gain

            field = self.main_fiber.propagate(field, dt)
            if self.spectral_filter is not None:
                field = self.spectral_filter.apply(field, dt)

            field, nalm_state = self.nalm.apply(field, dt)

            output_field = sqrt_transmission * field
            field = sqrt_reflection * field

            intracavity_energy = pulse_energy(field, dt)
            output_energy = pulse_energy(output_field, dt)

            peak_power, pulse_duration, spectral_width, tbp, contrast = self._pulse_diagnostics(field)

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
                    pump_bias=pump_bias_current,
                    peak_power=peak_power,
                    pulse_duration=pulse_duration,
                    spectral_width=spectral_width,
                    time_bandwidth_product=tbp,
                    pulse_contrast=contrast,
                )
            )

            snapshot_field = field.copy()
            snapshot_output = output_field.copy()
            recent_buffer.append((round_trip, snapshot_field, snapshot_output))

            if round_trip % self.store_every == 0 or round_trip == max_round_trips - 1:
                stored_map.setdefault(round_trip, (snapshot_field, snapshot_output))

            pump_bias_current, error_state = self._apply_pump_control(
                pump_bias_current,
                intracavity_energy,
                adaptive_pump=adaptive_pump,
                target_energy=target_energy_value,
                pump_adjustment=pump_adjustment,
                pump_min=pump_minimum,
                pump_max=pump_maximum,
                smoothing=smoothing,
                error_state=error_state,
            )

            if round_trip + 1 < min_round_trips:
                continue

            report = self._assess_mode_locking(
                history,
                energy_window=energy_window,
                energy_tolerance=relative_tolerance,
                contrast_threshold=contrast_threshold,
                tbp_threshold=tbp_threshold,
                peak_power_threshold=peak_power_threshold,
            )
            if report is None:
                continue

            mode_lock_report = report
            if (
                report.met_energy_stability
                and report.met_contrast
                and report.met_tbp
                and report.met_peak_power
            ):
                locked = True
                stored_map.setdefault(round_trip, (snapshot_field, snapshot_output))
                break

        for rt, snap_field, snap_output in recent_buffer:
            stored_map.setdefault(rt, (snap_field, snap_output))

        if stored_map:
            ordered = sorted(stored_map.items())
            stored_round_trips_array = np.array([rt for rt, _ in ordered], dtype=int)
            field_history = np.stack([snap[0] for _, snap in ordered], axis=0)
            output_history = np.stack([snap[1] for _, snap in ordered], axis=0)
        else:
            stored_round_trips_array = np.empty(0, dtype=int)
            field_history = np.empty((0, self.num_samples), dtype=np.complex128)
            output_history = np.empty((0, self.num_samples), dtype=np.complex128)

        return SimulationResult(
            field_history=field_history,
            output_history=output_history,
            history=history,
            stored_round_trips=stored_round_trips_array,
            time_axis=self.time_axis,
            frequency_axis=self.frequency_axis,
            mode_locked=locked,
            mode_lock_report=mode_lock_report,
        )
